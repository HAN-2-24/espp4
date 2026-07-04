#include "eye_state_detector.hpp"

#include "esp_log.h"
#include "helmet_dl_runtime_lock.h"

#if HELMET_EYE_STATE_MODEL_AVAILABLE && !HELMET_EYE_STATE_MODEL_EMBEDDED
#include "esp_partition.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <new>

#if HELMET_EYE_STATE_MODEL_AVAILABLE
#include "dl_define.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"

#if HELMET_EYE_STATE_MODEL_EMBEDDED
extern const uint8_t eye_state_model_start[] asm("_binary_eye_state_detect_s8_v1_espdl_start");
#endif
#endif

static const char *TAG = "eye_state_detector";
static constexpr const char *kModelPartition = "eye_model";
static constexpr float kMinConfidence = 0.45f;
static constexpr float kMinBoxSize = 4.0f;
static constexpr int kDecodedBoxChannels = 4;
static constexpr int kRawDflBoxChannels = 64;
static constexpr int kDflBins = 16;
static constexpr int kExpectedScoreChannels = 2;
static constexpr int kClassificationChannels = 3;
static constexpr int kDirectRawChannels = 7;
static constexpr int kDirectRawTx = 0;
static constexpr int kDirectRawTy = 1;
static constexpr int kDirectRawTw = 2;
static constexpr int kDirectRawTh = 3;
static constexpr int kDirectRawObj = 4;
static constexpr int kDirectRawOpen = 5;
static constexpr int kDirectRawClosed = 6;
static constexpr int kInputSize = 320;
static constexpr int kEspDlInternalArenaBytes = 0;
static constexpr int kOpenClass = 0;
static constexpr int kClosedClass = 1;
static constexpr int kBackgroundClass = 2;

static bool s_initialized;

#if HELMET_EYE_STATE_MODEL_AVAILABLE
static dl::Model *s_model;
static dl::image::ImagePreprocessor *s_preprocessor;

static float clamp_float_local(float value, float low, float high)
{
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

#if !HELMET_EYE_STATE_MODEL_EMBEDDED
static void log_model_partition(void)
{
    const esp_partition_t *partition =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, kModelPartition);
    if (!partition) {
        ESP_LOGE(TAG, "eye-state partition '%s' not found; flash partition table and eye_model", kModelPartition);
        return;
    }

    uint8_t header[16] = {};
    esp_err_t err = esp_partition_read(partition, 0, header, sizeof(header));
    if (err != ESP_OK) {
        ESP_LOGE(TAG,
                 "read eye-state partition '%s' failed: %s offset=0x%lx size=%lu",
                 kModelPartition,
                 esp_err_to_name(err),
                 static_cast<unsigned long>(partition->address),
                 static_cast<unsigned long>(partition->size));
        return;
    }

    bool blank = true;
    for (uint8_t byte : header) {
        if (byte != 0xff) {
            blank = false;
            break;
        }
    }

    ESP_LOGI(TAG,
             "eye-state partition '%s' offset=0x%lx size=%lu header=%02x %02x %02x %02x %02x %02x %02x %02x%s",
             kModelPartition,
             static_cast<unsigned long>(partition->address),
             static_cast<unsigned long>(partition->size),
             header[0],
             header[1],
             header[2],
             header[3],
             header[4],
             header[5],
             header[6],
             header[7],
             blank ? " blank" : "");
}
#endif
#endif

static eye_state_detector_result_t make_invalid_result(helmet_vision_eye_state_t state, helmet_vision_reason_t reason)
{
    eye_state_detector_result_t result = {};
    result.valid = false;
    result.closed = false;
    result.eye_state = state;
    result.reason = reason;
    result.open_ratio = 0.0f;
    result.confidence = 0.0f;
    result.open_score = 0.0f;
    result.closed_score = 0.0f;
    result.background_score = 0.0f;
    return result;
}

#if HELMET_EYE_STATE_MODEL_AVAILABLE
static bool is_chw_tensor(const dl::TensorBase *tensor, int channels)
{
    return tensor && tensor->shape.size() == 3 && tensor->shape[0] == 1 && tensor->shape[1] == channels &&
           tensor->shape[2] > 0;
}

static bool is_hwc_tensor(const dl::TensorBase *tensor, int channels)
{
    return tensor && tensor->shape.size() == 3 && tensor->shape[0] == 1 && tensor->shape[1] > 0 &&
           tensor->shape[2] == channels;
}

static bool is_nchw4_tensor(const dl::TensorBase *tensor, int channels)
{
    return tensor && tensor->shape.size() == 4 && tensor->shape[0] == 1 && tensor->shape[1] == channels &&
           tensor->shape[2] > 0 && tensor->shape[3] > 0;
}

static bool is_nhwc4_tensor(const dl::TensorBase *tensor, int channels)
{
    return tensor && tensor->shape.size() == 4 && tensor->shape[0] == 1 && tensor->shape[1] > 0 &&
           tensor->shape[2] > 0 && tensor->shape[3] == channels;
}

static int tensor_candidate_count(const dl::TensorBase *tensor, int channels)
{
    if (is_chw_tensor(tensor, channels)) {
        return tensor->shape[2];
    }
    if (is_hwc_tensor(tensor, channels)) {
        return tensor->shape[1];
    }
    return 0;
}

static int direct_raw_candidate_count(const dl::TensorBase *tensor)
{
    if (is_nchw4_tensor(tensor, kDirectRawChannels)) {
        return tensor->shape[2] * tensor->shape[3];
    }
    if (is_nhwc4_tensor(tensor, kDirectRawChannels)) {
        return tensor->shape[1] * tensor->shape[2];
    }
    return tensor_candidate_count(tensor, kDirectRawChannels);
}

static bool direct_raw_grid_dims(const dl::TensorBase *tensor, int &grid_w, int &grid_h)
{
    if (is_nchw4_tensor(tensor, kDirectRawChannels)) {
        grid_h = tensor->shape[2];
        grid_w = tensor->shape[3];
        return true;
    }
    if (is_nhwc4_tensor(tensor, kDirectRawChannels)) {
        grid_h = tensor->shape[1];
        grid_w = tensor->shape[2];
        return true;
    }

    int count = tensor_candidate_count(tensor, kDirectRawChannels);
    int side = (int)std::lround(std::sqrt((float)count));
    if (side <= 0 || side * side != count) {
        return false;
    }
    grid_w = side;
    grid_h = side;
    return true;
}

static float tensor_scale(const dl::TensorBase *tensor)
{
    return tensor ? DL_SCALE(tensor->exponent) : 0.0f;
}

static float tensor_raw_value(const dl::TensorBase *tensor, int index)
{
    if (!tensor || !tensor->data || index < 0 || index >= tensor->size) {
        return 0.0f;
    }

    switch (tensor->dtype) {
    case dl::DATA_TYPE_FLOAT:
        return static_cast<float *>(tensor->data)[index];
    case dl::DATA_TYPE_INT8:
        return dl::dequantize<int8_t, float>(static_cast<int8_t *>(tensor->data)[index], tensor_scale(tensor));
    case dl::DATA_TYPE_UINT8:
        return dl::dequantize<int8_t, float>(static_cast<int8_t *>(tensor->data)[index], tensor_scale(tensor));
    case dl::DATA_TYPE_INT16:
        return dl::dequantize<int16_t, float>(static_cast<int16_t *>(tensor->data)[index], tensor_scale(tensor));
    case dl::DATA_TYPE_UINT16:
        return dl::dequantize<int16_t, float>(static_cast<int16_t *>(tensor->data)[index], tensor_scale(tensor));
    default:
        return 0.0f;
    }
}

static int tensor_raw_i8(const dl::TensorBase *tensor, int index)
{
    if (!tensor || !tensor->data || index < 0 || index >= tensor->size) {
        return 0;
    }
    if (tensor->dtype == dl::DATA_TYPE_INT8 || tensor->dtype == dl::DATA_TYPE_UINT8) {
        return static_cast<int8_t *>(tensor->data)[index];
    }
    return 0;
}

static float tensor_value_3d(const dl::TensorBase *tensor, int channels, int candidate, int channel)
{
    if (!tensor || candidate < 0 || channel < 0 || channel >= channels) {
        return 0.0f;
    }

    if (is_chw_tensor(tensor, channels)) {
        int count = tensor->shape[2];
        return tensor_raw_value(tensor, channel * count + candidate);
    }
    if (is_hwc_tensor(tensor, channels)) {
        return tensor_raw_value(tensor, candidate * channels + channel);
    }
    return 0.0f;
}

static bool is_flat_classification_tensor(const dl::TensorBase *tensor)
{
    return tensor && tensor->size == kClassificationChannels &&
           ((tensor->shape.size() == 1 && tensor->shape[0] == kClassificationChannels) ||
            (tensor->shape.size() == 2 && tensor->shape[0] == 1 && tensor->shape[1] == kClassificationChannels) ||
            (tensor->shape.size() == 3 && tensor->shape[0] == 1 &&
             ((tensor->shape[1] == kClassificationChannels && tensor->shape[2] == 1) ||
              (tensor->shape[1] == 1 && tensor->shape[2] == kClassificationChannels))) ||
            (tensor->shape.size() == 4 && tensor->shape[0] == 1 &&
             ((tensor->shape[1] == kClassificationChannels && tensor->shape[2] == 1 && tensor->shape[3] == 1) ||
              (tensor->shape[1] == 1 && tensor->shape[2] == 1 && tensor->shape[3] == kClassificationChannels))));
}

static dl::TensorBase *find_classification_output(void)
{
    if (!s_model) {
        return nullptr;
    }

    std::map<std::string, dl::TensorBase *> &outputs = s_model->get_outputs();
    for (auto &item : outputs) {
        dl::TensorBase *tensor = item.second;
        if (is_flat_classification_tensor(tensor)) {
            return tensor;
        }
    }
    return nullptr;
}

static float tensor_classification_logit(const dl::TensorBase *tensor, int class_id)
{
    if (!is_flat_classification_tensor(tensor) || class_id < 0 || class_id >= kClassificationChannels) {
        return 0.0f;
    }
    return tensor_raw_value(tensor, class_id);
}

static helmet_vision_bbox_t make_full_frame_bbox(uint16_t width, uint16_t height)
{
    helmet_vision_bbox_t bbox = {};
    bbox.x = 0;
    bbox.y = 0;
    bbox.w = width;
    bbox.h = height;
    return bbox;
}

static bool postprocess_classification_output(const dl::TensorBase *tensor,
                                              uint16_t width,
                                              uint16_t height,
                                              eye_state_detector_result_t &result)
{
    if (!is_flat_classification_tensor(tensor)) {
        return false;
    }

    float logits[kClassificationChannels] = {
        tensor_classification_logit(tensor, kOpenClass),
        tensor_classification_logit(tensor, kClosedClass),
        tensor_classification_logit(tensor, kBackgroundClass),
    };
    float max_logit = std::max(logits[0], std::max(logits[1], logits[2]));
    float exps[kClassificationChannels] = {
        std::exp(logits[0] - max_logit),
        std::exp(logits[1] - max_logit),
        std::exp(logits[2] - max_logit),
    };
    float denom = exps[0] + exps[1] + exps[2];
    if (denom <= 0.0f || !std::isfinite(denom)) {
        result = make_invalid_result(HELMET_VISION_EYE_STATE_INVALID, HELMET_VISION_REASON_LOW_CONFIDENCE);
        return true;
    }

    float open_score = exps[kOpenClass] / denom;
    float closed_score = exps[kClosedClass] / denom;
    float background_score = exps[kBackgroundClass] / denom;
    int best_class = kOpenClass;
    float best_confidence = open_score;
    if (closed_score > best_confidence) {
        best_class = kClosedClass;
        best_confidence = closed_score;
    }
    if (background_score > best_confidence) {
        best_class = kBackgroundClass;
        best_confidence = background_score;
    }

    if (best_class == kBackgroundClass) {
        result = make_invalid_result(HELMET_VISION_EYE_STATE_INVALID, HELMET_VISION_REASON_NO_DETECTION);
        result.confidence = clamp_float_local(background_score, 0.0f, 1.0f);
        result.open_score = clamp_float_local(open_score, 0.0f, 1.0f);
        result.closed_score = clamp_float_local(closed_score, 0.0f, 1.0f);
        result.background_score = clamp_float_local(background_score, 0.0f, 1.0f);
        ESP_LOGD(TAG,
                 "classification background open=%.3f closed=%.3f background=%.3f",
                 open_score,
                 closed_score,
                 background_score);
        return true;
    }

    if (best_confidence < kMinConfidence) {
        result = make_invalid_result(HELMET_VISION_EYE_STATE_INVALID, HELMET_VISION_REASON_LOW_CONFIDENCE);
        result.confidence = clamp_float_local(best_confidence, 0.0f, 1.0f);
        result.open_score = clamp_float_local(open_score, 0.0f, 1.0f);
        result.closed_score = clamp_float_local(closed_score, 0.0f, 1.0f);
        result.background_score = clamp_float_local(background_score, 0.0f, 1.0f);
        ESP_LOGD(TAG,
                 "classification low confidence open=%.3f closed=%.3f background=%.3f",
                 open_score,
                 closed_score,
                 background_score);
        return true;
    }

    bool closed = best_class == kClosedClass;
    float eye_sum = open_score + closed_score;
    float open_ratio = eye_sum > 0.0f ? open_score / eye_sum : (closed ? 0.0f : 1.0f);

    result = {};
    result.valid = true;
    result.closed = closed;
    result.eye_state = closed ? HELMET_VISION_EYE_STATE_CLOSED : HELMET_VISION_EYE_STATE_OPEN;
    result.reason = HELMET_VISION_REASON_NONE;
    result.bbox = make_full_frame_bbox(width, height);
    result.open_ratio = clamp_float_local(open_ratio, 0.0f, 1.0f);
    result.confidence = clamp_float_local(best_confidence, 0.0f, 1.0f);
    result.open_score = clamp_float_local(open_score, 0.0f, 1.0f);
    result.closed_score = clamp_float_local(closed_score, 0.0f, 1.0f);
    result.background_score = clamp_float_local(background_score, 0.0f, 1.0f);

    ESP_LOGD(TAG,
             "classification open=%.3f closed=%.3f background=%.3f state=%s conf=%.3f",
             open_score,
             closed_score,
             background_score,
             closed ? "closed" : "open",
             best_confidence);
    return true;
}

static int tensor_raw_i8_3d(const dl::TensorBase *tensor, int channels, int candidate, int channel)
{
    if (!tensor || candidate < 0 || channel < 0 || channel >= channels) {
        return 0;
    }

    if (is_chw_tensor(tensor, channels)) {
        int count = tensor->shape[2];
        return tensor_raw_i8(tensor, channel * count + candidate);
    }
    if (is_hwc_tensor(tensor, channels)) {
        return tensor_raw_i8(tensor, candidate * channels + channel);
    }
    return 0;
}

static float tensor_score_3d(const dl::TensorBase *tensor, int candidate, int channel)
{
    float value = tensor_value_3d(tensor, kExpectedScoreChannels, candidate, channel);
    return clamp_float_local(value, 0.0f, 1.0f);
}

static float tensor_value_direct_raw(const dl::TensorBase *tensor, int candidate, int channel)
{
    if (!tensor || candidate < 0 || channel < 0 || channel >= kDirectRawChannels) {
        return 0.0f;
    }

    if (is_nchw4_tensor(tensor, kDirectRawChannels)) {
        int height = tensor->shape[2];
        int width = tensor->shape[3];
        if (candidate >= height * width) {
            return 0.0f;
        }
        int y = candidate / width;
        int x = candidate % width;
        return tensor_raw_value(tensor, (channel * height + y) * width + x);
    }

    if (is_nhwc4_tensor(tensor, kDirectRawChannels)) {
        int height = tensor->shape[1];
        int width = tensor->shape[2];
        if (candidate >= height * width) {
            return 0.0f;
        }
        int y = candidate / width;
        int x = candidate % width;
        return tensor_raw_value(tensor, (y * width + x) * kDirectRawChannels + channel);
    }

    return tensor_value_3d(tensor, kDirectRawChannels, candidate, channel);
}

static float sigmoid_float(float value)
{
    if (value >= 0.0f) {
        float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(value);
    return z / (1.0f + z);
}

static bool read_direct_raw_candidate(const dl::TensorBase *tensor,
                                      int candidate,
                                      float &center_x,
                                      float &center_y,
                                      float &box_w,
                                      float &box_h,
                                      float &open_score,
                                      float &closed_score)
{
    int grid_w = 0;
    int grid_h = 0;
    if (!direct_raw_grid_dims(tensor, grid_w, grid_h) || candidate < 0 || candidate >= grid_w * grid_h) {
        return false;
    }

    int cell_x = candidate % grid_w;
    int cell_y = candidate / grid_w;
    float stride_x = (float)kInputSize / (float)grid_w;
    float stride_y = (float)kInputSize / (float)grid_h;

    center_x = ((float)cell_x + sigmoid_float(tensor_value_direct_raw(tensor, candidate, kDirectRawTx))) * stride_x;
    center_y = ((float)cell_y + sigmoid_float(tensor_value_direct_raw(tensor, candidate, kDirectRawTy))) * stride_y;
    box_w = sigmoid_float(tensor_value_direct_raw(tensor, candidate, kDirectRawTw)) * (float)kInputSize;
    box_h = sigmoid_float(tensor_value_direct_raw(tensor, candidate, kDirectRawTh)) * (float)kInputSize;

    float objectness = sigmoid_float(tensor_value_direct_raw(tensor, candidate, kDirectRawObj));
    open_score = objectness * sigmoid_float(tensor_value_direct_raw(tensor, candidate, kDirectRawOpen));
    closed_score = objectness * sigmoid_float(tensor_value_direct_raw(tensor, candidate, kDirectRawClosed));
    return std::isfinite(center_x) && std::isfinite(center_y) && std::isfinite(box_w) && std::isfinite(box_h) &&
           std::isfinite(open_score) && std::isfinite(closed_score);
}

static dl::TensorBase *find_direct_raw_output(void)
{
    if (!s_model) {
        return nullptr;
    }

    std::map<std::string, dl::TensorBase *> &outputs = s_model->get_outputs();
    for (auto &item : outputs) {
        dl::TensorBase *tensor = item.second;
        if (direct_raw_candidate_count(tensor) > 0) {
            return tensor;
        }
    }
    return nullptr;
}

static int box_tensor_channels(const dl::TensorBase *tensor)
{
    if (is_chw_tensor(tensor, kRawDflBoxChannels) || is_hwc_tensor(tensor, kRawDflBoxChannels)) {
        return kRawDflBoxChannels;
    }
    if (is_chw_tensor(tensor, kDecodedBoxChannels) || is_hwc_tensor(tensor, kDecodedBoxChannels)) {
        return kDecodedBoxChannels;
    }
    return 0;
}

static bool find_model_outputs(dl::TensorBase **out_boxes, dl::TensorBase **out_scores)
{
    if (!s_model || !out_boxes || !out_scores) {
        return false;
    }

    *out_boxes = nullptr;
    *out_scores = nullptr;

    std::map<std::string, dl::TensorBase *> &outputs = s_model->get_outputs();
    for (auto &item : outputs) {
        dl::TensorBase *tensor = item.second;
        if (!tensor) {
            continue;
        }
        if (!*out_boxes && box_tensor_channels(tensor) > 0) {
            *out_boxes = tensor;
        } else if (!*out_scores &&
                   (is_chw_tensor(tensor, kExpectedScoreChannels) || is_hwc_tensor(tensor, kExpectedScoreChannels))) {
            *out_scores = tensor;
        }
    }

    int box_channels = box_tensor_channels(*out_boxes);
    return box_channels > 0 && *out_scores != nullptr &&
           tensor_candidate_count(*out_boxes, box_channels) == tensor_candidate_count(*out_scores, kExpectedScoreChannels);
}

static bool yolo_anchor_for_candidate(int candidate, float &anchor_x, float &anchor_y, float &stride)
{
    static constexpr int kLevel0 = 40 * 40;
    static constexpr int kLevel1 = 20 * 20;
    static constexpr int kLevel2 = 10 * 10;

    int index = candidate;
    int grid = 40;
    stride = 8.0f;
    if (index >= kLevel0) {
        index -= kLevel0;
        grid = 20;
        stride = 16.0f;
    }
    if (index >= kLevel1) {
        index -= kLevel1;
        grid = 10;
        stride = 32.0f;
    }
    int cells = grid * grid;
    if (index < 0 || index >= cells) {
        return false;
    }

    anchor_x = (float)(index % grid) + 0.5f;
    anchor_y = (float)(index / grid) + 0.5f;
    return true;
}

static float dfl_expected_value(const dl::TensorBase *tensor, int candidate, int side)
{
    float max_logit = -3.402823466e+38F;
    for (int bin = 0; bin < kDflBins; ++bin) {
        float logit = tensor_value_3d(tensor, kRawDflBoxChannels, candidate, side * kDflBins + bin);
        if (std::isfinite(logit)) {
            max_logit = std::max(max_logit, logit);
        }
    }
    if (!std::isfinite(max_logit)) {
        return 0.0f;
    }

    float sum = 0.0f;
    float weighted = 0.0f;
    for (int bin = 0; bin < kDflBins; ++bin) {
        float weight = std::exp(tensor_value_3d(tensor, kRawDflBoxChannels, candidate, side * kDflBins + bin) - max_logit);
        sum += weight;
        weighted += weight * (float)bin;
    }
    return sum > 0.0f ? weighted / sum : 0.0f;
}

static bool decode_yolo_dfl_box(const dl::TensorBase *tensor,
                                int candidate,
                                float &center_x,
                                float &center_y,
                                float &box_w,
                                float &box_h)
{
    float anchor_x = 0.0f;
    float anchor_y = 0.0f;
    float stride = 0.0f;
    if (!yolo_anchor_for_candidate(candidate, anchor_x, anchor_y, stride)) {
        return false;
    }

    float left = dfl_expected_value(tensor, candidate, 0);
    float top = dfl_expected_value(tensor, candidate, 1);
    float right = dfl_expected_value(tensor, candidate, 2);
    float bottom = dfl_expected_value(tensor, candidate, 3);

    float x1 = (anchor_x - left) * stride;
    float y1 = (anchor_y - top) * stride;
    float x2 = (anchor_x + right) * stride;
    float y2 = (anchor_y + bottom) * stride;

    center_x = (x1 + x2) * 0.5f;
    center_y = (y1 + y2) * 0.5f;
    box_w = x2 - x1;
    box_h = y2 - y1;
    return std::isfinite(center_x) && std::isfinite(center_y) && std::isfinite(box_w) && std::isfinite(box_h);
}

static bool read_box(const dl::TensorBase *tensor,
                     int box_channels,
                     int candidate,
                     float &center_x,
                     float &center_y,
                     float &box_w,
                     float &box_h)
{
    if (box_channels == kRawDflBoxChannels) {
        return decode_yolo_dfl_box(tensor, candidate, center_x, center_y, box_w, box_h);
    }
    if (box_channels == kDecodedBoxChannels) {
        center_x = tensor_value_3d(tensor, kDecodedBoxChannels, candidate, 0);
        center_y = tensor_value_3d(tensor, kDecodedBoxChannels, candidate, 1);
        box_w = tensor_value_3d(tensor, kDecodedBoxChannels, candidate, 2);
        box_h = tensor_value_3d(tensor, kDecodedBoxChannels, candidate, 3);
        return true;
    }
    return false;
}

static helmet_vision_bbox_t make_bbox(float center_x,
                                       float center_y,
                                       float box_w,
                                       float box_h,
                                       uint16_t width,
                                       uint16_t height)
{
    helmet_vision_bbox_t bbox = {};
    float scale_x = s_preprocessor ? s_preprocessor->get_resize_scale_x() : 0.0f;
    float scale_y = s_preprocessor ? s_preprocessor->get_resize_scale_y() : 0.0f;
    if (scale_x <= 0.0f) {
        scale_x = (float)kInputSize / (float)width;
    }
    if (scale_y <= 0.0f) {
        scale_y = (float)kInputSize / (float)height;
    }

    float x1 = (center_x - box_w * 0.5f) / scale_x;
    float y1 = (center_y - box_h * 0.5f) / scale_y;
    float x2 = (center_x + box_w * 0.5f) / scale_x;
    float y2 = (center_y + box_h * 0.5f) / scale_y;

    x1 = clamp_float_local(x1, 0.0f, (float)width - 1.0f);
    y1 = clamp_float_local(y1, 0.0f, (float)height - 1.0f);
    x2 = clamp_float_local(x2, 0.0f, (float)width - 1.0f);
    y2 = clamp_float_local(y2, 0.0f, (float)height - 1.0f);

    if (x2 <= x1 || y2 <= y1) {
        return {};
    }

    bbox.x = (uint16_t)std::lround(x1);
    bbox.y = (uint16_t)std::lround(y1);
    bbox.w = (uint16_t)std::max(1L, std::lround(x2 - x1));
    bbox.h = (uint16_t)std::max(1L, std::lround(y2 - y1));
    return bbox;
}

static eye_state_detector_result_t postprocess_outputs(uint16_t width, uint16_t height)
{
    dl::TensorBase *classification = find_classification_output();
    if (classification) {
        eye_state_detector_result_t result = {};
        if (postprocess_classification_output(classification, width, height, result)) {
            return result;
        }
    }

    dl::TensorBase *raw = find_direct_raw_output();
    if (raw) {
        int candidate_count = direct_raw_candidate_count(raw);
        float best_confidence = 0.0f;
        float best_open = 0.0f;
        float best_closed = 0.0f;
        float best_center_x = 0.0f;
        float best_center_y = 0.0f;
        float best_box_w = 0.0f;
        float best_box_h = 0.0f;
        int best_index = -1;

        for (int i = 0; i < candidate_count; ++i) {
            float center_x = 0.0f;
            float center_y = 0.0f;
            float box_w = 0.0f;
            float box_h = 0.0f;
            float open_score = 0.0f;
            float closed_score = 0.0f;
            if (!read_direct_raw_candidate(raw, i, center_x, center_y, box_w, box_h, open_score, closed_score)) {
                continue;
            }
            if (box_w < kMinBoxSize || box_h < kMinBoxSize) {
                continue;
            }

            float confidence = std::max(open_score, closed_score);
            if (confidence > best_confidence) {
                best_confidence = confidence;
                best_open = open_score;
                best_closed = closed_score;
                best_center_x = center_x;
                best_center_y = center_y;
                best_box_w = box_w;
                best_box_h = box_h;
                best_index = i;
            }
        }

        if (best_index < 0) {
            return make_invalid_result(HELMET_VISION_EYE_STATE_INVALID, HELMET_VISION_REASON_NO_DETECTION);
        }
        if (best_confidence < kMinConfidence) {
            return make_invalid_result(HELMET_VISION_EYE_STATE_INVALID, HELMET_VISION_REASON_LOW_CONFIDENCE);
        }

        helmet_vision_bbox_t bbox = make_bbox(best_center_x, best_center_y, best_box_w, best_box_h, width, height);
        if (bbox.w == 0 || bbox.h == 0) {
            return make_invalid_result(HELMET_VISION_EYE_STATE_INVALID, HELMET_VISION_REASON_NO_DETECTION);
        }

        float sum = best_open + best_closed;
        float open_ratio = sum > 0.0f ? best_open / sum : (best_closed > best_open ? 0.0f : 1.0f);
        bool closed = best_closed > best_open;

        eye_state_detector_result_t result = {};
        result.valid = true;
        result.closed = closed;
        result.eye_state = closed ? HELMET_VISION_EYE_STATE_CLOSED : HELMET_VISION_EYE_STATE_OPEN;
        result.reason = HELMET_VISION_REASON_NONE;
        result.bbox = bbox;
        result.open_ratio = clamp_float_local(open_ratio, 0.0f, 1.0f);
        result.confidence = clamp_float_local(best_confidence, 0.0f, 1.0f);
        result.open_score = clamp_float_local(best_open, 0.0f, 1.0f);
        result.closed_score = clamp_float_local(best_closed, 0.0f, 1.0f);
        result.background_score = 0.0f;

        ESP_LOGD(TAG,
                 "direct raw eye candidate idx=%d open=%.3f closed=%.3f conf=%.3f bbox=%u,%u,%u,%u",
                 best_index,
                 best_open,
                 best_closed,
                 best_confidence,
                 bbox.x,
                 bbox.y,
                 bbox.w,
                 bbox.h);

        return result;
    }

    dl::TensorBase *boxes = nullptr;
    dl::TensorBase *scores = nullptr;
    if (!find_model_outputs(&boxes, &scores)) {
        ESP_LOGW(TAG, "unexpected eye model outputs");
        return make_invalid_result(HELMET_VISION_EYE_STATE_INVALID, HELMET_VISION_REASON_NO_DETECTION);
    }

    int box_channels = box_tensor_channels(boxes);
    int candidate_count = tensor_candidate_count(boxes, box_channels);
    float best_confidence = 0.0f;
    float best_open = 0.0f;
    float best_closed = 0.0f;
    float best_center_x = 0.0f;
    float best_center_y = 0.0f;
    float best_box_w = 0.0f;
    float best_box_h = 0.0f;
    int best_index = -1;

    for (int i = 0; i < candidate_count; ++i) {
        float center_x = 0.0f;
        float center_y = 0.0f;
        float box_w = 0.0f;
        float box_h = 0.0f;
        if (!read_box(boxes, box_channels, i, center_x, center_y, box_w, box_h)) {
            continue;
        }
        if (!std::isfinite(center_x) || !std::isfinite(center_y) || !std::isfinite(box_w) || !std::isfinite(box_h) ||
            box_w < kMinBoxSize || box_h < kMinBoxSize) {
            continue;
        }

        float open_score = tensor_score_3d(scores, i, kOpenClass);
        float closed_score = tensor_score_3d(scores, i, kClosedClass);
        float confidence = std::max(open_score, closed_score);
        if (confidence > best_confidence) {
            best_confidence = confidence;
            best_open = open_score;
            best_closed = closed_score;
            best_center_x = center_x;
            best_center_y = center_y;
            best_box_w = box_w;
            best_box_h = box_h;
            best_index = i;
        }
    }

    if (best_index < 0) {
        return make_invalid_result(HELMET_VISION_EYE_STATE_INVALID, HELMET_VISION_REASON_NO_DETECTION);
    }
    if (best_confidence < kMinConfidence) {
        return make_invalid_result(HELMET_VISION_EYE_STATE_INVALID, HELMET_VISION_REASON_LOW_CONFIDENCE);
    }

    helmet_vision_bbox_t bbox = make_bbox(best_center_x, best_center_y, best_box_w, best_box_h, width, height);
    if (bbox.w == 0 || bbox.h == 0) {
        return make_invalid_result(HELMET_VISION_EYE_STATE_INVALID, HELMET_VISION_REASON_NO_DETECTION);
    }

    float sum = best_open + best_closed;
    float open_ratio = sum > 0.0f ? best_open / sum : (best_closed > best_open ? 0.0f : 1.0f);
    bool closed = best_closed > best_open;

    eye_state_detector_result_t result = {};
    result.valid = true;
    result.closed = closed;
    result.eye_state = closed ? HELMET_VISION_EYE_STATE_CLOSED : HELMET_VISION_EYE_STATE_OPEN;
    result.reason = HELMET_VISION_REASON_NONE;
    result.bbox = bbox;
    result.open_ratio = clamp_float_local(open_ratio, 0.0f, 1.0f);
    result.confidence = clamp_float_local(best_confidence, 0.0f, 1.0f);
    result.open_score = clamp_float_local(best_open, 0.0f, 1.0f);
    result.closed_score = clamp_float_local(best_closed, 0.0f, 1.0f);
    result.background_score = 0.0f;

    ESP_LOGD(TAG,
             "eye candidate idx=%d raw_open=%d raw_closed=%d open=%.3f closed=%.3f conf=%.3f bbox=%u,%u,%u,%u",
             best_index,
             tensor_raw_i8_3d(scores, kExpectedScoreChannels, best_index, kOpenClass),
             tensor_raw_i8_3d(scores, kExpectedScoreChannels, best_index, kClosedClass),
             best_open,
             best_closed,
             best_confidence,
             bbox.x,
             bbox.y,
             bbox.w,
             bbox.h);

    return result;
}
#endif

esp_err_t eye_state_detector_init(void)
{
    if (!s_initialized) {
        esp_err_t lock_ret = helmet_dl_runtime_lock_init();
        if (lock_ret != ESP_OK) {
            ESP_LOGE(TAG, "init dl runtime lock failed: %s", esp_err_to_name(lock_ret));
            return lock_ret;
        }

        s_initialized = true;
#if HELMET_EYE_STATE_MODEL_AVAILABLE
        lock_ret = helmet_dl_runtime_lock_acquire(10000);
        if (lock_ret != ESP_OK) {
            ESP_LOGE(TAG, "acquire dl runtime lock for model init failed: %s", esp_err_to_name(lock_ret));
            s_initialized = false;
            return lock_ret;
        }

#if HELMET_EYE_STATE_MODEL_EMBEDDED
        s_model = new (std::nothrow) dl::Model((const char *)eye_state_model_start,
                                               fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                                               kEspDlInternalArenaBytes,
                                               dl::MEMORY_MANAGER_GREEDY,
                                               nullptr,
                                               false);
#else
        log_model_partition();
        s_model = new (std::nothrow) dl::Model(kModelPartition,
                                               fbs::MODEL_LOCATION_IN_FLASH_PARTITION,
                                               kEspDlInternalArenaBytes,
                                               dl::MEMORY_MANAGER_GREEDY,
                                               nullptr,
                                               false);
#endif
        if (!s_model) {
            helmet_dl_runtime_lock_release();
            s_initialized = false;
            ESP_LOGE(TAG, "create eye-state model failed");
            return ESP_ERR_NO_MEM;
        }

        if (s_model->get_inputs().size() != 1 || s_model->get_outputs().empty()) {
            ESP_LOGE(TAG,
                     "eye-state model is not loadable from %s; inputs=%u outputs=%u",
                     HELMET_EYE_STATE_MODEL_EMBEDDED ? "rodata" : kModelPartition,
                     static_cast<unsigned>(s_model->get_inputs().size()),
                     static_cast<unsigned>(s_model->get_outputs().size()));
            delete s_model;
            s_model = nullptr;
            helmet_dl_runtime_lock_release();
            s_initialized = false;
            return ESP_OK;
        }

        s_preprocessor = new (std::nothrow)
            dl::image::ImagePreprocessor(s_model, {0, 0, 0}, {255, 255, 255}, DL_IMAGE_CAP_RGB565_BIG_ENDIAN);
        if (!s_preprocessor) {
            delete s_model;
            s_model = nullptr;
            helmet_dl_runtime_lock_release();
            s_initialized = false;
            ESP_LOGE(TAG, "create eye-state preprocessor failed");
            return ESP_ERR_NO_MEM;
        }

        helmet_dl_runtime_lock_release();
        ESP_LOGI(TAG,
                 "eye-state ESP-DL model loaded from %s",
                 HELMET_EYE_STATE_MODEL_EMBEDDED ? "rodata" : kModelPartition);
#else
        ESP_LOGW(TAG, "eye-state ESP-DL model is not configured; vision will report NO_MODEL");
#endif
    }
    return ESP_OK;
}

bool eye_state_detector_model_ready(void)
{
#if HELMET_EYE_STATE_MODEL_AVAILABLE
    return s_model != nullptr && s_preprocessor != nullptr;
#else
    return false;
#endif
}

eye_state_detector_result_t eye_state_detector_detect(const uint8_t *rgb565,
                                                       uint16_t width,
                                                       uint16_t height,
                                                       size_t stride_bytes)
{
    (void)stride_bytes;

#if HELMET_EYE_STATE_MODEL_AVAILABLE
    if (!s_model || !s_preprocessor) {
        return make_invalid_result(HELMET_VISION_EYE_STATE_NO_MODEL, HELMET_VISION_REASON_NO_MODEL);
    }
    if (!rgb565 || width == 0 || height == 0) {
        return make_invalid_result(HELMET_VISION_EYE_STATE_INVALID, HELMET_VISION_REASON_NO_DETECTION);
    }

    if (helmet_dl_runtime_lock_acquire(0) != ESP_OK) {
        return make_invalid_result(HELMET_VISION_EYE_STATE_INVALID, HELMET_VISION_REASON_RUNTIME_BUSY);
    }

    dl::image::img_t img = {
        .data = const_cast<uint8_t *>(rgb565),
        .width = width,
        .height = height,
        .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565,
    };
    s_preprocessor->preprocess(img);
    s_model->run();

    eye_state_detector_result_t result = postprocess_outputs(width, height);
    helmet_dl_runtime_lock_release();
    return result;
#else
    (void)rgb565;
    (void)width;
    (void)height;
    return make_invalid_result(HELMET_VISION_EYE_STATE_NO_MODEL, HELMET_VISION_REASON_NO_MODEL);
#endif
}
