#include "helmet_vision.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "bsp/esp-bsp.h"
#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "esp_cache.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_private/esp_cache_private.h"
#include "esp_timer.h"
#include "esp_video_device.h"
#include "esp_video_init.h"
#include "eye_state_detector.hpp"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "helmet_state.h"
#include "helmet_voice.h"
#include "linux/videodev2.h"

static const char *TAG = "helmet_vision";

static constexpr int kFrameBufferCount = 2;
static constexpr uint16_t kTargetFrameWidth = 640;
static constexpr uint16_t kTargetFrameHeight = 480;
static constexpr int kVisionTaskStack = 8192;
static constexpr int kVisionTaskPriority = 4;
static constexpr int kDetectorTaskStack = 8192;
static constexpr int kDetectorTaskPriority = 2;
#if CONFIG_FREERTOS_NUMBER_OF_CORES > 1
static constexpr BaseType_t kVisionTaskCore = 0;
static constexpr BaseType_t kDetectorTaskCore = 1;
#else
static constexpr BaseType_t kVisionTaskCore = 0;
static constexpr BaseType_t kDetectorTaskCore = 0;
#endif
static constexpr int kInferIntervalMs = 100;
static constexpr int kEyeStateIntervalMs = 100;
static constexpr int kPerclosWindowSamples = 300;
static constexpr int64_t kNoEyeTimeoutUs = 3000000;
static constexpr float kFatiguePerclosThreshold = 0.45f;
static constexpr int64_t kFatigueAlarmRetryUs = 5000000;
static constexpr int64_t kFatigueAlarmRepeatUs = 30000000;

static SemaphoreHandle_t s_lock;
static TaskHandle_t s_task;
static TaskHandle_t s_detector_task;
static SemaphoreHandle_t s_infer_signal;
static bool s_video_inited;
static int s_video_fd = -1;
static uint8_t *s_frame_buffers[kFrameBufferCount];
static uint8_t *s_latest_frame;
static uint8_t *s_infer_frame;
static size_t s_frame_size;
static size_t s_infer_frame_size;
static size_t s_cache_line_size = 64;
static helmet_vision_status_t s_status;

static bool s_tracker_ready;
static uint8_t s_closed_history[kPerclosWindowSamples];
static int s_history_index;
static int s_history_count;
static int s_closed_count;
static int s_closed_streak;
static bool s_prev_closed;
static uint32_t s_blink_count;
static float s_smoothed_open;
static int64_t s_last_sample_us;
static int64_t s_last_infer_submit_us;
static int64_t s_last_valid_eye_us;
static int64_t s_eye_invalid_since_us;
static int64_t s_last_diag_us;
static int64_t s_last_infer_diag_us;
static int64_t s_last_fatigue_alarm_attempt_us;
static int64_t s_last_fatigue_alarm_ok_us;
static int64_t s_last_fatigue_alarm_block_us;
static bool s_fatigue_alarm_latched;
static bool s_eye_state_published;
static bool s_infer_pending;
static bool s_infer_running;
static uint16_t s_infer_width;
static uint16_t s_infer_height;
static uint32_t s_infer_frame_seq;
static uint32_t s_infer_skip_count;

static float clamp_float(float value, float low, float high)
{
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static void lock_state(void)
{
    if (s_lock) {
        xSemaphoreTake(s_lock, portMAX_DELAY);
    }
}

static void unlock_state(void)
{
    if (s_lock) {
        xSemaphoreGive(s_lock);
    }
}

static void log_heap_snapshot(const char *stage)
{
    ESP_LOGI(
        TAG,
        "heap %s int_free=%u int_largest=%u dma_free=%u dma_largest=%u psram_free=%u psram_largest=%u",
        stage ? stage : "?",
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
        (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA),
        (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA),
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
        (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM)
    );
}

static void reset_eye_tracker_locked(void)
{
    memset(s_closed_history, 0, sizeof(s_closed_history));
    s_history_index = 0;
    s_history_count = 0;
    s_closed_count = 0;
    s_closed_streak = 0;
    s_prev_closed = false;
    s_blink_count = 0;
    s_smoothed_open = 0.0f;
    s_tracker_ready = false;
}

static void clear_eye_business_state(void)
{
    if (s_eye_state_published) {
        s_eye_state_published = false;
        helmet_state_clear_eye();
        helmet_state_update_fusion();
    }
}

static void update_fatigue_alarm_from_state(void)
{
    helmet_state_t state = helmet_state_get_copy();
    bool fatigue_source = state.eye.valid &&
                          (state.eye.perclos >= kFatiguePerclosThreshold || state.eye.yawn_detected);

    if (!fatigue_source) {
        s_fatigue_alarm_latched = false;
        return;
    }

    int64_t now = esp_timer_get_time();

    if (state.alarm_suppressed || state.risk != HELMET_RISK_FATIGUE || !state.alarm_active) {
        if (s_last_fatigue_alarm_block_us == 0 ||
            (now - s_last_fatigue_alarm_block_us) >= 3000000) {
            s_last_fatigue_alarm_block_us = now;
            ESP_LOGW(TAG,
                     "fatigue alarm blocked: risk=%s active=%d suppressed=%d perclos=%.2f",
                     helmet_risk_to_string(state.risk),
                     state.alarm_active ? 1 : 0,
                     state.alarm_suppressed ? 1 : 0,
                     state.eye.perclos);
        }
        return;
    }

    bool repeat_due = s_last_fatigue_alarm_ok_us == 0 ||
                      (now - s_last_fatigue_alarm_ok_us) >= kFatigueAlarmRepeatUs;
    bool retry_due = s_last_fatigue_alarm_attempt_us == 0 ||
                     (now - s_last_fatigue_alarm_attempt_us) >= kFatigueAlarmRetryUs;

    if (s_fatigue_alarm_latched && !repeat_due) {
        return;
    }
    if (!retry_due) {
        return;
    }

    s_last_fatigue_alarm_attempt_us = now;
    esp_err_t voice_ret = helmet_voice_play_alert(HELMET_VOICE_ALERT_FATIGUE);
    if (voice_ret == ESP_OK) {
        s_last_fatigue_alarm_ok_us = now;
        s_fatigue_alarm_latched = true;
    }

    ESP_LOGW(TAG,
             "fatigue alarm trigger: perclos=%.2f blink=%" PRIu32 " voice=%s",
             state.eye.perclos,
             state.eye.blink_count,
             esp_err_to_name(voice_ret));
}

static void mark_eye_invalid(helmet_vision_reason_t reason,
                             helmet_vision_eye_state_t eye_state,
                             float confidence,
                             float open_score,
                             float closed_score,
                             float background_score)
{
    if (reason == HELMET_VISION_REASON_RUNTIME_BUSY) {
        return;
    }

    int64_t now = esp_timer_get_time();
    bool should_clear_business = false;

    lock_state();
    if (s_eye_invalid_since_us == 0) {
        s_eye_invalid_since_us = now;
    }

    uint32_t no_eye_ms = (uint32_t)((now - s_eye_invalid_since_us) / 1000);
    bool immediate_clear = (reason == HELMET_VISION_REASON_NO_MODEL || reason == HELMET_VISION_REASON_CAMERA_ERROR);
    bool timed_out = no_eye_ms >= (uint32_t)(kNoEyeTimeoutUs / 1000);

    s_status.model_ready = eye_state_detector_model_ready();
    s_status.eye_valid = false;
    s_status.eye_closed = false;
    s_status.eye_state = eye_state;
    s_status.last_reason = reason;
    s_status.eye_confidence = clamp_float(confidence, 0.0f, 1.0f);
    s_status.open_score = clamp_float(open_score, 0.0f, 1.0f);
    s_status.closed_score = clamp_float(closed_score, 0.0f, 1.0f);
    s_status.background_score = clamp_float(background_score, 0.0f, 1.0f);
    s_status.no_eye_ms = no_eye_ms;

    if (immediate_clear || timed_out) {
        reset_eye_tracker_locked();
        s_status.eye_open_ratio = 0.0f;
        s_status.perclos = 0.0f;
        s_status.blink_count = 0;
        s_status.eye_bbox = {};
        should_clear_business = s_eye_state_published;
    }
    unlock_state();

    if (immediate_clear || timed_out || should_clear_business) {
        clear_eye_business_state();
    }
}

static void update_eye_state_from_detector(const eye_state_detector_result_t &result)
{
    if (result.reason == HELMET_VISION_REASON_RUNTIME_BUSY) {
        return;
    }

    if (!result.valid) {
        helmet_vision_eye_state_t eye_state = result.eye_state == HELMET_VISION_EYE_STATE_UNKNOWN
                                                  ? HELMET_VISION_EYE_STATE_INVALID
                                                  : result.eye_state;
        mark_eye_invalid(result.reason,
                         eye_state,
                         result.confidence,
                         result.open_score,
                         result.closed_score,
                         result.background_score);
        return;
    }

    int64_t now = esp_timer_get_time();

    if (s_last_sample_us > 0 && (now - s_last_sample_us) < (int64_t)kEyeStateIntervalMs * 1000) {
        return;
    }
    s_last_sample_us = now;
    s_last_valid_eye_us = now;

    bool closed = result.closed || result.eye_state == HELMET_VISION_EYE_STATE_CLOSED;
    float perclos = 0.0f;
    uint32_t blink_count = 0;
    float raw_open = clamp_float(result.open_ratio, 0.0f, 1.0f);
    float smoothed = raw_open;
    helmet_vision_bbox_t bbox = result.bbox;
    uint16_t frame_width = 0;
    uint16_t frame_height = 0;

    lock_state();
    s_eye_invalid_since_us = 0;

    if (!s_tracker_ready) {
        memset(s_closed_history, 0, sizeof(s_closed_history));
        s_history_index = 0;
        s_history_count = 0;
        s_closed_count = 0;
        s_closed_streak = 0;
        s_prev_closed = false;
        s_blink_count = 0;
        s_smoothed_open = raw_open;
        s_tracker_ready = true;
    } else {
        s_smoothed_open = s_smoothed_open * 0.65f + raw_open * 0.35f;
    }

    smoothed = s_smoothed_open;

    if (s_history_count == kPerclosWindowSamples) {
        s_closed_count -= s_closed_history[s_history_index] ? 1 : 0;
    } else {
        s_history_count++;
    }
    s_closed_history[s_history_index] = closed ? 1 : 0;
    s_closed_count += closed ? 1 : 0;
    s_history_index = (s_history_index + 1) % kPerclosWindowSamples;

    if (closed) {
        s_closed_streak++;
    } else {
        if (s_prev_closed && s_closed_streak >= 1 && s_closed_streak <= 15) {
            s_blink_count++;
        }
        s_closed_streak = 0;
    }
    s_prev_closed = closed;

    perclos = s_history_count > 0 ? (float)s_closed_count / (float)s_history_count : 0.0f;
    blink_count = s_blink_count;

    s_status.eye_valid = true;
    s_status.eye_closed = closed;
    s_status.model_ready = true;
    s_status.eye_state = closed ? HELMET_VISION_EYE_STATE_CLOSED : HELMET_VISION_EYE_STATE_OPEN;
    s_status.last_reason = HELMET_VISION_REASON_NONE;
    s_status.eye_open_ratio = smoothed;
    s_status.eye_confidence = clamp_float(result.confidence, 0.0f, 1.0f);
    s_status.open_score = clamp_float(result.open_score, 0.0f, 1.0f);
    s_status.closed_score = clamp_float(result.closed_score, 0.0f, 1.0f);
    s_status.background_score = clamp_float(result.background_score, 0.0f, 1.0f);
    s_status.perclos = perclos;
    s_status.blink_count = blink_count;
    s_status.no_eye_ms = 0;
    s_status.eye_bbox = bbox;
    frame_width = s_status.frame_width;
    frame_height = s_status.frame_height;

    unlock_state();

    if ((now - s_last_diag_us) >= 5000000) {
        s_last_diag_us = now;
        ESP_LOGI(TAG,
                 "eye model frame=%ux%u bbox=%u,%u,%u,%u conf=%.2f open=%.2f perclos=%.2f closed=%d blink=%" PRIu32,
                 frame_width,
                 frame_height,
                 bbox.x,
                 bbox.y,
                 bbox.w,
                 bbox.h,
                 result.confidence,
                 smoothed,
                 perclos,
                 closed ? 1 : 0,
                 blink_count);
    }

    helmet_state_set_eye(smoothed, perclos, blink_count, false);
    s_eye_state_published = true;
    helmet_state_update_fusion();
    update_fatigue_alarm_from_state();
}

static void update_latest_frame(const uint8_t *frame)
{
    if (!frame || !s_latest_frame || s_frame_size == 0) {
        return;
    }

    lock_state();
    memcpy(s_latest_frame, frame, s_frame_size);
    s_status.frame_valid = true;
    s_status.frame_seq++;
    unlock_state();
}

static bool submit_inference_frame(const uint8_t *frame, uint16_t width, uint16_t height, int64_t now)
{
    if (!frame || !s_infer_signal || width == 0 || height == 0) {
        return false;
    }

    size_t frame_size = (size_t)width * height * sizeof(uint16_t);
    if (frame_size == 0) {
        return false;
    }

    lock_state();
    bool ready = s_infer_frame != NULL && s_infer_frame_size >= frame_size && !s_infer_pending && !s_infer_running &&
                 (s_last_infer_submit_us == 0 ||
                  (now - s_last_infer_submit_us) >= (int64_t)kInferIntervalMs * 1000);
    if (ready) {
        s_infer_pending = true;
        s_infer_width = width;
        s_infer_height = height;
        s_infer_frame_seq = s_status.frame_seq;
        s_last_infer_submit_us = now;
    } else if (s_infer_pending || s_infer_running) {
        s_infer_skip_count++;
    }
    unlock_state();

    if (!ready) {
        return false;
    }

    memcpy(s_infer_frame, frame, frame_size);
    xSemaphoreGive(s_infer_signal);
    return true;
}

static void detector_task(void *arg)
{
    (void)arg;

    while (true) {
        if (xSemaphoreTake(s_infer_signal, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        uint8_t *frame = NULL;
        uint16_t width = 0;
        uint16_t height = 0;
        uint32_t frame_seq = 0;

        lock_state();
        if (s_infer_pending && s_infer_frame != NULL && s_infer_width > 0 && s_infer_height > 0) {
            s_infer_pending = false;
            s_infer_running = true;
            frame = s_infer_frame;
            width = s_infer_width;
            height = s_infer_height;
            frame_seq = s_infer_frame_seq;
        }
        unlock_state();

        if (!frame) {
            continue;
        }

        int64_t start = esp_timer_get_time();
        eye_state_detector_result_t result =
            eye_state_detector_detect(frame, width, height, (size_t)width * sizeof(uint16_t));
        int64_t elapsed_us = esp_timer_get_time() - start;

        update_eye_state_from_detector(result);

        uint32_t skip_count = 0;
        lock_state();
        s_infer_running = false;
        s_status.last_infer_ms = (uint32_t)(elapsed_us / 1000);
        skip_count = s_infer_skip_count;
        s_infer_skip_count = 0;
        unlock_state();

        int64_t now = esp_timer_get_time();
        if (elapsed_us >= 150000 || (now - s_last_infer_diag_us) >= 5000000) {
            s_last_infer_diag_us = now;
            ESP_LOGI(TAG,
                     "eye infer frame=%" PRIu32 " time=%" PRIi64 "ms skipped=%" PRIu32,
                     frame_seq,
                     elapsed_us / 1000,
                     skip_count);
        }

        vTaskDelay(1);
    }
}

static esp_err_t init_video_once(void)
{
    if (s_video_inited) {
        return ESP_OK;
    }

#if CONFIG_EXAMPLE_ENABLE_MIPI_CSI_CAM_SENSOR
    esp_video_init_csi_config_t csi_config[] = {
        {
            .sccb_config = {
                .init_sccb = true,
                .i2c_config = {
                    .port = CONFIG_EXAMPLE_MIPI_CSI_SCCB_I2C_PORT,
                    .scl_pin = static_cast<gpio_num_t>(CONFIG_EXAMPLE_MIPI_CSI_SCCB_I2C_SCL_PIN),
                    .sda_pin = static_cast<gpio_num_t>(CONFIG_EXAMPLE_MIPI_CSI_SCCB_I2C_SDA_PIN),
                },
                .freq = CONFIG_EXAMPLE_MIPI_CSI_SCCB_I2C_FREQ,
            },
            .reset_pin = static_cast<gpio_num_t>(CONFIG_EXAMPLE_MIPI_CSI_CAM_SENSOR_RESET_PIN),
            .pwdn_pin = static_cast<gpio_num_t>(CONFIG_EXAMPLE_MIPI_CSI_CAM_SENSOR_PWDN_PIN),
        },
    };

    i2c_master_bus_handle_t i2c_bus_handle = bsp_i2c_get_handle();
    if (i2c_bus_handle != NULL) {
        csi_config[0].sccb_config.init_sccb = false;
        csi_config[0].sccb_config.i2c_handle = i2c_bus_handle;
    }
#endif

    esp_video_init_config_t cam_config = {};
#if CONFIG_EXAMPLE_ENABLE_MIPI_CSI_CAM_SENSOR
    cam_config.csi = csi_config;
#endif

    esp_err_t ret = esp_video_init(&cam_config);
    if (ret == ESP_OK) {
        s_video_inited = true;
        ESP_LOGI(TAG, "esp_video initialized");
    } else {
        ESP_LOGE(TAG, "esp_video_init failed: %s", esp_err_to_name(ret));
    }
    return ret;
}

static void free_camera_buffers(void)
{
    for (int i = 0; i < kFrameBufferCount; i++) {
        if (s_frame_buffers[i]) {
            heap_caps_free(s_frame_buffers[i]);
            s_frame_buffers[i] = NULL;
        }
    }
    if (s_latest_frame) {
        heap_caps_free(s_latest_frame);
        s_latest_frame = NULL;
    }
    s_frame_size = 0;
}

static esp_err_t allocate_camera_buffers(uint16_t width, uint16_t height)
{
    size_t frame_size = (size_t)width * height * sizeof(uint16_t);
    if (frame_size == 0) {
        return ESP_ERR_INVALID_SIZE;
    }

    if (s_frame_size == frame_size && s_latest_frame) {
        return ESP_OK;
    }

    free_camera_buffers();

    for (int i = 0; i < kFrameBufferCount; i++) {
        s_frame_buffers[i] = (uint8_t *)heap_caps_aligned_alloc(s_cache_line_size, frame_size, MALLOC_CAP_SPIRAM);
        ESP_RETURN_ON_FALSE(s_frame_buffers[i], ESP_ERR_NO_MEM, TAG, "alloc camera fb %d failed", i);
    }

    s_latest_frame = (uint8_t *)heap_caps_aligned_alloc(s_cache_line_size, frame_size, MALLOC_CAP_SPIRAM);
    ESP_RETURN_ON_FALSE(s_latest_frame, ESP_ERR_NO_MEM, TAG, "alloc latest frame failed");

    if (s_infer_frame_size < frame_size) {
        if (s_infer_frame) {
            heap_caps_free(s_infer_frame);
            s_infer_frame = NULL;
            s_infer_frame_size = 0;
        }
        s_infer_frame = (uint8_t *)heap_caps_aligned_alloc(s_cache_line_size, frame_size, MALLOC_CAP_SPIRAM);
        ESP_RETURN_ON_FALSE(s_infer_frame, ESP_ERR_NO_MEM, TAG, "alloc inference frame failed");
        s_infer_frame_size = frame_size;
    }

    s_frame_size = frame_size;
    ESP_LOGI(TAG, "allocated vision buffers count=%d frame=%u", kFrameBufferCount, (unsigned)frame_size);
    log_heap_snapshot("after vision buffer alloc");
    return ESP_OK;
}

static esp_err_t request_v4l2_buffers(int fd)
{
    struct v4l2_requestbuffers req = {};
    req.count = kFrameBufferCount;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_USERPTR;

    ESP_RETURN_ON_FALSE(ioctl(fd, VIDIOC_REQBUFS, &req) == 0, ESP_FAIL, TAG, "VIDIOC_REQBUFS failed errno=%d", errno);

    for (int i = 0; i < kFrameBufferCount; i++) {
        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_USERPTR;
        buf.index = i;

        ESP_RETURN_ON_FALSE(ioctl(fd, VIDIOC_QUERYBUF, &buf) == 0, ESP_FAIL, TAG, "VIDIOC_QUERYBUF failed errno=%d", errno);
        buf.m.userptr = (unsigned long)s_frame_buffers[i];
        buf.length = s_frame_size;
        ESP_RETURN_ON_FALSE(ioctl(fd, VIDIOC_QBUF, &buf) == 0, ESP_FAIL, TAG, "VIDIOC_QBUF failed errno=%d", errno);
    }
    return ESP_OK;
}

static esp_err_t open_camera(void)
{
    log_heap_snapshot("before video init");
    ESP_RETURN_ON_ERROR(init_video_once(), TAG, "video init failed");
    log_heap_snapshot("after video init");
    ESP_RETURN_ON_ERROR(esp_cache_get_alignment(MALLOC_CAP_SPIRAM, &s_cache_line_size), TAG, "get cache alignment failed");

    int fd = open(ESP_VIDEO_MIPI_CSI_DEVICE_NAME, O_RDONLY);
    if (fd < 0) {
        ESP_LOGE(TAG, "open %s failed errno=%d", ESP_VIDEO_MIPI_CSI_DEVICE_NAME, errno);
        return ESP_FAIL;
    }

    struct v4l2_capability capability = {};
    if (ioctl(fd, VIDIOC_QUERYCAP, &capability) != 0) {
        ESP_LOGE(TAG, "VIDIOC_QUERYCAP failed errno=%d", errno);
        close(fd);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "driver=%s card=%s bus=%s", capability.driver, capability.card, capability.bus_info);

    struct v4l2_format format = {};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_FMT, &format) != 0) {
        ESP_LOGE(TAG, "VIDIOC_G_FMT failed errno=%d", errno);
        close(fd);
        return ESP_FAIL;
    }

    uint32_t original_width = format.fmt.pix.width;
    uint32_t original_height = format.fmt.pix.height;
    uint32_t original_pixfmt = format.fmt.pix.pixelformat;

    if (original_width != kTargetFrameWidth || original_height != kTargetFrameHeight || original_pixfmt != V4L2_PIX_FMT_RGB565) {
        format.fmt.pix.width = kTargetFrameWidth;
        format.fmt.pix.height = kTargetFrameHeight;
        format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB565;
        if (ioctl(fd, VIDIOC_S_FMT, &format) != 0) {
            ESP_LOGW(
                TAG,
                "VIDIOC_S_FMT target %ux%u RGB565 failed errno=%d; fallback to driver default",
                kTargetFrameWidth,
                kTargetFrameHeight,
                errno
            );
            memset(&format, 0, sizeof(format));
            format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            format.fmt.pix.width = original_width;
            format.fmt.pix.height = original_height;
            format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB565;
            if (ioctl(fd, VIDIOC_S_FMT, &format) != 0) {
                ESP_LOGE(TAG, "VIDIOC_S_FMT fallback failed errno=%d", errno);
                close(fd);
                return ESP_FAIL;
            }
        }
    }

    memset(&format, 0, sizeof(format));
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_FMT, &format) != 0) {
        ESP_LOGE(TAG, "VIDIOC_G_FMT after set failed errno=%d", errno);
        close(fd);
        return ESP_FAIL;
    }

    uint16_t width = (uint16_t)format.fmt.pix.width;
    uint16_t height = (uint16_t)format.fmt.pix.height;
    ESP_LOGI(TAG, "camera format width=%u height=%u pixfmt=0x%08" PRIx32, width, height, format.fmt.pix.pixelformat);
    if (width != kTargetFrameWidth || height != kTargetFrameHeight) {
        ESP_LOGW(TAG, "camera target %ux%u not active; driver returned %ux%u", kTargetFrameWidth, kTargetFrameHeight, width, height);
    }

    esp_err_t ret = allocate_camera_buffers(width, height);
    if (ret != ESP_OK) {
        close(fd);
        return ret;
    }

    ret = request_v4l2_buffers(fd);
    if (ret != ESP_OK) {
        close(fd);
        return ret;
    }
    log_heap_snapshot("after VIDIOC_REQBUFS");

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) != 0) {
        ESP_LOGE(TAG, "VIDIOC_STREAMON failed errno=%d", errno);
        close(fd);
        return ESP_FAIL;
    }
    log_heap_snapshot("after VIDIOC_STREAMON");

    lock_state();
    s_video_fd = fd;
    s_status.running = true;
    s_status.last_error = ESP_OK;
    s_status.model_ready = eye_state_detector_model_ready();
    s_status.eye_state = s_status.model_ready ? HELMET_VISION_EYE_STATE_UNKNOWN : HELMET_VISION_EYE_STATE_NO_MODEL;
    s_status.last_reason = s_status.model_ready ? HELMET_VISION_REASON_NONE : HELMET_VISION_REASON_NO_MODEL;
    s_status.frame_width = width;
    s_status.frame_height = height;
    unlock_state();

    ESP_LOGI(TAG, "vision stream started");
    return ESP_OK;
}

static void close_camera(void)
{
    if (s_video_fd >= 0) {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(s_video_fd, VIDIOC_STREAMOFF, &type);
        close(s_video_fd);
        s_video_fd = -1;
    }

    lock_state();
    s_status.running = false;
    s_status.frame_valid = false;
    s_status.eye_valid = false;
    s_status.eye_state = HELMET_VISION_EYE_STATE_INVALID;
    s_status.last_reason = HELMET_VISION_REASON_CAMERA_ERROR;
    unlock_state();

    free_camera_buffers();
}

static esp_err_t receive_frame_and_process(void)
{
    struct v4l2_buffer buf = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_USERPTR;

    ESP_RETURN_ON_FALSE(ioctl(s_video_fd, VIDIOC_DQBUF, &buf) == 0, ESP_FAIL, TAG, "VIDIOC_DQBUF failed errno=%d", errno);

    uint8_t *frame = NULL;
    if (buf.index < kFrameBufferCount) {
        frame = s_frame_buffers[buf.index];
    }

    if (frame) {
        uint16_t width = 0;
        uint16_t height = 0;
        int64_t now = esp_timer_get_time();

        lock_state();
        width = s_status.frame_width;
        height = s_status.frame_height;
        unlock_state();

        update_latest_frame(frame);
        submit_inference_frame(frame, width, height, now);
    }

    buf.m.userptr = (unsigned long)frame;
    buf.length = s_frame_size;
    ESP_RETURN_ON_FALSE(ioctl(s_video_fd, VIDIOC_QBUF, &buf) == 0, ESP_FAIL, TAG, "VIDIOC_QBUF failed errno=%d", errno);

    return ESP_OK;
}

static void vision_task(void *arg)
{
    (void)arg;

    while (true) {
        esp_err_t ret = open_camera();
        if (ret != ESP_OK) {
            lock_state();
            s_status.last_error = ret;
            s_status.running = false;
            s_status.frame_valid = false;
            s_status.eye_valid = false;
            s_status.eye_state = HELMET_VISION_EYE_STATE_INVALID;
            s_status.last_reason = HELMET_VISION_REASON_CAMERA_ERROR;
            unlock_state();

            helmet_state_clear_eye();
            helmet_state_update_fusion();
            vTaskDelay(pdMS_TO_TICKS(3000));
            continue;
        }

        while (true) {
            ret = receive_frame_and_process();
            if (ret != ESP_OK) {
                lock_state();
                s_status.last_error = ret;
                s_status.eye_valid = false;
                s_status.eye_state = HELMET_VISION_EYE_STATE_INVALID;
                s_status.last_reason = HELMET_VISION_REASON_CAMERA_ERROR;
                unlock_state();
                ESP_LOGE(TAG, "vision stream error: %s", esp_err_to_name(ret));
                break;
            }

            vTaskDelay(1);
        }

        close_camera();
        helmet_state_clear_eye();
        helmet_state_update_fusion();
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

extern "C" esp_err_t helmet_vision_init(void)
{
    if (s_lock == NULL) {
        s_lock = xSemaphoreCreateMutex();
        ESP_RETURN_ON_FALSE(s_lock, ESP_ERR_NO_MEM, TAG, "create vision mutex failed");
    }
    if (s_infer_signal == NULL) {
        s_infer_signal = xSemaphoreCreateBinary();
        ESP_RETURN_ON_FALSE(s_infer_signal, ESP_ERR_NO_MEM, TAG, "create inference signal failed");
    }

    log_heap_snapshot("before eye detector init");
    ESP_RETURN_ON_ERROR(eye_state_detector_init(), TAG, "init eye detector failed");
    log_heap_snapshot("after eye detector init");

    lock_state();
    s_status.model_ready = eye_state_detector_model_ready();
    s_status.eye_state = s_status.model_ready ? HELMET_VISION_EYE_STATE_UNKNOWN : HELMET_VISION_EYE_STATE_NO_MODEL;
    s_status.last_reason = s_status.model_ready ? HELMET_VISION_REASON_NONE : HELMET_VISION_REASON_NO_MODEL;
    unlock_state();

    ESP_LOGI(TAG, "vision service initialized");
    return ESP_OK;
}

extern "C" esp_err_t helmet_vision_start(void)
{
    if (s_detector_task == NULL) {
        BaseType_t ok =
            xTaskCreatePinnedToCore(detector_task,
                                    "eye detector",
                                    kDetectorTaskStack,
                                    NULL,
                                    kDetectorTaskPriority,
                                    &s_detector_task,
                                    kDetectorTaskCore);
        ESP_RETURN_ON_FALSE(ok == pdPASS, ESP_ERR_NO_MEM, TAG, "create eye detector task failed");
    }

    if (s_task == NULL) {
        BaseType_t ok =
            xTaskCreatePinnedToCore(vision_task, "helmet vision", kVisionTaskStack, NULL, kVisionTaskPriority, &s_task, kVisionTaskCore);
        ESP_RETURN_ON_FALSE(ok == pdPASS, ESP_ERR_NO_MEM, TAG, "create vision task failed");
    }

    ESP_LOGI(TAG, "vision capture started");
    return ESP_OK;
}

extern "C" esp_err_t helmet_vision_poll_once(void)
{
    return s_status.last_error;
}

extern "C" esp_err_t helmet_vision_get_status(helmet_vision_status_t *out_status)
{
    ESP_RETURN_ON_FALSE(out_status, ESP_ERR_INVALID_ARG, TAG, "status output is NULL");
    lock_state();
    *out_status = s_status;
    unlock_state();
    return ESP_OK;
}

extern "C" esp_err_t helmet_vision_copy_frame(void *dst, size_t dst_size, helmet_vision_status_t *out_status)
{
    ESP_RETURN_ON_FALSE(dst, ESP_ERR_INVALID_ARG, TAG, "frame dst is NULL");

    lock_state();
    if (out_status) {
        *out_status = s_status;
    }
    if (!s_status.frame_valid || !s_latest_frame || s_frame_size == 0) {
        unlock_state();
        return ESP_ERR_INVALID_STATE;
    }
    if (dst_size < s_frame_size) {
        unlock_state();
        return ESP_ERR_INVALID_SIZE;
    }
    memcpy(dst, s_latest_frame, s_frame_size);
    unlock_state();
    return ESP_OK;
}

extern "C" esp_err_t helmet_vision_copy_scaled_frame(void *dst,
                                                      size_t dst_size,
                                                      uint8_t scale,
                                                      helmet_vision_status_t *out_status)
{
    ESP_RETURN_ON_FALSE(dst, ESP_ERR_INVALID_ARG, TAG, "scaled frame dst is NULL");
    ESP_RETURN_ON_FALSE(scale > 0, ESP_ERR_INVALID_ARG, TAG, "scaled frame scale is zero");

    lock_state();
    if (out_status) {
        *out_status = s_status;
    }
    if (!s_status.frame_valid || !s_latest_frame || s_frame_size == 0 || s_status.frame_width == 0 || s_status.frame_height == 0) {
        unlock_state();
        return ESP_ERR_INVALID_STATE;
    }

    uint16_t out_w = s_status.frame_width / scale;
    uint16_t out_h = s_status.frame_height / scale;
    if (out_w == 0 || out_h == 0) {
        unlock_state();
        return ESP_ERR_INVALID_SIZE;
    }

    size_t payload_size = (size_t)out_w * out_h * sizeof(uint16_t);
    if (dst_size < payload_size) {
        unlock_state();
        return ESP_ERR_INVALID_SIZE;
    }

    const uint16_t *src = reinterpret_cast<const uint16_t *>(s_latest_frame);
    uint16_t *out = reinterpret_cast<uint16_t *>(dst);
    for (uint16_t y = 0; y < out_h; ++y) {
        const uint16_t *src_row = src + (uint32_t)y * scale * s_status.frame_width;
        uint16_t *out_row = out + (uint32_t)y * out_w;
        for (uint16_t x = 0; x < out_w; ++x) {
            out_row[x] = src_row[(uint32_t)x * scale];
        }
    }

    unlock_state();
    return ESP_OK;
}
