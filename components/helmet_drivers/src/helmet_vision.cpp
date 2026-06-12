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
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "helmet_state.h"
#include "linux/videodev2.h"
#include "nvs.h"

static const char *TAG = "helmet_vision";

static constexpr const char *kNvsNamespace = "helmet_vision";
static constexpr const char *kNvsRoiKey = "eye_roi";
static constexpr uint32_t kRoiMagic = 0x45594531; /* EYE1 */
static constexpr int kFrameBufferCount = 2;
static constexpr uint16_t kTargetFrameWidth = 640;
static constexpr uint16_t kTargetFrameHeight = 480;
static constexpr int kVisionTaskStack = 8192;
static constexpr int kVisionTaskPriority = 4;
static constexpr int kSampleIntervalMs = 100;
static constexpr int kPerclosWindowSamples = 300;
static constexpr int64_t kNoEyeTimeoutUs = 3000000;
static constexpr float kClosedThreshold = 0.35f;

typedef struct {
    uint32_t magic;
    uint16_t x;
    uint16_t y;
    uint16_t w;
    uint16_t h;
} vision_roi_nvs_t;

static SemaphoreHandle_t s_lock;
static TaskHandle_t s_task;
static bool s_video_inited;
static int s_video_fd = -1;
static uint8_t *s_frame_buffers[kFrameBufferCount];
static uint8_t *s_latest_frame;
static size_t s_frame_size;
static size_t s_cache_line_size = 64;
static helmet_vision_status_t s_status;
static vision_roi_nvs_t s_saved_roi;
static bool s_saved_roi_valid;

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
static int64_t s_last_valid_eye_us;
static int64_t s_last_diag_us;

static int clamp_int(int value, int low, int high)
{
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

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

static uint8_t rgb565_luma(uint16_t pixel)
{
    uint8_t r = (uint8_t)(((pixel >> 11) & 0x1f) * 255 / 31);
    uint8_t g = (uint8_t)(((pixel >> 5) & 0x3f) * 255 / 63);
    uint8_t b = (uint8_t)((pixel & 0x1f) * 255 / 31);
    return (uint8_t)((77 * r + 150 * g + 29 * b) >> 8);
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

static helmet_vision_roi_t default_roi(uint16_t width, uint16_t height)
{
    helmet_vision_roi_t roi = {};
    roi.x = width / 4;
    roi.y = (uint16_t)((height * 32) / 100);
    roi.w = width / 2;
    roi.h = (uint16_t)((height * 28) / 100);
    return roi;
}

static helmet_vision_roi_t clamp_roi(helmet_vision_roi_t roi, uint16_t width, uint16_t height)
{
    if (width == 0 || height == 0) {
        return {};
    }

    int min_w = width / 32;
    int min_h = height / 32;
    if (min_w < 16) {
        min_w = 16;
    }
    if (min_h < 12) {
        min_h = 12;
    }

    roi.w = (uint16_t)clamp_int(roi.w, min_w, width);
    roi.h = (uint16_t)clamp_int(roi.h, min_h, height);
    roi.x = (uint16_t)clamp_int(roi.x, 0, width - roi.w);
    roi.y = (uint16_t)clamp_int(roi.y, 0, height - roi.h);
    return roi;
}

static helmet_vision_roi_t roi_from_nvs(const vision_roi_nvs_t &saved, uint16_t width, uint16_t height)
{
    helmet_vision_roi_t roi = {};
    roi.x = (uint16_t)((uint32_t)saved.x * width / 10000);
    roi.y = (uint16_t)((uint32_t)saved.y * height / 10000);
    roi.w = (uint16_t)((uint32_t)saved.w * width / 10000);
    roi.h = (uint16_t)((uint32_t)saved.h * height / 10000);
    return clamp_roi(roi, width, height);
}

static vision_roi_nvs_t roi_to_nvs(helmet_vision_roi_t roi, uint16_t width, uint16_t height)
{
    vision_roi_nvs_t saved = {};
    saved.magic = kRoiMagic;
    if (width > 0 && height > 0) {
        saved.x = (uint16_t)((uint32_t)roi.x * 10000 / width);
        saved.y = (uint16_t)((uint32_t)roi.y * 10000 / height);
        saved.w = (uint16_t)((uint32_t)roi.w * 10000 / width);
        saved.h = (uint16_t)((uint32_t)roi.h * 10000 / height);
    }
    return saved;
}

static void load_roi_from_nvs(void)
{
    nvs_handle_t nvs = 0;
    esp_err_t ret = nvs_open(kNvsNamespace, NVS_READONLY, &nvs);
    if (ret != ESP_OK) {
        return;
    }

    vision_roi_nvs_t saved = {};
    size_t len = sizeof(saved);
    ret = nvs_get_blob(nvs, kNvsRoiKey, &saved, &len);
    nvs_close(nvs);

    if (ret == ESP_OK && len == sizeof(saved) && saved.magic == kRoiMagic && saved.w > 0 && saved.h > 0) {
        s_saved_roi = saved;
        s_saved_roi_valid = true;
        ESP_LOGI(TAG, "loaded ROI from NVS x=%u y=%u w=%u h=%u", saved.x, saved.y, saved.w, saved.h);
    }
}

static esp_err_t save_roi_to_nvs(helmet_vision_roi_t roi, uint16_t width, uint16_t height)
{
    nvs_handle_t nvs = 0;
    ESP_RETURN_ON_ERROR(nvs_open(kNvsNamespace, NVS_READWRITE, &nvs), TAG, "open NVS failed");

    vision_roi_nvs_t saved = roi_to_nvs(roi, width, height);
    esp_err_t ret = nvs_set_blob(nvs, kNvsRoiKey, &saved, sizeof(saved));
    if (ret == ESP_OK) {
        ret = nvs_commit(nvs);
    }
    nvs_close(nvs);

    if (ret == ESP_OK) {
        s_saved_roi = saved;
        s_saved_roi_valid = true;
        ESP_LOGI(TAG, "saved ROI x=%u y=%u w=%u h=%u", saved.x, saved.y, saved.w, saved.h);
    }
    return ret;
}

static void apply_initial_roi_locked(uint16_t width, uint16_t height)
{
    if (s_saved_roi_valid) {
        s_status.roi = roi_from_nvs(s_saved_roi, width, height);
    } else {
        s_status.roi = default_roi(width, height);
    }
}

static float estimate_eye_open_ratio(const uint8_t *frame, uint16_t width, uint16_t height, helmet_vision_roi_t roi)
{
    if (!frame || width == 0 || height == 0 || roi.w < 16 || roi.h < 12) {
        return -1.0f;
    }

    roi = clamp_roi(roi, width, height);
    const uint16_t *pixels = reinterpret_cast<const uint16_t *>(frame);
    int step_x = (roi.w > 360) ? 3 : ((roi.w > 180) ? 2 : 1);
    int step_y = (roi.h > 240) ? 3 : ((roi.h > 120) ? 2 : 1);

    uint32_t sum = 0;
    int samples = 0;
    uint8_t min_luma = 255;
    uint8_t max_luma = 0;

    for (int y = roi.y; y < roi.y + roi.h; y += step_y) {
        const uint16_t *row = pixels + y * width;
        for (int x = roi.x; x < roi.x + roi.w; x += step_x) {
            uint8_t y8 = rgb565_luma(row[x]);
            sum += y8;
            samples++;
            if (y8 < min_luma) {
                min_luma = y8;
            }
            if (y8 > max_luma) {
                max_luma = y8;
            }
        }
    }

    if (samples < 64 || max_luma <= min_luma + 10) {
        return -1.0f;
    }

    int mean = (int)(sum / samples);
    int contrast = max_luma - min_luma;
    int dark_threshold = mean - clamp_int(contrast / 5, 8, 28);
    dark_threshold = clamp_int(dark_threshold, 0, 255);

    int active_first = -1;
    int active_last = -1;
    int row_index = 0;
    int row_count = 0;
    int dark_count = 0;

    for (int y = roi.y; y < roi.y + roi.h; y += step_y, row_index++) {
        const uint16_t *row = pixels + y * width;
        int row_dark = 0;
        int row_samples = 0;

        for (int x = roi.x; x < roi.x + roi.w; x += step_x) {
            if (rgb565_luma(row[x]) < dark_threshold) {
                row_dark++;
            }
            row_samples++;
        }

        if (row_samples > 0) {
            int active_threshold = row_samples / 12;
            if (active_threshold < 2) {
                active_threshold = 2;
            }
            if (row_dark >= active_threshold) {
                if (active_first < 0) {
                    active_first = row_index;
                }
                active_last = row_index;
            }
        }
        dark_count += row_dark;
        row_count++;
    }

    if (row_count == 0 || active_first < 0) {
        return -1.0f;
    }

    float span_ratio = (float)(active_last - active_first + 1) / (float)row_count;
    float dark_ratio = (float)dark_count / (float)samples;
    float span_score = (span_ratio - 0.12f) / 0.36f;
    float density_score = (dark_ratio - 0.05f) / 0.20f;
    float contrast_score = (contrast - 10.0f) / 70.0f;
    float open = span_score * 0.65f + density_score * 0.25f + contrast_score * 0.10f;

    return clamp_float(open, 0.0f, 1.0f);
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

static void update_eye_state(float raw_open)
{
    int64_t now = esp_timer_get_time();
    if (raw_open < 0.0f) {
        uint32_t no_eye_ms = 0;
        bool should_clear = false;

        lock_state();
        if (s_last_valid_eye_us > 0) {
            no_eye_ms = (uint32_t)((now - s_last_valid_eye_us) / 1000);
            should_clear = (now - s_last_valid_eye_us) >= kNoEyeTimeoutUs;
        }
        s_status.eye_valid = false;
        s_status.no_eye_ms = no_eye_ms;
        if (should_clear) {
            reset_eye_tracker_locked();
            s_status.eye_closed = false;
            s_status.eye_open_ratio = 0.0f;
            s_status.perclos = 0.0f;
        }
        unlock_state();

        if (should_clear) {
            helmet_state_clear_eye();
            helmet_state_update_fusion();
        }
        return;
    }

    if (s_last_sample_us > 0 && (now - s_last_sample_us) < (int64_t)kSampleIntervalMs * 1000) {
        return;
    }
    s_last_sample_us = now;
    s_last_valid_eye_us = now;

    bool closed = false;
    float perclos = 0.0f;
    uint32_t blink_count = 0;
    float smoothed = raw_open;
    helmet_vision_roi_t roi = {};
    uint16_t frame_width = 0;
    uint16_t frame_height = 0;

    lock_state();

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
    closed = smoothed < kClosedThreshold;

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
    s_status.eye_open_ratio = smoothed;
    s_status.perclos = perclos;
    s_status.blink_count = blink_count;
    s_status.no_eye_ms = 0;
    roi = s_status.roi;
    frame_width = s_status.frame_width;
    frame_height = s_status.frame_height;

    unlock_state();

    if ((now - s_last_diag_us) >= 5000000) {
        s_last_diag_us = now;
        ESP_LOGI(TAG,
                 "eye diag frame=%ux%u roi=%u,%u,%u,%u open=%.2f perclos=%.2f closed=%d blink=%" PRIu32,
                 frame_width,
                 frame_height,
                 roi.x,
                 roi.y,
                 roi.w,
                 roi.h,
                 smoothed,
                 perclos,
                 closed ? 1 : 0,
                 blink_count);
    }

    helmet_state_set_eye(smoothed, perclos, blink_count, false);
    helmet_state_update_fusion();
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
    s_status.frame_width = width;
    s_status.frame_height = height;
    apply_initial_roi_locked(width, height);
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
        helmet_vision_roi_t roi = {};
        uint16_t width = 0;
        uint16_t height = 0;

        lock_state();
        roi = s_status.roi;
        width = s_status.frame_width;
        height = s_status.frame_height;
        unlock_state();

        float open_ratio = estimate_eye_open_ratio(frame, width, height, roi);
        update_eye_state(open_ratio);
        update_latest_frame(frame);
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
                unlock_state();
                ESP_LOGE(TAG, "vision stream error: %s", esp_err_to_name(ret));
                break;
            }
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

    load_roi_from_nvs();

    if (s_task == NULL) {
        BaseType_t ok = xTaskCreatePinnedToCore(vision_task, "helmet vision", kVisionTaskStack, NULL, kVisionTaskPriority, &s_task, 1);
        ESP_RETURN_ON_FALSE(ok == pdPASS, ESP_ERR_NO_MEM, TAG, "create vision task failed");
    }

    ESP_LOGI(TAG, "vision service initialized");
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

extern "C" esp_err_t helmet_vision_get_roi(helmet_vision_roi_t *out_roi)
{
    ESP_RETURN_ON_FALSE(out_roi, ESP_ERR_INVALID_ARG, TAG, "ROI output is NULL");
    lock_state();
    *out_roi = s_status.roi;
    unlock_state();
    return ESP_OK;
}

extern "C" esp_err_t helmet_vision_set_roi(const helmet_vision_roi_t *roi, bool save_to_nvs)
{
    ESP_RETURN_ON_FALSE(roi, ESP_ERR_INVALID_ARG, TAG, "ROI input is NULL");

    helmet_vision_roi_t clamped = {};
    uint16_t width = 0;
    uint16_t height = 0;

    lock_state();
    width = s_status.frame_width;
    height = s_status.frame_height;
    if (width == 0 || height == 0) {
        unlock_state();
        return ESP_ERR_INVALID_STATE;
    }
    clamped = clamp_roi(*roi, width, height);
    s_status.roi = clamped;
    unlock_state();

    if (save_to_nvs) {
        return save_roi_to_nvs(clamped, width, height);
    }
    return ESP_OK;
}

extern "C" esp_err_t helmet_vision_reset_roi(void)
{
    helmet_vision_roi_t roi = {};
    uint16_t width = 0;
    uint16_t height = 0;

    lock_state();
    width = s_status.frame_width;
    height = s_status.frame_height;
    if (width == 0 || height == 0) {
        unlock_state();
        return ESP_ERR_INVALID_STATE;
    }
    roi = default_roi(width, height);
    s_status.roi = roi;
    unlock_state();

    return save_roi_to_nvs(roi, width, height);
}
