#include "helmet_camera_usb.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_check.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "helmet_vision.h"
#include "tinyusb.h"
#include "tinyusb_cdc_acm.h"
#include "tinyusb_default_config.h"

static const char *TAG = "helmet_camera_usb";

#define HELMET_USB_CMD_MAX_LEN 32
#define HELMET_USB_QUEUE_DEPTH 4
#define HELMET_USB_TASK_STACK 4096
#define HELMET_USB_TASK_PRIORITY 3
#define HELMET_USB_DEFAULT_SCALE 4
#define HELMET_USB_MIN_SCALE 1
#define HELMET_USB_MAX_SCALE 8
#define HELMET_USB_WRITE_CHUNK 4096
#define HELMET_USB_WRITE_TIMEOUT_MS 1000
#define HELMET_USB_FRAME_VERSION 3

#define HELMET_USB_STATUS_FLAG_RUNNING (1U << 0)
#define HELMET_USB_STATUS_FLAG_FRAME_VALID (1U << 1)
#define HELMET_USB_STATUS_FLAG_MODEL_READY (1U << 2)
#define HELMET_USB_STATUS_FLAG_EYE_VALID (1U << 3)
#define HELMET_USB_STATUS_FLAG_EYE_CLOSED (1U << 4)

typedef struct __attribute__((packed)) {
    char magic[4];
    uint16_t version;
    uint16_t header_size;
    uint16_t width;
    uint16_t height;
    uint16_t src_width;
    uint16_t src_height;
    uint16_t scale;
    uint16_t pixel_format;
    uint32_t frame_seq;
    uint32_t payload_size;
    int32_t status;
    uint16_t flags;
    uint16_t eye_state;
    uint16_t reason;
    uint16_t eye_bbox_x;
    uint16_t eye_bbox_y;
    uint16_t eye_bbox_w;
    uint16_t eye_bbox_h;
    uint16_t eye_confidence_milli;
    uint16_t eye_open_milli;
    uint16_t perclos_milli;
    uint16_t reserved0;
    uint32_t blink_count;
    uint32_t no_eye_ms;
    uint32_t last_infer_ms;
    uint16_t open_score_milli;
    uint16_t closed_score_milli;
    uint16_t background_score_milli;
    uint16_t reserved1;
} helmet_camera_usb_frame_header_t;

typedef enum {
    HELMET_CAMERA_USB_CMD_PING = 0,
    HELMET_CAMERA_USB_CMD_SNAP,
} helmet_camera_usb_cmd_type_t;

typedef struct {
    helmet_camera_usb_cmd_type_t type;
    uint8_t scale;
} helmet_camera_usb_cmd_t;

static QueueHandle_t s_cmd_queue;
static TaskHandle_t s_task;
static bool s_inited;
static uint8_t s_rx_buf[64];
static char s_cmd_line[HELMET_USB_CMD_MAX_LEN];
static size_t s_cmd_line_len;
static uint16_t *s_snapshot_payload;
static size_t s_snapshot_payload_size;

static void camera_usb_event_callback(tinyusb_event_t *event, void *arg)
{
    (void)arg;

    if (!event) {
        return;
    }

    switch (event->id) {
    case TINYUSB_EVENT_ATTACHED:
        ESP_LOGI(TAG, "USB attached rhport=%u", (unsigned)event->rhport);
        break;
    case TINYUSB_EVENT_DETACHED:
        ESP_LOGI(TAG, "USB detached rhport=%u", (unsigned)event->rhport);
        break;
    default:
        ESP_LOGI(TAG, "USB event id=%d rhport=%u", (int)event->id, (unsigned)event->rhport);
        break;
    }
}

static int clamp_scale(int scale)
{
    if (scale < HELMET_USB_MIN_SCALE) {
        return HELMET_USB_MIN_SCALE;
    }
    if (scale > HELMET_USB_MAX_SCALE) {
        return HELMET_USB_MAX_SCALE;
    }
    return scale;
}

static uint16_t clamp_metric_milli(float value)
{
    if (value < 0.0f) {
        value = 0.0f;
    } else if (value > 1.0f) {
        value = 1.0f;
    }
    return (uint16_t)(value * 1000.0f + 0.5f);
}

static uint16_t rgb565_color(uint8_t r, uint8_t g, uint8_t b)
{
    return (uint16_t)(((r & 0xf8) << 8) | ((g & 0xfc) << 3) | (b >> 3));
}

static const uint8_t *glyph3x5(char ch)
{
    static const uint8_t blank[5] = {0, 0, 0, 0, 0};
    static const uint8_t underscore[5] = {0, 0, 0, 0, 7};
    static const uint8_t dot[5] = {0, 0, 0, 0, 2};
    static const uint8_t equal[5] = {0, 7, 0, 7, 0};
    static const uint8_t dash[5] = {0, 0, 7, 0, 0};
    static const uint8_t digits[10][5] = {
        {7, 5, 5, 5, 7},
        {2, 6, 2, 2, 7},
        {7, 1, 7, 4, 7},
        {7, 1, 7, 1, 7},
        {5, 5, 7, 1, 1},
        {7, 4, 7, 1, 7},
        {7, 4, 7, 5, 7},
        {7, 1, 1, 1, 1},
        {7, 5, 7, 5, 7},
        {7, 5, 7, 1, 7},
    };
    static const uint8_t letters[26][5] = {
        {7, 5, 7, 5, 5}, {6, 5, 6, 5, 6}, {7, 4, 4, 4, 7}, {6, 5, 5, 5, 6}, {7, 4, 6, 4, 7},
        {7, 4, 6, 4, 4}, {7, 4, 5, 5, 7}, {5, 5, 7, 5, 5}, {7, 2, 2, 2, 7}, {1, 1, 1, 5, 7},
        {5, 5, 6, 5, 5}, {4, 4, 4, 4, 7}, {5, 7, 7, 5, 5}, {5, 7, 7, 7, 5}, {7, 5, 5, 5, 7},
        {7, 5, 7, 4, 4}, {7, 5, 5, 7, 1}, {7, 5, 7, 6, 5}, {7, 4, 7, 1, 7}, {7, 2, 2, 2, 2},
        {5, 5, 5, 5, 7}, {5, 5, 5, 5, 2}, {5, 5, 7, 7, 5}, {5, 5, 2, 5, 5}, {5, 5, 2, 2, 2},
        {7, 1, 2, 4, 7},
    };

    if (ch >= '0' && ch <= '9') {
        return digits[ch - '0'];
    }
    if (ch >= 'a' && ch <= 'z') {
        ch = (char)(ch - 'a' + 'A');
    }
    if (ch >= 'A' && ch <= 'Z') {
        return letters[ch - 'A'];
    }
    switch (ch) {
    case '_':
        return underscore;
    case '.':
        return dot;
    case '=':
        return equal;
    case '-':
        return dash;
    default:
        return blank;
    }
}

static void fill_rect_rgb565(uint16_t *buffer, int width, int height, int x, int y, int w, int h, uint16_t color)
{
    if (!buffer || width <= 0 || height <= 0 || w <= 0 || h <= 0) {
        return;
    }
    int x1 = x < 0 ? 0 : x;
    int y1 = y < 0 ? 0 : y;
    int x2 = x + w;
    int y2 = y + h;
    if (x2 > width) {
        x2 = width;
    }
    if (y2 > height) {
        y2 = height;
    }
    for (int py = y1; py < y2; ++py) {
        uint16_t *row = buffer + py * width;
        for (int px = x1; px < x2; ++px) {
            row[px] = color;
        }
    }
}

static void draw_char3x5_rgb565(uint16_t *buffer, int width, int height, int x, int y, char ch, int scale, uint16_t color)
{
    const uint8_t *glyph = glyph3x5(ch);
    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 3; ++col) {
            if ((glyph[row] & (1U << (2 - col))) == 0) {
                continue;
            }
            fill_rect_rgb565(buffer, width, height, x + col * scale, y + row * scale, scale, scale, color);
        }
    }
}

static void draw_text3x5_rgb565(uint16_t *buffer, int width, int height, int x, int y, const char *text, int scale, uint16_t color)
{
    if (!text) {
        return;
    }
    int cursor_x = x;
    int advance = 4 * scale;
    while (*text != '\0') {
        draw_char3x5_rgb565(buffer, width, height, cursor_x, y, *text, scale, color);
        cursor_x += advance;
        text++;
    }
}

static const char *overlay_pred_name(const helmet_vision_status_t *status)
{
    if (!status || !status->model_ready || status->eye_state == HELMET_VISION_EYE_STATE_NO_MODEL) {
        return "NO_MODEL";
    }
    if (status->eye_valid && status->eye_state == HELMET_VISION_EYE_STATE_CLOSED) {
        return "CLOSED";
    }
    if (status->eye_valid && status->eye_state == HELMET_VISION_EYE_STATE_OPEN) {
        return "OPEN";
    }
    if (status->last_reason == HELMET_VISION_REASON_NO_DETECTION) {
        return "BACKGROUND";
    }
    if (status->last_reason == HELMET_VISION_REASON_LOW_CONFIDENCE) {
        return "LOW_CONF";
    }
    if (status->last_reason == HELMET_VISION_REASON_RUNTIME_BUSY) {
        return "BUSY";
    }
    if (status->last_reason == HELMET_VISION_REASON_CAMERA_ERROR) {
        return "CAMERA_ERROR";
    }
    return "INVALID";
}

static uint16_t overlay_pred_color(const helmet_vision_status_t *status)
{
    if (!status || !status->model_ready || status->eye_state == HELMET_VISION_EYE_STATE_NO_MODEL) {
        return rgb565_color(255, 208, 64);
    }
    if (status->eye_valid && status->eye_state == HELMET_VISION_EYE_STATE_CLOSED) {
        return rgb565_color(255, 64, 64);
    }
    if (status->eye_valid && status->eye_state == HELMET_VISION_EYE_STATE_OPEN) {
        return rgb565_color(64, 255, 128);
    }
    return rgb565_color(255, 160, 64);
}

static void draw_inference_overlay(uint16_t *buffer, uint16_t width, uint16_t height, const helmet_vision_status_t *status)
{
    if (!buffer || width == 0 || height == 0 || !status) {
        return;
    }

    int draw_scale = width >= 480 ? 3 : (width >= 240 ? 2 : 1);
    int x = 8;
    int y = 8;
    int line_h = 6 * draw_scale;
    int char_w = 4 * draw_scale;
    uint16_t bg = rgb565_color(0, 0, 0);
    uint16_t fg = overlay_pred_color(status);
    uint16_t white = rgb565_color(245, 245, 245);

    char line1[40];
    char line2[72];
    char line3[48];
    snprintf(line1, sizeof(line1), "PRED %s", overlay_pred_name(status));
    snprintf(line2,
             sizeof(line2),
             "CONF=%.3f O=%.3f C=%.3f BG=%.3f",
             status->eye_confidence,
             status->open_score,
             status->closed_score,
             status->background_score);
    snprintf(line3, sizeof(line3), "INFER=%luMS FRAME=%lu", (unsigned long)status->last_infer_ms, (unsigned long)status->frame_seq);

    size_t max_chars = strlen(line1);
    if (strlen(line2) > max_chars) {
        max_chars = strlen(line2);
    }
    if (strlen(line3) > max_chars) {
        max_chars = strlen(line3);
    }

    fill_rect_rgb565(buffer, width, height, x - 4, y - 4, (int)max_chars * char_w + 8, line_h * 3 + 8, bg);
    draw_text3x5_rgb565(buffer, width, height, x, y, line1, draw_scale, fg);
    draw_text3x5_rgb565(buffer, width, height, x, y + line_h, line2, draw_scale, white);
    draw_text3x5_rgb565(buffer, width, height, x, y + line_h * 2, line3, draw_scale, white);
}

static uint16_t status_flags(const helmet_vision_status_t *status)
{
    if (!status) {
        return 0;
    }

    uint16_t flags = 0;
    if (status->running) {
        flags |= HELMET_USB_STATUS_FLAG_RUNNING;
    }
    if (status->frame_valid) {
        flags |= HELMET_USB_STATUS_FLAG_FRAME_VALID;
    }
    if (status->model_ready) {
        flags |= HELMET_USB_STATUS_FLAG_MODEL_READY;
    }
    if (status->eye_valid) {
        flags |= HELMET_USB_STATUS_FLAG_EYE_VALID;
    }
    if (status->eye_closed) {
        flags |= HELMET_USB_STATUS_FLAG_EYE_CLOSED;
    }
    return flags;
}

static void queue_command_line(const char *line)
{
    if (!line || line[0] == '\0') {
        return;
    }

    helmet_camera_usb_cmd_t cmd = {
        .type = HELMET_CAMERA_USB_CMD_PING,
        .scale = HELMET_USB_DEFAULT_SCALE,
    };

    if (strncmp(line, "PING", 4) == 0) {
        cmd.type = HELMET_CAMERA_USB_CMD_PING;
    } else if (strncmp(line, "SNAP", 4) == 0) {
        cmd.type = HELMET_CAMERA_USB_CMD_SNAP;
        const char *arg = line + 4;
        while (*arg == ' ' || *arg == '\t') {
            arg++;
        }
        if (*arg >= '0' && *arg <= '9') {
            cmd.scale = (uint8_t)clamp_scale(atoi(arg));
        }
    } else {
        return;
    }

    if (s_cmd_queue != NULL) {
        (void)xQueueSend(s_cmd_queue, &cmd, 0);
    }
}

static void append_rx_byte(uint8_t byte)
{
    if (byte == '\r') {
        return;
    }
    if (byte == '\n') {
        s_cmd_line[s_cmd_line_len] = '\0';
        queue_command_line(s_cmd_line);
        s_cmd_line_len = 0;
        return;
    }
    if (s_cmd_line_len + 1 < sizeof(s_cmd_line)) {
        s_cmd_line[s_cmd_line_len++] = (char)byte;
    } else {
        s_cmd_line_len = 0;
    }
}

static void camera_usb_rx_callback(int itf, cdcacm_event_t *event)
{
    (void)event;

    size_t rx_size = 0;
    esp_err_t ret = tinyusb_cdcacm_read(itf, s_rx_buf, sizeof(s_rx_buf), &rx_size);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "cdc read failed: %s", esp_err_to_name(ret));
        return;
    }

    for (size_t i = 0; i < rx_size; ++i) {
        append_rx_byte(s_rx_buf[i]);
    }
}

static void camera_usb_line_state_callback(int itf, cdcacm_event_t *event)
{
    if (!event) {
        return;
    }
    ESP_LOGI(
        TAG,
        "CDC line state itf=%d dtr=%d rts=%d",
        itf,
        event->line_state_changed_data.dtr,
        event->line_state_changed_data.rts
    );
}

static esp_err_t cdc_write_all(const void *data, size_t len)
{
    const uint8_t *ptr = (const uint8_t *)data;
    while (len > 0) {
        size_t chunk = len > HELMET_USB_WRITE_CHUNK ? HELMET_USB_WRITE_CHUNK : len;

        size_t queued_total = 0;
        int idle_retries = 0;
        while (queued_total < chunk) {
            size_t queued = tinyusb_cdcacm_write_queue(
                TINYUSB_CDC_ACM_0,
                ptr + queued_total,
                chunk - queued_total
            );

            if (queued == 0) {
                esp_err_t flush_ret = tinyusb_cdcacm_write_flush(
                    TINYUSB_CDC_ACM_0,
                    pdMS_TO_TICKS(HELMET_USB_WRITE_TIMEOUT_MS)
                );
                if (flush_ret != ESP_OK) {
                    return flush_ret;
                }
                if (++idle_retries > 20) {
                    return ESP_ERR_TIMEOUT;
                }
                vTaskDelay(pdMS_TO_TICKS(1));
                continue;
            }

            queued_total += queued;
            idle_retries = 0;
        }

        esp_err_t ret = tinyusb_cdcacm_write_flush(TINYUSB_CDC_ACM_0, pdMS_TO_TICKS(HELMET_USB_WRITE_TIMEOUT_MS));
        if (ret != ESP_OK) {
            return ret;
        }
        ptr += queued_total;
        len -= queued_total;
    }
    return ESP_OK;
}

static esp_err_t send_status_header(esp_err_t status)
{
    helmet_camera_usb_frame_header_t header = {
        .magic = {'H', 'V', 'C', '1'},
        .version = HELMET_USB_FRAME_VERSION,
        .header_size = sizeof(helmet_camera_usb_frame_header_t),
        .pixel_format = 1,
        .status = status,
    };
    return cdc_write_all(&header, sizeof(header));
}

static esp_err_t ensure_snapshot_payload(size_t payload_size)
{
    if (s_snapshot_payload && s_snapshot_payload_size >= payload_size) {
        return ESP_OK;
    }

    if (s_snapshot_payload) {
        heap_caps_free(s_snapshot_payload);
        s_snapshot_payload = NULL;
        s_snapshot_payload_size = 0;
    }

    s_snapshot_payload = (uint16_t *)heap_caps_malloc(payload_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_snapshot_payload) {
        ESP_LOGE(TAG, "snapshot payload alloc failed size=%u", (unsigned)payload_size);
        return ESP_ERR_NO_MEM;
    }
    s_snapshot_payload_size = payload_size;
    return ESP_OK;
}

static esp_err_t send_snapshot(uint8_t scale)
{
    helmet_vision_status_t status = {};
    esp_err_t ret = helmet_vision_get_status(&status);
    if (ret != ESP_OK || !status.frame_valid || status.frame_width == 0 || status.frame_height == 0) {
        return send_status_header(ret == ESP_OK ? ESP_ERR_INVALID_STATE : ret);
    }

    scale = (uint8_t)clamp_scale(scale);
    uint16_t out_w = status.frame_width / scale;
    uint16_t out_h = status.frame_height / scale;
    if (out_w == 0 || out_h == 0) {
        return send_status_header(ESP_ERR_INVALID_SIZE);
    }

    size_t payload_size = (size_t)out_w * out_h * sizeof(uint16_t);
    ret = ensure_snapshot_payload(payload_size);
    if (ret != ESP_OK) {
        return send_status_header(ret);
    }

    ret = helmet_vision_copy_scaled_frame(s_snapshot_payload, payload_size, scale, &status);
    if (ret != ESP_OK) {
        return send_status_header(ret);
    }

    out_w = status.frame_width / scale;
    out_h = status.frame_height / scale;
    payload_size = (size_t)out_w * out_h * sizeof(uint16_t);
    draw_inference_overlay(s_snapshot_payload, out_w, out_h, &status);

    helmet_camera_usb_frame_header_t header = {
        .magic = {'H', 'V', 'C', '1'},
        .version = HELMET_USB_FRAME_VERSION,
        .header_size = sizeof(helmet_camera_usb_frame_header_t),
        .width = out_w,
        .height = out_h,
        .src_width = status.frame_width,
        .src_height = status.frame_height,
        .scale = scale,
        .pixel_format = 1,
        .frame_seq = status.frame_seq,
        .payload_size = (uint32_t)payload_size,
        .status = ESP_OK,
        .flags = status_flags(&status),
        .eye_state = (uint16_t)status.eye_state,
        .reason = (uint16_t)status.last_reason,
        .eye_bbox_x = status.eye_bbox.x,
        .eye_bbox_y = status.eye_bbox.y,
        .eye_bbox_w = status.eye_bbox.w,
        .eye_bbox_h = status.eye_bbox.h,
        .eye_confidence_milli = clamp_metric_milli(status.eye_confidence),
        .eye_open_milli = clamp_metric_milli(status.eye_open_ratio),
        .perclos_milli = clamp_metric_milli(status.perclos),
        .blink_count = status.blink_count,
        .no_eye_ms = status.no_eye_ms,
        .last_infer_ms = status.last_infer_ms,
        .open_score_milli = clamp_metric_milli(status.open_score),
        .closed_score_milli = clamp_metric_milli(status.closed_score),
        .background_score_milli = clamp_metric_milli(status.background_score),
    };

    ret = cdc_write_all(&header, sizeof(header));
    if (ret != ESP_OK) {
        return ret;
    }

    ret = cdc_write_all(s_snapshot_payload, payload_size);
    return ret;
}

static void camera_usb_task(void *arg)
{
    (void)arg;
    helmet_camera_usb_cmd_t cmd;

    while (true) {
        if (xQueueReceive(s_cmd_queue, &cmd, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        if (cmd.type == HELMET_CAMERA_USB_CMD_PING) {
            (void)cdc_write_all("PONG\n", 5);
        } else if (cmd.type == HELMET_CAMERA_USB_CMD_SNAP) {
            esp_err_t ret = send_snapshot(cmd.scale);
            if (ret != ESP_OK) {
                ESP_LOGW(TAG, "snapshot send failed: %s", esp_err_to_name(ret));
            }
        }
    }
}

esp_err_t helmet_camera_usb_init(void)
{
    if (s_inited) {
        return ESP_OK;
    }

    s_cmd_queue = xQueueCreate(HELMET_USB_QUEUE_DEPTH, sizeof(helmet_camera_usb_cmd_t));
    ESP_RETURN_ON_FALSE(s_cmd_queue, ESP_ERR_NO_MEM, TAG, "create command queue failed");

    const tinyusb_config_t tusb_cfg = TINYUSB_DEFAULT_CONFIG(camera_usb_event_callback);
    ESP_LOGI(TAG, "install TinyUSB camera CDC rhport=%u", (unsigned)tusb_cfg.port);
    ESP_RETURN_ON_ERROR(tinyusb_driver_install(&tusb_cfg), TAG, "install tinyusb failed");

    tinyusb_config_cdcacm_t acm_cfg = {
        .cdc_port = TINYUSB_CDC_ACM_0,
        .callback_rx = camera_usb_rx_callback,
        .callback_rx_wanted_char = NULL,
        .callback_line_state_changed = camera_usb_line_state_callback,
        .callback_line_coding_changed = NULL,
    };
    ESP_RETURN_ON_ERROR(tinyusb_cdcacm_init(&acm_cfg), TAG, "init cdc acm failed");

    BaseType_t ok = xTaskCreatePinnedToCore(
        camera_usb_task,
        "helmet cam usb",
        HELMET_USB_TASK_STACK,
        NULL,
        HELMET_USB_TASK_PRIORITY,
        &s_task,
        0
    );
    ESP_RETURN_ON_FALSE(ok == pdPASS, ESP_ERR_NO_MEM, TAG, "create camera usb task failed");

    s_inited = true;
    ESP_LOGI(TAG, "USB camera snapshot service ready on USB-OTG CDC");
    return ESP_OK;
}
