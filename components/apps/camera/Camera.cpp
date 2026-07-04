/*
 * SPDX-FileCopyrightText: 2023-2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Camera.hpp"

#include <stdio.h>
#include <string.h>

#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "helmet_vision.h"
#include "ui/ui.h"

LV_IMG_DECLARE(img_app_camera);

static const char *TAG = "Camera";

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
    static const uint8_t colon[5] = {0, 2, 0, 2, 0};
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
        {7, 5, 7, 5, 5}, // A
        {6, 5, 6, 5, 6}, // B
        {7, 4, 4, 4, 7}, // C
        {6, 5, 5, 5, 6}, // D
        {7, 4, 6, 4, 7}, // E
        {7, 4, 6, 4, 4}, // F
        {7, 4, 5, 5, 7}, // G
        {5, 5, 7, 5, 5}, // H
        {7, 2, 2, 2, 7}, // I
        {1, 1, 1, 5, 7}, // J
        {5, 5, 6, 5, 5}, // K
        {4, 4, 4, 4, 7}, // L
        {5, 7, 7, 5, 5}, // M
        {5, 7, 7, 7, 5}, // N
        {7, 5, 5, 5, 7}, // O
        {7, 5, 7, 4, 4}, // P
        {7, 5, 5, 7, 1}, // Q
        {7, 5, 7, 6, 5}, // R
        {7, 4, 7, 1, 7}, // S
        {7, 2, 2, 2, 2}, // T
        {5, 5, 5, 5, 7}, // U
        {5, 5, 5, 5, 2}, // V
        {5, 5, 7, 7, 5}, // W
        {5, 5, 2, 5, 5}, // X
        {5, 5, 2, 2, 2}, // Y
        {7, 1, 2, 4, 7}, // Z
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
    case ':':
        return colon;
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

static size_t text_len(const char *text)
{
    return text ? strlen(text) : 0;
}

static const char *eye_state_name(helmet_vision_eye_state_t state)
{
    switch (state) {
    case HELMET_VISION_EYE_STATE_OPEN:
        return "OPEN";
    case HELMET_VISION_EYE_STATE_CLOSED:
        return "CLOSED";
    case HELMET_VISION_EYE_STATE_INVALID:
        return "INVALID";
    case HELMET_VISION_EYE_STATE_NO_MODEL:
        return "NO_MODEL";
    case HELMET_VISION_EYE_STATE_UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

static const char *reason_name(helmet_vision_reason_t reason)
{
    switch (reason) {
    case HELMET_VISION_REASON_NONE:
        return "OK";
    case HELMET_VISION_REASON_NO_MODEL:
        return "NO_MODEL";
    case HELMET_VISION_REASON_NO_DETECTION:
        return "NO_DETECTION";
    case HELMET_VISION_REASON_RUNTIME_BUSY:
        return "BUSY";
    case HELMET_VISION_REASON_LOW_CONFIDENCE:
        return "LOW_CONF";
    case HELMET_VISION_REASON_CAMERA_ERROR:
        return "CAMERA_ERROR";
    default:
        return "UNKNOWN";
    }
}

static const char *overlay_pred_name(const helmet_vision_status_t &status)
{
    if (!status.model_ready || status.eye_state == HELMET_VISION_EYE_STATE_NO_MODEL) {
        return "NO_MODEL";
    }
    if (status.eye_valid && status.eye_state == HELMET_VISION_EYE_STATE_CLOSED) {
        return "CLOSED";
    }
    if (status.eye_valid && status.eye_state == HELMET_VISION_EYE_STATE_OPEN) {
        return "OPEN";
    }
    if (status.last_reason == HELMET_VISION_REASON_NO_DETECTION) {
        return "BACKGROUND";
    }
    if (status.last_reason == HELMET_VISION_REASON_LOW_CONFIDENCE) {
        return "LOW_CONF";
    }
    return reason_name(status.last_reason);
}

static uint16_t overlay_pred_color(const helmet_vision_status_t &status)
{
    if (!status.model_ready || status.eye_state == HELMET_VISION_EYE_STATE_NO_MODEL) {
        return rgb565_color(255, 208, 64);
    }
    if (status.eye_valid && status.eye_state == HELMET_VISION_EYE_STATE_CLOSED) {
        return rgb565_color(255, 64, 64);
    }
    if (status.eye_valid && status.eye_state == HELMET_VISION_EYE_STATE_OPEN) {
        return rgb565_color(64, 255, 128);
    }
    if (status.last_reason == HELMET_VISION_REASON_NO_DETECTION) {
        return rgb565_color(255, 208, 64);
    }
    return rgb565_color(255, 160, 64);
}

static void draw_rect_rgb565(uint16_t *buffer, int width, int height, helmet_vision_bbox_t bbox, uint16_t color, int thickness)
{
    if (!buffer || width <= 0 || height <= 0 || bbox.w == 0 || bbox.h == 0) {
        return;
    }

    int x1 = bbox.x;
    int y1 = bbox.y;
    int x2 = bbox.x + bbox.w - 1;
    int y2 = bbox.y + bbox.h - 1;

    if (x1 < 0) {
        x1 = 0;
    }
    if (y1 < 0) {
        y1 = 0;
    }
    if (x2 >= width) {
        x2 = width - 1;
    }
    if (y2 >= height) {
        y2 = height - 1;
    }
    if (x2 <= x1 || y2 <= y1) {
        return;
    }

    for (int t = 0; t < thickness; t++) {
        int top = y1 + t;
        int bottom = y2 - t;
        int left = x1 + t;
        int right = x2 - t;
        if (top > bottom || left > right) {
            break;
        }

        for (int x = left; x <= right; x++) {
            buffer[top * width + x] = color;
            buffer[bottom * width + x] = color;
        }
        for (int y = top; y <= bottom; y++) {
            buffer[y * width + left] = color;
            buffer[y * width + right] = color;
        }
    }
}

Camera::Camera(uint16_t hor_res, uint16_t ver_res):
    ESP_Brookesia_PhoneApp("Camera", &img_app_camera, false),
    _hor_res(hor_res),
    _ver_res(ver_res),
    _frame_buffer(NULL),
    _frame_buffer_size(0),
    _frame_dsc({}),
    _timer(NULL),
    _status_label(NULL),
    _canvas_ready(false)
{
}

Camera::~Camera()
{
}

bool Camera::init(void)
{
    return true;
}

bool Camera::run(void)
{
    ESP_LOGI(TAG, "run debug camera view");

    ui_camera_init();
    extraUiInit();
    updateFrame();

    if (_timer == NULL) {
        _timer = lv_timer_create(onTimer, 250, this);
    }

    return true;
}

bool Camera::pause(void)
{
    if (_timer) {
        lv_timer_pause(_timer);
    }
    return true;
}

bool Camera::resume(void)
{
    if (_timer) {
        lv_timer_resume(_timer);
    }
    updateFrame();
    return true;
}

bool Camera::back(void)
{
    notifyCoreClosed();
    return true;
}

bool Camera::close(void)
{
    if (_timer) {
        lv_timer_del(_timer);
        _timer = NULL;
    }

    if (_frame_buffer) {
        heap_caps_free(_frame_buffer);
        _frame_buffer = NULL;
    }
    _frame_buffer_size = 0;
    _status_label = NULL;
    _canvas_ready = false;

    return true;
}

void Camera::extraUiInit(void)
{
    lv_obj_add_flag(ui_PanelCameraShotTitle, LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_flag(ui_PanelCameraShotControlBg, LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_flag(ui_ButtonCameraShotBtn, LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_flag(ui_PanelCameraShotAlbum, LV_OBJ_FLAG_HIDDEN);
    lv_obj_clear_flag(ui_ImageCameraShotImage, LV_OBJ_FLAG_SCROLLABLE);

    _status_label = NULL;
}

bool Camera::ensureFrameBuffer(uint16_t width, uint16_t height)
{
    size_t needed = (size_t)width * height * sizeof(uint16_t);
    if (needed == 0) {
        return false;
    }

    if (_frame_buffer && _frame_buffer_size == needed) {
        return true;
    }

    if (_frame_buffer) {
        heap_caps_free(_frame_buffer);
        _frame_buffer = NULL;
        _frame_buffer_size = 0;
    }

    _frame_buffer = (uint8_t *)heap_caps_aligned_alloc(64, needed, MALLOC_CAP_SPIRAM);
    if (!_frame_buffer) {
        ESP_LOGE(TAG, "alloc debug frame buffer failed, size=%u", (unsigned)needed);
        return false;
    }

    _frame_buffer_size = needed;
    _canvas_ready = false;
    _frame_dsc.header.cf = LV_IMG_CF_TRUE_COLOR;
    _frame_dsc.header.always_zero = 0;
    _frame_dsc.header.reserved = 0;
    _frame_dsc.header.w = width;
    _frame_dsc.header.h = height;
    _frame_dsc.data_size = needed;
    _frame_dsc.data = _frame_buffer;

    return true;
}

void Camera::drawEyeBox(const helmet_vision_status_t &status)
{
    if (!_frame_buffer || !status.frame_valid || status.eye_bbox.w == 0 || status.eye_bbox.h == 0) {
        return;
    }

    uint16_t color = status.eye_valid
                         ? (status.eye_closed ? rgb565_color(255, 64, 64) : rgb565_color(64, 255, 128))
                         : rgb565_color(255, 208, 64);
    draw_rect_rgb565((uint16_t *)_frame_buffer, status.frame_width, status.frame_height, status.eye_bbox, color, 4);
}

void Camera::drawInferenceOverlay(const helmet_vision_status_t &status)
{
    if (!_frame_buffer || !status.frame_valid || status.frame_width == 0 || status.frame_height == 0) {
        return;
    }

    uint16_t *buffer = (uint16_t *)_frame_buffer;
    int width = status.frame_width;
    int height = status.frame_height;
    int scale = width >= 480 ? 3 : (width >= 240 ? 2 : 1);
    int x = 8;
    int y = 8;
    int line_h = 6 * scale;
    int char_w = 4 * scale;
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
             status.eye_confidence,
             status.open_score,
             status.closed_score,
             status.background_score);
    snprintf(line3, sizeof(line3), "INFER=%luMS FRAME=%lu", (unsigned long)status.last_infer_ms, (unsigned long)status.frame_seq);

    size_t max_chars = text_len(line1);
    if (text_len(line2) > max_chars) {
        max_chars = text_len(line2);
    }
    if (text_len(line3) > max_chars) {
        max_chars = text_len(line3);
    }

    fill_rect_rgb565(buffer, width, height, x - 4, y - 4, (int)max_chars * char_w + 8, line_h * 3 + 8, bg);
    draw_text3x5_rgb565(buffer, width, height, x, y, line1, scale, fg);
    draw_text3x5_rgb565(buffer, width, height, x, y + line_h, line2, scale, white);
    draw_text3x5_rgb565(buffer, width, height, x, y + line_h * 2, line3, scale, white);
}

void Camera::updateStatusLabel(const helmet_vision_status_t &status, esp_err_t frame_ret)
{
    if (!_status_label) {
        return;
    }

    char text[256];
    if (frame_ret == ESP_OK) {
        if (status.eye_valid) {
            snprintf(text,
                     sizeof(text),
                     "Vision %s | Model READY | Eye %s | Conf %.2f\nOpen %.2f | PERCLOS %.2f | Blink %lu | Box x%u y%u w%u h%u | Frame %lu",
                     status.running ? "RUN" : "STOP",
                     eye_state_name(status.eye_state),
                     status.eye_confidence,
                     status.eye_open_ratio,
                     status.perclos,
                     (unsigned long)status.blink_count,
                     status.eye_bbox.x,
                     status.eye_bbox.y,
                     status.eye_bbox.w,
                     status.eye_bbox.h,
                     (unsigned long)status.frame_seq);
        } else {
            snprintf(text,
                     sizeof(text),
                     "Vision %s | Model %s | Eye %s | Reason %s\nNoEye %lums | Frame %lu",
                     status.running ? "RUN" : "STOP",
                     status.model_ready ? "READY" : "NO_MODEL",
                     eye_state_name(status.eye_state),
                     reason_name(status.last_reason),
                     (unsigned long)status.no_eye_ms,
                     (unsigned long)status.frame_seq);
        }
    } else {
        snprintf(text,
                 sizeof(text),
                 "Vision %s | Frame unavailable: %s\nLast error: %s | Eye %s | Reason %s",
                 status.running ? "RUN" : "STOP",
                 esp_err_to_name(frame_ret),
                 esp_err_to_name(status.last_error),
                 eye_state_name(status.eye_state),
                 reason_name(status.last_reason));
    }
    lv_label_set_text(_status_label, text);
}

void Camera::updateFrame(void)
{
    helmet_vision_status_t status = {};
    helmet_vision_get_status(&status);

    esp_err_t frame_ret = ESP_ERR_INVALID_STATE;
    if (status.frame_width > 0 && status.frame_height > 0 && ensureFrameBuffer(status.frame_width, status.frame_height)) {
        frame_ret = helmet_vision_copy_frame(_frame_buffer, _frame_buffer_size, &status);
        if (frame_ret == ESP_OK) {
            drawEyeBox(status);
            drawInferenceOverlay(status);
            if (!_canvas_ready) {
                lv_canvas_set_buffer(ui_ImageCameraShotImage,
                                     _frame_buffer,
                                     status.frame_width,
                                     status.frame_height,
                                     LV_IMG_CF_TRUE_COLOR);
                lv_obj_set_size(ui_ImageCameraShotImage, status.frame_width, status.frame_height);
                lv_obj_center(ui_ImageCameraShotImage);
                _canvas_ready = true;
            } else {
                lv_obj_invalidate(ui_ImageCameraShotImage);
            }
        }
    }

    updateStatusLabel(status, frame_ret);
}

void Camera::onTimer(lv_timer_t *timer)
{
    Camera *camera = static_cast<Camera *>(timer->user_data);
    if (camera) {
        camera->updateFrame();
    }
}
