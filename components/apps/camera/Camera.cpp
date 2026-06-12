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

static void draw_rect_rgb565(uint16_t *buffer, int width, int height, helmet_vision_roi_t roi, uint16_t color, int thickness)
{
    if (!buffer || width <= 0 || height <= 0 || roi.w == 0 || roi.h == 0) {
        return;
    }

    int x1 = roi.x;
    int y1 = roi.y;
    int x2 = roi.x + roi.w - 1;
    int y2 = roi.y + roi.h - 1;

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
    _control_panel(NULL),
    _button_ctx({}),
    _button_ctx_count(0)
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
        _timer = lv_timer_create(onTimer, 120, this);
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
    _control_panel = NULL;
    _button_ctx_count = 0;

    return true;
}

void Camera::extraUiInit(void)
{
    lv_obj_add_flag(ui_PanelCameraShotTitle, LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_flag(ui_PanelCameraShotControlBg, LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_flag(ui_ButtonCameraShotBtn, LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_flag(ui_PanelCameraShotAlbum, LV_OBJ_FLAG_HIDDEN);
    lv_obj_clear_flag(ui_ImageCameraShotImage, LV_OBJ_FLAG_SCROLLABLE);

    _status_label = lv_label_create(ui_ImageCameraShotImage);
    lv_obj_set_width(_status_label, 620);
    lv_obj_set_style_text_font(_status_label, &lv_font_montserrat_16, LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_set_style_text_color(_status_label, lv_color_hex(0xffffff), LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_set_style_bg_color(_status_label, lv_color_hex(0x101820), LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_set_style_bg_opa(_status_label, LV_OPA_70, LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_set_style_pad_all(_status_label, 8, LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_align(_status_label, LV_ALIGN_TOP_LEFT, 16, 16);

    _control_panel = lv_obj_create(ui_ImageCameraShotImage);
    lv_obj_set_size(_control_panel, 560, 104);
    lv_obj_set_style_radius(_control_panel, 6, LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_set_style_bg_color(_control_panel, lv_color_hex(0x101820), LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_set_style_bg_opa(_control_panel, LV_OPA_70, LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_set_style_border_width(_control_panel, 0, LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_set_style_pad_all(_control_panel, 8, LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_set_style_pad_row(_control_panel, 8, LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_set_style_pad_column(_control_panel, 8, LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_set_flex_flow(_control_panel, LV_FLEX_FLOW_ROW_WRAP);
    lv_obj_align(_control_panel, LV_ALIGN_BOTTOM_LEFT, 16, -16);

    createRoiButton(_control_panel, "X-", ROI_X_DEC);
    createRoiButton(_control_panel, "X+", ROI_X_INC);
    createRoiButton(_control_panel, "Y-", ROI_Y_DEC);
    createRoiButton(_control_panel, "Y+", ROI_Y_INC);
    createRoiButton(_control_panel, "W-", ROI_W_DEC);
    createRoiButton(_control_panel, "W+", ROI_W_INC);
    createRoiButton(_control_panel, "H-", ROI_H_DEC);
    createRoiButton(_control_panel, "H+", ROI_H_INC);
    createRoiButton(_control_panel, "Save", ROI_SAVE);
    createRoiButton(_control_panel, "Reset", ROI_RESET);
}

lv_obj_t *Camera::createRoiButton(lv_obj_t *parent, const char *text, RoiAction action)
{
    if (_button_ctx_count >= ROI_ACTION_MAX) {
        return NULL;
    }

    RoiButtonContext *ctx = &_button_ctx[_button_ctx_count++];
    ctx->camera = this;
    ctx->action = action;

    lv_obj_t *btn = lv_btn_create(parent);
    lv_obj_set_size(btn, 92, 38);
    lv_obj_set_style_radius(btn, 5, LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_set_style_bg_color(btn, lv_color_hex(0x2d6cdf), LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_add_event_cb(btn, onRoiButtonClick, LV_EVENT_CLICKED, ctx);

    lv_obj_t *label = lv_label_create(btn);
    lv_label_set_text(label, text);
    lv_obj_set_style_text_font(label, &lv_font_montserrat_16, LV_PART_MAIN | LV_STATE_DEFAULT);
    lv_obj_center(label);

    return btn;
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
    _frame_dsc.header.cf = LV_IMG_CF_TRUE_COLOR;
    _frame_dsc.header.always_zero = 0;
    _frame_dsc.header.reserved = 0;
    _frame_dsc.header.w = width;
    _frame_dsc.header.h = height;
    _frame_dsc.data_size = needed;
    _frame_dsc.data = _frame_buffer;

    return true;
}

void Camera::drawRoi(const helmet_vision_status_t &status)
{
    if (!_frame_buffer || !status.frame_valid) {
        return;
    }

    uint16_t color = status.eye_valid
                         ? (status.eye_closed ? rgb565_color(255, 64, 64) : rgb565_color(64, 255, 128))
                         : rgb565_color(255, 208, 64);
    draw_rect_rgb565((uint16_t *)_frame_buffer, status.frame_width, status.frame_height, status.roi, color, 4);
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
                     "Vision %s | Eye %s | Open %.2f | PERCLOS %.2f | Blink %lu\nROI x%u y%u w%u h%u | Frame %lu",
                     status.running ? "RUN" : "STOP",
                     status.eye_closed ? "CLOSED" : "OPEN",
                     status.eye_open_ratio,
                     status.perclos,
                     (unsigned long)status.blink_count,
                     status.roi.x,
                     status.roi.y,
                     status.roi.w,
                     status.roi.h,
                     (unsigned long)status.frame_seq);
        } else {
            snprintf(text,
                     sizeof(text),
                     "Vision %s | Eye INVALID | NoEye %lums\nROI x%u y%u w%u h%u | Frame %lu",
                     status.running ? "RUN" : "STOP",
                     (unsigned long)status.no_eye_ms,
                     status.roi.x,
                     status.roi.y,
                     status.roi.w,
                     status.roi.h,
                     (unsigned long)status.frame_seq);
        }
    } else {
        snprintf(text,
                 sizeof(text),
                 "Vision %s | Frame unavailable: %s\nLast error: %s | ROI x%u y%u w%u h%u",
                 status.running ? "RUN" : "STOP",
                 esp_err_to_name(frame_ret),
                 esp_err_to_name(status.last_error),
                 status.roi.x,
                 status.roi.y,
                 status.roi.w,
                 status.roi.h);
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
            drawRoi(status);
            lv_canvas_set_buffer(ui_ImageCameraShotImage,
                                 _frame_buffer,
                                 status.frame_width,
                                 status.frame_height,
                                 LV_IMG_CF_TRUE_COLOR);
            lv_obj_set_size(ui_ImageCameraShotImage, status.frame_width, status.frame_height);
            lv_obj_center(ui_ImageCameraShotImage);
        }
    }

    updateStatusLabel(status, frame_ret);
}

void Camera::adjustRoi(RoiAction action)
{
    helmet_vision_status_t status = {};
    if (helmet_vision_get_status(&status) != ESP_OK || status.frame_width == 0 || status.frame_height == 0) {
        return;
    }

    helmet_vision_roi_t roi = status.roi;
    int step_x = status.frame_width / 80;
    int step_y = status.frame_height / 80;
    if (step_x < 4) {
        step_x = 4;
    }
    if (step_y < 4) {
        step_y = 4;
    }

    switch (action) {
    case ROI_X_DEC:
        roi.x = (uint16_t)((roi.x > step_x) ? roi.x - step_x : 0);
        break;
    case ROI_X_INC:
        roi.x = (uint16_t)(roi.x + step_x);
        break;
    case ROI_Y_DEC:
        roi.y = (uint16_t)((roi.y > step_y) ? roi.y - step_y : 0);
        break;
    case ROI_Y_INC:
        roi.y = (uint16_t)(roi.y + step_y);
        break;
    case ROI_W_DEC:
        roi.w = (uint16_t)((roi.w > step_x * 2) ? roi.w - step_x : roi.w);
        break;
    case ROI_W_INC:
        roi.w = (uint16_t)(roi.w + step_x);
        break;
    case ROI_H_DEC:
        roi.h = (uint16_t)((roi.h > step_y * 2) ? roi.h - step_y : roi.h);
        break;
    case ROI_H_INC:
        roi.h = (uint16_t)(roi.h + step_y);
        break;
    case ROI_SAVE:
        helmet_vision_set_roi(&roi, true);
        updateFrame();
        return;
    case ROI_RESET:
        helmet_vision_reset_roi();
        updateFrame();
        return;
    default:
        return;
    }

    helmet_vision_set_roi(&roi, false);
    updateFrame();
}

void Camera::onTimer(lv_timer_t *timer)
{
    Camera *camera = static_cast<Camera *>(timer->user_data);
    if (camera) {
        camera->updateFrame();
    }
}

void Camera::onRoiButtonClick(lv_event_t *e)
{
    RoiButtonContext *ctx = static_cast<RoiButtonContext *>(lv_event_get_user_data(e));
    if (ctx && ctx->camera) {
        ctx->camera->adjustRoi(ctx->action);
    }
}
