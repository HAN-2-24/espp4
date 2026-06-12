/*
 * SPDX-FileCopyrightText: 2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "lvgl.h"
#include "esp_brookesia.hpp"
#include "helmet_vision.h"

class Camera: public ESP_Brookesia_PhoneApp {
public:
    Camera(uint16_t hor_res, uint16_t ver_res);
    ~Camera();

    bool run(void);
    bool pause(void);
    bool resume(void);
    bool back(void);
    bool close(void);

    bool init(void) override;

private:
    enum RoiAction {
        ROI_X_DEC,
        ROI_X_INC,
        ROI_Y_DEC,
        ROI_Y_INC,
        ROI_W_DEC,
        ROI_W_INC,
        ROI_H_DEC,
        ROI_H_INC,
        ROI_SAVE,
        ROI_RESET,
        ROI_ACTION_MAX,
    };

    struct RoiButtonContext {
        Camera *camera;
        RoiAction action;
    };

    static void onTimer(lv_timer_t *timer);
    static void onRoiButtonClick(lv_event_t *e);

    void extraUiInit(void);
    void updateFrame(void);
    void updateStatusLabel(const helmet_vision_status_t &status, esp_err_t frame_ret);
    bool ensureFrameBuffer(uint16_t width, uint16_t height);
    void drawRoi(const helmet_vision_status_t &status);
    void adjustRoi(RoiAction action);
    lv_obj_t *createRoiButton(lv_obj_t *parent, const char *text, RoiAction action);

    uint16_t _hor_res;
    uint16_t _ver_res;
    uint8_t *_frame_buffer;
    size_t _frame_buffer_size;
    lv_img_dsc_t _frame_dsc;
    lv_timer_t *_timer;
    lv_obj_t *_status_label;
    lv_obj_t *_control_panel;
    RoiButtonContext _button_ctx[ROI_ACTION_MAX];
    int _button_ctx_count;
};

