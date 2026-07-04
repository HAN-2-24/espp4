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
    static void onTimer(lv_timer_t *timer);

    void extraUiInit(void);
    void updateFrame(void);
    void updateStatusLabel(const helmet_vision_status_t &status, esp_err_t frame_ret);
    bool ensureFrameBuffer(uint16_t width, uint16_t height);
    void drawEyeBox(const helmet_vision_status_t &status);
    void drawInferenceOverlay(const helmet_vision_status_t &status);

    uint16_t _hor_res;
    uint16_t _ver_res;
    uint8_t *_frame_buffer;
    size_t _frame_buffer_size;
    lv_img_dsc_t _frame_dsc;
    lv_timer_t *_timer;
    lv_obj_t *_status_label;
    bool _canvas_ready;
};

