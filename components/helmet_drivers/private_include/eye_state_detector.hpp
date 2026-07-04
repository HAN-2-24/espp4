#pragma once

#include "esp_err.h"
#include "helmet_vision.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    bool valid;
    bool closed;
    helmet_vision_eye_state_t eye_state;
    helmet_vision_reason_t reason;
    helmet_vision_bbox_t bbox;
    float open_ratio;
    float confidence;
    float open_score;
    float closed_score;
    float background_score;
} eye_state_detector_result_t;

esp_err_t eye_state_detector_init(void);
bool eye_state_detector_model_ready(void);
eye_state_detector_result_t eye_state_detector_detect(const uint8_t *rgb565,
                                                       uint16_t width,
                                                       uint16_t height,
                                                       size_t stride_bytes);
