#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "esp_err.h"

typedef struct {
    float eye_open_ratio;
    float perclos;
    unsigned long blink_count;
    int yawn_detected;
} helmet_vision_data_t;

esp_err_t helmet_vision_init(void);
esp_err_t helmet_vision_poll_once(void);

#ifdef __cplusplus
}
#endif