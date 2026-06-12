#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "esp_err.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    float eye_open_ratio;
    float perclos;
    unsigned long blink_count;
    int yawn_detected;
} helmet_vision_data_t;

typedef struct {
    uint16_t x;
    uint16_t y;
    uint16_t w;
    uint16_t h;
} helmet_vision_roi_t;

typedef struct {
    bool running;
    bool frame_valid;
    bool eye_valid;
    bool eye_closed;
    uint16_t frame_width;
    uint16_t frame_height;
    uint32_t frame_seq;
    helmet_vision_roi_t roi;
    float eye_open_ratio;
    float perclos;
    uint32_t blink_count;
    uint32_t no_eye_ms;
    esp_err_t last_error;
} helmet_vision_status_t;

esp_err_t helmet_vision_init(void);
esp_err_t helmet_vision_poll_once(void);
esp_err_t helmet_vision_get_status(helmet_vision_status_t *out_status);
esp_err_t helmet_vision_copy_frame(void *dst, size_t dst_size, helmet_vision_status_t *out_status);
esp_err_t helmet_vision_get_roi(helmet_vision_roi_t *out_roi);
esp_err_t helmet_vision_set_roi(const helmet_vision_roi_t *roi, bool save_to_nvs);
esp_err_t helmet_vision_reset_roi(void);

#ifdef __cplusplus
}
#endif
