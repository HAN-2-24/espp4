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
} helmet_vision_bbox_t;

typedef enum {
    HELMET_VISION_EYE_STATE_UNKNOWN = 0,
    HELMET_VISION_EYE_STATE_OPEN,
    HELMET_VISION_EYE_STATE_CLOSED,
    HELMET_VISION_EYE_STATE_INVALID,
    HELMET_VISION_EYE_STATE_NO_MODEL,
} helmet_vision_eye_state_t;

typedef enum {
    HELMET_VISION_REASON_NONE = 0,
    HELMET_VISION_REASON_NO_MODEL,
    HELMET_VISION_REASON_NO_DETECTION,
    HELMET_VISION_REASON_RUNTIME_BUSY,
    HELMET_VISION_REASON_LOW_CONFIDENCE,
    HELMET_VISION_REASON_CAMERA_ERROR,
} helmet_vision_reason_t;

typedef struct {
    bool running;
    bool frame_valid;
    bool eye_valid;
    bool eye_closed;
    bool model_ready;
    uint16_t frame_width;
    uint16_t frame_height;
    uint32_t frame_seq;
    helmet_vision_bbox_t eye_bbox;
    helmet_vision_eye_state_t eye_state;
    helmet_vision_reason_t last_reason;
    float eye_open_ratio;
    float eye_confidence;
    float open_score;
    float closed_score;
    float background_score;
    float perclos;
    uint32_t blink_count;
    uint32_t no_eye_ms;
    uint32_t last_infer_ms;
    esp_err_t last_error;
} helmet_vision_status_t;

esp_err_t helmet_vision_init(void);
esp_err_t helmet_vision_start(void);
esp_err_t helmet_vision_poll_once(void);
esp_err_t helmet_vision_get_status(helmet_vision_status_t *out_status);
esp_err_t helmet_vision_copy_frame(void *dst, size_t dst_size, helmet_vision_status_t *out_status);
esp_err_t helmet_vision_copy_scaled_frame(void *dst, size_t dst_size, uint8_t scale, helmet_vision_status_t *out_status);

#ifdef __cplusplus
}
#endif
