#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

typedef enum {
    HELMET_RISK_NORMAL = 0,
    HELMET_RISK_ATTENTION,
    HELMET_RISK_FATIGUE,
    HELMET_RISK_DANGER,
} helmet_risk_level_t;

typedef enum {
    HELMET_VOICE_IDLE = 0,
    HELMET_VOICE_READY,
    HELMET_VOICE_PLAYING,
    HELMET_VOICE_ERROR,
} helmet_voice_state_t;

typedef enum {
    HELMET_CLOUD_OFFLINE = 0,
    HELMET_CLOUD_CONNECTING,
    HELMET_CLOUD_ONLINE,
    HELMET_CLOUD_ERROR,
} helmet_cloud_state_t;

typedef struct {
    bool valid;
    float eye_open_ratio;
    float perclos;
    uint32_t blink_count;
    bool yawn_detected;
} helmet_eye_state_t;

typedef struct {
    bool valid;
    float roll;
    float pitch;
    float acc_total;
    bool fall_detected;
    bool impact_detected;
} helmet_pose_state_t;

typedef struct {
    bool valid;
    double latitude;
    double longitude;
    float speed;
    float course;
} helmet_location_state_t;

typedef struct {
    helmet_eye_state_t eye;
    helmet_pose_state_t pose;
    helmet_location_state_t location;

    helmet_risk_level_t risk;
    helmet_voice_state_t voice;
    helmet_cloud_state_t cloud;

    bool alarm_active;
    bool voice_enabled;
    bool cloud_enabled;

    char explanation[256];
    char report[256];
} helmet_state_t;

void helmet_state_init(void);

void helmet_state_set_eye(float eye_open_ratio, float perclos, uint32_t blink_count, bool yawn_detected);
void helmet_state_set_pose(float roll, float pitch, float acc_total, bool fall_detected, bool impact_detected);
void helmet_state_set_location(bool valid, double latitude, double longitude, float speed, float course);
void helmet_state_set_voice(helmet_voice_state_t voice);
void helmet_state_set_cloud(helmet_cloud_state_t cloud);

void helmet_state_request_alarm(void);
void helmet_state_cancel_alarm(void);

void helmet_state_update_fusion(void);
helmet_state_t helmet_state_get_copy(void);

const char *helmet_risk_to_string(helmet_risk_level_t risk);
const char *helmet_voice_to_string(helmet_voice_state_t voice);
const char *helmet_cloud_to_string(helmet_cloud_state_t cloud);

#ifdef __cplusplus
}
#endif