#pragma once

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    HELMET_RISK_NORMAL = 0,
    HELMET_RISK_ATTENTION,
    HELMET_RISK_FATIGUE,
    HELMET_RISK_DANGER,
} HelmetRiskLevel;

typedef enum {
    HELMET_VOICE_IDLE = 0,
    HELMET_VOICE_PLAYING,
    HELMET_VOICE_ERROR,
} HelmetVoiceState;

typedef enum {
    HELMET_CLOUD_OFFLINE = 0,
    HELMET_CLOUD_CONNECTING,
    HELMET_CLOUD_ONLINE,
    HELMET_CLOUD_ERROR,
} HelmetCloudState;

typedef struct {
    bool valid;
    float eye_open_ratio;
    float perclos;
    uint32_t blink_count;
    bool yawn_detected;
} HelmetEyeState;

typedef struct {
    bool valid;
    float roll;
    float pitch;
    float acc_total;
    bool fall_detected;
    bool impact_detected;
} HelmetPoseState;

typedef struct {
    bool valid;
    double latitude;
    double longitude;
    float speed;
    float course;
} HelmetLocationState;

typedef struct {
    HelmetEyeState eye;
    HelmetPoseState pose;
    HelmetLocationState location;

    HelmetRiskLevel risk;
    HelmetVoiceState voice;
    HelmetCloudState cloud;

    bool alarm_active;
    bool local_voice_enabled;
    bool cloud_llm_enabled;

    char explanation[256];
    char report_summary[256];
} HelmetSystemState;

class HelmetSystem {
public:
    HelmetSystem();

    void init(void);
    void update(void);

    const HelmetSystemState &getState(void) const;

    void setEyeState(float eye_open_ratio, float perclos, uint32_t blink_count, bool yawn_detected);
    void setPoseState(float roll, float pitch, float acc_total, bool fall_detected, bool impact_detected);
    void setLocationState(bool valid, double latitude, double longitude, float speed, float course);
    void setVoiceState(HelmetVoiceState state);
    void setCloudState(HelmetCloudState state);

    void setLocalVoiceEnabled(bool enabled);
    void setCloudLlmEnabled(bool enabled);

    void requestAlarm(void);
    void cancelAlarm(void);

private:
    void evaluateRisk(void);
    void updateExplanation(void);

private:
    HelmetSystemState _state;
};