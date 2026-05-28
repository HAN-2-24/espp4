#include "helmet_state.h"

#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static helmet_state_t s_state;
static SemaphoreHandle_t s_mutex;

static void lock_state(void)
{
    if (s_mutex) {
        xSemaphoreTake(s_mutex, portMAX_DELAY);
    }
}

static void unlock_state(void)
{
    if (s_mutex) {
        xSemaphoreGive(s_mutex);
    }
}

const char *helmet_risk_to_string(helmet_risk_level_t risk)
{
    switch (risk) {
    case HELMET_RISK_NORMAL: return "NORMAL";
    case HELMET_RISK_ATTENTION: return "ATTENTION";
    case HELMET_RISK_FATIGUE: return "FATIGUE";
    case HELMET_RISK_DANGER: return "DANGER";
    default: return "UNKNOWN";
    }
}

const char *helmet_voice_to_string(helmet_voice_state_t voice)
{
    switch (voice) {
    case HELMET_VOICE_IDLE: return "IDLE";
    case HELMET_VOICE_READY: return "READY";
    case HELMET_VOICE_PLAYING: return "PLAYING";
    case HELMET_VOICE_ERROR: return "ERROR";
    default: return "UNKNOWN";
    }
}

const char *helmet_cloud_to_string(helmet_cloud_state_t cloud)
{
    switch (cloud) {
    case HELMET_CLOUD_OFFLINE: return "OFFLINE";
    case HELMET_CLOUD_CONNECTING: return "CONNECTING";
    case HELMET_CLOUD_ONLINE: return "ONLINE";
    case HELMET_CLOUD_ERROR: return "ERROR";
    default: return "UNKNOWN";
    }
}

void helmet_state_init(void)
{
    if (!s_mutex) {
        s_mutex = xSemaphoreCreateMutex();
    }

    lock_state();

    memset(&s_state, 0, sizeof(s_state));

    s_state.risk = HELMET_RISK_NORMAL;
    s_state.voice = HELMET_VOICE_IDLE;
    s_state.cloud = HELMET_CLOUD_OFFLINE;
    s_state.voice_enabled = true;
    s_state.cloud_enabled = true;

    snprintf(s_state.explanation, sizeof(s_state.explanation), "Helmet system initialized.");
    snprintf(s_state.report, sizeof(s_state.report), "No fatigue report.");

    unlock_state();
}

void helmet_state_set_eye(float eye_open_ratio, float perclos, uint32_t blink_count, bool yawn_detected)
{
    lock_state();

    s_state.eye.valid = true;
    s_state.eye.eye_open_ratio = eye_open_ratio;
    s_state.eye.perclos = perclos;
    s_state.eye.blink_count = blink_count;
    s_state.eye.yawn_detected = yawn_detected;

    unlock_state();
}

void helmet_state_set_pose(float roll, float pitch, float acc_total, bool fall_detected, bool impact_detected)
{
    lock_state();

    s_state.pose.valid = true;
    s_state.pose.roll = roll;
    s_state.pose.pitch = pitch;
    s_state.pose.acc_total = acc_total;
    s_state.pose.fall_detected = fall_detected;
    s_state.pose.impact_detected = impact_detected;

    unlock_state();
}

void helmet_state_set_location(bool valid, double latitude, double longitude, float speed, float course)
{
    lock_state();

    s_state.location.valid = valid;
    s_state.location.latitude = latitude;
    s_state.location.longitude = longitude;
    s_state.location.speed = speed;
    s_state.location.course = course;

    unlock_state();
}

void helmet_state_set_voice(helmet_voice_state_t voice)
{
    lock_state();
    s_state.voice = voice;
    unlock_state();
}

void helmet_state_set_cloud(helmet_cloud_state_t cloud)
{
    lock_state();
    s_state.cloud = cloud;
    unlock_state();
}

void helmet_state_request_alarm(void)
{
    lock_state();
    s_state.alarm_active = true;
    unlock_state();
}

void helmet_state_cancel_alarm(void)
{
    lock_state();
    s_state.alarm_active = false;
    unlock_state();
}

void helmet_state_update_fusion(void)
{
    lock_state();

    helmet_risk_level_t risk = HELMET_RISK_NORMAL;

    if (s_state.pose.valid && (s_state.pose.fall_detected || s_state.pose.impact_detected)) {
        risk = HELMET_RISK_DANGER;
        s_state.alarm_active = true;
    } else if (s_state.eye.valid && (s_state.eye.perclos >= 0.45f || s_state.eye.yawn_detected)) {
        risk = HELMET_RISK_FATIGUE;
        s_state.alarm_active = true;
    } else if (s_state.eye.valid && s_state.eye.perclos >= 0.25f) {
        risk = HELMET_RISK_ATTENTION;
    } else if (s_state.pose.valid && (s_state.pose.roll >= 45.0f || s_state.pose.roll <= -45.0f)) {
        risk = HELMET_RISK_ATTENTION;
    }

    s_state.risk = risk;

    snprintf(
        s_state.explanation,
        sizeof(s_state.explanation),
        "Risk=%s, PERCLOS=%.2f, Roll=%.1f, Pitch=%.1f, Acc=%.2f, GPS=%s.",
        helmet_risk_to_string(s_state.risk),
        s_state.eye.perclos,
        s_state.pose.roll,
        s_state.pose.pitch,
        s_state.pose.acc_total,
        s_state.location.valid ? "VALID" : "INVALID"
    );

    snprintf(
        s_state.report,
        sizeof(s_state.report),
        "EyeOpen %.2f | PERCLOS %.2f | Blink %lu | Roll %.1f | Pitch %.1f | Lat %.6f | Lon %.6f",
        s_state.eye.eye_open_ratio,
        s_state.eye.perclos,
        (unsigned long)s_state.eye.blink_count,
        s_state.pose.roll,
        s_state.pose.pitch,
        s_state.location.latitude,
        s_state.location.longitude
    );

    unlock_state();
}

helmet_state_t helmet_state_get_copy(void)
{
    helmet_state_t copy;

    lock_state();
    copy = s_state;
    unlock_state();

    return copy;
}