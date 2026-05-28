#include "HelmetSystem.hpp"
#include <string.h>
#include <stdio.h>

HelmetSystem::HelmetSystem()
{
    memset(&_state, 0, sizeof(_state));
}

void HelmetSystem::init(void)
{
    memset(&_state, 0, sizeof(_state));

    _state.risk = HELMET_RISK_NORMAL;
    _state.voice = HELMET_VOICE_IDLE;
    _state.cloud = HELMET_CLOUD_OFFLINE;
    _state.local_voice_enabled = true;
    _state.cloud_llm_enabled = true;

    snprintf(_state.explanation, sizeof(_state.explanation), "System initialized.");
    snprintf(_state.report_summary, sizeof(_state.report_summary), "No fatigue report generated.");
}

void HelmetSystem::update(void)
{
    evaluateRisk();
    updateExplanation();
}

const HelmetSystemState &HelmetSystem::getState(void) const
{
    return _state;
}

void HelmetSystem::setEyeState(float eye_open_ratio, float perclos, uint32_t blink_count, bool yawn_detected)
{
    _state.eye.valid = true;
    _state.eye.eye_open_ratio = eye_open_ratio;
    _state.eye.perclos = perclos;
    _state.eye.blink_count = blink_count;
    _state.eye.yawn_detected = yawn_detected;
}

void HelmetSystem::setPoseState(float roll, float pitch, float acc_total, bool fall_detected, bool impact_detected)
{
    _state.pose.valid = true;
    _state.pose.roll = roll;
    _state.pose.pitch = pitch;
    _state.pose.acc_total = acc_total;
    _state.pose.fall_detected = fall_detected;
    _state.pose.impact_detected = impact_detected;
}

void HelmetSystem::setLocationState(bool valid, double latitude, double longitude, float speed, float course)
{
    _state.location.valid = valid;
    _state.location.latitude = latitude;
    _state.location.longitude = longitude;
    _state.location.speed = speed;
    _state.location.course = course;
}

void HelmetSystem::setVoiceState(HelmetVoiceState state)
{
    _state.voice = state;
}

void HelmetSystem::setCloudState(HelmetCloudState state)
{
    _state.cloud = state;
}

void HelmetSystem::setLocalVoiceEnabled(bool enabled)
{
    _state.local_voice_enabled = enabled;
}

void HelmetSystem::setCloudLlmEnabled(bool enabled)
{
    _state.cloud_llm_enabled = enabled;
}

void HelmetSystem::requestAlarm(void)
{
    _state.alarm_active = true;
}

void HelmetSystem::cancelAlarm(void)
{
    _state.alarm_active = false;
}

void HelmetSystem::evaluateRisk(void)
{
    HelmetRiskLevel risk = HELMET_RISK_NORMAL;

    if (_state.pose.fall_detected || _state.pose.impact_detected) {
        risk = HELMET_RISK_DANGER;
        _state.alarm_active = true;
    } else if (_state.eye.valid) {
        if (_state.eye.perclos >= 0.45f || _state.eye.yawn_detected) {
            risk = HELMET_RISK_FATIGUE;
            _state.alarm_active = true;
        } else if (_state.eye.perclos >= 0.25f) {
            risk = HELMET_RISK_ATTENTION;
        }
    }

    if (_state.pose.valid && (_state.pose.roll > 45.0f || _state.pose.roll < -45.0f)) {
        if (risk < HELMET_RISK_ATTENTION) {
            risk = HELMET_RISK_ATTENTION;
        }
    }

    _state.risk = risk;
}

void HelmetSystem::updateExplanation(void)
{
    const char *risk_text = "NORMAL";

    switch (_state.risk) {
    case HELMET_RISK_NORMAL:
        risk_text = "NORMAL";
        break;
    case HELMET_RISK_ATTENTION:
        risk_text = "ATTENTION";
        break;
    case HELMET_RISK_FATIGUE:
        risk_text = "FATIGUE";
        break;
    case HELMET_RISK_DANGER:
        risk_text = "DANGER";
        break;
    default:
        break;
    }

    snprintf(
        _state.explanation,
        sizeof(_state.explanation),
        "Risk=%s, PERCLOS=%.2f, Roll=%.1f, Pitch=%.1f, Acc=%.2f, GPS=%s.",
        risk_text,
        _state.eye.perclos,
        _state.pose.roll,
        _state.pose.pitch,
        _state.pose.acc_total,
        _state.location.valid ? "VALID" : "INVALID"
    );

    snprintf(
        _state.report_summary,
        sizeof(_state.report_summary),
        "Eye: %.2f / PERCLOS %.2f / Blink %lu | Pose: R %.1f P %.1f | Location: %.6f, %.6f",
        _state.eye.eye_open_ratio,
        _state.eye.perclos,
        (unsigned long)_state.eye.blink_count,
        _state.pose.roll,
        _state.pose.pitch,
        _state.location.latitude,
        _state.location.longitude
    );
}