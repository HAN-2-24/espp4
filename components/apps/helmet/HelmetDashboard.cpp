#include "HelmetDashboard.hpp"

#include <stdio.h>

#include "esp_log.h"

#include "helmet_board.h"
#include "helmet_state.h"
#include "helmet_gnss.h"
#include "helmet_imu.h"
#include "helmet_voice.h"
#include "helmet_vision.h"
#include "helmet_cloud.h"
#include "helmet_wifi.h"
#include "ui/ui_helmet.h"

static const char *TAG = "HelmetDashboard";

LV_IMG_DECLARE(img_app_setting);

HelmetDashboard::HelmetDashboard():
    ESP_Brookesia_PhoneApp("Helmet", &img_app_setting, false),
    _is_ui_del(true),
    _is_ui_resumed(false),
    _timer(nullptr)
{
}

HelmetDashboard::~HelmetDashboard()
{
}

bool HelmetDashboard::init(void)
{
    ESP_LOGI(TAG, "init");

    helmet_board_init();
    helmet_state_init();
    helmet_gnss_init();
    helmet_imu_init();
    helmet_voice_init();
    helmet_vision_init();
   
    helmet_cloud_init();

    return true;
}

bool HelmetDashboard::run(void)
{
    ESP_LOGI(TAG, "run");

    _is_ui_del = false;
    _is_ui_resumed = true;

    ui_helmet_init();
    extraUiInit();

    feedBootDemoState();
    helmet_state_update_fusion();
    updateUi();

    if (_timer == nullptr) {
        _timer = lv_timer_create(onTimer, 1000, this);
    }

    return true;
}

bool HelmetDashboard::back(void)
{
    ESP_LOGI(TAG, "back");
    notifyCoreClosed();
    return true;
}

bool HelmetDashboard::close(void)
{
    ESP_LOGI(TAG, "close");

    _is_ui_del = true;
    _is_ui_resumed = false;

    if (_timer != nullptr) {
        lv_timer_del(_timer);
        _timer = nullptr;
    }

    return true;
}

bool HelmetDashboard::pause(void)
{
    _is_ui_resumed = false;
    return true;
}

bool HelmetDashboard::resume(void)
{
    _is_ui_resumed = true;
    updateUi();
    return true;
}

void HelmetDashboard::extraUiInit(void)
{
    lv_obj_add_event_cb(ui_ButtonRefresh, onRefreshClicked, LV_EVENT_CLICKED, this);
    lv_obj_add_event_cb(ui_ButtonCancelAlarm, onCancelAlarmClicked, LV_EVENT_CLICKED, this);
}

void HelmetDashboard::feedBootDemoState(void)
{
    /*
     * 只用于第一阶段验证 UI 与状态层。
     * 接入真实驱动后删除这里的模拟输入。
     */
    helmet_state_set_eye(0.72f, 0.18f, 26, false);
    helmet_state_set_pose(3.5f, -2.1f, 0.12f, false, false);
    //helmet_state_set_location(true, 31.230416, 121.473701, 0.0f, 0.0f);
    helmet_state_set_voice(HELMET_VOICE_READY);
    helmet_state_set_cloud(HELMET_CLOUD_OFFLINE);
}

void HelmetDashboard::updateUi(void)
{
    if (_is_ui_del || !_is_ui_resumed) {
        return;
    }

    helmet_state_t state = helmet_state_get_copy();

    char buf[256];

    snprintf(buf, sizeof(buf), "Risk: %s", helmet_risk_to_string(state.risk));
    lv_label_set_text(ui_LabelRisk, buf);

    snprintf(buf, sizeof(buf), "Alarm: %s", state.alarm_active ? "ACTIVE" : "NONE");
    lv_label_set_text(ui_LabelAlarm, buf);

    snprintf(
        buf,
        sizeof(buf),
        "EyeOpen %.2f | PERCLOS %.2f | Blink %lu | Yawn %s",
        state.eye.eye_open_ratio,
        state.eye.perclos,
        (unsigned long)state.eye.blink_count,
        state.eye.yawn_detected ? "YES" : "NO"
    );
    lv_label_set_text(ui_LabelEye, buf);

    snprintf(
        buf,
        sizeof(buf),
        "Roll %.1f | Pitch %.1f | Acc %.2f | Fall %s | Impact %s",
        state.pose.roll,
        state.pose.pitch,
        state.pose.acc_total,
        state.pose.fall_detected ? "YES" : "NO",
        state.pose.impact_detected ? "YES" : "NO"
    );
    lv_label_set_text(ui_LabelPose, buf);

    snprintf(
        buf,
        sizeof(buf),
        "%s | Lat %.6f | Lon %.6f | Speed %.1f",
        state.location.valid ? "VALID" : "INVALID",
        state.location.latitude,
        state.location.longitude,
        state.location.speed
    );
    lv_label_set_text(ui_LabelLocation, buf);

    snprintf(
        buf,
        sizeof(buf),
        "%s | Enabled %s",
        helmet_voice_to_string(state.voice),
        state.voice_enabled ? "YES" : "NO"
    );
    lv_label_set_text(ui_LabelVoice, buf);

    snprintf(
        buf,
        sizeof(buf),
        "%s | Enabled %s",
        helmet_cloud_to_string(state.cloud),
        state.cloud_enabled ? "YES" : "NO"
    );
    lv_label_set_text(ui_LabelCloud, buf);

    lv_label_set_text(ui_LabelExplanation, state.explanation);
    lv_label_set_text(ui_LabelReport, state.report);
}

void HelmetDashboard::onRefreshClicked(lv_event_t *e)
{
    HelmetDashboard *app = static_cast<HelmetDashboard *>(lv_event_get_user_data(e));
    if (app == nullptr) {
        return;
    }

    //helmet_gnss_poll_once();
    helmet_imu_poll_once();
    helmet_vision_poll_once();
    helmet_cloud_update_once();

    helmet_state_update_fusion();
    app->updateUi();
}

void HelmetDashboard::onCancelAlarmClicked(lv_event_t *e)
{
    HelmetDashboard *app = static_cast<HelmetDashboard *>(lv_event_get_user_data(e));
    if (app == nullptr) {
        return;
    }

    helmet_state_cancel_alarm();
    helmet_voice_stop();
    helmet_state_update_fusion();
    app->updateUi();
}

void HelmetDashboard::onTimer(lv_timer_t *timer)
{
    HelmetDashboard *app = static_cast<HelmetDashboard *>(timer->user_data);
    if (app == nullptr) {
        return;
    }

   // helmet_gnss_poll_once();
    helmet_imu_poll_once();
    helmet_vision_poll_once();
    helmet_cloud_update_once();

    helmet_state_update_fusion();
    app->updateUi();
}