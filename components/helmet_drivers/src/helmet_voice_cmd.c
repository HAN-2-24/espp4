#include "helmet_voice_cmd.h"

#include "helmet_state.h"
#include "helmet_voice.h"

#include "esp_log.h"

#include <stdbool.h>
#include <string.h>

static const char *TAG = "helmet_voice_cmd";

static bool text_has(const char *text, const char *needle)
{
    return text != NULL && needle != NULL && strstr(text, needle) != NULL;
}

helmet_voice_cmd_t helmet_voice_cmd_parse(const char *text)
{
    if (text == NULL) {
        return HELMET_VOICE_CMD_UNKNOWN;
    }

    if (text_has(text, "取消报警") ||
        text_has(text, "停止报警") ||
        text_has(text, "关闭报警")) {
        return HELMET_VOICE_CMD_CANCEL_ALARM;
    }

    if (text_has(text, "刷新") ||
        text_has(text, "查看状态") ||
        text_has(text, "更新状态")) {
        return HELMET_VOICE_CMD_REFRESH_STATUS;
    }

    if (text_has(text, "播报状态") ||
        text_has(text, "当前状态") ||
        text_has(text, "什么状态")) {
        return HELMET_VOICE_CMD_REPORT_STATUS;
    }

    if (text_has(text, "求救") ||
        text_has(text, "救命") ||
        text_has(text, "SOS") ||
        text_has(text, "sos")) {
        return HELMET_VOICE_CMD_SOS;
    }

    return HELMET_VOICE_CMD_UNKNOWN;
}

esp_err_t helmet_voice_cmd_execute(helmet_voice_cmd_t cmd)
{
    ESP_LOGI(TAG, "execute cmd=%d", (int)cmd);

    switch (cmd) {
    case HELMET_VOICE_CMD_CANCEL_ALARM:
        ESP_LOGI(TAG, "CMD: CANCEL_ALARM state placeholder");
        helmet_state_cancel_alarm();
        return helmet_voice_stop();

    case HELMET_VOICE_CMD_REFRESH_STATUS:
        ESP_LOGI(TAG, "CMD: REFRESH_STATUS");
        helmet_state_update_fusion();
        helmet_state_set_voice(HELMET_VOICE_READY);
        return ESP_OK;

    case HELMET_VOICE_CMD_REPORT_STATUS:
        ESP_LOGI(TAG, "CMD: REPORT_STATUS beep placeholder");
        return helmet_voice_play_alert(HELMET_VOICE_ALERT_FATIGUE);

    case HELMET_VOICE_CMD_SOS:
        ESP_LOGI(TAG, "CMD: SOS state placeholder");
        helmet_state_request_alarm();
        helmet_state_set_voice(HELMET_VOICE_READY);
        return ESP_OK;

    case HELMET_VOICE_CMD_UNKNOWN:
    default:
        ESP_LOGW(TAG, "unknown command");
        return ESP_ERR_NOT_FOUND;
    }
}
