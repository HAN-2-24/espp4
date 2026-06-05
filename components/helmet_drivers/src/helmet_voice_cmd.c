#include "helmet_voice_cmd.h"

#include <stdbool.h>
#include <string.h>

#include "esp_log.h"

#include "helmet_cloud.h"
#include "helmet_state.h"
#include "helmet_voice.h"

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

    if (text_has(text, "取消警报") || text_has(text, "取消报警")) {
        return HELMET_VOICE_CMD_CANCEL_ALARM;
    }

    if (text_has(text, "状态查询") || text_has(text, "查询状态")) {
        return HELMET_VOICE_CMD_REPORT_STATUS;
    }

    return HELMET_VOICE_CMD_UNKNOWN;
}

esp_err_t helmet_voice_cmd_execute(helmet_voice_cmd_t cmd)
{
    ESP_LOGI(TAG, "execute cmd=%d", (int)cmd);

    switch (cmd) {
    case HELMET_VOICE_CMD_CANCEL_ALARM:
        ESP_LOGI(TAG, "CMD: CANCEL_ALARM");
        helmet_state_cancel_alarm();
        return helmet_voice_stop();

    case HELMET_VOICE_CMD_REPORT_STATUS:
        ESP_LOGI(TAG, "CMD: REPORT_STATUS cloud pcm");
        helmet_state_update_fusion();
        helmet_state_set_voice(HELMET_VOICE_READY);
        return helmet_cloud_request_status_pcm();

    case HELMET_VOICE_CMD_REFRESH_STATUS:
    case HELMET_VOICE_CMD_SOS:
        ESP_LOGW(TAG, "unsupported command: %d", (int)cmd);
        return ESP_ERR_NOT_SUPPORTED;

    case HELMET_VOICE_CMD_UNKNOWN:
    default:
        ESP_LOGW(TAG, "unknown command");
        return ESP_ERR_NOT_FOUND;
    }
}
