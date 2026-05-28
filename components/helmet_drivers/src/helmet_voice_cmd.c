#include "helmet_voice_cmd.h"

#include "helmet_voice.h"
#include "helmet_state.h"

#include "esp_log.h"

#include <string.h>

static const char *TAG = "helmet_voice_cmd";

helmet_voice_cmd_t helmet_voice_cmd_parse(const char *text)
{
    if (text == NULL) {
        return HELMET_VOICE_CMD_UNKNOWN;
    }

    if (strstr(text, "取消报警") || strstr(text, "停止报警") || strstr(text, "关闭报警")) {
        return HELMET_VOICE_CMD_CANCEL_ALARM;
    }

    if (strstr(text, "刷新") || strstr(text, "查看状态")) {
        return HELMET_VOICE_CMD_REFRESH_STATUS;
    }

    if (strstr(text, "播报状态") || strstr(text, "当前状态")) {
        return HELMET_VOICE_CMD_REPORT_STATUS;
    }

    if (strstr(text, "求救") || strstr(text, "SOS") || strstr(text, "救命")) {
        return HELMET_VOICE_CMD_SOS;
    }

    return HELMET_VOICE_CMD_UNKNOWN;
}

esp_err_t helmet_voice_cmd_execute(helmet_voice_cmd_t cmd)
{
    ESP_LOGI(TAG, "execute cmd=%d", cmd);

    switch (cmd) {
    case HELMET_VOICE_CMD_CANCEL_ALARM:
        /*
         * 这里替换成你项目里真实的取消报警函数。
         * 例如：
         * helmet_state_cancel_alarm();
         */
        helmet_voice_stop();
        return ESP_OK;

    case HELMET_VOICE_CMD_REFRESH_STATUS:
        /*
         * 这里替换成你的刷新状态逻辑。
         */
        return ESP_OK;

    case HELMET_VOICE_CMD_REPORT_STATUS:
        /*
         * 后续可以根据 helmet_state 当前状态播放不同提示音。
         */
        return helmet_voice_play_alert(0);

    case HELMET_VOICE_CMD_SOS:
        /*
         * 后续接 SOS / Cloud / GNSS。
         */
        return ESP_OK;

    default:
        ESP_LOGW(TAG, "unknown command");
        return ESP_ERR_NOT_FOUND;
    }
}
