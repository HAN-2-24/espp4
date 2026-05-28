#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HELMET_VOICE_CMD_UNKNOWN = 0,
    HELMET_VOICE_CMD_CANCEL_ALARM,
    HELMET_VOICE_CMD_REFRESH_STATUS,
    HELMET_VOICE_CMD_REPORT_STATUS,
    HELMET_VOICE_CMD_SOS,
} helmet_voice_cmd_t;

helmet_voice_cmd_t helmet_voice_cmd_parse(const char *text);
esp_err_t helmet_voice_cmd_execute(helmet_voice_cmd_t cmd);

#ifdef __cplusplus
}
#endif