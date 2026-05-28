#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "esp_err.h"

esp_err_t helmet_wifi_init(void);
bool helmet_wifi_is_connected(void);
esp_err_t helmet_wifi_wait_connected(uint32_t timeout_ms);
const char *helmet_wifi_get_ip_string(void);

#ifdef __cplusplus
}
#endif