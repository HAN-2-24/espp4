#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "esp_err.h"

esp_err_t helmet_cloud_init(void);
esp_err_t helmet_cloud_update_once(void);
esp_err_t helmet_cloud_request_status_pcm(void);

#ifdef __cplusplus
}
#endif
