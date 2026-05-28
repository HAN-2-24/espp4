#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "esp_err.h"

esp_err_t helmet_imu_init(void);
esp_err_t helmet_imu_poll_once(void);

#ifdef __cplusplus
}
#endif