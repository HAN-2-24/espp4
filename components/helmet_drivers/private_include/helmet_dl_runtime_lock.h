#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "esp_err.h"

#include <stdint.h>

esp_err_t helmet_dl_runtime_lock_init(void);
esp_err_t helmet_dl_runtime_lock_acquire(uint32_t timeout_ms);
void helmet_dl_runtime_lock_release(void);

#ifdef __cplusplus
}
#endif
