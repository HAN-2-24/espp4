#include "helmet_dl_runtime_lock.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static SemaphoreHandle_t s_dl_runtime_lock;

esp_err_t helmet_dl_runtime_lock_init(void)
{
    if (s_dl_runtime_lock != NULL) {
        return ESP_OK;
    }

    s_dl_runtime_lock = xSemaphoreCreateMutex();
    if (s_dl_runtime_lock == NULL) {
        return ESP_ERR_NO_MEM;
    }

    return ESP_OK;
}

esp_err_t helmet_dl_runtime_lock_acquire(uint32_t timeout_ms)
{
    if (s_dl_runtime_lock == NULL) {
        esp_err_t ret = helmet_dl_runtime_lock_init();
        if (ret != ESP_OK) {
            return ret;
        }
    }

    TickType_t ticks = pdMS_TO_TICKS(timeout_ms);
    if (timeout_ms > 0U && ticks == 0) {
        ticks = 1;
    }

    return xSemaphoreTake(s_dl_runtime_lock, ticks) == pdTRUE ? ESP_OK : ESP_ERR_TIMEOUT;
}

void helmet_dl_runtime_lock_release(void)
{
    if (s_dl_runtime_lock != NULL) {
        xSemaphoreGive(s_dl_runtime_lock);
    }
}
