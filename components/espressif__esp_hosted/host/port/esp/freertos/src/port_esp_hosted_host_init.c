/*
 * SPDX-FileCopyrightText: 2024-2025 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_hosted.h"

#include "port_esp_hosted_host_log.h"

#include "esp_private/startup_internal.h"

/* Reserve internal DMA memory pool for SDIO operations.
 * Declared here instead of including esp_private/esp_psram_extram.h
 * to avoid adding extra include path dependencies. */
esp_err_t esp_psram_extram_reserve_dma_pool(size_t size);

DEFINE_LOG_TAG(host_init);

//ESP_SYSTEM_INIT_FN(esp_hosted_host_init, BIT(0), 120)
static void __attribute__((constructor)) esp_hosted_host_init(void)
{
	ESP_LOGI(TAG, "ESP Hosted : Host chip_ip[%d]", CONFIG_IDF_FIRMWARE_CHIP_ID);

	/* Reserve internal DMA memory before PSRAM allocations consume it.
	 * 96KB: ~48KB for SDIO mempool + ~48KB for bounce buffer + audio/camera DMA */
	esp_psram_extram_reserve_dma_pool(96 * 1024);

	ESP_ERROR_CHECK(esp_hosted_init());
}

static void __attribute__((destructor)) esp_hosted_host_deinit(void)
{
	ESP_LOGI(TAG, "ESP Hosted deinit");
	esp_hosted_deinit();
}
