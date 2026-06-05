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

#define ESP_HOSTED_INTERNAL_DMA_RESERVE_TARGET_BYTES (64 * 1024)
#define ESP_HOSTED_INTERNAL_DMA_STARTUP_MARGIN_BYTES (48 * 1024)

static size_t esp_hosted_get_dma_reserve_size(void)
{
	size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
	size_t largest_dma = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
	size_t reserve = ESP_HOSTED_INTERNAL_DMA_RESERVE_TARGET_BYTES;

	if (free_internal <= ESP_HOSTED_INTERNAL_DMA_STARTUP_MARGIN_BYTES ||
		largest_dma <= ESP_HOSTED_INTERNAL_DMA_STARTUP_MARGIN_BYTES) {
		return 0;
	}

	if (reserve > free_internal - ESP_HOSTED_INTERNAL_DMA_STARTUP_MARGIN_BYTES) {
		reserve = free_internal - ESP_HOSTED_INTERNAL_DMA_STARTUP_MARGIN_BYTES;
	}

	if (reserve > largest_dma - ESP_HOSTED_INTERNAL_DMA_STARTUP_MARGIN_BYTES) {
		reserve = largest_dma - ESP_HOSTED_INTERNAL_DMA_STARTUP_MARGIN_BYTES;
	}

	return reserve;
}

//ESP_SYSTEM_INIT_FN(esp_hosted_host_init, BIT(0), 120)
static void __attribute__((constructor)) esp_hosted_host_init(void)
{
	ESP_LOGI(TAG, "ESP Hosted : Host chip_ip[%d]", CONFIG_IDF_FIRMWARE_CHIP_ID);

	/* Reserve internal DMA memory before PSRAM allocations consume it.
	 * Keep this below the old 96KB value so FreeRTOS startup can still create
	 * its timer task on memory-constrained builds. */
	size_t reserve_size = esp_hosted_get_dma_reserve_size();
	size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
	size_t largest_dma = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
	ESP_LOGI(TAG, "Internal DMA reserve request=%uK target=%uK margin=%uK free=%uK largest_dma=%uK",
		(unsigned)(reserve_size / 1024),
		(unsigned)(ESP_HOSTED_INTERNAL_DMA_RESERVE_TARGET_BYTES / 1024),
		(unsigned)(ESP_HOSTED_INTERNAL_DMA_STARTUP_MARGIN_BYTES / 1024),
		(unsigned)(free_internal / 1024),
		(unsigned)(largest_dma / 1024));
	if (reserve_size > 0) {
		esp_err_t reserve_ret = esp_psram_extram_reserve_dma_pool(reserve_size);
		if (reserve_ret != ESP_OK) {
			ESP_LOGW(TAG, "Internal DMA reserve failed: %s", esp_err_to_name(reserve_ret));
		}
	} else {
		ESP_LOGW(TAG, "Skip internal DMA reserve; not enough startup heap margin");
	}

	ESP_ERROR_CHECK(esp_hosted_init());
}

static void __attribute__((destructor)) esp_hosted_host_deinit(void)
{
	ESP_LOGI(TAG, "ESP Hosted deinit");
	esp_hosted_deinit();
}
