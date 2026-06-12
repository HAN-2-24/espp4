/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_check.h"
#include "esp_memory_utils.h"
#include "esp_heap_caps.h"
#include "lvgl.h"
#include "bsp/esp-bsp.h"
#include "bsp/display.h"
#include "bsp_board_extra.h"
#include "esp_lv_adapter.h"
#include "lvgl_adapter_init.h"
#include "esp_ldo_regulator.h"
#include "esp_brookesia.hpp"
#include "app_examples/phone/squareline/src/phone_app_squareline.hpp"
#include "apps.h"

static const char *TAG = "main";

static esp_ldo_channel_handle_t sd_ldo_handle = NULL;

extern "C" void p4_tcm_heap_reserve_linker_hook(void);

static void heap_alloc_failed_hook(size_t size, uint32_t caps, const char *function_name)
{
    ESP_EARLY_LOGE(
        "heap_fail",
        "size=%u caps=0x%lx func=%s int_free=%u int_largest=%u dma_free=%u dma_largest=%u psram_free=%u psram_largest=%u",
        (unsigned)size,
        (unsigned long)caps,
        function_name ? function_name : "?",
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
        (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA),
        (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA),
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
        (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM)
    );
}

static esp_err_t init_sd_ldo_only(void)
{
    esp_ldo_channel_config_t ldo_cfg = {
        .chan_id = 4,
        .voltage_mv = 3300,
    };
    return esp_ldo_acquire_channel(&ldo_cfg, &sd_ldo_handle);
}

extern "C" void app_main(void)
{
    p4_tcm_heap_reserve_linker_hook();
    ESP_ERROR_CHECK(heap_caps_register_failed_alloc_callback(heap_alloc_failed_hook));

    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    ESP_ERROR_CHECK(bsp_spiffs_mount());
    ESP_LOGI(TAG, "SPIFFS mount successfully");

#if CONFIG_EXAMPLE_ENABLE_SD_CARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
    ESP_LOGI(TAG, "SD card mount successfully");
#else
    ESP_ERROR_CHECK(init_sd_ldo_only());
#endif

    ESP_ERROR_CHECK(bsp_extra_codec_init());

    bsp_display_cfg_t cfg = {
        .hw_cfg = {
            .hdmi_resolution = BSP_HDMI_RES_NONE,
            .dsi_bus = {
                .lane_bit_rate_mbps = BSP_LCD_MIPI_DSI_LANE_BITRATE_MBPS,
            },
        },
    };
    lv_display_t *disp = lvgl_adapter_init(&cfg);
    assert(disp != nullptr && "Failed to init LVGL adapter");
    bsp_display_backlight_on();

    ESP_ERROR_CHECK(esp_lv_adapter_lock(-1));

    ESP_Brookesia_Phone *phone = new ESP_Brookesia_Phone();
    assert(phone != nullptr && "Failed to create phone");

    ESP_Brookesia_PhoneStylesheet_t *phone_stylesheet = new ESP_Brookesia_PhoneStylesheet_t ESP_BROOKESIA_PHONE_1024_600_DARK_STYLESHEET();
    ESP_BROOKESIA_CHECK_NULL_EXIT(phone_stylesheet, "Create phone stylesheet failed");
    ESP_BROOKESIA_CHECK_FALSE_EXIT(phone->addStylesheet(*phone_stylesheet), "Add phone stylesheet failed");
    ESP_BROOKESIA_CHECK_FALSE_EXIT(phone->activateStylesheet(*phone_stylesheet), "Activate phone stylesheet failed");

    assert(phone->begin() && "Failed to begin phone");

    PhoneAppSquareline *smart_gadget = new PhoneAppSquareline();
    assert(smart_gadget != nullptr && "Failed to create phone app squareline");
    assert((phone->installApp(smart_gadget) >= 0) && "Failed to install phone app squareline");



    HelmetDashboard *helmet_dashboard = new HelmetDashboard();
    assert(helmet_dashboard != nullptr && "Failed to create helmet_dashboard");
    assert((phone->installApp(helmet_dashboard) >= 0) && "Failed to install helmet_dashboard");

    Camera *camera = new Camera(1280, 720);
    assert(camera != nullptr && "Failed to create camera");
    assert((phone->installApp(camera) >= 0) && "Failed to begin camera");



#if CONFIG_EXAMPLE_ENABLE_SD_CARD
    ESP_LOGW(TAG, "Using Video Player example requires inserting the SD card in advance and saving an MJPEG format video on the SD card");
    AppVideoPlayer *app_video_player = new AppVideoPlayer();
    assert(app_video_player != nullptr && "Failed to create app_video_player");
    assert((phone->installApp(app_video_player) >= 0) && "Failed to begin app_video_player");
#endif

    esp_lv_adapter_unlock();
}
