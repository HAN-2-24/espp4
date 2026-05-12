/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "lvgl_adapter_init.h"

#include "esp_check.h"
#include "esp_log.h"
#include "esp_lv_adapter.h"
#include "bsp/display.h"
#include "bsp/touch.h"

static const char *TAG = "lvgl_adapter_init";

#define LVGL_ADAPTER_BUFFER_HEIGHT 20

static void lvgl_adapter_get_resolution(uint32_t *out_hres, uint32_t *out_vres)
{
    if (out_hres) {
        *out_hres = BSP_LCD_H_RES;
    }
    if (out_vres) {
        *out_vres = BSP_LCD_V_RES;
    }
}

lv_display_t *lvgl_adapter_init(const bsp_display_cfg_t *cfg)
{
    if (cfg == NULL) {
        ESP_LOGE(TAG, "Display config is NULL");
        return NULL;
    }

    bsp_lcd_handles_t handles = { 0 };
    esp_err_t err = bsp_display_new_with_handles(&cfg->hw_cfg, &handles);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "BSP display init failed (%d)", err);
        return NULL;
    }

    const esp_lv_adapter_config_t adapter_cfg = ESP_LV_ADAPTER_DEFAULT_CONFIG();
    err = esp_lv_adapter_init(&adapter_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "LVGL adapter init failed (%d)", err);
        return NULL;
    }

    uint32_t hres = 0;
    uint32_t vres = 0;
    lvgl_adapter_get_resolution(&hres, &vres);

    esp_lv_adapter_display_config_t disp_cfg = ESP_LV_ADAPTER_DISPLAY_MIPI_DEFAULT_CONFIG(
        handles.panel, handles.io, hres, vres, ESP_LV_ADAPTER_ROTATE_0);

    disp_cfg.profile.buffer_height = LVGL_ADAPTER_BUFFER_HEIGHT;
    lv_display_t *disp = esp_lv_adapter_register_display(&disp_cfg);
    if (disp == NULL) {
        ESP_LOGE(TAG, "Register display failed");
        return NULL;
    }

    esp_lcd_touch_handle_t touch = NULL;
    err = bsp_touch_new(NULL, &touch);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Touch init failed (%d)", err);
        return NULL;
    }

    const esp_lv_adapter_touch_config_t touch_cfg = ESP_LV_ADAPTER_TOUCH_DEFAULT_CONFIG(disp, touch);
    lv_indev_t *indev = esp_lv_adapter_register_touch(&touch_cfg);
    if (indev == NULL) {
        ESP_LOGE(TAG, "Register touch failed");
        return NULL;
    }

    err = esp_lv_adapter_start();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "LVGL adapter start failed (%d)", err);
        return NULL;
    }

    return disp;
}
