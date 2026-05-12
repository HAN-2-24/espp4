/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "bsp/config.h"
#include "bsp/display.h"
#include "lvgl.h"
#include "esp_lv_adapter_display.h"

#if (BSP_CONFIG_NO_GRAPHIC_LIB == 1)
typedef struct {
    bsp_display_config_t hw_cfg;    /*!< Display HW configuration */
} bsp_display_cfg_t;
#endif

lv_display_t *lvgl_adapter_init(const bsp_display_cfg_t *cfg);

#ifdef __cplusplus
}
#endif
