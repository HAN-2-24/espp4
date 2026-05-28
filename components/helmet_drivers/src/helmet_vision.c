#include "helmet_vision.h"

#include "esp_log.h"
#include "helmet_state.h"

static const char *TAG = "helmet_vision";

esp_err_t helmet_vision_init(void)
{
    /*
     * 第一阶段不在这里初始化摄像头。
     * MIPI CSI 应复用官方 camera/bsp 初始化链路。
     */
    ESP_LOGI(TAG, "vision interface initialized");
    return ESP_OK;
}

esp_err_t helmet_vision_poll_once(void)
{
    /*
     * 后续实现:
     * 1. 从 MIPI CSI 摄像头取帧
     * 2. 眼部检测
     * 3. PERCLOS / blink / yawn
     * 4. helmet_state_set_eye(...)
     */
    return ESP_OK;
}
