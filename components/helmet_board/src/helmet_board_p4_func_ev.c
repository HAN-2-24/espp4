#include "helmet_board.h"

#include "esp_log.h"
#include "esp_err.h"
#include "driver/gpio.h"

/*
 * 这里用你工程里 BSP extra 实际声明的头文件。
 * 如果你当前工程没有公开头文件，也可以先用 extern 声明。
 */
// #include "bsp_board_extra.h"
extern esp_err_t bsp_extra_codec_init(void);

static const char *TAG = "helmet_board";

static bool s_bsp_extra_inited = false;

esp_err_t helmet_board_init(void)
{
    ESP_LOGI(TAG, "board init: %s", HELMET_BOARD_NAME);

    /*
     * 关键：
     * MPU6050 和 codec 共用 I2C0。
     * I2C0 由 BSP extra codec 初始化流程创建。
     *
     * 所以这里必须先调用 bsp_extra_codec_init()，
     * 后面的 helmet_imu_init() 才能通过 i2c_master_get_bus_handle()
     * 拿到已经存在的 I2C bus。
     *
     * 注意：这里仍然没有手动 i2c_new_master_bus()。
     */
    if (!s_bsp_extra_inited) {
        esp_err_t ret = bsp_extra_codec_init();
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "bsp_extra_codec_init failed: %s", esp_err_to_name(ret));
            return ret;
        }

        s_bsp_extra_inited = true;
        ESP_LOGI(TAG, "bsp_extra_codec_init done");
    }

    gpio_config_t key_cfg = {
        .pin_bit_mask = 1ULL << HELMET_CANCEL_KEY_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };

    esp_err_t ret = gpio_config(&key_cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "cancel key gpio config failed: %s", esp_err_to_name(ret));
        return ret;
    }

    return ESP_OK;
}

int helmet_board_cancel_key_pressed(void)
{
    return gpio_get_level(HELMET_CANCEL_KEY_GPIO) == 0;
}