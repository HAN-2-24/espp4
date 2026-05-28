#include "helmet_gnss.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "esp_log.h"
#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "helmet_board.h"
#include "helmet_state.h"

static const char *TAG = "helmet_gnss";

#define GNSS_UART_RX_BUF_SIZE     1024
#define GNSS_TASK_STACK_SIZE      4096
#define GNSS_TASK_PRIORITY        5
#define GNSS_READ_BUF_SIZE        128
#define GNSS_LINE_BUF_SIZE        160

#define GNSS_RAW_ECHO             1
#define GNSS_RMC_OK_LOG           1
#define GNSS_CMD_LOG              1

static TaskHandle_t s_gnss_task_handle = NULL;
static int s_gnss_uart_inited = 0;

static double nmea_raw_to_decimal(double raw, const char *dir)
{
    int degree = (int)(raw / 100.0);
    double minute = raw - degree * 100.0;
    double decimal = degree + minute / 60.0;

    if (dir != NULL && (dir[0] == 'S' || dir[0] == 'W')) {
        decimal = -decimal;
    }

    return decimal;
}

static esp_err_t helmet_gnss_send_cmd(const char *cmd)
{
    if (cmd == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

#if GNSS_CMD_LOG
    ESP_LOGI(TAG, "GNSS TX CMD: %s", cmd);
#endif

    int len = strlen(cmd);
    int written = uart_write_bytes(HELMET_GNSS_UART_PORT, cmd, len);

    if (written != len) {
        ESP_LOGE(TAG, "GNSS cmd write failed: written=%d expected=%d", written, len);
        return ESP_FAIL;
    }

    return ESP_OK;
}

static void helmet_gnss_config_module(void)
{
    /*
     * 使用老工程思路：
     * 先关闭无用输出，再打开 RMC。
     *
     * 注意：
     * 这一步要求 GNSS RX 接到 ESP32-P4 GPIO5。
     * 如果只接 GNSS TX -> GPIO6，则只能接收，不能配置模块。
     */

    vTaskDelay(pdMS_TO_TICKS(100));

    const char *cmd_close_all = "$CCRMO,GGA,3,1*3F\r\n";
    helmet_gnss_send_cmd(cmd_close_all);

    vTaskDelay(pdMS_TO_TICKS(100));

    const char *cmd_open_rmc = "$CCRMO,RMC,2,1*23\r\n";
    helmet_gnss_send_cmd(cmd_open_rmc);

    vTaskDelay(pdMS_TO_TICKS(100));
}

int helmet_gnss_parse_rmc(const char *sentence, helmet_gnss_data_t *out)
{
    if (sentence == NULL || out == NULL) {
        return 0;
    }

    memset(out, 0, sizeof(*out));

    char type[8] = {0};
    char time_str[16] = {0};
    char status = 0;
    double latitude_raw = 0.0;
    char lat_dir[2] = {0};
    double longitude_raw = 0.0;
    char lon_dir[2] = {0};
    float speed = 0.0f;
    float course = 0.0f;

    int matched = sscanf(
        sentence,
        "$%7[^,],%15[^,],%c,%lf,%1s,%lf,%1s,%f,%f",
        type,
        time_str,
        &status,
        &latitude_raw,
        lat_dir,
        &longitude_raw,
        lon_dir,
        &speed,
        &course
    );

    if (matched < 9) {
        return 0;
    }

    if (strcmp(type, "GNRMC") != 0 && strcmp(type, "GPRMC") != 0) {
        return 0;
    }

    if (status != 'A') {
        return 0;
    }

    if (lat_dir[0] == '\0' || lon_dir[0] == '\0') {
        return 0;
    }

    out->valid = 1;
    out->latitude = nmea_raw_to_decimal(latitude_raw, lat_dir);
    out->longitude = nmea_raw_to_decimal(longitude_raw, lon_dir);
    out->speed = speed;
    out->course = course;

    return 1;
}

static void helmet_gnss_process_line(const char *line)
{
    if (line == NULL || line[0] == '\0') {
        return;
    }

    helmet_gnss_data_t data = {0};

    if (helmet_gnss_parse_rmc(line, &data)) {
        helmet_state_set_location(
            true,
            data.latitude,
            data.longitude,
            data.speed,
            data.course
        );

#if GNSS_RMC_OK_LOG
        ESP_LOGI(
            TAG,
            "RMC OK lat=%.6f lon=%.6f speed=%.2f course=%.2f",
            data.latitude,
            data.longitude,
            data.speed,
            data.course
        );
#endif
    }
}

static void helmet_gnss_task(void *arg)
{
    uint8_t rx_buf[GNSS_READ_BUF_SIZE];
    char line[GNSS_LINE_BUF_SIZE];
    int pos = 0;

    ESP_LOGI(TAG, "GNSS task started");

    while (1) {
        int len = uart_read_bytes(
            HELMET_GNSS_UART_PORT,
            rx_buf,
            sizeof(rx_buf),
            pdMS_TO_TICKS(200)
        );

        if (len <= 0) {
            continue;
        }

#if GNSS_RAW_ECHO
        printf("%.*s", len, (char *)rx_buf);
        fflush(stdout);
#endif

        for (int i = 0; i < len; i++) {
            uint8_t ch = rx_buf[i];

            if (ch == '\n') {
                line[pos] = '\0';

                helmet_gnss_process_line(line);

                pos = 0;
            } else if (ch != '\r') {
                if (pos < GNSS_LINE_BUF_SIZE - 1) {
                    line[pos++] = (char)ch;
                } else {
                    pos = 0;
                }
            }
        }
    }
}

esp_err_t helmet_gnss_init(void)
{
    if (s_gnss_uart_inited) {
        return ESP_OK;
    }

    uart_config_t cfg = {
        .baud_rate = HELMET_GNSS_UART_BAUDRATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    esp_err_t ret;

    ret = uart_driver_install(
        HELMET_GNSS_UART_PORT,
        GNSS_UART_RX_BUF_SIZE,
        0,
        0,
        NULL,
        0
    );
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "uart_driver_install failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = uart_param_config(HELMET_GNSS_UART_PORT, &cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "uart_param_config failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = uart_set_pin(
        HELMET_GNSS_UART_PORT,
        HELMET_GNSS_UART_TX_GPIO,
        HELMET_GNSS_UART_RX_GPIO,
        UART_PIN_NO_CHANGE,
        UART_PIN_NO_CHANGE
    );
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "uart_set_pin failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /*
     * 配置 GNSS 模块输出：
     * 1. 关闭无用语句；
     * 2. 打开 RMC。
     *
     * 硬件要求：
     * ESP32-P4 GPIO5 -> GNSS RX
     * ESP32-P4 GPIO6 <- GNSS TX
     */
    helmet_gnss_config_module();

    s_gnss_uart_inited = 1;

    if (s_gnss_task_handle == NULL) {
        BaseType_t task_ret = xTaskCreate(
            helmet_gnss_task,
            "helmet_gnss",
            GNSS_TASK_STACK_SIZE,
            NULL,
            GNSS_TASK_PRIORITY,
            &s_gnss_task_handle
        );

        if (task_ret != pdPASS) {
            ESP_LOGE(TAG, "create GNSS task failed");
            s_gnss_task_handle = NULL;
            return ESP_FAIL;
        }
    }

    ESP_LOGI(
        TAG,
        "GNSS UART initialized: uart=%d tx=%d rx=%d baud=%d",
        HELMET_GNSS_UART_PORT,
        HELMET_GNSS_UART_TX_GPIO,
        HELMET_GNSS_UART_RX_GPIO,
        HELMET_GNSS_UART_BAUDRATE
    );

    return ESP_OK;
}

esp_err_t helmet_gnss_poll_once(void)
{
    /*
     * 保留接口，避免影响 HelmetDashboard.cpp 编译。
     * GNSS 已经由 helmet_gnss_task 后台持续采集。
     */
    return ESP_OK;
}