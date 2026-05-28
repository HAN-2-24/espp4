#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "esp_err.h"
#include "driver/gpio.h"

#define HELMET_BOARD_NAME "ESP32-P4-Function-EV-Board"

/*
 * 第一阶段只声明外接传感器建议脚位。
 * 不碰 LCD、MIPI、Touch、Audio Codec、ESP32-C6 这些官方 BSP 已经管理的资源。
 *
 * 避开：
 * GPIO26: LCD PWM
 * GPIO27: LCD reset
 * GPIO37/GPIO38: U0TXD/U0RXD
 */

#define HELMET_I2C_PORT              0
#define HELMET_I2C_SDA_GPIO          GPIO_NUM_7
#define HELMET_I2C_SCL_GPIO          GPIO_NUM_8
#define HELMET_I2C_FREQ_HZ           400000

#define HELMET_GNSS_UART_PORT        1
#define HELMET_GNSS_UART_TX_GPIO     GPIO_NUM_5
#define HELMET_GNSS_UART_RX_GPIO     GPIO_NUM_6
#define HELMET_GNSS_UART_BAUDRATE    9600

#define HELMET_CANCEL_KEY_GPIO       GPIO_NUM_23

esp_err_t helmet_board_init(void);
int helmet_board_cancel_key_pressed(void);

#ifdef __cplusplus
}
#endif