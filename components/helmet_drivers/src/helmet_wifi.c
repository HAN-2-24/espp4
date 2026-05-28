#include "helmet_wifi.h"

#include <string.h>
#include <stdio.h>

#include "esp_log.h"
#include "esp_err.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

/*
 * 这里先用宏写死，方便第一阶段调通。
 * 后续可以改成 menuconfig / NVS 保存 / 配网页面。
 */
#ifndef HELMET_WIFI_SSID
#define HELMET_WIFI_SSID      "ZTZTH6"
#endif

#ifndef HELMET_WIFI_PASSWORD
#define HELMET_WIFI_PASSWORD  "66666666"
#endif

#define HELMET_WIFI_CONNECTED_BIT  BIT0
#define HELMET_WIFI_FAIL_BIT       BIT1

static const char *TAG = "helmet_wifi";

static EventGroupHandle_t s_wifi_event_group = NULL;
static esp_netif_t *s_wifi_netif = NULL;

static bool s_wifi_inited = false;
static volatile bool s_wifi_connected = false;
static char s_ip_string[16] = "0.0.0.0";

static void helmet_wifi_event_handler(
    void *arg,
    esp_event_base_t event_base,
    int32_t event_id,
    void *event_data
)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "WiFi STA start, connecting...");
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        s_wifi_connected = false;
        snprintf(s_ip_string, sizeof(s_ip_string), "0.0.0.0");

        if (s_wifi_event_group != NULL) {
            xEventGroupClearBits(s_wifi_event_group, HELMET_WIFI_CONNECTED_BIT);
            xEventGroupSetBits(s_wifi_event_group, HELMET_WIFI_FAIL_BIT);
        }

        ESP_LOGW(TAG, "WiFi disconnected, reconnecting...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;

        s_wifi_connected = true;

        snprintf(
            s_ip_string,
            sizeof(s_ip_string),
            IPSTR,
            IP2STR(&event->ip_info.ip)
        );

        if (s_wifi_event_group != NULL) {
            xEventGroupClearBits(s_wifi_event_group, HELMET_WIFI_FAIL_BIT);
            xEventGroupSetBits(s_wifi_event_group, HELMET_WIFI_CONNECTED_BIT);
        }

        ESP_LOGI(TAG, "WiFi connected, IP=%s", s_ip_string);
    }
}

esp_err_t helmet_wifi_init(void)
{
    if (s_wifi_inited) {
        return ESP_OK;
    }

    ESP_LOGI(TAG, "WiFi init start");

    s_wifi_event_group = xEventGroupCreate();
    if (s_wifi_event_group == NULL) {
        ESP_LOGE(TAG, "create wifi event group failed");
        return ESP_FAIL;
    }

    esp_err_t ret;

    ret = esp_netif_init();
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "esp_netif_init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = esp_event_loop_create_default();
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "esp_event_loop_create_default failed: %s", esp_err_to_name(ret));
        return ret;
    }

    if (s_wifi_netif == NULL) {
        s_wifi_netif = esp_netif_create_default_wifi_sta();
        if (s_wifi_netif == NULL) {
            ESP_LOGE(TAG, "create default wifi sta failed");
            return ESP_FAIL;
        }
    }

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();

    ret = esp_wifi_init(&cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = esp_event_handler_instance_register(
        WIFI_EVENT,
        ESP_EVENT_ANY_ID,
        helmet_wifi_event_handler,
        NULL,
        NULL
    );
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "register WIFI_EVENT failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = esp_event_handler_instance_register(
        IP_EVENT,
        IP_EVENT_STA_GOT_IP,
        helmet_wifi_event_handler,
        NULL,
        NULL
    );
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "register IP_EVENT failed: %s", esp_err_to_name(ret));
        return ret;
    }

    wifi_config_t wifi_config;
    memset(&wifi_config, 0, sizeof(wifi_config));

    strncpy(
        (char *)wifi_config.sta.ssid,
        HELMET_WIFI_SSID,
        sizeof(wifi_config.sta.ssid) - 1
    );

    strncpy(
        (char *)wifi_config.sta.password,
        HELMET_WIFI_PASSWORD,
        sizeof(wifi_config.sta.password) - 1
    );

    ret = esp_wifi_set_mode(WIFI_MODE_STA);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_mode failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_config failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = esp_wifi_start();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_start failed: %s", esp_err_to_name(ret));
        return ret;
    }

    s_wifi_inited = true;

    ESP_LOGI(TAG, "WiFi init done, ssid=%s", HELMET_WIFI_SSID);

    return ESP_OK;
}

bool helmet_wifi_is_connected(void)
{
    return s_wifi_connected;
}

esp_err_t helmet_wifi_wait_connected(uint32_t timeout_ms)
{
    if (s_wifi_event_group == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_event_group,
        HELMET_WIFI_CONNECTED_BIT,
        pdFALSE,
        pdTRUE,
        pdMS_TO_TICKS(timeout_ms)
    );

    if (bits & HELMET_WIFI_CONNECTED_BIT) {
        return ESP_OK;
    }

    return ESP_ERR_TIMEOUT;
}

const char *helmet_wifi_get_ip_string(void)
{
    return s_ip_string;
}