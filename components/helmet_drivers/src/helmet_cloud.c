#include "helmet_cloud.h"

#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "esp_log.h"
#include "esp_err.h"
#include "mqtt_client.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "helmet_state.h"
#include "helmet_wifi.h"

static const char *TAG = "helmet_cloud";

/*
 * MQTT 参数
 * 先按你旧 ESP32-01S.ino 的云平台改这里。
 */
#ifndef HELMET_MQTT_URI
#define HELMET_MQTT_URI        "mqtt://sh-3-mqtt.iot-api.com:1883"
#endif

#ifndef HELMET_MQTT_USERNAME
#define HELMET_MQTT_USERNAME   "填你的MQTT用户名"
#endif

#ifndef HELMET_MQTT_PASSWORD
#define HELMET_MQTT_PASSWORD   "填你的MQTT密码"
#endif

#ifndef HELMET_MQTT_TOPIC_ATTR
#define HELMET_MQTT_TOPIC_ATTR "attributes"
#endif

#define HELMET_CLOUD_REPORT_INTERVAL_MS   30000
#define HELMET_CLOUD_TASK_STACK_SIZE      8192
#define HELMET_CLOUD_TASK_PRIORITY        4

#define HELMET_CLOUD_DEBUG_LOG            1

static esp_mqtt_client_handle_t s_mqtt_client = NULL;
static TaskHandle_t s_cloud_task_handle = NULL;

static bool s_cloud_inited = false;
static bool s_mqtt_started = false;
static volatile bool s_mqtt_connected = false;

static const char *json_bool(bool value)
{
    return value ? "true" : "false";
}

static void json_escape(char *dst, size_t dst_size, const char *src)
{
    if (dst == NULL || dst_size == 0) {
        return;
    }

    if (src == NULL) {
        dst[0] = '\0';
        return;
    }

    size_t di = 0;

    for (size_t si = 0; src[si] != '\0' && di < dst_size - 1; si++) {
        char c = src[si];

        if (c == '"' || c == '\\') {
            if (di + 2 >= dst_size) {
                break;
            }
            dst[di++] = '\\';
            dst[di++] = c;
        } else if ((unsigned char)c < 32) {
            dst[di++] = ' ';
        } else {
            dst[di++] = c;
        }
    }

    dst[di] = '\0';
}

static int helmet_cloud_risk_to_warning(helmet_risk_level_t risk)
{
    switch (risk) {
    case HELMET_RISK_NORMAL:
        return 0;

    case HELMET_RISK_ATTENTION:
    case HELMET_RISK_FATIGUE:
        return 1;

    case HELMET_RISK_DANGER:
        return 2;

    default:
        return 0;
    }
}

static int helmet_cloud_calc_fatigue_index(const helmet_state_t *state)
{
    if (state == NULL || !state->eye.valid) {
        return 0;
    }

    int index = (int)(state->eye.perclos * 100.0f);

    if (index < 0) {
        index = 0;
    }

    if (index > 100) {
        index = 100;
    }

    return index;
}

static esp_err_t helmet_cloud_build_payload(char *payload, size_t payload_size)
{
    if (payload == NULL || payload_size == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    helmet_state_t state = helmet_state_get_copy();

    char explanation[192];
    char report[192];

    json_escape(explanation, sizeof(explanation), state.explanation);
    json_escape(report, sizeof(report), state.report);

    int warning = helmet_cloud_risk_to_warning(state.risk);
    int fatigue_index = helmet_cloud_calc_fatigue_index(&state);

    /*
     * 上传内容包含：
     * 1. 地图 location.lat / location.lng
     * 2. 仪表盘 Risk / Alarm
     * 3. Eye Vision
     * 4. Pose / IMU
     * 5. Voice
     * 6. Cloud
     * 7. Explanation / Report
     */
    int len = snprintf(
        payload,
        payload_size,
        "{"
            "\"location\":{"
                "\"valid\":%s,"
                "\"lat\":%.6f,"
                "\"lng\":%.6f,"
                "\"speed\":%.2f,"
                "\"course\":%.2f"
            "},"
            "\"risk\":\"%s\","
            "\"warning\":%d,"
            "\"alarm_active\":%s,"
            "\"eye\":{"
                "\"valid\":%s,"
                "\"eye_open_ratio\":%.2f,"
                "\"perclos\":%.2f,"
                "\"blink_count\":%lu,"
                "\"yawn_detected\":%s"
            "},"
            "\"pose\":{"
                "\"valid\":%s,"
                "\"roll\":%.2f,"
                "\"pitch\":%.2f,"
                "\"acc_total\":%.2f,"
                "\"fall_detected\":%s,"
                "\"impact_detected\":%s"
            "},"
            "\"voice\":{"
                "\"state\":\"%s\","
                "\"enabled\":%s"
            "},"
            "\"cloud\":{"
                "\"state\":\"%s\","
                "\"enabled\":%s,"
                "\"wifi_connected\":%s,"
                "\"ip\":\"%s\""
            "},"
            "\"fatigue_index\":%d,"
            "\"explanation\":\"%s\","
            "\"report\":\"%s\""
        "}",
        json_bool(state.location.valid),
        state.location.latitude,
        state.location.longitude,
        state.location.speed,
        state.location.course,

        helmet_risk_to_string(state.risk),
        warning,
        json_bool(state.alarm_active),

        json_bool(state.eye.valid),
        state.eye.eye_open_ratio,
        state.eye.perclos,
        (unsigned long)state.eye.blink_count,
        json_bool(state.eye.yawn_detected),

        json_bool(state.pose.valid),
        state.pose.roll,
        state.pose.pitch,
        state.pose.acc_total,
        json_bool(state.pose.fall_detected),
        json_bool(state.pose.impact_detected),

        helmet_voice_to_string(state.voice),
        json_bool(state.voice_enabled),

        helmet_cloud_to_string(state.cloud),
        json_bool(state.cloud_enabled),
        json_bool(helmet_wifi_is_connected()),
        helmet_wifi_get_ip_string(),

        fatigue_index,
        explanation,
        report
    );

    if (len <= 0 || len >= (int)payload_size) {
        ESP_LOGE(TAG, "MQTT payload too long");
        return ESP_ERR_INVALID_SIZE;
    }

    return ESP_OK;
}

static esp_err_t helmet_cloud_publish_state_once(void)
{
    if (!helmet_wifi_is_connected()) {
        helmet_state_set_cloud(HELMET_CLOUD_OFFLINE);

#if HELMET_CLOUD_DEBUG_LOG
        ESP_LOGW(TAG, "WiFi not connected, skip MQTT publish");
#endif

        return ESP_ERR_INVALID_STATE;
    }

    if (s_mqtt_client == NULL || !s_mqtt_started || !s_mqtt_connected) {
#if HELMET_CLOUD_DEBUG_LOG
        ESP_LOGW(TAG, "MQTT not connected, skip publish");
#endif

        return ESP_ERR_INVALID_STATE;
    }

    char payload[1536];

    esp_err_t ret = helmet_cloud_build_payload(payload, sizeof(payload));
    if (ret != ESP_OK) {
        return ret;
    }

    int msg_id = esp_mqtt_client_publish(
        s_mqtt_client,
        HELMET_MQTT_TOPIC_ATTR,
        payload,
        0,
        0,
        0
    );

#if HELMET_CLOUD_DEBUG_LOG
    ESP_LOGI(
        TAG,
        "MQTT publish msg_id=%d topic=%s payload=%s",
        msg_id,
        HELMET_MQTT_TOPIC_ATTR,
        payload
    );
#endif

    if (msg_id < 0) {
        return ESP_FAIL;
    }

    return ESP_OK;
}

static void helmet_mqtt_event_handler(
    void *handler_args,
    esp_event_base_t base,
    int32_t event_id,
    void *event_data
)
{
    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;

    switch ((esp_mqtt_event_id_t)event_id) {
    case MQTT_EVENT_CONNECTED:
        s_mqtt_connected = true;
        helmet_state_set_cloud(HELMET_CLOUD_ONLINE);
        ESP_LOGI(TAG, "MQTT connected");
        break;

    case MQTT_EVENT_DISCONNECTED:
        s_mqtt_connected = false;
        helmet_state_set_cloud(HELMET_CLOUD_OFFLINE);
        ESP_LOGW(TAG, "MQTT disconnected");
        break;

    case MQTT_EVENT_ERROR:
        s_mqtt_connected = false;
        helmet_state_set_cloud(HELMET_CLOUD_ERROR);
        ESP_LOGE(TAG, "MQTT error");
        break;

    case MQTT_EVENT_PUBLISHED:
#if HELMET_CLOUD_DEBUG_LOG
        ESP_LOGI(TAG, "MQTT published msg_id=%d", event->msg_id);
#endif
        break;

    default:
        break;
    }
}

static esp_err_t helmet_cloud_start_mqtt(void)
{
    if (s_mqtt_started) {
        return ESP_OK;
    }

    if (!helmet_wifi_is_connected()) {
        return ESP_ERR_INVALID_STATE;
    }

    helmet_state_set_cloud(HELMET_CLOUD_CONNECTING);

    esp_mqtt_client_config_t mqtt_cfg = {
        .broker = {
            .address = {
                .uri = HELMET_MQTT_URI,
            },
        },
        .credentials = {
            .username = HELMET_MQTT_USERNAME,
            .authentication = {
                .password = HELMET_MQTT_PASSWORD,
            },
        },
        .session = {
            .keepalive = 60,
        },
    };

    s_mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    if (s_mqtt_client == NULL) {
        ESP_LOGE(TAG, "esp_mqtt_client_init failed");
        helmet_state_set_cloud(HELMET_CLOUD_ERROR);
        return ESP_FAIL;
    }

    esp_err_t ret = esp_mqtt_client_register_event(
        s_mqtt_client,
        ESP_EVENT_ANY_ID,
        helmet_mqtt_event_handler,
        NULL
    );
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "mqtt register event failed: %s", esp_err_to_name(ret));
        helmet_state_set_cloud(HELMET_CLOUD_ERROR);
        return ret;
    }

    ret = esp_mqtt_client_start(s_mqtt_client);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "mqtt start failed: %s", esp_err_to_name(ret));
        helmet_state_set_cloud(HELMET_CLOUD_ERROR);
        return ret;
    }

    s_mqtt_started = true;

    ESP_LOGI(TAG, "MQTT client started");

    return ESP_OK;
}

static void helmet_cloud_task(void *arg)
{
    ESP_LOGI(TAG, "cloud task started");

    while (1) {
        if (!helmet_wifi_is_connected()) {
            helmet_state_set_cloud(HELMET_CLOUD_OFFLINE);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        if (!s_mqtt_started) {
            helmet_cloud_start_mqtt();
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        helmet_cloud_publish_state_once();

        vTaskDelay(pdMS_TO_TICKS(HELMET_CLOUD_REPORT_INTERVAL_MS));
    }
}

esp_err_t helmet_cloud_init(void)
{
    if (s_cloud_inited) {
        return ESP_OK;
    }

    helmet_state_set_cloud(HELMET_CLOUD_CONNECTING);

    if (s_cloud_task_handle == NULL) {
        BaseType_t task_ret = xTaskCreate(
            helmet_cloud_task,
            "helmet_cloud",
            HELMET_CLOUD_TASK_STACK_SIZE,
            NULL,
            HELMET_CLOUD_TASK_PRIORITY,
            &s_cloud_task_handle
        );

        if (task_ret != pdPASS) {
            ESP_LOGE(TAG, "create cloud task failed");
            helmet_state_set_cloud(HELMET_CLOUD_ERROR);
            return ESP_FAIL;
        }
    }

    s_cloud_inited = true;

    ESP_LOGI(TAG, "cloud init done");

    return ESP_OK;
}

esp_err_t helmet_cloud_update_once(void)
{
    return helmet_cloud_publish_state_once();
}