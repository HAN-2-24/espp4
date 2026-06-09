#include "helmet_cloud.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sdkconfig.h"
#include "esp_event.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_hosted.h"
#include "mqtt_client.h"

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"

#include "helmet_state.h"
#include "helmet_voice.h"

static const char *TAG = "helmet_cloud";

#define HELMET_MQTT_URI                  CONFIG_HELMET_MQTT_URI
#define HELMET_MQTT_CLIENT_ID            CONFIG_HELMET_MQTT_CLIENT_ID
#define HELMET_MQTT_USERNAME             CONFIG_HELMET_MQTT_USERNAME
#define HELMET_MQTT_PASSWORD             CONFIG_HELMET_MQTT_PASSWORD
#define HELMET_MQTT_TOPIC_ATTR           CONFIG_HELMET_MQTT_TOPIC_ATTR
#define HELMET_MQTT_TOPIC_STATUS_QUERY   CONFIG_HELMET_MQTT_TOPIC_STATUS_QUERY
#define HELMET_MQTT_TOPIC_STATUS_PCM     CONFIG_HELMET_MQTT_TOPIC_STATUS_PCM

#define HELMET_CLOUD_REPORT_INTERVAL_MS   CONFIG_HELMET_CLOUD_REPORT_INTERVAL_MS
#define HELMET_CLOUD_TASK_STACK_SIZE      6144
#define HELMET_CLOUD_TASK_PRIORITY        4
#define HELMET_CLOUD_WIFI_TASK_STACK_SIZE 6144
#define HELMET_CLOUD_WIFI_TASK_PRIORITY   5
#define HELMET_CLOUD_WIFI_SSID            "ZTZTH6"
#define HELMET_CLOUD_WIFI_PASSWORD        "66666666"
#define HELMET_CLOUD_WIFI_RETRY_DELAY_MS  5000
#define HELMET_CLOUD_WIFI_CONNECT_WAIT_MS 15000
#define HELMET_CLOUD_WIFI_CONNECTED_BIT   BIT0
#define HELMET_CLOUD_WIFI_DISCONNECTED_BIT BIT1

#if CONFIG_HELMET_CLOUD_DEBUG_LOG
#define HELMET_CLOUD_DEBUG_LOG            1
#else
#define HELMET_CLOUD_DEBUG_LOG            0
#endif

#define HELMET_CLOUD_TOPIC_MAX_LEN        128
#define HELMET_CLOUD_STATUS_PAYLOAD_SIZE  2048
#define HELMET_CLOUD_STATE_PAYLOAD_SIZE   1536
#define HELMET_CLOUD_MQTT_BUFFER_SIZE     1024
#define HELMET_CLOUD_MQTT_OUT_BUFFER_SIZE 1024
#define HELMET_CLOUD_MQTT_OUTBOX_LIMIT    1024
#define HELMET_CLOUD_MQTT_TASK_STACK_SIZE 4096
#define HELMET_CLOUD_PCM_FLAG_FINAL       0x01U
#define HELMET_CLOUD_PCM_FLAG_ADPCM       0x02U
#define HELMET_CLOUD_STATUS_PCM_QOS       0
#define HELMET_CLOUD_STATUS_PCM_IDLE_TIMEOUT_MS 2500
#define HELMET_CLOUD_STATUS_PCM_RESPONSE_TIMEOUT_MS 15000
#define HELMET_CLOUD_ADPCM_DECODE_BYTES   256
#define HELMET_CLOUD_ADPCM_DECODE_SAMPLES (HELMET_CLOUD_ADPCM_DECODE_BYTES * 2)
#ifdef CONFIG_BSP_SPIFFS_MOUNT_POINT
#define HELMET_CLOUD_STATUS_PCM_FILE_PATH CONFIG_BSP_SPIFFS_MOUNT_POINT "/status_reply.pcm"
#else
#define HELMET_CLOUD_STATUS_PCM_FILE_PATH "/spiffs/status_reply.pcm"
#endif
#define HELMET_CLOUD_STATUS_PCM_FILE_IO_BYTES 512
#define HELMET_CLOUD_STATUS_PCM_MAX_BYTES (256U * 1024U)

#if CONFIG_HELMET_MQTT_USE_EMBEDDED_CA
extern const uint8_t emqx_ca_pem_start[] asm("_binary_emqx_ca_pem_start");
#endif

static esp_mqtt_client_handle_t s_mqtt_client = NULL;
static TaskHandle_t s_cloud_task_handle = NULL;
static TaskHandle_t s_cloud_wifi_task_handle = NULL;

static bool s_cloud_inited = false;
static bool s_cloud_wifi_started = false;
static bool s_cloud_wifi_netif_ready = false;
static bool s_cloud_wifi_hosted_started = false;
static bool s_cloud_wifi_driver_ready = false;
static bool s_cloud_wifi_handlers_registered = false;
static bool s_mqtt_started = false;
static bool s_mqtt_config_warned = false;
static volatile bool s_cloud_wifi_connected = false;
static volatile bool s_mqtt_connected = false;
static volatile uint32_t s_status_request_id = 0;
static volatile uint32_t s_status_pcm_expected_seq = 0;
static volatile bool s_status_request_queued = false;
static volatile bool s_status_request_pending = false;
static volatile bool s_status_pcm_stream_open = false;
static volatile bool s_status_pcm_play_queued = false;
static bool s_status_pcm_stream_adpcm = false;
static volatile TickType_t s_status_request_tick = 0;
static volatile TickType_t s_status_pcm_last_tick = 0;
static volatile uint32_t s_status_pcm_play_request_id = 0;
static char s_cloud_ip_string[16] = "0.0.0.0";
static char s_cloud_state_payload[HELMET_CLOUD_STATE_PAYLOAD_SIZE];
static char s_cloud_status_payload[HELMET_CLOUD_STATUS_PAYLOAD_SIZE];
static FILE *s_status_pcm_file = NULL;
static size_t s_status_pcm_file_bytes = 0;
static int16_t s_status_adpcm_decode_pcm[HELMET_CLOUD_ADPCM_DECODE_SAMPLES];
static uint8_t s_status_pcm_file_io_buf[HELMET_CLOUD_STATUS_PCM_FILE_IO_BYTES];
static int s_status_adpcm_predictor = 0;
static int s_status_adpcm_index = 0;
static EventGroupHandle_t s_cloud_wifi_event_group = NULL;
static esp_event_handler_instance_t s_cloud_wifi_any_id_handler;
static esp_event_handler_instance_t s_cloud_wifi_got_ip_handler;

static esp_err_t helmet_cloud_publish_status_pcm_request(void);

static const char *json_bool(bool value)
{
    return value ? "true" : "false";
}

static const char *helmet_cloud_optional_config(const char *value)
{
    return (value != NULL && value[0] != '\0') ? value : NULL;
}

static bool helmet_cloud_mqtt_is_configured(void)
{
    return HELMET_MQTT_URI[0] != '\0';
}

static void helmet_cloud_log_unconfigured_once(void)
{
    if (!s_mqtt_config_warned) {
        ESP_LOGW(TAG, "MQTT broker URI is not configured. Set CONFIG_HELMET_MQTT_URI.");
        s_mqtt_config_warned = true;
    }
}

static void helmet_cloud_wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    (void)arg;

    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        snprintf(s_cloud_ip_string, sizeof(s_cloud_ip_string), IPSTR, IP2STR(&event->ip_info.ip));
        s_cloud_wifi_connected = true;
        if (s_cloud_wifi_event_group != NULL) {
            xEventGroupClearBits(s_cloud_wifi_event_group, HELMET_CLOUD_WIFI_DISCONNECTED_BIT);
            xEventGroupSetBits(s_cloud_wifi_event_group, HELMET_CLOUD_WIFI_CONNECTED_BIT);
        }
        ESP_LOGI(TAG, "fixed WiFi got IP: %s", s_cloud_ip_string);
        return;
    }

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        s_cloud_wifi_connected = false;
        snprintf(s_cloud_ip_string, sizeof(s_cloud_ip_string), "0.0.0.0");
        if (s_cloud_wifi_event_group != NULL) {
            xEventGroupClearBits(s_cloud_wifi_event_group, HELMET_CLOUD_WIFI_CONNECTED_BIT);
            xEventGroupSetBits(s_cloud_wifi_event_group, HELMET_CLOUD_WIFI_DISCONNECTED_BIT);
        }
        ESP_LOGW(TAG, "fixed WiFi disconnected");
    }
}

static bool helmet_cloud_wifi_is_connected(void)
{
    if (!s_cloud_wifi_connected) {
        snprintf(s_cloud_ip_string, sizeof(s_cloud_ip_string), "0.0.0.0");
        return false;
    }

    return true;
}

static esp_err_t helmet_cloud_wifi_register_handlers_once(void)
{
    if (s_cloud_wifi_handlers_registered) {
        return ESP_OK;
    }

    esp_err_t ret = esp_event_handler_instance_register(
        WIFI_EVENT,
        ESP_EVENT_ANY_ID,
        helmet_cloud_wifi_event_handler,
        NULL,
        &s_cloud_wifi_any_id_handler
    );
    if (ret != ESP_OK) {
        return ret;
    }

    ret = esp_event_handler_instance_register(
        IP_EVENT,
        IP_EVENT_STA_GOT_IP,
        helmet_cloud_wifi_event_handler,
        NULL,
        &s_cloud_wifi_got_ip_handler
    );
    if (ret != ESP_OK) {
        return ret;
    }

    s_cloud_wifi_handlers_registered = true;
    return ESP_OK;
}

static esp_err_t helmet_cloud_wifi_prepare_netif(void)
{
    if (s_cloud_wifi_netif_ready) {
        return ESP_OK;
    }

    esp_err_t ret = esp_netif_init();
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        return ret;
    }

    ret = esp_event_loop_create_default();
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        return ret;
    }

    esp_netif_t *sta_netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    if (sta_netif == NULL) {
        sta_netif = esp_netif_create_default_wifi_sta();
    }
    if (sta_netif == NULL) {
        return ESP_FAIL;
    }

    s_cloud_wifi_netif_ready = true;
    return ESP_OK;
}

static esp_err_t helmet_cloud_wifi_prepare_driver(void)
{
    esp_err_t ret = helmet_cloud_wifi_prepare_netif();
    if (ret != ESP_OK) {
        return ret;
    }

    if (!s_cloud_wifi_hosted_started) {
        ret = (esp_err_t)esp_hosted_init();
        if (ret != ESP_OK) {
            return ret;
        }
        s_cloud_wifi_hosted_started = true;
        ESP_LOGI(TAG, "ESP-Hosted init done for fixed WiFi");
    }

    ret = (esp_err_t)esp_hosted_connect_to_slave();
    if (ret != ESP_OK) {
        return ret;
    }

    if (!s_cloud_wifi_driver_ready) {
        wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
        ret = esp_wifi_init(&cfg);
        if (ret != ESP_OK && ret != ESP_ERR_WIFI_INIT_STATE) {
            return ret;
        }

        ret = helmet_cloud_wifi_register_handlers_once();
        if (ret != ESP_OK) {
            return ret;
        }

        s_cloud_wifi_driver_ready = true;
    }

    return ESP_OK;
}

static esp_err_t helmet_cloud_wifi_connect_fixed(void)
{
    esp_err_t ret = helmet_cloud_wifi_prepare_driver();
    if (ret != ESP_OK) {
        return ret;
    }

    wifi_config_t wifi_config = {0};
    snprintf((char *)wifi_config.sta.ssid, sizeof(wifi_config.sta.ssid), "%s", HELMET_CLOUD_WIFI_SSID);
    snprintf((char *)wifi_config.sta.password, sizeof(wifi_config.sta.password), "%s", HELMET_CLOUD_WIFI_PASSWORD);

    ret = esp_wifi_set_mode(WIFI_MODE_STA);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = esp_wifi_start();
    if (ret != ESP_OK && ret != ESP_ERR_WIFI_CONN && ret != ESP_ERR_WIFI_STATE) {
        return ret;
    }

    ret = esp_wifi_connect();
    if (ret != ESP_OK && ret != ESP_ERR_WIFI_CONN && ret != ESP_ERR_WIFI_STATE) {
        return ret;
    }

    return ESP_OK;
}

static void helmet_cloud_wifi_task(void *arg)
{
    (void)arg;

    ESP_LOGI(TAG, "fixed WiFi task started ssid=%s", HELMET_CLOUD_WIFI_SSID);

    while (1) {
        if (!s_cloud_wifi_connected) {
            esp_err_t ret = helmet_cloud_wifi_connect_fixed();
            if (ret != ESP_OK) {
                ESP_LOGW(TAG, "fixed WiFi setup waiting: %s", esp_err_to_name(ret));
                vTaskDelay(pdMS_TO_TICKS(HELMET_CLOUD_WIFI_RETRY_DELAY_MS));
                continue;
            }

            EventBits_t bits = xEventGroupWaitBits(
                s_cloud_wifi_event_group,
                HELMET_CLOUD_WIFI_CONNECTED_BIT,
                pdFALSE,
                pdFALSE,
                pdMS_TO_TICKS(HELMET_CLOUD_WIFI_CONNECT_WAIT_MS)
            );
            if (!(bits & HELMET_CLOUD_WIFI_CONNECTED_BIT)) {
                ESP_LOGW(TAG, "fixed WiFi connection timeout");
                esp_wifi_disconnect();
            }
        } else if (s_cloud_wifi_event_group != NULL) {
            xEventGroupWaitBits(
                s_cloud_wifi_event_group,
                HELMET_CLOUD_WIFI_DISCONNECTED_BIT,
                pdTRUE,
                pdFALSE,
                pdMS_TO_TICKS(HELMET_CLOUD_WIFI_RETRY_DELAY_MS)
            );
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

static esp_err_t helmet_cloud_wifi_start_once(void)
{
    if (s_cloud_wifi_started) {
        return ESP_OK;
    }

    if (s_cloud_wifi_event_group == NULL) {
        s_cloud_wifi_event_group = xEventGroupCreate();
        if (s_cloud_wifi_event_group == NULL) {
            return ESP_ERR_NO_MEM;
        }
    }

    BaseType_t task_ret = xTaskCreate(
        helmet_cloud_wifi_task,
        "helmet_wifi",
        HELMET_CLOUD_WIFI_TASK_STACK_SIZE,
        NULL,
        HELMET_CLOUD_WIFI_TASK_PRIORITY,
        &s_cloud_wifi_task_handle
    );
    if (task_ret != pdPASS) {
        return ESP_FAIL;
    }

    s_cloud_wifi_started = true;
    return ESP_OK;
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
            "\"alarm_suppressed\":%s,"
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
        json_bool(state.alarm_suppressed),

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
        json_bool(helmet_cloud_wifi_is_connected()),
        s_cloud_ip_string,

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
    if (!helmet_cloud_mqtt_is_configured()) {
        helmet_cloud_log_unconfigured_once();
        helmet_state_set_cloud(HELMET_CLOUD_OFFLINE);
        return ESP_ERR_INVALID_STATE;
    }

    if (!helmet_cloud_wifi_is_connected()) {
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

    esp_err_t ret = helmet_cloud_build_payload(s_cloud_state_payload, sizeof(s_cloud_state_payload));
    if (ret != ESP_OK) {
        return ret;
    }

    int payload_len = (int)strlen(s_cloud_state_payload);
    int msg_id = esp_mqtt_client_publish(
        s_mqtt_client,
        HELMET_MQTT_TOPIC_ATTR,
        s_cloud_state_payload,
        payload_len,
        0,
        0
    );

#if HELMET_CLOUD_DEBUG_LOG
    ESP_LOGI(
        TAG,
        "MQTT publish msg_id=%d topic=%s bytes=%d",
        msg_id,
        HELMET_MQTT_TOPIC_ATTR,
        payload_len
    );
#endif

    if (msg_id < 0) {
        return ESP_FAIL;
    }

    return ESP_OK;
}

static bool helmet_cloud_topic_to_cstr(
    esp_mqtt_event_handle_t event,
    char *topic,
    size_t topic_size
)
{
    if (event == NULL || topic == NULL || topic_size == 0) {
        return false;
    }

    if (event->topic_len <= 0 || event->topic_len >= (int)topic_size) {
        return false;
    }

    memcpy(topic, event->topic, event->topic_len);
    topic[event->topic_len] = '\0';

    return true;
}

static bool helmet_cloud_parse_pcm_topic(
    const char *topic,
    uint32_t *request_id,
    uint32_t *seq,
    uint32_t *flags
)
{
    const char *prefix = HELMET_MQTT_TOPIC_STATUS_PCM "/";
    size_t prefix_len = strlen(prefix);

    if (topic == NULL || request_id == NULL || seq == NULL || flags == NULL) {
        return false;
    }

    if (strncmp(topic, prefix, prefix_len) != 0) {
        return false;
    }

    const char *cursor = topic + prefix_len;
    char *end = NULL;

    unsigned long parsed_request = strtoul(cursor, &end, 10);
    if (end == cursor || *end != '/') {
        return false;
    }

    cursor = end + 1;
    unsigned long parsed_seq = strtoul(cursor, &end, 10);
    if (end == cursor || *end != '/') {
        return false;
    }

    cursor = end + 1;
    unsigned long parsed_flags = strtoul(cursor, &end, 10);
    if (end == cursor || *end != '\0') {
        return false;
    }

    *request_id = (uint32_t)parsed_request;
    *seq = (uint32_t)parsed_seq;
    *flags = (uint32_t)parsed_flags;

    return true;
}

static void helmet_cloud_status_file_close(void)
{
    if (s_status_pcm_file != NULL) {
        fclose(s_status_pcm_file);
        s_status_pcm_file = NULL;
    }
}

static esp_err_t helmet_cloud_status_file_open(uint32_t request_id)
{
    helmet_cloud_status_file_close();
    remove(HELMET_CLOUD_STATUS_PCM_FILE_PATH);

    s_status_pcm_file = fopen(HELMET_CLOUD_STATUS_PCM_FILE_PATH, "wb");
    s_status_pcm_file_bytes = 0;

    if (s_status_pcm_file == NULL) {
        ESP_LOGW(TAG, "status pcm file open failed request=%lu path=%s",
                 (unsigned long)request_id, HELMET_CLOUD_STATUS_PCM_FILE_PATH);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "status pcm file begin request=%lu path=%s",
             (unsigned long)request_id, HELMET_CLOUD_STATUS_PCM_FILE_PATH);

    return ESP_OK;
}

static esp_err_t helmet_cloud_status_file_write(const void *data, size_t bytes)
{
    if (bytes == 0) {
        return ESP_OK;
    }

    if (data == NULL || s_status_pcm_file == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    if (s_status_pcm_file_bytes + bytes > HELMET_CLOUD_STATUS_PCM_MAX_BYTES) {
        ESP_LOGW(TAG, "status pcm file too large: %u + %u",
                 (unsigned)s_status_pcm_file_bytes, (unsigned)bytes);
        return ESP_ERR_NO_MEM;
    }

    size_t written = fwrite(data, 1, bytes, s_status_pcm_file);
    if (written != bytes) {
        ESP_LOGW(TAG, "status pcm file write failed: %u/%u",
                 (unsigned)written, (unsigned)bytes);
        return ESP_FAIL;
    }

    s_status_pcm_file_bytes += written;

    return ESP_OK;
}

static esp_err_t helmet_cloud_play_status_pcm_file(uint32_t request_id)
{
    FILE *file = fopen(HELMET_CLOUD_STATUS_PCM_FILE_PATH, "rb");
    if (file == NULL) {
        ESP_LOGW(TAG, "status pcm playback open failed request=%lu path=%s",
                 (unsigned long)request_id, HELMET_CLOUD_STATUS_PCM_FILE_PATH);
        return ESP_FAIL;
    }

    esp_err_t ret = helmet_voice_stream_pcm_begin();
    if (ret != ESP_OK) {
        fclose(file);
        ESP_LOGW(TAG, "status pcm playback begin failed request=%lu ret=%s",
                 (unsigned long)request_id, esp_err_to_name(ret));
        return ret;
    }

    size_t total = 0;

    while (true) {
        size_t bytes_read = fread(
            s_status_pcm_file_io_buf,
            1,
            sizeof(s_status_pcm_file_io_buf),
            file
        );

        if (bytes_read == 0) {
            if (!feof(file)) {
                ret = ESP_FAIL;
            }
            break;
        }

        if ((bytes_read & 1U) != 0U) {
            bytes_read--;
        }

        if (bytes_read == 0) {
            continue;
        }

        ret = helmet_voice_stream_pcm_write(s_status_pcm_file_io_buf, bytes_read);
        if (ret != ESP_OK) {
            break;
        }

        total += bytes_read;
    }

    esp_err_t end_ret = helmet_voice_stream_pcm_end();
    fclose(file);
    remove(HELMET_CLOUD_STATUS_PCM_FILE_PATH);

    if (ret == ESP_OK) {
        ret = end_ret;
    }

    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "status pcm playback done request=%lu bytes=%u",
                 (unsigned long)request_id, (unsigned)total);
    } else {
        ESP_LOGW(TAG, "status pcm playback failed request=%lu ret=%s bytes=%u",
                 (unsigned long)request_id, esp_err_to_name(ret), (unsigned)total);
    }

    return ret;
}

static void helmet_cloud_service_status_pcm_playback(void)
{
    if (!s_status_pcm_play_queued) {
        return;
    }

    uint32_t request_id = s_status_pcm_play_request_id;
    s_status_pcm_play_queued = false;
    s_status_pcm_play_request_id = 0;

    (void)helmet_cloud_play_status_pcm_file(request_id);
}

static void helmet_cloud_adpcm_reset(void)
{
    s_status_adpcm_predictor = 0;
    s_status_adpcm_index = 0;
}

static int16_t helmet_cloud_adpcm_decode_nibble(uint8_t nibble)
{
    static const int index_table[16] = {
        -1, -1, -1, -1, 2, 4, 6, 8,
        -1, -1, -1, -1, 2, 4, 6, 8,
    };
    static const int step_table[89] = {
        7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
        19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
        50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
        130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
        337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
        876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
        2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
        5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
        15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
    };

    int step = step_table[s_status_adpcm_index];
    int diff = step >> 3;

    if ((nibble & 0x01U) != 0U) {
        diff += step >> 2;
    }
    if ((nibble & 0x02U) != 0U) {
        diff += step >> 1;
    }
    if ((nibble & 0x04U) != 0U) {
        diff += step;
    }

    if ((nibble & 0x08U) != 0U) {
        s_status_adpcm_predictor -= diff;
    } else {
        s_status_adpcm_predictor += diff;
    }

    if (s_status_adpcm_predictor > 32767) {
        s_status_adpcm_predictor = 32767;
    } else if (s_status_adpcm_predictor < -32768) {
        s_status_adpcm_predictor = -32768;
    }

    s_status_adpcm_index += index_table[nibble & 0x0FU];
    if (s_status_adpcm_index < 0) {
        s_status_adpcm_index = 0;
    } else if (s_status_adpcm_index > 88) {
        s_status_adpcm_index = 88;
    }

    return (int16_t)s_status_adpcm_predictor;
}

static esp_err_t helmet_cloud_write_status_adpcm(const uint8_t *data, size_t bytes)
{
    if (data == NULL || bytes == 0) {
        return ESP_OK;
    }

    size_t offset = 0;
    while (offset < bytes) {
        size_t block = bytes - offset;
        if (block > HELMET_CLOUD_ADPCM_DECODE_BYTES) {
            block = HELMET_CLOUD_ADPCM_DECODE_BYTES;
        }

        for (size_t i = 0; i < block; i++) {
            uint8_t packed = data[offset + i];
            s_status_adpcm_decode_pcm[i * 2] = helmet_cloud_adpcm_decode_nibble(packed & 0x0FU);
            s_status_adpcm_decode_pcm[i * 2 + 1] = helmet_cloud_adpcm_decode_nibble((packed >> 4) & 0x0FU);
        }

        esp_err_t ret = helmet_cloud_status_file_write(
            s_status_adpcm_decode_pcm,
            block * 2U * sizeof(int16_t)
        );

        if (ret != ESP_OK) {
            return ret;
        }

        offset += block;
    }

    return ESP_OK;
}

static void helmet_cloud_close_status_stream(void)
{
    if (s_status_pcm_stream_open) {
        helmet_cloud_status_file_close();
        s_status_pcm_stream_open = false;
        s_status_pcm_stream_adpcm = false;
        helmet_cloud_adpcm_reset();
    }
}

static void helmet_cloud_service_status_pcm_timeout(void)
{
    if (!s_status_request_pending) {
        return;
    }

    if (!s_status_pcm_stream_open) {
        if (s_status_request_tick == 0) {
            return;
        }

        TickType_t elapsed = xTaskGetTickCount() - s_status_request_tick;
        if (elapsed < pdMS_TO_TICKS(HELMET_CLOUD_STATUS_PCM_RESPONSE_TIMEOUT_MS)) {
            return;
        }

        ESP_LOGW(TAG, "status pcm response timeout request=%lu", (unsigned long)s_status_request_id);
        s_status_request_pending = false;
        s_status_request_tick = 0;
        s_status_pcm_last_tick = 0;
        return;
    }

    if (s_status_pcm_last_tick == 0) {
        return;
    }

    TickType_t elapsed = xTaskGetTickCount() - s_status_pcm_last_tick;
    if (elapsed < pdMS_TO_TICKS(HELMET_CLOUD_STATUS_PCM_IDLE_TIMEOUT_MS)) {
        return;
    }

    ESP_LOGW(TAG, "status pcm idle timeout, closing stream request=%lu", (unsigned long)s_status_request_id);
    helmet_cloud_close_status_stream();
    s_status_request_pending = false;
    s_status_request_tick = 0;
    s_status_pcm_last_tick = 0;
}

static void helmet_cloud_handle_status_pcm(esp_mqtt_event_handle_t event)
{
    char topic[HELMET_CLOUD_TOPIC_MAX_LEN];
    uint32_t request_id = 0;
    uint32_t seq = 0;
    uint32_t flags = 0;
    bool packet_adpcm = false;

    if (!helmet_cloud_topic_to_cstr(event, topic, sizeof(topic))) {
        return;
    }

    if (!helmet_cloud_parse_pcm_topic(topic, &request_id, &seq, &flags)) {
        return;
    }

    if (!s_status_request_pending || request_id != s_status_request_id) {
        ESP_LOGW(
            TAG,
            "drop status pcm: request=%lu active=%lu pending=%d",
            (unsigned long)request_id,
            (unsigned long)s_status_request_id,
            s_status_request_pending ? 1 : 0
        );
        return;
    }

    if (event->current_data_offset != 0 || event->data_len != event->total_data_len) {
        ESP_LOGW(TAG, "drop fragmented pcm chunk: topic=%s", topic);
        return;
    }

    packet_adpcm = (flags & HELMET_CLOUD_PCM_FLAG_ADPCM) != 0U;

    if (!s_status_pcm_stream_open) {
        esp_err_t ret = helmet_cloud_status_file_open(request_id);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "status pcm file begin failed: %s", esp_err_to_name(ret));
            s_status_request_pending = false;
            s_status_request_tick = 0;
            s_status_pcm_last_tick = 0;
            return;
        }

        s_status_pcm_stream_open = true;
        s_status_pcm_stream_adpcm = packet_adpcm;
        if (s_status_pcm_stream_adpcm) {
            helmet_cloud_adpcm_reset();
        }
        s_status_pcm_expected_seq = seq;
    } else if (packet_adpcm != s_status_pcm_stream_adpcm) {
        ESP_LOGW(TAG, "status pcm codec flag changed, closing stream");
        helmet_cloud_close_status_stream();
        s_status_request_pending = false;
        s_status_request_tick = 0;
        s_status_pcm_last_tick = 0;
        return;
    }

    s_status_pcm_last_tick = xTaskGetTickCount();

    if (seq < s_status_pcm_expected_seq) {
        return;
    }

    if (seq != s_status_pcm_expected_seq) {
        ESP_LOGW(
            TAG,
            "status pcm seq mismatch: got=%lu expected=%lu",
            (unsigned long)seq,
            (unsigned long)s_status_pcm_expected_seq
        );
        s_status_pcm_expected_seq = seq;
    }

    if (event->data_len > 0) {
        esp_err_t ret = s_status_pcm_stream_adpcm ?
                        helmet_cloud_write_status_adpcm((const uint8_t *)event->data, (size_t)event->data_len) :
                        helmet_cloud_status_file_write(event->data, (size_t)event->data_len);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "status pcm file write failed: %s", esp_err_to_name(ret));
            helmet_cloud_close_status_stream();
            s_status_request_pending = false;
            s_status_request_tick = 0;
            s_status_pcm_last_tick = 0;
            return;
        }
    }

    s_status_pcm_expected_seq++;

    if ((flags & HELMET_CLOUD_PCM_FLAG_FINAL) != 0U) {
        size_t file_bytes = s_status_pcm_file_bytes;
        helmet_cloud_close_status_stream();
        s_status_request_pending = false;
        s_status_request_tick = 0;
        s_status_pcm_last_tick = 0;
        if (file_bytes > 0) {
            s_status_pcm_play_request_id = request_id;
            s_status_pcm_play_queued = true;
            if (s_cloud_task_handle != NULL) {
                xTaskNotifyGive(s_cloud_task_handle);
            }
        } else {
            remove(HELMET_CLOUD_STATUS_PCM_FILE_PATH);
        }
        ESP_LOGI(TAG, "status pcm file complete request=%lu bytes=%u",
                 (unsigned long)request_id, (unsigned)file_bytes);
    }
}

static void helmet_mqtt_event_handler(
    void *handler_args,
    esp_event_base_t base,
    int32_t event_id,
    void *event_data
)
{
    (void)handler_args;
    (void)base;

    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;

    switch ((esp_mqtt_event_id_t)event_id) {
    case MQTT_EVENT_CONNECTED:
        s_mqtt_connected = true;
        helmet_state_set_cloud(HELMET_CLOUD_ONLINE);
        ESP_LOGI(TAG, "MQTT connected");
        esp_mqtt_client_subscribe(
            s_mqtt_client,
            HELMET_MQTT_TOPIC_STATUS_PCM "/#",
            HELMET_CLOUD_STATUS_PCM_QOS
        );
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

    case MQTT_EVENT_DATA:
        helmet_cloud_handle_status_pcm(event);
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

    if (!helmet_cloud_mqtt_is_configured()) {
        helmet_cloud_log_unconfigured_once();
        helmet_state_set_cloud(HELMET_CLOUD_OFFLINE);
        return ESP_ERR_INVALID_STATE;
    }

    if (!helmet_cloud_wifi_is_connected()) {
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
            .client_id = helmet_cloud_optional_config(HELMET_MQTT_CLIENT_ID),
            .username = helmet_cloud_optional_config(HELMET_MQTT_USERNAME),
            .authentication = {
                .password = helmet_cloud_optional_config(HELMET_MQTT_PASSWORD),
            },
        },
        .session = {
            .keepalive = 60,
        },
        .task = {
            .stack_size = HELMET_CLOUD_MQTT_TASK_STACK_SIZE,
        },
        .buffer = {
            .size = HELMET_CLOUD_MQTT_BUFFER_SIZE,
            .out_size = HELMET_CLOUD_MQTT_OUT_BUFFER_SIZE,
        },
        .outbox = {
            .limit = HELMET_CLOUD_MQTT_OUTBOX_LIMIT,
        },
    };

#if CONFIG_HELMET_MQTT_USE_EMBEDDED_CA
    mqtt_cfg.broker.verification.certificate = (const char *)emqx_ca_pem_start;
#endif

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

    ESP_LOGI(TAG, "MQTT client started uri=%s topic=%s", HELMET_MQTT_URI, HELMET_MQTT_TOPIC_ATTR);

    return ESP_OK;
}

static void helmet_cloud_task(void *arg)
{
    (void)arg;

    ESP_LOGI(TAG, "cloud task started");

    while (1) {
        helmet_cloud_service_status_pcm_playback();
        helmet_cloud_service_status_pcm_timeout();

        if (!helmet_cloud_mqtt_is_configured()) {
            helmet_cloud_log_unconfigured_once();
            helmet_state_set_cloud(HELMET_CLOUD_OFFLINE);
            vTaskDelay(pdMS_TO_TICKS(5000));
            continue;
        }

        if (!helmet_cloud_wifi_is_connected()) {
            helmet_state_set_cloud(HELMET_CLOUD_OFFLINE);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        if (!s_mqtt_started) {
            helmet_cloud_start_mqtt();
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        if (s_status_request_queued) {
            s_status_request_queued = false;

            esp_err_t request_ret = helmet_cloud_publish_status_pcm_request();
            if (request_ret != ESP_OK) {
                ESP_LOGW(TAG, "status pcm request failed: %s", esp_err_to_name(request_ret));
            }
        } else if (s_status_request_pending || s_status_pcm_stream_open || s_status_pcm_play_queued) {
            /* Status downlink shares ESP-Hosted with telemetry; keep the link quiet while audio is arriving. */
        } else {
            helmet_cloud_publish_state_once();
        }

        uint32_t wait_ms = s_status_request_pending ?
                           250U :
                           (uint32_t)HELMET_CLOUD_REPORT_INTERVAL_MS;
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(wait_ms));
    }
}

esp_err_t helmet_cloud_init(void)
{
    if (s_cloud_inited) {
        return ESP_OK;
    }

    esp_err_t netif_ret = esp_netif_init();
    if (netif_ret != ESP_OK && netif_ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "esp_netif_init failed: %s", esp_err_to_name(netif_ret));
        helmet_state_set_cloud(HELMET_CLOUD_ERROR);
        return netif_ret;
    }

    esp_err_t wifi_ret = helmet_cloud_wifi_start_once();
    if (wifi_ret != ESP_OK) {
        ESP_LOGE(TAG, "fixed WiFi task start failed: %s", esp_err_to_name(wifi_ret));
        helmet_state_set_cloud(HELMET_CLOUD_ERROR);
        return wifi_ret;
    }

    helmet_state_set_cloud(HELMET_CLOUD_OFFLINE);
    if (!helmet_cloud_mqtt_is_configured()) {
        helmet_cloud_log_unconfigured_once();
    }

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

static esp_err_t helmet_cloud_publish_status_pcm_request(void)
{
    if (!helmet_cloud_mqtt_is_configured()) {
        helmet_cloud_log_unconfigured_once();
        helmet_state_set_cloud(HELMET_CLOUD_OFFLINE);
        return ESP_ERR_INVALID_STATE;
    }

    if (!helmet_cloud_wifi_is_connected()) {
        helmet_state_set_cloud(HELMET_CLOUD_OFFLINE);
        ESP_LOGW(TAG, "status pcm request skipped: WiFi offline");
        return ESP_ERR_INVALID_STATE;
    }

    if (s_mqtt_client == NULL || !s_mqtt_started || !s_mqtt_connected) {
        ESP_LOGW(TAG, "status pcm request skipped: MQTT offline");
        return ESP_ERR_INVALID_STATE;
    }

    helmet_cloud_close_status_stream();

    s_status_request_id++;
    if (s_status_request_id == 0) {
        s_status_request_id = 1;
    }

    s_status_request_pending = true;
    s_status_pcm_expected_seq = 0;
    s_status_request_tick = xTaskGetTickCount();
    s_status_pcm_last_tick = 0;

    int prefix_len = snprintf(
        s_cloud_status_payload,
        sizeof(s_cloud_status_payload),
        "{"
            "\"type\":\"helmet_status_query\","
            "\"request_id\":%lu,"
            "\"audio\":{"
                "\"format\":\"pcm_s16le\","
                "\"sample_rate\":16000,"
                "\"channels\":1"
            "},"
            "\"state\":",
        (unsigned long)s_status_request_id
    );

    if (prefix_len <= 0 || prefix_len >= ((int)sizeof(s_cloud_status_payload) - 2)) {
        s_status_request_pending = false;
        ESP_LOGE(TAG, "status pcm request payload too long");
        return ESP_ERR_INVALID_SIZE;
    }

    size_t state_capacity = sizeof(s_cloud_status_payload) - (size_t)prefix_len - 1;
    esp_err_t ret = helmet_cloud_build_payload(s_cloud_status_payload + prefix_len, state_capacity);
    if (ret != ESP_OK) {
        s_status_request_pending = false;
        return ret;
    }

    size_t state_len = strlen(s_cloud_status_payload + prefix_len);
    size_t len = (size_t)prefix_len + state_len;
    if (len + 1 >= sizeof(s_cloud_status_payload)) {
        s_status_request_pending = false;
        ESP_LOGE(TAG, "status pcm request payload too long");
        return ESP_ERR_INVALID_SIZE;
    }

    s_cloud_status_payload[len++] = '}';
    s_cloud_status_payload[len] = '\0';

    int msg_id = esp_mqtt_client_publish(
        s_mqtt_client,
        HELMET_MQTT_TOPIC_STATUS_QUERY,
        s_cloud_status_payload,
        (int)len,
        0,
        0
    );

    if (msg_id < 0) {
        s_status_request_pending = false;
        return ESP_FAIL;
    }

    ESP_LOGI(
        TAG,
        "status pcm request msg_id=%d request=%lu topic=%s",
        msg_id,
        (unsigned long)s_status_request_id,
        HELMET_MQTT_TOPIC_STATUS_QUERY
    );

    return ESP_OK;
}

esp_err_t helmet_cloud_request_status_pcm(void)
{
    if (!helmet_cloud_mqtt_is_configured()) {
        helmet_cloud_log_unconfigured_once();
        helmet_state_set_cloud(HELMET_CLOUD_OFFLINE);
        return ESP_ERR_INVALID_STATE;
    }

    if (!helmet_cloud_wifi_is_connected()) {
        helmet_state_set_cloud(HELMET_CLOUD_OFFLINE);
        ESP_LOGW(TAG, "status pcm request skipped: WiFi offline");
        return ESP_ERR_INVALID_STATE;
    }

    if (s_mqtt_client == NULL || !s_mqtt_started || !s_mqtt_connected) {
        ESP_LOGW(TAG, "status pcm request skipped: MQTT offline");
        return ESP_ERR_INVALID_STATE;
    }

    s_status_request_queued = true;

    if (s_cloud_task_handle != NULL) {
        xTaskNotifyGive(s_cloud_task_handle);
    }

    ESP_LOGI(TAG, "status pcm request queued");

    return ESP_OK;
}
