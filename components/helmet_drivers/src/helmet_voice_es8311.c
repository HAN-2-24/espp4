#include "helmet_voice.h"

#include "helmet_state.h"
#include "helmet_voice_cmd.h"

#include "sdkconfig.h"
#include "esp_err.h"
#include "esp_log.h"

#include "bsp/esp32_p4_function_ev_board.h"
#include "driver/gpio.h"
#include "esp_codec_dev.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "esp_afe_config.h"
#include "esp_afe_sr_iface.h"
#include "esp_mn_iface.h"
#include "esp_mn_models.h"
#include "model_path.h"

/*
 * Some esp-sr package versions used by this project do not expose this symbol
 * in the public headers even though the library provides it.
 */
extern const esp_afe_sr_iface_t *esp_afe_handle_from_config(afe_config_t *afe_config);

#include <stdbool.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const char *TAG = "helmet_voice";

static bool s_voice_inited = false;
static TaskHandle_t s_voice_capture_task_handle = NULL;
static TaskHandle_t s_voice_sr_init_task_handle = NULL;
static TaskHandle_t s_voice_alert_task_handle = NULL;
static SemaphoreHandle_t s_voice_output_mutex = NULL;
static SemaphoreHandle_t s_voice_alert_signal = NULL;

static esp_codec_dev_handle_t s_spk_codec = NULL;
static esp_codec_dev_handle_t s_mic_codec = NULL;

#define HELMET_AUDIO_SAMPLE_RATE          16000
#define HELMET_AUDIO_BITS                 16
#define HELMET_AUDIO_CHANNELS             1
#define HELMET_AUDIO_VOLUME               100
#define HELMET_MIC_GAIN                   42.0f

#define VOICE_CAPTURE_FRAMES              512
#define VOICE_CAPTURE_INTERVAL_MS         20

#define VOICE_MN_TIMEOUT_MS               6000
#define VOICE_MN_DET_THRESHOLD            0.35f
#define VOICE_CMD_COOLDOWN_MS             1500

#define VOICE_LOG_EVERY_N_FRAMES          500
#define VOICE_AFE_VAD_ENABLED             0
#define VOICE_SR_INIT_TASK_STACK_SIZE      8192
#define VOICE_SR_INIT_TASK_PRIORITY        3
#define VOICE_ALERT_TASK_STACK_SIZE        3072
#define VOICE_ALERT_TASK_PRIORITY          5
#define VOICE_ALERT_COUNT                  3
#define VOICE_FATIGUE_REPEAT_MS            30000
#define VOICE_DANGER_REPEAT_MS             10000
#define VOICE_DEFAULT_REPEAT_MS            30000
#if CONFIG_FREERTOS_NUMBER_OF_CORES > 1
#define VOICE_SR_INIT_TASK_CORE            1
#define VOICE_ALERT_TASK_CORE              1
#else
#define VOICE_SR_INIT_TASK_CORE            0
#define VOICE_ALERT_TASK_CORE              0
#endif

#define VOICE_ACCIDENT_PCM_PATH           BSP_SPIFFS_MOUNT_POINT "/music/accident.pcm"
#define VOICE_TIRED_PCM_PATH              BSP_SPIFFS_MOUNT_POINT "/music/tired.pcm"
#define VOICE_ALERT_NONE                  (-1)
#define VOICE_MN_CANCEL_ID                HELMET_VOICE_CMD_CANCEL_ALARM
#define VOICE_MN_STATUS_ID                HELMET_VOICE_CMD_REPORT_STATUS

static int16_t s_capture_pcm[VOICE_CAPTURE_FRAMES];

#define TEST_BEEP_FRAMES                  512
#define TEST_BEEP_AMPLITUDE               16000
static int16_t s_beep[TEST_BEEP_FRAMES];

typedef enum {
    VOICE_RUNTIME_IDLE = 0,
    VOICE_RUNTIME_LISTENING,
    VOICE_RUNTIME_PLAYING,
    VOICE_RUNTIME_ERROR,
} voice_runtime_state_t;

static volatile voice_runtime_state_t s_voice_runtime = VOICE_RUNTIME_IDLE;
static volatile bool s_voice_output_busy = false;
static volatile bool s_voice_stop_requested = false;
static volatile int s_pending_alert = VOICE_ALERT_NONE;
static volatile int s_current_alert = VOICE_ALERT_NONE;
static volatile bool s_output_prepared = false;
static uint32_t s_last_alert_ms[VOICE_ALERT_COUNT];

static const esp_afe_sr_iface_t *s_afe_handle = NULL;
static esp_afe_sr_data_t *s_afe_data = NULL;
static int s_afe_feed_chunksize = 0;
static int s_afe_feed_channel_num = 0;

static srmodel_list_t *s_sr_models = NULL;
static esp_mn_iface_t *s_mn_handle = NULL;
static model_iface_data_t *s_mn_data = NULL;
static int s_mn_chunksize = 0;
static bool s_mn_ready = false;
static volatile uint32_t s_mn_detecting_count = 0;
static volatile uint32_t s_mn_detected_count = 0;
static volatile uint32_t s_mn_timeout_count = 0;
static volatile uint32_t s_mn_frame_mismatch_count = 0;
static volatile int s_mn_last_afe_frames = 0;
static volatile bool s_voice_streaming = false;

static void make_test_beep(void)
{
    for (int i = 0; i < TEST_BEEP_FRAMES; i++) {
        int16_t v = ((i / 8) % 2 == 0) ? TEST_BEEP_AMPLITUDE : -TEST_BEEP_AMPLITUDE;
        s_beep[i] = v;
    }
}

static esp_err_t voice_prepare_output(void)
{
    if (s_spk_codec == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    gpio_set_direction(BSP_POWER_AMP_IO, GPIO_MODE_OUTPUT);
    gpio_set_level(BSP_POWER_AMP_IO, 1);

    int ret = esp_codec_dev_set_out_mute(s_spk_codec, false);
    if (ret != ESP_CODEC_DEV_OK) {
        ESP_LOGW(TAG, "codec unmute failed: %s", esp_err_to_name(ret));
    }

    if (s_output_prepared) {
        return ESP_OK;
    }

    ret = esp_codec_dev_set_out_vol(s_spk_codec, HELMET_AUDIO_VOLUME);
    if (ret != ESP_CODEC_DEV_OK) {
        ESP_LOGW(TAG, "codec volume failed: %s", esp_err_to_name(ret));
    }

    s_output_prepared = true;

    return ESP_OK;
}

static esp_err_t voice_write_pcm_bytes(const void *pcm, size_t bytes)
{
    if (s_spk_codec == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    if (pcm == NULL || bytes == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    if ((bytes & 1U) != 0U) {
        bytes--;
    }

    if (bytes == 0) {
        return ESP_OK;
    }

    size_t done = 0;
    while (done < bytes) {
        size_t chunk = bytes - done;
        if (chunk > sizeof(s_beep)) {
            chunk = sizeof(s_beep);
        }

        if (pcm != (const void *)s_beep) {
            memcpy(s_beep, (const uint8_t *)pcm + done, chunk);
        }

        int ret = esp_codec_dev_write(s_spk_codec, s_beep, (int)chunk);
        if (ret != ESP_CODEC_DEV_OK) {
            return (esp_err_t)ret;
        }

        done += chunk;
    }

    return ESP_OK;
}

static bool voice_output_lock(TickType_t ticks_to_wait)
{
    if (s_voice_output_mutex == NULL) {
        return true;
    }

    return xSemaphoreTake(s_voice_output_mutex, ticks_to_wait) == pdTRUE;
}

static void voice_output_unlock(void)
{
    if (s_voice_output_mutex != NULL) {
        xSemaphoreGive(s_voice_output_mutex);
    }
}

static void helmet_multinet_clean(void)
{
    if (s_mn_ready && s_mn_handle != NULL && s_mn_data != NULL && s_mn_handle->clean != NULL) {
        s_mn_handle->clean(s_mn_data);
    }
}

static esp_err_t helmet_multinet_init(srmodel_list_t *models)
{
    if (models == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    char *mn_name = esp_srmodel_filter(models, ESP_MN_PREFIX, ESP_MN_CHINESE);
    if (mn_name == NULL) {
        ESP_LOGW(TAG, "MultiNet model not found; fixed commands disabled");
        return ESP_OK;
    }

    s_mn_handle = esp_mn_handle_from_name(mn_name);
    if (s_mn_handle == NULL) {
        ESP_LOGW(TAG, "esp_mn_handle_from_name failed: %s", mn_name);
        return ESP_OK;
    }

    s_mn_data = s_mn_handle->create(mn_name, VOICE_MN_TIMEOUT_MS);
    if (s_mn_data == NULL) {
        ESP_LOGW(TAG, "MultiNet create failed: %s", mn_name);
        return ESP_OK;
    }

    s_mn_chunksize = s_mn_handle->get_samp_chunksize(s_mn_data);
    int sample_rate = s_mn_handle->get_samp_rate(s_mn_data);
    const char *loader_mode = "default";

    if (s_mn_chunksize <= 0) {
        ESP_LOGW(TAG, "MultiNet chunksize invalid: %d", s_mn_chunksize);
        return ESP_OK;
    }

    if (sample_rate != HELMET_AUDIO_SAMPLE_RATE) {
        ESP_LOGW(TAG, "MultiNet sample rate mismatch: got=%d expected=%d", sample_rate, HELMET_AUDIO_SAMPLE_RATE);
    }

    if (s_mn_handle->set_det_threshold != NULL) {
        s_mn_handle->set_det_threshold(s_mn_data, VOICE_MN_DET_THRESHOLD);
    }

    s_mn_ready = true;
    helmet_multinet_clean();

    ESP_LOGI(
        TAG,
        "MultiNet ready: model=%s chunksize=%d sample_rate=%d commands=offline cancel_id=%d status_id=%d loader=%s threshold=%.2f",
        mn_name,
        s_mn_chunksize,
        sample_rate,
        VOICE_MN_CANCEL_ID,
        VOICE_MN_STATUS_ID,
        loader_mode,
        (double)VOICE_MN_DET_THRESHOLD
    );

    return ESP_OK;
}

static helmet_voice_cmd_t helmet_multinet_detect_command(
    afe_fetch_result_t *afe_res,
    bool cancel_only
)
{
    if (!s_mn_ready || s_mn_handle == NULL || s_mn_data == NULL || afe_res == NULL || afe_res->data == NULL) {
        return HELMET_VOICE_CMD_UNKNOWN;
    }

    int frames = afe_res->data_size / (int)sizeof(int16_t);
    s_mn_last_afe_frames = frames;

    if (frames != s_mn_chunksize) {
        s_mn_frame_mismatch_count++;
        return HELMET_VOICE_CMD_UNKNOWN;
    }

    esp_mn_state_t state = s_mn_handle->detect(s_mn_data, afe_res->data);
    if (state == ESP_MN_STATE_TIMEOUT) {
        s_mn_timeout_count++;
        helmet_multinet_clean();
        return HELMET_VOICE_CMD_UNKNOWN;
    }

    if (state != ESP_MN_STATE_DETECTED) {
        s_mn_detecting_count++;
        return HELMET_VOICE_CMD_UNKNOWN;
    }

    s_mn_detected_count++;

    esp_mn_results_t *results = s_mn_handle->get_results(s_mn_data);
    if (results == NULL || results->num <= 0) {
        helmet_multinet_clean();
        return HELMET_VOICE_CMD_UNKNOWN;
    }

    int command_id = results->command_id[0];
    helmet_voice_cmd_t cmd = HELMET_VOICE_CMD_UNKNOWN;

    if (command_id == VOICE_MN_CANCEL_ID) {
        cmd = HELMET_VOICE_CMD_CANCEL_ALARM;
    } else if (command_id == VOICE_MN_STATUS_ID) {
        cmd = HELMET_VOICE_CMD_REPORT_STATUS;
    }

    ESP_LOGI(
        TAG,
        "MultiNet command detected: id=%d phrase=%d prob=%.3f text=%s raw=%s",
        command_id,
        results->phrase_id[0],
        (double)results->prob[0],
        results->string,
        results->raw_string
    );

    helmet_multinet_clean();

    if (cancel_only && cmd != HELMET_VOICE_CMD_CANCEL_ALARM) {
        ESP_LOGI(TAG, "ignore non-cancel command during alert: %d", (int)cmd);
        return HELMET_VOICE_CMD_UNKNOWN;
    }

    if (cmd != HELMET_VOICE_CMD_CANCEL_ALARM && cmd != HELMET_VOICE_CMD_REPORT_STATUS) {
        ESP_LOGW(TAG, "ignore unsupported MultiNet command id=%d", (int)cmd);
        return HELMET_VOICE_CMD_UNKNOWN;
    }

    return cmd;
}

static esp_err_t helmet_sr_init(void)
{
    ESP_LOGI(TAG, "helmet_sr_init: enter");

    s_sr_models = esp_srmodel_init("model");
    if (s_sr_models == NULL) {
        ESP_LOGE(TAG, "esp_srmodel_init failed");
        return ESP_FAIL;
    }

    afe_config_t *afe_config = afe_config_init(
        "M",
        s_sr_models,
        AFE_TYPE_SR,
        AFE_MODE_LOW_COST
    );

    if (afe_config == NULL) {
        ESP_LOGE(TAG, "afe_config_init failed");
        return ESP_FAIL;
    }

#if VOICE_AFE_VAD_ENABLED
    afe_config->vad_init = true;
    afe_config->vad_mode = VAD_MODE_2;
    afe_config->vad_min_noise_ms = 600;
#else
    afe_config->vad_init = false;
#endif
    afe_config->wakenet_init = false;
    afe_config->wakenet_model_name = NULL;
    afe_config->wakenet_model_name_2 = NULL;
    afe_config->fixed_first_channel = true;
    afe_config->fixed_output_channel = true;

    s_afe_handle = esp_afe_handle_from_config(afe_config);
    if (s_afe_handle == NULL) {
        ESP_LOGE(TAG, "esp_afe_handle_from_config failed");
        afe_config_free(afe_config);
        return ESP_FAIL;
    }

    s_afe_data = s_afe_handle->create_from_config(afe_config);
    afe_config_free(afe_config);

    if (s_afe_data == NULL) {
        ESP_LOGE(TAG, "AFE create_from_config failed");
        return ESP_FAIL;
    }

    s_afe_feed_chunksize = s_afe_handle->get_feed_chunksize(s_afe_data);
    s_afe_feed_channel_num = s_afe_handle->get_feed_channel_num(s_afe_data);

    ESP_LOGI(
        TAG,
        "AFE ready: direct MultiNet mode=low_cost feed_chunksize=%d feed_channel_num=%d vad=%s",
        s_afe_feed_chunksize,
        s_afe_feed_channel_num,
        VOICE_AFE_VAD_ENABLED ? "on" : "off"
    );

    s_afe_handle->print_pipeline(s_afe_data);

    esp_err_t mn_ret = helmet_multinet_init(s_sr_models);
    if (mn_ret != ESP_OK) {
        ESP_LOGW(TAG, "helmet_multinet_init ret=%s", esp_err_to_name(mn_ret));
    }

    return ESP_OK;
}

static void helmet_sr_reset(void)
{
    if (s_afe_handle != NULL && s_afe_data != NULL) {
        s_afe_handle->reset_buffer(s_afe_data);
    }
}

static afe_fetch_result_t *helmet_afe_feed_fetch(const int16_t *pcm, size_t frames)
{
    if (s_afe_handle == NULL || s_afe_data == NULL || pcm == NULL || frames == 0) {
        return NULL;
    }

    if ((int)frames != s_afe_feed_chunksize) {
        static int warn_count = 0;

        if ((warn_count++ % 50) == 0) {
            ESP_LOGW(
                TAG,
                "AFE feed size mismatch: got=%u need=%d",
                (unsigned)frames,
                s_afe_feed_chunksize
            );
        }

        return NULL;
    }

    if (s_afe_feed_channel_num != 1) {
        static int channel_warn_count = 0;

        if ((channel_warn_count++ % 50) == 0) {
            ESP_LOGW(
                TAG,
                "AFE feed channel mismatch: mono input but AFE needs channel_num=%d",
                s_afe_feed_channel_num
            );
        }

        return NULL;
    }

    s_afe_handle->feed(s_afe_data, pcm);

    afe_fetch_result_t *res = s_afe_handle->fetch_with_delay(s_afe_data, 0);
    if (res == NULL) {
        return NULL;
    }

    if (res->ret_value == ESP_FAIL) {
        ESP_LOGW(TAG, "AFE fetch failed");
        return NULL;
    }

    return res;
}

static int voice_calc_avg_abs(const int16_t *pcm, size_t frames)
{
    if (pcm == NULL || frames == 0) {
        return 0;
    }

    int64_t sum_abs = 0;

    for (size_t i = 0; i < frames; i++) {
        int16_t v = pcm[i];
        sum_abs += (v >= 0) ? v : -v;
    }

    return (int)(sum_abs / frames);
}

static bool voice_try_cancel_from_mic(void)
{
    if (!s_mn_ready) {
        return false;
    }

    size_t frames_read = 0;
    esp_err_t ret = helmet_voice_read_pcm(
        s_capture_pcm,
        VOICE_CAPTURE_FRAMES,
        &frames_read
    );

    if (ret != ESP_OK || frames_read == 0) {
        return false;
    }

    afe_fetch_result_t *afe_res = helmet_afe_feed_fetch(s_capture_pcm, frames_read);
    helmet_voice_cmd_t cmd = helmet_multinet_detect_command(afe_res, true);

    if (cmd != HELMET_VOICE_CMD_CANCEL_ALARM) {
        return false;
    }

    ESP_LOGI(TAG, "cancel command detected during alert playback");
    ret = helmet_voice_cmd_execute(cmd);
    ESP_LOGI(TAG, "cancel command executed ret=%s", esp_err_to_name(ret));

    return ret == ESP_OK;
}

static esp_err_t test_speaker_once(void)
{
    make_test_beep();

    s_voice_output_busy = true;
    s_voice_runtime = VOICE_RUNTIME_PLAYING;

    ESP_LOGI(TAG, "speaker test: mono_frames=%u bytes=%u",
             (unsigned)TEST_BEEP_FRAMES,
             (unsigned)sizeof(s_beep));

    esp_err_t ret = ESP_ERR_TIMEOUT;
    if (voice_output_lock(pdMS_TO_TICKS(200))) {
        esp_err_t prep_ret = voice_prepare_output();
        ret = (prep_ret == ESP_OK) ? voice_write_pcm_bytes(s_beep, sizeof(s_beep)) : prep_ret;
        voice_output_unlock();
    }

    vTaskDelay(pdMS_TO_TICKS(150));

    s_voice_output_busy = false;
    s_voice_runtime = VOICE_RUNTIME_IDLE;

    ESP_LOGI(TAG, "speaker test ret=%s", esp_err_to_name(ret));

    return ret;
}

static esp_err_t voice_play_pcm_file(const char *path)
{
    if (path == NULL || path[0] == '\0') {
        return ESP_ERR_INVALID_ARG;
    }

    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        ESP_LOGE(TAG, "open pcm failed: %s", path);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "play pcm file: %s", path);

    esp_err_t ret = ESP_OK;
    size_t total_bytes = 0;

    if (!voice_output_lock(pdMS_TO_TICKS(200))) {
        ESP_LOGW(TAG, "speaker output lock timeout");
        fclose(file);
        return ESP_ERR_TIMEOUT;
    }

    ret = voice_prepare_output();
    voice_output_unlock();

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "prepare speaker failed: %s", esp_err_to_name(ret));
        fclose(file);
        return ret;
    }

    while (!s_voice_stop_requested) {
        size_t bytes_read = fread(
            s_beep,
            1,
            sizeof(s_beep),
            file
        );

        if (bytes_read == 0) {
            if (feof(file)) {
                ret = ESP_OK;
            } else {
                ESP_LOGE(TAG, "read pcm failed: %s", path);
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

        if (!voice_output_lock(pdMS_TO_TICKS(200))) {
            ESP_LOGW(TAG, "speaker output lock timeout");
            ret = ESP_ERR_TIMEOUT;
            break;
        }

        ret = voice_write_pcm_bytes(s_beep, bytes_read);
        voice_output_unlock();

        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "write pcm failed: %s", esp_err_to_name(ret));
            break;
        }

        total_bytes += bytes_read;

        if (voice_try_cancel_from_mic()) {
            break;
        }
    }

    fclose(file);

    if (s_voice_stop_requested && ret == ESP_OK) {
        ESP_LOGI(TAG, "pcm playback stopped by request");
    }

    ESP_LOGI(TAG, "pcm playback done: %s bytes=%u ret=%s",
             path,
             (unsigned)total_bytes,
             esp_err_to_name(ret));

    return ret;
}

static int voice_alert_priority(int alert)
{
    switch ((helmet_voice_alert_t)alert) {
    case HELMET_VOICE_ALERT_DANGER:
        return 3;
    case HELMET_VOICE_ALERT_FATIGUE:
        return 2;
    case HELMET_VOICE_ALERT_GNSS_LOST:
        return 1;
    default:
        return 0;
    }
}

static uint32_t voice_alert_repeat_ms(helmet_voice_alert_t alert)
{
    switch (alert) {
    case HELMET_VOICE_ALERT_DANGER:
        return VOICE_DANGER_REPEAT_MS;

    case HELMET_VOICE_ALERT_FATIGUE:
        return VOICE_FATIGUE_REPEAT_MS;

    case HELMET_VOICE_ALERT_GNSS_LOST:
    default:
        return VOICE_DEFAULT_REPEAT_MS;
    }
}

static bool voice_alert_is_valid(helmet_voice_alert_t alert)
{
    return alert >= HELMET_VOICE_ALERT_FATIGUE && alert <= HELMET_VOICE_ALERT_GNSS_LOST;
}

static const char *voice_alert_pcm_path(helmet_voice_alert_t alert)
{
    switch (alert) {
    case HELMET_VOICE_ALERT_DANGER:
        return VOICE_ACCIDENT_PCM_PATH;

    case HELMET_VOICE_ALERT_FATIGUE:
        return VOICE_TIRED_PCM_PATH;

    case HELMET_VOICE_ALERT_GNSS_LOST:
    default:
        return NULL;
    }
}

static esp_err_t voice_play_alert_now(helmet_voice_alert_t alert)
{
    const char *pcm_path = voice_alert_pcm_path(alert);
    esp_err_t ret;

    s_voice_stop_requested = false;
    s_voice_output_busy = true;
    s_current_alert = (int)alert;
    s_voice_runtime = VOICE_RUNTIME_PLAYING;
    helmet_state_set_voice(HELMET_VOICE_PLAYING);
    helmet_multinet_clean();

    if (pcm_path != NULL) {
        ret = voice_play_pcm_file(pcm_path);
    } else {
        ret = test_speaker_once();
    }

    if (ret == ESP_OK) {
        helmet_state_set_voice(HELMET_VOICE_READY);
    } else {
        helmet_state_set_voice(HELMET_VOICE_ERROR);
    }

    helmet_sr_reset();

    s_voice_runtime = VOICE_RUNTIME_IDLE;
    s_voice_output_busy = false;
    s_voice_stop_requested = false;
    s_current_alert = VOICE_ALERT_NONE;

    if (ret == ESP_OK) {
        s_last_alert_ms[(int)alert] = esp_log_timestamp();
    }

    return ret;
}

static void voice_service_pending_alert(void)
{
    if (!voice_output_lock(pdMS_TO_TICKS(50))) {
        return;
    }

    int pending = s_pending_alert;
    if (pending == VOICE_ALERT_NONE || s_voice_output_busy || s_voice_streaming) {
        voice_output_unlock();
        return;
    }

    s_pending_alert = VOICE_ALERT_NONE;
    s_voice_output_busy = true;
    voice_output_unlock();

    esp_err_t ret = voice_play_alert_now((helmet_voice_alert_t)pending);

    ESP_LOGI(TAG, "pending alert done ret=%s", esp_err_to_name(ret));
}

static void voice_alert_task(void *arg)
{
    (void)arg;

    ESP_LOGI(TAG, "voice alert task started");

    while (1) {
        if (s_voice_alert_signal != NULL) {
            xSemaphoreTake(s_voice_alert_signal, pdMS_TO_TICKS(250));
        } else {
            vTaskDelay(pdMS_TO_TICKS(250));
        }

        voice_service_pending_alert();
    }
}

static esp_err_t start_voice_alert_task(void)
{
    if (s_voice_alert_task_handle != NULL) {
        return ESP_OK;
    }

    if (s_voice_alert_signal == NULL) {
        s_voice_alert_signal = xSemaphoreCreateBinary();
        if (s_voice_alert_signal == NULL) {
            ESP_LOGE(TAG, "create voice alert signal failed");
            return ESP_ERR_NO_MEM;
        }
    }

    BaseType_t task_ret = xTaskCreatePinnedToCore(
        voice_alert_task,
        "voice_alert",
        VOICE_ALERT_TASK_STACK_SIZE,
        NULL,
        VOICE_ALERT_TASK_PRIORITY,
        &s_voice_alert_task_handle,
        VOICE_ALERT_TASK_CORE
    );

    if (task_ret != pdPASS) {
        s_voice_alert_task_handle = NULL;
        ESP_LOGE(TAG, "voice alert task: create failed");
        return ESP_FAIL;
    }

    return ESP_OK;
}

esp_err_t helmet_voice_read_pcm(int16_t *mono_buffer, size_t frames, size_t *frames_read)
{
    if (!s_voice_inited || s_mic_codec == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    if (mono_buffer == NULL || frames == 0 || frames_read == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    size_t bytes_to_read = frames * sizeof(int16_t);

    esp_err_t ret = esp_codec_dev_read(s_mic_codec, mono_buffer, (int)bytes_to_read);

    if (ret != ESP_CODEC_DEV_OK) {
        *frames_read = 0;
        ESP_LOGE(TAG, "helmet_voice_read_pcm: read failed ret=%s", esp_err_to_name(ret));
        return (esp_err_t)ret;
    }

    *frames_read = frames;

    return ESP_OK;
}

static void voice_capture_task(void *arg)
{
    (void)arg;

    ESP_LOGI(TAG, "voice capture task: start direct MultiNet mode");

    int log_count = 0;
    helmet_voice_cmd_t last_cmd = HELMET_VOICE_CMD_UNKNOWN;
    TickType_t last_cmd_tick = 0;

    while (1) {
        if (s_voice_output_busy) {
            vTaskDelay(pdMS_TO_TICKS(VOICE_CAPTURE_INTERVAL_MS));
            continue;
        }

        size_t frames_read = 0;

        esp_err_t ret = helmet_voice_read_pcm(
            s_capture_pcm,
            VOICE_CAPTURE_FRAMES,
            &frames_read
        );

        if (ret == ESP_OK && frames_read > 0) {
            int avg_abs = voice_calc_avg_abs(s_capture_pcm, frames_read);
            afe_fetch_result_t *afe_res = helmet_afe_feed_fetch(s_capture_pcm, frames_read);
            int vad_state = -1;

#if VOICE_AFE_VAD_ENABLED
            vad_state = (afe_res != NULL) ? (int)afe_res->vad_state : -1;
#endif

            helmet_voice_cmd_t detected_cmd = helmet_multinet_detect_command(afe_res, false);
            if (detected_cmd != HELMET_VOICE_CMD_UNKNOWN) {
                TickType_t now = xTaskGetTickCount();
                bool duplicate = (detected_cmd == last_cmd) &&
                                 (last_cmd_tick != 0) &&
                                 ((now - last_cmd_tick) < pdMS_TO_TICKS(VOICE_CMD_COOLDOWN_MS));

                if (duplicate) {
                    ESP_LOGI(TAG, "command ignored by cooldown: id=%d", (int)detected_cmd);
                } else {
                    esp_err_t cmd_ret = helmet_voice_cmd_execute(detected_cmd);

                    ESP_LOGI(
                        TAG,
                        "command executed: id=%d ret=%s",
                        (int)detected_cmd,
                        esp_err_to_name(cmd_ret)
                    );

                    last_cmd = detected_cmd;
                    last_cmd_tick = now;
                }

                helmet_sr_reset();
                vTaskDelay(pdMS_TO_TICKS(VOICE_CAPTURE_INTERVAL_MS));
                continue;
            }

            if ((log_count++ % VOICE_LOG_EVERY_N_FRAMES) == 0) {
                ESP_LOGI(
                    TAG,
                    "direct command monitor: mn_ready=%d vad=%d avg_abs=%d afe_frames=%d mn_need=%d detecting=%lu detected=%lu timeout=%lu mismatch=%lu",
                    s_mn_ready ? 1 : 0,
                    vad_state,
                    avg_abs,
                    s_mn_last_afe_frames,
                    s_mn_chunksize,
                    (unsigned long)s_mn_detecting_count,
                    (unsigned long)s_mn_detected_count,
                    (unsigned long)s_mn_timeout_count,
                    (unsigned long)s_mn_frame_mismatch_count
                );
            }
        } else {
            ESP_LOGW(
                TAG,
                "voice pcm read failed ret=%s frames=%u",
                esp_err_to_name(ret),
                (unsigned)frames_read
            );
        }

        vTaskDelay(pdMS_TO_TICKS(VOICE_CAPTURE_INTERVAL_MS));
    }
}

static esp_err_t start_voice_capture_task(void)
{
    if (s_voice_capture_task_handle != NULL) {
        ESP_LOGI(TAG, "voice capture task: already running");
        return ESP_OK;
    }

    BaseType_t task_ret = xTaskCreate(
        voice_capture_task,
        "voice_capture",
        3072,
        NULL,
        5,
        &s_voice_capture_task_handle
    );

    if (task_ret != pdPASS) {
        s_voice_capture_task_handle = NULL;
        ESP_LOGE(TAG, "voice capture task: create failed");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "voice capture task: created");

    return ESP_OK;
}

static void voice_sr_init_task(void *arg)
{
    (void)arg;

    ESP_LOGI(TAG, "voice SR init task: start core=%d", VOICE_SR_INIT_TASK_CORE);

    uint32_t start_ms = esp_log_timestamp();
    esp_err_t ret = helmet_sr_init();
    uint32_t elapsed_ms = esp_log_timestamp() - start_ms;

    ESP_LOGI(TAG, "helmet_sr_init ret=%s elapsed=%" PRIu32 "ms",
             esp_err_to_name(ret), elapsed_ms);

    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "helmet_voice_init: skip speaker test at boot");
        ret = start_voice_capture_task();
    }

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "voice SR init task failed: %s", esp_err_to_name(ret));
        helmet_state_set_voice(HELMET_VOICE_ERROR);
        s_voice_runtime = VOICE_RUNTIME_ERROR;
    } else {
        helmet_state_set_voice(HELMET_VOICE_READY);
        s_voice_runtime = VOICE_RUNTIME_IDLE;
        ESP_LOGI(TAG, "voice SR init task: done");
    }

    s_voice_sr_init_task_handle = NULL;
    vTaskDelete(NULL);
}

esp_err_t helmet_voice_init(void)
{
    ESP_LOGI(TAG, "helmet_voice_init: enter");

    if (s_voice_inited) {
        ESP_LOGI(TAG, "helmet_voice_init: already initialized");
        helmet_state_set_voice(HELMET_VOICE_READY);
        return ESP_OK;
    }

    if (s_voice_output_mutex == NULL) {
        s_voice_output_mutex = xSemaphoreCreateMutex();
        if (s_voice_output_mutex == NULL) {
            ESP_LOGE(TAG, "create voice output mutex failed");
            helmet_state_set_voice(HELMET_VOICE_ERROR);
            return ESP_ERR_NO_MEM;
        }
    }

    s_spk_codec = bsp_audio_codec_speaker_init();
    if (s_spk_codec == NULL) {
        ESP_LOGE(TAG, "bsp_audio_codec_speaker_init failed");
        helmet_state_set_voice(HELMET_VOICE_ERROR);
        return ESP_FAIL;
    }

    s_mic_codec = bsp_audio_codec_microphone_init();
    if (s_mic_codec == NULL) {
        ESP_LOGE(TAG, "bsp_audio_codec_microphone_init failed");
        helmet_state_set_voice(HELMET_VOICE_ERROR);
        return ESP_FAIL;
    }

    esp_codec_dev_sample_info_t fs = {
        .sample_rate = HELMET_AUDIO_SAMPLE_RATE,
        .channel = HELMET_AUDIO_CHANNELS,
        .bits_per_sample = HELMET_AUDIO_BITS,
    };

    int codec_ret = esp_codec_dev_open(s_spk_codec, &fs);
    ESP_LOGI(TAG, "speaker esp_codec_dev_open ret=%s", esp_err_to_name(codec_ret));
    if (codec_ret != ESP_CODEC_DEV_OK) {
        helmet_state_set_voice(HELMET_VOICE_ERROR);
        return (esp_err_t)codec_ret;
    }

    codec_ret = esp_codec_dev_open(s_mic_codec, &fs);
    ESP_LOGI(TAG, "mic esp_codec_dev_open ret=%s", esp_err_to_name(codec_ret));
    if (codec_ret != ESP_CODEC_DEV_OK) {
        helmet_state_set_voice(HELMET_VOICE_ERROR);
        return (esp_err_t)codec_ret;
    }

    codec_ret = esp_codec_dev_set_in_gain(s_mic_codec, HELMET_MIC_GAIN);
    ESP_LOGI(TAG, "esp_codec_dev_set_in_gain ret=%s", esp_err_to_name(codec_ret));
    if (codec_ret != ESP_CODEC_DEV_OK) {
        ESP_LOGW(TAG, "set mic gain failed, continue");
    }

    esp_err_t ret = voice_prepare_output();
    ESP_LOGI(TAG, "voice_prepare_output ret=%s", esp_err_to_name(ret));

    if (ret != ESP_OK) {
        helmet_state_set_voice(HELMET_VOICE_ERROR);
        return ret;
    }

    s_voice_inited = true;
    s_voice_runtime = VOICE_RUNTIME_IDLE;
    helmet_state_set_voice(HELMET_VOICE_READY);

    ret = start_voice_alert_task();
    if (ret != ESP_OK) {
        helmet_state_set_voice(HELMET_VOICE_ERROR);
        s_voice_runtime = VOICE_RUNTIME_ERROR;
        return ret;
    }

    ret = test_speaker_once();
    ESP_LOGI(TAG, "startup speaker self-test ret=%s", esp_err_to_name(ret));
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "startup speaker self-test failed; alerts will still be queued");
    }

    if (s_voice_sr_init_task_handle == NULL && s_voice_capture_task_handle == NULL) {
        BaseType_t task_ret = xTaskCreatePinnedToCore(
            voice_sr_init_task,
            "voice_sr_init",
            VOICE_SR_INIT_TASK_STACK_SIZE,
            NULL,
            VOICE_SR_INIT_TASK_PRIORITY,
            &s_voice_sr_init_task_handle,
            VOICE_SR_INIT_TASK_CORE
        );

        if (task_ret != pdPASS) {
            s_voice_sr_init_task_handle = NULL;
            ESP_LOGE(TAG, "helmet_voice_init: create voice SR init task failed");
            helmet_state_set_voice(HELMET_VOICE_ERROR);
            s_voice_runtime = VOICE_RUNTIME_ERROR;
            return ESP_FAIL;
        }

        ESP_LOGI(TAG, "helmet_voice_init: voice SR init deferred");
    } else {
        ESP_LOGI(TAG, "helmet_voice_init: voice SR/capture already active");
    }

    ESP_LOGI(TAG, "helmet_voice_init: done");

    return ESP_OK;
}

esp_err_t helmet_voice_play_alert(helmet_voice_alert_t alert)
{
    ESP_LOGI(TAG, "play alert: %d", alert);

    if (!s_voice_inited) {
        return ESP_ERR_INVALID_STATE;
    }

    if (!voice_alert_is_valid(alert)) {
        return ESP_ERR_INVALID_ARG;
    }

    if (!voice_output_lock(pdMS_TO_TICKS(50))) {
        ESP_LOGW(TAG, "voice output lock timeout, skip alert=%d", alert);
        return ESP_ERR_TIMEOUT;
    }

    bool output_busy = s_voice_output_busy || s_voice_streaming;
    uint32_t now_ms = esp_log_timestamp();
    uint32_t repeat_ms = voice_alert_repeat_ms(alert);

    if (output_busy && s_current_alert == (int)alert) {
        ESP_LOGI(TAG, "alert=%d already playing, skip duplicate", alert);
        voice_output_unlock();
        return ESP_OK;
    }

    if (!output_busy && s_pending_alert != (int)alert && s_last_alert_ms[(int)alert] != 0) {
        uint32_t elapsed_ms = now_ms - s_last_alert_ms[(int)alert];
        if (elapsed_ms < repeat_ms) {
            ESP_LOGI(TAG, "alert=%d cooldown %u/%u ms, skip duplicate",
                     alert,
                     (unsigned)elapsed_ms,
                     (unsigned)repeat_ms);
            voice_output_unlock();
            return ESP_OK;
        }
    }

    if (s_pending_alert != VOICE_ALERT_NONE) {
        if (voice_alert_priority((int)alert) > voice_alert_priority(s_pending_alert)) {
            ESP_LOGW(TAG, "replace pending alert %d with %d", s_pending_alert, alert);
            s_pending_alert = (int)alert;
        } else {
            ESP_LOGW(TAG, "voice alert already pending=%d, skip alert=%d", s_pending_alert, alert);
            voice_output_unlock();
            return s_pending_alert == (int)alert ? ESP_OK : ESP_ERR_INVALID_STATE;
        }
    } else {
        if (output_busy) {
            s_pending_alert = (int)alert;
        }
    }

    voice_output_unlock();

    if (!output_busy) {
        return voice_play_alert_now(alert);
    }

    if (s_voice_alert_signal != NULL) {
        xSemaphoreGive(s_voice_alert_signal);
    }
    ESP_LOGI(TAG, "voice output busy, queued alert=%d", alert);

    return ESP_OK;
}

esp_err_t helmet_voice_stream_pcm_begin(void)
{
    if (!s_voice_inited) {
        return ESP_ERR_INVALID_STATE;
    }

    if (!voice_output_lock(pdMS_TO_TICKS(200))) {
        return ESP_ERR_TIMEOUT;
    }

    if (s_voice_output_busy || s_pending_alert != VOICE_ALERT_NONE) {
        voice_output_unlock();
        ESP_LOGW(TAG, "voice output busy, skip pcm stream");
        return ESP_ERR_INVALID_STATE;
    }

    s_voice_stop_requested = false;
    s_voice_streaming = true;
    s_voice_output_busy = true;
    s_voice_runtime = VOICE_RUNTIME_PLAYING;
    helmet_state_set_voice(HELMET_VOICE_PLAYING);

    esp_err_t ret = voice_prepare_output();
    if (ret != ESP_OK) {
        s_voice_streaming = false;
        s_voice_output_busy = false;
        s_voice_runtime = VOICE_RUNTIME_ERROR;
        helmet_state_set_voice(HELMET_VOICE_ERROR);
        voice_output_unlock();
        ESP_LOGE(TAG, "prepare pcm stream failed: %s", esp_err_to_name(ret));
        return ret;
    }

    voice_output_unlock();

    ESP_LOGI(TAG, "pcm stream begin");

    return ESP_OK;
}

esp_err_t helmet_voice_stream_pcm_write(const void *pcm, size_t bytes)
{
    if (!s_voice_inited) {
        return ESP_ERR_INVALID_STATE;
    }

    if (pcm == NULL || bytes == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    if (!s_voice_streaming || s_voice_stop_requested) {
        return ESP_ERR_INVALID_STATE;
    }

    if ((bytes & 1U) != 0U) {
        bytes--;
    }

    if (bytes == 0) {
        return ESP_OK;
    }

    if (!voice_output_lock(pdMS_TO_TICKS(500))) {
        return ESP_ERR_TIMEOUT;
    }

    esp_err_t ret = voice_write_pcm_bytes(pcm, bytes);
    voice_output_unlock();

    return ret;
}

esp_err_t helmet_voice_stream_pcm_end(void)
{
    if (!voice_output_lock(pdMS_TO_TICKS(200))) {
        return ESP_ERR_TIMEOUT;
    }

    if (s_voice_streaming) {
        ESP_LOGI(TAG, "pcm stream end");
    }

    s_voice_streaming = false;
    s_voice_output_busy = false;
    s_voice_stop_requested = false;
    s_voice_runtime = VOICE_RUNTIME_IDLE;
    helmet_state_set_voice(HELMET_VOICE_READY);

    voice_output_unlock();

    return ESP_OK;
}

esp_err_t helmet_voice_stop(void)
{
    ESP_LOGI(TAG, "voice stop");

    s_voice_stop_requested = true;
    s_pending_alert = VOICE_ALERT_NONE;

    if (s_voice_streaming) {
        s_voice_streaming = false;
        s_voice_output_busy = false;
        s_voice_runtime = VOICE_RUNTIME_IDLE;
        s_voice_stop_requested = false;
        helmet_state_set_voice(HELMET_VOICE_READY);
        return ESP_OK;
    }

    if (!s_voice_output_busy) {
        s_voice_runtime = VOICE_RUNTIME_IDLE;
        s_voice_stop_requested = false;
        helmet_state_set_voice(HELMET_VOICE_READY);
        return ESP_OK;
    }

    helmet_state_set_voice(HELMET_VOICE_READY);

    return ESP_OK;
}
