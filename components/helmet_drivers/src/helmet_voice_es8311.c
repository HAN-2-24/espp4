#include "helmet_voice.h"

#include "helmet_state.h"

#include "esp_err.h"
#include "esp_log.h"

#include "bsp/esp32_p4_function_ev_board.h"
#include "esp_codec_dev.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_afe_config.h"
#include "esp_afe_sr_iface.h"
#include "model_path.h"

/*
 * Some esp-sr package versions used by this project do not expose this symbol
 * in the public headers even though the library provides it.
 */
extern const esp_afe_sr_iface_t *esp_afe_handle_from_config(afe_config_t *afe_config);

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

static const char *TAG = "helmet_voice";

static bool s_voice_inited = false;
static TaskHandle_t s_voice_capture_task_handle = NULL;

static esp_codec_dev_handle_t s_spk_codec = NULL;
static esp_codec_dev_handle_t s_mic_codec = NULL;

#define HELMET_AUDIO_SAMPLE_RATE          16000
#define HELMET_AUDIO_BITS                 16
#define HELMET_AUDIO_CHANNELS             1
#define HELMET_AUDIO_VOLUME               100
#define HELMET_MIC_GAIN                   42.0f

#define VOICE_CAPTURE_FRAMES              512
#define VOICE_CAPTURE_INTERVAL_MS         20

#define VOICE_VAD_STOP_THRESHOLD          350
#define VOICE_VAD_STOP_COUNT              35
#define VOICE_VAD_AFE_STOP_COUNT          18
#define VOICE_VAD_GRACE_FRAMES            24
#define VOICE_VAD_NOISE_MARGIN            120
#define VOICE_WAKE_TAIL_MIN_FRAMES        18
#define VOICE_WAKE_TAIL_SILENCE_COUNT     8
#define VOICE_WAKE_TAIL_MAX_FRAMES        55
#define VOICE_COMMAND_START_TIMEOUT_FRAMES 120
#define VOICE_SPEECH_START_COUNT          2
#define VOICE_LISTEN_MAX_FRAMES           280

#define VOICE_LOG_EVERY_N_FRAMES          50
#define VOICE_PCM_EVENT_LOG_EVERY_N       25

#define VOICE_ACCIDENT_PCM_PATH           BSP_SPIFFS_MOUNT_POINT "/music/accident.pcm"
#define VOICE_ALERT_NONE                  (-1)

static int16_t s_capture_pcm[VOICE_CAPTURE_FRAMES];

#define TEST_BEEP_FRAMES                  512
#define TEST_BEEP_AMPLITUDE               6000
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

static const esp_afe_sr_iface_t *s_afe_handle = NULL;
static esp_afe_sr_data_t *s_afe_data = NULL;
static int s_afe_feed_chunksize = 0;
static int s_afe_feed_channel_num = 0;

static void make_test_beep(void)
{
    for (int i = 0; i < TEST_BEEP_FRAMES; i++) {
        int16_t v = ((i / 8) % 2 == 0) ? TEST_BEEP_AMPLITUDE : -TEST_BEEP_AMPLITUDE;
        s_beep[i] = v;
    }
}

static esp_err_t helmet_wake_init(void)
{
    ESP_LOGI(TAG, "helmet_wake_init: enter");

    srmodel_list_t *models = esp_srmodel_init("model");
    if (models == NULL) {
        ESP_LOGE(TAG, "esp_srmodel_init failed");
        return ESP_FAIL;
    }

    afe_config_t *afe_config = afe_config_init(
        "M",
        models,
        AFE_TYPE_SR,
        AFE_MODE_HIGH_PERF
    );

    if (afe_config == NULL) {
        ESP_LOGE(TAG, "afe_config_init failed");
        return ESP_FAIL;
    }

    afe_config->vad_init = true;
    afe_config->vad_mode = VAD_MODE_2;
    afe_config->vad_min_noise_ms = 600;

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
        "WakeNet ready: feed_chunksize=%d feed_channel_num=%d",
        s_afe_feed_chunksize,
        s_afe_feed_channel_num
    );

    s_afe_handle->print_pipeline(s_afe_data);

    return ESP_OK;
}

static void helmet_wake_reset(void)
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

static bool helmet_wake_detect_feed(const int16_t *pcm, size_t frames)
{
    afe_fetch_result_t *res = helmet_afe_feed_fetch(pcm, frames);
    if (res == NULL) {
        return false;
    }

    if (res->wakeup_state == WAKENET_DETECTED) {
        ESP_LOGI(
            TAG,
            "WAKE DETECTED wake_word_index=%d",
            res->wake_word_index
        );

        return true;
    }

    return false;
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

static int voice_update_noise_floor(int current, int avg_abs)
{
    if (avg_abs <= 0) {
        return current;
    }

    if (current <= 0) {
        return avg_abs;
    }

    return ((current * 7) + avg_abs) / 8;
}

static void voice_capture_print_stats(const int16_t *pcm, size_t frames, int avg_abs)
{
    if (pcm == NULL || frames < 4) {
        return;
    }

    int16_t min_v = pcm[0];
    int16_t max_v = pcm[0];

    for (size_t i = 0; i < frames; i++) {
        int16_t v = pcm[i];

        if (v < min_v) {
            min_v = v;
        }

        if (v > max_v) {
            max_v = v;
        }
    }

    ESP_LOGI(
        TAG,
        "voice pcm: frames=%u min=%d max=%d avg_abs=%d runtime=%d first=%d %d %d %d",
        (unsigned)frames,
        min_v,
        max_v,
        avg_abs,
        (int)s_voice_runtime,
        pcm[0],
        pcm[1],
        pcm[2],
        pcm[3]
    );
}

static void voice_on_listen_start(int avg_abs)
{
    s_voice_runtime = VOICE_RUNTIME_LISTENING;

    ESP_LOGI(
        TAG,
        "EVENT: LISTEN_START by WakeNet avg_abs=%d",
        avg_abs
    );
}

static void voice_on_pcm_frame(const int16_t *pcm, size_t frames)
{
    static int pcm_event_count = 0;

    if (pcm == NULL || frames < 4) {
        return;
    }

    if ((pcm_event_count++ % VOICE_PCM_EVENT_LOG_EVERY_N) == 0) {
        ESP_LOGI(
            TAG,
            "EVENT: PCM_FRAME frames=%u first=%d %d %d %d",
            (unsigned)frames,
            pcm[0],
            pcm[1],
            pcm[2],
            pcm[3]
        );
    }
}

static void voice_on_listen_stop(int avg_abs, const char *reason)
{
    s_voice_runtime = VOICE_RUNTIME_IDLE;

    ESP_LOGI(
        TAG,
        "EVENT: LISTEN_STOP reason=%s avg_abs=%d no recognizer active",
        (reason != NULL) ? reason : "unknown",
        avg_abs
    );
}

static esp_err_t test_speaker_once(void)
{
    if (s_spk_codec == NULL) {
        ESP_LOGE(TAG, "speaker test failed: speaker codec is NULL");
        return ESP_ERR_INVALID_STATE;
    }

    make_test_beep();

    s_voice_output_busy = true;
    s_voice_runtime = VOICE_RUNTIME_PLAYING;

    ESP_LOGI(TAG, "speaker test: write bytes=%u", (unsigned)sizeof(s_beep));

    esp_err_t ret = esp_codec_dev_write(
        s_spk_codec,
        s_beep,
        sizeof(s_beep)
    );

    vTaskDelay(pdMS_TO_TICKS(150));

    s_voice_output_busy = false;
    s_voice_runtime = VOICE_RUNTIME_IDLE;

    ESP_LOGI(TAG, "speaker test: ret=%s", esp_err_to_name(ret));

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

        ret = esp_codec_dev_write(
            s_spk_codec,
            s_beep,
            bytes_read
        );

        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "write pcm failed: %s", esp_err_to_name(ret));
            break;
        }
    }

    fclose(file);

    if (s_voice_stop_requested && ret == ESP_OK) {
        ESP_LOGI(TAG, "pcm playback stopped by request");
    }

    return ret;
}

static const char *voice_alert_pcm_path(helmet_voice_alert_t alert)
{
    switch (alert) {
    case HELMET_VOICE_ALERT_DANGER:
        return VOICE_ACCIDENT_PCM_PATH;

    case HELMET_VOICE_ALERT_FATIGUE:
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
    s_voice_runtime = VOICE_RUNTIME_PLAYING;
    helmet_state_set_voice(HELMET_VOICE_PLAYING);

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

    helmet_wake_reset();

    s_voice_runtime = VOICE_RUNTIME_IDLE;
    s_voice_output_busy = false;
    s_voice_stop_requested = false;

    return ret;
}

static void voice_service_pending_alert(void)
{
    int pending = s_pending_alert;

    if (pending == VOICE_ALERT_NONE || s_voice_output_busy) {
        return;
    }

    s_pending_alert = VOICE_ALERT_NONE;

    esp_err_t ret = voice_play_alert_now((helmet_voice_alert_t)pending);

    ESP_LOGI(TAG, "pending alert done ret=%s", esp_err_to_name(ret));
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

    esp_err_t ret = esp_codec_dev_read(
        s_mic_codec,
        mono_buffer,
        bytes_to_read
    );

    if (ret != ESP_OK) {
        *frames_read = 0;
        ESP_LOGE(TAG, "helmet_voice_read_pcm: read failed ret=%s", esp_err_to_name(ret));
        return ret;
    }

    *frames_read = frames;

    return ESP_OK;
}

static void voice_capture_task(void *arg)
{
    (void)arg;

    ESP_LOGI(TAG, "voice capture task: start");

    bool speaking = false;
    int stop_count = 0;
    int log_count = 0;
    int listen_frame_count = 0;
    int ambient_avg_abs = 0;
    int listen_noise_floor = 0;
    int speech_count = 0;
    int wake_tail_silence_count = 0;
    int command_wait_count = 0;
    bool heard_speech = false;
    bool command_armed = false;

    while (1) {
        voice_service_pending_alert();

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

            if (!speaking) {
                if (helmet_wake_detect_feed(s_capture_pcm, frames_read)) {
                    speaking = true;
                    stop_count = 0;
                    log_count = 0;
                    listen_frame_count = 0;
                    listen_noise_floor = ambient_avg_abs;
                    speech_count = 0;
                    wake_tail_silence_count = 0;
                    command_wait_count = 0;
                    heard_speech = false;
                    command_armed = false;

                    voice_on_listen_start(avg_abs);
                } else {
                    ambient_avg_abs = voice_update_noise_floor(ambient_avg_abs, avg_abs);
                }
            } else {
                afe_fetch_result_t *afe_res = helmet_afe_feed_fetch(s_capture_pcm, frames_read);
                int vad_state = (afe_res != NULL) ? (int)afe_res->vad_state : -1;
                int energy_stop_threshold = VOICE_VAD_STOP_THRESHOLD;

                if (listen_noise_floor > 0 &&
                    (listen_noise_floor + VOICE_VAD_NOISE_MARGIN) > energy_stop_threshold) {
                    energy_stop_threshold = listen_noise_floor + VOICE_VAD_NOISE_MARGIN;
                }

                bool afe_silence = (afe_res != NULL && afe_res->vad_state == VAD_SILENCE);
                bool afe_speech = (afe_res != NULL && afe_res->vad_state == VAD_SPEECH);
                bool energy_silence = (avg_abs <= energy_stop_threshold);
                bool energy_speech = (avg_abs > energy_stop_threshold);
                bool speech_detected = afe_speech || energy_speech;
                bool silence_detected = (afe_silence && !energy_speech) ||
                                        (!afe_speech && energy_silence);
                bool can_stop = (listen_frame_count >= VOICE_VAD_GRACE_FRAMES);
                bool stop_by_timeout = false;
                const char *stop_reason = NULL;

                if (!command_armed) {
                    if (silence_detected && listen_frame_count >= VOICE_WAKE_TAIL_MIN_FRAMES) {
                        wake_tail_silence_count++;
                    } else if (!silence_detected && wake_tail_silence_count > 0) {
                        wake_tail_silence_count--;
                    }

                    if (wake_tail_silence_count >= VOICE_WAKE_TAIL_SILENCE_COUNT ||
                        listen_frame_count >= VOICE_WAKE_TAIL_MAX_FRAMES) {
                        command_armed = true;
                        command_wait_count = 0;
                        speech_count = 0;
                        stop_count = 0;

                        ESP_LOGI(
                            TAG,
                            "EVENT: COMMAND_ARMED vad=%d avg_abs=%d threshold=%d frame=%d tail_silence=%d",
                            vad_state,
                            avg_abs,
                            energy_stop_threshold,
                            listen_frame_count,
                            wake_tail_silence_count
                        );
                    }
                } else if (!heard_speech) {
                    command_wait_count++;

                    if (speech_detected) {
                        if (speech_count < VOICE_SPEECH_START_COUNT) {
                            speech_count++;
                        }
                    } else if (speech_count > 0) {
                        speech_count--;
                    }

                    if (speech_count >= VOICE_SPEECH_START_COUNT) {
                        heard_speech = true;
                        stop_count = 0;

                        ESP_LOGI(
                            TAG,
                            "EVENT: COMMAND_SPEECH_START vad=%d avg_abs=%d threshold=%d",
                            vad_state,
                            avg_abs,
                            energy_stop_threshold
                        );
                    }
                }

                voice_on_pcm_frame(s_capture_pcm, frames_read);

                if ((log_count++ % VOICE_LOG_EVERY_N_FRAMES) == 0) {
                    voice_capture_print_stats(s_capture_pcm, frames_read, avg_abs);
                    ESP_LOGI(
                        TAG,
                        "listen monitor: vad=%d avg_abs=%d threshold=%d stop=%d frame=%d armed=%d heard=%d speech=%d wait=%d tail=%d",
                        vad_state,
                        avg_abs,
                        energy_stop_threshold,
                        stop_count,
                        listen_frame_count,
                        command_armed ? 1 : 0,
                        heard_speech ? 1 : 0,
                        speech_count,
                        command_wait_count,
                        wake_tail_silence_count
                    );
                }

                listen_frame_count++;

                if (listen_frame_count >= VOICE_LISTEN_MAX_FRAMES) {
                    stop_by_timeout = true;
                    stop_reason = "timeout";
                } else if (command_armed &&
                           !heard_speech &&
                           command_wait_count >= VOICE_COMMAND_START_TIMEOUT_FRAMES) {
                    stop_by_timeout = true;
                    stop_reason = "speech_start_timeout";
                } else if (heard_speech && can_stop && silence_detected) {
                    stop_count++;

                    if (afe_silence && stop_count >= VOICE_VAD_AFE_STOP_COUNT) {
                        stop_reason = "afe_silence";
                    } else if (!afe_speech && stop_count >= VOICE_VAD_STOP_COUNT) {
                        stop_reason = "energy_silence";
                    }
                } else {
                    stop_count = 0;
                }

                if (stop_by_timeout || stop_reason != NULL) {
                    speaking = false;
                    stop_count = 0;
                    log_count = 0;
                    listen_frame_count = 0;
                    speech_count = 0;
                    wake_tail_silence_count = 0;
                    command_wait_count = 0;
                    heard_speech = false;
                    command_armed = false;

                    voice_on_listen_stop(avg_abs, stop_reason);
                    helmet_wake_reset();
                }
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

esp_err_t helmet_voice_init(void)
{
    ESP_LOGI(TAG, "helmet_voice_init: enter");

    if (s_voice_inited) {
        ESP_LOGI(TAG, "helmet_voice_init: already initialized");
        helmet_state_set_voice(HELMET_VOICE_READY);
        return ESP_OK;
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

    esp_err_t ret = esp_codec_dev_open(s_spk_codec, &fs);
    ESP_LOGI(TAG, "speaker esp_codec_dev_open ret=%s", esp_err_to_name(ret));

    if (ret != ESP_OK) {
        helmet_state_set_voice(HELMET_VOICE_ERROR);
        return ret;
    }

    ret = esp_codec_dev_open(s_mic_codec, &fs);
    ESP_LOGI(TAG, "mic esp_codec_dev_open ret=%s", esp_err_to_name(ret));

    if (ret != ESP_OK) {
        helmet_state_set_voice(HELMET_VOICE_ERROR);
        return ret;
    }

    ret = esp_codec_dev_set_out_vol(s_spk_codec, HELMET_AUDIO_VOLUME);
    ESP_LOGI(TAG, "esp_codec_dev_set_out_vol ret=%s", esp_err_to_name(ret));

    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "set speaker volume failed, continue");
    }

    ret = esp_codec_dev_set_in_gain(s_mic_codec, HELMET_MIC_GAIN);
    ESP_LOGI(TAG, "esp_codec_dev_set_in_gain ret=%s", esp_err_to_name(ret));

    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "set mic gain failed, continue");
    }

    s_voice_inited = true;
    s_voice_runtime = VOICE_RUNTIME_IDLE;
    helmet_state_set_voice(HELMET_VOICE_READY);

    ret = helmet_wake_init();
    ESP_LOGI(TAG, "helmet_wake_init ret=%s", esp_err_to_name(ret));

    if (ret != ESP_OK) {
        helmet_state_set_voice(HELMET_VOICE_ERROR);
        s_voice_inited = false;
        s_voice_runtime = VOICE_RUNTIME_ERROR;
        return ret;
    }

    ESP_LOGI(TAG, "helmet_voice_init: skip speaker test at boot");

    ret = start_voice_capture_task();

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "helmet_voice_init: start voice capture task failed");
        helmet_state_set_voice(HELMET_VOICE_ERROR);
        s_voice_inited = false;
        s_voice_runtime = VOICE_RUNTIME_ERROR;
        return ret;
    }

    ESP_LOGI(TAG, "helmet_voice_init: done");

    return ESP_OK;
}

esp_err_t helmet_voice_play_alert(helmet_voice_alert_t alert)
{
    ESP_LOGI(TAG, "play alert: %d", alert);

    if (!s_voice_inited || s_spk_codec == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    if (s_voice_output_busy) {
        ESP_LOGW(TAG, "voice output busy, skip alert");
        return ESP_ERR_INVALID_STATE;
    }

    if (s_pending_alert != VOICE_ALERT_NONE) {
        ESP_LOGW(TAG, "voice alert already pending, skip alert=%d", alert);
        return ESP_ERR_INVALID_STATE;
    }

    s_pending_alert = (int)alert;

    return ESP_OK;
}

esp_err_t helmet_voice_stop(void)
{
    ESP_LOGI(TAG, "voice stop");

    s_voice_stop_requested = true;
    s_pending_alert = VOICE_ALERT_NONE;

    if (!s_voice_output_busy) {
        s_voice_runtime = VOICE_RUNTIME_IDLE;
        s_voice_stop_requested = false;
        helmet_state_set_voice(HELMET_VOICE_READY);
        return ESP_OK;
    }

    helmet_state_set_voice(HELMET_VOICE_READY);

    return ESP_OK;
}
