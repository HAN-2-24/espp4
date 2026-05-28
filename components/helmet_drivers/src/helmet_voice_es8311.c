#include "helmet_voice.h"

#include "helmet_state.h"

#include "esp_log.h"
#include "esp_err.h"

#include "bsp/esp32_p4_function_ev_board.h"
#include "esp_codec_dev.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/*
 * WakeNet / AFE
 */
#include "esp_afe_sr_iface.h"
#include "esp_afe_config.h"
#include "model_path.h"

/*
 * 当前 esp-sr 头文件可能没有暴露这个声明，但库里有这个符号。
 * 用 extern 补声明，避免 implicit declaration。
 */
extern const esp_afe_sr_iface_t *esp_afe_handle_from_config(afe_config_t *afe_config);

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

static const char *TAG = "helmet_voice";

static bool s_voice_inited = false;
static TaskHandle_t s_voice_capture_task_handle = NULL;

static esp_codec_dev_handle_t s_spk_codec = NULL;
static esp_codec_dev_handle_t s_mic_codec = NULL;

#define HELMET_AUDIO_SAMPLE_RATE          16000
#define HELMET_AUDIO_BITS                 16
#define HELMET_AUDIO_CHANNELS             1
#define HELMET_AUDIO_VOLUME               60
#define HELMET_MIC_GAIN                   42.0f

/*
 * WakeNet AFE feed size。
 * 你之前日志里 need=512，所以这里固定 512。
 */
#define VOICE_CAPTURE_FRAMES              512
#define VOICE_CAPTURE_INTERVAL_MS         20

/*
 * 唤醒后停止判断。
 * 512 frames @ 16kHz 约 32ms。
 * 35 次约 1.1 秒静音后认为一句话结束。
 */
#define VOICE_VAD_STOP_THRESHOLD          350
#define VOICE_VAD_STOP_COUNT              35

/*
 * 日志降频，避免刷爆串口。
 */
#define VOICE_LOG_EVERY_N_FRAMES          50
#define VOICE_PCM_EVENT_LOG_EVERY_N       25

/*
 * 低内存安全版：
 * 1. 不 malloc
 * 2. 不 calloc
 * 3. 不缓存整句 PCM
 * 4. 不申请 160KB 音频缓存
 *
 * 这个 buffer 是固定静态区，占用约 1024 bytes。
 * 不放在 task 栈里，避免栈压力。
 */
static int16_t s_capture_pcm[VOICE_CAPTURE_FRAMES];

/*
 * 测试 beep buffer 也是静态区。
 * 这版默认启动时不自动播放。
 */
#define TEST_BEEP_FRAMES                  800
#define TEST_BEEP_AMPLITUDE               6000
static int16_t s_beep[TEST_BEEP_FRAMES];

typedef enum {
    VOICE_RUNTIME_IDLE = 0,
    VOICE_RUNTIME_LISTENING,
    VOICE_RUNTIME_PLAYING,
    VOICE_RUNTIME_ERROR,
} voice_runtime_state_t;

typedef enum {
    VOICE_CMD_UNKNOWN = 0,
    VOICE_CMD_CANCEL_ALARM,
    VOICE_CMD_REFRESH_STATUS,
    VOICE_CMD_REPORT_STATUS,
    VOICE_CMD_SOS,
} voice_cmd_t;

static volatile voice_runtime_state_t s_voice_runtime = VOICE_RUNTIME_IDLE;
static volatile bool s_voice_output_busy = false;

/*
 * ESP-SR AFE / WakeNet
 */
static const esp_afe_sr_iface_t *s_afe_handle = NULL;
static esp_afe_sr_data_t *s_afe_data = NULL;
static int s_afe_feed_chunksize = 0;
static int s_afe_feed_channel_num = 0;

/*
 * Forward declarations
 */
static void helmet_voice_handle_text_command(const char *text);
static voice_cmd_t helmet_voice_parse_command_text(const char *text);
static esp_err_t helmet_voice_execute_command(voice_cmd_t cmd);

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

    /*
     * 单麦克风输入。
     * M = microphone channel。
     */
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

static bool helmet_wake_detect_feed(const int16_t *pcm, size_t frames)
{
    if (s_afe_handle == NULL || s_afe_data == NULL || pcm == NULL || frames == 0) {
        return false;
    }

    if ((int)frames != s_afe_feed_chunksize) {
        static int warn_count = 0;

        if ((warn_count++ % 50) == 0) {
            ESP_LOGW(
                TAG,
                "WakeNet feed size mismatch: got=%u need=%d",
                (unsigned)frames,
                s_afe_feed_chunksize
            );
        }

        return false;
    }

    if (s_afe_feed_channel_num != 1) {
        static int channel_warn_count = 0;

        if ((channel_warn_count++ % 50) == 0) {
            ESP_LOGW(
                TAG,
                "WakeNet feed channel mismatch: mono input but AFE need channel_num=%d",
                s_afe_feed_channel_num
            );
        }

        return false;
    }

    s_afe_handle->feed(s_afe_data, pcm);

    afe_fetch_result_t *res = s_afe_handle->fetch_with_delay(
        s_afe_data,
        0
    );

    if (res == NULL) {
        return false;
    }

    if (res->ret_value == ESP_FAIL) {
        ESP_LOGW(TAG, "AFE fetch failed");
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

    if (pcm == NULL || frames == 0) {
        return;
    }

    /*
     * 低内存安全版：
     * 不缓存 PCM。
     * 只在唤醒后打印降频日志，确认链路。
     */
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

static void voice_on_listen_stop(int avg_abs)
{
    s_voice_runtime = VOICE_RUNTIME_IDLE;

    ESP_LOGI(
        TAG,
        "EVENT: LISTEN_STOP avg_abs=%d",
        avg_abs
    );

    /*
     * 低内存安全版：
     * 当前不做 ASR，不缓存音频。
     * 只用假文本打通命令解析链路。
     */
    helmet_voice_handle_text_command("取消报警");
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

            if (!speaking) {
                /*
                 * 未唤醒状态：
                 * 只喂 WakeNet，不打印 PCM，不转发 PCM，不缓存 PCM。
                 */
                if (helmet_wake_detect_feed(s_capture_pcm, frames_read)) {
                    speaking = true;
                    stop_count = 0;
                    log_count = 0;

                    voice_on_listen_start(avg_abs);
                }
            } else {
                /*
                 * 已唤醒状态：
                 * 只做轻量监听和静音结束判断。
                 */
                voice_on_pcm_frame(s_capture_pcm, frames_read);

                if ((log_count++ % VOICE_LOG_EVERY_N_FRAMES) == 0) {
                    voice_capture_print_stats(s_capture_pcm, frames_read, avg_abs);
                }

                if (avg_abs <= VOICE_VAD_STOP_THRESHOLD) {
                    stop_count++;

                    if (stop_count >= VOICE_VAD_STOP_COUNT) {
                        speaking = false;
                        stop_count = 0;
                        log_count = 0;

                        voice_on_listen_stop(avg_abs);

                        /*
                         * 一句话结束后 reset AFE，
                         * 准备下一次唤醒。
                         */
                        helmet_wake_reset();
                    }
                } else {
                    stop_count = 0;
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

    /*
     * s_capture_pcm 已经放到静态区，不占任务栈。
     * 任务栈不用太大。
     */
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

    /*
     * 低内存安全版：
     * 不申请 PCM 缓存。
     * 不 malloc。
     * 不 calloc。
     */

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

    /*
     * helmet_voice_read_pcm() 会检查 s_voice_inited。
     * 启动采集任务前必须置 true。
     */
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

    /*
     * 关键修改：
     * 启动阶段不自动 test_speaker_once。
     * 避免开机阶段抢 I2S / DMA / codec / I2C 资源。
     */
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

    helmet_state_set_voice(HELMET_VOICE_PLAYING);

    esp_err_t ret = test_speaker_once();

    if (ret == ESP_OK) {
        helmet_state_set_voice(HELMET_VOICE_READY);
    } else {
        helmet_state_set_voice(HELMET_VOICE_ERROR);
    }

    return ret;
}

esp_err_t helmet_voice_stop(void)
{
    ESP_LOGI(TAG, "voice stop");

    s_voice_runtime = VOICE_RUNTIME_IDLE;
    s_voice_output_busy = false;

    helmet_state_set_voice(HELMET_VOICE_READY);

    return ESP_OK;
}

/*
 * ============================================================
 * Voice command layer
 * ============================================================
 *
 * 低内存安全版：
 * - 不缓存 PCM
 * - 不做 ASR
 * - 唤醒结束后直接走 fake_text
 * - 只验证命令链路
 */

static void helmet_voice_handle_text_command(const char *text)
{
    if (text == NULL || text[0] == '\0') {
        ESP_LOGW(TAG, "voice command text empty");
        return;
    }

    ESP_LOGI(TAG, "voice command text: %s", text);

    voice_cmd_t cmd = helmet_voice_parse_command_text(text);
    esp_err_t ret = helmet_voice_execute_command(cmd);

    ESP_LOGI(
        TAG,
        "voice command handled: cmd=%d ret=%s",
        (int)cmd,
        esp_err_to_name(ret)
    );
}

static voice_cmd_t helmet_voice_parse_command_text(const char *text)
{
    if (text == NULL) {
        return VOICE_CMD_UNKNOWN;
    }

    if (strstr(text, "取消报警") != NULL ||
        strstr(text, "停止报警") != NULL ||
        strstr(text, "关闭报警") != NULL) {
        return VOICE_CMD_CANCEL_ALARM;
    }

    if (strstr(text, "刷新") != NULL ||
        strstr(text, "查看状态") != NULL ||
        strstr(text, "更新状态") != NULL) {
        return VOICE_CMD_REFRESH_STATUS;
    }

    if (strstr(text, "播报状态") != NULL ||
        strstr(text, "当前状态") != NULL ||
        strstr(text, "什么状态") != NULL) {
        return VOICE_CMD_REPORT_STATUS;
    }

    if (strstr(text, "求救") != NULL ||
        strstr(text, "救命") != NULL ||
        strstr(text, "SOS") != NULL ||
        strstr(text, "sos") != NULL) {
        return VOICE_CMD_SOS;
    }

    return VOICE_CMD_UNKNOWN;
}

static esp_err_t helmet_voice_execute_command(voice_cmd_t cmd)
{
    ESP_LOGI(TAG, "execute voice cmd=%d", (int)cmd);

    switch (cmd) {
    case VOICE_CMD_CANCEL_ALARM:
        ESP_LOGI(TAG, "CMD: CANCEL_ALARM");

        /*
         * 低内存安全版：
         * 不在命令执行里自动播放 beep。
         * 避免唤醒结束后立即抢音频输出。
         *
         * 后面如果你有真正的报警取消函数，
         * 在这里接：
         * helmet_state_cancel_alarm();
         * 或 helmet_alarm_cancel();
         */
        helmet_state_set_voice(HELMET_VOICE_READY);
        return ESP_OK;

    case VOICE_CMD_REFRESH_STATUS:
        ESP_LOGI(TAG, "CMD: REFRESH_STATUS");

        /*
         * TODO:
         * 接你的状态刷新逻辑。
         */
        helmet_state_set_voice(HELMET_VOICE_READY);
        return ESP_OK;

    case VOICE_CMD_REPORT_STATUS:
        ESP_LOGI(TAG, "CMD: REPORT_STATUS");

        /*
         * TODO:
         * 后续再接播报或 UI 提示。
         */
        helmet_state_set_voice(HELMET_VOICE_READY);
        return ESP_OK;

    case VOICE_CMD_SOS:
        ESP_LOGI(TAG, "CMD: SOS");

        /*
         * TODO:
         * 后续接 SOS 上报。
         */
        helmet_state_set_voice(HELMET_VOICE_READY);
        return ESP_OK;

    case VOICE_CMD_UNKNOWN:
    default:
        ESP_LOGW(TAG, "CMD: UNKNOWN");
        helmet_state_set_voice(HELMET_VOICE_READY);
        return ESP_OK;
    }
}