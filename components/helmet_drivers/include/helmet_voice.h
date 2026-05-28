#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "esp_err.h"

#include <stdint.h>
#include <stddef.h>

typedef enum {
    HELMET_VOICE_ALERT_FATIGUE = 0,
    HELMET_VOICE_ALERT_DANGER,
    HELMET_VOICE_ALERT_GNSS_LOST,
} helmet_voice_alert_t;

/*
 * 初始化语音模块。
 *
 * 当前实现：
 *   1. 初始化 ES8311 / I2S
 *   2. 设置 16kHz / 16bit / stereo
 *   3. 设置音量并取消静音
 *   4. 播放一次 beep
 *   5. 做一次麦克风 smoke test
 *   6. 启动 voice_capture_task
 */
esp_err_t helmet_voice_init(void);

/*
 * 播放报警音。
 *
 * 当前测试阶段复用 beep。
 */
esp_err_t helmet_voice_play_alert(helmet_voice_alert_t alert);

/*
 * 停止语音播放。
 *
 * 当前测试阶段只恢复状态。
 */
esp_err_t helmet_voice_stop(void);

/*
 * 读取麦克风 PCM。
 *
 * 输出格式：
 *   16000 Hz
 *   16-bit signed PCM
 *   mono
 *
 * 内部实际做：
 *   stereo L/R -> 取 left -> mono
 */
esp_err_t helmet_voice_read_pcm(int16_t *mono_buffer, size_t frames, size_t *frames_read);

#ifdef __cplusplus
}
#endif
