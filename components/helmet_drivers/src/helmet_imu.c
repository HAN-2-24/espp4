#include "helmet_imu.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "esp_log.h"
#include "esp_err.h"

#include "driver/i2c_master.h"
#include "bsp/esp32_p4_function_ev_board.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "helmet_board.h"
#include "helmet_state.h"
#include "helmet_voice.h"

static const char *TAG = "helmet_imu";

#define MPU6050_ADDR_0               0x68
#define MPU6050_ADDR_1               0x69

#define MPU_SAMPLE_RATE_REG          0x19
#define MPU_CFG_REG                  0x1A
#define MPU_GYRO_CFG_REG             0x1B
#define MPU_ACCEL_CFG_REG            0x1C
#define MPU_FIFO_EN_REG              0x23
#define MPU_INTBP_CFG_REG            0x37
#define MPU_INT_EN_REG               0x38
#define MPU_ACCEL_XOUTH_REG          0x3B
#define MPU_GYRO_XOUTH_REG           0x43
#define MPU_USER_CTRL_REG            0x6A
#define MPU_PWR_MGMT1_REG            0x6B
#define MPU_PWR_MGMT2_REG            0x6C
#define MPU_DEVICE_ID_REG            0x75

#define IMU_TASK_STACK_SIZE          4096
#define IMU_TASK_PRIORITY            5
#define IMU_SAMPLE_PERIOD_MS         10
#define IMU_CALIBRATION_SAMPLES      2000

#define PI_F                         3.14159265358979f
#define RAD2DEG_F                    57.29578f
#define DEG2RAD_F                    0.017453292519943295f
#define GRAVITY_F                    9.80665f

#define IMU_DEBUG_LOG                1

#define IMU_ACCIDENT_ALARM_COOLDOWN_MS 10000
#define IMU_ACCIDENT_ALARM_RETRY_MS    1000

#ifndef HELMET_I2C_FREQ_HZ
#define HELMET_I2C_FREQ_HZ           100000
#endif

typedef struct {
    float x;
    float y;
    float z;
} imu_vec3_t;

typedef struct {
    float q[3];
    float r[3];
    float x_est[3];
    float p[3];
    float k[3];
} imu_kalman3d_t;

static TaskHandle_t s_imu_task_handle = NULL;
static bool s_imu_inited = false;

static i2c_master_bus_handle_t s_i2c_bus = NULL;
static i2c_master_dev_handle_t s_mpu_dev = NULL;
static uint8_t s_mpu_addr = MPU6050_ADDR_0;

static imu_vec3_t s_acc_offset = {0};
static float s_acc_gravity_sign = 1.0f;
static imu_vec3_t s_gyr_offset = {0};

static double s_roll_v = 0.0;
static double s_pitch_v = 0.0;
static double s_gyro_roll = 0.0;
static double s_gyro_pitch = 0.0;
static double s_acc_roll = 0.0;
static double s_acc_pitch = 0.0;
static double s_k_roll = 0.0;
static double s_k_pitch = 0.0;

static double s_e_p[2][2] = {
    {1, 0},
    {0, 1}
};

static double s_k_k[2][2] = {
    {0, 0},
    {0, 0}
};

static imu_kalman3d_t s_acc_body_kf;

static float sqf_local(float x)
{
    return x * x;
}

static void imu_kalman3d_init(
    imu_kalman3d_t *kf,
    const float q[3],
    const float r[3],
    const float x_init[3],
    const float p_init[3]
)
{
    if (kf == NULL) {
        return;
    }

    for (int i = 0; i < 3; i++) {
        kf->q[i] = q[i];
        kf->r[i] = r[i];
        kf->x_est[i] = x_init[i];
        kf->p[i] = p_init[i];
        kf->k[i] = 0.0f;
    }
}

static void imu_kalman3d_update(
    imu_kalman3d_t *kf,
    const imu_vec3_t *measurement,
    imu_vec3_t *out
)
{
    if (kf == NULL || measurement == NULL || out == NULL) {
        return;
    }

    float p_predict[3];

    p_predict[0] = kf->p[0] + kf->q[0];
    p_predict[1] = kf->p[1] + kf->q[1];
    p_predict[2] = kf->p[2] + kf->q[2];

    kf->k[0] = p_predict[0] / (p_predict[0] + kf->r[0]);
    kf->k[1] = p_predict[1] / (p_predict[1] + kf->r[1]);
    kf->k[2] = p_predict[2] / (p_predict[2] + kf->r[2]);

    kf->x_est[0] = kf->x_est[0] + kf->k[0] * (measurement->x - kf->x_est[0]);
    kf->x_est[1] = kf->x_est[1] + kf->k[1] * (measurement->y - kf->x_est[1]);
    kf->x_est[2] = kf->x_est[2] + kf->k[2] * (measurement->z - kf->x_est[2]);

    kf->p[0] = (1.0f - kf->k[0]) * p_predict[0];
    kf->p[1] = (1.0f - kf->k[1]) * p_predict[1];
    kf->p[2] = (1.0f - kf->k[2]) * p_predict[2];

    out->x = kf->x_est[0];
    out->y = kf->x_est[1];
    out->z = kf->x_est[2];
}

/*
 * I2C 修复点：
 *
 * 保留完整 IMU 算法版，只把 I2C bus 获取改为 BSP bus。
 * 不在这里创建自定义 bus，不改算法，不改任务，不改采样周期。
 */
static esp_err_t imu_i2c_bus_init_if_needed(void)
{
    if (s_i2c_bus != NULL) {
        return ESP_OK;
    }

    ESP_LOGI(
        TAG,
        "use BSP I2C: BSP_I2C_NUM=%d SDA=%d SCL=%d",
        BSP_I2C_NUM,
        BSP_I2C_SDA,
        BSP_I2C_SCL
    );

    esp_err_t ret = bsp_i2c_init();

    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "bsp_i2c_init failed: %s", esp_err_to_name(ret));
        s_i2c_bus = NULL;
        return ret;
    }

    s_i2c_bus = bsp_i2c_get_handle();

    if (s_i2c_bus == NULL) {
        ESP_LOGE(TAG, "bsp_i2c_get_handle returned NULL");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "BSP I2C bus handle ready");

    return ESP_OK;
}

static esp_err_t imu_add_i2c_device(uint8_t addr)
{
    if (s_i2c_bus == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    if (s_mpu_dev != NULL) {
        i2c_master_bus_rm_device(s_mpu_dev);
        s_mpu_dev = NULL;
    }

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = addr,
        .scl_speed_hz = HELMET_I2C_FREQ_HZ,
    };

    esp_err_t ret = i2c_master_bus_add_device(
        s_i2c_bus,
        &dev_cfg,
        &s_mpu_dev
    );

    if (ret != ESP_OK) {
        ESP_LOGE(
            TAG,
            "add MPU6050 I2C device 0x%02X failed: %s",
            addr,
            esp_err_to_name(ret)
        );
        s_mpu_dev = NULL;
        return ret;
    }

    s_mpu_addr = addr;
    return ESP_OK;
}

static esp_err_t imu_write_byte(uint8_t reg, uint8_t data)
{
    if (s_mpu_dev == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    uint8_t buf[2] = {reg, data};

    return i2c_master_transmit(
        s_mpu_dev,
        buf,
        sizeof(buf),
        pdMS_TO_TICKS(100)
    );
}

static esp_err_t imu_read_len(uint8_t reg, uint8_t *buf, size_t len)
{
    if (s_mpu_dev == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    if (buf == NULL || len == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    return i2c_master_transmit_receive(
        s_mpu_dev,
        &reg,
        1,
        buf,
        len,
        pdMS_TO_TICKS(100)
    );
}

static esp_err_t imu_read_byte(uint8_t reg, uint8_t *data)
{
    return imu_read_len(reg, data, 1);
}

static esp_err_t imu_probe_mpu6050(uint8_t *who_am_i_out)
{
    uint8_t who = 0;
    esp_err_t ret;

    ret = imu_add_i2c_device(MPU6050_ADDR_0);
    if (ret == ESP_OK) {
        ret = imu_read_byte(MPU_DEVICE_ID_REG, &who);
        if (ret == ESP_OK && (who == 0x68 || who == 0x69)) {
            if (who_am_i_out != NULL) {
                *who_am_i_out = who;
            }

            ESP_LOGI(
                TAG,
                "MPU6050 found at addr=0x%02X WHO_AM_I=0x%02X",
                MPU6050_ADDR_0,
                who
            );
            return ESP_OK;
        }
    }

    ESP_LOGW(
        TAG,
        "MPU6050 not found at 0x%02X, try 0x%02X",
        MPU6050_ADDR_0,
        MPU6050_ADDR_1
    );

    ret = imu_add_i2c_device(MPU6050_ADDR_1);
    if (ret == ESP_OK) {
        ret = imu_read_byte(MPU_DEVICE_ID_REG, &who);
        if (ret == ESP_OK && (who == 0x68 || who == 0x69)) {
            if (who_am_i_out != NULL) {
                *who_am_i_out = who;
            }

            ESP_LOGI(
                TAG,
                "MPU6050 found at addr=0x%02X WHO_AM_I=0x%02X",
                MPU6050_ADDR_1,
                who
            );
            return ESP_OK;
        }
    }

    ESP_LOGE(TAG, "MPU6050 probe failed");
    return ESP_FAIL;
}

static esp_err_t imu_set_gyro_fsr(uint8_t fsr)
{
    return imu_write_byte(MPU_GYRO_CFG_REG, fsr << 3);
}

static esp_err_t imu_set_accel_fsr(uint8_t fsr)
{
    return imu_write_byte(MPU_ACCEL_CFG_REG, fsr << 3);
}

static esp_err_t imu_set_lpf(uint16_t lpf)
{
    uint8_t data = 0;

    if (lpf >= 188) {
        data = 1;
    } else if (lpf >= 98) {
        data = 2;
    } else if (lpf >= 42) {
        data = 3;
    } else if (lpf >= 20) {
        data = 4;
    } else if (lpf >= 10) {
        data = 5;
    } else {
        data = 6;
    }

    return imu_write_byte(MPU_CFG_REG, data);
}

static esp_err_t imu_set_rate(uint16_t rate)
{
    if (rate > 1000) {
        rate = 1000;
    }

    if (rate < 4) {
        rate = 4;
    }

    uint8_t data = (uint8_t)(1000 / rate - 1);

    esp_err_t ret = imu_write_byte(MPU_SAMPLE_RATE_REG, data);
    if (ret != ESP_OK) {
        return ret;
    }

    return imu_set_lpf(rate / 2);
}

static esp_err_t imu_mpu6050_init_device(void)
{
    esp_err_t ret;

    ret = imu_write_byte(MPU_PWR_MGMT1_REG, 0x80);
    if (ret != ESP_OK) {
        return ret;
    }

    vTaskDelay(pdMS_TO_TICKS(100));

    ret = imu_write_byte(MPU_PWR_MGMT1_REG, 0x00);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = imu_set_gyro_fsr(3);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = imu_set_accel_fsr(3);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = imu_set_rate(250);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = imu_write_byte(MPU_INT_EN_REG, 0x00);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = imu_write_byte(MPU_USER_CTRL_REG, 0x00);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = imu_write_byte(MPU_FIFO_EN_REG, 0x00);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = imu_write_byte(MPU_INTBP_CFG_REG, 0x80);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = imu_write_byte(MPU_PWR_MGMT1_REG, 0x01);
    if (ret != ESP_OK) {
        return ret;
    }

    ret = imu_write_byte(MPU_PWR_MGMT2_REG, 0x00);
    if (ret != ESP_OK) {
        return ret;
    }

    return ESP_OK;
}

static esp_err_t imu_get_gyroscope_raw(short *gx, short *gy, short *gz)
{
    if (gx == NULL || gy == NULL || gz == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t buf[6];

    esp_err_t ret = imu_read_len(MPU_GYRO_XOUTH_REG, buf, 6);
    if (ret != ESP_OK) {
        return ret;
    }

    int16_t raw_x = (int16_t)(((uint16_t)buf[0] << 8) | buf[1]);
    int16_t raw_y = (int16_t)(((uint16_t)buf[2] << 8) | buf[3]);
    int16_t raw_z = (int16_t)(((uint16_t)buf[4] << 8) | buf[5]);

    *gy = (short)(-raw_x);
    *gz = (short)(-raw_y);
    *gx = (short)( raw_z);

    return ESP_OK;
}

static esp_err_t imu_get_accelerometer_raw(short *ax, short *ay, short *az)
{
    if (ax == NULL || ay == NULL || az == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t buf[6];

    esp_err_t ret = imu_read_len(MPU_ACCEL_XOUTH_REG, buf, 6);
    if (ret != ESP_OK) {
        return ret;
    }

    int16_t raw_x = (int16_t)(((uint16_t)buf[0] << 8) | buf[1]);
    int16_t raw_y = (int16_t)(((uint16_t)buf[2] << 8) | buf[3]);
    int16_t raw_z = (int16_t)(((uint16_t)buf[4] << 8) | buf[5]);

    *ay = (short)(-raw_x);
    *az = (short)(-raw_y);
    *ax = (short)( raw_z);

    return ESP_OK;
}

static esp_err_t imu_calibrate_offsets(void)
{
    int64_t ax_sum = 0;
    int64_t ay_sum = 0;
    int64_t az_sum = 0;

    int64_t gx_sum = 0;
    int64_t gy_sum = 0;
    int64_t gz_sum = 0;

    short ax = 0;
    short ay = 0;
    short az = 0;

    short gx = 0;
    short gy = 0;
    short gz = 0;

    ESP_LOGI(TAG, "MPU6050 calibrating offsets, samples=%d", IMU_CALIBRATION_SAMPLES);

    for (int i = 0; i < IMU_CALIBRATION_SAMPLES; i++) {
        esp_err_t ret1 = imu_get_gyroscope_raw(&gx, &gy, &gz);
        esp_err_t ret2 = imu_get_accelerometer_raw(&ax, &ay, &az);

        if (ret1 != ESP_OK || ret2 != ESP_OK) {
            ESP_LOGE(
                TAG,
                "calibration read failed: gyro=%s acc=%s",
                esp_err_to_name(ret1),
                esp_err_to_name(ret2)
            );
            return ESP_FAIL;
        }

        ax_sum += ax;
        ay_sum += ay;
        az_sum += az;

        gx_sum += gx;
        gy_sum += gy;
        gz_sum += gz;

        vTaskDelay(pdMS_TO_TICKS(1));
    }

    int32_t ax_offset_raw = (int32_t)(ax_sum / IMU_CALIBRATION_SAMPLES);
    int32_t ay_offset_raw = (int32_t)(ay_sum / IMU_CALIBRATION_SAMPLES);
    int32_t az_mean_raw = (int32_t)(az_sum / IMU_CALIBRATION_SAMPLES);
    int32_t az_expected_raw = (az_mean_raw < 0) ? -2048 : 2048;
    int32_t az_offset_raw = az_mean_raw - az_expected_raw;

    s_acc_gravity_sign = (az_expected_raw < 0) ? -1.0f : 1.0f;

    int32_t gx_offset_raw = (int32_t)(gx_sum / IMU_CALIBRATION_SAMPLES);
    int32_t gy_offset_raw = (int32_t)(gy_sum / IMU_CALIBRATION_SAMPLES);
    int32_t gz_offset_raw = (int32_t)(gz_sum / IMU_CALIBRATION_SAMPLES);

    s_acc_offset.x = 9.8f * 16.0f * (float)ax_offset_raw / 32768.0f;
    s_acc_offset.y = 9.8f * 16.0f * (float)ay_offset_raw / 32768.0f;
    s_acc_offset.z = 9.8f * 16.0f * (float)az_offset_raw / 32768.0f;

    s_gyr_offset.x = 2000.0f * (float)gx_offset_raw / 32768.0f;
    s_gyr_offset.y = 2000.0f * (float)gy_offset_raw / 32768.0f;
    s_gyr_offset.z = 2000.0f * (float)gz_offset_raw / 32768.0f;

#if IMU_DEBUG_LOG
    ESP_LOGI(
        TAG,
        "offset acc=(%.3f, %.3f, %.3f) gyr=(%.3f, %.3f, %.3f)",
        s_acc_offset.x,
        s_acc_offset.y,
        s_acc_offset.z,
        s_gyr_offset.x,
        s_gyr_offset.y,
        s_gyr_offset.z
    );
#endif

    return ESP_OK;
}

static void imu_remove_gravity_acceleration(
    const imu_vec3_t *acc_raw,
    double roll,
    double pitch,
    imu_vec3_t *acc_body
)
{
    if (acc_raw == NULL || acc_body == NULL) {
        return;
    }

    const float g = 9.81f;

    float gx = g * sinf((float)pitch * (PI_F / 180.0f));
    float gy = g * cosf((float)pitch * (PI_F / 180.0f)) * sinf((float)roll * (PI_F / 180.0f));
    float gz = g * cosf((float)pitch * (PI_F / 180.0f)) * cosf((float)roll * (PI_F / 180.0f));

    gx *= s_acc_gravity_sign;
    gy *= s_acc_gravity_sign;
    gz *= s_acc_gravity_sign;

    acc_body->x = acc_raw->x + gx;
    acc_body->y = acc_raw->y - gy;
    acc_body->z = acc_raw->z - gz;
}

static void imu_update_attitude(
    const imu_vec3_t *real_acc,
    const imu_vec3_t *real_gyr,
    double dt
)
{
    if (real_acc == NULL || real_gyr == NULL) {
        return;
    }

    if (dt <= 0.0 || dt > 1.0) {
        dt = (double)IMU_SAMPLE_PERIOD_MS / 1000.0;
    }

    double roll_rad = s_k_roll * DEG2RAD_F;
    double pitch_rad = s_k_pitch * DEG2RAD_F;
    double cos_pitch = cos(pitch_rad);

    if (fabs(cos_pitch) < 1e-6) {
        cos_pitch = (cos_pitch < 0.0) ? -1e-6 : 1e-6;
    }

    s_roll_v =
        real_gyr->x +
        ((sin(pitch_rad) * sin(roll_rad)) / cos_pitch) * real_gyr->y +
        ((sin(pitch_rad) * cos(roll_rad)) / cos_pitch) * real_gyr->z;

    s_pitch_v =
        cos(roll_rad) * real_gyr->y -
        sin(roll_rad) * real_gyr->z;

    s_gyro_roll = s_k_roll + dt * s_roll_v;
    s_gyro_pitch = s_k_pitch + dt * s_pitch_v;

    s_e_p[0][0] = s_e_p[0][0] + 0.0025;
    s_e_p[0][1] = s_e_p[0][1] + 0.0;
    s_e_p[1][0] = s_e_p[1][0] + 0.0;
    s_e_p[1][1] = s_e_p[1][1] + 0.0025;

    s_k_k[0][0] = s_e_p[0][0] / (s_e_p[0][0] + 0.3);
    s_k_k[0][1] = 0.0;
    s_k_k[1][0] = 0.0;
    s_k_k[1][1] = s_e_p[1][1] / (s_e_p[1][1] + 0.3);

    double acc_x = (double)s_acc_gravity_sign * (double)real_acc->x;
    double acc_y = (double)s_acc_gravity_sign * (double)real_acc->y;
    double acc_z = (double)s_acc_gravity_sign * (double)real_acc->z;

    s_acc_roll = atan2(acc_y, acc_z) * RAD2DEG_F;

    s_acc_pitch =
        -1.0 *
        atan2(acc_x, sqrt((acc_y * acc_y) + (acc_z * acc_z))) *
        RAD2DEG_F;

    s_k_roll = s_gyro_roll + s_k_k[0][0] * (s_acc_roll - s_gyro_roll);
    s_k_pitch = s_gyro_pitch + s_k_k[1][1] * (s_acc_pitch - s_gyro_pitch);

    s_e_p[0][0] = (1.0 - s_k_k[0][0]) * s_e_p[0][0];
    s_e_p[0][1] = 0.0;
    s_e_p[1][0] = 0.0;
    s_e_p[1][1] = (1.0 - s_k_k[1][1]) * s_e_p[1][1];
}

static void imu_check_accident(
    const imu_vec3_t *kelman_answer,
    bool *fall_detected,
    bool *impact_detected
)
{
    static uint32_t fall_cont = 0;
    static uint32_t abnormal_x_count = 0;
    static uint32_t abnormal_y_count = 0;

    static bool accident_latched = false;

    if (fall_detected != NULL) {
        *fall_detected = false;
    }

    if (impact_detected != NULL) {
        *impact_detected = false;
    }

    if (kelman_answer == NULL) {
        return;
    }

    if (s_k_roll >= 45.0 || s_k_roll <= -45.0) {
        fall_cont++;

        if (fall_cont >= 100) {
            accident_latched = true;

            if (fall_detected != NULL) {
                *fall_detected = true;
            }
        }
    } else {
        fall_cont = 0;
    }

    if (kelman_answer->x >= 1.0f * GRAVITY_F) {
        abnormal_x_count++;

        if (abnormal_x_count >= 15 && accident_latched) {
            if (impact_detected != NULL) {
                *impact_detected = true;
            }

            abnormal_x_count = 0;
            accident_latched = false;
        }
    } else {
        abnormal_x_count = 0;
    }

    if (kelman_answer->y >= 1.0f * GRAVITY_F) {
        abnormal_y_count++;

        if (abnormal_y_count >= 15 && accident_latched) {
            if (impact_detected != NULL) {
                *impact_detected = true;
            }

            abnormal_y_count = 0;
            accident_latched = false;
        }
    } else {
        abnormal_y_count = 0;
    }
}

static esp_err_t imu_read_and_update_once(double dt)
{
    static bool alarm_latched = false;
    static TickType_t last_alarm_tick = 0;
    static TickType_t last_alarm_attempt_tick = 0;

    short ax_raw = 0;
    short ay_raw = 0;
    short az_raw = 0;

    short gx_raw = 0;
    short gy_raw = 0;
    short gz_raw = 0;

    esp_err_t ret1 = imu_get_gyroscope_raw(&gx_raw, &gy_raw, &gz_raw);
    esp_err_t ret2 = imu_get_accelerometer_raw(&ax_raw, &ay_raw, &az_raw);

    if (ret1 != ESP_OK || ret2 != ESP_OK) {
        ESP_LOGE(
            TAG,
            "read raw failed: gyro=%s acc=%s",
            esp_err_to_name(ret1),
            esp_err_to_name(ret2)
        );
        return ESP_FAIL;
    }

    imu_vec3_t acc_measure;
    imu_vec3_t gyr_measure;
    imu_vec3_t real_acc;
    imu_vec3_t real_gyr;
    imu_vec3_t real_acc_body;
    imu_vec3_t kelman_answer;

    acc_measure.x = 9.8f * 16.0f * (float)ax_raw / 32768.0f;
    acc_measure.y = 9.8f * 16.0f * (float)ay_raw / 32768.0f;
    acc_measure.z = 9.8f * 16.0f * (float)az_raw / 32768.0f;

    gyr_measure.x = 2000.0f * (float)gx_raw / 32768.0f;
    gyr_measure.y = 2000.0f * (float)gy_raw / 32768.0f;
    gyr_measure.z = 2000.0f * (float)gz_raw / 32768.0f;

    real_acc.x = acc_measure.x - s_acc_offset.x;
    real_acc.y = acc_measure.y - s_acc_offset.y;
    real_acc.z = acc_measure.z - s_acc_offset.z;

    real_gyr.x = gyr_measure.x - s_gyr_offset.x;
    real_gyr.y = gyr_measure.y - s_gyr_offset.y;
    real_gyr.z = gyr_measure.z - s_gyr_offset.z;

    imu_update_attitude(&real_acc, &real_gyr, dt);

    imu_remove_gravity_acceleration(
        &real_acc,
        s_k_roll,
        s_k_pitch,
        &real_acc_body
    );

    imu_kalman3d_update(
        &s_acc_body_kf,
        &real_acc_body,
        &kelman_answer
    );

    float acc_all = sqrtf(
        sqf_local(kelman_answer.x) +
        sqf_local(kelman_answer.y) +
        sqf_local(kelman_answer.z)
    );

    bool fall_detected = false;
    bool impact_detected = false;

    imu_check_accident(
        &kelman_answer,
        &fall_detected,
        &impact_detected
    );

    helmet_state_set_pose(
        (float)s_k_roll,
        (float)s_k_pitch,
        acc_all,
        fall_detected,
        impact_detected
    );

    bool accident_detected = fall_detected || impact_detected;

    if (accident_detected && !alarm_latched) {
        TickType_t now = xTaskGetTickCount();
        TickType_t cooldown = pdMS_TO_TICKS(IMU_ACCIDENT_ALARM_COOLDOWN_MS);
        TickType_t retry_interval = pdMS_TO_TICKS(IMU_ACCIDENT_ALARM_RETRY_MS);
        bool can_attempt = (last_alarm_attempt_tick == 0) ||
                           ((now - last_alarm_attempt_tick) >= retry_interval);
        bool cooldown_elapsed = (last_alarm_tick == 0) ||
                                ((now - last_alarm_tick) >= cooldown);

        if (can_attempt && cooldown_elapsed) {
            helmet_state_request_alarm();
            esp_err_t voice_ret = helmet_voice_play_alert(HELMET_VOICE_ALERT_DANGER);
            last_alarm_attempt_tick = now;

            ESP_LOGW(
                TAG,
                "accident alarm triggered: fall=%d impact=%d roll=%.2f pitch=%.2f voice=%s",
                fall_detected ? 1 : 0,
                impact_detected ? 1 : 0,
                (float)s_k_roll,
                (float)s_k_pitch,
                esp_err_to_name(voice_ret)
            );

            if (voice_ret == ESP_OK) {
                last_alarm_tick = now;
                alarm_latched = true;
            }
        }
    } else if (!accident_detected) {
        alarm_latched = false;
    }

#if IMU_DEBUG_LOG
    static uint32_t log_count = 0;
    log_count++;

    /*
     * IMU_SAMPLE_PERIOD_MS = 10ms
     * 200 samples = 2000ms
     *
     * 只节流日志，不改变采样周期、卡尔曼更新频率、事故检测计数时间基准。
     */
    if ((log_count % 200) == 0) {
        ESP_LOGI(
            TAG,
            "raw acc=(%d,%d,%d) gyr=(%d,%d,%d) roll=%.2f pitch=%.2f acc=%.2f fall=%d impact=%d",
            ax_raw,
            ay_raw,
            az_raw,
            gx_raw,
            gy_raw,
            gz_raw,
            (float)s_k_roll,
            (float)s_k_pitch,
            acc_all,
            fall_detected ? 1 : 0,
            impact_detected ? 1 : 0
        );
    }
#endif

    return ESP_OK;
}

static void helmet_imu_task(void *arg)
{
    (void)arg;

    TickType_t last_wake = xTaskGetTickCount();
    TickType_t last_tick = xTaskGetTickCount();

    ESP_LOGI(TAG, "IMU task started");

    while (1) {
        TickType_t now_tick = xTaskGetTickCount();
        double dt = ((double)(now_tick - last_tick) * (double)portTICK_PERIOD_MS) / 1000.0;
        last_tick = now_tick;

        esp_err_t ret = imu_read_and_update_once(dt);
        if (ret == ESP_OK) {
            helmet_state_update_fusion();
        }

        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(IMU_SAMPLE_PERIOD_MS));
    }
}

esp_err_t helmet_imu_init(void)
{
    if (s_imu_inited) {
        return ESP_OK;
    }

    esp_err_t ret;

    ret = imu_i2c_bus_init_if_needed();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "IMU I2C bus not ready");
        return ret;
    }

    uint8_t who_am_i = 0;

    ret = imu_probe_mpu6050(&who_am_i);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "MPU6050 probe failed");
        return ret;
    }

    ESP_LOGI(
        TAG,
        "MPU6050 using addr=0x%02X WHO_AM_I=0x%02X",
        s_mpu_addr,
        who_am_i
    );

    ret = imu_mpu6050_init_device();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "MPU6050 init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    vTaskDelay(pdMS_TO_TICKS(100));

    ret = imu_calibrate_offsets();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "MPU6050 calibration failed");
        return ret;
    }

    float q[3] = {0.05f, 0.05f, 0.05f};
    float r[3] = {0.1f, 0.1f, 0.1f};
    float x_init[3] = {0.0f, 0.0f, 0.0f};
    float p_init[3] = {0.1f, 0.1f, 0.1f};

    imu_kalman3d_init(
        &s_acc_body_kf,
        q,
        r,
        x_init,
        p_init
    );

    if (s_imu_task_handle == NULL) {
        BaseType_t task_ret = xTaskCreate(
            helmet_imu_task,
            "helmet_imu",
            IMU_TASK_STACK_SIZE,
            NULL,
            IMU_TASK_PRIORITY,
            &s_imu_task_handle
        );

        if (task_ret != pdPASS) {
            ESP_LOGE(TAG, "create IMU task failed");
            s_imu_task_handle = NULL;
            return ESP_FAIL;
        }
    }

    s_imu_inited = true;

    ESP_LOGI(TAG, "MPU6050 init done");

    return ESP_OK;
}

esp_err_t helmet_imu_poll_once(void)
{
    /*
     * 正式采样由 helmet_imu_task 后台执行。
     * 保留这个接口是为了兼容外部调用，不在这里重复采样，避免破坏 10ms 后台算法节奏。
     */
    return ESP_OK;
}

esp_err_t helmet_imu_poll(void)
{
    return helmet_imu_poll_once();
}
