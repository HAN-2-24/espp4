#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "esp_err.h"

typedef struct {
    int valid;
    double latitude;
    double longitude;
    float speed;
    float course;
} helmet_gnss_data_t;

esp_err_t helmet_gnss_init(void);
esp_err_t helmet_gnss_poll_once(void);
int helmet_gnss_parse_rmc(const char *sentence, helmet_gnss_data_t *out);

#ifdef __cplusplus
}
#endif