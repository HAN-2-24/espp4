#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "lvgl.h"

extern lv_obj_t *ui_HelmetScreenMain;

extern lv_obj_t *ui_LabelRisk;
extern lv_obj_t *ui_LabelAlarm;
extern lv_obj_t *ui_LabelEye;
extern lv_obj_t *ui_LabelPose;
extern lv_obj_t *ui_LabelLocation;
extern lv_obj_t *ui_LabelVoice;
extern lv_obj_t *ui_LabelCloud;
extern lv_obj_t *ui_LabelExplanation;
extern lv_obj_t *ui_LabelReport;

extern lv_obj_t *ui_ButtonRefresh;
extern lv_obj_t *ui_ButtonCancelAlarm;

void ui_helmet_init(void);

#ifdef __cplusplus
}
#endif