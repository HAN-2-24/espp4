#include "ui_helmet.h"

lv_obj_t *ui_HelmetScreenMain = NULL;

lv_obj_t *ui_LabelRisk = NULL;
lv_obj_t *ui_LabelAlarm = NULL;
lv_obj_t *ui_LabelEye = NULL;
lv_obj_t *ui_LabelPose = NULL;
lv_obj_t *ui_LabelLocation = NULL;
lv_obj_t *ui_LabelVoice = NULL;
lv_obj_t *ui_LabelCloud = NULL;
lv_obj_t *ui_LabelExplanation = NULL;
lv_obj_t *ui_LabelReport = NULL;

lv_obj_t *ui_ButtonRefresh = NULL;
lv_obj_t *ui_ButtonCancelAlarm = NULL;

static lv_obj_t *create_card(lv_obj_t *parent, const char *title, lv_obj_t **value)
{
    lv_obj_t *card = lv_obj_create(parent);
    lv_obj_set_width(card, lv_pct(96));
    lv_obj_set_height(card, 72);
    lv_obj_set_style_radius(card, 12, 0);
    lv_obj_set_style_border_width(card, 0, 0);
    lv_obj_set_style_pad_all(card, 10, 0);
    lv_obj_set_style_bg_color(card, lv_color_hex(0x20242C), 0);

    lv_obj_t *title_label = lv_label_create(card);
    lv_label_set_text(title_label, title);
    lv_obj_set_style_text_color(title_label, lv_color_hex(0xAAB0BC), 0);
    lv_obj_align(title_label, LV_ALIGN_TOP_LEFT, 0, 0);

    *value = lv_label_create(card);
    lv_label_set_text(*value, "--");
    lv_obj_set_width(*value, lv_pct(100));
    lv_label_set_long_mode(*value, LV_LABEL_LONG_DOT);
    lv_obj_set_style_text_color(*value, lv_color_hex(0xFFFFFF), 0);
    lv_obj_align(*value, LV_ALIGN_BOTTOM_LEFT, 0, 0);

    return card;
}

static lv_obj_t *create_button(lv_obj_t *parent, const char *text)
{
    lv_obj_t *btn = lv_btn_create(parent);
    lv_obj_set_size(btn, 210, 54);
    lv_obj_set_style_radius(btn, 14, 0);

    lv_obj_t *label = lv_label_create(btn);
    lv_label_set_text(label, text);
    lv_obj_center(label);

    return btn;
}

void ui_helmet_init(void)
{
    if (ui_HelmetScreenMain != NULL) {
        lv_scr_load(ui_HelmetScreenMain);
        return;
    }

    ui_HelmetScreenMain = lv_obj_create(NULL);
    lv_obj_clear_flag(ui_HelmetScreenMain, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_style_bg_color(ui_HelmetScreenMain, lv_color_hex(0x10131A), 0);
    lv_obj_set_style_bg_opa(ui_HelmetScreenMain, LV_OPA_COVER, 0);

    lv_obj_t *title = lv_label_create(ui_HelmetScreenMain);
    lv_label_set_text(title, "ESP32-P4 Fatigue Helmet");
    lv_obj_set_style_text_color(title, lv_color_hex(0xFFFFFF), 0);
    lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 18);

    ui_LabelRisk = lv_label_create(ui_HelmetScreenMain);
    lv_label_set_text(ui_LabelRisk, "Risk: --");
    lv_obj_set_style_text_color(ui_LabelRisk, lv_color_hex(0xFFFFFF), 0);
    lv_obj_align(ui_LabelRisk, LV_ALIGN_TOP_LEFT, 24, 58);

    ui_LabelAlarm = lv_label_create(ui_HelmetScreenMain);
    lv_label_set_text(ui_LabelAlarm, "Alarm: --");
    lv_obj_set_style_text_color(ui_LabelAlarm, lv_color_hex(0xFFFFFF), 0);
    lv_obj_align(ui_LabelAlarm, LV_ALIGN_TOP_RIGHT, -24, 58);

    lv_obj_t *container = lv_obj_create(ui_HelmetScreenMain);
    lv_obj_set_width(container, lv_pct(94));
    lv_obj_set_height(container, lv_pct(68));
    lv_obj_align(container, LV_ALIGN_TOP_MID, 0, 92);
    lv_obj_set_style_bg_opa(container, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(container, 0, 0);
    lv_obj_set_style_pad_all(container, 0, 0);
    lv_obj_set_style_pad_row(container, 8, 0);
    lv_obj_set_flex_flow(container, LV_FLEX_FLOW_COLUMN);
    lv_obj_set_flex_align(container, LV_FLEX_ALIGN_START, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);

    create_card(container, "Eye Vision", &ui_LabelEye);
    create_card(container, "Pose / IMU", &ui_LabelPose);
    create_card(container, "BeiDou / GNSS", &ui_LabelLocation);
    create_card(container, "Onboard Voice", &ui_LabelVoice);
    create_card(container, "Cloud LLM", &ui_LabelCloud);
    create_card(container, "State Explanation", &ui_LabelExplanation);
    create_card(container, "Fatigue Report", &ui_LabelReport);

    ui_ButtonRefresh = create_button(ui_HelmetScreenMain, "Refresh");
    lv_obj_align(ui_ButtonRefresh, LV_ALIGN_BOTTOM_LEFT, 36, -24);

    ui_ButtonCancelAlarm = create_button(ui_HelmetScreenMain, "Cancel Alarm");
    lv_obj_align(ui_ButtonCancelAlarm, LV_ALIGN_BOTTOM_RIGHT, -36, -24);

    lv_scr_load(ui_HelmetScreenMain);
}