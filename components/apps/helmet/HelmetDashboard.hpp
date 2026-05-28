#pragma once

#include "lvgl.h"
#include "esp_brookesia.hpp"

class HelmetDashboard : public ESP_Brookesia_PhoneApp {
public:
    HelmetDashboard();
    ~HelmetDashboard();

    bool init(void) override;
    bool run(void) override;
    bool back(void) override;
    bool close(void) override;
    bool pause(void) override;
    bool resume(void) override;

private:
    void extraUiInit(void);
    void updateUi(void);
    void feedBootDemoState(void);

    static void onRefreshClicked(lv_event_t *e);
    static void onCancelAlarmClicked(lv_event_t *e);
    static void onTimer(lv_timer_t *timer);

private:
    bool _is_ui_del;
    bool _is_ui_resumed;
    lv_timer_t *_timer;
};