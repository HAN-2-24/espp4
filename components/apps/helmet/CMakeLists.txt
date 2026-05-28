file(GLOB_RECURSE HELMET_APP_SRCS
    "*.c"
    "*.cpp"
)

idf_component_register(
    SRCS
        ${HELMET_APP_SRCS}
    INCLUDE_DIRS
        "."
        "ui"
    REQUIRES
        helmet_board
        helmet_core
        helmet_drivers
        lvgl
        esp_brookesia
)