file(GLOB_RECURSE HELMET_DRIVER_SRCS
    "src/*.c"
    "src/*.cpp"
)

idf_component_register(
    SRCS
        ${HELMET_DRIVER_SRCS}
    INCLUDE_DIRS
        "include"
    REQUIRES
        driver
        esp_timer
        helmet_board
        helmet_core
	mqtt
        esp_wifi
        esp_event
        esp_netif
	esp32_p4_function_ev_board_noglib
)
