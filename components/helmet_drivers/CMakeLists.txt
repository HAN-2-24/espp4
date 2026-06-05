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
        esp_event
        esp_netif
        esp_wifi
        espressif__esp_hosted
        espressif__esp-sr
        esp32_p4_function_ev_board_noglib
    EMBED_TXTFILES
        "certs/emqx_ca.pem"
)
