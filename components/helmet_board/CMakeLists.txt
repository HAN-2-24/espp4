file(GLOB_RECURSE HELMET_BOARD_SRCS
    "src/*.c"
    "src/*.cpp"
)

idf_component_register(
    SRCS
        ${HELMET_BOARD_SRCS}
    INCLUDE_DIRS
        "include"
    REQUIRES
        driver
	esp_lcd
)