file(GLOB_RECURSE HELMET_DRIVER_SRCS
    "src/*.c"
    "src/*.cpp"
)

idf_component_register(
    SRCS
        ${HELMET_DRIVER_SRCS}
    INCLUDE_DIRS
        "include"
    PRIV_INCLUDE_DIRS
        "private_include"
    REQUIRES
        driver
        esp-dl
        esp_tinyusb
        esp_timer
        esp_video
        nvs_flash
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

set(EYE_STATE_MODEL_PATH "${COMPONENT_DIR}/models/esp32p4/eye_state_detect_s8_v1.espdl")
if(EXISTS "${EYE_STATE_MODEL_PATH}")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
        HELMET_EYE_STATE_MODEL_AVAILABLE=1
        HELMET_EYE_STATE_MODEL_EMBEDDED=0)

    add_custom_target(helmet_eye_state_model ALL DEPENDS "${EYE_STATE_MODEL_PATH}")
    add_dependencies(flash helmet_eye_state_model)

    partition_table_get_partition_info(EYE_STATE_MODEL_PARTITION_SIZE "--partition-name eye_model" "size")
    partition_table_get_partition_info(EYE_STATE_MODEL_PARTITION_OFFSET "--partition-name eye_model" "offset")
    if("${EYE_STATE_MODEL_PARTITION_SIZE}" AND "${EYE_STATE_MODEL_PARTITION_OFFSET}")
        file(SIZE "${EYE_STATE_MODEL_PATH}" EYE_STATE_MODEL_FILE_SIZE)
        if(EYE_STATE_MODEL_FILE_SIZE GREATER EYE_STATE_MODEL_PARTITION_SIZE)
            message(FATAL_ERROR
                "eye_state_detect_s8_v1.espdl (${EYE_STATE_MODEL_FILE_SIZE} bytes) does not fit eye_model partition "
                "(${EYE_STATE_MODEL_PARTITION_SIZE} bytes)")
        endif()
        esptool_py_flash_to_partition(flash "eye_model" "${EYE_STATE_MODEL_PATH}")
    else()
        message(FATAL_ERROR "eye_model partition is required for eye_state_detect_s8_v1.espdl")
    endif()
else()
    target_compile_definitions(${COMPONENT_LIB} PRIVATE
        HELMET_EYE_STATE_MODEL_AVAILABLE=0
        HELMET_EYE_STATE_MODEL_EMBEDDED=0)
endif()
