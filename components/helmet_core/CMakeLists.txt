file(GLOB_RECURSE HELMET_CORE_SRCS
    "src/*.c"
    "src/*.cpp"
)

idf_component_register(
    SRCS
        ${HELMET_CORE_SRCS}
    INCLUDE_DIRS
        "include"
    REQUIRES
        freertos
)