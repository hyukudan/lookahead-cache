# Run after building the library:
# cmake -DSL_BUILD_DIR=/absolute/build -P tests/install_consumer.cmake
# A non-default SDK can be supplied through CMAKE_PREFIX_PATH in the environment
# or -DSL_SDK_PREFIX_PATH=... and -DSL_CUDATOOLKIT_ROOT=... here.
cmake_minimum_required(VERSION 3.21)
if(NOT IS_DIRECTORY "${SL_BUILD_DIR}")
    message(FATAL_ERROR "Set SL_BUILD_DIR to a completed library build")
endif()
if(NOT DEFINED SL_CONFIG)
    set(SL_CONFIG Release)
endif()
set(stage "${SL_BUILD_DIR}/installed-consumer/prefix")
set(consumer_build "${SL_BUILD_DIR}/installed-consumer/build")
if(SL_TEST_LEGACY_PACKAGE)
    set(consumer_build "${SL_BUILD_DIR}/installed-consumer/build-legacy")
endif()

function(run_checked)
    execute_process(COMMAND ${ARGV} RESULT_VARIABLE code)
    if(NOT code STREQUAL "0")
        message(FATAL_ERROR "Consumer validation failed (${code}): ${ARGV}")
    endif()
endfunction()

run_checked("${CMAKE_COMMAND}" --install "${SL_BUILD_DIR}" --prefix "${stage}" --config "${SL_CONFIG}")
set(sdk_arguments)
if(SL_CUDATOOLKIT_ROOT)
    list(APPEND sdk_arguments "-DCUDAToolkit_ROOT=${SL_CUDATOOLKIT_ROOT}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${CMAKE_CURRENT_LIST_DIR}/consumer" -B "${consumer_build}"
    "-DCMAKE_BUILD_TYPE=${SL_CONFIG}" "-DCMAKE_PREFIX_PATH=${stage};${SL_SDK_PREFIX_PATH}"
    "-DSL_TEST_LEGACY_PACKAGE=${SL_TEST_LEGACY_PACKAGE}"
    ${sdk_arguments} RESULT_VARIABLE configure_code)
if(NOT configure_code STREQUAL "0")
    message(FATAL_ERROR "Consumer configuration failed (${configure_code})")
endif()
run_checked("${CMAKE_COMMAND}" --build "${consumer_build}" --config "${SL_CONFIG}")
