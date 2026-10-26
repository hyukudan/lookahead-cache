cmake_minimum_required(VERSION 3.20)
project(StrixLookahead VERSION 0.1.0 LANGUAGES C CXX)

add_library(strix_lookahead STATIC src/cache.cpp)
add_library(StrixLookahead::cache ALIAS strix_lookahead)
set_target_properties(strix_lookahead PROPERTIES
    EXPORT_NAME cache CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_features(strix_lookahead PUBLIC cxx_std_20)
target_include_directories(strix_lookahead PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strix_lookahead PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(sl_cache_test tests/cache_test.cpp)
    target_link_libraries(sl_cache_test PRIVATE StrixLookahead::cache)
    add_test(NAME cache_contract COMMAND sl_cache_test)
    add_executable(sl_c_api_smoke tests/c_api_smoke.c)
    set_target_properties(sl_c_api_smoke PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON LINKER_LANGUAGE CXX)
    target_link_libraries(sl_c_api_smoke PRIVATE StrixLookahead::cache)
    add_test(NAME c_abi_smoke COMMAND sl_c_api_smoke)
endif()

option(SL_BUILD_EXAMPLES "Build the host reference example" ON)
if(SL_BUILD_EXAMPLES)
    add_executable(sl_cache_example examples/cache_example.cpp)
    target_link_libraries(sl_cache_example PRIVATE StrixLookahead::cache)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS strix_lookahead EXPORT StrixLookaheadTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT StrixLookaheadTargets NAMESPACE StrixLookahead:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StrixLookahead)
configure_package_config_file(cmake/StrixLookaheadConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/StrixLookaheadConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StrixLookahead)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/StrixLookaheadConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/StrixLookaheadConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/StrixLookaheadConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StrixLookahead)
