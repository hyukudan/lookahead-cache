cmake_minimum_required(VERSION 3.21)
project(StrixLookaheadConsumer LANGUAGES C CXX)

# Deliberately enable only C/C++. Linking the installed library must not require
# consumers of its C ABI to compile a HIP or CUDA translation unit.
find_package(StrixLookahead 0.2.1 CONFIG REQUIRED)
add_executable(sl_installed_consumer main.c)
set_target_properties(sl_installed_consumer PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON LINKER_LANGUAGE CXX)
target_link_libraries(sl_installed_consumer PRIVATE StrixLookahead::cache)
foreach(backend IN ITEMS hip cuda)
    if(TARGET StrixLookahead::${backend})
        string(TOUPPER ${backend} upper)
        target_compile_definitions(sl_installed_consumer PRIVATE SL_CONSUMER_${upper}=1)
        target_link_libraries(sl_installed_consumer PRIVATE StrixLookahead::${backend})
    endif()
endforeach()
