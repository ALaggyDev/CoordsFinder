cmake_minimum_required(VERSION 3.24)

project(CoordsFinder VERSION 1.0.0 LANGUAGES CXX)

# Keep CUDA opt-in: configuring the default build must not require a CUDA toolkit.
option(COORDSFINDER_ENABLE_CUDA "Build the optional CUDA backend" OFF)
if(APPLE)
    set(COORDSFINDER_METAL_DEFAULT ON)
else()
    set(COORDSFINDER_METAL_DEFAULT OFF)
endif()
option(COORDSFINDER_ENABLE_METAL "Build the optional Apple Metal backend" ${COORDSFINDER_METAL_DEFAULT})

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

find_package(Threads REQUIRED)

add_library(coordsfinder_core STATIC
    src/config.cpp
    src/cpu_backend.cpp
    src/scan.cpp)
target_include_directories(coordsfinder_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(coordsfinder_core PUBLIC cxx_std_20)
target_link_libraries(coordsfinder_core PUBLIC Threads::Threads)

if(COORDSFINDER_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "COORDSFINDER_ENABLE_METAL requires macOS")
    endif()
    enable_language(OBJCXX)
    target_sources(coordsfinder_core PRIVATE src/metal_backend.mm)
    target_compile_definitions(coordsfinder_core PUBLIC COORDSFINDER_HAS_METAL=1)
    target_link_libraries(coordsfinder_core PUBLIC "-framework Foundation" "-framework Metal")
    set_source_files_properties(src/metal_backend.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(coordsfinder_core PROPERTIES
        OBJCXX_STANDARD 20
        OBJCXX_STANDARD_REQUIRED YES)
endif()

add_executable(coordsfinder src/main.cpp)
target_link_libraries(coordsfinder PRIVATE coordsfinder_core)
target_compile_definitions(coordsfinder PRIVATE COORDSFINDER_VERSION="${PROJECT_VERSION}")

if(COORDSFINDER_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    # CUDA is enabled only after the option check so the CPU preset stays portable.
    enable_language(CUDA)
    target_sources(coordsfinder PRIVATE src/cuda_backend.cu)
    target_compile_definitions(coordsfinder PRIVATE COORDSFINDER_HAS_CUDA=1)
    set_target_properties(coordsfinder PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED YES)
    if(MSVC)
        target_compile_options(coordsfinder PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endif()

if(MSVC)
    target_compile_options(coordsfinder_core PRIVATE /W4 /permissive- /utf-8)
    target_compile_options(coordsfinder PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/utf-8>)
else()
    target_compile_options(coordsfinder_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(coordsfinder PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(coordsfinder_tests tests/tests.cpp)
    target_link_libraries(coordsfinder_tests PRIVATE coordsfinder_core)
    target_compile_definitions(coordsfinder_tests PRIVATE COORDSFINDER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME coordsfinder_tests COMMAND coordsfinder_tests)
endif()

install(TARGETS coordsfinder RUNTIME DESTINATION bin)
