cmake_minimum_required(VERSION 3.24)

project(CoordsFinder VERSION 1.0.0 LANGUAGES CXX)

# Keep GPU backends opt-in: configuring the default build must not require a CUDA toolkit or ROCm installation.
option(COORDSFINDER_ENABLE_CUDA "Build the optional CUDA backend" OFF)
option(COORDSFINDER_ENABLE_HIP "Build the optional HIP (AMD ROCm) backend" OFF)

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

add_executable(coordsfinder src/main.cpp)
target_link_libraries(coordsfinder PRIVATE coordsfinder_core)
target_compile_definitions(coordsfinder PRIVATE COORDSFINDER_VERSION="${PROJECT_VERSION}")

include(CTest)
if(BUILD_TESTING)
    add_executable(coordsfinder_tests tests/tests.cpp)
    target_link_libraries(coordsfinder_tests PRIVATE coordsfinder_core)
    target_compile_definitions(coordsfinder_tests PRIVATE COORDSFINDER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME coordsfinder_tests COMMAND coordsfinder_tests)
endif()

# GPU backends attach to both the main executable and the test suite, so the
# tests can exercise them wherever a compatible device is available.
if(COORDSFINDER_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    # CUDA is enabled only after the option check so the CPU preset stays portable.
    enable_language(CUDA)
    foreach(target coordsfinder coordsfinder_tests)
        if(TARGET ${target})
            target_sources(${target} PRIVATE src/cuda_backend.cu)
            target_compile_definitions(${target} PRIVATE COORDSFINDER_HAS_CUDA=1)
            set_target_properties(${target} PROPERTIES
                CUDA_STANDARD 20
                CUDA_STANDARD_REQUIRED YES)
        endif()
    endforeach()
    if(MSVC)
        target_compile_options(coordsfinder PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endif()

if(COORDSFINDER_ENABLE_HIP)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        # Broad RDNA3/RDNA4 default so the binary runs on most modern AMD GPUs
        # (desktops, laptops and APUs) out of the box; override for other
        # targets, e.g. -DCMAKE_HIP_ARCHITECTURES=gfx1030 for RX 6000 series.
        set(CMAKE_HIP_ARCHITECTURES "gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1200;gfx1201")
    endif()
    find_package(hip QUIET CONFIG)
    if(NOT hip_FOUND)
        message(FATAL_ERROR
            "COORDSFINDER_ENABLE_HIP is ON but the ROCm HIP package was not found. "
            "Install rocm-hip-sdk (or set CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake/hip).")
    endif()
    enable_language(HIP)
    # .cpp does not imply the HIP language for CMake; opt this file in
    # explicitly so hipcc compiles its device code.
    set_source_files_properties(src/hip_backend.cpp PROPERTIES LANGUAGE HIP)
    foreach(target coordsfinder coordsfinder_tests)
        if(TARGET ${target})
            target_sources(${target} PRIVATE src/hip_backend.cpp)
            target_compile_definitions(${target} PRIVATE COORDSFINDER_HAS_HIP=1)
            # Targets created before enable_language(HIP) miss the variable
            # default, so set the architecture property explicitly.
            set_target_properties(${target} PROPERTIES
                HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
                HIP_STANDARD 20
                HIP_STANDARD_REQUIRED YES)
        endif()
    endforeach()
    target_link_libraries(coordsfinder PRIVATE hip::host)
endif()

if(MSVC)
    target_compile_options(coordsfinder_core PRIVATE /W4 /permissive- /utf-8)
    target_compile_options(coordsfinder PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/utf-8>)
else()
    target_compile_options(coordsfinder_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(coordsfinder PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
endif()

install(TARGETS coordsfinder RUNTIME DESTINATION bin)
