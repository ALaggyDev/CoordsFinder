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

if(COORDSFINDER_ENABLE_HIP)
    if(COORDSFINDER_ENABLE_CUDA)
        message(FATAL_ERROR
            "COORDSFINDER_ENABLE_CUDA and COORDSFINDER_ENABLE_HIP cannot be combined; "
            "the GPU backends define the same device symbols and would collide at link time.")
    endif()
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        # Broad RDNA3/RDNA4 default so the binary runs on most modern AMD GPUs
        # out of the box; override for other targets,
        # e.g. -DCMAKE_HIP_ARCHITECTURES=gfx1030 for RX 6000 series.
        set(CMAKE_HIP_ARCHITECTURES "gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201")
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
    target_sources(coordsfinder PRIVATE src/hip_backend.cpp)
    target_compile_definitions(coordsfinder PRIVATE COORDSFINDER_HAS_HIP=1)
    set_target_properties(coordsfinder PROPERTIES
        HIP_STANDARD 20
        HIP_STANDARD_REQUIRED YES)
    target_link_libraries(coordsfinder PRIVATE hip::host)
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
