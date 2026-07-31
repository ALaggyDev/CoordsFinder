cmake_minimum_required(VERSION 3.24)

project(CoordsFinder VERSION 0.2.0 LANGUAGES CXX)

# Keep CUDA opt-in: configuring the default build must not require a CUDA toolkit.
option(COORDSFINDER_ENABLE_CUDA "Build the optional CUDA backend" OFF)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

find_package(Threads REQUIRED)

add_library(coordsfinder_core STATIC
    src/config.cpp
    src/cpu_backend.cpp
    src/scan.cpp)
target_include_directories(coordsfinder_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(coordsfinder_core PUBLIC cxx_std_17)
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
        CUDA_STANDARD 17
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
