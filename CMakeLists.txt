cmake_minimum_required(VERSION 3.28)

project(NeuroServe
    VERSION 0.0.1
    DESCRIPTION "GPU-aware LLM inference runtime"
    LANGUAGES CXX)

# ---- Options -------------------------------------------------------------
option(NS_ENABLE_CUDA         "Build CUDA components and GPU tests" ON)
option(NS_BUILD_TESTS         "Build unit tests"                    ON)
option(NS_WARNINGS_AS_ERRORS  "Treat compiler warnings as errors"   OFF)

# ---- Language standards --------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # lets VS Code IntelliSense see real flags

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# ---- CUDA ------------------------------------------------------------------
if(NS_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)  # Ada Lovelace (RTX 4050 Laptop)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
endif()

# ---- Shared warning settings --------------------------------------------
add_library(ns_warnings INTERFACE)
target_compile_options(ns_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow>)
if(NS_WARNINGS_AS_ERRORS)
    target_compile_options(ns_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()

# ---- Components ------------------------------------------------------------
add_subdirectory(runtime)

if(NS_ENABLE_CUDA)
    # Execution engine (git submodule). EXCLUDE_FROM_ALL: build only what NeuroServe links.
    # SYSTEM: its headers don't trigger NeuroServe's strict warnings.
    add_subdirectory(third_party/cuda-llm-engine EXCLUDE_FROM_ALL SYSTEM)

    add_subdirectory(cuda)
    add_subdirectory(tools)
endif()

if(NS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "NeuroServe ${PROJECT_VERSION} | build=${CMAKE_BUILD_TYPE} | CUDA=${NS_ENABLE_CUDA} | archs=${CMAKE_CUDA_ARCHITECTURES}")


