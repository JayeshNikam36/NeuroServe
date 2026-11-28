add_library(neuroserve_core
    core/src/version.cpp
    core/src/generate.cpp
    core/src/request.cpp
    core/src/engine_loop.cpp
    core/src/log.cpp
    core/src/scheduler.cpp
    core/src/latency_histogram.cpp
    core/src/decision_log.cpp
    core/src/kv_reference.cpp
    core/src/kv_cache_manager.cpp
    core/src/kv_prefix_hash.cpp
    core/src/kv_memory.cpp
    core/src/memory_ledger.cpp)
add_library(neuroserve::core ALIAS neuroserve_core)

target_include_directories(neuroserve_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/include)

target_compile_definitions(neuroserve_core
    PRIVATE NS_VERSION_STRING="${PROJECT_VERSION}")

if(NS_ENABLE_CUDA)
    target_compile_definitions(neuroserve_core PRIVATE NS_WITH_CUDA=1)
    # NVTX ranges for Nsight Systems (core/include/neuroserve/core/nvtx.hpp): header-only NVTX 3.
    if(TARGET CUDA::nvtx3)
        target_link_libraries(neuroserve_core PRIVATE CUDA::nvtx3)
        target_compile_definitions(neuroserve_core PRIVATE NS_NVTX=1)
    endif()
endif()
if(NS_CHECK_INVARIANTS)
    # PUBLIC: EngineLoopOptions' default must be the same in the library and everything using it.
    target_compile_definitions(neuroserve_core PUBLIC NS_CHECK_INVARIANTS=1)
endif()

find_package(Threads REQUIRED)
target_link_libraries(neuroserve_core
    PUBLIC  Threads::Threads
    PRIVATE ns_warnings)

# Linked into the Python extension module, which is a shared library.
set_target_properties(neuroserve_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NS_ENABLE_CUDA)
    add_subdirectory(kernels)
    add_subdirectory(backend)
endif()