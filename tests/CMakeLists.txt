include(FetchContent)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# ---- CPU unit tests (run everywhere, including CI) ----
add_executable(ns_unit_tests
    unit/test_version.cpp
    unit/test_generate.cpp
    unit/test_request.cpp
    unit/test_engine_loop.cpp
    unit/test_log.cpp
    unit/test_scheduler.cpp
    unit/test_latency_histogram.cpp
    unit/test_decision_log.cpp
    unit/test_kv_block_manager.cpp
    unit/test_kv_property.cpp
    unit/test_kv_invariants.cpp
    unit/test_kv_sharing.cpp
    unit/test_kv_prefix_hash.cpp
    unit/test_kv_prefix_index.cpp
    unit/test_kv_prefix_cache.cpp
    unit/test_preemption_recompute.cpp
    unit/test_kv_memory.cpp
    unit/test_step_profile.cpp
    unit/test_engine_prefill_batch.cpp
    unit/test_engine_chunked_prefill.cpp
    unit/test_engine_mixed_steps.cpp)
target_include_directories(ns_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support)
# Golden scheduler traces live in tests/data; NS_UPDATE_GOLDEN=1 rewrites their expected sections.
target_compile_definitions(ns_unit_tests PRIVATE NS_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(ns_unit_tests PRIVATE neuroserve::core GTest::gtest_main ns_warnings)
gtest_discover_tests(ns_unit_tests PROPERTIES LABELS "cpu")

if(NS_ENABLE_CUDA)
    # ---- GPU unit tests (fast, no model needed) ----
    add_executable(ns_cuda_tests
        cuda/test_smoke.cpp
        cuda/test_device_info.cpp
        cuda/test_attention_varlen.cpp)
    target_link_libraries(ns_cuda_tests PRIVATE neuroserve::cuda neuroserve::kernels GTest::gtest_main ns_warnings)
    gtest_discover_tests(ns_cuda_tests PROPERTIES LABELS "gpu")

    # ---- Integration tests (GPU + model weights + reference outputs) ----
    add_executable(ns_integration_tests
        integration/test_cuda_llm_engine_backend.cpp)
    target_link_libraries(ns_integration_tests PRIVATE neuroserve::backend GTest::gtest_main ns_warnings)
    gtest_discover_tests(ns_integration_tests PROPERTIES LABELS "integration")
endif()

# ---- Python binding tests (GPU + model, run with pytest) ----
if(NS_BUILD_PYTHON)
    # Compute Sanitizer (Day 41): memcheck and racecheck on the KV, scheduler and GPU suites.
    #   cmake --build build/dev-cuda --target compute_sanitizer   (logs and summary in tests/logs)
    find_program(NS_COMPUTE_SANITIZER compute-sanitizer HINTS "${CUDAToolkit_BIN_DIR}" /usr/local/cuda/bin)
    if(NS_COMPUTE_SANITIZER)
        add_custom_target(compute_sanitizer
            COMMAND bash "${PROJECT_SOURCE_DIR}/scripts/compute_sanitizer.sh" "${NS_COMPUTE_SANITIZER}"
                    $<TARGET_FILE:ns_unit_tests> $<TARGET_FILE:ns_cuda_tests> $<TARGET_FILE:ns_integration_tests>
                    "${PROJECT_SOURCE_DIR}/tests/logs"
            DEPENDS ns_unit_tests ns_cuda_tests ns_integration_tests
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            USES_TERMINAL
            COMMENT "Compute Sanitizer: memcheck and racecheck (see tests/logs)")
    endif()

    add_test(NAME python_bindings
        COMMAND ${Python_EXECUTABLE} -m pytest -q -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}/python)
    set_tests_properties(python_bindings PROPERTIES
        LABELS "integration;python"
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/python")
endif()