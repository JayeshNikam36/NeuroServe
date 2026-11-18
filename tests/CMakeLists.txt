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
    unit/test_kv_prefix_hash.cpp)
target_include_directories(ns_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support)
# Golden scheduler traces live in tests/data; NS_UPDATE_GOLDEN=1 rewrites their expected sections.
target_compile_definitions(ns_unit_tests PRIVATE NS_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(ns_unit_tests PRIVATE neuroserve::core GTest::gtest_main ns_warnings)
gtest_discover_tests(ns_unit_tests PROPERTIES LABELS "cpu")

if(NS_ENABLE_CUDA)
    # ---- GPU unit tests (fast, no model needed) ----
    add_executable(ns_cuda_tests
        cuda/test_smoke.cpp
        cuda/test_device_info.cpp)
    target_link_libraries(ns_cuda_tests PRIVATE neuroserve::cuda GTest::gtest_main ns_warnings)
    gtest_discover_tests(ns_cuda_tests PROPERTIES LABELS "gpu")

    # ---- Integration tests (GPU + model weights + reference outputs) ----
    add_executable(ns_integration_tests
        integration/test_cuda_llm_engine_backend.cpp)
    target_link_libraries(ns_integration_tests PRIVATE neuroserve::backend GTest::gtest_main ns_warnings)
    gtest_discover_tests(ns_integration_tests PROPERTIES LABELS "integration")
endif()

# ---- Python binding tests (GPU + model, run with pytest) ----
if(NS_BUILD_PYTHON)
    add_test(NAME python_bindings
        COMMAND ${Python_EXECUTABLE} -m pytest -q -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}/python)
    set_tests_properties(python_bindings PROPERTIES
        LABELS "integration;python"
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/python")
endif()