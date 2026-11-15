include(FetchContent)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# ---- CPU unit tests (run everywhere, including CI) ----
add_executable(ns_unit_tests
    unit/test_version.cpp)
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
    gtest_discover_tests(ns_integration_tests PROPERTIES LABELS "gpu;integration")
endif()