# NeuroServe's own CUDA kernels (Phase 4), independent of cuda-llm-engine: tested on their own in
# ns_cuda_tests, without a model.
add_library(neuroserve_kernels
    src/attention_varlen.cu)
add_library(neuroserve::kernels ALIAS neuroserve_kernels)

target_include_directories(neuroserve_kernels
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(neuroserve_kernels
    PUBLIC  CUDA::cudart
    PRIVATE ns_warnings)

set_target_properties(neuroserve_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
