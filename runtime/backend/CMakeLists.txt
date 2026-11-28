add_library(neuroserve_backend
    src/cuda_llm_engine_backend.cpp
    src/gpu_memory.cpp
    src/prefill_from_offset.cu
    src/decode_graph.cu)
add_library(neuroserve::backend ALIAS neuroserve_backend)

target_include_directories(neuroserve_backend
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(neuroserve_backend
    PUBLIC  neuroserve::core
    PRIVATE cuda_engine::core neuroserve::cuda neuroserve::kernels ns_warnings)

set_target_properties(neuroserve_backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON)