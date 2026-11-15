add_library(neuroserve_cuda
    check/check.cpp
    device/device_info.cpp
    smoke/smoke.cu)
add_library(neuroserve::cuda ALIAS neuroserve_cuda)

target_include_directories(neuroserve_cuda
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(neuroserve_cuda
    PUBLIC CUDA::cudart
    PRIVATE ns_warnings)

target_compile_options(neuroserve_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

set_target_properties(neuroserve_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON)