add_library(neuroserve_core
    core/src/version.cpp)
add_library(neuroserve::core ALIAS neuroserve_core)

target_include_directories(neuroserve_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/include)

target_compile_definitions(neuroserve_core
    PRIVATE NS_VERSION_STRING="${PROJECT_VERSION}")

if(NS_ENABLE_CUDA)
    target_compile_definitions(neuroserve_core PRIVATE NS_WITH_CUDA=1)
endif()

target_link_libraries(neuroserve_core PRIVATE ns_warnings)


if(NS_ENABLE_CUDA)
    add_subdirectory(backend)
endif()