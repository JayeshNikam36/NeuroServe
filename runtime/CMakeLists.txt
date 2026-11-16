add_library(neuroserve_core
    core/src/version.cpp
    core/src/generate.cpp
    core/src/request.cpp
    core/src/engine_loop.cpp
    core/src/log.cpp
    core/src/scheduler.cpp
    core/src/latency_histogram.cpp)
add_library(neuroserve::core ALIAS neuroserve_core)

target_include_directories(neuroserve_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/include)

target_compile_definitions(neuroserve_core
    PRIVATE NS_VERSION_STRING="${PROJECT_VERSION}")

if(NS_ENABLE_CUDA)
    target_compile_definitions(neuroserve_core PRIVATE NS_WITH_CUDA=1)
endif()

find_package(Threads REQUIRED)
target_link_libraries(neuroserve_core
    PUBLIC  Threads::Threads
    PRIVATE ns_warnings)

# Linked into the Python extension module, which is a shared library.
set_target_properties(neuroserve_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NS_ENABLE_CUDA)
    add_subdirectory(backend)
endif()