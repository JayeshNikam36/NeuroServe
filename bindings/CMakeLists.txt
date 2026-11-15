include(FetchContent)
FetchContent_Declare(pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
    SYSTEM)
set(PYBIND11_FINDPYTHON ON)
FetchContent_MakeAvailable(pybind11)

# CMake target "ns_python"; Python module file "_core" (imported as neuroserve._core).
pybind11_add_module(ns_python MODULE module.cpp)
set_target_properties(ns_python PROPERTIES OUTPUT_NAME _core)
target_link_libraries(ns_python PRIVATE neuroserve::backend neuroserve::cuda neuroserve::core ns_warnings)

if(SKBUILD)
    # pip install / wheel build (scikit-build-core): install into the package.
    install(TARGETS ns_python LIBRARY DESTINATION neuroserve)
else()
    # Development build: put the module inside the source package, so PYTHONPATH=python works.
    set_target_properties(ns_python PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/neuroserve)
endif()