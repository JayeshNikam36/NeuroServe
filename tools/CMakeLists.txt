add_executable(ns_device_info ns_device_info/main.cpp)
target_link_libraries(ns_device_info PRIVATE neuroserve::cuda neuroserve::core ns_warnings)