#include <cstring>
#include <exception>
#include <iostream>
#include <string>

#include "neuroserve/core/version.hpp"
#include "neuroserve/cuda/device_info.hpp"

namespace nc = neuroserve::cuda;

// Exit codes: 0 ok, 1 CUDA/initialization error, 2 no GPU found.
int main(int argc, char** argv) {
    const bool json = argc > 1 && std::strcmp(argv[1], "--json") == 0;

    std::string error;
    const int count = nc::device_count(&error);
    if (count == 0) {
        std::cerr << "neuroserve: no CUDA-capable GPU detected";
        if (!error.empty()) {
            std::cerr << " (" << error << ")";
        }
        std::cerr << "\n";
        return 2;
    }

    try {
        const nc::InitResult init = nc::initialize_device(0);
        if (json) {
            std::cout << "{\"neuroserve_version\":\"" << neuroserve::version() << "\""
                      << ",\"init_ms\":" << init.init_ms << ",\"devices\":[";
            for (int i = 0; i < count; ++i) {
                if (i > 0) {
                    std::cout << ",";
                }
                std::cout << nc::to_json(nc::query_device(i));
            }
            std::cout << "]}\n";
        } else {
            std::cout << "NeuroServe " << neuroserve::version() << " | " << count
                      << " CUDA device(s)\n";
            for (int i = 0; i < count; ++i) {
                std::cout << nc::summary(nc::query_device(i)) << "\n";
            }
            std::cout << "Initialized GPU 0 in " << init.init_ms << " ms\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "neuroserve: " << e.what() << "\n";
        return 1;
    }
    return 0;
}