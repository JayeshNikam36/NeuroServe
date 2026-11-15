#include "neuroserve/cuda/check.hpp"

#include <string>

namespace neuroserve::cuda {

void throw_cuda_error(cudaError_t code, const char* expr, const char* file, int line) {
    throw CudaError(code, std::string(file) + ":" + std::to_string(line) + ": " + expr +
                              " failed: " + cudaGetErrorName(code) + " (" +
                              cudaGetErrorString(code) + ")");
}

}  // namespace neuroserve::cuda