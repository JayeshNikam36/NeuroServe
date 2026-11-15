#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace neuroserve::cuda {

// Exception for a failed CUDA runtime call. Keeps the original error code.
class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    cudaError_t code() const noexcept { return code_; }

private:
    cudaError_t code_;
};

[[noreturn]] void throw_cuda_error(cudaError_t code, const char* expr, const char* file, int line);

}  // namespace neuroserve::cuda

// Wrap every CUDA runtime call: NS_CUDA_CHECK(cudaMalloc(&p, bytes));
#define NS_CUDA_CHECK(expr)                                                         \
    do {                                                                            \
        const cudaError_t ns_cuda_err_ = (expr);                                    \
        if (ns_cuda_err_ != cudaSuccess) {                                          \
            ::neuroserve::cuda::throw_cuda_error(ns_cuda_err_, #expr, __FILE__, __LINE__); \
        }                                                                           \
    } while (0)