#include "neuroserve/cuda/smoke.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace neuroserve::cuda {
namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

__global__ void axpb_kernel(float a, float b, float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) {
        out[i] = a * static_cast<float>(i) + b;
    }
}

__global__ void arch_kernel(int* out) {
#ifdef __CUDA_ARCH__
    *out = __CUDA_ARCH__;
#endif
}

}  // namespace

void run_axpb(float a, float b, float* host_out, std::size_t n) {
    if (n == 0) {
        return;
    }
    float* d_out = nullptr;
    check(cudaMalloc(&d_out, n * sizeof(float)), "cudaMalloc");

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    axpb_kernel<<<blocks, kThreads>>>(a, b, d_out, n);

    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaMemcpy(host_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_out);  // always free, even on error
    check(err, "axpb_kernel / cudaMemcpy");
}

int compiled_arch() {
    int* d_arch = nullptr;
    check(cudaMalloc(&d_arch, sizeof(int)), "cudaMalloc");
    arch_kernel<<<1, 1>>>(d_arch);

    int arch = 0;
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaMemcpy(&arch, d_arch, sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_arch);
    check(err, "arch_kernel / cudaMemcpy");
    return arch;
}

}  // namespace neuroserve::cuda