#pragma once

#include <cstddef>

namespace neuroserve::cuda {

// Fills host_out[i] = a * i + b using a GPU kernel. Throws std::runtime_error on CUDA failure.
void run_axpb(float a, float b, float* host_out, std::size_t n);

// Returns __CUDA_ARCH__ as seen by a running kernel (e.g. 890 for sm_89).
int compiled_arch();

}  // namespace neuroserve::cuda