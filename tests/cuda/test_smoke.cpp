#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <vector>

#include "neuroserve/cuda/smoke.hpp"

TEST(CudaSmoke, KernelWritesExpectedValues) {
    constexpr std::size_t n = std::size_t{1} << 20;  // 1M elements
    std::vector<float> out(n, -1.0f);
    neuroserve::cuda::run_axpb(2.0f, 1.0f, out.data(), n);

    const std::array<std::size_t, 4> probes{0, 1, 1000, n - 1};
    for (std::size_t i : probes) {
        EXPECT_FLOAT_EQ(out[i], 2.0f * static_cast<float>(i) + 1.0f) << "index " << i;
    }
}

TEST(CudaSmoke, ZeroLengthIsNoOp) {
    EXPECT_NO_THROW(neuroserve::cuda::run_axpb(1.0f, 0.0f, nullptr, 0));
}

TEST(CudaSmoke, CompiledForAdaSm89) {
    EXPECT_EQ(neuroserve::cuda::compiled_arch(), 890);
}