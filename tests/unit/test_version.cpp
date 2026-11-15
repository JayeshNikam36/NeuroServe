#include <gtest/gtest.h>

#include <algorithm>

#include "neuroserve/core/version.hpp"

TEST(Version, IsNonEmptySemver) {
    const auto v = neuroserve::version();
    ASSERT_FALSE(v.empty());
    EXPECT_EQ(std::count(v.begin(), v.end(), '.'), 2) << "expected MAJOR.MINOR.PATCH, got " << v;
}

TEST(Version, ReportsCudaFlag) {
    // Only checks that the call works; the value depends on the build preset.
    const bool enabled = neuroserve::cuda_enabled();
    SUCCEED() << "cuda_enabled=" << enabled;
}