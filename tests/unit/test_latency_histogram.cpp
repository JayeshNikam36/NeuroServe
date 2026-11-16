#include "neuroserve/core/latency_histogram.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using neuroserve::core::LatencyHistogram;

namespace {

constexpr double kBucketError = 0.0906;  // one bucket width: 2^(1/8) - 1 = 0.0905, rounded up

// The nearest-rank percentile of the samples: the value the histogram estimates.
double exact_percentile(std::vector<double> samples, double q) {
    std::sort(samples.begin(), samples.end());
    const auto rank = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(q * static_cast<double>(samples.size()))));
    return samples[rank - 1];
}

LatencyHistogram histogram_of(const std::vector<double>& samples) {
    LatencyHistogram h;
    for (const double s : samples) {
        h.record(s);
    }
    return h;
}

// Every estimate must be within one bucket width of the true value (see latency_histogram.hpp).
void expect_within_one_bucket(const LatencyHistogram& h, const std::vector<double>& samples) {
    for (const double q : {0.5, 0.9, 0.95, 0.99, 0.999}) {
        const double exact = exact_percentile(samples, q);
        EXPECT_NEAR(h.percentile(q), exact, exact * kBucketError) << "q = " << q;
    }
}

}  // namespace

TEST(LatencyHistogram, EmptyReportsZero) {
    const LatencyHistogram h;
    EXPECT_EQ(h.count(), 0u);
    EXPECT_EQ(h.percentile(0.5), 0.0);
    EXPECT_EQ(h.mean(), 0.0);
    EXPECT_EQ(h.max(), 0.0);
}

TEST(LatencyHistogram, OneSampleIsReportedExactly) {
    LatencyHistogram h;
    h.record(7.3);
    EXPECT_DOUBLE_EQ(h.percentile(0.5), 7.3);  // clamped to the smallest and largest sample
    EXPECT_DOUBLE_EQ(h.percentile(0.99), 7.3);
    EXPECT_DOUBLE_EQ(h.mean(), 7.3);
}

TEST(LatencyHistogram, IdenticalSamplesAreReportedExactly) {
    const auto h = histogram_of(std::vector<double>(1000, 25.0));
    EXPECT_DOUBLE_EQ(h.percentile(0.5), 25.0);
    EXPECT_DOUBLE_EQ(h.percentile(0.99), 25.0);
}

TEST(LatencyHistogram, UniformSamplesAreWithinOneBucket) {
    std::vector<double> samples;
    for (int ms = 1; ms <= 1000; ++ms) {
        samples.push_back(ms);
    }
    const auto h = histogram_of(samples);
    EXPECT_EQ(h.count(), 1000u);
    EXPECT_DOUBLE_EQ(h.mean(), 500.5);  // the mean is exact: it comes from the running sum
    expect_within_one_bucket(h, samples);
}

TEST(LatencyHistogram, ASlowTailShowsUpInP99NotInP50) {
    std::vector<double> samples(980, 10.0);           // 98% of requests take 10 ms ...
    samples.insert(samples.end(), 20, 1000.0);          // ... 2% take a second
    const auto h = histogram_of(samples);
    expect_within_one_bucket(h, samples);
    EXPECT_LT(h.percentile(0.95), 11.0);
    EXPECT_GT(h.percentile(0.99), 900.0);
}

TEST(LatencyHistogram, LogNormalSamplesAreWithinOneBucket) {
    std::mt19937 rng(42);  // fixed seed: the same samples on every run
    std::lognormal_distribution<double> latency(std::log(20.0), 1.0);  // median 20 ms, long right tail
    std::vector<double> samples(100'000);
    for (auto& s : samples) {
        s = latency(rng);
    }
    const auto h = histogram_of(samples);
    expect_within_one_bucket(h, samples);
    EXPECT_LE(h.percentile(0.5), h.percentile(0.95));
    EXPECT_LE(h.percentile(0.95), h.percentile(0.99));
    EXPECT_LE(h.percentile(0.99), h.max());
}

TEST(LatencyHistogram, ExtremesLandInTheEdgeBuckets) {
    LatencyHistogram h;
    h.record(-1.0);            // clamped to 0
    h.record(0.0);
    h.record(1'000'000.0);     // above kMaxMs: the overflow bucket
    EXPECT_EQ(h.bucket_counts().front(), 2u);
    EXPECT_EQ(h.bucket_counts().back(), 1u);
    EXPECT_DOUBLE_EQ(h.min(), 0.0);
    EXPECT_DOUBLE_EQ(h.percentile(1.0), 1'000'000.0);
}

TEST(LatencyHistogram, BucketsAreGeometricAndCoverTheRange) {
    const LatencyHistogram h;
    const auto bounds = h.upper_bounds();
    EXPECT_DOUBLE_EQ(bounds.front(), LatencyHistogram::kMinMs);
    EXPECT_GE(bounds.back(), LatencyHistogram::kMaxMs);
    const double ratio = std::pow(2.0, 1.0 / LatencyHistogram::kBucketsPerDoubling);
    for (std::size_t i = 1; i < bounds.size(); ++i) {
        EXPECT_NEAR(bounds[i] / bounds[i - 1], ratio, 1e-9);
    }
    EXPECT_EQ(h.bucket_counts().size(), bounds.size() + 1);
}
