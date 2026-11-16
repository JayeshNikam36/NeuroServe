#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace neuroserve::core {

// A latency histogram with fixed, geometric buckets: kBucketsPerDoubling buckets for every factor
// of two from kMinMs to kMaxMs, plus one bucket below and one above. Recording is O(log buckets)
// and memory is constant, however many samples are recorded (a server records millions).
//
// Percentiles are estimated by linear interpolation inside the bucket that holds the requested
// rank, with the bucket's range narrowed to the smallest and largest sample seen. The estimate and
// the true sample at that rank (nearest-rank definition) lie in the same bucket, so the relative
// error is below one bucket width: 2^(1/8) - 1, about 9.05%. The tests check exactly that bound.
//
// Not thread-safe: the owner synchronizes (the engine loop records under its mutex).
class LatencyHistogram {
public:
    static constexpr double kMinMs = 0.01;          // 10 microseconds
    static constexpr double kMaxMs = 600'000.0;     // 10 minutes
    static constexpr int kBucketsPerDoubling = 8;   // bucket width: 2^(1/8) - 1 = 9.05% of its lower bound

    LatencyHistogram();

    void record(double ms);

    std::uint64_t count() const noexcept { return count_; }
    double mean() const noexcept { return count_ == 0 ? 0.0 : sum_ / static_cast<double>(count_); }
    double min() const noexcept { return count_ == 0 ? 0.0 : min_; }
    double max() const noexcept { return count_ == 0 ? 0.0 : max_; }

    // q in [0, 1] (0.5 = median, 0.99 = P99). 0 when nothing was recorded.
    double percentile(double q) const;

    // Bucket i holds samples in (upper_bounds()[i-1], upper_bounds()[i]]; the last bucket has no
    // upper bound (it holds everything above kMaxMs).
    std::span<const double> upper_bounds() const noexcept { return bounds_; }
    std::span<const std::uint64_t> bucket_counts() const noexcept { return counts_; }

private:
    std::vector<double> bounds_;          // upper bound of every bucket but the last
    std::vector<std::uint64_t> counts_;   // bounds_.size() + 1 buckets
    std::uint64_t count_ = 0;
    double sum_ = 0.0;
    double min_ = 0.0;
    double max_ = 0.0;
};

// What the engine reports for one latency metric.
struct LatencySummary {
    std::uint64_t count = 0;
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

LatencySummary summarize(const LatencyHistogram& histogram);

}  // namespace neuroserve::core
