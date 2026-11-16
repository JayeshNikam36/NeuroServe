#include "neuroserve/core/latency_histogram.hpp"

#include <algorithm>
#include <cmath>

namespace neuroserve::core {

LatencyHistogram::LatencyHistogram() {
    const double ratio = std::pow(2.0, 1.0 / kBucketsPerDoubling);
    for (double bound = kMinMs; bound < kMaxMs * ratio; bound *= ratio) {
        bounds_.push_back(bound);
    }
    counts_.assign(bounds_.size() + 1, 0);
}

void LatencyHistogram::record(double ms) {
    ms = std::max(ms, 0.0);  // a clock can only go forward; guard against rounding below zero
    const auto bucket = static_cast<std::size_t>(std::lower_bound(bounds_.begin(), bounds_.end(), ms) - bounds_.begin());
    ++counts_[bucket];
    if (count_ == 0) {
        min_ = ms;
        max_ = ms;
    } else {
        min_ = std::min(min_, ms);
        max_ = std::max(max_, ms);
    }
    ++count_;
    sum_ += ms;
}

double LatencyHistogram::percentile(double q) const {
    if (count_ == 0) {
        return 0.0;
    }
    q = std::clamp(q, 0.0, 1.0);
    // Nearest rank: the smallest sample with at least q of all samples at or below it.
    const auto rank = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(std::ceil(q * static_cast<double>(count_))));
    std::uint64_t below = 0;
    for (std::size_t i = 0; i < counts_.size(); ++i) {
        if (below + counts_[i] >= rank) {
            // The bucket's range, narrowed to the samples seen (exact for the first and last bucket).
            const double lower = std::max(i == 0 ? 0.0 : bounds_[i - 1], min_);
            const double upper = std::min(i < bounds_.size() ? bounds_[i] : max_, max_);
            const double fraction = static_cast<double>(rank - below) / static_cast<double>(counts_[i]);
            return lower + fraction * (upper - lower);
        }
        below += counts_[i];
    }
    return max_;  // not reached: the counts add up to count_
}

LatencySummary summarize(const LatencyHistogram& histogram) {
    LatencySummary s;
    s.count = histogram.count();
    s.mean = histogram.mean();
    s.p50 = histogram.percentile(0.50);
    s.p95 = histogram.percentile(0.95);
    s.p99 = histogram.percentile(0.99);
    s.max = histogram.max();
    return s;
}

}  // namespace neuroserve::core
