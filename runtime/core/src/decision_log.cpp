#include "neuroserve/core/decision_log.hpp"

#include <algorithm>
#include <stdexcept>

namespace neuroserve::core {

std::string_view to_string(DecisionAction action) noexcept {
    switch (action) {
        case DecisionAction::kAdmit:   return "admit";
        case DecisionAction::kDefer:   return "defer";
        case DecisionAction::kReject:  return "reject";
        case DecisionAction::kPreempt: return "preempt";
    }
    return "unknown";
}

DecisionLog::DecisionLog(std::size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) {
        throw std::invalid_argument("the decision log needs a capacity of at least 1");
    }
}

void DecisionLog::add(Explanation explanation) {
    explanation.seq = next_seq_++;
    if (entries_.size() == capacity_) {
        entries_.pop_front();
    }
    entries_.push_back(std::move(explanation));
}

DecisionPage DecisionLog::since(std::uint64_t since, std::size_t limit) const {
    DecisionPage page;
    page.next_since = since;
    if (entries_.empty() || since >= entries_.back().seq) {
        return page;  // nothing newer
    }
    const std::uint64_t oldest = entries_.front().seq;
    if (since + 1 < oldest) {
        page.dropped = oldest - since - 1;  // the reader fell behind by this many
    }
    // Sequence numbers are consecutive, so the first wanted entry is found by arithmetic.
    const std::uint64_t first = std::max(since + 1, oldest);
    const auto begin = entries_.begin() + static_cast<std::ptrdiff_t>(first - oldest);
    const auto count = std::min<std::size_t>(limit, static_cast<std::size_t>(entries_.end() - begin));
    page.decisions.assign(begin, begin + static_cast<std::ptrdiff_t>(count));
    if (!page.decisions.empty()) {
        page.next_since = page.decisions.back().seq;
    }
    return page;
}

}  // namespace neuroserve::core
