#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace neuroserve::core {

enum class DecisionAction { kAdmit, kDefer, kReject, kPreempt };

std::string_view to_string(DecisionAction action) noexcept;

// Why the engine did what it did with one request, with the numbers it decided on.
// See docs/scheduling.md ("Decision log") for every reason code.
struct Explanation {
    std::uint64_t seq = 0;      // position in the log: 1, 2, 3, ... (the cursor for since=)
    std::uint64_t step = 0;     // engine step (scheduling pass) the decision was made in
    std::int64_t unix_ms = 0;   // wall-clock time, for people reading the log
    std::string request_id;     // the engine's id (ns-...)
    std::string client_request_id;
    DecisionAction action = DecisionAction::kDefer;
    std::string reason;         // short code, e.g. "batch_full", "kv_pressure", "queue_full"
    std::string policy;         // the scheduling policy that was deciding
    std::vector<std::pair<std::string, std::int64_t>> numbers;  // e.g. {"pages_needed", 3}, {"free_pages", 1}
};

struct DecisionPage {
    std::vector<Explanation> decisions;  // oldest first
    std::uint64_t next_since = 0;        // pass as `since` to continue after these
    std::uint64_t dropped = 0;           // decisions after `since` that were already overwritten
};

// A bounded ring buffer of explanations: the newest `capacity` are kept, older ones are dropped,
// so memory stays bounded however long the server runs. Readers page through it with a cursor
// (`since` = the last seq they saw). Not thread-safe: the engine loop guards it with its mutex.
class DecisionLog {
public:
    explicit DecisionLog(std::size_t capacity);

    // Stores the explanation with the next seq number (overwriting the oldest if full).
    void add(Explanation explanation);

    // Up to `limit` explanations with seq > since, oldest first.
    DecisionPage since(std::uint64_t since, std::size_t limit) const;

    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t size() const noexcept { return entries_.size(); }

private:
    std::size_t capacity_;
    std::deque<Explanation> entries_;
    std::uint64_t next_seq_ = 1;
};

}  // namespace neuroserve::core
