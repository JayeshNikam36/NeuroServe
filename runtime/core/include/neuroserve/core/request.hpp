#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace neuroserve::core {

// Lifecycle of one inference request. See docs/request-lifecycle.md for the diagram.
enum class RequestState : std::uint8_t {
    kReceived,   // created by the API, nothing checked yet
    kValidated,  // passed request checks (prompt length, max_tokens)
    kQueued,     // waiting for the scheduler
    kAdmitted,   // KV-cache pages reserved by the scheduler
    kPrefill,    // prompt being processed
    kDecode,     // generating one token per engine step
    kPreempted,  // paused, pages reclaimed; goes back to the queue
    kCompleted,  // terminal: finished normally (stop token or max_tokens)
    kCancelled,  // terminal: client cancelled or disconnected
    kRejected,   // terminal: never ran (invalid, queue full, queue timeout); safe to retry
    kFailed,     // terminal: error while running (e.g. GPU error)
};

inline constexpr std::size_t kNumRequestStates = 11;

std::string_view to_string(RequestState state) noexcept;
std::ostream& operator<<(std::ostream& os, RequestState state);
bool is_terminal(RequestState state) noexcept;
bool is_legal_transition(RequestState from, RequestState to) noexcept;

class IllegalTransition : public std::logic_error {
public:
    IllegalTransition(const std::string& request_id, RequestState from, RequestState to);
    RequestState from() const noexcept { return from_; }
    RequestState to() const noexcept { return to_; }

private:
    RequestState from_;
    RequestState to_;
};

// A request's state and the time it entered each state.
// Not thread-safe: only the engine thread changes a Request.
class Request {
public:
    using Clock = std::chrono::steady_clock;  // monotonic: durations are never negative

    explicit Request(std::string id, Clock::time_point now = Clock::now());

    const std::string& id() const noexcept { return id_; }
    RequestState state() const noexcept { return state_; }
    bool is_finished() const noexcept { return is_terminal(state_); }

    // Throws IllegalTransition for a move the state machine does not allow, and
    // std::invalid_argument if `now` is earlier than the previous transition.
    // On any throw the request is unchanged.
    void transition_to(RequestState next, Clock::time_point now = Clock::now());

    // First time the request entered `state`; empty if it never did.
    std::optional<Clock::time_point> entered_at(RequestState state) const noexcept;
    Clock::time_point last_transition_at() const noexcept { return last_transition_at_; }
    std::uint32_t preemption_count() const noexcept { return preemption_count_; }

private:
    std::string id_;
    RequestState state_ = RequestState::kReceived;
    std::array<std::optional<Clock::time_point>, kNumRequestStates> entered_at_{};
    Clock::time_point last_transition_at_;
    std::uint32_t preemption_count_ = 0;
};

// Monotonic, process-unique request ids: ns-000001, ns-000002, ... (grows past 6 digits).
// Thread-safe.
class RequestIdGenerator {
public:
    explicit RequestIdGenerator(std::uint64_t first = 1) noexcept : next_(first) {}
    std::string next();

private:
    std::atomic<std::uint64_t> next_;
};

}  // namespace neuroserve::core
