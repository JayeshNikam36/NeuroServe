#include "neuroserve/core/request.hpp"

#include <cstdio>
#include <ostream>
#include <utility>

namespace neuroserve::core {

namespace {

std::size_t index_of(RequestState state) noexcept { return static_cast<std::size_t>(state); }

std::string transition_message(const std::string& request_id, RequestState from, RequestState to) {
    std::string msg = "request ";
    msg += request_id;
    msg += ": illegal transition ";
    msg += to_string(from);
    msg += " -> ";
    msg += to_string(to);
    return msg;
}

}  // namespace

std::string_view to_string(RequestState state) noexcept {
    switch (state) {
        case RequestState::kReceived:  return "RECEIVED";
        case RequestState::kValidated: return "VALIDATED";
        case RequestState::kQueued:    return "QUEUED";
        case RequestState::kAdmitted:  return "ADMITTED";
        case RequestState::kPrefill:   return "PREFILL";
        case RequestState::kDecode:    return "DECODE";
        case RequestState::kPreempted: return "PREEMPTED";
        case RequestState::kCompleted: return "COMPLETED";
        case RequestState::kCancelled: return "CANCELLED";
        case RequestState::kRejected:  return "REJECTED";
        case RequestState::kFailed:    return "FAILED";
    }
    return "UNKNOWN";
}

std::ostream& operator<<(std::ostream& os, RequestState state) { return os << to_string(state); }

bool is_terminal(RequestState state) noexcept {
    return state == RequestState::kCompleted || state == RequestState::kCancelled ||
           state == RequestState::kRejected || state == RequestState::kFailed;
}

bool is_legal_transition(RequestState from, RequestState to) noexcept {
    using S = RequestState;
    switch (from) {
        case S::kReceived:  return to == S::kValidated || to == S::kRejected || to == S::kCancelled;
        case S::kValidated: return to == S::kQueued || to == S::kRejected || to == S::kCancelled;
        case S::kQueued:    return to == S::kAdmitted || to == S::kRejected || to == S::kCancelled;
        case S::kAdmitted:  return to == S::kPrefill || to == S::kCancelled || to == S::kFailed;
        // Prefill emits the first token, so a request can finish without ever decoding.
        case S::kPrefill:   return to == S::kDecode || to == S::kCompleted || to == S::kPreempted ||
                                   to == S::kCancelled || to == S::kFailed;
        case S::kDecode:    return to == S::kCompleted || to == S::kPreempted || to == S::kCancelled ||
                                   to == S::kFailed;
        // A preempted request lost its pages: it re-enters the queue, never jumps to ADMITTED.
        case S::kPreempted: return to == S::kQueued || to == S::kCancelled || to == S::kFailed;
        case S::kCompleted:
        case S::kCancelled:
        case S::kRejected:
        case S::kFailed:    return false;
    }
    return false;
}

IllegalTransition::IllegalTransition(const std::string& request_id, RequestState from, RequestState to)
    : std::logic_error(transition_message(request_id, from, to)), from_(from), to_(to) {}

Request::Request(std::string id, Clock::time_point now) : id_(std::move(id)), last_transition_at_(now) {
    if (id_.empty()) {
        throw std::invalid_argument("request id must not be empty");
    }
    entered_at_[index_of(RequestState::kReceived)] = now;
}

void Request::transition_to(RequestState next, Clock::time_point now) {
    if (!is_legal_transition(state_, next)) {
        throw IllegalTransition(id_, state_, next);
    }
    if (now < last_transition_at_) {
        throw std::invalid_argument("request " + id_ + ": transition time is earlier than the previous transition");
    }
    state_ = next;
    last_transition_at_ = now;
    auto& first_entry = entered_at_[index_of(next)];
    if (!first_entry) {
        first_entry = now;  // keep the first entry: e.g. queue wait counts from the first QUEUED
    }
    if (next == RequestState::kPreempted) {
        ++preemption_count_;
    }
}

std::optional<Request::Clock::time_point> Request::entered_at(RequestState state) const noexcept {
    return entered_at_[index_of(state)];
}

std::string RequestIdGenerator::next() {
    const std::uint64_t n = next_.fetch_add(1, std::memory_order_relaxed);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "ns-%06llu", static_cast<unsigned long long>(n));
    return buf;
}

}  // namespace neuroserve::core
