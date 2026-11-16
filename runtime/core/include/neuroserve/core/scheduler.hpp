#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// Scheduling policies decide which queued requests the engine admits next.
//
// A policy is a pure decision function: it sees a snapshot of the queue and the running batch
// (SchedulerInput) and returns a ScheduleDecision. It never touches the backend, the KV cache or a
// request, so it can be unit-tested and replayed against recorded traces without a GPU. The engine
// owns the mechanism (admitting, prefilling, decoding) and checks every decision before applying it.
namespace neuroserve::core {

struct QueuedRequest {
    std::string id;
    int prompt_tokens = 0;
    int max_new_tokens = 0;
    int pages = 0;  // KV pages the request reserves when admitted
};

struct RunningRequest {
    std::string id;
    int pages = 0;
    int generated_tokens = 0;
    int max_new_tokens = 0;
};

struct SchedulerInput {
    std::vector<QueuedRequest> queued;    // in arrival order
    std::vector<RunningRequest> running;  // admitted and still generating
    int max_batch = 0;                    // most requests running at once
    int free_pages = 0;                   // KV pages not reserved by running requests
};

struct ScheduleDecision {
    std::vector<std::size_t> admit;  // indices into SchedulerInput::queued, in admission order
};

// Throws std::logic_error if the decision breaks a limit: an index out of range or repeated,
// more running requests than max_batch, or more pages than are free. The engine calls this on
// every decision, so a faulty policy fails loudly instead of corrupting the engine's state.
void check_decision(const SchedulerInput& input, const ScheduleDecision& decision);

class SchedulingPolicy {
public:
    virtual ~SchedulingPolicy() = default;
    virtual std::string_view name() const noexcept = 0;
    // Called on the engine thread once per step. Must be deterministic for a given input.
    virtual ScheduleDecision schedule(const SchedulerInput& input) = 0;
};

// First come, first served: admits queued requests in arrival order while they fit, and stops at
// the first one that does not, so a large request is never overtaken (no starvation), at the cost
// of head-of-line blocking.
class FcfsPolicy final : public SchedulingPolicy {
public:
    std::string_view name() const noexcept override { return "fcfs"; }
    ScheduleDecision schedule(const SchedulerInput& input) override;
};

// Creates a policy by its config name ("fcfs"). Throws std::invalid_argument for an unknown name.
std::unique_ptr<SchedulingPolicy> make_policy(std::string_view name);

}  // namespace neuroserve::core
