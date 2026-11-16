#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// Scheduling policies decide, once per engine step, which running requests to preempt and which
// queued requests to admit.
//
// A policy is a pure decision function: it sees a snapshot of the queue and the running batch
// (SchedulerInput) and returns a ScheduleDecision. It never touches the backend, the KV cache or a
// request, so it can be unit-tested and replayed against recorded traces without a GPU. The engine
// owns the mechanism (admitting, preempting, prefilling, decoding) and checks every decision before
// applying it.
//
// KV pages are allocated on demand: a request holds pages for the tokens it has cached and takes
// one more page each time it crosses a page boundary. The snapshot therefore says, for this step,
// how many new pages each running request needs and how many each queued request would need.
//
// One step = the prefills of the requests admitted in it + one decode token for every request that
// was already running. The step's token budget (max_step_tokens) bounds that sum, so one long
// prompt cannot stall every running request; decode tokens are counted first.
namespace neuroserve::core {

struct QueuedRequest {
    std::string id;
    int prompt_tokens = 0;
    int max_new_tokens = 0;
    int pages = 0;  // pages its prefill needs if admitted now
};

struct RunningRequest {
    std::string id;
    int pages = 0;            // pages it holds now
    int next_step_pages = 0;  // new pages its decode in this step needs (0 or 1)
    int generated_tokens = 0;
    int max_new_tokens = 0;
};

struct SchedulerInput {
    std::vector<QueuedRequest> queued;    // in arrival order (preempted requests first)
    std::vector<RunningRequest> running;  // in admission order (most recently admitted last)
    int max_batch = 0;                    // most requests running at once
    int free_pages = 0;                   // KV pages not held by any request
    int max_step_tokens = 0;              // decode tokens of kept running requests + admitted prompt tokens
};

struct ScheduleDecision {
    std::vector<std::size_t> preempt;  // indices into SchedulerInput::running
    std::vector<std::size_t> admit;    // indices into SchedulerInput::queued, in admission order
};

// Throws std::logic_error if the decision breaks a limit: an index out of range or repeated, more
// running requests than max_batch, a step that needs more KV pages than are available (free pages
// plus the pages preempted requests give back), or more tokens than max_step_tokens. The engine calls this on every decision, so
// a faulty policy fails loudly instead of corrupting the engine's state.
void check_decision(const SchedulerInput& input, const ScheduleDecision& decision);

class SchedulingPolicy {
public:
    virtual ~SchedulingPolicy() = default;
    virtual std::string_view name() const noexcept = 0;
    // Called on the engine thread once per step. Must be deterministic for a given input.
    virtual ScheduleDecision schedule(const SchedulerInput& input) = 0;
};

// First come, first served, with preemption under memory pressure:
//   1. If the running batch cannot take its next decode step, preempt the most recently admitted
//      requests until it can (they restart later), and admit nothing this step. The oldest
//      running request is never preempted, so it always finishes: no livelock.
//   2. Otherwise admit queued requests in arrival order while they fit (slots, pages, and the token
//      budget left after one decode token per running request); stop at the first one that does
//      not, so a large request is never overtaken (no starvation), at the cost of head-of-line
//      blocking.
class FcfsPolicy final : public SchedulingPolicy {
public:
    std::string_view name() const noexcept override { return "fcfs"; }
    ScheduleDecision schedule(const SchedulerInput& input) override;
};

// Creates a policy by its config name ("fcfs"). Throws std::invalid_argument for an unknown name.
std::unique_ptr<SchedulingPolicy> make_policy(std::string_view name);

}  // namespace neuroserve::core
