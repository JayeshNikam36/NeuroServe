#include "neuroserve/core/scheduler.hpp"

#include <stdexcept>

namespace neuroserve::core {

namespace {

// Marks each index as taken; throws for an index out of range or listed twice.
std::vector<bool> check_indices(const std::vector<std::size_t>& indices, std::size_t size,
                                const std::string& what) {
    std::vector<bool> taken(size, false);
    for (const std::size_t index : indices) {
        if (index >= size) {
            throw std::logic_error("scheduling decision " + what + " index " + std::to_string(index) +
                                   ", but there are only " + std::to_string(size));
        }
        if (taken[index]) {
            throw std::logic_error("scheduling decision " + what + " index " + std::to_string(index) + " twice");
        }
        taken[index] = true;
    }
    return taken;
}

}  // namespace

void check_decision(const SchedulerInput& input, const ScheduleDecision& decision) {
    const auto preempted = check_indices(decision.preempt, input.running.size(), "preempts running");
    check_indices(decision.admit, input.queued.size(), "admits queued");

    long long available = input.free_pages;
    long long needed = 0;
    for (std::size_t i = 0; i < input.running.size(); ++i) {
        if (preempted[i]) {
            available += input.running[i].pages;  // given back
        } else {
            needed += input.running[i].next_step_pages;
        }
    }
    for (const std::size_t index : decision.admit) {
        needed += input.queued[index].pages;
    }

    const auto running = input.running.size() - decision.preempt.size() + decision.admit.size();
    if (running > static_cast<std::size_t>(input.max_batch)) {
        throw std::logic_error("scheduling decision would run " + std::to_string(running) +
                               " requests; max_batch is " + std::to_string(input.max_batch));
    }
    if (needed > available) {
        throw std::logic_error("scheduling decision needs " + std::to_string(needed) +
                               " new KV pages this step; " + std::to_string(available) + " are available");
    }
}

ScheduleDecision FcfsPolicy::schedule(const SchedulerInput& input) {
    ScheduleDecision decision;
    long long free_pages = input.free_pages;
    long long growth = 0;
    for (const auto& r : input.running) {
        growth += r.next_step_pages;
    }

    // 1. Memory pressure: preempt the most recently admitted requests until the rest can grow.
    std::size_t kept = input.running.size();
    while (growth > free_pages && kept > 1) {
        --kept;
        decision.preempt.push_back(kept);
        free_pages += input.running[kept].pages;
        growth -= input.running[kept].next_step_pages;
    }
    if (!decision.preempt.empty()) {
        return decision;  // no admissions while memory is short
    }

    // 2. Admit in strict arrival order while there are slots and pages.
    free_pages -= growth;
    auto slots = input.max_batch - static_cast<long long>(input.running.size());
    for (std::size_t i = 0; i < input.queued.size() && slots > 0; ++i) {
        if (input.queued[i].pages > free_pages) {
            break;  // nobody behind this request may overtake it
        }
        decision.admit.push_back(i);
        free_pages -= input.queued[i].pages;
        --slots;
    }
    return decision;
}

std::unique_ptr<SchedulingPolicy> make_policy(std::string_view name) {
    if (name == "fcfs") {
        return std::make_unique<FcfsPolicy>();
    }
    throw std::invalid_argument("unknown scheduling policy '" + std::string(name) + "' (available: fcfs)");
}

}  // namespace neuroserve::core
