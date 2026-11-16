#include "neuroserve/core/scheduler.hpp"

#include <stdexcept>

namespace neuroserve::core {

void check_decision(const SchedulerInput& input, const ScheduleDecision& decision) {
    std::vector<bool> taken(input.queued.size(), false);
    long long pages = 0;
    for (const std::size_t index : decision.admit) {
        if (index >= input.queued.size()) {
            throw std::logic_error("scheduling decision admits queue index " + std::to_string(index) +
                                   ", but only " + std::to_string(input.queued.size()) + " requests are queued");
        }
        if (taken[index]) {
            throw std::logic_error("scheduling decision admits " + input.queued[index].id + " twice");
        }
        taken[index] = true;
        pages += input.queued[index].pages;
    }
    const auto running = input.running.size() + decision.admit.size();
    if (running > static_cast<std::size_t>(input.max_batch)) {
        throw std::logic_error("scheduling decision would run " + std::to_string(running) +
                               " requests; max_batch is " + std::to_string(input.max_batch));
    }
    if (pages > input.free_pages) {
        throw std::logic_error("scheduling decision needs " + std::to_string(pages) + " KV pages; " +
                               std::to_string(input.free_pages) + " are free");
    }
}

ScheduleDecision FcfsPolicy::schedule(const SchedulerInput& input) {
    ScheduleDecision decision;
    auto slots = input.max_batch - static_cast<long long>(input.running.size());
    long long free_pages = input.free_pages;
    for (std::size_t i = 0; i < input.queued.size() && slots > 0; ++i) {
        if (input.queued[i].pages > free_pages) {
            break;  // strict arrival order: nobody behind this request may overtake it
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
