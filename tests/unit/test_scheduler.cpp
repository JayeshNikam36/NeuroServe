// Scheduling policies: FCFS unit tests, decision checks, golden traces, and the engine loop's use
// of a policy.

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "fake_backend.hpp"
#include "neuroserve/core/engine_loop.hpp"
#include "neuroserve/core/scheduler.hpp"
#include "scheduler_sim.hpp"

namespace {

using neuroserve::core::check_decision;
using neuroserve::core::DecodeFirstPolicy;
using neuroserve::core::FcfsPolicy;
using neuroserve::core::make_policy;
using neuroserve::core::PriorityPolicy;
using neuroserve::core::QueuedRequest;
using neuroserve::core::RunningRequest;
using neuroserve::core::ScheduleDecision;
using neuroserve::core::SchedulerInput;
using neuroserve::core::SchedulingPolicy;
using neuroserve::core::TokenBudgetPolicy;
using namespace std::chrono_literals;

// A snapshot with queued requests needing `queued_pages` each, and `running` requests that each
// hold `held` pages and need `grow` new pages in their next decode step.
SchedulerInput input(int max_batch, int free_pages, std::vector<int> queued_pages, int running = 0, int held = 1,
                     int grow = 0) {
    SchedulerInput in;
    in.max_batch = max_batch;
    in.free_pages = free_pages;
    in.max_step_tokens = 1 << 20;  // budget not the limit here; the token tests set it
    for (std::size_t i = 0; i < queued_pages.size(); ++i) {
        in.queued.push_back(QueuedRequest{"q" + std::to_string(i), 10, 10, queued_pages[i]});
    }
    for (int i = 0; i < running; ++i) {
        in.running.push_back(RunningRequest{"r" + std::to_string(i), held, grow, 1, 10});
    }
    return in;
}

ScheduleDecision decide(std::vector<std::size_t> preempt, std::vector<std::size_t> admit) {
    ScheduleDecision decision;
    decision.preempt = std::move(preempt);
    decision.admit = std::move(admit);
    return decision;
}

using Indices = std::vector<std::size_t>;

ScheduleDecision fcfs(const SchedulerInput& in) { return FcfsPolicy().schedule(in); }

// ---- FCFS: admission -------------------------------------------------------------------------

TEST(Fcfs, AdmitsInArrivalOrderUpToMaxBatch) {
    EXPECT_EQ(fcfs(input(3, 100, {1, 1, 1, 1, 1})).admit, (Indices{0, 1, 2}));
}

TEST(Fcfs, RunningRequestsUseSlotsAndTheirGrowthUsesPages) {
    EXPECT_EQ(fcfs(input(3, 100, {1, 1, 1}, /*running=*/2)).admit, (Indices{0}));
    EXPECT_TRUE(fcfs(input(3, 100, {1}, /*running=*/3)).admit.empty());
    // 3 free pages, but the 2 running requests each need a new page this step: 1 page left.
    EXPECT_EQ(fcfs(input(8, 3, {1, 1, 1}, /*running=*/2, /*held=*/2, /*grow=*/1)).admit, (Indices{0}));
}

TEST(Fcfs, StopsAtTheFirstRequestThatDoesNotFit) {
    // q1 needs 8 pages but only 5 are left after q0: q2 (1 page) would fit, but must not overtake q1.
    EXPECT_EQ(fcfs(input(8, 7, {2, 8, 1})).admit, (Indices{0}));
}

TEST(Fcfs, EmptyQueueAdmitsNothing) { EXPECT_TRUE(fcfs(input(8, 100, {})).admit.empty()); }

// ---- FCFS: token budget ----------------------------------------------------------------------

SchedulerInput prompts(int budget, std::vector<int> prompt_tokens, int running = 0) {
    auto in = input(8, 1000, std::vector<int>(prompt_tokens.size(), 1), running);
    in.max_step_tokens = budget;
    for (std::size_t i = 0; i < prompt_tokens.size(); ++i) {
        in.queued[i].prompt_tokens = prompt_tokens[i];
    }
    return in;
}

TEST(Fcfs, PrefillsFitTheTokenBudget) {
    EXPECT_EQ(fcfs(prompts(100, {40, 50, 30})).admit, (Indices{0, 1}));  // 40 + 50; 30 more would be 120
}

TEST(Fcfs, DecodeTokensAreCountedFirst) {
    // 6 running requests each decode one token: 94 of 100 left, so the 90-token prompt fits...
    EXPECT_EQ(fcfs(prompts(100, {90}, /*running=*/6)).admit, (Indices{0}));
    // ...but with 6 + 90 = 96 used, a 5-token prompt behind it does not.
    EXPECT_EQ(fcfs(prompts(100, {90, 5}, /*running=*/6)).admit, (Indices{0}));
    EXPECT_TRUE(fcfs(prompts(100, {95}, /*running=*/6)).admit.empty());
}

TEST(Fcfs, ALongPromptBlocksShorterOnesBehindIt) {
    // Strict arrival order applies to the token budget too: the 80-token prompt waits, and so
    // does the 10-token prompt behind it.
    EXPECT_TRUE(fcfs(prompts(100, {80, 10}, /*running=*/30)).admit.empty());
}

// ---- FCFS: preemption ------------------------------------------------------------------------

TEST(Fcfs, PreemptsTheNewestWhenTheBatchCannotGrow) {
    // 3 running requests hold 2 pages each and each needs 1 more; only 1 page is free.
    // Preempting r2 frees 2 pages: r0 and r1 can grow. Nothing is admitted while memory is short.
    const auto decision = fcfs(input(8, 1, {1}, /*running=*/3, /*held=*/2, /*grow=*/1));
    EXPECT_EQ(decision.preempt, (Indices{2}));
    EXPECT_TRUE(decision.admit.empty());
}

TEST(Fcfs, NeverPreemptsTheOldestRunningRequest) {
    // No free pages at all: preempt newest first until the rest fit, but always keep r0.
    const auto decision = fcfs(input(8, 0, {}, /*running=*/3, /*held=*/1, /*grow=*/1));
    EXPECT_EQ(decision.preempt, (Indices{2, 1}));
    EXPECT_NO_THROW(check_decision(input(8, 0, {}, 3, 1, 1), decision));
}

TEST(Fcfs, DoesNotPreemptWhenTheBatchCanGrow) {
    EXPECT_TRUE(fcfs(input(8, 3, {}, /*running=*/3, /*held=*/2, /*grow=*/1)).preempt.empty());
}

// ---- Decision checks and the factory ----------------------------------------------------------

TEST(CheckDecision, RejectsDecisionsThatBreakALimit) {
    const auto in = input(2, 5, {2, 2, 2});
    EXPECT_NO_THROW(check_decision(in, decide({}, {0, 1})));
    EXPECT_THROW(check_decision(in, decide({}, {3})), std::logic_error);        // no such queued request
    EXPECT_THROW(check_decision(in, decide({}, {1, 1})), std::logic_error);     // admitted twice
    EXPECT_THROW(check_decision(in, decide({}, {0, 1, 2})), std::logic_error);  // over max_batch
    EXPECT_THROW(check_decision(input(8, 3, {2, 2}), decide({}, {0, 1})), std::logic_error);  // pages

    auto budget = prompts(50, {30, 30});
    EXPECT_NO_THROW(check_decision(budget, decide({}, {0})));
    EXPECT_THROW(check_decision(budget, decide({}, {0, 1})), std::logic_error);  // 60 tokens > 50

    const auto busy = input(8, 0, {}, /*running=*/2, /*held=*/2, /*grow=*/1);  // needs 2, has 0
    EXPECT_THROW(check_decision(busy, decide({}, {})), std::logic_error);       // cannot grow
    EXPECT_NO_THROW(check_decision(busy, decide({1}, {})));                     // preempting r1 frees 2
    EXPECT_THROW(check_decision(busy, decide({2}, {})), std::logic_error);      // no such running request
}

TEST(MakePolicy, KnowsEveryPolicyAndRejectsUnknownNames) {
    for (const auto name : neuroserve::core::kPolicyNames) {
        EXPECT_EQ(make_policy(name)->name(), name);
    }
    EXPECT_THROW(make_policy("lifo"), std::invalid_argument);
}

// ---- Token-budget policy ---------------------------------------------------------------------

TEST(TokenBudget, SkipsWhatDoesNotFitAndPacksTheBudget) {
    // FCFS stops at q1 (80 tokens > 60 left); token-budget skips it and admits q2 and q3.
    const auto in = prompts(100, {40, 80, 10, 20});
    EXPECT_EQ(fcfs(in).admit, (Indices{0}));
    EXPECT_EQ(TokenBudgetPolicy().schedule(in).admit, (Indices{0, 2, 3}));
}

TEST(TokenBudget, AnAgedRequestCannotBeOvertaken) {
    auto in = prompts(100, {40, 80, 10});
    in.queued[1].waited_steps = 4;  // q1 has waited long enough: from it on, FCFS order applies
    EXPECT_EQ(TokenBudgetPolicy(/*max_wait_steps=*/4).schedule(in).admit, (Indices{0}));
    EXPECT_EQ(TokenBudgetPolicy(/*max_wait_steps=*/5).schedule(in).admit, (Indices{0, 2}));
}

// ---- Decode-first policy ---------------------------------------------------------------------

TEST(DecodeFirst, CapsPrefillTokensWhileRequestsAreDecoding) {
    // Budget 100 with 2 decoding: prefills are capped at 100 / 4 = 25 tokens this step.
    EXPECT_EQ(DecodeFirstPolicy().schedule(prompts(100, {20, 10}, /*running=*/2)).admit, (Indices{0}));
    EXPECT_TRUE(DecodeFirstPolicy().schedule(prompts(100, {30, 5}, /*running=*/2)).admit.empty());  // strict order
    EXPECT_EQ(fcfs(prompts(100, {30, 5}, /*running=*/2)).admit, (Indices{0, 1}));  // FCFS would take both
}

TEST(DecodeFirst, UsesTheWholeBudgetWhenNothingIsRunning) {
    EXPECT_EQ(DecodeFirstPolicy().schedule(prompts(100, {60, 30})).admit, (Indices{0, 1}));
}

TEST(Policies, AllPreemptTheSameWayUnderMemoryPressure) {
    // All requests have the same priority, so the priority policy preempts like the others.
    const auto in = input(8, 1, {1}, /*running=*/3, /*held=*/2, /*grow=*/1);
    for (const auto name : neuroserve::core::kPolicyNames) {
        const auto decision = make_policy(name)->schedule(in);
        EXPECT_EQ(decision.preempt, (Indices{2})) << name;
        EXPECT_TRUE(decision.admit.empty()) << name;
    }
}

// ---- Priority policy --------------------------------------------------------------------------

using neuroserve::core::kHighPriority;
using neuroserve::core::kLowPriority;
using neuroserve::core::kNormalPriority;

SchedulerInput with_priorities(SchedulerInput in, std::vector<int> queued, std::vector<int> running = {}) {
    for (std::size_t i = 0; i < queued.size(); ++i) {
        in.queued[i].priority = queued[i];
    }
    for (std::size_t i = 0; i < running.size(); ++i) {
        in.running[i].priority = running[i];
    }
    return in;
}

TEST(Priority, AdmitsHigherPriorityFirstThenInQueueOrder) {
    const auto in = with_priorities(input(3, 100, {1, 1, 1, 1}), {kLowPriority, kNormalPriority, kHighPriority,
                                                                  kNormalPriority});
    EXPECT_EQ(PriorityPolicy().schedule(in).admit, (Indices{2, 1, 3}));
    EXPECT_EQ(fcfs(in).admit, (Indices{0, 1, 2}));  // FCFS ignores priorities
}

TEST(Priority, StopsAtTheFirstRequestThatDoesNotFitInPriorityOrder) {
    // The high-priority q1 needs 8 pages but only 5 are free: nothing may overtake it.
    const auto in = with_priorities(input(8, 5, {1, 8, 1}), {kNormalPriority, kHighPriority, kNormalPriority});
    EXPECT_TRUE(PriorityPolicy().schedule(in).admit.empty());
}

TEST(Priority, AgingRaisesPriorityOneLevelPerInterval) {
    const PriorityPolicy policy(/*aging_ms=*/100);
    EXPECT_EQ(policy.effective_priority(kLowPriority, 0), kLowPriority);
    EXPECT_EQ(policy.effective_priority(kLowPriority, 99), kLowPriority);
    EXPECT_EQ(policy.effective_priority(kLowPriority, 100), kNormalPriority);
    EXPECT_EQ(policy.effective_priority(kLowPriority, 250), kHighPriority);
    EXPECT_EQ(policy.effective_priority(kLowPriority, 1'000'000), kHighPriority);  // capped at high
    EXPECT_EQ(policy.effective_priority(kHighPriority, 0), kHighPriority);
    EXPECT_THROW(PriorityPolicy(0), std::invalid_argument);
}

TEST(Priority, AnAgedRequestIsServedBeforeNewerHighPriorityOnes) {
    auto in = with_priorities(input(1, 100, {1, 1}), {kLowPriority, kHighPriority});
    EXPECT_EQ(PriorityPolicy(/*aging_ms=*/100).schedule(in).admit, (Indices{1}));  // not aged yet
    in.queued[0].age_ms = 200;  // two intervals: now high too, and ahead in the queue
    EXPECT_EQ(PriorityPolicy(/*aging_ms=*/100).schedule(in).admit, (Indices{0}));
}

TEST(Priority, PreemptsTheLowestPriorityFirstUnderMemoryPressure) {
    // As in Fcfs.PreemptsTheNewestWhenTheBatchCannotGrow, but r1 is low priority and r2 is high.
    const auto in = with_priorities(input(8, 1, {1}, /*running=*/3, /*held=*/2, /*grow=*/1), {kNormalPriority},
                                    {kNormalPriority, kLowPriority, kHighPriority});
    EXPECT_EQ(PriorityPolicy().schedule(in).preempt, (Indices{1}));
    EXPECT_EQ(fcfs(in).preempt, (Indices{2}));  // FCFS preempts the newest, whatever its priority
}

TEST(Priority, NeverPreemptsTheOldestRunningRequest) {
    // r0 has the lowest priority, but it is the oldest: the others go first, newest first.
    const auto in = with_priorities(input(8, 0, {}, /*running=*/3, /*held=*/1, /*grow=*/1), {},
                                    {kLowPriority, kHighPriority, kHighPriority});
    const auto decision = PriorityPolicy().schedule(in);
    EXPECT_EQ(decision.preempt, (Indices{2, 1}));
    EXPECT_NO_THROW(check_decision(in, decision));
}

// The starvation test: one low-priority request and a stream of high-priority requests that keeps
// the single slot busy. With aging the low request is admitted after 2 x aging_ms; without it, only
// once the stream ends.
int step_admitted(const std::string& log, const std::string& id) {
    std::istringstream lines(log);
    for (std::string line; std::getline(lines, line);) {
        if (line.find(" admit=[" + id + "]") != std::string::npos) {
            return std::stoi(line.substr(5));  // "step N ..."
        }
    }
    return -1;
}

TEST(Priority, ALowPriorityRequestIsNotStarvedByAStreamOfUrgentOnes) {
    neuroserve::test_support::TraceSpec spec;
    spec.max_batch = 1;
    spec.total_pages = 64;
    spec.max_batch_tokens = 64;
    spec.arrivals.push_back({0, "low", 4, 2, kLowPriority});
    for (int i = 0; i < 50; ++i) {  // one urgent request per step; each needs two steps
        spec.arrivals.push_back({i, "h" + std::to_string(i), 4, 2, kHighPriority});
    }
    constexpr int kAgingMs = 40;  // 4 simulated steps
    PriorityPolicy aging(kAgingMs);
    const int with_aging = step_admitted(neuroserve::test_support::simulate(spec, aging), "low");
    EXPECT_GE(with_aging, 0);
    EXPECT_LE(with_aging * neuroserve::test_support::kSimStepMs, 2 * kAgingMs);  // the starvation bound

    PriorityPolicy no_aging(/*aging_ms=*/1'000'000);
    EXPECT_GE(step_admitted(neuroserve::test_support::simulate(spec, no_aging), "low"), 100);  // after all 50
}

// ---- Golden traces ----------------------------------------------------------------------------
//
// Each file in tests/data/scheduler_traces holds a workload and the log its policy must produce.
// After an intended change to a policy, regenerate the expected logs and review the diff:
//   NS_UPDATE_GOLDEN=1 ./build/<preset>/tests/ns_unit_tests --gtest_filter='GoldenTrace.*'

void check_golden(const std::string& name) {
    const std::string path = std::string(NS_TEST_DATA_DIR) + "/scheduler_traces/" + name;
    const auto trace = neuroserve::test_support::parse_trace(neuroserve::test_support::read_file(path));
    const auto policy = neuroserve::test_support::make_trace_policy(trace.spec);
    const std::string actual = neuroserve::test_support::simulate(trace.spec, *policy);

    const char* update = std::getenv("NS_UPDATE_GOLDEN");
    if (update != nullptr && std::string(update) == "1") {
        std::ofstream(path) << trace.header << actual;
        std::cout << "updated " << path << "\n";
        return;
    }
    EXPECT_EQ(actual, trace.expected) << "policy decisions differ from " << path;
}

TEST(GoldenTrace, BurstLimitedByMaxBatch) { check_golden("burst_max_batch.trace"); }
TEST(GoldenTrace, KvBoundHeadOfLineBlocking) { check_golden("kv_head_of_line.trace"); }
TEST(GoldenTrace, StaggeredArrivals) { check_golden("staggered_arrivals.trace"); }
TEST(GoldenTrace, PreemptionUnderKvPressure) { check_golden("kv_preemption.trace"); }
TEST(GoldenTrace, TokenBudget) { check_golden("token_budget.trace"); }
TEST(GoldenTrace, TokenBudgetPolicyPacksTheBudget) { check_golden("token_budget_policy.trace"); }
TEST(GoldenTrace, TokenBudgetPolicyAging) { check_golden("token_budget_aging.trace"); }
TEST(GoldenTrace, DecodeFirstPolicy) { check_golden("decode_first_policy.trace"); }
TEST(GoldenTrace, PriorityOrder) { check_golden("priority_order.trace"); }
TEST(GoldenTrace, PriorityAging) { check_golden("priority_aging.trace"); }
TEST(GoldenTrace, PriorityPreemption) { check_golden("priority_preemption.trace"); }

// ---- The engine loop uses the policy ------------------------------------------------------------

// Admits the most recent arrival first (newest first), and records what it admitted.
class NewestFirstPolicy final : public SchedulingPolicy {
public:
    std::string_view name() const noexcept override { return "newest-first"; }
    ScheduleDecision schedule(const SchedulerInput& in) override {
        ScheduleDecision decision;
        auto slots = in.max_batch - static_cast<long long>(in.running.size());
        for (std::size_t i = in.queued.size(); i-- > 0 && slots > 0; --slots) {
            decision.admit.push_back(i);
            admitted.push_back(in.queued[i].id);
        }
        return decision;
    }
    std::vector<std::string> admitted;  // engine thread only; read after the loop has stopped
};

// Asks the engine to admit a request that does not exist.
class FaultyPolicy final : public SchedulingPolicy {
public:
    std::string_view name() const noexcept override { return "faulty"; }
    ScheduleDecision schedule(const SchedulerInput&) override { return decide({}, {99}); }
};

TEST(EngineLoopPolicy, ThePolicyDecidesTheAdmissionOrder) {
    neuroserve::test_support::FakeBackend backend;
    std::atomic<bool> first{true};
    std::promise<void> entered;
    std::promise<void> release;
    std::shared_future<void> released = release.get_future().share();
    backend.on_decode = [&](std::size_t) {
        if (first.exchange(false)) {
            entered.set_value();
            released.wait();
        }
    };
    auto policy = std::make_unique<NewestFirstPolicy>();
    NewestFirstPolicy* recorder = policy.get();
    neuroserve::core::EngineLoop loop(backend, {.max_batch = 1}, std::move(policy));
    EXPECT_EQ(loop.policy_name(), "newest-first");

    neuroserve::core::GenerationParams p;
    p.prompt = {1};
    p.max_new_tokens = 3;
    auto a = loop.submit(p);
    ASSERT_EQ(entered.get_future().wait_for(10s), std::future_status::ready);  // a is running
    auto b = loop.submit(p);
    auto c = loop.submit(p);
    release.set_value();
    for (const auto* handle : {&a, &b, &c}) {
        ASSERT_TRUE(handle->wait_for(10s));
    }
    loop.stop();
    EXPECT_EQ(recorder->admitted, (std::vector<std::string>{"ns-000001", "ns-000003", "ns-000002"}));
}

TEST(EngineLoopPolicy, ThePolicyCanBeSwitchedWhileServing) {
    neuroserve::test_support::FakeBackend backend;
    neuroserve::core::EngineLoop loop(backend);
    EXPECT_EQ(loop.policy_name(), "fcfs");
    neuroserve::core::GenerationParams p;
    p.prompt = {1};
    p.max_new_tokens = 3;
    ASSERT_TRUE(loop.submit(p).wait_for(10s));

    loop.set_policy(make_policy("decode-first"));
    auto handle = loop.submit(p);  // the switch happens at the next engine step
    ASSERT_TRUE(handle.wait_for(10s));
    EXPECT_EQ(handle.wait().tokens, (std::vector<neuroserve::TokenId>{2, 3, 4}));
    EXPECT_EQ(loop.policy_name(), "decode-first");
    EXPECT_THROW(loop.set_policy(nullptr), std::invalid_argument);
}

TEST(EngineLoopPolicy, AFaultyPolicyStopsTheEngineLoudly) {
    neuroserve::test_support::FakeBackend backend;
    neuroserve::core::EngineLoop loop(backend, {}, std::make_unique<FaultyPolicy>());
    neuroserve::core::GenerationParams p;
    p.prompt = {1};
    p.max_new_tokens = 3;
    auto handle = loop.submit(p);
    ASSERT_TRUE(handle.wait_for(10s));
    EXPECT_THROW(handle.wait(), neuroserve::core::EngineStopped);  // cancelled, not left hanging
    EXPECT_TRUE(loop.stop_requested());
    EXPECT_THROW(loop.submit(p), neuroserve::core::EngineStopped);
    loop.stop();
    EXPECT_EQ(backend.live_sequences(), 0);
}

}  // namespace
