// Scheduling policies: FCFS unit tests, decision checks, golden traces, and the engine loop's use
// of a policy.

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fake_backend.hpp"
#include "neuroserve/core/engine_loop.hpp"
#include "neuroserve/core/scheduler.hpp"
#include "scheduler_sim.hpp"

namespace {

using neuroserve::core::check_decision;
using neuroserve::core::FcfsPolicy;
using neuroserve::core::make_policy;
using neuroserve::core::QueuedRequest;
using neuroserve::core::RunningRequest;
using neuroserve::core::ScheduleDecision;
using neuroserve::core::SchedulerInput;
using neuroserve::core::SchedulingPolicy;
using namespace std::chrono_literals;

SchedulerInput input(int max_batch, int free_pages, std::vector<int> queued_pages, int running = 0) {
    SchedulerInput in;
    in.max_batch = max_batch;
    in.free_pages = free_pages;
    for (std::size_t i = 0; i < queued_pages.size(); ++i) {
        in.queued.push_back(QueuedRequest{"q" + std::to_string(i), 10, 10, queued_pages[i]});
    }
    for (int i = 0; i < running; ++i) {
        in.running.push_back(RunningRequest{"r" + std::to_string(i), 1, 1, 10});
    }
    return in;
}

std::vector<std::size_t> fcfs(const SchedulerInput& in) { return FcfsPolicy().schedule(in).admit; }

// ---- FCFS ------------------------------------------------------------------------------------

TEST(Fcfs, AdmitsInArrivalOrderUpToMaxBatch) {
    EXPECT_EQ(fcfs(input(3, 100, {1, 1, 1, 1, 1})), (std::vector<std::size_t>{0, 1, 2}));
}

TEST(Fcfs, RunningRequestsUseUpSlots) {
    EXPECT_EQ(fcfs(input(3, 100, {1, 1, 1}, /*running=*/2)), (std::vector<std::size_t>{0}));
    EXPECT_TRUE(fcfs(input(3, 100, {1}, /*running=*/3)).empty());
}

TEST(Fcfs, StopsAtTheFirstRequestThatDoesNotFit) {
    // q1 needs 8 pages but only 5 are left after q0: q2 (1 page) would fit, but must not overtake q1.
    EXPECT_EQ(fcfs(input(8, 7, {2, 8, 1})), (std::vector<std::size_t>{0}));
}

TEST(Fcfs, EmptyQueueAdmitsNothing) { EXPECT_TRUE(fcfs(input(8, 100, {})).empty()); }

// ---- Decision checks and the factory ----------------------------------------------------------

TEST(CheckDecision, RejectsDecisionsThatBreakALimit) {
    const auto in = input(2, 5, {2, 2, 2}, /*running=*/0);
    EXPECT_NO_THROW(check_decision(in, ScheduleDecision{{0, 1}}));
    EXPECT_THROW(check_decision(in, ScheduleDecision{{3}}), std::logic_error);        // no such request
    EXPECT_THROW(check_decision(in, ScheduleDecision{{1, 1}}), std::logic_error);     // admitted twice
    EXPECT_THROW(check_decision(in, ScheduleDecision{{0, 1, 2}}), std::logic_error);  // over max_batch
    EXPECT_THROW(check_decision(input(8, 3, {2, 2}), ScheduleDecision{{0, 1}}), std::logic_error);  // pages
}

TEST(MakePolicy, KnowsFcfsAndRejectsUnknownNames) {
    EXPECT_EQ(make_policy("fcfs")->name(), "fcfs");
    EXPECT_THROW(make_policy("lifo"), std::invalid_argument);
}

// ---- Golden traces ----------------------------------------------------------------------------
//
// Each file in tests/data/scheduler_traces holds a workload and the log its policy must produce.
// After an intended change to a policy, regenerate the expected logs and review the diff:
//   NS_UPDATE_GOLDEN=1 ./build/<preset>/tests/ns_unit_tests --gtest_filter='GoldenTrace.*'

void check_golden(const std::string& name) {
    const std::string path = std::string(NS_TEST_DATA_DIR) + "/scheduler_traces/" + name;
    const auto trace = neuroserve::test_support::parse_trace(neuroserve::test_support::read_file(path));
    const auto policy = make_policy(trace.spec.policy);
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
    ScheduleDecision schedule(const SchedulerInput&) override { return ScheduleDecision{{99}}; }
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
