// EngineLoop::step_profile: where the engine thread's time goes, step by step (FakeBackend, no GPU).

#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "fake_backend.hpp"
#include "neuroserve/core/engine_loop.hpp"

namespace {

using neuroserve::TokenId;
using neuroserve::core::EngineLoop;
using neuroserve::core::EngineLoopOptions;
using neuroserve::core::GenerationParams;
using neuroserve::core::RequestHandle;
using neuroserve::core::StepProfile;
using neuroserve::core::StepRecord;
using neuroserve::test_support::FakeBackend;
using namespace std::chrono_literals;

GenerationParams params(std::vector<TokenId> prompt, int max_new_tokens) {
    GenerationParams p;
    p.prompt = std::move(prompt);
    p.max_new_tokens = max_new_tokens;
    return p;
}

void finish_all(const std::vector<RequestHandle>& handles) {
    for (const auto& h : handles) {
        ASSERT_TRUE(h.wait_for(10s));
        h.wait();
    }
}

// Every phase is >= 0 and they add up to the step's total.
void expect_consistent(const StepRecord& r) {
    for (const double v : {r.housekeeping_ms, r.schedule_ms, r.preempt_ms, r.prefill_ms, r.decode_ms, r.host_ms,
                           r.invariants_ms}) {
        EXPECT_GE(v, 0.0);
    }
    const double sum = r.housekeeping_ms + r.schedule_ms + r.preempt_ms + r.prefill_ms + r.decode_ms + r.host_ms +
                       r.invariants_ms;
    EXPECT_NEAR(sum, r.total_ms, 1e-6 + 1e-9 * r.total_ms);
}

TEST(StepProfile, SplitsTheBackendsPrefillAndDecodeFromTheHost) {
    FakeBackend backend;
    backend.on_prefill = [] { std::this_thread::sleep_for(5ms); };
    backend.on_decode = [](std::size_t) { std::this_thread::sleep_for(2ms); };
    EngineLoopOptions options;
    options.max_batch = 4;
    EngineLoop loop(backend, options);
    std::vector<RequestHandle> handles;
    for (int i = 0; i < 3; ++i) {
        handles.push_back(loop.submit(params({1, 2, 3, 4 + i}, 6)));
    }
    finish_all(handles);
    const StepProfile p = loop.step_profile(/*with_records=*/true);

    EXPECT_EQ(p.totals.prefill_requests, 3);
    EXPECT_EQ(p.totals.prefill_tokens, 12);
    EXPECT_EQ(p.totals.decode_batch, 3 * 5);  // the first token of each comes from its prefill
    EXPECT_GE(p.totals.prefill_ms, 3 * 5.0);
    const auto decode_steps = static_cast<double>(loop.stats().decode_steps);
    EXPECT_GE(p.totals.decode_ms, decode_steps * 2.0);
    EXPECT_GE(p.busy_steps, static_cast<std::uint64_t>(decode_steps));
    EXPECT_GE(p.steps, p.busy_steps);
    expect_consistent(p.totals);
    // Backend time dominates here: the fake's sleeps are far longer than the loop's own work.
    EXPECT_GT(p.totals.prefill_ms + p.totals.decode_ms, 0.5 * p.totals.total_ms);
    EXPECT_LE(p.totals.total_ms + p.idle_ms, p.elapsed_ms + 1.0);

    ASSERT_EQ(p.records.size(), p.steps);
    double prefill = 0.0;
    int batch = 0;
    for (std::size_t i = 0; i < p.records.size(); ++i) {
        const StepRecord& r = p.records[i];
        expect_consistent(r);
        prefill += r.prefill_ms;
        batch += r.decode_batch;
        if (i > 0) {
            EXPECT_GE(r.start_ms, p.records[i - 1].start_ms + p.records[i - 1].total_ms - 1e-6);
        }
        if (r.decode_batch == 0) {
            EXPECT_EQ(r.decode_ms, 0.0);
        }
        if (r.prefill_requests == 0) {
            EXPECT_EQ(r.prefill_ms, 0.0);
        }
    }
    EXPECT_DOUBLE_EQ(prefill, p.totals.prefill_ms);
    EXPECT_EQ(batch, p.totals.decode_batch);
    EXPECT_TRUE(loop.step_profile(/*with_records=*/false).records.empty());
}

TEST(StepProfile, CountsTheTimeWaitingForWorkAsIdle) {
    FakeBackend backend;
    EngineLoop loop(backend);
    loop.reset_step_profile();
    std::this_thread::sleep_for(50ms);
    finish_all({loop.submit(params({1, 2}, 2))});
    const StepProfile p = loop.step_profile(false);
    EXPECT_GE(p.idle_ms, 40.0);
    EXPECT_GE(p.elapsed_ms, p.idle_ms);
    EXPECT_GE(p.busy_steps, 1U);
}

TEST(StepProfile, ResetStartsOver) {
    FakeBackend backend;
    EngineLoop loop(backend);
    finish_all({loop.submit(params({1, 2, 3}, 4))});
    ASSERT_GT(loop.step_profile(false).steps, 0U);
    loop.reset_step_profile();
    const StepProfile empty = loop.step_profile(true);
    EXPECT_EQ(empty.steps, 0U);
    EXPECT_EQ(empty.busy_steps, 0U);
    EXPECT_EQ(empty.totals.total_ms, 0.0);
    EXPECT_EQ(empty.totals.decode_batch, 0);
    EXPECT_TRUE(empty.records.empty());
    EXPECT_EQ(empty.dropped, 0U);
    finish_all({loop.submit(params({5, 6}, 3))});
    const StepProfile again = loop.step_profile(true);
    EXPECT_EQ(again.totals.prefill_requests, 1);
    EXPECT_EQ(again.totals.prefill_tokens, 2);
    ASSERT_FALSE(again.records.empty());
    EXPECT_GE(again.records.front().start_ms, 0.0);
}

TEST(StepProfile, KeepsOnlyTheNewestRecordsAndCountsTheRest) {
    FakeBackend backend;
    EngineLoopOptions options;
    options.step_profile_capacity = 4;
    EngineLoop loop(backend, options);
    finish_all({loop.submit(params({1, 2, 3}, 20))});
    const StepProfile p = loop.step_profile(true);
    ASSERT_GT(p.steps, 4U);
    ASSERT_EQ(p.records.size(), 4U);
    EXPECT_EQ(p.dropped, p.steps - 4);
    for (std::size_t i = 1; i < p.records.size(); ++i) {
        EXPECT_GT(p.records[i].start_ms, p.records[i - 1].start_ms);  // oldest first, across the wrap
    }
    // The totals still cover every step.
    EXPECT_EQ(p.totals.decode_batch, 19);
}

TEST(StepProfile, ZeroCapacityKeepsTotalsOnly) {
    FakeBackend backend;
    EngineLoopOptions options;
    options.step_profile_capacity = 0;
    EngineLoop loop(backend, options);
    finish_all({loop.submit(params({1, 2, 3}, 5))});
    const StepProfile p = loop.step_profile(true);
    EXPECT_TRUE(p.records.empty());
    EXPECT_EQ(p.dropped, 0U);
    EXPECT_EQ(p.totals.decode_batch, 4);
}

TEST(StepProfile, IsSafeToReadAndResetWhileServing) {
    FakeBackend backend;
    EngineLoop loop(backend);
    std::vector<RequestHandle> handles;
    for (int i = 0; i < 16; ++i) {
        handles.push_back(loop.submit(params({1, 2, 3 + i}, 30)));
    }
    for (int i = 0; i < 200; ++i) {
        const StepProfile p = loop.step_profile(i % 2 == 0);
        EXPECT_GE(p.elapsed_ms, 0.0);
        if (i % 50 == 0) {
            loop.reset_step_profile();
        }
    }
    finish_all(handles);
}

}  // namespace
