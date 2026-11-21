// Preemption by recompute (Day 38, SCH-008): a preempted request keeps the tokens it generated and
// resumes by prefilling its prompt and those tokens. Nothing is generated twice, so what a client
// was streamed can never diverge (ADR-0006), and with prefix caching the recompute reuses the
// request's own CACHED pages. docs/kv-cache.md, "Preemption".
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>

#include "fake_backend.hpp"
#include "prefill_gate.hpp"
#include "neuroserve/core/engine_loop.hpp"

namespace {

using neuroserve::TokenId;
using neuroserve::core::EngineLoop;
using neuroserve::core::EngineLoopOptions;
using neuroserve::core::GenerationParams;
using neuroserve::test_support::FakeBackend;
using neuroserve::test_support::FirstPrefillGate;
using namespace std::chrono_literals;

constexpr int kRequests = 4;
constexpr int kPrompt = 20;
constexpr int kNewTokens = 60;  // 80 positions: 5 pages each, 20 together, on a 10-page pool

GenerationParams request(int r) {
    GenerationParams p;
    p.prompt = std::vector<TokenId>(kPrompt, static_cast<TokenId>(100 * (r + 1)));
    p.max_new_tokens = kNewTokens;
    return p;
}

std::vector<TokenId> expected(const GenerationParams& p) {  // what FakeBackend generates
    std::vector<TokenId> out;
    for (int i = 1; i <= p.max_new_tokens; ++i) {
        out.push_back((p.prompt.back() + i) % 1000);
    }
    return out;
}

// Runs the requests concurrently on a pool too small for all of them; returns the loop's stats.
neuroserve::core::EngineLoopStats run(FakeBackend& backend, EngineLoopOptions options = {}) {
    options.check_invariants = true;
    FirstPrefillGate gate(backend);  // all queued before the first prefill: they must compete
    EngineLoop loop(backend, options);
    std::vector<neuroserve::core::RequestHandle> handles;
    for (int r = 0; r < kRequests; ++r) {
        handles.push_back(loop.submit(request(r)));
    }
    gate.open();
    for (int r = 0; r < kRequests; ++r) {
        EXPECT_TRUE(handles[static_cast<std::size_t>(r)].wait_for(10s)) << "request " << r;
        EXPECT_EQ(handles[static_cast<std::size_t>(r)].wait().tokens, expected(request(r))) << "request " << r;
    }
    EXPECT_FALSE(loop.stop_requested()) << "every step's invariant checks passed";
    const auto kv = loop.kv_stats();
    EXPECT_EQ(kv.free_pages, kv.total_pages) << "KV pages leaked";
    return loop.stats();
}

TEST(PreemptionByRecompute, APreemptedRequestResumesFromItsTokensAndNothingIsGeneratedTwice) {
    FakeBackend backend(1000, 10);
    const auto stats = run(backend);
    ASSERT_GT(stats.preemptions, 0u) << "the pool was never short: preemption was not exercised";
    EXPECT_EQ(backend.generated(), kRequests * kNewTokens)
        << "every token was generated exactly once: a resumed request does not regenerate";
    const auto& sizes = backend.prefill_sizes();
    EXPECT_EQ(sizes.size(), kRequests + stats.preemptions) << "one prefill per admission";
    EXPECT_GT(*std::max_element(sizes.begin(), sizes.end()), kPrompt)
        << "a resumption prefills its prompt and the tokens it kept";
}

TEST(PreemptionByRecompute, ARecomputeReusesTheRequestsOwnCachedPages) {
    // 14 pages: a shortage that eases. The victim's pages are CACHED when it is preempted; the
    // others grow and evict some of them, then finish, and the victim reuses what survived. (On
    // 10 pages the shortage never eases: everything is evicted before it resumes, and it is all
    // recomputed. Correct either way.)
    FakeBackend backend(1000, 14);
    backend.prefix_caching = true;
    const auto stats = run(backend);
    ASSERT_GT(stats.preemptions, 0u);
    EXPECT_EQ(backend.generated(), kRequests * kNewTokens);
    const auto& sizes = backend.prefill_sizes();
    const auto input = std::accumulate(sizes.begin(), sizes.end(), std::int64_t{0});
    EXPECT_GT(backend.kv_stats().prefix_hit_pages, 0) << "its own pages, CACHED when it was preempted";
    EXPECT_LT(backend.prefilled_tokens(), input) << "only what was evicted meanwhile is computed again";
}

TEST(PreemptionByRecompute, ARecomputeLongerThanTheStepBudgetIsStillAdmitted) {
    // max_batch_tokens 32: every prompt fits, but a resumption of up to 79 tokens does not. It
    // counts as one full step's budget and is admitted, so no request starves.
    FakeBackend backend(1000, 10);
    EngineLoopOptions options;
    options.max_batch_tokens = 32;
    const auto stats = run(backend, options);
    ASSERT_GT(stats.preemptions, 0u);
    const auto& sizes = backend.prefill_sizes();
    EXPECT_GT(*std::max_element(sizes.begin(), sizes.end()), 32) << "the over-budget resumption happened";
    EXPECT_EQ(backend.generated(), kRequests * kNewTokens);
}

}  // namespace
