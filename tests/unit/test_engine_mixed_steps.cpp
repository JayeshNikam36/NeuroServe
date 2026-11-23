// Mixed steps (Day 53): a step's prefills and decode go to the backend in one mixed_step call,
// which a backend that supports it runs as one forward pass (FakeBackend, no GPU).

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>
#include <utility>
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
using neuroserve::test_support::FakeBackend;
using namespace std::chrono_literals;

GenerationParams params(std::vector<TokenId> prompt, int max_new_tokens) {
    GenerationParams p;
    p.prompt = std::move(prompt);
    p.max_new_tokens = max_new_tokens;
    return p;
}

std::vector<TokenId> prompt_of(int length, TokenId first) {
    std::vector<TokenId> p(static_cast<std::size_t>(length));
    std::iota(p.begin(), p.end(), first);
    return p;
}

EngineLoopOptions options(int chunk, bool mixed) {
    EngineLoopOptions o;
    o.prefill_chunk_tokens = chunk;
    o.mixed_steps = mixed;
    return o;
}

// A runner decodes while a 200-token prompt is prefilled in chunks of 64: the engine holds its
// first prefill (the runner's) until the long prompt is queued, so the steps are known.
struct Overlap {
    std::vector<TokenId> runner_tokens;
    std::vector<TokenId> long_tokens;
    std::vector<int> prefill_sizes;
    std::vector<std::pair<std::size_t, std::size_t>> mixed_calls;
    std::uint64_t mixed_steps = 0;
};

Overlap run_overlap(bool engine_mixed, bool backend_mixed, int fail_decode_at = -1) {
    FakeBackend backend(1000, 128);
    backend.mixed_steps = backend_mixed;
    if (fail_decode_at >= 0) {
        backend.fail_decode_once_at(fail_decode_at);
    }
    std::atomic<int> prefills{0};
    std::atomic<bool> open{false};
    backend.on_prefill = [&] {
        if (prefills++ == 0) {
            while (!open.load()) std::this_thread::sleep_for(1ms);
        }
    };
    EngineLoop loop(backend, options(64, engine_mixed));
    RequestHandle runner = loop.submit(params({5}, 40));
    while (prefills.load() == 0) std::this_thread::sleep_for(1ms);
    RequestHandle long_one = loop.submit(params(prompt_of(200, 100), 3));
    open = true;
    Overlap out;
    out.long_tokens = long_one.wait().tokens;
    try {
        out.runner_tokens = runner.wait().tokens;
    } catch (...) {
        out.runner_tokens.assign(1, -1);  // failed
    }
    loop.stop();
    out.prefill_sizes = backend.prefill_sizes();
    out.mixed_calls = backend.mixed_calls;
    out.mixed_steps = loop.stats().mixed_steps;
    EXPECT_EQ(backend.live_sequences(), 0);
    return out;
}

TEST(EngineMixedSteps, PrefillChunksAndDecodesGoToTheBackendInOneCall) {
    const Overlap mixed = run_overlap(true, true);
    // The runner's prefill alone, then four steps with a chunk of the long prompt and the runner's
    // decode in one call each.
    const std::vector<std::pair<std::size_t, std::size_t>> want(4, {1, 1});
    EXPECT_EQ(mixed.mixed_calls, want);
    EXPECT_EQ(mixed.mixed_steps, 4u);
    EXPECT_EQ(mixed.prefill_sizes, (std::vector<int>{1, 64, 64, 64, 8}));
    EXPECT_EQ(mixed.long_tokens, (std::vector<TokenId>{300, 301, 302}));
    EXPECT_EQ(mixed.runner_tokens.size(), 40u);
}

TEST(EngineMixedSteps, OutputsAreTheSameWithTwoCalls) {
    const Overlap mixed = run_overlap(true, true);
    for (const auto& [engine_mixed, backend_mixed] : {std::pair{false, true}, std::pair{true, false}}) {
        const Overlap two = run_overlap(engine_mixed, backend_mixed);
        EXPECT_TRUE(two.mixed_calls.empty());
        EXPECT_EQ(two.mixed_steps, 0u);
        EXPECT_EQ(two.prefill_sizes, mixed.prefill_sizes);
        EXPECT_EQ(two.long_tokens, mixed.long_tokens);
        EXPECT_EQ(two.runner_tokens, mixed.runner_tokens);
    }
}

TEST(EngineMixedSteps, AFailedDecodeFailsOnlyTheDecodeBatch) {
    // The first decode call is the first mixed step's: the runner fails, the long prompt goes on.
    const Overlap r = run_overlap(true, true, 0);
    EXPECT_EQ(r.runner_tokens, (std::vector<TokenId>{-1}));
    EXPECT_EQ(r.long_tokens, (std::vector<TokenId>{300, 301, 302}));
    EXPECT_EQ(r.prefill_sizes, (std::vector<int>{1, 64, 64, 64, 8}));
}

TEST(EngineMixedSteps, ManyRequestsMatchTheTwoCallRun) {
    auto run = [](bool mixed) {
        FakeBackend backend(2000, 256);
        EngineLoop loop(backend, options(32, mixed));
        std::vector<RequestHandle> handles;
        for (int i = 0; i < 24; ++i) {
            handles.push_back(loop.submit(params(prompt_of(5 + 17 * i % 120, 1 + i), 8 + i % 5)));
        }
        std::vector<std::vector<TokenId>> out;
        for (std::size_t i = 0; i < handles.size(); ++i) {
            const auto r = handles[i].wait();
            EXPECT_EQ(r.tokens.size(), static_cast<std::size_t>(8 + i % 5));
            out.push_back(r.tokens);
        }
        loop.stop();
        EXPECT_EQ(backend.live_sequences(), 0);
        return std::pair{out, loop.stats().mixed_steps};
    };
    const auto [mixed, mixed_steps] = run(true);
    const auto [two, two_steps] = run(false);
    EXPECT_EQ(mixed, two);
    EXPECT_EQ(two_steps, 0u);
    EXPECT_GT(mixed_steps, 0u) << "no step mixed prefills and decodes";
}

}  // namespace
