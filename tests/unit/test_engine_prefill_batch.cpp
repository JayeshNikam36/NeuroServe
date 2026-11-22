// The engine prefills the requests admitted in one step together (InferenceBackend::prefill_batch),
// and a request that fails in the batch fails alone (FakeBackend, no GPU).

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "fake_backend.hpp"
#include "neuroserve/core/engine_loop.hpp"

namespace {

using neuroserve::TokenId;
using neuroserve::core::EngineLoop;
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

// Holds the engine inside its first prefill until the test has queued the rest, so the next step
// admits them all at once.
struct Gate {
    std::atomic<bool> entered{false};
    std::atomic<bool> open{false};
    void hold() {
        entered = true;
        while (!open.load()) std::this_thread::sleep_for(1ms);
    }
    void wait_entered() const {
        while (!entered.load()) std::this_thread::sleep_for(1ms);
    }
};

TEST(EnginePrefillBatch, RequestsAdmittedInOneStepArePrefilledInOneCall) {
    FakeBackend backend;
    Gate gate;
    backend.on_prefill = [&] { gate.hold(); };
    EngineLoop loop(backend);
    RequestHandle first = loop.submit(params({1, 2}, 3));  // prefilled alone; holds the engine
    gate.wait_entered();
    std::vector<RequestHandle> rest;
    for (TokenId t = 10; t < 15; ++t) {
        rest.push_back(loop.submit(params({t, t}, 4)));
    }
    gate.open = true;
    first.wait();
    for (TokenId i = 0; i < 5; ++i) {
        const auto r = rest[static_cast<std::size_t>(i)].wait();
        // The fake's tokens continue from the prompt's last token: a request is unaffected by its pack.
        EXPECT_EQ(r.tokens, (std::vector<TokenId>{11 + i, 12 + i, 13 + i, 14 + i}));
    }
    loop.stop();
    EXPECT_EQ(backend.prefill_batch_sizes, (std::vector<std::size_t>{1, 5}));
    EXPECT_EQ(backend.prefill_calls(), 6);  // the default: one prefill per item
}

TEST(EnginePrefillBatch, AFailedPrefillInTheBatchFailsOnlyItsRequest) {
    FakeBackend backend;
    backend.fail_prefill_when_first_token(13);
    Gate gate;
    backend.on_prefill = [&] { gate.hold(); };
    EngineLoop loop(backend);
    RequestHandle hold = loop.submit(params({1}, 3));
    gate.wait_entered();
    RequestHandle good_a = loop.submit(params({20}, 3));
    RequestHandle bad = loop.submit(params({13, 1}, 3));
    RequestHandle good_b = loop.submit(params({30}, 3));
    gate.open = true;
    hold.wait();
    EXPECT_EQ(good_a.wait().tokens, (std::vector<TokenId>{21, 22, 23}));
    EXPECT_THROW(bad.wait(), std::runtime_error);
    EXPECT_EQ(good_b.wait().tokens, (std::vector<TokenId>{31, 32, 33}));
    loop.stop();
    EXPECT_EQ(loop.stats().failed, 1u);
    EXPECT_EQ(backend.live_sequences(), 0);  // the failed request's sequence was freed
    EXPECT_EQ(backend.prefill_batch_sizes, (std::vector<std::size_t>{1, 3}));  // the bad one shared its call
}

}  // namespace
