// EngineLoop: one engine thread serving many caller threads (FakeBackend, no GPU).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "fake_backend.hpp"
#include "neuroserve/core/engine_loop.hpp"

namespace {

using neuroserve::TokenId;
using neuroserve::core::EngineLoop;
using neuroserve::core::EngineStopped;
using neuroserve::core::FinishReason;
using neuroserve::core::GenerationParams;
using neuroserve::core::GenerationResult;
using neuroserve::core::RequestHandle;
using neuroserve::test_support::FakeBackend;
using namespace std::chrono_literals;

// Thrown when a request does not finish in time. Deliberately not a std::exception, so an
// EXPECT_THROW(..., std::runtime_error) cannot mistake a hang for the expected error.
struct TimedOut {};

GenerationResult finish(const RequestHandle& handle) {
    if (!handle.wait_for(10s)) {
        throw TimedOut{};
    }
    return handle.wait();
}

GenerationParams params(std::vector<TokenId> prompt, int max_new_tokens, std::vector<TokenId> stop = {}) {
    GenerationParams p;
    p.prompt = std::move(prompt);
    p.max_new_tokens = max_new_tokens;
    p.stop_token_ids = std::move(stop);
    return p;
}

// What FakeBackend must produce for these params: last prompt token + 1, + 2, ... (mod vocab),
// ending at the first stop token or after max_new_tokens.
std::vector<TokenId> expected_tokens(const GenerationParams& p, int vocab = 1000) {
    std::vector<TokenId> out;
    TokenId t = p.prompt.back();
    while (static_cast<int>(out.size()) < p.max_new_tokens) {
        t = (t + 1) % vocab;
        out.push_back(t);
        if (std::find(p.stop_token_ids.begin(), p.stop_token_ids.end(), t) != p.stop_token_ids.end()) {
            break;
        }
    }
    return out;
}

FinishReason expected_reason(const GenerationParams& p) {
    const auto tokens = expected_tokens(p);
    const bool stopped = std::find(p.stop_token_ids.begin(), p.stop_token_ids.end(), tokens.back()) !=
                         p.stop_token_ids.end();
    return stopped ? FinishReason::kStop : FinishReason::kLength;
}

// Blocks the engine inside its first decode step until open() is called, so tests can queue
// requests while the engine is busy, at a known point.
class FirstDecodeGate {
public:
    void install(FakeBackend& backend) {
        backend.on_decode = [this](std::size_t) {
            if (first_.exchange(false)) {
                entered_.set_value();
                released_.wait();
            }
        };
    }
    bool wait_entered() { return entered_future_.wait_for(10s) == std::future_status::ready; }
    void open() { release_.set_value(); }

private:
    std::atomic<bool> first_{true};
    std::promise<void> entered_;
    std::future<void> entered_future_ = entered_.get_future();
    std::promise<void> release_;
    std::shared_future<void> released_ = release_.get_future().share();
};

TEST(EngineLoop, SingleRequestMatchesGreedyReference) {
    FakeBackend backend;
    EngineLoop loop(backend);
    const auto result = finish(loop.submit(params({5, 6, 7}, 4)));
    EXPECT_EQ(result.tokens, (std::vector<TokenId>{8, 9, 10, 11}));
    EXPECT_EQ(result.finish_reason, FinishReason::kLength);
    EXPECT_EQ(result.prompt_tokens, 3);
    loop.stop();
    EXPECT_EQ(backend.live_sequences(), 0);
}

TEST(EngineLoop, StopsOnTheStepTheStopTokenIsProduced) {
    FakeBackend backend;
    EngineLoop loop(backend);
    const auto result = finish(loop.submit(params({5}, 50, {9})));
    EXPECT_EQ(result.tokens, (std::vector<TokenId>{6, 7, 8, 9}));  // includes the stop token
    EXPECT_EQ(result.finish_reason, FinishReason::kStop);
    loop.stop();
    EXPECT_EQ(backend.decode_calls(), 3) << "decoding continued after the stop token";
}

TEST(EngineLoop, StopTokenFromPrefillNeedsNoDecode) {
    FakeBackend backend;
    EngineLoop loop(backend);
    const auto result = finish(loop.submit(params({41}, 10, {42})));
    EXPECT_EQ(result.tokens, (std::vector<TokenId>{42}));
    EXPECT_EQ(result.finish_reason, FinishReason::kStop);
    loop.stop();
    EXPECT_EQ(backend.decode_calls(), 0);
    EXPECT_EQ(backend.live_sequences(), 0);
}

TEST(EngineLoop, InvalidRequestsAreRejectedSynchronously) {
    FakeBackend backend;  // vocab 1000, max prompt 512, 64 pages x 16 tokens per sequence
    EngineLoop loop(backend);
    EXPECT_THROW(loop.submit(params({}, 4)), std::invalid_argument);                     // empty prompt
    EXPECT_THROW(loop.submit(params({1}, 0)), std::invalid_argument);                    // nothing to generate
    EXPECT_THROW(loop.submit(params(std::vector<TokenId>(513, 1), 4)), std::invalid_argument);  // prompt too long
    EXPECT_THROW(loop.submit(params({1, 1000}, 4)), std::invalid_argument);              // outside vocabulary
    EXPECT_THROW(loop.submit(params(std::vector<TokenId>(500, 1), 600)), std::invalid_argument);  // > 1024 tokens
    const auto stats = loop.stats();
    EXPECT_EQ(stats.rejected, 5u);
    EXPECT_EQ(stats.submitted, 0u);
    EXPECT_EQ(stats.pending, 0u);
}

TEST(EngineLoop, HundredConcurrentSubmittersAllCompleteCorrectly) {
    FakeBackend backend;
    FirstDecodeGate gate;
    gate.install(backend);
    EngineLoop loop(backend, {.max_batch = 8});

    constexpr std::size_t kRequests = 100;
    std::vector<GenerationParams> inputs;
    for (std::size_t i = 0; i < kRequests; ++i) {
        const auto first = static_cast<TokenId>(i + 1);
        std::vector<TokenId> stop;
        if (i % 3 == 0) {
            stop.push_back(first + 1 + 5);  // stops after 5 tokens
        }
        inputs.push_back(params({first, first + 1}, 10 + static_cast<int>(i % 30), std::move(stop)));
    }

    // 100 threads submit at the same time while the engine is paused inside its first step.
    std::vector<std::optional<RequestHandle>> handles(kRequests);
    std::vector<std::thread> submitters;
    for (std::size_t i = 0; i < kRequests; ++i) {
        submitters.emplace_back([&, i] { handles[i].emplace(loop.submit(inputs[i])); });
    }
    for (auto& t : submitters) {
        t.join();
    }
    gate.open();

    for (std::size_t i = 0; i < kRequests; ++i) {
        const auto result = finish(*handles[i]);
        EXPECT_EQ(result.tokens, expected_tokens(inputs[i])) << "request " << i;
        EXPECT_EQ(result.finish_reason, expected_reason(inputs[i])) << "request " << i;
        EXPECT_EQ(result.request_id, handles[i]->id());
    }
    loop.stop();
    const auto stats = loop.stats();
    EXPECT_EQ(stats.submitted, kRequests);
    EXPECT_EQ(stats.completed, kRequests);
    EXPECT_EQ(stats.failed + stats.cancelled, 0u);
    EXPECT_EQ(stats.max_batch_seen, 8u) << "requests were not batched, or max_batch was exceeded";
    EXPECT_EQ(backend.live_sequences(), 0) << "sequences leaked";
}

TEST(EngineLoop, KvReservationLimitsConcurrentRequests) {
    FakeBackend backend(1000, /*total_pages=*/8);  // each request below needs 4 of the 8 pages
    FirstDecodeGate gate;
    gate.install(backend);
    EngineLoop loop(backend, {.max_batch = 8});

    std::vector<RequestHandle> handles;
    for (TokenId i = 0; i < 6; ++i) {
        handles.push_back(loop.submit(params({i}, 63)));  // 1 + 63 = 64 tokens = 4 pages
    }
    gate.open();
    for (TokenId i = 0; i < 6; ++i) {
        EXPECT_EQ(finish(handles[static_cast<std::size_t>(i)]).tokens, expected_tokens(params({i}, 63)));
    }
    loop.stop();
    EXPECT_EQ(loop.stats().max_batch_seen, 2u) << "admitted more requests than the KV cache can hold";
    EXPECT_EQ(backend.live_sequences(), 0);
}

TEST(EngineLoop, PrefillFailureFailsOnlyThatRequest) {
    FakeBackend backend;
    backend.fail_prefill_when_first_token(13);
    EngineLoop loop(backend);
    auto bad = loop.submit(params({13, 1}, 5));
    auto good = loop.submit(params({20}, 5));
    EXPECT_THROW(finish(bad), std::runtime_error);
    EXPECT_EQ(finish(good).tokens, (std::vector<TokenId>{21, 22, 23, 24, 25}));
    loop.stop();
    EXPECT_EQ(loop.stats().failed, 1u);
    EXPECT_EQ(backend.live_sequences(), 0);
}

TEST(EngineLoop, DecodeFailureFailsTheBatchAndServingContinues) {
    FakeBackend backend;
    backend.fail_decode_once_at(0);
    EngineLoop loop(backend);
    EXPECT_THROW(finish(loop.submit(params({1}, 10))), std::runtime_error);
    EXPECT_EQ(finish(loop.submit(params({1}, 3))).tokens, (std::vector<TokenId>{2, 3, 4}));
    loop.stop();
    const auto stats = loop.stats();
    EXPECT_EQ(stats.failed, 1u);
    EXPECT_EQ(stats.completed, 1u);
    EXPECT_EQ(backend.live_sequences(), 0);
}

TEST(EngineLoop, StopCancelsPendingAndActiveRequests) {
    FakeBackend backend;
    FirstDecodeGate gate;
    gate.install(backend);
    EngineLoop loop(backend, {.max_batch = 1});

    auto running = loop.submit(params({1}, 100));
    ASSERT_TRUE(gate.wait_entered());               // `running` is inside its first decode step
    auto queued = loop.submit(params({2}, 100));    // max_batch 1: stays in the queue

    std::thread stopper([&loop] { loop.stop(); });  // stop() blocks until the engine thread exits
    while (!loop.stop_requested()) {
        std::this_thread::yield();
    }
    gate.open();
    stopper.join();

    EXPECT_THROW(finish(running), EngineStopped);
    EXPECT_THROW(finish(queued), EngineStopped);
    const auto stats = loop.stats();
    EXPECT_EQ(stats.cancelled, 2u);
    EXPECT_EQ(stats.completed, 0u);
    EXPECT_EQ(stats.pending + stats.active, 0u);
    EXPECT_EQ(backend.live_sequences(), 0);
}

TEST(EngineLoop, SubmitAfterStopThrows) {
    FakeBackend backend;
    EngineLoop loop(backend);
    loop.stop();
    loop.stop();  // idempotent
    EXPECT_THROW(loop.submit(params({1}, 4)), EngineStopped);
    EXPECT_EQ(loop.stats().rejected, 1u);
}

TEST(EngineLoop, ResultsCarryTimings) {
    FakeBackend backend;
    EngineLoop loop(backend);
    const auto result = finish(loop.submit(params({1}, 8)));
    EXPECT_GE(result.queue_ms, 0.0);
    EXPECT_GE(result.ttft_ms, 0.0);
    EXPECT_LE(result.ttft_ms, result.e2e_ms);
}

TEST(EngineLoop, RequestIdsAreSequential) {
    FakeBackend backend;
    EngineLoop loop(backend);
    auto a = loop.submit(params({1}, 2));
    auto b = loop.submit(params({1}, 2));
    EXPECT_EQ(a.id(), "ns-000001");
    EXPECT_EQ(b.id(), "ns-000002");
    EXPECT_EQ(finish(b).request_id, "ns-000002");
    EXPECT_EQ(finish(a).request_id, "ns-000001");
}

}  // namespace
