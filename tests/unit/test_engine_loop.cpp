// EngineLoop: one engine thread serving many caller threads (FakeBackend, no GPU).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <iterator>
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
using neuroserve::core::EngineLoopOptions;
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
    auto urgent = params({1}, 4);
    urgent.priority = 3;
    EXPECT_THROW(loop.submit(urgent), std::invalid_argument);                            // priority is 0..2
    const auto stats = loop.stats();
    EXPECT_EQ(stats.rejected, 6u);
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

// The correctness tests below run once per scheduling policy: every policy must produce exactly
// the same tokens; only the order and timing of the work may differ.
class EveryPolicy : public ::testing::TestWithParam<std::string> {};

std::vector<std::string> all_policies() {
    return {std::begin(neuroserve::core::kPolicyNames), std::end(neuroserve::core::kPolicyNames)};
}

INSTANTIATE_TEST_SUITE_P(Policies, EveryPolicy, ::testing::ValuesIn(all_policies()),
                         [](const ::testing::TestParamInfo<std::string>& param_info) {
                             std::string name = param_info.param;
                             std::replace(name.begin(), name.end(), '-', '_');  // test names: no '-'
                             return name;
                         });

TEST_P(EveryPolicy, KvPressurePreemptsAndEveryRequestStillCompletesCorrectly) {
    // 8 pages of 16 tokens; six requests that each grow to 64 tokens (4 pages). Pages are taken on
    // demand, so all six start; when they outgrow the cache the newest are preempted, restart from
    // their prompt, and still produce exactly the right tokens.
    FakeBackend backend(1000, /*total_pages=*/8);
    FirstDecodeGate gate;
    gate.install(backend);
    EngineLoop loop(backend, {.max_batch = 8, .policy = GetParam()});

    std::vector<RequestHandle> handles;
    for (TokenId i = 0; i < 6; ++i) {
        handles.push_back(loop.submit(params({i}, 63)));  // 1 + 63 = 64 tokens = 4 pages
    }
    gate.open();
    for (TokenId i = 0; i < 6; ++i) {
        EXPECT_EQ(finish(handles[static_cast<std::size_t>(i)]).tokens, expected_tokens(params({i}, 63)));
    }
    loop.stop();
    const auto stats = loop.stats();
    EXPECT_GT(stats.preemptions, 0u) << "the cache was never short: the test does not exercise preemption";
    EXPECT_GT(stats.max_batch_seen, 2u) << "on-demand pages should run more than 2 requests at once";
    EXPECT_EQ(stats.completed, 6u);
    EXPECT_EQ(backend.live_sequences(), 0);
}

// Reads a streaming request until it finishes; returns every token it streamed.
std::vector<TokenId> read_all(const RequestHandle& handle) {
    std::vector<TokenId> out;
    for (int reads = 0; reads < 10'000; ++reads) {
        auto batch = handle.read_tokens(10s);
        out.insert(out.end(), batch.tokens.begin(), batch.tokens.end());
        if (batch.finished) {
            return out;
        }
    }
    throw TimedOut{};
}

GenerationParams streaming(GenerationParams p) {
    p.stream = true;
    return p;
}

TEST_P(EveryPolicy, StreamedRequestsReturnEachTokenOnceDespitePreemption) {
    // The KV-pressure workload above, streamed: preempted requests generate their tokens again
    // from the prompt, but each reader sees every token exactly once, in order.
    FakeBackend backend(1000, /*total_pages=*/8);
    FirstDecodeGate gate;
    gate.install(backend);
    EngineLoop loop(backend, {.max_batch = 8, .policy = GetParam()});

    std::vector<RequestHandle> handles;
    for (TokenId i = 0; i < 6; ++i) {
        handles.push_back(loop.submit(streaming(params({i}, 63))));
    }
    gate.open();
    for (TokenId i = 0; i < 6; ++i) {
        const auto& handle = handles[static_cast<std::size_t>(i)];
        const auto expected = expected_tokens(params({i}, 63));
        EXPECT_EQ(read_all(handle), expected);
        EXPECT_EQ(handle.wait().tokens, expected);
    }
    loop.stop();
    EXPECT_GT(loop.stats().preemptions, 0u) << "the cache was never short: the test does not exercise preemption";
}

TEST_P(EveryPolicy, TwentyFourRequestsJoinAndLeaveTheRunningBatch) {
    // The plan's acceptance test: 24 requests of different lengths all complete correctly,
    // requests join a batch that is already running and leave it when done (continuous
    // batching), and no KV page leaks.
    FakeBackend backend(1000, /*total_pages=*/16);
    backend.record_batches = true;
    FirstDecodeGate gate;
    gate.install(backend);
    EngineLoop loop(backend, {.max_batch = 8, .policy = GetParam()});

    std::vector<GenerationParams> inputs;
    std::vector<RequestHandle> handles;
    for (int i = 0; i < 24; ++i) {
        inputs.push_back(params({static_cast<TokenId>(10 * i + 1)}, 3 + (i * 7) % 29));
        handles.push_back(loop.submit(inputs.back()));
    }
    gate.open();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_EQ(finish(handles[i]).tokens, expected_tokens(inputs[i])) << "request " << i;
    }
    loop.stop();

    int joins = 0;   // a step whose batch has a new sequence next to ones from the previous step
    int leaves = 0;  // a step that lost a sequence while others carried on
    const auto& batches = backend.decode_batches;
    for (std::size_t s = 1; s < batches.size(); ++s) {
        const auto& prev = batches[s - 1];
        const auto& cur = batches[s];
        const auto in = [](const std::vector<neuroserve::SeqId>& v, neuroserve::SeqId id) {
            return std::find(v.begin(), v.end(), id) != v.end();
        };
        const bool kept_any = std::any_of(cur.begin(), cur.end(), [&](auto id) { return in(prev, id); });
        if (kept_any && std::any_of(cur.begin(), cur.end(), [&](auto id) { return !in(prev, id); })) {
            ++joins;
        }
        if (kept_any && std::any_of(prev.begin(), prev.end(), [&](auto id) { return !in(cur, id); })) {
            ++leaves;
        }
    }
    EXPECT_GT(joins, 5) << "requests did not join a running batch";
    EXPECT_GT(leaves, 5) << "finished requests did not leave a running batch";
    EXPECT_EQ(loop.stats().completed, 24u);
    EXPECT_EQ(backend.live_sequences(), 0);
    EXPECT_EQ(loop.kv_stats().free_pages, loop.kv_stats().total_pages) << "KV pages leaked";
}

// Runs one request, queues a low- and then a high-priority request behind it (one slot), and
// returns the order in which the three were prefilled, by the first token of their prompts.
std::vector<TokenId> admission_order(const std::string& policy) {
    FakeBackend backend;
    backend.record_batches = true;
    FirstDecodeGate gate;
    gate.install(backend);
    EngineLoopOptions options{.max_batch = 1, .policy = policy, .policy_options = {.priority_aging_ms = 3'600'000}};
    EngineLoop loop(backend, options);
    const auto request = [](TokenId first, int priority) {
        auto p = params({first}, 3);
        p.priority = priority;
        return p;
    };
    auto running = loop.submit(request(100, neuroserve::core::kNormalPriority));
    EXPECT_TRUE(gate.wait_entered());  // 100 is running; the engine is paused in its first decode
    auto low = loop.submit(request(200, neuroserve::core::kLowPriority));
    auto high = loop.submit(request(300, neuroserve::core::kHighPriority));
    gate.open();
    for (const auto* handle : {&running, &low, &high}) {
        EXPECT_TRUE(handle->wait_for(10s));
    }
    loop.stop();
    return backend.prefill_first_tokens;
}

TEST(EngineLoop, HigherPriorityRequestsAreAdmittedFirstUnderThePriorityPolicy) {
    EXPECT_EQ(admission_order("priority"), (std::vector<TokenId>{100, 300, 200}));
    EXPECT_EQ(admission_order("fcfs"), (std::vector<TokenId>{100, 200, 300}));  // arrival order
}

TEST(EngineLoop, StreamedTokensCanBeReadBeforeTheRequestFinishes) {
    FakeBackend backend;
    FirstDecodeGate gate;
    gate.install(backend);
    EngineLoop loop(backend);
    const auto handle = loop.submit(streaming(params({5}, 4)));
    ASSERT_TRUE(gate.wait_entered());  // prefilled (first token out), paused in the first decode

    const auto first = handle.read_tokens(10s);
    EXPECT_EQ(first.tokens, (std::vector<TokenId>{6}));
    EXPECT_FALSE(first.finished);
    EXPECT_TRUE(handle.read_tokens(1ms).tokens.empty());  // nothing new yet: the timeout passes
    gate.open();
    EXPECT_EQ(read_all(handle), (std::vector<TokenId>{7, 8, 9}));
    EXPECT_EQ(handle.wait().tokens, (std::vector<TokenId>{6, 7, 8, 9}));
}

TEST(EngineLoop, AStreamEndsWithTheErrorWhenTheRequestFails) {
    FakeBackend backend;
    backend.fail_decode_after(1);  // the first decode works, the second fails
    EngineLoop loop(backend);
    const auto handle = loop.submit(streaming(params({1}, 10)));
    EXPECT_EQ(read_all(handle), (std::vector<TokenId>{2, 3}));  // what was generated before the error
    EXPECT_THROW(handle.wait(), std::runtime_error);
}

TEST(EngineLoop, OnlyStreamingRequestsCanBeRead) {
    FakeBackend backend;
    EngineLoop loop(backend);
    const auto handle = loop.submit(params({1}, 2));
    EXPECT_THROW(handle.read_tokens(1ms), std::logic_error);
    EXPECT_EQ(finish(handle).tokens, (std::vector<TokenId>{2, 3}));
}

TEST(EngineLoop, NoStepExceedsTheTokenBudget) {
    // Prompts of 40-100 tokens against a 128-token budget: at most one or two prefills fit next to
    // the running decodes in any step. Every step stays within the budget, and every request still
    // completes correctly.
    FakeBackend backend;
    std::vector<neuroserve::core::StepStats> steps;  // engine thread only; read after stop()
    neuroserve::core::EngineLoopOptions options;
    options.max_batch = 8;
    options.max_batch_tokens = 128;
    options.on_step = [&steps](const neuroserve::core::StepStats& s) { steps.push_back(s); };
    FirstDecodeGate gate;
    gate.install(backend);
    EngineLoop loop(backend, options);

    std::vector<GenerationParams> inputs;
    std::vector<RequestHandle> handles;
    for (int i = 0; i < 12; ++i) {
        inputs.push_back(params(std::vector<TokenId>(static_cast<std::size_t>(40 + 5 * i), i + 1), 12));
        handles.push_back(loop.submit(inputs.back()));
    }
    gate.open();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_EQ(finish(handles[i]).tokens, expected_tokens(inputs[i])) << "request " << i;
    }
    loop.stop();

    ASSERT_FALSE(steps.empty());
    int prefill_tokens = 0;
    for (const auto& s : steps) {
        EXPECT_LE(s.prefill_tokens + s.decode_tokens, 128) << "a step exceeded the token budget";
        prefill_tokens += s.prefill_tokens;
    }
    const auto stats = loop.stats();
    EXPECT_LE(stats.max_step_tokens, 128);
    EXPECT_GT(stats.max_step_tokens, 64) << "the budget was never approached: the test is too easy";
    EXPECT_EQ(stats.prefill_tokens, static_cast<std::uint64_t>(prefill_tokens));
}

TEST(EngineLoop, PromptLongerThanTheTokenBudgetIsRejected) {
    FakeBackend backend;
    neuroserve::core::EngineLoopOptions options;
    options.max_batch_tokens = 64;
    EngineLoop loop(backend, options);
    EXPECT_THROW(loop.submit(params(std::vector<TokenId>(65, 1), 4)), std::invalid_argument);
    EXPECT_EQ(finish(loop.submit(params(std::vector<TokenId>(64, 1), 4))).tokens.size(), 4u);
}

TEST(EngineLoop, AdmittedRequestsDecodeFromTheNextStep) {
    // A request is prefilled in the step that admits it and decoded from the following step: the
    // step that prefills it never counts it as a decode token, and each later step decodes it once.
    FakeBackend backend;
    std::vector<neuroserve::core::StepStats> steps;
    neuroserve::core::EngineLoopOptions options;
    options.on_step = [&steps](const neuroserve::core::StepStats& s) { steps.push_back(s); };
    EngineLoop loop(backend, options);
    EXPECT_EQ(finish(loop.submit(params({5, 6, 7}, 4))).tokens, (std::vector<TokenId>{8, 9, 10, 11}));
    loop.stop();

    ASSERT_EQ(steps.size(), 4u);  // 1 prefill step (token 8) + 3 decode steps (9, 10, 11)
    EXPECT_EQ(steps[0].prefill_requests, 1u);
    EXPECT_EQ(steps[0].prefill_tokens, 3);
    EXPECT_EQ(steps[0].decode_tokens, 0);
    for (std::size_t s = 1; s < steps.size(); ++s) {
        EXPECT_EQ(steps[s].prefill_tokens, 0);
        EXPECT_EQ(steps[s].decode_tokens, 1);
    }
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
