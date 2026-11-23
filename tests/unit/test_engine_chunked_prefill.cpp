// Chunked prefill (Day 52): a prompt is prefilled across steps, at most prefill_chunk_tokens per
// step, from its cached offset, while the running requests keep decoding (FakeBackend, no GPU).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <numeric>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "fake_backend.hpp"
#include "neuroserve/core/engine_loop.hpp"

namespace {

using neuroserve::SeqId;
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

EngineLoopOptions chunked(int chunk) {
    EngineLoopOptions o;
    o.prefill_chunk_tokens = chunk;
    return o;
}

TEST(EngineChunkedPrefill, ALongPromptIsPrefilledInChunksWhileAnotherRequestDecodes) {
    FakeBackend backend(1000, 128);
    backend.record_batches = true;
    std::atomic<int> prefills{0};
    std::atomic<bool> open{false};
    backend.on_prefill = [&] {  // hold the engine in the first prefill until the long prompt is queued
        if (prefills++ == 0) {
            while (!open.load()) std::this_thread::sleep_for(1ms);
        }
    };
    EngineLoop loop(backend, chunked(64));
    RequestHandle runner = loop.submit(params({5}, 40));
    while (prefills.load() == 0) std::this_thread::sleep_for(1ms);
    RequestHandle long_one = loop.submit(params(prompt_of(200, 100), 3));
    open = true;
    const auto r = long_one.wait();
    EXPECT_EQ(r.tokens, (std::vector<TokenId>{300, 301, 302}));  // the fake continues from the last prompt token
    EXPECT_EQ(runner.wait().tokens.size(), 40u);
    loop.stop();
    // The runner's prompt, then the long prompt in chunks of 64, 64, 64 and 8.
    EXPECT_EQ(backend.prefill_sizes(), (std::vector<int>{1, 64, 64, 64, 8}));
    // The runner decoded in every step of the long prompt's prefill: 4 steps with a batch of one
    // (the long request joins the decode batch after its last chunk).
    const auto& batches = backend.decode_batches;
    ASSERT_GE(batches.size(), 5u);
    for (std::size_t s = 0; s < 4; ++s) {
        EXPECT_EQ(batches[s].size(), 1u) << "step " << s;
    }
    EXPECT_EQ(batches[4].size(), 2u);
    EXPECT_EQ(backend.live_sequences(), 0);
}

std::vector<std::vector<TokenId>> run_with_chunk(int chunk, std::vector<int>& sizes, int& max_step_tokens) {
    FakeBackend backend(1000, 256);
    EngineLoop loop(backend, chunked(chunk));
    std::vector<RequestHandle> handles;
    const std::vector<int> lengths = {100, 300, 500, 20, 512, 1, 257};
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        handles.push_back(loop.submit(params(prompt_of(lengths[i], static_cast<TokenId>(10 * i + 1)), 6)));
    }
    std::vector<std::vector<TokenId>> out;
    for (auto& h : handles) {
        out.push_back(h.wait().tokens);
    }
    loop.stop();
    sizes = backend.prefill_sizes();
    max_step_tokens = loop.stats().max_step_tokens;
    EXPECT_EQ(backend.live_sequences(), 0);
    return out;
}

TEST(EngineChunkedPrefill, OutputsAreTheSameForEveryChunkSize) {
    std::vector<int> sizes;
    int max_step = 0;
    const auto whole = run_with_chunk(0, sizes, max_step);
    for (const int chunk : {64, 128, 256, 512}) {
        SCOPED_TRACE("chunk " + std::to_string(chunk));
        const auto out = run_with_chunk(chunk, sizes, max_step);
        EXPECT_EQ(out, whole);
        EXPECT_LE(*std::max_element(sizes.begin(), sizes.end()), chunk);  // no prefill exceeds a chunk
        EXPECT_EQ(std::accumulate(sizes.begin(), sizes.end(), 0), 100 + 300 + 500 + 20 + 512 + 1 + 257);
        EXPECT_LE(max_step, 512);  // chunks and decodes together stay within max_batch_tokens
    }
}

TEST(EngineChunkedPrefill, TheStepBudgetBoundsChunksAndDecodesTogether) {
    FakeBackend backend(1000, 256);
    EngineLoopOptions o = chunked(64);
    o.max_batch_tokens = 70;
    EngineLoop loop(backend, o);
    std::vector<RequestHandle> handles;
    for (int i = 0; i < 6; ++i) {
        handles.push_back(loop.submit(params(prompt_of(10, 1 + i), 30)));
    }
    handles.push_back(loop.submit(params(prompt_of(70, 500), 4)));
    for (auto& h : handles) {
        h.wait();
    }
    loop.stop();
    EXPECT_LE(loop.stats().max_step_tokens, 70);
    EXPECT_EQ(backend.live_sequences(), 0);
}

TEST(EngineChunkedPrefill, ABackendThatCannotChunkPrefillsWhole) {
    FakeBackend backend(1000, 128);
    backend.chunked_prefill = false;
    EngineLoop loop(backend, chunked(64));
    EXPECT_EQ(loop.submit(params(prompt_of(200, 100), 2)).wait().tokens, (std::vector<TokenId>{300, 301}));
    loop.stop();
    EXPECT_EQ(backend.prefill_sizes(), (std::vector<int>{200}));
}

TEST(EngineChunkedPrefill, CancellingARequestMidPrefillFreesItsPages) {
    FakeBackend backend(1000, 128);
    EngineLoop* loop_ptr = nullptr;
    std::mutex mu;
    std::string victim;  // set by the test thread, read by the engine thread: under mu
    bool cancelled = false;
    std::atomic<int> calls{0};
    backend.on_prefill = [&] {
        std::lock_guard lock(mu);
        if (++calls >= 3 && !victim.empty() && !cancelled) {  // from its third chunk on
            cancelled = true;
            loop_ptr->cancel_request(victim);
        }
    };
    EngineLoop loop(backend, chunked(16));
    loop_ptr = &loop;
    RequestHandle runner = loop.submit(params({7}, 20));
    RequestHandle h = loop.submit(params(prompt_of(200, 100), 5));
    {
        std::lock_guard lock(mu);
        victim = h.id();
    }
    EXPECT_THROW(h.wait(), neuroserve::core::RequestCancelled);
    EXPECT_EQ(runner.wait().tokens.size(), 20u);
    loop.stop();
    EXPECT_EQ(backend.live_sequences(), 0);
    EXPECT_EQ(backend.kv_stats().free_pages, backend.kv_stats().total_pages);
}

TEST(EngineChunkedPrefill, APromptLongerThanAStepIsAcceptedOnlyWhenChunking) {
    FakeBackend backend(1000, 256);
    EngineLoopOptions o = chunked(100);
    o.max_batch_tokens = 128;
    EngineLoop loop(backend, o);
    RequestHandle h = loop.submit(params(prompt_of(300, 3), 4));  // 300 > 128: three steps
    EXPECT_EQ(h.wait().tokens.size(), 4u);
    loop.stop();
    EXPECT_LE(loop.stats().max_step_tokens, 128);

    FakeBackend whole(1000, 256);
    EngineLoopOptions off;
    off.max_batch_tokens = 128;
    EngineLoop plain(whole, off);
    EXPECT_THROW(plain.submit(params(prompt_of(300, 3), 4)), std::invalid_argument);
    plain.stop();
}

TEST(EngineChunkedPrefill, NegativeChunkSizesAreRefused) {
    FakeBackend backend;
    EXPECT_THROW(EngineLoop(backend, chunked(-1)), std::invalid_argument);
}

}  // namespace
