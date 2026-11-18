// Shared KV pages through the backend and the engine (Day 31, KV-006): a shared page is freed only
// when its last holder releases it, whether that holder completes, is cancelled or is preempted,
// and the engine plans preemptions with the pages a request alone holds.
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "fake_backend.hpp"
#include "prefill_gate.hpp"
#include "neuroserve/core/engine_loop.hpp"

namespace {

using neuroserve::SeqId;
using neuroserve::TokenId;
using neuroserve::core::EngineLoop;
using neuroserve::core::EngineLoopOptions;
using neuroserve::core::GenerationParams;
using neuroserve::core::RequestHandle;
using neuroserve::test_support::FakeBackend;
using neuroserve::test_support::FirstPrefillGate;
using namespace std::chrono_literals;

GenerationParams params(std::vector<TokenId> prompt, int max_new_tokens) {
    GenerationParams p;
    p.prompt = std::move(prompt);
    p.max_new_tokens = max_new_tokens;
    return p;
}

// What FakeBackend generates: last prompt token + 1, + 2, ... (mod 1000).
std::vector<TokenId> expected(const GenerationParams& p) {
    std::vector<TokenId> out;
    for (int i = 1; i <= p.max_new_tokens; ++i) {
        out.push_back((p.prompt.back() + i) % 1000);
    }
    return out;
}

// 32 common tokens (two full pages), then `tail` tokens of its own.
std::vector<TokenId> with_common_prefix(TokenId own, int tail) {
    std::vector<TokenId> prompt;
    for (int i = 0; i < 32; ++i) {
        prompt.push_back(100 + i);
    }
    for (int i = 0; i < tail; ++i) {
        prompt.push_back(own);
    }
    return prompt;
}

EngineLoopOptions checked() {
    EngineLoopOptions o;
    o.check_invariants = true;  // I11 must hold with shared pages counted once
    return o;
}

TEST(KvSharing, TheBackendCountsASharedPageOnce) {
    FakeBackend b;
    const SeqId a = b.create_sequence();
    b.prefill(a, std::vector<TokenId>(40, 7));  // pages 0, 1, 2
    const SeqId c = b.create_sequence();
    b.share_prefix(c, a, 2);
    EXPECT_EQ(b.sequence_length(c), 32);
    EXPECT_EQ(b.sequence_pages(a).held, 3);
    EXPECT_EQ(b.sequence_pages(a).exclusive, 1);
    EXPECT_EQ(b.sequence_pages(c).held, 2);
    EXPECT_EQ(b.sequence_pages(c).exclusive, 0);
    EXPECT_EQ(b.kv_stats().shared_pages, 2);
    EXPECT_EQ(b.kv_stats().free_pages, 64 - 3) << "sharing took nothing";

    const SeqId ids[] = {c};
    const TokenId last[] = {1};
    b.decode_step(ids, last);  // position 32: a new page of c's own
    EXPECT_EQ(b.sequence_pages(c).held, 3);
    EXPECT_EQ(b.sequence_pages(c).exclusive, 1);

    b.free_sequence(a);  // the first owner goes; the shared pages stay with c
    EXPECT_EQ(b.sequence_pages(c).exclusive, 3);
    EXPECT_EQ(b.kv_stats().shared_pages, 0);
    EXPECT_EQ(b.kv_stats().free_pages, 64 - 3);
    b.free_sequence(c);
    EXPECT_EQ(b.kv_stats().free_pages, 64);
    b.check_kv_invariants();
}

TEST(KvSharing, RequestsWithACommonPrefixShareItsPages) {
    FakeBackend backend;
    backend.share_prompt_prefixes = true;
    int most_shared = 0;
    backend.on_decode = [&](std::size_t) { most_shared = std::max(most_shared, backend.kv_stats().shared_pages); };
    FirstPrefillGate gate(backend);  // both queued before the engine goes on: they run together
    EngineLoop loop(backend, checked());
    const auto pa = params(with_common_prefix(1, 8), 20);
    const auto pb = params(with_common_prefix(2, 8), 20);
    auto a = loop.submit(pa);
    auto b = loop.submit(pb);
    gate.open();
    ASSERT_TRUE(a.wait_for(10s));
    ASSERT_TRUE(b.wait_for(10s));
    EXPECT_EQ(a.wait().tokens, expected(pa));
    EXPECT_EQ(b.wait().tokens, expected(pb));
    EXPECT_EQ(most_shared, 2) << "the two common pages were held once";
    EXPECT_FALSE(loop.stop_requested()) << "every step's checks passed with shared pages";
    EXPECT_EQ(loop.kv_stats().free_pages, loop.kv_stats().total_pages);
}

TEST(KvSharing, SharedPagesOutliveTheRequestThatWroteThem) {
    FakeBackend backend;
    backend.share_prompt_prefixes = true;
    bool shared_while_both_ran = false;
    bool own_after_first_finished = false;
    backend.on_decode = [&](std::size_t batch) {
        if (batch == 2 && backend.kv_stats().shared_pages == 2) {
            shared_while_both_ran = true;
        }
        if (batch == 1 && shared_while_both_ran && backend.kv_stats().shared_pages == 0) {
            own_after_first_finished = true;  // the survivor now holds the pages alone
        }
    };
    FirstPrefillGate gate(backend);
    EngineLoop loop(backend, checked());
    const auto first = params(with_common_prefix(1, 4), 3);    // writes the prefix, finishes early
    const auto second = params(with_common_prefix(2, 4), 40);  // reuses it, keeps running
    auto a = loop.submit(first);
    auto b = loop.submit(second);
    gate.open();
    ASSERT_TRUE(b.wait_for(10s));
    EXPECT_EQ(a.wait().tokens, expected(first));
    EXPECT_EQ(b.wait().tokens, expected(second)) << "the shared pages were not freed under it";
    EXPECT_TRUE(shared_while_both_ran);
    EXPECT_TRUE(own_after_first_finished);
    EXPECT_FALSE(loop.stop_requested());
}

TEST(KvSharing, ACancelledSharerReleasesOnlyItsOwnPages) {
    FakeBackend backend;
    backend.share_prompt_prefixes = true;
    std::atomic<int> decodes{0};
    backend.on_decode = [&](std::size_t) {
        if (++decodes == 3) {
            std::this_thread::sleep_for(20ms);  // room for the cancel to land between steps
        }
    };
    FirstPrefillGate gate(backend);
    EngineLoop loop(backend, checked());
    const auto keeper = params(with_common_prefix(1, 4), 60);
    auto a = loop.submit(keeper);
    auto b = loop.submit(params(with_common_prefix(2, 4), 60));
    gate.open();
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (decodes < 3 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
    }
    ASSERT_GE(decodes.load(), 3) << "the engine stopped decoding";
    EXPECT_TRUE(loop.cancel_request(b.id()));
    ASSERT_TRUE(a.wait_for(10s));
    EXPECT_EQ(a.wait().tokens, expected(keeper));
    EXPECT_THROW(b.wait(), neuroserve::core::RequestCancelled);
    EXPECT_FALSE(loop.stop_requested());
    EXPECT_EQ(loop.kv_stats().free_pages, loop.kv_stats().total_pages);
}

TEST(KvSharing, PreemptingSharersNeverPlansWithPagesItCannotFree) {
    // 16 requests with the same two-page prefix on a 10-page pool: each grows to 5 pages (2 shared,
    // 3 its own), so the engine must preempt. A policy that counted shared pages as freed by a
    // preemption would admit requests the memory cannot hold, and their prefill would fail.
    FakeBackend backend(1000, /*total_pages=*/10);
    backend.share_prompt_prefixes = true;
    FirstPrefillGate gate(backend);  // all 16 queued, so they compete for the pages
    EngineLoop loop(backend, checked());
    std::vector<GenerationParams> all;
    std::vector<RequestHandle> handles;
    for (int i = 0; i < 16; ++i) {
        all.push_back(params(with_common_prefix(static_cast<TokenId>(i + 1), 4), 30));
        handles.push_back(loop.submit(all.back()));
    }
    gate.open();
    for (std::size_t i = 0; i < handles.size(); ++i) {
        ASSERT_TRUE(handles[i].wait_for(30s));
        EXPECT_EQ(handles[i].wait().tokens, expected(all[i])) << "request " << i;  // throws if it failed
    }
    EXPECT_GT(loop.stats().preemptions, 0u);
    EXPECT_EQ(loop.stats().failed, 0u);
    EXPECT_FALSE(loop.stop_requested());
    EXPECT_EQ(loop.kv_stats().free_pages, loop.kv_stats().total_pages);
}

}  // namespace
