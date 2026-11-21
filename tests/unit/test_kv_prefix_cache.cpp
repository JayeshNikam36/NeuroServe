// Prefix caching through the engine (Day 33, KV-007): the engine loop with a backend that admits
// prompts through the prefix index (FakeBackend on the real KVCacheManager), with every step's
// invariant checks on.
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
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

// A 64-token system prompt (4 full pages), then the user's own 8 tokens.
GenerationParams request(TokenId user, int max_new_tokens = 20) {
    GenerationParams p;
    for (int i = 0; i < 64; ++i) {
        p.prompt.push_back(200 + i);
    }
    for (int i = 0; i < 8; ++i) {
        p.prompt.push_back(user);
    }
    p.max_new_tokens = max_new_tokens;
    return p;
}

std::vector<TokenId> expected(const GenerationParams& p) {  // what FakeBackend generates
    std::vector<TokenId> out;
    for (int i = 1; i <= p.max_new_tokens; ++i) {
        out.push_back((p.prompt.back() + i) % 1000);
    }
    return out;
}

EngineLoopOptions checked() {
    EngineLoopOptions o;
    o.check_invariants = true;
    return o;
}

TEST(KvPrefixCache, ASecondRequestWithTheSameSystemPromptReusesItsFullPages) {
    FakeBackend backend;
    backend.prefix_caching = true;
    int most_shared = 0;
    backend.on_decode = [&](std::size_t) { most_shared = std::max(most_shared, backend.kv_stats().shared_pages); };
    FirstPrefillGate gate(backend);  // both queued while the first is admitted
    EngineLoop loop(backend, checked());
    const auto pa = request(1);
    const auto pb = request(2);
    auto a = loop.submit(pa);
    auto b = loop.submit(pb);
    gate.open();
    ASSERT_TRUE(a.wait_for(10s));
    ASSERT_TRUE(b.wait_for(10s));
    EXPECT_EQ(a.wait().tokens, expected(pa));
    EXPECT_EQ(b.wait().tokens, expected(pb)) << "the reused pages hold the right K/V";
    EXPECT_FALSE(loop.stop_requested()) << "every step's invariant checks passed";

    const auto kv = loop.kv_stats();
    EXPECT_EQ(kv.prefix_lookups, 2);
    EXPECT_EQ(kv.prefix_lookup_pages, 8) << "4 reusable pages per prompt";
    EXPECT_EQ(kv.prefix_hit_pages, 4) << "the second request's hit is counted";
    EXPECT_EQ(most_shared, 4);
    EXPECT_EQ(backend.prefilled_tokens(), 72 + 8) << "the second prefill computed only its own 8 tokens";
    EXPECT_EQ(kv.free_pages, kv.total_pages);
}

TEST(KvPrefixCache, AFinishedRequestsPagesAreReusedByALaterOne) {
    // CACHED pages (Day 37): the first request finishes before the second arrives, and its pages,
    // released but still indexed, are reused.
    FakeBackend backend;
    backend.prefix_caching = true;
    EngineLoop loop(backend, checked());
    EXPECT_EQ(loop.submit(request(1)).wait().tokens, expected(request(1)));
    EXPECT_EQ(loop.submit(request(2)).wait().tokens, expected(request(2)));
    const auto kv = loop.kv_stats();
    EXPECT_EQ(kv.prefix_lookups, 2);
    EXPECT_EQ(kv.prefix_hit_pages, 4);
    EXPECT_EQ(backend.prefilled_tokens(), 72 + 8);
    EXPECT_EQ(kv.free_pages, kv.total_pages) << "cached pages count as free";
    EXPECT_FALSE(loop.stop_requested());
}

TEST(KvPrefixCache, UnderMemoryPressureTheOldestCachedPagesGoAndOutputsStayCorrect) {
    // 12 pages, 5 per request (72 prompt + 8 new tokens): five system prompts in turn fill the
    // pool with CACHED pages, so later ones evict the oldest. Then the newest and the oldest again.
    FakeBackend backend(1000, 12);
    backend.prefix_caching = true;
    EngineLoop loop(backend, checked());
    auto with_system = [](TokenId system, TokenId user) {
        GenerationParams p;
        for (int i = 0; i < 64; ++i) {
            p.prompt.push_back(system * 100 + i % 50);
        }
        for (int i = 0; i < 8; ++i) {
            p.prompt.push_back(user);
        }
        p.max_new_tokens = 8;  // 80 positions: 5 pages
        return p;
    };
    for (TokenId system = 1; system <= 5; ++system) {
        const auto p = with_system(system, 1);
        EXPECT_EQ(loop.submit(p).wait().tokens, expected(p)) << "system prompt " << system;
    }
    const auto before = loop.kv_stats();
    const auto again = with_system(5, 2);  // the most recent system prompt: still cached
    EXPECT_EQ(loop.submit(again).wait().tokens, expected(again));
    const auto after = loop.kv_stats();
    EXPECT_EQ(after.prefix_hit_pages - before.prefix_hit_pages, 4) << "the newest cached pages survived";
    const auto old = with_system(1, 2);    // the oldest: evicted long ago
    EXPECT_EQ(loop.submit(old).wait().tokens, expected(old));
    EXPECT_EQ(loop.kv_stats().prefix_hit_pages, after.prefix_hit_pages) << "evicted, so computed again";
    EXPECT_GT(backend.kv().stats().evicted_blocks, 0);
    EXPECT_FALSE(loop.stop_requested()) << "every step's invariant checks passed";
}

TEST(KvPrefixCache, DecodedPagesAreIndexedAsTheyFill) {
    FakeBackend backend;
    backend.prefix_caching = true;
    int most_indexed = 0;
    backend.on_decode = [&](std::size_t) { most_indexed = std::max(most_indexed, backend.kv_stats().indexed_pages); };
    EngineLoop loop(backend, checked());
    GenerationParams p;
    p.prompt = std::vector<TokenId>(10, 3);
    p.max_new_tokens = 40;  // 10 + 40 tokens; K/V written for the first 49: 3 full pages
    EXPECT_EQ(loop.submit(p).wait().tokens.size(), 40u);
    EXPECT_EQ(most_indexed, 3);
    EXPECT_FALSE(loop.stop_requested());
}

TEST(KvPrefixCache, AnIdenticalPromptCopiesItsLastPageAndPrefillsOneToken) {
    // Copy-on-write (Day 35): a 64-token prompt is 4 full pages. The second request reuses all 4,
    // but the 4th holds the last prompt token, whose logits must be computed: it gets a private copy
    // of that page and its prefill computes that one token.
    FakeBackend backend;
    backend.prefix_caching = true;
    FirstPrefillGate gate(backend);
    EngineLoop loop(backend, checked());
    GenerationParams p;
    for (int i = 0; i < 64; ++i) {
        p.prompt.push_back(200 + i);
    }
    p.max_new_tokens = 20;
    auto a = loop.submit(p);
    auto b = loop.submit(p);
    gate.open();
    ASSERT_TRUE(a.wait_for(10s));
    ASSERT_TRUE(b.wait_for(10s));
    EXPECT_EQ(a.wait().tokens, expected(p));
    EXPECT_EQ(b.wait().tokens, expected(p));
    EXPECT_EQ(backend.block_copies(), 1);
    EXPECT_EQ(backend.prefilled_tokens(), 64 + 1);
    const auto kv = loop.kv_stats();
    EXPECT_EQ(kv.prefix_hit_pages, 4);
    EXPECT_EQ(kv.free_pages, kv.total_pages);
    EXPECT_FALSE(loop.stop_requested()) << "every step's invariant checks passed";
}

}  // namespace
