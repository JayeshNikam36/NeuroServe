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

TEST(KvPrefixCache, PagesOfAFinishedRequestAreFreedNotKeptYet) {
    // Keeping a finished request's pages for later requests is the CACHED state, with eviction
    // (Day 37). Until then, an indexed page is freed with its last holder, and a later request with
    // the same system prompt computes it again.
    FakeBackend backend;
    backend.prefix_caching = true;
    EngineLoop loop(backend, checked());
    EXPECT_EQ(loop.submit(request(1)).wait().tokens, expected(request(1)));
    EXPECT_EQ(loop.submit(request(2)).wait().tokens, expected(request(2)));
    const auto kv = loop.kv_stats();
    EXPECT_EQ(kv.prefix_lookups, 2);
    EXPECT_EQ(kv.prefix_hit_pages, 0);
    EXPECT_EQ(kv.indexed_pages, 0);
    EXPECT_EQ(backend.prefilled_tokens(), 2 * 72);
    EXPECT_FALSE(loop.stop_requested());
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

}  // namespace
