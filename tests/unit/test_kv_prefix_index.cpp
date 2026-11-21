// The prefix index and lookup at admission (Day 33, KV-007): the contract, run on the reference and
// on the KVCacheManager alike. docs/kv-cache.md, "Lookup at admission".
#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "neuroserve/core/kv_cache_manager.hpp"
#include "neuroserve/core/kv_reference.hpp"

namespace {

using neuroserve::KvCacheExhausted;
using neuroserve::TokenId;
using neuroserve::core::BlockId;
using neuroserve::core::KvBlockConfig;
using neuroserve::core::KvBlockManager;
using neuroserve::core::KvCacheIdentity;
using neuroserve::core::KVCacheManager;
using neuroserve::core::ReferenceBlockManager;

KvCacheIdentity identity(int block_size = 16) {
    KvCacheIdentity id;
    id.model_id = "llama-3.2-1b";
    id.weights_digest = "test";
    id.dtype = "fp16";
    id.block_size = block_size;
    id.rope = {500000.0, "llama3", 32.0, 1.0, 4.0, 8192};
    return id;
}

using Factory = std::function<std::unique_ptr<KvBlockManager>(KvBlockConfig, std::optional<KvCacheIdentity>)>;
struct Implementation {
    std::string name;
    Factory make;
};

// A system prompt of `n` tokens, then `tail` tokens of the request's own.
std::vector<TokenId> prompt(int n, TokenId own = 0, int tail = 0) {
    std::vector<TokenId> p;
    for (int i = 0; i < n; ++i) {
        p.push_back(1000 + i);
    }
    for (int i = 0; i < tail; ++i) {
        p.push_back(own);
    }
    return p;
}

class KvPrefixIndex : public ::testing::TestWithParam<Implementation> {
protected:
    std::unique_ptr<KvBlockManager> make(int num_blocks, int max_per_sequence = 16) {
        auto kv = GetParam().make(KvBlockConfig{num_blocks, 16, max_per_sequence}, identity());
        EXPECT_TRUE(kv->prefix_caching());
        return kv;
    }
    // Admits `p` as a new sequence, as a backend does: admit, then (K/V written) commit.
    static int admit_and_commit(KvBlockManager& kv, neuroserve::SeqId seq, const std::vector<TokenId>& p) {
        kv.create(seq);
        const int cached = kv.admit(seq, p).cached_tokens;
        kv.commit(seq, p);
        kv.check_invariants();
        return cached;
    }
};

TEST_P(KvPrefixIndex, ASecondRequestWithTheSameSystemPromptReusesItsFullBlocks) {
    auto kv = make(16);
    const auto a = prompt(64, 1, 8);  // 72 tokens: 4 full blocks of system prompt, 1 partial
    const auto b = prompt(64, 2, 8);
    EXPECT_EQ(admit_and_commit(*kv, 1, a), 0) << "nothing to reuse yet";
    EXPECT_EQ(kv->stats().indexed_blocks, 4) << "the partial block is not indexed";

    EXPECT_EQ(admit_and_commit(*kv, 2, b), 64);
    const std::vector<BlockId> ta = kv->table(1);
    const std::vector<BlockId> tb = kv->table(2);
    ASSERT_EQ(tb.size(), 5u);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(tb[static_cast<std::size_t>(i)], ta[static_cast<std::size_t>(i)]) << "block " << i;
        EXPECT_EQ(kv->ref_count(ta[static_cast<std::size_t>(i)]), 2);
    }
    EXPECT_NE(tb[4], ta[4]) << "the requests' own tails are their own";
    EXPECT_EQ(kv->length(2), 72);

    const auto s = kv->stats();
    EXPECT_EQ(s.used_blocks, 6) << "4 shared + 2 own, instead of 10";
    EXPECT_EQ(s.shared_blocks, 4);
    EXPECT_EQ(s.prefix_lookups, 2);
    EXPECT_EQ(s.prefix_lookup_blocks, 8);
    EXPECT_EQ(s.prefix_hit_blocks, 4) << "the hit is counted";
}

TEST_P(KvPrefixIndex, TheBlockHoldingTheLastPromptTokenIsCopiedNotShared) {
    // Copy-on-write (Day 35): the last prompt token's logits must be computed, which writes its K/V,
    // so the block holding it is reused as a private copy, and the prefill computes one token.
    auto kv = make(16);
    const auto p = prompt(32);  // exactly 2 full blocks
    admit_and_commit(*kv, 1, p);
    EXPECT_EQ(kv->stats().indexed_blocks, 2);
    kv->create(2);
    const auto a = kv->admit(2, p);
    EXPECT_EQ(a.cached_tokens, 31) << "every token but the last is reused";
    ASSERT_TRUE(a.copy.has_value());
    EXPECT_EQ(a.copy->from, kv->table(1)[1]);
    EXPECT_EQ(a.copy->to, kv->table(2)[1]) << "the copy sits where the original was matched";
    EXPECT_NE(a.copy->to, a.copy->from);
    EXPECT_EQ(kv->table(2)[0], kv->table(1)[0]) << "the first block is shared";
    EXPECT_EQ(kv->ref_count(a.copy->from), 1) << "the original stays its writer's alone";
    EXPECT_EQ(kv->ref_count(a.copy->to), 1);
    kv->commit(2, p);
    kv->check_invariants();
    EXPECT_FALSE(kv->indexed(a.copy->to)) << "a copy duplicates an indexed block: not indexed";
    const auto s = kv->stats();
    EXPECT_EQ(s.indexed_blocks, 2);
    EXPECT_EQ(s.shared_blocks, 1);
    EXPECT_EQ(s.prefix_lookup_blocks, 4) << "each lookup could reuse both blocks";
    EXPECT_EQ(s.prefix_hit_blocks, 2) << "the copied block counts as a hit";
    EXPECT_EQ(s.copied_blocks, 1);

    // A hit that stops before the last block needs no copy.
    kv->create(3);
    auto other = p;
    other[20] = 9;
    const auto b = kv->admit(3, other);
    EXPECT_EQ(b.cached_tokens, 16);
    EXPECT_FALSE(b.copy.has_value());
}

TEST_P(KvPrefixIndex, ACopyNeedsAFreeBlockOrNothingHappens) {
    auto kv = make(3);
    admit_and_commit(*kv, 1, prompt(32));  // 2 blocks; 1 free
    admit_and_commit(*kv, 2, prompt(32));  // shares 1, copies 1: no block free now
    kv->create(3);
    const auto before = kv->stats();
    EXPECT_THROW(kv->admit(3, prompt(32)), KvCacheExhausted) << "the copy needs a block";
    EXPECT_EQ(kv->stats().copied_blocks, before.copied_blocks);
    EXPECT_EQ(kv->stats().shared_blocks, before.shared_blocks);
    EXPECT_TRUE(kv->table(3).empty());
    kv->check_invariants();
}

TEST_P(KvPrefixIndex, LookupStopsAtTheFirstMiss) {
    auto kv = make(16);
    admit_and_commit(*kv, 1, prompt(64, 0, 1));
    auto changed = prompt(64, 0, 1);
    changed[20] = 7;  // block 1 differs; blocks 2 and 3 have the same tokens as sequence 1's
    EXPECT_EQ(kv->match(changed), 1) << "a block is reusable only if everything before it is";
    EXPECT_EQ(admit_and_commit(*kv, 2, changed), 16);
    EXPECT_EQ(kv->stats().indexed_blocks, 4 + 3) << "its own blocks 1..3 are a different prefix";
}

TEST_P(KvPrefixIndex, IndexedBlocksOutliveTheirLastHolderAsCached) {
    auto kv = make(16);
    admit_and_commit(*kv, 1, prompt(64, 1, 4));
    admit_and_commit(*kv, 2, prompt(64, 2, 4));
    kv->release(1);  // the request that wrote them
    kv->check_invariants();
    EXPECT_EQ(kv->stats().indexed_blocks, 4) << "sequence 2 still holds them";
    EXPECT_EQ(admit_and_commit(*kv, 3, prompt(64, 3, 4)), 64);

    kv->release(2);
    kv->release(3);
    kv->check_invariants();
    auto s = kv->stats();
    EXPECT_EQ(s.indexed_blocks, 4) << "CACHED (Day 37): still indexed with no holder";
    EXPECT_EQ(s.cached_blocks, 4);
    EXPECT_EQ(kv->free_blocks(), 16) << "a CACHED block counts as free: it is reclaimed on demand";
    EXPECT_EQ(s.used_blocks, 0);
    EXPECT_EQ(kv->match(prompt(64, 4, 4)), 4) << "a later request with the same system prompt can reuse them";
    EXPECT_EQ(admit_and_commit(*kv, 4, prompt(64, 4, 4)), 64);
    s = kv->stats();
    EXPECT_EQ(s.cached_blocks, 0) << "reused: OWNED again";
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(kv->ref_count(kv->table(4)[static_cast<std::size_t>(i)]), 1);
    }
}

TEST_P(KvPrefixIndex, CommitIndexesEachBlockOnceAsItFills) {
    auto kv = make(16);
    std::vector<TokenId> tokens = prompt(10);
    kv->create(1);
    kv->admit(1, tokens);
    EXPECT_EQ(kv->commit(1, tokens), 0);
    for (int step = 0; step < 40; ++step) {  // decode: one token at a time
        kv->append(1, 1);
        tokens.push_back(500 + step);
        const int added = kv->commit(1, tokens);
        EXPECT_EQ(added, tokens.size() % 16 == 0 ? 1 : 0) << tokens.size() << " tokens";
        kv->check_invariants();
    }
    EXPECT_EQ(kv->stats().indexed_blocks, 3);
    // A later request whose prompt continues with the generated tokens reuses them too.
    std::vector<TokenId> follow_up = tokens;
    follow_up.push_back(9);
    EXPECT_EQ(kv->match(follow_up), 3);
}

TEST_P(KvPrefixIndex, ADuplicateIsIndexedOnceTheOriginalIsGone) {
    auto kv = make(16);
    const auto p = prompt(32);
    auto with_tail = p;
    with_tail.push_back(5);
    admit_and_commit(*kv, 1, with_tail);  // blocks A0, A1 indexed
    auto other = p;
    other[20] = 9;  // block 1 differs from A1
    admit_and_commit(*kv, 2, other);      // reuses A0, its own block 1
    kv->release(2);
    kv->create(3);
    kv->append(3, 32);                    // no admit: a sequence that computed the same 32 tokens
    EXPECT_EQ(kv->commit(3, p), 0) << "A0 holds that prefix: a duplicate, not indexed";
    kv->release(1);                       // A0 and A1 are CACHED, still holding the prefix
    EXPECT_EQ(kv->commit(3, p), 0);
    EXPECT_EQ(kv->evict_cached(), 3) << "A0, A1 and sequence 2's block 1";
    EXPECT_EQ(kv->commit(3, p), 2) << "the retry indexes sequence 3's copies";
    kv->check_invariants();
    EXPECT_EQ(kv->match(with_tail), 2);
}

TEST_P(KvPrefixIndex, EvictionTakesTheLeastRecentlyReleasedFirstLeavesFirst) {
    auto kv = make(6);
    const auto a = prompt(48, 0, 1);       // 4 blocks, 3 indexed
    std::vector<TokenId> b(33, 7);         // 3 blocks, 2 indexed
    admit_and_commit(*kv, 1, a);
    kv->release(1);                        // released first: A2, A1, A0 (leaves first)
    admit_and_commit(*kv, 2, b);           // takes FREE blocks only
    kv->release(2);
    EXPECT_EQ(kv->stats().cached_blocks, 5);
    EXPECT_EQ(kv->stats().evicted_blocks, 0);

    // 6 blocks, 5 CACHED, 1 FREE: a 3-block request takes the FREE one and evicts the two oldest.
    kv->create(3);
    kv->append(3, 40);
    kv->check_invariants();
    auto s = kv->stats();
    EXPECT_EQ(s.evicted_blocks, 2);
    EXPECT_EQ(kv->match(a), 1) << "A's last two blocks went first; A0, released after them, stays";
    EXPECT_EQ(kv->match(b), 2) << "B, released later, is untouched";

    kv->release(3);
    kv->create(4);
    EXPECT_THROW(kv->append(4, 7 * 16), KvCacheExhausted) << "7 blocks: more than the pool";
    kv->append(4, 6 * 16);                 // the whole pool: every CACHED block is evicted
    EXPECT_EQ(kv->stats().cached_blocks, 0);
    EXPECT_EQ(kv->stats().indexed_blocks, 0);
    EXPECT_EQ(kv->match(b), 0);
    kv->check_invariants();
}

TEST_P(KvPrefixIndex, OnlyUnreferencedBlocksAreEvicted) {
    auto kv = make(4);
    admit_and_commit(*kv, 1, prompt(32, 0, 1));  // 3 blocks, held
    admit_and_commit(*kv, 2, std::vector<TokenId>(16, 3));  // 1 block, indexed
    kv->release(2);                              // 1 CACHED
    kv->create(3);
    EXPECT_THROW(kv->append(3, 32), KvCacheExhausted) << "1 reclaimable block, 2 needed";
    EXPECT_EQ(kv->stats().cached_blocks, 1) << "a failed allocation evicts nothing";
    kv->append(3, 16);                           // evicts the CACHED block, never an OWNED one
    EXPECT_EQ(kv->stats().evicted_blocks, 1);
    EXPECT_EQ(kv->match(prompt(32, 0, 1)), 2) << "the held blocks are untouched";
    kv->check_invariants();
}

TEST_P(KvPrefixIndex, ACopysSourceIsNotEvictedToMakeItsCopy) {
    auto kv = make(3);
    const auto p = prompt(32);                   // 2 full blocks
    admit_and_commit(*kv, 1, p);
    kv->release(1);                              // CACHED: block 1 oldest, then block 0
    admit_and_commit(*kv, 2, std::vector<TokenId>(16, 3));
    kv->release(2);                              // CACHED, newest
    kv->create(3);
    const auto a = kv->admit(3, p);              // revives block 0; copies block 1 into a new block
    ASSERT_TRUE(a.copy.has_value());
    EXPECT_EQ(a.cached_tokens, 31);
    EXPECT_NE(a.copy->to, a.copy->from) << "the oldest CACHED block is the source: the newest was evicted instead";
    EXPECT_TRUE(kv->indexed(a.copy->from)) << "the source is still there to copy from";
    EXPECT_EQ(kv->ref_count(a.copy->from), 0);
    EXPECT_EQ(kv->stats().evicted_blocks, 1);
    kv->check_invariants();
}

TEST_P(KvPrefixIndex, EvictCachedForgetsOnlyWhatIsCached) {
    auto kv = make(8);
    admit_and_commit(*kv, 1, prompt(32, 0, 1));
    admit_and_commit(*kv, 2, std::vector<TokenId>(33, 4));
    kv->release(2);
    EXPECT_EQ(kv->evict_cached(), 2);
    EXPECT_EQ(kv->stats().cached_blocks, 0);
    EXPECT_EQ(kv->stats().indexed_blocks, 2) << "sequence 1's blocks are held, so they stay";
    EXPECT_EQ(kv->match(std::vector<TokenId>(33, 4)), 0);
    EXPECT_EQ(kv->evict_cached(), 0);
    kv->check_invariants();
}

TEST_P(KvPrefixIndex, AdmitIsAllOrNothing) {
    auto kv = make(6);
    admit_and_commit(*kv, 1, prompt(64, 1, 8));  // 5 blocks, 4 indexed; 1 free
    kv->create(2);
    const auto before = kv->stats();
    // Reuses 4, but needs 2 more for its own 20-token tail: only 1 is free.
    EXPECT_THROW(kv->admit(2, prompt(64, 2, 20)), KvCacheExhausted);
    const auto after = kv->stats();
    EXPECT_EQ(kv->length(2), 0);
    EXPECT_TRUE(kv->table(2).empty());
    EXPECT_EQ(after.shared_blocks, before.shared_blocks) << "the reused blocks were given back";
    EXPECT_EQ(after.prefix_lookups, before.prefix_lookups) << "a failed admission is not counted";
    EXPECT_EQ(after.prefix_hit_blocks, before.prefix_hit_blocks);
    kv->check_invariants();
    EXPECT_EQ(kv->admit(2, prompt(64, 2, 8)).cached_tokens, 64) << "one that fits is admitted";

    kv->create(3);
    EXPECT_THROW(kv->admit(3, prompt(16 * 17)), std::length_error);  // over max_blocks_per_sequence
}

TEST_P(KvPrefixIndex, BadCallsAreRejectedAndChangeNothing) {
    auto kv = make(8);
    admit_and_commit(*kv, 1, prompt(40));
    EXPECT_THROW(kv->admit(1, prompt(8)), std::invalid_argument) << "not empty";
    EXPECT_THROW(kv->admit(9, prompt(8)), std::invalid_argument) << "unknown";
    kv->create(2);
    EXPECT_THROW(kv->admit(2, {}), std::invalid_argument) << "empty prompt";
    EXPECT_THROW(kv->commit(1, prompt(39)), std::invalid_argument) << "wrong number of tokens";
    auto tampered = prompt(40);
    tampered[3] = 1;
    EXPECT_THROW(kv->commit(1, tampered), std::invalid_argument) << "disagrees with an indexed block";
    EXPECT_EQ(kv->stats().indexed_blocks, 2);
    kv->check_invariants();
}

TEST_P(KvPrefixIndex, MatchChangesNothing) {
    auto kv = make(8);
    admit_and_commit(*kv, 1, prompt(40));
    const auto before = kv->stats();
    EXPECT_EQ(kv->match(prompt(40)), 2);
    EXPECT_EQ(kv->match(prompt(33)), 2);
    EXPECT_EQ(kv->match(prompt(32)), 2) << "the last token's block too (copied on admission)";
    const auto after = kv->stats();
    EXPECT_EQ(after.prefix_lookups, before.prefix_lookups);
    EXPECT_EQ(after.shared_blocks, before.shared_blocks);
}

TEST_P(KvPrefixIndex, WithoutAnIdentityPrefixCachingIsOff) {
    auto kv = GetParam().make(KvBlockConfig{8, 16, 8}, std::nullopt);
    EXPECT_FALSE(kv->prefix_caching());
    kv->create(1);
    EXPECT_THROW(kv->admit(1, prompt(8)), std::logic_error);
    EXPECT_THROW(kv->commit(1, {}), std::logic_error);
    EXPECT_THROW((void)kv->match(prompt(8)), std::logic_error);
    EXPECT_THROW(GetParam().make(KvBlockConfig{8, 16, 8}, identity(8)), std::invalid_argument)
        << "the identity's block size must be the pool's";
}

INSTANTIATE_TEST_SUITE_P(
    BothImplementations, KvPrefixIndex,
    ::testing::Values(Implementation{"Reference", [](KvBlockConfig c, std::optional<KvCacheIdentity> id) {
                                         return std::make_unique<ReferenceBlockManager>(c, std::move(id));
                                     }},
                      Implementation{"KVCacheManager", [](KvBlockConfig c, std::optional<KvCacheIdentity> id) {
                                         return std::make_unique<KVCacheManager>(c, std::move(id));
                                     }}),
    [](const ::testing::TestParamInfo<Implementation>& p) { return p.param.name; });

}  // namespace
