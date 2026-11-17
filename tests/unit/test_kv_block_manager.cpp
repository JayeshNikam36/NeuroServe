// The contract every KV block manager must meet (docs/kv-cache.md). The tests run once per
// implementation: today the CPU reference; from Day 28 also the KVCacheManager. Breaking the state
// on purpose (fault injection) to test check_invariants() comes with the checker on Day 30.
#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "neuroserve/core/kv_reference.hpp"

using neuroserve::KvCacheExhausted;
using neuroserve::core::BlockId;
using neuroserve::core::KvBlockConfig;
using neuroserve::core::KvBlockManager;
using neuroserve::core::KvInvariantViolation;
using neuroserve::core::ReferenceBlockManager;

namespace {

using Factory = std::function<std::unique_ptr<KvBlockManager>(KvBlockConfig)>;

struct Implementation {
    std::string name;
    Factory make;
};

void PrintTo(const Implementation& implementation, std::ostream* os) { *os << implementation.name; }

std::vector<BlockId> ids(std::initializer_list<BlockId> list) { return list; }

class KvBlockContract : public ::testing::TestWithParam<Implementation> {
protected:
    std::unique_ptr<KvBlockManager> make(int num_blocks, int max_per_sequence = 64) {
        auto kv = GetParam().make(KvBlockConfig{num_blocks, 16, max_per_sequence});
        kv->check_invariants();
        return kv;
    }
};

TEST_P(KvBlockContract, TablesCoverExactlyTheTokens) {
    // The worked example of the technical specification (16-token blocks).
    auto kv = make(32);
    const std::vector<std::pair<int, int>> requests = {{30, 2}, {64, 4}, {100, 7}, {16, 1}};
    for (std::size_t r = 0; r < requests.size(); ++r) {
        kv->create(r);
        kv->append(r, requests[r].first);
        EXPECT_EQ(kv->length(r), requests[r].first);
        EXPECT_EQ(kv->table(r).size(), static_cast<std::size_t>(requests[r].second)) << "request " << r;
        kv->check_invariants();
    }
    EXPECT_EQ(kv->free_blocks(), 32 - (2 + 4 + 7 + 1));
    const auto s = kv->stats();
    EXPECT_EQ(s.total_blocks, 32);
    EXPECT_EQ(s.used_blocks, 14);
    EXPECT_EQ(s.shared_blocks, 0);
    EXPECT_EQ(s.sequences, 4);
    EXPECT_EQ(s.cached_tokens, 30 + 64 + 100 + 16);
}

TEST_P(KvBlockContract, TakesTheLowestFreeBlockFirst) {
    auto kv = make(8);
    kv->create(1);
    kv->append(1, 40);  // 3 blocks
    kv->create(2);
    kv->append(2, 20);  // 2 blocks
    EXPECT_EQ(kv->table(1), ids({0, 1, 2}));
    EXPECT_EQ(kv->table(2), ids({3, 4}));
    kv->release(1);
    kv->create(3);
    kv->append(3, 33);  // 3 blocks: the freed 0, 1, 2 again
    EXPECT_EQ(kv->table(3), ids({0, 1, 2}));
    kv->check_invariants();
}

TEST_P(KvBlockContract, NewBlocksOnlyAtBlockBoundaries) {
    auto kv = make(8);
    kv->create(7);
    EXPECT_EQ(kv->blocks_needed(7, 0), 0);
    EXPECT_EQ(kv->blocks_needed(7, 1), 1);
    EXPECT_EQ(kv->blocks_needed(7, 16), 1);
    EXPECT_EQ(kv->blocks_needed(7, 17), 2);
    kv->append(7, 15);
    EXPECT_EQ(kv->blocks_needed(7, 1), 0) << "position 15 is still in block 0";
    kv->append(7, 1);
    EXPECT_EQ(kv->table(7).size(), 1u);
    EXPECT_EQ(kv->blocks_needed(7, 1), 1) << "position 16 starts block 1";
    for (int t = 0; t < 16; ++t) {  // one token at a time, as decode does
        kv->append(7, 1);
        EXPECT_EQ(kv->table(7).size(), 2u);
    }
    kv->append(7, 1);
    EXPECT_EQ(kv->table(7).size(), 3u);
    EXPECT_EQ(kv->length(7), 33);
    kv->check_invariants();
}

TEST_P(KvBlockContract, ExhaustionThrowsAndChangesNothing) {
    auto kv = make(4);
    kv->create(1);
    kv->append(1, 48);  // 3 of 4 blocks
    kv->create(2);
    EXPECT_THROW(kv->append(2, 17), KvCacheExhausted);  // needs 2, 1 free
    EXPECT_EQ(kv->length(2), 0);
    EXPECT_TRUE(kv->table(2).empty());
    EXPECT_EQ(kv->free_blocks(), 1);
    kv->append(2, 16);  // exactly the last block
    EXPECT_EQ(kv->free_blocks(), 0);
    EXPECT_THROW(kv->append(1, 1), KvCacheExhausted);
    EXPECT_EQ(kv->length(1), 48);
    kv->append(2, 0);  // nothing needed: fine even with no free block
    kv->check_invariants();
}

TEST_P(KvBlockContract, ASequenceCannotOutgrowItsLimit) {
    auto kv = make(16, /*max_per_sequence=*/2);
    kv->create(1);
    kv->append(1, 32);
    EXPECT_THROW(kv->append(1, 1), std::length_error);
    EXPECT_EQ(kv->length(1), 32);
    EXPECT_EQ(kv->table(1).size(), 2u);
    kv->check_invariants();
}

TEST_P(KvBlockContract, ReleaseReturnsEveryBlock) {
    auto kv = make(8);
    for (neuroserve::SeqId s = 0; s < 4; ++s) {
        kv->create(s);
        kv->append(s, 20 + static_cast<int>(s));
    }
    EXPECT_EQ(kv->free_blocks(), 0);
    for (neuroserve::SeqId s = 0; s < 4; ++s) {
        kv->release(s);
        EXPECT_FALSE(kv->contains(s));
        kv->check_invariants();
    }
    EXPECT_EQ(kv->free_blocks(), 8);
    for (BlockId b = 0; b < 8; ++b) {
        EXPECT_EQ(kv->ref_count(b), 0);
    }
    kv->create(2);  // a released id can be used again
    EXPECT_EQ(kv->length(2), 0);
}

TEST_P(KvBlockContract, SharedPrefixBlocksAreCountedAndOutliveTheirFirstOwner) {
    auto kv = make(16);
    kv->create(1);
    kv->append(1, 40);  // blocks 0, 1 full; block 2 partial
    kv->create(2);
    kv->share_prefix(2, 1, 2);
    EXPECT_EQ(kv->table(2), ids({0, 1}));
    EXPECT_EQ(kv->length(2), 32);
    EXPECT_EQ(kv->ref_count(0), 2);
    EXPECT_EQ(kv->ref_count(1), 2);
    EXPECT_EQ(kv->ref_count(2), 1);
    EXPECT_EQ(kv->stats().shared_blocks, 2);
    EXPECT_EQ(kv->free_blocks(), 13) << "sharing takes no new block";
    kv->check_invariants();

    kv->append(2, 5);  // the shared blocks are full: the new tokens go to a new block of 2's own
    EXPECT_EQ(kv->table(2), ids({0, 1, 3}));
    EXPECT_EQ(kv->table(1), ids({0, 1, 2})) << "the first owner is unchanged";
    kv->check_invariants();

    kv->release(1);  // shared blocks survive; only 1's own block is freed
    EXPECT_EQ(kv->ref_count(0), 1);
    EXPECT_EQ(kv->ref_count(1), 1);
    EXPECT_EQ(kv->ref_count(2), 0);
    EXPECT_EQ(kv->free_blocks(), 13);
    kv->check_invariants();

    kv->release(2);  // the last owner: everything is free
    EXPECT_EQ(kv->free_blocks(), 16);
    kv->check_invariants();
}

TEST_P(KvBlockContract, OnlyFullBlocksCanBeShared) {
    auto kv = make(16);
    kv->create(1);
    kv->append(1, 40);  // 2 full blocks
    kv->create(2);
    EXPECT_THROW(kv->share_prefix(2, 1, 3), std::invalid_argument) << "block 2 is partial";
    EXPECT_THROW(kv->share_prefix(2, 1, -1), std::invalid_argument);
    EXPECT_THROW(kv->share_prefix(1, 1, 1), std::invalid_argument) << "with itself";
    EXPECT_THROW(kv->share_prefix(2, 99, 1), std::invalid_argument) << "unknown source";
    kv->append(2, 1);
    EXPECT_THROW(kv->share_prefix(2, 1, 1), std::invalid_argument) << "destination not empty";
    EXPECT_EQ(kv->table(2).size(), 1u);
    EXPECT_EQ(kv->ref_count(0), 1);
    kv->create(3);
    kv->share_prefix(3, 1, 0);  // zero blocks: a no-op
    EXPECT_EQ(kv->length(3), 0);
    kv->check_invariants();
}

TEST_P(KvBlockContract, ManySharersOfOnePrefix) {
    // 16 requests with a 512-token (32-block) prefix hold 32 blocks between them, not 512.
    auto kv = make(64);
    kv->create(0);
    kv->append(0, 512);
    for (neuroserve::SeqId s = 1; s < 16; ++s) {
        kv->create(s);
        kv->share_prefix(s, 0, 32);
        kv->append(s, 8);  // each continues in a block of its own
    }
    EXPECT_EQ(kv->ref_count(0), 16);
    EXPECT_EQ(kv->stats().shared_blocks, 32);
    EXPECT_EQ(kv->stats().used_blocks, 32 + 15);
    kv->check_invariants();
    for (neuroserve::SeqId s = 0; s < 16; ++s) {
        kv->release(s);
    }
    EXPECT_EQ(kv->free_blocks(), 64);
    kv->check_invariants();
}

TEST_P(KvBlockContract, BadCallsAreRejected) {
    auto kv = make(4);
    kv->create(1);
    EXPECT_THROW(kv->create(1), std::invalid_argument);
    EXPECT_THROW(kv->append(9, 1), std::invalid_argument);
    EXPECT_THROW(kv->append(1, -1), std::invalid_argument);
    EXPECT_THROW(kv->blocks_needed(9, 1), std::invalid_argument);
    EXPECT_THROW(kv->release(9), std::invalid_argument);
    EXPECT_THROW(kv->table(9), std::invalid_argument);
    EXPECT_THROW(kv->length(9), std::invalid_argument);
    EXPECT_THROW(kv->ref_count(4), std::invalid_argument);
    EXPECT_THROW(kv->ref_count(-1), std::invalid_argument);
    EXPECT_THROW(GetParam().make(KvBlockConfig{0, 16, 4}), std::invalid_argument);
    EXPECT_THROW(GetParam().make(KvBlockConfig{4, 0, 4}), std::invalid_argument);
    EXPECT_THROW(GetParam().make(KvBlockConfig{4, 16, 0}), std::invalid_argument);
    kv->check_invariants();
}

TEST_P(KvBlockContract, InvariantsHoldThroughARandomWorkload) {
    // A smoke test of the contract under churn; Day 29 compares implementations op by op.
    auto kv = make(48, 8);
    std::mt19937 rng(2026);
    std::vector<neuroserve::SeqId> live;
    neuroserve::SeqId next = 0;
    for (int op = 0; op < 3000; ++op) {
        const int kind = static_cast<int>(rng() % 4);
        if (kind == 0 || live.empty()) {
            kv->create(next);
            live.push_back(next++);
        } else if (kind == 1) {
            const auto s = live[rng() % live.size()];
            try {
                kv->append(s, static_cast<int>(rng() % 40));
            } catch (const KvCacheExhausted&) {
            } catch (const std::length_error&) {
            }
        } else if (kind == 2) {
            const auto src = live[rng() % live.size()];
            kv->create(next);
            const int full = kv->length(src) / 16;
            kv->share_prefix(next, src, static_cast<int>(rng() % static_cast<unsigned>(full + 1)));
            live.push_back(next++);
        } else {
            const std::size_t i = rng() % live.size();
            kv->release(live[i]);
            live.erase(live.begin() + static_cast<std::ptrdiff_t>(i));
        }
        kv->check_invariants();
    }
    for (const auto s : live) {
        kv->release(s);
    }
    EXPECT_EQ(kv->free_blocks(), 48);
    kv->check_invariants();
}

INSTANTIATE_TEST_SUITE_P(
    Implementations, KvBlockContract,
    ::testing::Values(Implementation{"Reference",
                                     [](KvBlockConfig c) { return std::make_unique<ReferenceBlockManager>(c); }}),
    [](const ::testing::TestParamInfo<Implementation>& p) { return p.param.name; });

}  // namespace
