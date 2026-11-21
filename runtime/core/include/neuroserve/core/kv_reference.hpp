#pragma once

#include <map>
#include <optional>
#include <vector>

#include "neuroserve/core/kv_block_manager.hpp"

namespace neuroserve::core {

// The CPU reference block manager: the test oracle for the real KVCacheManager (Days 28–29).
// Written to be obviously correct rather than fast: a reference count per block, a map of block
// tables, and linear scans (the lowest free block is found by searching from 0; a prefix hit by
// scanning every block's record; commit re-walks the whole chain and keeps no cursor).
class ReferenceBlockManager final : public KvBlockManager {
public:
    // std::invalid_argument unless num_blocks, block_size and max_blocks_per_sequence are positive,
    // or if prefix_cache is given with another block size.
    explicit ReferenceBlockManager(KvBlockConfig config, std::optional<KvCacheIdentity> prefix_cache = std::nullopt);

    const KvBlockConfig& config() const noexcept override { return config_; }
    void create(SeqId seq) override;
    int blocks_needed(SeqId seq, int tokens) const override;
    void append(SeqId seq, int tokens) override;
    void share_prefix(SeqId dst, SeqId src, int blocks) override;
    void release(SeqId seq) override;
    bool contains(SeqId seq) const override;
    std::vector<BlockId> table(SeqId seq) const override;
    int length(SeqId seq) const override;
    int ref_count(BlockId block) const override;
    int free_blocks() const override;
    KvBlockStats stats() const override;
    bool prefix_caching() const noexcept override { return hasher_.has_value(); }
    Admission admit(SeqId seq, std::span<const TokenId> prompt) override;
    int commit(SeqId seq, std::span<const TokenId> tokens) override;
    int match(std::span<const TokenId> prompt) const override;
    bool indexed(BlockId block) const override;
    int evict_cached() override;
    void check_invariants() const override;

private:
    struct Record {  // of an indexed block
        bool indexed = false;
        BlockHash hash = 0;
        BlockId parent = -1;  // the block before it in the table that indexed it; -1 for the first
        std::vector<TokenId> tokens;
        std::uint64_t released_at = 0;  // CACHED (indexed, count 0): when it was released
    };
    struct Sequence {
        std::vector<BlockId> table;
        int length = 0;
    };

    const Sequence& get(SeqId seq) const;
    Sequence& get(SeqId seq);
    int blocks_for(long long tokens) const;
    const PrefixHasher& hasher(const char* op) const;
    std::vector<BlockId> walk(std::span<const TokenId> prompt) const;  // the blocks admit() would reuse
    BlockId take_block(BlockId keep = -1);  // a FREE block, lowest id; else evicts the oldest CACHED

    KvBlockConfig config_;
    std::vector<int> refs_;               // reference count per block
    std::map<SeqId, Sequence> sequences_;  // ordered: deterministic iteration and messages
    std::optional<PrefixHasher> hasher_;
    std::vector<Record> records_;          // per block
    std::int64_t lookups_ = 0;
    std::int64_t lookup_blocks_ = 0;
    std::int64_t hit_blocks_ = 0;
    std::int64_t copied_blocks_ = 0;
    std::int64_t reused_tokens_ = 0;
    std::uint64_t release_clock_ = 0;
    std::int64_t evicted_blocks_ = 0;
};

}  // namespace neuroserve::core
