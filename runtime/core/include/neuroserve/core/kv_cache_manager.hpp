#pragma once

#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include "neuroserve/core/kv_block_manager.hpp"

namespace neuroserve::core {

struct KVCacheManagerTestPeer;  // defined only in tests: injects faults to test check_invariants()

// The KV block manager the engine uses: the single owner of KV memory accounting
// (docs/kv-cache.md). It meets the KvBlockManager contract exactly like the CPU reference, which is
// its test oracle, but every operation is proportional to the blocks it touches: an ordered set of
// free blocks (lowest id first, O(log n)), running counts for the statistics, and a hash map for
// the prefix index (a lookup or a commit costs one hash and one map probe per block).
class KVCacheManager final : public KvBlockManager {
public:
    // std::invalid_argument unless num_blocks, block_size and max_blocks_per_sequence are positive,
    // or if prefix_cache is given with another block size. With prefix_cache, prefix caching is on.
    explicit KVCacheManager(KvBlockConfig config, std::optional<KvCacheIdentity> prefix_cache = std::nullopt);

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

    // The sequence that took a used block from the free list (and so wrote its K/V). Sharers do not
    // change it, and it stays when that sequence is released while others still use the block, so
    // a leaked or wrongly written block names the request it came from (KV-003).
    // std::invalid_argument for a free block or an id outside the pool.
    SeqId allocated_by(BlockId block) const;

private:
    friend struct KVCacheManagerTestPeer;

    struct Block {
        int refs = 0;
        SeqId allocated_by = 0;
        bool indexed = false;          // in index_ under `hash`
        BlockHash hash = 0;
        BlockId parent = -1;           // the block before it in the table that indexed it; -1 first
        std::vector<TokenId> tokens;   // its tokens, to verify a hit (only while indexed)
        std::uint64_t released_at = 0; // CACHED: its place in the LRU queue
    };
    struct Sequence {
        std::vector<BlockId> table;
        int length = 0;
    };

    const Sequence& get(SeqId seq) const;
    Sequence& get(SeqId seq);
    int blocks_for(long long tokens) const;
    const Block& block_at(BlockId block) const;
    const PrefixHasher& hasher(const char* op) const;
    std::vector<BlockId> walk(std::span<const TokenId> prompt) const;  // the blocks admit() would reuse
    void take_free_block(SeqId seq, Sequence& s, BlockId keep = -1);  // FREE first, else evicts LRU
    void unindex(BlockId block);

    KvBlockConfig config_;
    std::vector<Block> blocks_;
    std::set<BlockId> free_;                         // reference count 0, lowest id first
    std::unordered_map<SeqId, Sequence> sequences_;
    int shared_blocks_ = 0;                          // reference count >= 2
    std::int64_t cached_tokens_ = 0;                 // sum of the sequences' lengths
    std::optional<PrefixHasher> hasher_;
    std::unordered_map<BlockHash, BlockId> index_;   // the prefix index: full, written blocks
    std::int64_t lookups_ = 0;
    std::int64_t lookup_blocks_ = 0;
    std::int64_t hit_blocks_ = 0;
    std::int64_t copied_blocks_ = 0;
    std::int64_t reused_tokens_ = 0;
    std::set<std::pair<std::uint64_t, BlockId>> lru_;  // CACHED blocks, least recently released first
    std::uint64_t release_clock_ = 0;
    std::int64_t evicted_blocks_ = 0;
};

}  // namespace neuroserve::core
