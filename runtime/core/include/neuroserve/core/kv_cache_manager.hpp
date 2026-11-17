#pragma once

#include <set>
#include <unordered_map>
#include <vector>

#include "neuroserve/core/kv_block_manager.hpp"

namespace neuroserve::core {

// The KV block manager the engine uses: the single owner of KV memory accounting
// (docs/kv-cache.md). It meets the KvBlockManager contract exactly like the CPU reference, which is
// its test oracle, but every operation is proportional to the blocks it touches: an ordered set of
// free blocks (lowest id first, O(log n)), and running counts for the statistics.
class KVCacheManager final : public KvBlockManager {
public:
    // std::invalid_argument unless num_blocks, block_size and max_blocks_per_sequence are positive.
    explicit KVCacheManager(KvBlockConfig config);

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
    void check_invariants() const override;

    // The sequence that took a used block from the free list (and so wrote its K/V). Sharers do not
    // change it, and it stays when that sequence is released while others still use the block, so
    // a leaked or wrongly written block names the request it came from (KV-003).
    // std::invalid_argument for a free block or an id outside the pool.
    SeqId allocated_by(BlockId block) const;

private:
    struct Block {
        int refs = 0;
        SeqId allocated_by = 0;
    };
    struct Sequence {
        std::vector<BlockId> table;
        int length = 0;
    };

    const Sequence& get(SeqId seq) const;
    Sequence& get(SeqId seq);
    int blocks_for(long long tokens) const;
    const Block& block_at(BlockId block) const;

    KvBlockConfig config_;
    std::vector<Block> blocks_;
    std::set<BlockId> free_;                         // reference count 0, lowest id first
    std::unordered_map<SeqId, Sequence> sequences_;
    int shared_blocks_ = 0;                          // reference count >= 2
    std::int64_t cached_tokens_ = 0;                 // sum of the sequences' lengths
};

}  // namespace neuroserve::core
