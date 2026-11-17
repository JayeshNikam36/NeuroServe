#pragma once

#include <map>
#include <vector>

#include "neuroserve/core/kv_block_manager.hpp"

namespace neuroserve::core {

// The CPU reference block manager: the test oracle for the real KVCacheManager (Days 28–29).
// Written to be obviously correct rather than fast: a reference count per block, a map of block
// tables, and linear scans (the lowest free block is found by searching from 0).
class ReferenceBlockManager final : public KvBlockManager {
public:
    // std::invalid_argument unless num_blocks, block_size and max_blocks_per_sequence are positive.
    explicit ReferenceBlockManager(KvBlockConfig config);

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

private:
    struct Sequence {
        std::vector<BlockId> table;
        int length = 0;
    };

    const Sequence& get(SeqId seq) const;
    Sequence& get(SeqId seq);
    int blocks_for(long long tokens) const;

    KvBlockConfig config_;
    std::vector<int> refs_;               // reference count per block
    std::map<SeqId, Sequence> sequences_;  // ordered: deterministic iteration and messages
};

}  // namespace neuroserve::core
