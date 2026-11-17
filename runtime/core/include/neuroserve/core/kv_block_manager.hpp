#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "neuroserve/core/inference_backend.hpp"  // SeqId, KvCacheExhausted

namespace neuroserve::core {

using BlockId = std::int32_t;

struct KvBlockConfig {
    int num_blocks = 0;               // blocks in the pool; ids are 0 .. num_blocks - 1
    int block_size = 16;              // tokens per block
    int max_blocks_per_sequence = 0;  // the longest block table a sequence may have
};

struct KvBlockStats {
    int total_blocks = 0;
    int free_blocks = 0;    // reference count 0
    int used_blocks = 0;    // reference count >= 1
    int shared_blocks = 0;  // reference count >= 2
    int sequences = 0;
    std::int64_t cached_tokens = 0;  // sum of the sequences' lengths
};

// A broken internal invariant of a block manager (see docs/kv-cache.md, "Invariants").
class KvInvariantViolation : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

// Accounting for a pool of fixed-size KV blocks (docs/kv-cache.md). It decides which physical block
// holds which positions of which sequence; the GPU memory itself belongs to the backend.
//
// The contract, shared by every implementation so that they can be compared operation by operation:
// - A sequence's block table lists its blocks in logical order: position p is in
//   table[p / block_size], slot p % block_size. It has exactly ceil(length / block_size) entries.
// - New blocks are taken lowest id first, so the same operations give the same tables everywhere.
// - A block's reference count is the number of tables that list it; at 0 it is free.
// - Only full blocks are shared (share_prefix), so a shared block is never written again.
// - Every operation either succeeds or throws and leaves the state unchanged.
// Not thread-safe: only the engine thread uses it (docs/kv-cache.md, "Ownership and threads").
class KvBlockManager {
public:
    virtual ~KvBlockManager() = default;
    KvBlockManager(const KvBlockManager&) = delete;
    KvBlockManager& operator=(const KvBlockManager&) = delete;

    virtual const KvBlockConfig& config() const noexcept = 0;

    // Starts an empty sequence (no blocks). std::invalid_argument if the id is in use.
    virtual void create(SeqId seq) = 0;

    // New blocks that appending `tokens` tokens to the sequence would take (0 within its last block).
    virtual int blocks_needed(SeqId seq, int tokens) const = 0;

    // Grows the sequence by `tokens` positions, taking new blocks at block boundaries.
    // KvCacheExhausted if there are not enough free blocks; std::length_error past
    // max_blocks_per_sequence; std::invalid_argument for an unknown sequence or tokens < 0.
    virtual void append(SeqId seq, int tokens) = 0;

    // Makes the empty sequence `dst` start with the first `blocks` blocks of `src`, which must be
    // full: their reference counts go up by one and dst's length becomes blocks * block_size.
    // std::invalid_argument if dst is not empty, src == dst, or src has fewer full blocks.
    virtual void share_prefix(SeqId dst, SeqId src, int blocks) = 0;

    // Drops the sequence's references; blocks no other sequence lists become free. The id may be
    // created again afterwards. std::invalid_argument for an unknown sequence.
    virtual void release(SeqId seq) = 0;

    virtual bool contains(SeqId seq) const = 0;
    virtual std::vector<BlockId> table(SeqId seq) const = 0;
    virtual int length(SeqId seq) const = 0;      // tokens (positions) the sequence holds
    virtual int ref_count(BlockId block) const = 0;
    virtual int free_blocks() const = 0;
    virtual KvBlockStats stats() const = 0;

    // Checks every invariant this implementation can see; KvInvariantViolation names the first
    // one broken, with the block or sequence and both sides of the equation.
    virtual void check_invariants() const = 0;

protected:
    KvBlockManager() = default;
};

}  // namespace neuroserve::core
