#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "neuroserve/core/inference_backend.hpp"  // SeqId, TokenId, KvCacheExhausted
#include "neuroserve/core/kv_prefix_hash.hpp"

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
    // Prefix caching (zero without it):
    int indexed_blocks = 0;               // full blocks in the prefix index
    std::int64_t prefix_lookups = 0;      // successful admit() calls
    std::int64_t prefix_lookup_blocks = 0;  // full prompt blocks they could have reused
    std::int64_t prefix_hit_blocks = 0;   // of those, the blocks they did reuse
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
// - Prefix caching (a manager built with a KvCacheIdentity): a full block whose K/V are written is
//   indexed by its chained hash (commit), and admit() reuses indexed blocks for a new prompt. A hit
//   is verified, not trusted: the indexed block's tokens must equal the prompt's, and its parent must
//   be the block the walk reused just before it. A block is unindexed when its last reference goes
//   (it is free again; keeping it as CACHED comes with eviction, docs/kv-cache.md).
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

    // ---- Prefix caching (docs/kv-cache.md, "Prefix caching"). Without a KvCacheIdentity,
    // prefix_caching() is false and admit, commit and match throw std::logic_error.
    virtual bool prefix_caching() const noexcept = 0;

    // Admits the empty sequence `seq` with `prompt`: walks the prompt's full blocks in order, reusing
    // each one the index holds (reference count + 1) and stopping at the first miss, then appends the
    // rest of the prompt. Returns the tokens it reused (a multiple of block_size): the prefill starts
    // there. The block that holds the last prompt token is never reused, because computing that
    // token's logits writes its K/V (copy-on-write, Day 35, lifts this), so at most
    // prompt.size() - 1 tokens are reused. Counts the lookup, the blocks it could have reused and the
    // hits. Throws like append (nothing reused, nothing counted); std::invalid_argument for an
    // unknown or non-empty sequence or an empty prompt.
    virtual int admit(SeqId seq, std::span<const TokenId> prompt) = 0;

    // Indexes the sequence's full blocks, now that their K/V are written. `tokens` are all its tokens
    // (tokens.size() == length(seq)). Block by block, in order: a block already indexed must hold
    // these tokens (std::invalid_argument otherwise); a block whose prefix another block already holds
    // in the index is a duplicate, and indexing stops there (a later call retries it); any other block
    // is indexed under its chained hash, with the block before it in the table as its parent.
    // Returns how many blocks it indexed.
    virtual int commit(SeqId seq, std::span<const TokenId> tokens) = 0;

    // How many leading full blocks of `prompt` admit() would reuse now. Changes nothing.
    virtual int match(std::span<const TokenId> prompt) const = 0;

    // Whether the block is in the prefix index. std::invalid_argument outside the pool.
    virtual bool indexed(BlockId block) const = 0;

    // Checks every invariant this implementation can see; KvInvariantViolation names the first
    // one broken, with the block or sequence and both sides of the equation.
    virtual void check_invariants() const = 0;

protected:
    KvBlockManager() = default;
};

}  // namespace neuroserve::core
