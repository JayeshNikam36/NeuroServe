#pragma once
// Prefill from a cached offset (docs/kv-cache.md, "Prefill from a cached offset", Day 34).
// Internal to the CUDA backend: uses cuda-llm-engine's types and kernels, without modifying it.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

#include "paged.h"

namespace neuroserve::backend_detail {

// Causal attention for T new query tokens over a paged cache: query t sits at position start + t
// and attends to positions 0 .. start + t, all read through the block table (the cached prefix and
// the new tokens' own K/V, which must already be scattered into the pool).
// q, out: [T, n_heads, head_dim]. k_pool, v_pool: [num_pages, page_size, n_kv_heads, head_dim].
// The per-query math is attention_decode_paged's, so a suffix token gets the same attention as the
// decode path would give it.
void attention_prefill_paged(const __half* q, const __half* k_pool, const __half* v_pool,
                             const int* block_table, __half* out, int start, int T, int page_size,
                             int n_heads, int n_kv_heads, int head_dim, cudaStream_t stream = 0);

// Prefills ids[start ..] for a sequence whose first `start` positions are already in its pages
// (reused from the prefix cache): positions and RoPE start at `start`, the new K/V go into the
// sequence's pages, and attention covers the whole prefix. seq.pages must already cover ids.size()
// tokens (the backend gives it the manager's blocks); 0 < start < ids.size(). The sequence's length
// is 0 (a prefix from the cache) or start (the next chunk of a chunked prefill). At most m.max_T new
// tokens. Sets seq.len and returns the last token's logits, like model_prefill_paged.
// Copies page `from`'s K and V, in every layer, into page `to` (copy-on-write: a reused page the
// prefill is about to write into). Stream-ordered before the prefill's kernels.
void copy_page(const Model& m, PagedKVCache& c, int from, int to);

void model_prefill_from(Model& m, PagedKVCache& c, Sequence& seq, const std::vector<int>& ids, int start,
                        std::vector<float>& logits);

// ---- Batched prefill (Day 51, docs/prefill-attention.md) ----------------------------------------

// One sequence of a batched prefill: its whole input ids, of which the first `start` positions are
// already in its pages (a prefix-cache hit; 0 for a fresh prompt). seq.pages must cover ids.size()
// tokens and seq.len must be 0.
struct PrefillPart {
    Sequence* seq = nullptr;
    const std::vector<int>* ids = nullptr;
    int start = 0;
};

// Device buffers for a batched prefill of up to max_seqs sequences: their block tables, offsets
// and tile map, and the last rows and logits of every sequence.
struct PrefillBatchScratch {
    int max_seqs = 0;
    int max_pages = 0;
    int max_tiles = 0;
    int* d_tables = nullptr;    // [max_seqs, max_pages]
    int* d_cu = nullptr;        // [max_seqs + 1]
    int* d_start = nullptr;     // [max_seqs]
    int* d_tile_map = nullptr;  // [2 * max_tiles]
    __half* d_last = nullptr;   // [max_seqs, hidden]: each sequence's last hidden row
    __half* d_last_norm = nullptr;
    __half* d_logits = nullptr;  // [max_seqs, vocab]
};

PrefillBatchScratch alloc_prefill_batch_scratch(const Model& m, int max_seqs, int max_pages);
void free_prefill_batch_scratch(PrefillBatchScratch& s);

// Prefills several sequences in one forward pass: their new tokens are packed (at most m.max_T in
// all), every projection and the MLP run once on the pack, each sequence's K/V go into its own
// pages, and attention is the varlen tiled kernel over the pack. Sets every seq.len and returns
// each sequence's next token (the argmax of its last position's logits, ties to the lowest id).
std::vector<int> model_prefill_batch(Model& m, PagedKVCache& c, const std::vector<PrefillPart>& parts,
                                     PrefillBatchScratch& s);

}  // namespace neuroserve::backend_detail
