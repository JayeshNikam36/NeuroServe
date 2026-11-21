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
// tokens (the backend gives it the manager's blocks); 0 < start < ids.size(). Sets seq.len and
// returns the last token's logits, like model_prefill_paged.
// Copies page `from`'s K and V, in every layer, into page `to` (copy-on-write: a reused page the
// prefill is about to write into). Stream-ordered before the prefill's kernels.
void copy_page(const Model& m, PagedKVCache& c, int from, int to);

void model_prefill_from(Model& m, PagedKVCache& c, Sequence& seq, const std::vector<int>& ids, int start,
                        std::vector<float>& logits);

}  // namespace neuroserve::backend_detail
