#pragma once
// Variable-length prefill attention (docs/prefill-attention.md).
//
// Causal attention for B sequences whose new tokens are packed one after another. Sequence b has
// start[b] cached tokens and T[b] = cu_seqlens_q[b + 1] - cu_seqlens_q[b] new tokens; the K and V
// of all start[b] + T[b] positions are in its KV pages already (the caller scatters the new ones
// first). New token i of sequence b is at position start[b] + i and attends to keys
// 0 .. start[b] + i, read through the sequence's row of block_tables.
//
// Layouts (FP16):
//   q, out          [total_q, n_heads, head_dim], token-major, total_q = cu_seqlens_q[B]
//   k_pool, v_pool  [num_pages * page_size, n_kv_heads, head_dim]: row page * page_size + slot
//   block_tables    [B, max_pages] page ids, row-major
//
// Each sequence's output depends only on its own inputs: packed with others or alone, in any
// position, it is bit-identical (tested). The same inputs always give the same bits.

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace neuroserve::kernels {

struct VarlenAttentionArgs {
    const __half* q = nullptr;
    const __half* k_pool = nullptr;
    const __half* v_pool = nullptr;
    const std::int32_t* cu_seqlens_q = nullptr;  // device, [B + 1]
    const std::int32_t* start = nullptr;         // device, [B]
    const std::int32_t* block_tables = nullptr;  // device, [B, max_pages]
    __half* out = nullptr;
    int num_seqs = 0;     // B
    int total_q = 0;      // cu_seqlens_q[B], known on the host
    int max_keys = 0;     // max over b of start[b] + T[b], known on the host: sizes shared memory
    int max_pages = 0;
    int n_heads = 0;
    int n_kv_heads = 0;   // n_heads must be a multiple of it (grouped-query attention)
    int head_dim = 0;
    int page_size = 0;
    float scale = 0.f;    // usually 1 / sqrt(head_dim)
};

// v0 (Day 48): one thread block per (new token, query head), the softmax in FP32 over every key
// in shared memory (max_keys floats). Simple and obviously correct; the reference for the tiled
// kernel. Throws std::invalid_argument for arguments it cannot serve (shared memory over 48 KB,
// i.e. more than 12,288 keys; head counts that do not divide). Launches on `stream`; does not
// synchronize.
void attention_prefill_varlen_v0(const VarlenAttentionArgs& args, cudaStream_t stream = nullptr);

}  // namespace neuroserve::kernels
