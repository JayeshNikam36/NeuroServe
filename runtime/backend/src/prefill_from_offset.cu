#include "prefill_from_offset.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "neuroserve/kernels/attention_varlen.hpp"

#include "embedding.cuh"
#include "kv_scatter.cuh"
#include "linear.cuh"
#include "residual_add.cuh"
#include "rmsnorm.cuh"
#include "rope.cuh"
#include "swiglu.cuh"
#include "util.h"

namespace neuroserve::backend_detail {
namespace {

// Row of the pool where position j is stored: page block_table[j / page_size], slot j % page_size.
__device__ __forceinline__ size_t paged_row(const int* __restrict__ block_table, int j, int page_size) {
    return static_cast<size_t>(block_table[j / page_size]) * page_size + (j % page_size);
}

// One block per (query head, new token). The math, step by step, is attention_decode_paged's:
// float dot products, softmax with the max subtracted, one thread summing in order, then the value
// blend in position order. Only the length differs per query: S = start + t + 1 (causal).
__global__ void attention_prefill_paged_kernel(const __half* __restrict__ q, const __half* __restrict__ k_pool,
                                               const __half* __restrict__ v_pool,
                                               const int* __restrict__ block_table, __half* __restrict__ out,
                                               int start, int page_size, int n_heads, int n_kv_heads, int head_dim) {
    extern __shared__ float scores[];  // S floats
    __shared__ float s_max, s_sum;

    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int S = start + t + 1;
    const int g = h / (n_heads / n_kv_heads);  // grouped-query sharing
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const __half* qv = q + (static_cast<size_t>(t) * n_heads + h) * head_dim;

    for (int j = threadIdx.x; j < S; j += blockDim.x) {
        const __half* kv = k_pool + (paged_row(block_table, j, page_size) * n_kv_heads + g) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++) dot += __half2float(qv[d]) * __half2float(kv[d]);
        scores[j] = dot * scale;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int j = 0; j < S; j++) mx = fmaxf(mx, scores[j]);
        s_max = mx;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < S; j += blockDim.x) scores[j] = expf(scores[j] - s_max);
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int j = 0; j < S; j++) sum += scores[j];
        s_sum = sum;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < S; j += blockDim.x) scores[j] /= s_sum;
    __syncthreads();

    __half* o = out + (static_cast<size_t>(t) * n_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < S; j++)
            acc += scores[j] * __half2float(v_pool[(paged_row(block_table, j, page_size) * n_kv_heads + g) * head_dim + d]);
        o[d] = __float2half_rn(acc);
    }
}

// The engine's layer_forward_paged (engine/paged.cu, static there), with the new tokens at
// positions start .. start + T - 1 and attention over the whole cached prefix.
void layer_forward_from(cublasHandle_t handle, const LayerConfig& c, const LayerWeights& w, LayerScratch& s,
                        __half* x, const int* d_positions, const float* d_inv_freq, int T, int start,
                        __half* k_pool, __half* v_pool, const int* d_table, int page_size) {
    const int qdim = c.n_heads * c.head_dim, kvdim = c.n_kv_heads * c.head_dim;
    const size_t n_hidden = static_cast<size_t>(T) * c.hidden;

    rmsnorm(x, w.attn_norm, s.norm, T, c.hidden, c.eps);
    linear(handle, s.norm, w.wq, s.q, T, c.hidden, qdim);
    linear(handle, s.norm, w.wk, s.k, T, c.hidden, kvdim);
    linear(handle, s.norm, w.wv, s.v, T, c.hidden, kvdim);
    rope_apply(s.q, s.k, d_positions, d_inv_freq, T, c.n_heads, c.n_kv_heads, c.head_dim);

    kv_scatter(s.k, k_pool, d_table, start, T, page_size, kvdim);  // the new tokens' K/V first:
    kv_scatter(s.v, v_pool, d_table, start, T, page_size, kvdim);  // each query also attends to itself
    attention_prefill_paged(s.q, k_pool, v_pool, d_table, s.attn, start, T, page_size, c.n_heads, c.n_kv_heads,
                            c.head_dim);

    linear(handle, s.attn, w.wo, s.proj, T, qdim, c.hidden);
    residual_add(x, s.proj, x, n_hidden);

    rmsnorm(x, w.mlp_norm, s.norm, T, c.hidden, c.eps);
    linear(handle, s.norm, w.w_gate, s.gate, T, c.hidden, c.inter);
    linear(handle, s.norm, w.w_up, s.up, T, c.hidden, c.inter);
    swiglu(s.gate, s.up, s.act, static_cast<size_t>(T) * c.inter);
    linear(handle, s.act, w.w_down, s.proj, T, c.inter, c.hidden);
    residual_add(x, s.proj, x, n_hidden);
}

// layer_forward_from for a pack of sequences: the projections, RoPE (packed positions) and the MLP
// run once on all T packed tokens; K/V are scattered into each sequence's own pages; attention is
// the varlen tiled kernel (`a` describes the pack; its q and out are the scratch's).
void layer_forward_batch(cublasHandle_t handle, const LayerConfig& c, const LayerWeights& w, LayerScratch& s,
                         __half* x, const int* d_positions, const float* d_inv_freq, int T, __half* k_pool,
                         __half* v_pool, const std::vector<PrefillPart>& parts, const std::vector<int>& cu,
                         int page_size, neuroserve::kernels::VarlenAttentionArgs a, const int* d_tile_map,
                         int num_tiles) {
    const int qdim = c.n_heads * c.head_dim, kvdim = c.n_kv_heads * c.head_dim;
    const size_t n_hidden = static_cast<size_t>(T) * c.hidden;

    rmsnorm(x, w.attn_norm, s.norm, T, c.hidden, c.eps);
    linear(handle, s.norm, w.wq, s.q, T, c.hidden, qdim);
    linear(handle, s.norm, w.wk, s.k, T, c.hidden, kvdim);
    linear(handle, s.norm, w.wv, s.v, T, c.hidden, kvdim);
    rope_apply(s.q, s.k, d_positions, d_inv_freq, T, c.n_heads, c.n_kv_heads, c.head_dim);

    for (size_t b = 0; b < parts.size(); ++b) {  // each sequence's new K/V into its own pages, first
        const PrefillPart& p = parts[b];
        const size_t row = static_cast<size_t>(cu[b]) * kvdim;
        const int Tb = cu[b + 1] - cu[b];
        kv_scatter(s.k + row, k_pool, p.seq->d_table, p.start, Tb, page_size, kvdim);
        kv_scatter(s.v + row, v_pool, p.seq->d_table, p.start, Tb, page_size, kvdim);
    }
    a.q = s.q;
    a.out = s.attn;
    a.k_pool = k_pool;
    a.v_pool = v_pool;
    neuroserve::kernels::attention_prefill_varlen_tiled(a, d_tile_map, num_tiles, neuroserve::kernels::TileShape::k64x32);

    linear(handle, s.attn, w.wo, s.proj, T, qdim, c.hidden);
    residual_add(x, s.proj, x, n_hidden);

    rmsnorm(x, w.mlp_norm, s.norm, T, c.hidden, c.eps);
    linear(handle, s.norm, w.w_gate, s.gate, T, c.hidden, c.inter);
    linear(handle, s.norm, w.w_up, s.up, T, c.hidden, c.inter);
    swiglu(s.gate, s.up, s.act, static_cast<size_t>(T) * c.inter);
    linear(handle, s.act, w.w_down, s.proj, T, c.inter, c.hidden);
    residual_add(x, s.proj, x, n_hidden);
}

// The engine's head_logits (engine/paged.cu, static there).
void head_logits(Model& m, int T, std::vector<float>& logits) {
    const int H = m.cfg.hidden;
    const __half* last = m.d_x + static_cast<size_t>(T - 1) * H;
    rmsnorm(last, m.final_norm, m.d_normed, 1, H, m.cfg.eps);
    linear(m.handle, m.d_normed, m.embed, m.d_logits, 1, H, m.vocab);
    CUDA_CHECK(cudaGetLastError());
    std::vector<__half> host(static_cast<size_t>(m.vocab));
    CUDA_CHECK(cudaMemcpy(host.data(), m.d_logits, static_cast<size_t>(m.vocab) * sizeof(__half), cudaMemcpyDeviceToHost));
    logits.resize(static_cast<size_t>(m.vocab));
    for (int v = 0; v < m.vocab; v++) logits[static_cast<size_t>(v)] = __half2float(host[static_cast<size_t>(v)]);
}

}  // namespace

void attention_prefill_paged(const __half* q, const __half* k_pool, const __half* v_pool, const int* block_table,
                             __half* out, int start, int T, int page_size, int n_heads, int n_kv_heads, int head_dim,
                             cudaStream_t stream) {
    const size_t shmem = static_cast<size_t>(start + T) * sizeof(float);  // the longest query's scores
    if (shmem > 48 * 1024) {
        fprintf(stderr, "attention_prefill_paged: %d positions need %zu bytes of shared memory (limit 48 KB)\n",
                start + T, shmem);
        exit(1);
    }
    const dim3 grid(static_cast<unsigned>(n_heads), static_cast<unsigned>(T));
    attention_prefill_paged_kernel<<<grid, 128, shmem, stream>>>(q, k_pool, v_pool, block_table, out, start,
                                                                 page_size, n_heads, n_kv_heads, head_dim);
}

void copy_page(const Model& m, PagedKVCache& c, int from, int to) {
    const size_t bytes = static_cast<size_t>(c.page_size) * m.cfg.n_kv_heads * m.cfg.head_dim * sizeof(__half);
    const size_t offset_from = static_cast<size_t>(from) * bytes;
    const size_t offset_to = static_cast<size_t>(to) * bytes;
    for (size_t l = 0; l < c.k_pool.size(); ++l) {
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char*>(c.k_pool[l]) + offset_to,
                                   reinterpret_cast<const char*>(c.k_pool[l]) + offset_from, bytes,
                                   cudaMemcpyDeviceToDevice, 0));
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char*>(c.v_pool[l]) + offset_to,
                                   reinterpret_cast<const char*>(c.v_pool[l]) + offset_from, bytes,
                                   cudaMemcpyDeviceToDevice, 0));
    }
}

void model_prefill_from(Model& m, PagedKVCache& c, Sequence& seq, const std::vector<int>& ids, int start,
                        std::vector<float>& logits) {
    const int total = static_cast<int>(ids.size());
    const int T = total - start;  // the uncached tokens
    if ((seq.len != 0 && seq.len != start) || start <= 0 || T <= 0) {  // cached prefix, or the next chunk
        fprintf(stderr, "model_prefill_from: needs 0 < start < %d and the sequence at 0 or start, got start %d, len %d\n",
                total, start, seq.len);
        exit(1);
    }
    if (T > m.max_T || static_cast<int>(seq.pages.size()) * c.page_size < total ||
        static_cast<int>(seq.pages.size()) > seq.max_pages) {
        fprintf(stderr, "model_prefill_from: %d tokens do not fit (%zu pages, scratch %d)\n", total, seq.pages.size(),
                m.max_T);
        exit(1);
    }
    CUDA_CHECK(cudaMemcpy(seq.d_table, seq.pages.data(), seq.pages.size() * sizeof(int), cudaMemcpyHostToDevice));
    std::vector<int> pos(static_cast<size_t>(T));
    for (int t = 0; t < T; t++) pos[static_cast<size_t>(t)] = start + t;  // position ids continue the prefix
    CUDA_CHECK(cudaMemcpy(m.d_ids, ids.data() + start, static_cast<size_t>(T) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.d_pos, pos.data(), static_cast<size_t>(T) * sizeof(int), cudaMemcpyHostToDevice));
    embedding_lookup(m.embed, m.d_ids, m.d_x, T, m.cfg.hidden);
    for (int l = 0; l < m.n_layers; l++) {
        layer_forward_from(m.handle, m.cfg, m.layers[static_cast<size_t>(l)], m.scratch, m.d_x, m.d_pos,
                           m.d_inv_freq, T, start, c.k_pool[static_cast<size_t>(l)], c.v_pool[static_cast<size_t>(l)],
                           seq.d_table, c.page_size);
    }
    head_logits(m, T, logits);
    seq.len = total;
}

PrefillBatchScratch alloc_prefill_batch_scratch(const Model& m, int max_seqs, int max_pages) {
    PrefillBatchScratch s;
    s.max_seqs = max_seqs;
    s.max_pages = max_pages;
    s.max_tiles = m.max_T / 64 + max_seqs;  // ceil(T_b / 64) summed over at most max_seqs sequences
    CUDA_CHECK(cudaMalloc(&s.d_tables, sizeof(int) * static_cast<size_t>(max_seqs) * static_cast<size_t>(max_pages)));
    CUDA_CHECK(cudaMalloc(&s.d_cu, sizeof(int) * static_cast<size_t>(max_seqs + 1)));
    CUDA_CHECK(cudaMalloc(&s.d_start, sizeof(int) * static_cast<size_t>(max_seqs)));
    CUDA_CHECK(cudaMalloc(&s.d_tile_map, sizeof(int) * 2 * static_cast<size_t>(s.max_tiles)));
    const size_t rows = static_cast<size_t>(max_seqs) * static_cast<size_t>(m.cfg.hidden);
    CUDA_CHECK(cudaMalloc(&s.d_last, sizeof(__half) * rows));
    CUDA_CHECK(cudaMalloc(&s.d_last_norm, sizeof(__half) * rows));
    CUDA_CHECK(cudaMalloc(&s.d_logits, sizeof(__half) * static_cast<size_t>(max_seqs) * static_cast<size_t>(m.vocab)));
    return s;
}

void free_prefill_batch_scratch(PrefillBatchScratch& s) {
    for (void* p : {static_cast<void*>(s.d_tables), static_cast<void*>(s.d_cu), static_cast<void*>(s.d_start),
                    static_cast<void*>(s.d_tile_map), static_cast<void*>(s.d_last), static_cast<void*>(s.d_last_norm),
                    static_cast<void*>(s.d_logits)}) {
        cudaFree(p);
    }
    s = PrefillBatchScratch{};
}

std::vector<int> model_prefill_batch(Model& m, PagedKVCache& c, const std::vector<PrefillPart>& parts,
                                     PrefillBatchScratch& s) {
    const int B = static_cast<int>(parts.size());
    if (B < 1 || B > s.max_seqs) {
        throw std::logic_error("model_prefill_batch: " + std::to_string(B) + " sequences, scratch for " +
                               std::to_string(s.max_seqs));
    }
    std::vector<int> cu{0}, start, ids, pos, tables(static_cast<size_t>(B) * static_cast<size_t>(s.max_pages), 0);
    int max_keys = 1;
    for (int b = 0; b < B; ++b) {
        const PrefillPart& p = parts[static_cast<size_t>(b)];
        const int total = static_cast<int>(p.ids->size());
        const int T = total - p.start;
        // A new prefill (length 0, a cached prefix or none) or the next chunk of one (length = start).
        if ((p.seq->len != 0 && p.seq->len != p.start) || p.start < 0 || T <= 0 ||
            static_cast<int>(p.seq->pages.size()) * c.page_size < total ||
            static_cast<int>(p.seq->pages.size()) > s.max_pages) {
            throw std::logic_error("model_prefill_batch: sequence " + std::to_string(b) + " does not fit (" +
                                   std::to_string(total) + " tokens from " + std::to_string(p.start) + ", " +
                                   std::to_string(p.seq->pages.size()) + " pages, length " +
                                   std::to_string(p.seq->len) + ")");
        }
        cu.push_back(cu.back() + T);
        start.push_back(p.start);
        max_keys = std::max(max_keys, total);
        for (int t = p.start; t < total; ++t) {
            ids.push_back((*p.ids)[static_cast<size_t>(t)]);
            pos.push_back(t);  // position ids continue each sequence's cached prefix
        }
        std::copy(p.seq->pages.begin(), p.seq->pages.end(), tables.begin() + static_cast<long>(b) * s.max_pages);
        // The sequence's own table, for kv_scatter and later the decode path.
        CUDA_CHECK(cudaMemcpy(p.seq->d_table, p.seq->pages.data(), p.seq->pages.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    const int T = cu.back();
    if (T > m.max_T) {
        throw std::logic_error("model_prefill_batch: " + std::to_string(T) + " tokens exceed the scratch (" +
                               std::to_string(m.max_T) + ")");
    }
    const std::vector<int> tile_map = neuroserve::kernels::varlen_tile_map(cu, start, 64);  // the 64 x 32 shape's rows
    if (static_cast<int>(tile_map.size() / 2) > s.max_tiles) {
        throw std::logic_error("model_prefill_batch: more tiles than the scratch holds");
    }
    CUDA_CHECK(cudaMemcpy(m.d_ids, ids.data(), sizeof(int) * static_cast<size_t>(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.d_pos, pos.data(), sizeof(int) * static_cast<size_t>(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_tables, tables.data(), sizeof(int) * tables.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_cu, cu.data(), sizeof(int) * cu.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_start, start.data(), sizeof(int) * start.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_tile_map, tile_map.data(), sizeof(int) * tile_map.size(), cudaMemcpyHostToDevice));

    neuroserve::kernels::VarlenAttentionArgs a;
    a.cu_seqlens_q = s.d_cu;
    a.start = s.d_start;
    a.block_tables = s.d_tables;
    a.num_seqs = B;
    a.total_q = T;
    a.max_keys = max_keys;
    a.max_pages = s.max_pages;
    a.n_heads = m.cfg.n_heads;
    a.n_kv_heads = m.cfg.n_kv_heads;
    a.head_dim = m.cfg.head_dim;
    a.page_size = c.page_size;
    a.scale = 1.f / std::sqrt(static_cast<float>(m.cfg.head_dim));  // = rsqrtf(head_dim), the engine's

    embedding_lookup(m.embed, m.d_ids, m.d_x, T, m.cfg.hidden);
    for (int l = 0; l < m.n_layers; l++) {
        layer_forward_batch(m.handle, m.cfg, m.layers[static_cast<size_t>(l)], m.scratch, m.d_x, m.d_pos,
                            m.d_inv_freq, T, c.k_pool[static_cast<size_t>(l)], c.v_pool[static_cast<size_t>(l)], parts,
                            cu, c.page_size, a, s.d_tile_map, static_cast<int>(tile_map.size() / 2));
    }

    // Each sequence's last row: the final norm and the head, once for all of them.
    const int H = m.cfg.hidden;
    for (int b = 0; b < B; ++b) {
        CUDA_CHECK(cudaMemcpyAsync(s.d_last + static_cast<size_t>(b) * H,
                                   m.d_x + static_cast<size_t>(cu[static_cast<size_t>(b) + 1] - 1) * H,
                                   sizeof(__half) * static_cast<size_t>(H), cudaMemcpyDeviceToDevice, 0));
    }
    rmsnorm(s.d_last, m.final_norm, s.d_last_norm, B, H, m.cfg.eps);
    linear(m.handle, s.d_last_norm, m.embed, s.d_logits, B, H, m.vocab);
    CUDA_CHECK(cudaGetLastError());
    std::vector<__half> host(static_cast<size_t>(B) * static_cast<size_t>(m.vocab));
    CUDA_CHECK(cudaMemcpy(host.data(), s.d_logits, sizeof(__half) * host.size(), cudaMemcpyDeviceToHost));
    std::vector<int> next(static_cast<size_t>(B));
    for (int b = 0; b < B; ++b) {
        const __half* row = host.data() + static_cast<size_t>(b) * m.vocab;
        int best = 0;
        float best_v = __half2float(row[0]);
        for (int v = 1; v < m.vocab; ++v) {
            const float x = __half2float(row[v]);
            if (x > best_v) {  // strictly greater: ties stay with the lowest id, like argmax_first
                best_v = x;
                best = v;
            }
        }
        next[static_cast<size_t>(b)] = best;
        parts[static_cast<size_t>(b)].seq->len = static_cast<int>(parts[static_cast<size_t>(b)].ids->size());
    }
    return next;
}

}  // namespace neuroserve::backend_detail
