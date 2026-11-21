#include "prefill_from_offset.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>

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

}  // namespace neuroserve::backend_detail
