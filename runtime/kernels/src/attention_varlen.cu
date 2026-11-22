#include "neuroserve/kernels/attention_varlen.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace neuroserve::kernels {
namespace {

constexpr int kThreads = 128;
constexpr size_t kMaxSharedBytes = 48 * 1024;  // without opting in to more

// Row of the pool holding position j of a sequence: page table[j / page_size], slot j % page_size.
__device__ __forceinline__ size_t pool_row(const std::int32_t* __restrict__ table, int j, int page_size) {
    return static_cast<size_t>(table[j / page_size]) * static_cast<size_t>(page_size) +
           static_cast<size_t>(j % page_size);
}

// The sequence a packed token belongs to: the last b with cu_seqlens_q[b] <= token.
__device__ __forceinline__ int sequence_of(const std::int32_t* __restrict__ cu, int num_seqs, int token) {
    int lo = 0;
    int hi = num_seqs - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (cu[mid] <= token) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// One block per (query head, packed token). Every step in a fixed order (scores in parallel, then
// one thread takes the max and the sum in key order, then each output element sums over keys in
// order), so the result depends only on this query's inputs.
__global__ void attention_prefill_varlen_v0_kernel(VarlenAttentionArgs a) {
    extern __shared__ float scores[];  // one per key of this query
    __shared__ float s_max;
    __shared__ float s_sum;

    const int h = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int b = sequence_of(a.cu_seqlens_q, a.num_seqs, token);
    const int i = token - a.cu_seqlens_q[b];  // index among this sequence's new tokens
    const int keys = a.start[b] + i + 1;      // causal: positions 0 .. start + i
    const int g = h / (a.n_heads / a.n_kv_heads);
    const std::int32_t* table = a.block_tables + static_cast<size_t>(b) * static_cast<size_t>(a.max_pages);
    const size_t kv_stride = static_cast<size_t>(a.n_kv_heads) * static_cast<size_t>(a.head_dim);
    const __half* qv = a.q + (static_cast<size_t>(token) * a.n_heads + h) * a.head_dim;

    for (int j = static_cast<int>(threadIdx.x); j < keys; j += kThreads) {
        const __half* kv = a.k_pool + pool_row(table, j, a.page_size) * kv_stride + static_cast<size_t>(g) * a.head_dim;
        float dot = 0.f;
        for (int d = 0; d < a.head_dim; ++d) {
            dot += __half2float(qv[d]) * __half2float(kv[d]);
        }
        scores[j] = dot * a.scale;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int j = 0; j < keys; ++j) {
            mx = fmaxf(mx, scores[j]);
        }
        s_max = mx;
    }
    __syncthreads();
    for (int j = static_cast<int>(threadIdx.x); j < keys; j += kThreads) {
        scores[j] = expf(scores[j] - s_max);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int j = 0; j < keys; ++j) {
            sum += scores[j];
        }
        s_sum = sum;
    }
    __syncthreads();

    const float inv = 1.f / s_sum;
    __half* o = a.out + (static_cast<size_t>(token) * a.n_heads + h) * a.head_dim;
    for (int d = static_cast<int>(threadIdx.x); d < a.head_dim; d += kThreads) {
        float acc = 0.f;
        for (int j = 0; j < keys; ++j) {
            const __half* vv = a.v_pool + pool_row(table, j, a.page_size) * kv_stride + static_cast<size_t>(g) * a.head_dim;
            acc += scores[j] * __half2float(vv[d]);
        }
        o[d] = __float2half_rn(acc * inv);
    }
}

void require(bool ok, const std::string& what) {
    if (!ok) {
        throw std::invalid_argument("attention_prefill_varlen: " + what);
    }
}

}  // namespace

void attention_prefill_varlen_v0(const VarlenAttentionArgs& a, cudaStream_t stream) {
    require(a.q && a.k_pool && a.v_pool && a.cu_seqlens_q && a.start && a.block_tables && a.out, "null pointer");
    require(a.num_seqs >= 1 && a.total_q >= 1, "needs at least one sequence and one token");
    require(a.total_q <= 65535, "v0 takes at most 65,535 packed tokens (one grid row each)");
    require(a.n_heads >= 1 && a.n_kv_heads >= 1 && a.n_heads % a.n_kv_heads == 0,
            "n_heads must be a positive multiple of n_kv_heads");
    require(a.head_dim >= 1 && a.page_size >= 1 && a.max_pages >= 1, "head_dim, page_size and max_pages must be >= 1");
    require(a.max_keys >= 1 && a.max_keys <= a.max_pages * a.page_size, "max_keys must fit in max_pages pages");
    const size_t shmem = static_cast<size_t>(a.max_keys) * sizeof(float);
    require(shmem <= kMaxSharedBytes, std::to_string(a.max_keys) + " keys need " + std::to_string(shmem) +
                                          " bytes of shared memory (v0's limit is 48 KB: 12,288 keys)");
    const dim3 grid(static_cast<unsigned>(a.n_heads), static_cast<unsigned>(a.total_q));
    attention_prefill_varlen_v0_kernel<<<grid, kThreads, shmem, stream>>>(a);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("attention_prefill_varlen_v0 launch: ") + cudaGetErrorString(err));
    }
}

}  // namespace neuroserve::kernels
