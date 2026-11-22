#include "neuroserve/kernels/attention_varlen.hpp"

#include <mma.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>

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

// ---- The tiled kernel (Day 49, docs/prefill-attention.md) ----------------------------------------
//
// One block per (tile of BR new tokens of one sequence, query head); WARPS = BR / 16 warps, each
// owning 16 query rows. Per K/V tile of BC keys:
//   1. the block loads K and V (128-bit loads through the block table; zeros past the last key)
//   2. each warp: S = Q K^T on tensor cores, stored to its slice of shared memory
//   3. lanes 0..15, one row each: scale, causal mask, running max m, P = exp2(S - m) in FP16, the
//      running sum l, and the factor alpha = exp2(m_old - m_new) for the old output
//   4. each warp: P V on tensor cores into the same slice
//   5. every lane: its 32 output elements (row k/2, columns lane and lane + 32) O = O * alpha + P V
// and at the end out = O / l. Every step is in a fixed order, so results do not depend on timing,
// and every row's numbers depend only on that row's query and keys (not on the rest of the tile or
// the pack).

constexpr int kD = 64;                 // head dimension
constexpr int kLdq = kD + 8;           // padded rows (halves): 144 bytes, 16-byte aligned
constexpr int kLdo = kD + 4;           // S / P V scratch rows (floats); also holds S, BC <= 64 columns
constexpr float kLog2e = 1.4426950408889634f;

template <int BR, int BC>
struct TiledLayout {
    static constexpr int kWarps = BR / 16;
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kLdp = BC + 8;  // P rows (halves)
    static constexpr size_t kQ = size_t{BR} * kLdq * 2;
    static constexpr size_t kK = size_t{BC} * kLdq * 2;
    static constexpr size_t kV = kK;
    static constexpr size_t kS = size_t{kWarps} * 16 * kLdo * 4;
    static constexpr size_t kP = size_t{kWarps} * 16 * kLdp * 2;
    static constexpr size_t kRows = size_t{kWarps} * 3 * 16 * 4;  // m, l, alpha per row
    static constexpr size_t kBytes = kQ + kK + kV + kS + kP + kRows;
    static_assert(BR % 16 == 0 && BC % 16 == 0 && BC <= kD, "tiles are multiples of 16; BC <= 64");
    static_assert(kQ % 128 == 0 && kK % 128 == 0 && kS % 128 == 0 && kP % 128 == 0, "128-byte aligned buffers");
};

// Copies `rows` rows of 64 halves (128 bytes, 8 x 16-byte loads each) into shared memory with row
// stride kLdq; rows at or past `valid` are zeros. row_ptr(r) gives the global row.
template <int ROWS, int THREADS, typename RowPtr>
__device__ __forceinline__ void load_rows(__half* dst, int valid, RowPtr row_ptr) {
    for (int c = static_cast<int>(threadIdx.x); c < ROWS * 8; c += THREADS) {
        const int r = c / 8;
        const int part = c % 8;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (r < valid) {
            v = reinterpret_cast<const uint4*>(row_ptr(r))[part];
        }
        reinterpret_cast<uint4*>(dst + r * kLdq)[part] = v;
    }
}

template <int BR, int BC>
__global__ void __launch_bounds__(TiledLayout<BR, BC>::kThreads)
attention_prefill_varlen_tiled_kernel(VarlenAttentionArgs a, const std::int32_t* __restrict__ tile_map) {
    using L = TiledLayout<BR, BC>;
    namespace wmma = nvcuda::wmma;
    extern __shared__ __align__(128) unsigned char smem[];
    __half* const qs = reinterpret_cast<__half*>(smem);
    __half* const ks = reinterpret_cast<__half*>(smem + L::kQ);
    __half* const vs = reinterpret_cast<__half*>(smem + L::kQ + L::kK);
    float* const ss = reinterpret_cast<float*>(smem + L::kQ + L::kK + L::kV);
    __half* const ps = reinterpret_cast<__half*>(smem + L::kQ + L::kK + L::kV + L::kS);
    float* const rows_state = reinterpret_cast<float*>(smem + L::kQ + L::kK + L::kV + L::kS + L::kP);

    const int h = static_cast<int>(blockIdx.x);
    const int tile = static_cast<int>(blockIdx.y);
    const int b = tile_map[2 * tile];
    const int i0 = tile_map[2 * tile + 1];
    const int first = a.cu_seqlens_q[b];
    const int T = a.cu_seqlens_q[b + 1] - first;
    const int rows = min(BR, T - i0);
    const int p0 = a.start[b] + i0;       // position of the tile's first query
    const int n_keys = p0 + rows;         // keys 0 .. last query's position
    const int g = h / (a.n_heads / a.n_kv_heads);
    const std::int32_t* table = a.block_tables + static_cast<size_t>(b) * static_cast<size_t>(a.max_pages);
    const size_t kv_stride = static_cast<size_t>(a.n_kv_heads) * kD;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const float scale_log2 = a.scale * kLog2e;

    float* const s_w = ss + warp * 16 * kLdo;
    __half* const p_w = ps + warp * 16 * L::kLdp;
    float* const m_w = rows_state + warp * 48;
    float* const l_w = m_w + 16;
    float* const alpha_w = m_w + 32;
    if (lane < 16) {
        m_w[lane] = -INFINITY;
        l_w[lane] = 0.f;
    }

    load_rows<BR, L::kThreads>(qs, rows, [&](int r) {
        return a.q + (static_cast<size_t>(first + i0 + r) * a.n_heads + h) * kD;
    });

    float o[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) o[k] = 0.f;

    const int n_tiles = (n_keys + BC - 1) / BC;
    for (int kt = 0; kt < n_tiles; ++kt) {
        const int k0 = kt * BC;
        __syncthreads();  // the previous tile's K and V are no longer read (and Q is loaded)
        const int valid = min(BC, n_keys - k0);
        const auto key_row = [&](int r) {
            const int key = k0 + r;
            return (static_cast<size_t>(table[key / a.page_size]) * a.page_size + key % a.page_size) * kv_stride +
                   static_cast<size_t>(g) * kD;
        };
        load_rows<BC, L::kThreads>(ks, valid, [&](int r) { return a.k_pool + key_row(r); });
        load_rows<BC, L::kThreads>(vs, valid, [&](int r) { return a.v_pool + key_row(r); });
        __syncthreads();

        // 2. S = Q K^T for this warp's 16 rows.
        for (int n = 0; n < BC / 16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
#pragma unroll
            for (int kk = 0; kk < kD / 16; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, qs + warp * 16 * kLdq + kk * 16, kLdq);
                wmma::load_matrix_sync(fb, ks + n * 16 * kLdq + kk * 16, kLdq);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(s_w + n * 16, acc, kLdo, wmma::mem_row_major);
        }
        __syncwarp();

        // 3. Online softmax, one row per lane.
        if (lane < 16) {
            const int p = p0 + warp * 16 + lane;  // this row's position
            float* srow = s_w + lane * kLdo;
            float mx = -INFINITY;
            for (int c = 0; c < BC; ++c) {
                const float v = (k0 + c <= p) ? srow[c] * scale_log2 : -INFINITY;
                srow[c] = v;
                mx = fmaxf(mx, v);
            }
            const float m_old = m_w[lane];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.f;
            __half* prow = p_w + lane * L::kLdp;
            for (int c = 0; c < BC; ++c) {
                const float e = (srow[c] == -INFINITY) ? 0.f : exp2f(srow[c] - m_new);
                prow[c] = __float2half_rn(e);
                sum += e;
            }
            const float alpha = (m_old == -INFINITY) ? 0.f : exp2f(m_old - m_new);
            l_w[lane] = l_w[lane] * alpha + sum;
            m_w[lane] = m_new;
            alpha_w[lane] = alpha;
        }
        __syncwarp();

        // 4. P V for this warp's rows, into the same scratch.
#pragma unroll
        for (int n = 0; n < kD / 16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
            for (int kk = 0; kk < BC / 16; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, p_w + kk * 16, L::kLdp);
                wmma::load_matrix_sync(fb, vs + kk * 16 * kLdq + n * 16, kLdq);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(s_w + n * 16, acc, kLdo, wmma::mem_row_major);
        }
        __syncwarp();

        // 5. O = O * alpha + P V: lane owns row k / 2, column lane + 32 * (k % 2).
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            const int r = k / 2;
            o[k] = o[k] * alpha_w[r] + s_w[r * kLdo + lane + 32 * (k % 2)];
        }
        __syncwarp();
    }

#pragma unroll
    for (int k = 0; k < 32; ++k) {
        const int r = warp * 16 + k / 2;
        if (r < rows) {
            a.out[(static_cast<size_t>(first + i0 + r) * a.n_heads + h) * kD + lane + 32 * (k % 2)] =
                __float2half_rn(o[k] / l_w[k / 2]);
        }
    }
}

void require(bool ok, const std::string& what) {
    if (!ok) {
        throw std::invalid_argument("attention_prefill_varlen: " + what);
    }
}

template <int BR, int BC>
void launch_tiled(const VarlenAttentionArgs& a, const std::int32_t* tile_map, int num_tiles, cudaStream_t stream) {
    using L = TiledLayout<BR, BC>;
    static const bool configured = [] {
        int device = 0;
        int optin = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        require(static_cast<size_t>(optin) >= L::kBytes,
                "tile " + std::to_string(BR) + "x" + std::to_string(BC) + " needs " + std::to_string(L::kBytes) +
                    " bytes of shared memory; this GPU allows " + std::to_string(optin));
        return cudaFuncSetAttribute(attention_prefill_varlen_tiled_kernel<BR, BC>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(L::kBytes)) == cudaSuccess;
    }();
    require(configured, "could not set the tiled kernel's shared memory size");
    const dim3 grid(static_cast<unsigned>(a.n_heads), static_cast<unsigned>(num_tiles));
    attention_prefill_varlen_tiled_kernel<BR, BC><<<grid, L::kThreads, L::kBytes, stream>>>(a, tile_map);
}

}  // namespace

int tile_rows(TileShape shape) {
    return (shape == TileShape::k128x64 || shape == TileShape::k128x32) ? 128 : 64;
}

int tile_keys(TileShape shape) {
    return (shape == TileShape::k64x32 || shape == TileShape::k128x32) ? 32 : 64;
}

const char* tile_shape_name(TileShape shape) {
    switch (shape) {
        case TileShape::k64x64: return "64x64";
        case TileShape::k64x32: return "64x32";
        case TileShape::k128x64: return "128x64";
        case TileShape::k128x32: return "128x32";
    }
    return "?";
}

std::vector<std::int32_t> varlen_tile_map(const std::vector<std::int32_t>& cu, const std::vector<std::int32_t>& start,
                                          int rows) {
    require(rows >= 1 && !cu.empty() && cu.size() == start.size() + 1, "tile map: cu_seqlens_q has B + 1 entries");
    std::vector<std::tuple<std::int64_t, std::int32_t, std::int32_t>> tiles;  // (-end, b, i0)
    for (size_t b = 0; b + 1 < cu.size(); ++b) {
        const std::int32_t T = cu[b + 1] - cu[b];
        for (std::int32_t i0 = 0; i0 < T; i0 += rows) {
            const std::int64_t end = static_cast<std::int64_t>(start[b]) + std::min(i0 + rows, T);
            tiles.emplace_back(-end, static_cast<std::int32_t>(b), i0);
        }
    }
    std::sort(tiles.begin(), tiles.end());  // longest first; ties by sequence, then tile
    std::vector<std::int32_t> flat;
    flat.reserve(tiles.size() * 2);
    for (const auto& [neg_end, b, i0] : tiles) {
        flat.push_back(b);
        flat.push_back(i0);
    }
    return flat;
}

void attention_prefill_varlen_tiled(const VarlenAttentionArgs& a, const std::int32_t* tile_map, int num_tiles,
                                    TileShape shape, cudaStream_t stream) {
    require(a.q && a.k_pool && a.v_pool && a.cu_seqlens_q && a.start && a.block_tables && a.out && tile_map,
            "null pointer");
    require(a.num_seqs >= 1 && num_tiles >= 1 && num_tiles <= 65535, "needs 1 to 65,535 tiles");
    require(a.n_heads >= 1 && a.n_kv_heads >= 1 && a.n_heads % a.n_kv_heads == 0,
            "n_heads must be a positive multiple of n_kv_heads");
    require(a.head_dim == kD, "the tiled kernel is built for head_dim 64");
    require(a.page_size >= 1 && a.max_pages >= 1, "page_size and max_pages must be >= 1");
    switch (shape) {
        case TileShape::k64x64: launch_tiled<64, 64>(a, tile_map, num_tiles, stream); break;
        case TileShape::k64x32: launch_tiled<64, 32>(a, tile_map, num_tiles, stream); break;
        case TileShape::k128x64: launch_tiled<128, 64>(a, tile_map, num_tiles, stream); break;
        case TileShape::k128x32: launch_tiled<128, 32>(a, tile_map, num_tiles, stream); break;
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("attention_prefill_varlen_tiled launch: ") + cudaGetErrorString(err));
    }
}

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
