// Variable-length prefill attention v0 (docs/prefill-attention.md), against a per-sequence FP32
// reference computed on the host from the same FP16 inputs. GPU, no model.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "neuroserve/kernels/attention_varlen.hpp"

namespace {

using neuroserve::kernels::TileShape;
using neuroserve::kernels::VarlenAttentionArgs;
using neuroserve::kernels::attention_prefill_varlen_tiled;
using neuroserve::kernels::attention_prefill_varlen_v0;
using neuroserve::kernels::tile_rows;
using neuroserve::kernels::varlen_tile_map;

// Which kernel a problem runs: v0, or the tiled kernel with a tile shape.
struct Kernel {
    bool tiled = false;
    TileShape shape = TileShape::k64x64;
};
constexpr Kernel kV0{};
constexpr Kernel kTiled{true, TileShape::k64x64};
const std::vector<TileShape> kShapes = {TileShape::k64x64, TileShape::k64x32, TileShape::k128x64, TileShape::k128x32};


constexpr int kPage = 16;
constexpr float kAtol = 2e-3f;  // |out - ref| <= kAtol + kRtol * |ref| (docs/prefill-attention.md)
constexpr float kRtol = 1e-2f;

struct Seq {
    int start;  // cached keys
    int T;      // new tokens
};

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(const std::vector<T>& host) {
        if (cudaMalloc(&ptr, host.size() * sizeof(T)) != cudaSuccess) throw std::runtime_error("cudaMalloc");
        if (cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("cudaMemcpy");
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { cudaFree(ptr); }
};

// One test case: sequences, their pages (shuffled, from a pool whose unused rows are NaN), Q, K, V.
struct Problem {
    int n_heads = 32;
    int n_kv_heads = 8;
    int head_dim = 64;
    std::vector<Seq> seqs;
    int max_pages = 0;
    int num_pages = 0;
    std::vector<std::int32_t> cu, start, tables;
    std::vector<__half> q, k_pool, v_pool;

    Problem(std::vector<Seq> s, std::uint32_t seed, float q_scale = 1.f, int heads = 32, int kv_heads = 8)
        : n_heads(heads), n_kv_heads(kv_heads), seqs(std::move(s)) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        cu.push_back(0);
        for (const auto& x : seqs) {
            cu.push_back(cu.back() + x.T);
            start.push_back(x.start);
            max_pages = std::max(max_pages, (x.start + x.T + kPage - 1) / kPage);
        }
        int used = 0;
        for (const auto& x : seqs) used += (x.start + x.T + kPage - 1) / kPage;
        num_pages = used + 7;  // spare pages stay NaN
        std::vector<std::int32_t> ids(static_cast<size_t>(num_pages));
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        tables.assign(seqs.size() * static_cast<size_t>(max_pages), -1);
        size_t next = 0;
        const size_t row = static_cast<size_t>(n_kv_heads) * head_dim;
        k_pool.assign(static_cast<size_t>(num_pages) * kPage * row, __float2half(NAN));
        v_pool = k_pool;
        for (size_t b = 0; b < seqs.size(); ++b) {
            const int keys = seqs[b].start + seqs[b].T;
            for (int p = 0; p < (keys + kPage - 1) / kPage; ++p) tables[b * max_pages + p] = ids[next++];
            for (int j = 0; j < keys; ++j) {  // only real positions get values: the rest of a page stays NaN
                const size_t r = static_cast<size_t>(tables[b * max_pages + j / kPage]) * kPage + j % kPage;
                for (size_t e = 0; e < row; ++e) {
                    k_pool[r * row + e] = __float2half(uni(rng));
                    v_pool[r * row + e] = __float2half(uni(rng));
                }
            }
        }
        q.resize(static_cast<size_t>(cu.back()) * n_heads * head_dim);
        for (auto& x : q) x = __float2half(q_scale * uni(rng));
    }

    int total_q() const { return cu.back(); }
    int max_keys() const {
        int m = 1;
        for (const auto& x : seqs) m = std::max(m, x.start + x.T);
        return m;
    }

    // Runs a kernel; returns out as floats.
    std::vector<float> run(Kernel kernel = kV0) const {
        DeviceBuffer<__half> dq(q), dk(k_pool), dv(v_pool);
        DeviceBuffer<std::int32_t> dcu(cu), dstart(start), dtables(tables);
        std::vector<__half> host_out(q.size(), __float2half(NAN));
        DeviceBuffer<__half> dout(host_out);
        VarlenAttentionArgs a;
        a.q = dq.ptr;
        a.k_pool = dk.ptr;
        a.v_pool = dv.ptr;
        a.cu_seqlens_q = dcu.ptr;
        a.start = dstart.ptr;
        a.block_tables = dtables.ptr;
        a.out = dout.ptr;
        a.num_seqs = static_cast<int>(seqs.size());
        a.total_q = total_q();
        a.max_keys = max_keys();
        a.max_pages = max_pages;
        a.n_heads = n_heads;
        a.n_kv_heads = n_kv_heads;
        a.head_dim = head_dim;
        a.page_size = kPage;
        a.scale = 1.f / std::sqrt(static_cast<float>(head_dim));
        if (kernel.tiled) {
            const std::vector<std::int32_t> map = varlen_tile_map(cu, start, tile_rows(kernel.shape));
            DeviceBuffer<std::int32_t> dmap(map);
            attention_prefill_varlen_tiled(a, dmap.ptr, static_cast<int>(map.size() / 2), kernel.shape);
            if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("tiled kernel failed");
        } else {
            attention_prefill_varlen_v0(a);
        }
        if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("kernel failed");
        if (cudaMemcpy(host_out.data(), dout.ptr, host_out.size() * sizeof(__half), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
            throw std::runtime_error("cudaMemcpy out");
        std::vector<float> out(host_out.size());
        for (size_t i = 0; i < out.size(); ++i) out[i] = __half2float(host_out[i]);
        return out;
    }

    // The reference: sequence b alone, FP32 (double sums), keys read through its own block table.
    void reference(size_t b, int i, int h, float* o) const {
        const int g = h / (n_heads / n_kv_heads);
        const int keys = seqs[b].start + i + 1;
        const size_t row = static_cast<size_t>(n_kv_heads) * head_dim;
        const __half* qv = &q[(static_cast<size_t>(cu[b] + i) * n_heads + h) * head_dim];
        const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
        std::vector<double> s(static_cast<size_t>(keys));
        double mx = -INFINITY;
        for (int j = 0; j < keys; ++j) {
            const size_t r = static_cast<size_t>(tables[b * max_pages + j / kPage]) * kPage + j % kPage;
            double dot = 0;
            for (int d = 0; d < head_dim; ++d)
                dot += static_cast<double>(__half2float(qv[d])) * __half2float(k_pool[r * row + g * head_dim + d]);
            s[j] = dot * scale;
            mx = std::max(mx, s[j]);
        }
        double sum = 0;
        for (auto& x : s) sum += (x = std::exp(x - mx));
        for (int d = 0; d < head_dim; ++d) {
            double acc = 0;
            for (int j = 0; j < keys; ++j) {
                const size_t r = static_cast<size_t>(tables[b * max_pages + j / kPage]) * kPage + j % kPage;
                acc += s[j] * __half2float(v_pool[r * row + g * head_dim + d]);
            }
            o[d] = static_cast<float>(acc / sum);
        }
    }
};

// Compares every token of every sequence; on long problems a sample of heads (all groups).
float expect_matches_reference(const Problem& p, const std::vector<float>& out) {
    const long long work = static_cast<long long>(p.total_q()) * p.max_keys();
    std::vector<int> heads;
    if (work <= 200000) {
        for (int h = 0; h < p.n_heads; ++h) heads.push_back(h);
    } else {
        heads = {0, p.n_heads / 4 + 1, p.n_heads / 2 + 2, p.n_heads - 1};
    }
    float worst = 0.f;
    std::vector<float> ref(static_cast<size_t>(p.head_dim));
    for (size_t b = 0; b < p.seqs.size(); ++b) {
        for (int i = 0; i < p.seqs[b].T; ++i) {
            for (const int h : heads) {
                p.reference(b, i, h, ref.data());
                const float* got = &out[(static_cast<size_t>(p.cu[b] + i) * p.n_heads + h) * p.head_dim];
                for (int d = 0; d < p.head_dim; ++d) {
                    const float err = std::fabs(got[d] - ref[static_cast<size_t>(d)]);
                    worst = std::max(worst, err);
                    if (!(err <= kAtol + kRtol * std::fabs(ref[static_cast<size_t>(d)]))) {
                        ADD_FAILURE() << "seq " << b << " (start " << p.seqs[b].start << ", T " << p.seqs[b].T
                                      << ") token " << i << " head " << h << " dim " << d << ": got " << got[d]
                                      << ", reference " << ref[static_cast<size_t>(d)];
                        return worst;
                    }
                }
            }
        }
    }
    return worst;
}

// Output rows of sequence b, as raw bits.
std::vector<std::uint16_t> rows_of(const Problem& p, size_t b, const std::vector<float>& out) {
    std::vector<std::uint16_t> bits;
    const size_t from = static_cast<size_t>(p.cu[b]) * p.n_heads * p.head_dim;
    const size_t to = static_cast<size_t>(p.cu[b + 1]) * p.n_heads * p.head_dim;
    for (size_t e = from; e < to; ++e) {
        const __half hv = __float2half(out[e]);
        std::uint16_t u;
        std::memcpy(&u, &hv, sizeof(u));
        bits.push_back(u);
    }
    return bits;
}

const std::vector<int> kLengths = {1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 129, 480, 1000, 2047, 2048};

TEST(AttentionVarlenV0, MatchesTheReferenceForEverySingleLengthFrom1To2048) {
    for (const int T : kLengths) {
        SCOPED_TRACE("T = " + std::to_string(T));
        const Problem p({{0, T}}, static_cast<std::uint32_t>(T));
        const float worst = expect_matches_reference(p, p.run());
        RecordProperty("max_err_T" + std::to_string(T), std::to_string(worst));
    }
}

TEST(AttentionVarlenV0, MatchesTheReferenceForAPackOfMixedLengthsWithCachedPrefixes) {
    // Starts of 0, whole pages and partial pages; every sequence at most 2,048 keys.
    const Problem p({{0, 1}, {16, 2}, {17, 15}, {100, 16}, {0, 17}, {33, 63}, {64, 64}, {0, 65}, {1, 127},
                     {48, 128}, {0, 129}, {256, 480}, {1048, 1000}, {0, 2048}, {1, 2047}},
                    7);
    expect_matches_reference(p, p.run());
}

TEST(AttentionVarlenV0, MatchesTheReferenceForRandomPacks) {
    std::mt19937 rng(42);
    for (int trial = 0; trial < 6; ++trial) {
        const int n = 1 + static_cast<int>(rng() % 16);
        std::vector<Seq> seqs;
        for (int s = 0; s < n; ++s) {
            const int T = 1 + static_cast<int>(rng() % 256);
            const int start = (rng() % 3 == 0) ? 0 : static_cast<int>(rng() % 1024u);
            seqs.push_back({start, T});
        }
        SCOPED_TRACE("trial " + std::to_string(trial));
        const Problem p(seqs, 1000u + static_cast<std::uint32_t>(trial));
        expect_matches_reference(p, p.run());
    }
}

TEST(AttentionVarlenV0, MatchesTheReferenceWithASharpSoftmax) {
    // Queries 8x larger: scores spread over tens of units, most weight on a few keys.
    const Problem p({{0, 64}, {200, 100}, {0, 513}}, 9, 8.f);
    expect_matches_reference(p, p.run());
}

TEST(AttentionVarlenV0, MatchesTheReferenceWithoutGroupedQueries) {
    const Problem p({{0, 40}, {30, 70}}, 11, 1.f, /*heads=*/8, /*kv_heads=*/8);
    expect_matches_reference(p, p.run());
}

void expect_independent_of_the_pack(Kernel kernel) {
    const std::vector<Seq> seqs = {{0, 37}, {16, 64}, {5, 129}, {0, 1}, {300, 200}};
    // The pack, and each sequence alone, from the same seed-derived data: build the pack once and
    // derive single-sequence problems that share its pages, Q, K and V.
    const Problem packed(seqs, 21);
    const std::vector<float> out = packed.run(kernel);
    for (size_t b = 0; b < seqs.size(); ++b) {
        Problem alone = packed;
        alone.seqs = {seqs[b]};
        alone.cu = {0, seqs[b].T};
        alone.start = {seqs[b].start};
        alone.tables.assign(packed.tables.begin() + static_cast<long>(b * packed.max_pages),
                            packed.tables.begin() + static_cast<long>((b + 1) * packed.max_pages));
        alone.q.assign(packed.q.begin() + static_cast<long>(static_cast<size_t>(packed.cu[b]) * 32 * 64),
                       packed.q.begin() + static_cast<long>(static_cast<size_t>(packed.cu[b + 1]) * 32 * 64));
        EXPECT_EQ(rows_of(alone, 0, alone.run(kernel)), rows_of(packed, b, out)) << "sequence " << b;
    }
    // Reversed order: same bits per sequence.
    Problem reversed = packed;
    reversed.seqs.assign(seqs.rbegin(), seqs.rend());
    reversed.cu = {0};
    reversed.start.clear();
    reversed.tables.clear();
    reversed.q.clear();
    for (size_t r = 0; r < seqs.size(); ++r) {
        const size_t b = seqs.size() - 1 - r;
        reversed.cu.push_back(reversed.cu.back() + seqs[b].T);
        reversed.start.push_back(seqs[b].start);
        reversed.tables.insert(reversed.tables.end(), packed.tables.begin() + static_cast<long>(b * packed.max_pages),
                               packed.tables.begin() + static_cast<long>((b + 1) * packed.max_pages));
        reversed.q.insert(reversed.q.end(), packed.q.begin() + static_cast<long>(static_cast<size_t>(packed.cu[b]) * 32 * 64),
                          packed.q.begin() + static_cast<long>(static_cast<size_t>(packed.cu[b + 1]) * 32 * 64));
    }
    const std::vector<float> rev_out = reversed.run(kernel);
    for (size_t r = 0; r < seqs.size(); ++r) {
        EXPECT_EQ(rows_of(reversed, r, rev_out), rows_of(packed, seqs.size() - 1 - r, out)) << "reversed position " << r;
    }
}

TEST(AttentionVarlenV0, ASequencesOutputDoesNotDependOnItsPack) { expect_independent_of_the_pack(kV0); }

TEST(AttentionVarlenV0, IsDeterministic) {
    const Problem p({{0, 300}, {77, 50}}, 5);
    const std::vector<float> a = p.run();
    const std::vector<float> b = p.run();
    EXPECT_EQ(rows_of(p, 0, a), rows_of(p, 0, b));
    EXPECT_EQ(rows_of(p, 1, a), rows_of(p, 1, b));
}

TEST(AttentionVarlenV0, RefusesWhatItCannotServe) {
    VarlenAttentionArgs a;
    EXPECT_THROW(attention_prefill_varlen_v0(a), std::invalid_argument);  // null pointers
    const Problem p({{0, 8}}, 3);
    DeviceBuffer<__half> dq(p.q), dk(p.k_pool), dv(p.v_pool);
    DeviceBuffer<std::int32_t> dcu(p.cu), dstart(p.start), dtables(p.tables);
    a.q = dq.ptr;
    a.k_pool = dk.ptr;
    a.v_pool = dv.ptr;
    a.cu_seqlens_q = dcu.ptr;
    a.start = dstart.ptr;
    a.block_tables = dtables.ptr;
    a.out = dq.ptr;
    a.num_seqs = 1;
    a.total_q = 8;
    a.max_keys = 8;
    a.max_pages = 1;
    a.n_heads = 32;
    a.n_kv_heads = 6;  // 32 is not a multiple of 6
    a.head_dim = 64;
    a.page_size = kPage;
    a.scale = 0.125f;
    EXPECT_THROW(attention_prefill_varlen_v0(a), std::invalid_argument);
    a.n_kv_heads = 8;
    a.max_keys = 17;  // more keys than one page holds
    EXPECT_THROW(attention_prefill_varlen_v0(a), std::invalid_argument);
    a.max_pages = 800;
    a.max_keys = 12289;  // 49,156 bytes of scores
    EXPECT_THROW(attention_prefill_varlen_v0(a), std::invalid_argument);
}

// ---- The tiled kernel (Day 49) ---------------------------------------------------------------------

// Elementwise: |a - b| <= kAtol + kRtol * |b|.
void expect_close(const Problem& p, const std::vector<float>& got, const std::vector<float>& want, const char* what) {
    float worst = 0.f;
    for (size_t e = 0; e < got.size(); ++e) {
        const float err = std::fabs(got[e] - want[e]);
        worst = std::max(worst, err);
        if (!(err <= kAtol + kRtol * std::fabs(want[e]))) {
            const size_t token = e / (static_cast<size_t>(p.n_heads) * p.head_dim);
            ADD_FAILURE() << what << ": token " << token << " element " << e << ": got " << got[e] << ", want " << want[e];
            return;
        }
    }
    ::testing::Test::RecordProperty(std::string("max_err_") + what, std::to_string(worst));
}

TEST(AttentionVarlenTiled, MatchesTheReferenceForEverySingleLengthFrom1To2048) {
    for (const int T : kLengths) {
        SCOPED_TRACE("T = " + std::to_string(T));
        const Problem p({{0, T}}, static_cast<std::uint32_t>(T));
        expect_matches_reference(p, p.run(kTiled));
    }
}

TEST(AttentionVarlenTiled, MatchesTheReferenceForAPackOfMixedLengthsWithCachedPrefixes) {
    const Problem p({{0, 1}, {16, 2}, {17, 15}, {100, 16}, {0, 17}, {33, 63}, {64, 64}, {0, 65}, {1, 127},
                     {48, 128}, {0, 129}, {256, 480}, {1048, 1000}, {0, 2048}, {1, 2047}},
                    7);
    expect_matches_reference(p, p.run(kTiled));
}

TEST(AttentionVarlenTiled, MatchesTheReferenceForRandomPacks) {
    std::mt19937 rng(4242);
    for (int trial = 0; trial < 4; ++trial) {
        const int n = 1 + static_cast<int>(rng() % 16);
        std::vector<Seq> seqs;
        for (int s = 0; s < n; ++s) {
            const int T = 1 + static_cast<int>(rng() % 256);
            const int start = (rng() % 3 == 0) ? 0 : static_cast<int>(rng() % 1024u);
            seqs.push_back({start, T});
        }
        SCOPED_TRACE("trial " + std::to_string(trial));
        const Problem p(seqs, 2000u + static_cast<std::uint32_t>(trial));
        expect_matches_reference(p, p.run(kTiled));
    }
}

TEST(AttentionVarlenTiled, MatchesTheReferenceWithASharpSoftmaxAndWithoutGroupedQueries) {
    const Problem sharp({{0, 64}, {200, 100}, {0, 513}}, 9, 8.f);
    expect_matches_reference(sharp, sharp.run(kTiled));
    const Problem mha({{0, 40}, {30, 70}}, 11, 1.f, /*heads=*/8, /*kv_heads=*/8);
    expect_matches_reference(mha, mha.run(kTiled));
}

TEST(AttentionVarlenTiled, MatchesV0WithinToleranceForEveryTileShape) {
    // Every tile edge: lengths around 16, 32, 64 and 128, cached starts on and off page and tile
    // boundaries, and the longest context.
    const Problem p({{0, 1}, {0, 31}, {0, 32}, {0, 33}, {5, 64}, {64, 65}, {0, 127}, {100, 128}, {31, 129},
                     {0, 480}, {700, 300}, {0, 2048}},
                    13);
    const std::vector<float> v0 = p.run(kV0);
    for (const TileShape shape : kShapes) {
        SCOPED_TRACE(neuroserve::kernels::tile_shape_name(shape));
        expect_close(p, p.run({true, shape}), v0, neuroserve::kernels::tile_shape_name(shape));
    }
}

TEST(AttentionVarlenTiled, ASequencesOutputDoesNotDependOnItsPack) {
    for (const TileShape shape : kShapes) {
        SCOPED_TRACE(neuroserve::kernels::tile_shape_name(shape));
        expect_independent_of_the_pack({true, shape});
    }
}

TEST(AttentionVarlenTiled, IsDeterministic) {
    const Problem p({{0, 300}, {77, 50}, {0, 1}}, 5);
    for (const TileShape shape : kShapes) {
        const std::vector<float> a = p.run({true, shape});
        const std::vector<float> b = p.run({true, shape});
        for (size_t s = 0; s < 3; ++s) EXPECT_EQ(rows_of(p, s, a), rows_of(p, s, b)) << s;
    }
}

TEST(AttentionVarlenTiled, TheTileMapCoversEveryTokenOnceLongestFirst) {
    const std::vector<std::int32_t> cu = {0, 1, 65, 200, 330};
    const std::vector<std::int32_t> start = {0, 1000, 0, 64};
    for (const int rows : {64, 128}) {
        const std::vector<std::int32_t> map = varlen_tile_map(cu, start, rows);
        std::vector<int> covered(330, 0);
        long long last_end = 1LL << 40;
        for (size_t t = 0; t < map.size(); t += 2) {
            const int b = map[t];
            const int i0 = map[t + 1];
            const int T = cu[static_cast<size_t>(b) + 1] - cu[static_cast<size_t>(b)];
            ASSERT_EQ(i0 % rows, 0);
            for (int i = i0; i < std::min(i0 + rows, T); ++i) ++covered[static_cast<size_t>(cu[static_cast<size_t>(b)] + i)];
            const long long end = start[static_cast<size_t>(b)] + std::min(i0 + rows, T);
            EXPECT_LE(end, last_end);
            last_end = end;
        }
        for (int c : covered) EXPECT_EQ(c, 1);
    }
    EXPECT_EQ(varlen_tile_map(cu, start, 64), varlen_tile_map(cu, start, 64));
}

TEST(AttentionVarlenTiled, RefusesWhatItCannotServe) {
    const Problem p({{0, 8}}, 3, 1.f, /*heads=*/4, /*kv_heads=*/4);
    DeviceBuffer<__half> dq(p.q), dk(p.k_pool), dv(p.v_pool);
    DeviceBuffer<std::int32_t> dcu(p.cu), dstart(p.start), dtables(p.tables);
    const std::vector<std::int32_t> map = varlen_tile_map(p.cu, p.start, 64);
    DeviceBuffer<std::int32_t> dmap(map);
    VarlenAttentionArgs a;
    a.q = dq.ptr;
    a.k_pool = dk.ptr;
    a.v_pool = dv.ptr;
    a.cu_seqlens_q = dcu.ptr;
    a.start = dstart.ptr;
    a.block_tables = dtables.ptr;
    a.out = dq.ptr;
    a.num_seqs = 1;
    a.total_q = 8;
    a.max_keys = 8;
    a.max_pages = 1;
    a.n_heads = 4;
    a.n_kv_heads = 4;
    a.head_dim = 128;  // only 64 is built
    a.page_size = kPage;
    a.scale = 0.125f;
    EXPECT_THROW(attention_prefill_varlen_tiled(a, dmap.ptr, 1), std::invalid_argument);
    a.head_dim = 64;
    EXPECT_THROW(attention_prefill_varlen_tiled(a, nullptr, 1), std::invalid_argument);
    EXPECT_THROW(attention_prefill_varlen_tiled(a, dmap.ptr, 0), std::invalid_argument);
    EXPECT_THROW(varlen_tile_map({0, 8}, {}, 64), std::invalid_argument);
}

}  // namespace
