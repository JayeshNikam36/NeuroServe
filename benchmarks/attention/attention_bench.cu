// Prefill attention, timed per layer with CUDA events (docs/prefill-attention.md, Days 48-49).
//
// Compares the kernel the engine uses today for prefill from a cached offset (Day 34's
// attention_prefill_paged, one launch per sequence) with the varlen v0 kernel (one launch for the
// whole pack) and the tiled kernel at each tile shape (Day 49), on Llama 3.2 1B's shapes: 32 query heads, 8 KV heads, head dimension 64, 16-token
// pages. Each case is timed as the median of 20 runs after 3 warm-up runs, and set against the
// whole prefill step Day 46 measured (0.28 ms per token) to show attention's share of it.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "neuroserve/kernels/attention_varlen.hpp"
#include "prefill_from_offset.cuh"

namespace {

constexpr int kHeads = 32, kKvHeads = 8, kDim = 64, kPage = 16, kLayers = 16;
constexpr int kWarmup = 3, kRuns = 20;
constexpr double kPrefillMsPerToken = 0.28;  // Day 46 baseline, the whole prefill step

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Case {
    std::string name;
    std::vector<std::pair<int, int>> seqs;  // (start, T)
};

struct Result {
    std::string name;
    int tokens = 0, max_keys = 0;
    double today_ms = 0, v0_ms = 0;
    std::vector<double> tiled_ms;  // per kShapes
};

const neuroserve::kernels::TileShape kShapes[] = {
    neuroserve::kernels::TileShape::k64x64, neuroserve::kernels::TileShape::k64x32,
    neuroserve::kernels::TileShape::k128x64, neuroserve::kernels::TileShape::k128x32};

template <typename F>
double median_ms(F&& launch) {
    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event");
    check(cudaEventCreate(&b), "event");
    for (int i = 0; i < kWarmup; ++i) launch();
    std::vector<float> ms;
    for (int i = 0; i < kRuns; ++i) {
        check(cudaEventRecord(a), "record");
        launch();
        check(cudaEventRecord(b), "record");
        check(cudaEventSynchronize(b), "sync");
        float t = 0;
        check(cudaEventElapsedTime(&t, a, b), "elapsed");
        ms.push_back(t);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(ms.begin(), ms.end());
    return (ms[kRuns / 2 - 1] + ms[kRuns / 2]) / 2.0;
}

template <typename T>
T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    check(cudaMalloc(&p, v.size() * sizeof(T)), "cudaMalloc");
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    return p;
}

Result run_case(const Case& c, std::mt19937& rng) {
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<int> cu{0}, start;
    int max_keys = 1, max_pages = 1, pages = 0;
    for (auto [s, t] : c.seqs) {
        cu.push_back(cu.back() + t);
        start.push_back(s);
        max_keys = std::max(max_keys, s + t);
        max_pages = std::max(max_pages, (s + t + kPage - 1) / kPage);
        pages += (s + t + kPage - 1) / kPage;
    }
    std::vector<int> ids(static_cast<size_t>(pages));
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<int> tables(c.seqs.size() * static_cast<size_t>(max_pages), 0);
    for (size_t b = 0, next = 0; b < c.seqs.size(); ++b)
        for (int p = 0; p < (c.seqs[b].first + c.seqs[b].second + kPage - 1) / kPage; ++p)
            tables[b * max_pages + p] = ids[next++];
    const size_t row = static_cast<size_t>(kKvHeads) * kDim;
    std::vector<__half> pool(static_cast<size_t>(pages) * kPage * row), q(static_cast<size_t>(cu.back()) * kHeads * kDim);
    for (auto& x : pool) x = __float2half(uni(rng));
    for (auto& x : q) x = __float2half(uni(rng));
    __half* dk = upload(pool);
    __half* dv = upload(pool);
    __half* dq = upload(q);
    __half* dout = upload(q);
    int* dcu = upload(cu);
    int* dstart = upload(start);
    int* dtables = upload(tables);

    neuroserve::kernels::VarlenAttentionArgs a;
    a.q = dq;
    a.k_pool = dk;
    a.v_pool = dv;
    a.cu_seqlens_q = dcu;
    a.start = dstart;
    a.block_tables = dtables;
    a.out = dout;
    a.num_seqs = static_cast<int>(c.seqs.size());
    a.total_q = cu.back();
    a.max_keys = max_keys;
    a.max_pages = max_pages;
    a.n_heads = kHeads;
    a.n_kv_heads = kKvHeads;
    a.head_dim = kDim;
    a.page_size = kPage;
    a.scale = 1.f / std::sqrt(static_cast<float>(kDim));

    Result r;
    r.name = c.name;
    r.tokens = cu.back();
    r.max_keys = max_keys;
    r.v0_ms = median_ms([&] { neuroserve::kernels::attention_prefill_varlen_v0(a); });
    r.today_ms = median_ms([&] {  // one launch per sequence, as the backend does
        for (size_t b = 0; b < c.seqs.size(); ++b) {
            const size_t off = static_cast<size_t>(cu[b]) * kHeads * kDim;
            neuroserve::backend_detail::attention_prefill_paged(dq + off, dk, dv, dtables + b * max_pages, dout + off,
                                                                c.seqs[b].first, c.seqs[b].second, kPage, kHeads,
                                                                kKvHeads, kDim);
        }
    });
    for (const auto shape : kShapes) {
        const std::vector<std::int32_t> map =
            neuroserve::kernels::varlen_tile_map(std::vector<std::int32_t>(cu.begin(), cu.end()),
                                                 std::vector<std::int32_t>(start.begin(), start.end()),
                                                 neuroserve::kernels::tile_rows(shape));
        std::int32_t* dmap = upload(map);
        const int tiles = static_cast<int>(map.size() / 2);
        r.tiled_ms.push_back(
            median_ms([&] { neuroserve::kernels::attention_prefill_varlen_tiled(a, dmap, tiles, shape); }));
        cudaFree(dmap);
    }
    check(cudaGetLastError(), "kernels");
    for (void* p : {static_cast<void*>(dk), static_cast<void*>(dv), static_cast<void*>(dq), static_cast<void*>(dout),
                    static_cast<void*>(dcu), static_cast<void*>(dstart), static_cast<void*>(dtables)})
        cudaFree(p);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    const char* out_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--out" && i + 1 < argc) out_path = argv[++i];
    }
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "device");
    const std::vector<Case> cases = {
        {"1 x 64", {{0, 64}}},       {"1 x 128", {{0, 128}}},    {"1 x 256", {{0, 256}}},
        {"1 x 480", {{0, 480}}},     {"1 x 1024", {{0, 1024}}},  {"1 x 2048", {{0, 2048}}},
        {"4 x 480", {{0, 480}, {0, 480}, {0, 480}, {0, 480}}},
        {"cached 1024 + 256", {{1024, 256}}},
        {"8 x 32 (short prompts)", {{0, 32}, {0, 32}, {0, 32}, {0, 32}, {0, 32}, {0, 32}, {0, 32}, {0, 32}}},
    };
    std::mt19937 rng(48);
    std::vector<Result> results;
    for (const auto& c : cases) {
        results.push_back(run_case(c, rng));
        const auto& r = results.back();
        std::printf("%-24s tokens=%5d keys<=%5d  today %8.3f  v0 %8.3f  tiled 64x64 %7.3f 64x32 %7.3f 128x64 %7.3f "
                    "128x32 %7.3f ms per layer\n",
                    r.name.c_str(), r.tokens, r.max_keys, r.today_ms, r.v0_ms, r.tiled_ms[0], r.tiled_ms[1],
                    r.tiled_ms[2], r.tiled_ms[3]);
    }

    std::string md = "# Prefill attention: today's kernel, varlen v0 and the tiled kernel\n\n";
    md += std::string("- **GPU:** ") + prop.name + ", " + std::to_string(prop.multiProcessorCount) + " SMs, compute capability " +
          std::to_string(prop.major) + "." + std::to_string(prop.minor) + "\n";
    md += "- **Shapes:** Llama 3.2 1B: 32 query heads, 8 KV heads, head dimension 64, 16-token pages (shuffled)\n";
    md += "- **Timing:** CUDA events, median of 20 runs after 3 warm-up runs, one layer's attention. `today` is "
          "Day 34's `attention_prefill_paged`, one launch per sequence (what the backend runs for a prefill from a "
          "cached offset); `v0` is `attention_prefill_varlen_v0`, one launch for the pack; `tiled RxC` is "
          "`attention_prefill_varlen_tiled` with R query rows per block and C keys per K/V tile.\n";
    md += "- **Against the step:** 16 layers of today's kernel against the whole prefill step at Day 46's 0.28 ms "
          "per token (measured up to 480 tokens; longer prompts extrapolated, since the matmuls scale linearly). "
          "A prompt prefilled from zero does not run this kernel (the engine's own prefill attention does; Day 46's "
          "flat cost per token shows it is cheap); a prefix hit, a chunk or a resumed request does. The last column "
          "is what this kernel would cost such a prefill.\n\n";
    md += "| Case | new tokens | keys (max) | today, ms per layer | v0, ms per layer | today x 16 layers | prefill step "
          "(Day 46) | this kernel against the step |\n|---|---|---|---|---|---|---|---|\n";
    char buf[512];
    for (const auto& r : results) {
        const double step = kPrefillMsPerToken * r.tokens;
        std::snprintf(buf, sizeof buf, "| %s | %d | %d | %.3f | %.3f | %.1f ms | %.1f ms | %.0f%% |\n", r.name.c_str(),
                      r.tokens, r.max_keys, r.today_ms, r.v0_ms, r.today_ms * kLayers, step,
                      100.0 * r.today_ms * kLayers / step);
        md += buf;
    }
    md += "\n## The tiled kernel, ms per layer\n\n| Case | v0 | tiled 64x64 | tiled 64x32 | tiled 128x64 | tiled 128x32 | "
          "best | speed-up on today's | best x 16 layers |\n|---|---|---|---|---|---|---|---|---|\n";
    for (const auto& r : results) {
        const auto best = std::min_element(r.tiled_ms.begin(), r.tiled_ms.end());
        const char* best_name = neuroserve::kernels::tile_shape_name(kShapes[best - r.tiled_ms.begin()]);
        std::snprintf(buf, sizeof buf, "| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %s | %.1fx | %.2f ms |\n",
                      r.name.c_str(), r.v0_ms, r.tiled_ms[0], r.tiled_ms[1], r.tiled_ms[2], r.tiled_ms[3], best_name,
                      r.today_ms / *best, *best * kLayers);
        md += buf;
    }
    std::printf("\n%s", md.c_str());
    if (out_path) {
        FILE* f = std::fopen(out_path, "w");
        if (!f) {
            std::perror(out_path);
            return 1;
        }
        std::fputs(md.c_str(), f);
        std::fclose(f);
        std::printf("wrote %s\n", out_path);
    }
    return 0;
}
