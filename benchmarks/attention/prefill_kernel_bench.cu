// Prefill attention kernel benchmark (Day 50, docs/prefill-attention.md).
//
//   ./build/dev-cuda/benchmarks/attention/ns_prefill_kernel_bench --out benchmarks/results/day50-prefill-kernel.md
//
// For prompt lengths 64-2,048 at batch 1 and batch 8 (fresh prompts, no cached prefix), times one
// layer's prefill attention with CUDA events:
//   - naive:  Day 34's attention_prefill_paged, one launch per sequence (what the engine runs today
//             for prefill from an offset)
//   - tiled:  attention_prefill_varlen_tiled at 64 x 32 (the default) and 128 x 64
// Every timed run starts with the L2 cache flushed (a memset larger than L2, outside the timed
// interval), so K and V come from DRAM as in a real step; the default shape is also timed with a
// warm L2, to show what flushing changes. Each run's SM clock, temperature, power and clock-event
// reasons are read through NVML right after it, and the report flags power or thermal capping.
// Achieved TFLOPS count the attention matmuls under the causal mask: 4 x heads x head_dim per
// (query, visible key) pair. Each case also checks the tiled outputs against the naive ones.

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
#include <nvml.h>

#include "neuroserve/kernels/attention_varlen.hpp"
#include "prefill_from_offset.cuh"

namespace {

namespace nk = neuroserve::kernels;

constexpr int kHeads = 32, kKvHeads = 8, kDim = 64, kPage = 16, kLayers = 16;
constexpr int kWarmup = 3;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ---- NVML: clocks, temperature, power, clock-event reasons ---------------------------------------
struct GpuSample {
    unsigned sm_mhz = 0, temp_c = 0, power_mw = 0;
    unsigned long long reasons = 0;
};

class Nvml {
public:
    Nvml() {
        ok_ = nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByIndex_v2(0, &dev_) == NVML_SUCCESS;
    }
    ~Nvml() {
        if (ok_) nvmlShutdown();
    }
    bool ok() const { return ok_; }
    GpuSample sample() const {
        GpuSample s;
        if (!ok_) return s;
        nvmlDeviceGetClockInfo(dev_, NVML_CLOCK_SM, &s.sm_mhz);
        nvmlDeviceGetTemperature(dev_, NVML_TEMPERATURE_GPU, &s.temp_c);
        nvmlDeviceGetPowerUsage(dev_, &s.power_mw);
        nvmlDeviceGetCurrentClocksEventReasons(dev_, &s.reasons);
        return s;
    }

private:
    bool ok_ = false;
    nvmlDevice_t dev_{};
};

constexpr unsigned long long kPowerReasons = 0x4ULL | 0x80ULL;            // software power cap, power brake
constexpr unsigned long long kThermalReasons = 0x20ULL | 0x40ULL | 0x8ULL;  // software/hardware thermal, HW slowdown

// ---- Timing -----------------------------------------------------------------------------------
struct Timing {
    double median_ms = 0, p10_ms = 0, p90_ms = 0;
    unsigned sm_mhz_median = 0, sm_mhz_min = 0, temp_max = 0;
    double power_w_mean = 0;
    int power_capped = 0, thermal = 0, runs = 0;
};

class L2Flusher {
public:
    L2Flusher() {
        int l2 = 0;
        check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0), "L2 size");
        bytes_ = std::max<size_t>(static_cast<size_t>(l2) * 4, size_t{64} << 20);
        check(cudaMalloc(&buf_, bytes_), "flush buffer");
    }
    ~L2Flusher() { cudaFree(buf_); }
    void flush(int i) { check(cudaMemsetAsync(buf_, i & 0xff, bytes_), "flush"); }
    size_t bytes() const { return bytes_; }

private:
    void* buf_ = nullptr;
    size_t bytes_ = 0;
};

template <typename F>
Timing time_it(F&& launch, L2Flusher& l2, const Nvml& nvml, bool flush, double budget_ms = 4000.0) {
    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event");
    check(cudaEventCreate(&b), "event");
    for (int i = 0; i < kWarmup; ++i) launch();
    check(cudaDeviceSynchronize(), "warm-up");
    std::vector<float> ms;
    std::vector<GpuSample> gpu;
    double spent = 0;
    for (int i = 0; i < 25 && (i < 5 || spent < budget_ms); ++i) {
        if (flush) l2.flush(i);
        check(cudaEventRecord(a), "record");
        launch();
        check(cudaEventRecord(b), "record");
        check(cudaEventSynchronize(b), "sync");
        float t = 0;
        check(cudaEventElapsedTime(&t, a, b), "elapsed");
        ms.push_back(t);
        spent += t;
        gpu.push_back(nvml.sample());
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::vector<float> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    const auto at = [&](double q) { return static_cast<double>(sorted[static_cast<size_t>(q * (sorted.size() - 1) + 0.5)]); };
    Timing t;
    t.runs = static_cast<int>(ms.size());
    t.median_ms = at(0.5);
    t.p10_ms = at(0.1);
    t.p90_ms = at(0.9);
    std::vector<unsigned> clocks;
    double power = 0;
    for (const auto& g : gpu) {
        clocks.push_back(g.sm_mhz);
        t.temp_max = std::max(t.temp_max, g.temp_c);
        power += g.power_mw / 1000.0;
        t.power_capped += (g.reasons & kPowerReasons) ? 1 : 0;
        t.thermal += (g.reasons & kThermalReasons) ? 1 : 0;
    }
    std::sort(clocks.begin(), clocks.end());
    t.sm_mhz_median = clocks[clocks.size() / 2];
    t.sm_mhz_min = clocks.front();
    t.power_w_mean = power / static_cast<double>(gpu.size());
    return t;
}

// ---- One case: batch x length ----------------------------------------------------------------
struct Row {
    int batch = 0, length = 0;
    double gflop = 0;
    Timing naive, t64x32, t128x64, t64x32_warm;
    float max_diff = 0;  // tiled 64 x 32 against naive
};

template <typename T>
T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    check(cudaMalloc(&p, v.size() * sizeof(T)), "cudaMalloc");
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    return p;
}

Row run_case(int batch, int length, std::mt19937& rng, L2Flusher& l2, const Nvml& nvml) {
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    const int pages_per = (length + kPage - 1) / kPage;
    const int pages = batch * pages_per;
    std::vector<int> ids(static_cast<size_t>(pages));
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<std::int32_t> tables(ids.begin(), ids.end()), cu{0}, start(static_cast<size_t>(batch), 0);
    for (int b = 0; b < batch; ++b) cu.push_back(cu.back() + length);
    const size_t row = static_cast<size_t>(kKvHeads) * kDim;
    std::vector<__half> pool(static_cast<size_t>(pages) * kPage * row), q(static_cast<size_t>(cu.back()) * kHeads * kDim);
    for (auto& x : pool) x = __float2half(uni(rng));
    for (auto& x : q) x = __float2half(uni(rng));
    __half* dk = upload(pool);
    __half* dv = upload(pool);
    __half* dq = upload(q);
    __half* dnaive = upload(q);
    __half* dtiled = upload(q);
    std::int32_t* dcu = upload(cu);
    std::int32_t* dstart = upload(start);
    std::int32_t* dtables = upload(tables);

    nk::VarlenAttentionArgs a;
    a.q = dq;
    a.k_pool = dk;
    a.v_pool = dv;
    a.cu_seqlens_q = dcu;
    a.start = dstart;
    a.block_tables = dtables;
    a.out = dtiled;
    a.num_seqs = batch;
    a.total_q = cu.back();
    a.max_keys = length;
    a.max_pages = pages_per;
    a.n_heads = kHeads;
    a.n_kv_heads = kKvHeads;
    a.head_dim = kDim;
    a.page_size = kPage;
    a.scale = 1.f / std::sqrt(static_cast<float>(kDim));

    Row r;
    r.batch = batch;
    r.length = length;
    // Causal: query i sees i + 1 keys; 2 matmuls x 2 FLOPs x head_dim per (query, key) pair, per head.
    r.gflop = 4.0 * kHeads * kDim * batch * (static_cast<double>(length) * (length + 1) / 2) / 1e9;

    const auto naive = [&] {
        for (int b = 0; b < batch; ++b) {
            const size_t off = static_cast<size_t>(cu[static_cast<size_t>(b)]) * kHeads * kDim;
            neuroserve::backend_detail::attention_prefill_paged(dq + off, dk, dv, dtables + b * pages_per, dnaive + off,
                                                                0, length, kPage, kHeads, kKvHeads, kDim);
        }
    };
    const auto tiled = [&](nk::TileShape shape) {
        const std::vector<std::int32_t> map = nk::varlen_tile_map(cu, start, nk::tile_rows(shape));
        std::int32_t* dmap = upload(map);
        const int tiles = static_cast<int>(map.size() / 2);
        return std::make_pair(dmap, tiles);
    };
    const auto m32 = tiled(nk::TileShape::k64x32);
    const auto m64 = tiled(nk::TileShape::k128x64);
    std::int32_t* const map32 = m32.first;
    std::int32_t* const map64 = m64.first;
    const int tiles32 = m32.second;
    const int tiles64 = m64.second;
    r.naive = time_it(naive, l2, nvml, true);
    r.t64x32 = time_it([&] { nk::attention_prefill_varlen_tiled(a, map32, tiles32, nk::TileShape::k64x32); }, l2, nvml, true);
    r.t128x64 = time_it([&] { nk::attention_prefill_varlen_tiled(a, map64, tiles64, nk::TileShape::k128x64); }, l2, nvml, true);
    r.t64x32_warm =
        time_it([&] { nk::attention_prefill_varlen_tiled(a, map32, tiles32, nk::TileShape::k64x32); }, l2, nvml, false);
    check(cudaDeviceSynchronize(), "kernels");

    // Sanity: the last tiled 64 x 32 output against the naive one.
    std::vector<__half> hn(q.size()), ht(q.size());
    check(cudaMemcpy(hn.data(), dnaive, hn.size() * sizeof(__half), cudaMemcpyDeviceToHost), "copy");
    check(cudaMemcpy(ht.data(), dtiled, ht.size() * sizeof(__half), cudaMemcpyDeviceToHost), "copy");
    for (size_t i = 0; i < hn.size(); ++i)
        r.max_diff = std::max(r.max_diff, std::fabs(__half2float(hn[i]) - __half2float(ht[i])));

    for (void* p : {static_cast<void*>(dk), static_cast<void*>(dv), static_cast<void*>(dq), static_cast<void*>(dnaive),
                    static_cast<void*>(dtiled), static_cast<void*>(dcu), static_cast<void*>(dstart),
                    static_cast<void*>(dtables), static_cast<void*>(map32), static_cast<void*>(map64)})
        cudaFree(p);
    return r;
}

std::string fmt(const char* f, double v) {
    char b[64];
    std::snprintf(b, sizeof b, f, v);
    return b;
}

std::string clock_cell(const Timing& t, bool nvml) {
    if (!nvml) return "–";
    std::string s = std::to_string(t.sm_mhz_median) + " (" + std::to_string(t.sm_mhz_min) + ")";
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    const char* out_path = nullptr;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--out" && i + 1 < argc) out_path = argv[++i];

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "device");
    Nvml nvml;
    L2Flusher l2;
    std::mt19937 rng(50);
    const std::vector<int> lengths = {64, 128, 256, 512, 1024, 2048};
    std::vector<Row> rows;
    for (const int batch : {1, 8}) {
        for (const int length : lengths) {
            rows.push_back(run_case(batch, length, rng, l2, nvml));
            const Row& r = rows.back();
            std::printf("batch %d x %4d: naive %9.3f ms | tiled 64x32 %7.3f ms %5.2f TFLOPS | 128x64 %7.3f ms %5.2f TFLOPS"
                        " | warm-L2 64x32 %7.3f ms | sm %u MHz (min %u) %u C | max diff %.4f\n",
                        r.batch, r.length, r.naive.median_ms, r.t64x32.median_ms, r.gflop / r.t64x32.median_ms,
                        r.t128x64.median_ms, r.gflop / r.t128x64.median_ms, r.t64x32_warm.median_ms,
                        r.t64x32.sm_mhz_median, r.t64x32.sm_mhz_min, r.t64x32.temp_max, r.max_diff);
            std::fflush(stdout);
        }
    }

    int capped = 0, thermal = 0, runs = 0;
    for (const auto& r : rows)
        for (const Timing* t : {&r.naive, &r.t64x32, &r.t128x64, &r.t64x32_warm}) {
            capped += t->power_capped;
            thermal += t->thermal;
            runs += t->runs;
        }

    std::string md = "# Prefill attention kernel: time and TFLOPS by length\n\n";
    md += std::string("- **GPU:** ") + prop.name + ", " + std::to_string(prop.multiProcessorCount) + " SMs, L2 " +
          std::to_string(prop.l2CacheSize >> 20) + " MB, compute capability " + std::to_string(prop.major) + "." +
          std::to_string(prop.minor) + "\n";
    md += "- **Shapes:** Llama 3.2 1B attention: 32 query heads, 8 KV heads, head dimension 64; fresh prompts "
          "(no cached prefix), 16-token pages in shuffled order; one layer\n";
    md += "- **Kernels:** naive = Day 34's `attention_prefill_paged`, one launch per sequence; tiled = "
          "`attention_prefill_varlen_tiled`, one launch for the batch, at 64 x 32 (the default) and 128 x 64\n";
    md += "- **Timing:** CUDA events; " + std::to_string(kWarmup) +
          " warm-up runs, then up to 25 timed runs (at least 5, about 4 s per kernel and case); median, with P10–P90. "
          "Every timed run starts with the L2 flushed: a " + std::to_string(l2.bytes() >> 20) +
          " MB memset outside the timed interval. `warm` = the default shape without flushing.\n";
    md += "- **TFLOPS:** the attention matmuls under the causal mask, 4 x 32 heads x 64 x (pairs of query and visible "
          "key) per layer, over the median time\n";
    md += std::string("- **Clocks:** ") +
          (nvml.ok() ? "NVML after every timed run: SM clock median (minimum), temperature, power, and clock-event "
                       "reasons. Power or thermal capping in " + std::to_string(capped) + " and " +
                           std::to_string(thermal) + " of " + std::to_string(runs) + " runs."
                     : "NVML not available; not logged.") +
          "\n";
    for (const int batch : {1, 8}) {
        md += "\n## Batch " + std::to_string(batch) + "\n\n";
        md += "| Length | GFLOP | naive (ms) | tiled 64x32 (ms) | P10–P90 | TFLOPS | tiled 128x64 (ms) | TFLOPS | "
              "speed-up (best) | warm L2 64x32 (ms) | SM MHz, 64x32 | max °C | W | x 16 layers, best |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n";
        for (const auto& r : rows) {
            if (r.batch != batch) continue;
            const double best = std::min(r.t64x32.median_ms, r.t128x64.median_ms);
            md += "| " + std::to_string(r.length) + " | " + fmt("%.2f", r.gflop) + " | " + fmt("%.3f", r.naive.median_ms) +
                  " | " + fmt("%.3f", r.t64x32.median_ms) + " | " + fmt("%.3f", r.t64x32.p10_ms) + "–" +
                  fmt("%.3f", r.t64x32.p90_ms) + " | " + fmt("%.2f", r.gflop / r.t64x32.median_ms) + " | " +
                  fmt("%.3f", r.t128x64.median_ms) + " | " + fmt("%.2f", r.gflop / r.t128x64.median_ms) + " | " +
                  fmt("%.1fx", r.naive.median_ms / best) + " | " + fmt("%.3f", r.t64x32_warm.median_ms) + " | " +
                  clock_cell(r.t64x32, nvml.ok()) + " | " + std::to_string(r.t64x32.temp_max) + " | " +
                  fmt("%.1f", r.t64x32.power_w_mean) + " | " + fmt("%.2f ms", best * kLayers) + " |\n";
        }
    }
    md += "\nNaive TFLOPS for reference: ";
    for (const auto& r : rows) {
        md += "batch " + std::to_string(r.batch) + " x " + std::to_string(r.length) + " " +
              fmt("%.3f", r.gflop / r.naive.median_ms) + (&r == &rows.back() ? "." : "; ");
    }
    float worst = 0;
    for (const auto& r : rows) worst = std::max(worst, r.max_diff);
    md += "\n\nTiled 64 x 32 against naive, largest absolute difference in any output element: " + fmt("%.4f", worst) +
          " (the tests check every case against an FP32 reference).\n";
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
