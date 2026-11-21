// Block-size sweep (Day 42): what the KV block size costs and buys, at 8, 16 and 32 tokens.
//
// - Capacity and waste, exactly, for 10,000 sequences drawn from a fixed, seeded length mix: a
//   block holds block_size tokens, so a sequence of L tokens wastes ceil(L/B)*B - L slots.
// - Decode attention time, measured with cuda-llm-engine's paged decode kernel
//   (attention_decode_paged, which takes the block size as a parameter) on one layer of
//   Llama-3.2-1B's shape, for a batch whose block tables point at shuffled pages, as in serving.
//   The block sizes run interleaved in one process, in rotating order every round, so clock
//   drift or warm-up cannot favour one of them.
//
// The engine's fast batched decode kernel (the one serving uses) is written for 16-token pages, so
// it cannot be swept; the paged kernel isolates what the block size changes: where each position's
// K and V are found.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "attention_decode_paged.cuh"

#define CHECK(x)                                                                                     \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                            \
        }                                                                                            \
    } while (0)

namespace {

// Llama-3.2-1B: 16 layers, 32 query heads, 8 KV heads of 64, FP16 K/V.
constexpr int kLayers = 16, kHeads = 32, kKvHeads = 8, kHeadDim = 64;
constexpr long long kBytesPerToken = 2LL * kLayers * kKvHeads * kHeadDim * 2;  // 32 KiB
constexpr int kBlockSizes[] = {8, 16, 32};
constexpr int kCapacityTokens = 32768;  // the timing pool: 1 GiB of K/V for the whole model

// A chat-like mix: prompt log-uniform in [16, 512] tokens, output uniform in [16, 256].
std::vector<int> sample_lengths(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> log_prompt(std::log(16.0), std::log(512.0));
    std::uniform_int_distribution<int> output(16, 256);
    std::vector<int> out;
    for (int i = 0; i < n; ++i) {
        out.push_back(static_cast<int>(std::lround(std::exp(log_prompt(rng)))) + output(rng));
    }
    return out;
}

struct Fit {
    double waste = 0;           // padding slots / allocated slots
    double blocks = 0;          // mean blocks (block-table entries) per sequence
    double allocated = 0;       // mean tokens allocated per sequence
    double per_gib = 0;         // sequences of the mix that fit in 1 GiB of K/V
};

Fit fit(const std::vector<int>& lengths, int bs) {
    double pad = 0, alloc = 0, blocks = 0;
    for (const int l : lengths) {
        const int b = (l + bs - 1) / bs;
        blocks += b;
        alloc += static_cast<double>(b) * bs;
        pad += static_cast<double>(b) * bs - l;
    }
    Fit f;
    const auto n = static_cast<double>(lengths.size());
    f.waste = pad / alloc;
    f.blocks = blocks / n;
    f.allocated = alloc / n;
    f.per_gib = (1024.0 * 1024 * 1024 / kBytesPerToken) / f.allocated;
    return f;
}

// One block size's pool and the batch's block tables, pages shuffled.
struct Setup {
    int bs = 0;
    __half* k = nullptr;
    __half* v = nullptr;
    std::vector<int*> tables;  // device, one per sequence
};

Setup make_setup(int bs, const std::vector<int>& batch, unsigned seed) {
    Setup s;
    s.bs = bs;
    const int pages = kCapacityTokens / bs;
    const size_t bytes = static_cast<size_t>(kCapacityTokens) * kKvHeads * kHeadDim * sizeof(__half);
    CHECK(cudaMalloc(&s.k, bytes));
    CHECK(cudaMalloc(&s.v, bytes));
    CHECK(cudaMemset(s.k, 0, bytes));
    CHECK(cudaMemset(s.v, 0, bytes));
    std::vector<int> order(static_cast<size_t>(pages));
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937(seed));
    size_t next = 0;
    for (const int len : batch) {
        const size_t need = static_cast<size_t>((len + bs - 1) / bs);
        if (next + need > order.size()) {
            std::fprintf(stderr, "the timing pool is too small for the batch\n");
            std::exit(1);
        }
        int* table = nullptr;
        CHECK(cudaMalloc(&table, need * sizeof(int)));
        CHECK(cudaMemcpy(table, order.data() + next, need * sizeof(int), cudaMemcpyHostToDevice));
        s.tables.push_back(table);
        next += need;
    }
    return s;
}

void free_setup(Setup& s) {
    for (int* t : s.tables) cudaFree(t);
    cudaFree(s.k);
    cudaFree(s.v);
}

// One decode step's attention for the whole batch (one layer): a launch per sequence.
void step(const Setup& s, const std::vector<int>& batch, const __half* q, __half* out) {
    for (size_t i = 0; i < batch.size(); ++i) {
        attention_decode_paged(q, s.k, s.v, s.tables[i], out, batch[i], s.bs, kHeads, kKvHeads, kHeadDim);
    }
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
}

struct Timing {
    double median_us = 0, min_us = 0, max_us = 0;
};

// Interleaved rounds: round r runs the block sizes starting at index r % 3, each for `reps` steps
// between two events. Returns per-step microseconds for one layer.
std::vector<Timing> time_scenario(const std::vector<int>& batch, int rounds, int reps, unsigned seed) {
    std::vector<Setup> setups;
    for (const int bs : kBlockSizes) setups.push_back(make_setup(bs, batch, seed));
    __half *q = nullptr, *out = nullptr;
    CHECK(cudaMalloc(&q, kHeads * kHeadDim * sizeof(__half)));
    CHECK(cudaMalloc(&out, kHeads * kHeadDim * sizeof(__half)));
    CHECK(cudaMemset(q, 0, kHeads * kHeadDim * sizeof(__half)));
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    for (const Setup& s : setups) {  // warm up every configuration
        for (int i = 0; i < 20; ++i) step(s, batch, q, out);
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<std::vector<double>> samples(setups.size());
    for (int r = 0; r < rounds; ++r) {
        for (size_t j = 0; j < setups.size(); ++j) {
            const size_t k = (static_cast<size_t>(r) + j) % setups.size();
            CHECK(cudaEventRecord(a));
            for (int i = 0; i < reps; ++i) step(setups[k], batch, q, out);
            CHECK(cudaEventRecord(b));
            CHECK(cudaEventSynchronize(b));
            float ms = 0;
            CHECK(cudaEventElapsedTime(&ms, a, b));
            samples[k].push_back(1000.0 * ms / reps);
        }
    }
    CHECK(cudaGetLastError());
    std::vector<Timing> out_t;
    for (auto& v : samples) {
        out_t.push_back({median(v), *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end())});
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaFree(q);
    cudaFree(out);
    for (Setup& s : setups) free_setup(s);
    return out_t;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_path;
    int rounds = 30, reps = 20;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (arg == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (arg == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: %s [--out FILE.md] [--rounds N] [--reps N]\n", argv[0]);
            return 2;
        }
    }
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int runtime = 0, driver = 0;
    CHECK(cudaRuntimeGetVersion(&runtime));
    CHECK(cudaDriverGetVersion(&driver));

    const std::vector<int> mix = sample_lengths(10000, 42);
    std::vector<Fit> fits;
    for (const int bs : kBlockSizes) fits.push_back(fit(mix, bs));

    // Two decode batches of 8 (max_batch): lengths from the mix (seed 7), and 8 long contexts.
    const std::vector<int> mixed_batch = sample_lengths(8, 7);
    const std::vector<int> long_batch(8, 1000);
    std::fprintf(stderr, "timing %d rounds x %d steps per block size (interleaved)...\n", rounds, reps);
    const std::vector<Timing> t_mixed = time_scenario(mixed_batch, rounds, reps, 1);
    const std::vector<Timing> t_long = time_scenario(long_batch, rounds, reps, 2);

    std::ostringstream md;
    char when[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(when, sizeof when, "%Y-%m-%dT%H:%MZ", std::gmtime(&now));
    md << "# Block-size sweep (Day 42)\n\n"
       << prop.name << ", compute " << prop.major << "." << prop.minor << ", CUDA runtime " << runtime / 1000 << "."
       << (runtime % 1000) / 10 << ", driver API " << driver / 1000 << "." << (driver % 1000) / 10 << "; " << when
       << ".\nLlama-3.2-1B shape (16 layers, 8 KV heads of 64, FP16): 32 KiB of K/V per token.\n\n"
       << "## Capacity and waste\n\n"
       << "10,000 sequences, prompt log-uniform in [16, 512] tokens plus output uniform in [16, 256] (seed 42).\n\n"
       << "| Block (tokens) | Waste (padding / allocated) | Blocks per sequence | Tokens allocated per sequence | Sequences per GiB of K/V |\n"
       << "| --- | --- | --- | --- | --- |\n";
    char line[256];
    for (size_t i = 0; i < fits.size(); ++i) {
        std::snprintf(line, sizeof line, "| %d | %.2f%% | %.1f | %.1f | %.1f |\n", kBlockSizes[i], 100 * fits[i].waste,
                      fits[i].blocks, fits[i].allocated, fits[i].per_gib);
        md << line;
    }
    auto timing_rows = [&](const char* name, const std::vector<Timing>& t) {
        const double ref = t[1].median_us;  // 16
        for (size_t i = 0; i < t.size(); ++i) {
            std::snprintf(line, sizeof line, "| %s | %d | %.1f | %.1f to %.1f | %+.1f%% | %.0f |\n", name, kBlockSizes[i],
                          t[i].median_us, t[i].min_us, t[i].max_us, 100 * (t[i].median_us / ref - 1),
                          kLayers * t[i].median_us);
            md << line;
        }
    };
    md << "\n## Decode attention time\n\n"
       << "`attention_decode_paged`, one layer, a batch of 8 per step (one launch per sequence), pages shuffled; "
       << rounds << " interleaved rounds of " << reps << " steps per block size. Per step, microseconds.\n\n"
       << "| Batch | Block | Median | Range | vs 16 | Median x 16 layers |\n| --- | --- | --- | --- | --- | --- |\n";
    std::ostringstream lens;
    for (size_t i = 0; i < mixed_batch.size(); ++i) lens << (i ? " " : "") << mixed_batch[i];
    timing_rows(("mix (" + lens.str() + ")").c_str(), t_mixed);
    timing_rows("8 x 1000 tokens", t_long);
    md << "\nThe engine's fast batched decode kernel, the one serving uses, is written for 16-token pages and is not "
          "swept; this kernel isolates what the block size changes, where each position's K and V are found.\n";

    // The reading: the trade-off in this run's own numbers.
    auto pct = [](double x) { return 100 * x; };
    auto vs16 = [](const std::vector<Timing>& t, size_t i) { return 100 * (t[i].median_us / t[1].median_us - 1); };
    std::snprintf(line, sizeof line,
                  "\n## Reading\n\n- **Waste:** %.2f%% of allocated slots at 8 tokens, %.2f%% at 16, %.2f%% at 32.\n",
                  pct(fits[0].waste), pct(fits[1].waste), pct(fits[2].waste));
    md << line;
    std::snprintf(line, sizeof line,
                  "- **Capacity:** %.1f, %.1f and %.1f sequences of the mix per GiB of K/V at 8, 16 and 32.\n",
                  fits[0].per_gib, fits[1].per_gib, fits[2].per_gib);
    md << line;
    std::snprintf(line, sizeof line,
                  "- **Decode attention vs 16:** %+.1f%% at 8 and %+.1f%% at 32 for the mixed batch; %+.1f%% and "
                  "%+.1f%% for 8 x 1000 tokens.\n",
                  vs16(t_mixed, 0), vs16(t_mixed, 2), vs16(t_long, 0), vs16(t_long, 2));
    md << line;
    std::snprintf(line, sizeof line,
                  "- **Block-table entries:** %.1f, %.1f and %.1f per sequence; a prefix is shared in whole blocks, so "
                  "a smaller block shares more of a prompt and a larger one less.\n",
                  fits[0].blocks, fits[1].blocks, fits[2].blocks);
    md << line;

    std::fputs(md.str().c_str(), stdout);
    if (!out_path.empty()) {
        std::ofstream(out_path) << md.str();
        std::fprintf(stderr, "wrote %s\n", out_path.c_str());
    }
    return 0;
}
