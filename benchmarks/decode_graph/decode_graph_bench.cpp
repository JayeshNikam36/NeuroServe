// CUDA Graphs for decode, A/B (Day 57, docs/cuda-graphs.md).
//
//   ./build/dev-cuda/benchmarks/decode_graph/ns_decode_graph_bench --out benchmarks/results/day57-cuda-graphs.md
//
// Times decode steps through the CUDA backend's three decode paths, in one process:
//   - eager:       the engine's batched decode (model_decode_batch_greedy), what runs today
//   - graph-eager: the graph's work issued kernel by kernel (padded to the bucket, its buffers,
//                  cuBLAS handle and stream), the same work as a replay without the graph
//   - graph:       the same step replayed from a captured CUDA Graph
// for every batch size 1..max_batch (3, 5, 6, 7 padded to their bucket) at one context length.
// Each run: fresh sequences prefilled to --context tokens, --warmup decode steps (a graph's first
// use, and its capture, fall here), then --steps timed steps: the wall time of each
// decode_step() call, as the engine loop sees it. Graph paths also report the host time of a
// step: writing the inputs, issuing the GPU work, waiting for it.
// Interleaved: every round runs every (batch, path), the path order rotating from round to round
// so each path takes each position equally often; the first round is discarded (warm-up). Each
// run's SM clock and clock-event reasons are read through NVML right after it.
// Model: $NS_MODEL_DIR (default /models/llama-3.2-1b-fp16).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <nvml.h>

#include "neuroserve/backend/cuda_llm_engine_backend.hpp"

namespace {

namespace ns = neuroserve;
using Backend = ns::CudaLlmEngineBackend;
using Path = Backend::DecodePath;
using Clock = std::chrono::steady_clock;

const char* path_name(Path p) {
    return p == Path::kEager ? "eager" : p == Path::kGraphEager ? "graph-eager" : "graph";
}

struct GpuSample {
    unsigned sm_mhz = 0, temp_c = 0;
    unsigned long long reasons = 0;
};

class Nvml {
public:
    Nvml() { ok_ = nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByIndex_v2(0, &dev_) == NVML_SUCCESS; }
    ~Nvml() {
        if (ok_) nvmlShutdown();
    }
    GpuSample sample() const {
        GpuSample s;
        if (!ok_) return s;
        nvmlDeviceGetClockInfo(dev_, NVML_CLOCK_SM, &s.sm_mhz);
        nvmlDeviceGetTemperature(dev_, NVML_TEMPERATURE_GPU, &s.temp_c);
        nvmlDeviceGetCurrentClocksEventReasons(dev_, &s.reasons);
        return s;
    }

private:
    bool ok_ = false;
    nvmlDevice_t dev_{};
};
constexpr unsigned long long kPowerReasons = 0x4ULL | 0x80ULL;
constexpr unsigned long long kThermalReasons = 0x20ULL | 0x40ULL | 0x8ULL;

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const auto i = static_cast<std::size_t>(q * static_cast<double>(v.size() - 1) + 0.5);
    return v[std::min(i, v.size() - 1)];
}
double median(std::vector<double> v) { return percentile(std::move(v), 0.5); }

struct Run {
    int round = 0, batch = 0;
    Path path = Path::kEager;
    double p50_ms = 0, p90_ms = 0, mean_ms = 0;
    double stage_ms = 0, issue_ms = 0, wait_ms = 0;  // per step, graph paths only
    GpuSample gpu;
};

int bucket_of(int n) {
    for (const int b : {1, 2, 4, 8, 16})
        if (n <= b) return b;
    return 0;
}

Run run_one(Backend& b, Path path, int batch, int context, int warmup, int steps, int round, const Nvml& nvml) {
    b.set_decode_path(path);
    std::vector<ns::SeqId> ids;
    std::vector<ns::TokenId> last;
    for (int s = 0; s < batch; ++s) {
        std::vector<ns::TokenId> prompt(static_cast<std::size_t>(context));
        for (int t = 0; t < context; ++t) prompt[static_cast<std::size_t>(t)] = 100 + (t * 7 + s * 131) % 5000;
        ids.push_back(b.create_sequence());
        last.push_back(b.prefill(ids.back(), prompt));
    }
    for (int i = 0; i < warmup; ++i) last = b.decode_step(ids, last);
    b.reset_decode_host_times();
    std::vector<double> wall;
    wall.reserve(static_cast<std::size_t>(steps));
    for (int i = 0; i < steps; ++i) {
        const auto t0 = Clock::now();
        last = b.decode_step(ids, last);
        wall.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    Run r;
    r.round = round;
    r.batch = batch;
    r.path = path;
    r.p50_ms = percentile(wall, 0.5);
    r.p90_ms = percentile(wall, 0.9);
    r.mean_ms = std::accumulate(wall.begin(), wall.end(), 0.0) / static_cast<double>(wall.size());
    const Backend::DecodeHostTimes t = b.decode_host_times();
    if (t.steps > 0) {
        r.stage_ms = t.stage_ms / static_cast<double>(t.steps);
        r.issue_ms = t.issue_ms / static_cast<double>(t.steps);
        r.wait_ms = t.wait_ms / static_cast<double>(t.steps);
    }
    r.gpu = nvml.sample();
    for (const ns::SeqId id : ids) b.free_sequence(id);
    b.set_decode_path(Path::kEager);
    return r;
}

std::string fmt(double v, int digits = 2) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*f", digits, v);
    return buf;
}
std::string pct(double ratio) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%+.1f%%", (ratio - 1.0) * 100.0);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_path;
    int rounds = 5, context = 512, warmup = 8, steps = 64, max_batch = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](int& v) {
            if (i + 1 < argc) v = std::atoi(argv[++i]);
        };
        if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--rounds") next(rounds);
        else if (a == "--context") next(context);
        else if (a == "--warmup") next(warmup);
        else if (a == "--steps") next(steps);
        else if (a == "--max-batch") next(max_batch);
        else {
            std::fprintf(stderr, "usage: %s [--out FILE] [--rounds 5] [--context 512] [--warmup 8] [--steps 64] "
                                 "[--max-batch 8]\n", argv[0]);
            return 2;
        }
    }
    if (rounds < 2 || steps < 1 || context < 1 || max_batch < 1 || max_batch > 16) {
        std::fprintf(stderr, "rounds >= 2 (the first is discarded), steps >= 1, 1 <= max-batch <= 16\n");
        return 2;
    }
    const char* env = std::getenv("NS_MODEL_DIR");
    ns::CudaLlmEngineConfig config;
    config.model_dir = env ? env : "/models/llama-3.2-1b-fp16";
    config.max_batch = max_batch;
    config.max_prompt_tokens = 512;
    const int tokens_per_seq = context + warmup + steps + 1;
    config.max_pages_per_sequence = (tokens_per_seq + 15) / 16 + 1;
    config.num_pages = max_batch * config.max_pages_per_sequence + 16;
    Backend backend(config);
    Nvml nvml;
    const Path paths[] = {Path::kEager, Path::kGraphEager, Path::kGraph};
    std::printf("%d rounds (first discarded) x %d batch sizes x 3 paths; context %d, %d warm-up + %d timed steps\n",
                rounds, max_batch, context, warmup, steps);
    std::vector<Run> runs;
    for (int round = 1; round <= rounds; ++round) {
        for (int batch = 1; batch <= max_batch; ++batch) {
            for (int k = 0; k < 3; ++k) {
                const Path path = paths[(k + round - 1) % 3];  // the order rotates each round
                runs.push_back(run_one(backend, path, batch, context, warmup, steps, round, nvml));
                const Run& r = runs.back();
                std::printf("round %d%s batch %d %-11s p50 %6.2f ms p90 %6.2f ms | stage %.3f issue %.3f wait %.2f ms"
                            " | sm %u MHz %u C%s\n",
                            round, round == 1 ? " (warm-up)" : "", batch, path_name(path), r.p50_ms, r.p90_ms,
                            r.stage_ms, r.issue_ms, r.wait_ms, r.gpu.sm_mhz, r.gpu.temp_c,
                            (r.gpu.reasons & (kPowerReasons | kThermalReasons)) ? " (capped)" : "");
                std::fflush(stdout);
            }
        }
    }

    // Medians over the measured rounds; paired ratios per round.
    auto measured = [&](int batch, Path path) {
        std::vector<const Run*> v;
        for (const Run& r : runs)
            if (r.round > 1 && r.batch == batch && r.path == path) v.push_back(&r);
        return v;
    };
    auto med = [&](int batch, Path path, double Run::*field) {
        std::vector<double> v;
        for (const Run* r : measured(batch, path)) v.push_back(r->*field);
        return median(v);
    };
    auto paired = [&](int batch, Path a, Path b) {  // median over rounds of a / b, and rounds where a < b
        std::vector<double> ratios;
        int wins = 0;
        for (const Run* ra : measured(batch, a))
            for (const Run* rb : measured(batch, b))
                if (ra->round == rb->round) {
                    ratios.push_back(ra->p50_ms / rb->p50_ms);
                    wins += ra->p50_ms < rb->p50_ms;
                }
        return std::pair<double, int>{median(ratios), wins};
    };
    const int measured_rounds = rounds - 1;
    int capped = 0, thermal = 0;
    unsigned sm_min = ~0u, sm_max = 0;
    for (const Run& r : runs) {
        if (r.round == 1) continue;
        capped += (r.gpu.reasons & kPowerReasons) != 0;
        thermal += (r.gpu.reasons & kThermalReasons) != 0;
        sm_min = std::min(sm_min, r.gpu.sm_mhz);
        sm_max = std::max(sm_max, r.gpu.sm_mhz);
    }

    std::ostringstream md;
    md << "# CUDA Graphs for decode: A/B\n\n"
       << "- **What:** the wall time of one `decode_step()` call (P50 over " << steps << " timed steps after " << warmup
       << " warm-up steps), each batch of sequences prefilled to " << context << " tokens; median over "
       << measured_rounds << " measured rounds (round 1 discarded), the path order rotating each round.\n"
       << "- **Paths:** `eager` = the engine's batched decode (today); `graph-eager` = the graph's work issued "
          "kernel by kernel (padded to the bucket); `graph` = replayed from a CUDA Graph.\n"
       << "- **GPU:** SM clock " << sm_min << "-" << sm_max << " MHz after the measured runs; power-capped in "
       << capped << ", thermally capped in " << thermal << " of " << measured_rounds * max_batch * 3 << " runs.\n\n"
       << "## Step time\n\n"
       << "| batch | bucket | eager (ms) | graph-eager (ms) | graph (ms) | graph vs eager | graph vs graph-eager |\n"
       << "| --- | --- | --- | --- | --- | --- | --- |\n";
    for (int batch = 1; batch <= max_batch; ++batch) {
        const auto [ve, we] = paired(batch, Path::kGraph, Path::kEager);
        const auto [vg, wg] = paired(batch, Path::kGraph, Path::kGraphEager);
        md << "| " << batch << " | " << bucket_of(batch) << " | " << fmt(med(batch, Path::kEager, &Run::p50_ms)) << " | "
           << fmt(med(batch, Path::kGraphEager, &Run::p50_ms)) << " | " << fmt(med(batch, Path::kGraph, &Run::p50_ms))
           << " | " << pct(ve) << " (" << we << "/" << measured_rounds << ") | " << pct(vg) << " (" << wg << "/"
           << measured_rounds << ") |\n";
    }
    md << "\nPaired with the same round's run: median change of the step time, and in how many rounds the graph "
          "was faster.\n\n"
       << "## Host time per step (graph paths)\n\n"
       << "| batch | graph-eager: stage / issue / wait (ms) | graph: stage / issue / wait (ms) |\n"
       << "| --- | --- | --- |\n";
    for (int batch = 1; batch <= max_batch; ++batch) {
        md << "| " << batch << " | " << fmt(med(batch, Path::kGraphEager, &Run::stage_ms), 3) << " / "
           << fmt(med(batch, Path::kGraphEager, &Run::issue_ms), 3) << " / "
           << fmt(med(batch, Path::kGraphEager, &Run::wait_ms)) << " | "
           << fmt(med(batch, Path::kGraph, &Run::stage_ms), 3) << " / " << fmt(med(batch, Path::kGraph, &Run::issue_ms), 3)
           << " / " << fmt(med(batch, Path::kGraph, &Run::wait_ms)) << " |\n";
    }
    md << "\nStage: writing the pinned inputs; issue: launching the GPU work (about 280 kernels kernel by kernel, "
          "one `cudaGraphLaunch` from a graph); wait: `cudaStreamSynchronize` until it is done.\n\n"
       << "## The cost of padding\n\n"
       << "A batch that is not a bucket size runs padded to its bucket. The graph's step time at each such batch "
          "against the graph's at its bucket, and against the engine's unpadded eager step:\n\n"
       << "| batch | bucket | graph (ms) | graph at the bucket (ms) | vs the bucket | eager, unpadded (ms) |\n"
       << "| --- | --- | --- | --- | --- | --- |\n";
    for (int batch = 1; batch <= max_batch; ++batch) {
        const int bucket = bucket_of(batch);
        if (bucket == batch || bucket > max_batch) continue;
        const double g = med(batch, Path::kGraph, &Run::p50_ms), gb = med(bucket, Path::kGraph, &Run::p50_ms);
        md << "| " << batch << " | " << bucket << " | " << fmt(g) << " | " << fmt(gb) << " | " << pct(g / gb) << " | "
           << fmt(med(batch, Path::kEager, &Run::p50_ms)) << " |\n";
    }
    md << "\nGraphs captured: " << backend.cuda_graphs_captured() << "; replays: " << backend.cuda_graph_replays()
       << ".\n";
    const std::string report = md.str();
    std::printf("\n%s", report.c_str());
    if (!out_path.empty()) {
        const std::filesystem::path dir = std::filesystem::path(out_path).parent_path();
        if (!dir.empty()) std::filesystem::create_directories(dir);
        std::ofstream(out_path) << report;
        std::printf("wrote %s\n", out_path.c_str());
    }
    return 0;
}
