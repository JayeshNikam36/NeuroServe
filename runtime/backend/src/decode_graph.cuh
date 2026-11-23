#pragma once
// CUDA Graphs for decode (Day 56, docs/cuda-graphs.md): the engine's batched greedy decode step
// (cuda-llm-engine engine/batched.cu, model_decode_batch_greedy), captured once per (bucket,
// split count) and replayed. Only NeuroServe's backend uses it.

#include <cstdint>
#include <map>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "paged.h"

namespace neuroserve::backend_detail {

// How a decode step runs. kEager is the engine's own call; kGraph replays a captured graph;
// kGraphEager does the graph's work without the graph (the same padding, buffers, cuBLAS handle and
// stream), so a replay can be compared with it bit for bit.
enum class DecodeMode { kEager, kGraphEager, kGraph };

class DecodeGraphs {
public:
    // Buckets of 1, 2, 4, 8 and 16 sequences, up to the first that holds max_batch (at most 16).
    // pad_page is a KV page no sequence holds: padding rows write their K/V there and nowhere else.
    DecodeGraphs(Model& m, PagedKVCache& c, int max_batch, int max_pages, int pad_page);
    ~DecodeGraphs();
    DecodeGraphs(const DecodeGraphs&) = delete;
    DecodeGraphs& operator=(const DecodeGraphs&) = delete;

    int max_bucket() const { return buckets_.back(); }
    static int bucket_for(const std::vector<int>& buckets, int n);  // the smallest bucket >= n, or 0

    // One greedy decode step, as model_decode_batch_greedy: tokens[i] is fed to seqs[i] at position
    // seqs[i]->len, whose pages must already cover it; next gets each sequence's next token and
    // every sequence advances by one. n <= max_bucket(). `graph` false runs the same work eagerly.
    void step(std::vector<Sequence*>& seqs, const std::vector<int>& tokens, std::vector<int>& next, bool graph);

    int graphs() const { return static_cast<int>(execs_.size()); }  // captured (bucket, splits) pairs
    std::uint64_t replays() const { return replays_; }

    // Host time of the steps so far (Day 57), in milliseconds: writing the pinned inputs (stage),
    // issuing the GPU work (issue: the ~280 launches eagerly, one cudaGraphLaunch from a graph),
    // and waiting for it (wait). A first use's capture is not counted.
    struct HostTimes {
        double stage_ms = 0, issue_ms = 0, wait_ms = 0;
        std::uint64_t steps = 0;
    };
    const HostTimes& host_times() const { return times_; }
    void reset_host_times() { times_ = {}; }

private:
    void enqueue(int bucket, int splits);   // the step's work on stream_, for `bucket` rows
    cudaGraphExec_t graph_for(int bucket, int splits);

    Model& m_;
    PagedKVCache& c_;
    int max_pages_ = 0;
    int pad_page_ = 0;
    std::vector<int> buckets_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t handle_ = nullptr;  // its own, with a workspace: nothing is allocated in a capture
    void* cublas_ws_ = nullptr;
    // Inputs, one contiguous block (ids, positions, lengths, then the tables, max_bucket rows each),
    // pinned on the host and mirrored on the device; outputs likewise.
    int* h_in_ = nullptr;
    int* d_in_ = nullptr;
    int* h_next_ = nullptr;
    int* d_next_ = nullptr;
    float* d_ws_ = nullptr;        // attention workspace for max_bucket rows and 32 splits
    __half* d_normed_ = nullptr;   // [max_bucket, hidden]
    __half* d_logits_ = nullptr;   // [max_bucket, vocab]
    std::map<int, cudaGraphExec_t> execs_;  // key: bucket * 64 + splits
    std::uint64_t replays_ = 0;
    HostTimes times_;
};

}  // namespace neuroserve::backend_detail
