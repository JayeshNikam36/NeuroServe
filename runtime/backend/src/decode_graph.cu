#include "decode_graph.cuh"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

#include "argmax.cuh"
#include "attention_decode_fast.cuh"
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

constexpr int kMaxSplits = 32;                       // the engine's workspace limit (batched.cu)
constexpr size_t kCublasWorkspaceBytes = 4u << 20;   // set, so cuBLAS allocates nothing in a capture
constexpr int kBuckets[] = {1, 2, 4, 8, 16};

using Clock = std::chrono::steady_clock;
double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// The engine's layer_forward_batched (engine/batched.cu, static there), with every launch on `stream`.
void layer_forward(cublasHandle_t handle, const LayerConfig& c, const LayerWeights& w, LayerScratch& s, __half* x,
                   const int* d_pos, const int* d_tables, const int* d_lens, float* d_ws, const float* d_inv_freq,
                   int n, __half* k_pool, __half* v_pool, int max_pages, int page_size, int splits,
                   cudaStream_t stream) {
    const int qdim = c.n_heads * c.head_dim, kvdim = c.n_kv_heads * c.head_dim;
    const size_t n_hidden = static_cast<size_t>(n) * c.hidden;

    rmsnorm(x, w.attn_norm, s.norm, n, c.hidden, c.eps, stream);
    linear(handle, s.norm, w.wq, s.q, n, c.hidden, qdim, stream);
    linear(handle, s.norm, w.wk, s.k, n, c.hidden, kvdim, stream);
    linear(handle, s.norm, w.wv, s.v, n, c.hidden, kvdim, stream);
    rope_apply(s.q, s.k, d_pos, d_inv_freq, n, c.n_heads, c.n_kv_heads, c.head_dim, stream);

    kv_scatter_batch(s.k, k_pool, d_tables, d_pos, n, max_pages, page_size, kvdim, stream);
    kv_scatter_batch(s.v, v_pool, d_tables, d_pos, n, max_pages, page_size, kvdim, stream);

    attention_decode_fast(s.q, k_pool, v_pool, d_tables, d_lens, s.attn, d_ws, n, max_pages, page_size, c.n_heads,
                          c.n_kv_heads, c.head_dim, splits, stream);

    linear(handle, s.attn, w.wo, s.proj, n, qdim, c.hidden, stream);
    residual_add(x, s.proj, x, n_hidden, stream);

    rmsnorm(x, w.mlp_norm, s.norm, n, c.hidden, c.eps, stream);
    linear(handle, s.norm, w.w_gate, s.gate, n, c.hidden, c.inter, stream);
    linear(handle, s.norm, w.w_up, s.up, n, c.hidden, c.inter, stream);
    swiglu(s.gate, s.up, s.act, static_cast<size_t>(n) * c.inter, stream);
    linear(handle, s.act, w.w_down, s.proj, n, c.inter, c.hidden, stream);
    residual_add(x, s.proj, x, n_hidden, stream);
}

}  // namespace

int DecodeGraphs::bucket_for(const std::vector<int>& buckets, int n) {
    for (const int b : buckets) {
        if (n <= b) {
            return b;
        }
    }
    return 0;
}

DecodeGraphs::DecodeGraphs(Model& m, PagedKVCache& c, int max_batch, int max_pages, int pad_page,
                           GpuMemoryManager& memory)
    : m_(m), c_(c), memory_(memory), max_pages_(max_pages), pad_page_(pad_page) {
    for (const int b : kBuckets) {
        buckets_.push_back(b);
        if (b >= max_batch) {
            break;
        }
    }
    const int B = max_bucket();
    if (B > m.max_T || max_pages < 1 || pad_page < 0) {
        throw std::logic_error("DecodeGraphs: " + std::to_string(B) + " rows do not fit the model's scratch (" +
                               std::to_string(m.max_T) + ")");
    }
    // A blocking stream: it stays ordered with the legacy default stream, which prefill uses.
    CUDA_CHECK(cudaStreamCreate(&stream_));
    if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("DecodeGraphs: cublasCreate failed");
    }
    using core::MemoryOwner;
    cublas_ws_ = memory_.allocate(MemoryOwner::kWorkspace, kCublasWorkspaceBytes, "cuBLAS workspace (CUDA Graph decode)");
    if (cublasSetWorkspace(handle_, cublas_ws_, kCublasWorkspaceBytes) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(handle_, stream_) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("DecodeGraphs: cublasSetWorkspace / cublasSetStream failed");
    }
    const size_t in_ints = static_cast<size_t>(B) * static_cast<size_t>(3 + max_pages);
    CUDA_CHECK(cudaMallocHost(&h_in_, in_ints * sizeof(int)));
    const auto get = [this](size_t bytes) {
        return memory_.allocate(MemoryOwner::kWorkspace, bytes, "CUDA Graph decode buffers");
    };
    d_in_ = static_cast<int*>(get(in_ints * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_next_, static_cast<size_t>(B) * sizeof(int)));
    d_next_ = static_cast<int*>(get(static_cast<size_t>(B) * sizeof(int)));
    d_ws_ = static_cast<float*>(get(attention_decode_fast_workspace_floats(B, m.cfg.n_heads, kMaxSplits) * sizeof(float)));
    d_normed_ = static_cast<__half*>(get(static_cast<size_t>(B) * m.cfg.hidden * sizeof(__half)));
    d_logits_ = static_cast<__half*>(get(static_cast<size_t>(B) * m.vocab * sizeof(__half)));
    // Allocated in the legacy stream's order: done before the graph's own stream first uses them.
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

DecodeGraphs::~DecodeGraphs() {
    for (auto& [key, exec] : execs_) {
        cudaGraphExecDestroy(exec);
    }
    cudaFreeHost(h_in_);
    cudaFreeHost(h_next_);
    if (stream_ != nullptr) {
        cudaStreamSynchronize(stream_);  // nothing of ours still running before its memory goes back
    }
    if (handle_ != nullptr) {
        cublasDestroy(handle_);
    }
    for (void* p : {static_cast<void*>(d_in_), static_cast<void*>(d_next_), static_cast<void*>(d_ws_),
                    static_cast<void*>(d_normed_), static_cast<void*>(d_logits_), cublas_ws_}) {
        try {
            memory_.free(p);  // nullptr is ignored
        } catch (...) {       // a destructor: the manager returns anything left when it goes
        }
    }
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

void DecodeGraphs::enqueue(int bucket, int splits) {
    const int B = max_bucket();
    const int H = m_.cfg.hidden;
    const size_t in_bytes = static_cast<size_t>(B) * static_cast<size_t>(3 + max_pages_) * sizeof(int);
    const int* d_ids = d_in_;
    const int* d_pos = d_in_ + B;
    const int* d_lens = d_in_ + 2 * B;
    const int* d_tables = d_in_ + 3 * B;
    CUDA_CHECK(cudaMemcpyAsync(d_in_, h_in_, in_bytes, cudaMemcpyHostToDevice, stream_));
    embedding_lookup(m_.embed, d_ids, m_.d_x, bucket, H, stream_);
    for (int l = 0; l < m_.n_layers; ++l) {
        layer_forward(handle_, m_.cfg, m_.layers[static_cast<size_t>(l)], m_.scratch, m_.d_x, d_pos, d_tables, d_lens,
                      d_ws_, m_.d_inv_freq, bucket, c_.k_pool[static_cast<size_t>(l)],
                      c_.v_pool[static_cast<size_t>(l)], max_pages_, c_.page_size, splits, stream_);
    }
    rmsnorm(m_.d_x, m_.final_norm, d_normed_, bucket, H, m_.cfg.eps, stream_);
    linear(handle_, d_normed_, m_.embed, d_logits_, bucket, H, m_.vocab, stream_);
    argmax_rows(d_logits_, d_next_, bucket, m_.vocab, stream_);
    CUDA_CHECK(cudaMemcpyAsync(h_next_, d_next_, static_cast<size_t>(bucket) * sizeof(int), cudaMemcpyDeviceToHost,
                               stream_));
}

cudaGraphExec_t DecodeGraphs::graph_for(int bucket, int splits) {
    const int key = bucket * 64 + splits;
    auto it = execs_.find(key);
    if (it != execs_.end()) {
        return it->second;
    }
    // Captured on first use. The caller has just run this (bucket, splits) eagerly, so every kernel
    // is loaded and cuBLAS has chosen its GEMMs for these shapes before the capture.
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    enqueue(bucket, splits);
    CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    execs_.emplace(key, exec);
    return exec;
}

void DecodeGraphs::step(std::vector<Sequence*>& seqs, const std::vector<int>& tokens, std::vector<int>& next,
                        bool graph) {
    const int n = static_cast<int>(seqs.size());
    const int bucket = bucket_for(buckets_, n);
    if (n < 1 || bucket == 0 || static_cast<int>(tokens.size()) != n) {
        throw std::logic_error("DecodeGraphs::step: " + std::to_string(n) + " sequences, buckets up to " +
                               std::to_string(max_bucket()));
    }
    const auto t0 = Clock::now();
    const int B = max_bucket();
    int* ids = h_in_;
    int* pos = h_in_ + B;
    int* lens = h_in_ + 2 * B;
    int* tables = h_in_ + 3 * B;
    int max_len = 1;
    for (int i = 0; i < bucket; ++i) {
        int* row = tables + static_cast<size_t>(i) * max_pages_;
        std::fill(row, row + max_pages_, pad_page_);  // unused entries, and all of a padding row's
        if (i >= n) {  // padding: token 0 at position 0, its K/V into the pad page only
            ids[i] = 0;
            pos[i] = 0;
            lens[i] = 1;
            continue;
        }
        const Sequence& s = *seqs[static_cast<size_t>(i)];
        if (static_cast<int>(s.pages.size()) * c_.page_size < s.len + 1 || static_cast<int>(s.pages.size()) > max_pages_) {
            throw std::logic_error("DecodeGraphs::step: sequence " + std::to_string(i) + " has " +
                                   std::to_string(s.pages.size()) + " pages for position " + std::to_string(s.len));
        }
        std::copy(s.pages.begin(), s.pages.end(), row);
        ids[i] = tokens[static_cast<size_t>(i)];
        pos[i] = s.len;
        lens[i] = s.len + 1;
        max_len = std::max(max_len, lens[i]);
    }
    // The eager path's choice, from the real rows (padding never changes it: docs/cuda-graphs.md).
    const int splits = choose_num_splits(n, m_.cfg.n_kv_heads, max_len);
    const bool captured = execs_.count(bucket * 64 + splits) > 0;
    const auto t1 = Clock::now();
    if (graph && captured) {
        CUDA_CHECK(cudaGraphLaunch(execs_.at(bucket * 64 + splits), stream_));
        ++replays_;
    } else {
        enqueue(bucket, splits);  // eagerly: a first use is also the capture's warm-up
    }
    const auto t2 = Clock::now();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const auto t3 = Clock::now();
    times_.stage_ms += ms_between(t0, t1);
    times_.issue_ms += ms_between(t1, t2);
    times_.wait_ms += ms_between(t2, t3);
    ++times_.steps;
    if (graph && !captured) {
        graph_for(bucket, splits);  // capture after the eager run: replayed from the next step on
    }
    next.assign(h_next_, h_next_ + n);
    for (Sequence* s : seqs) {
        s->len += 1;
    }
}

}  // namespace neuroserve::backend_detail
