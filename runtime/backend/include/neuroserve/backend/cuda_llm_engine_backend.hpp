#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "neuroserve/backend/gpu_memory.hpp"
#include "neuroserve/core/inference_backend.hpp"
#include "neuroserve/core/kv_prefix_hash.hpp"

namespace neuroserve {

struct CudaLlmEngineConfig {
    std::string model_dir;                  // directory with weights.bin and index.tsv
    std::string model_name = "llama-3.2-1b";
    int device_index = 0;
    int max_prompt_tokens = 512;            // prefill scratch size
    int page_size_tokens = 16;              // must be 16: the fast decode kernel is specialised to it
    int num_pages = 2048;                   // 2048 pages x 16 tokens x 32 KiB = 1 GiB for Llama-3.2-1B
    int max_batch = 8;                      // max sequences per decode_step
    int max_pages_per_sequence = 64;        // 64 x 16 = 1024 tokens per sequence
    // Prefix caching (docs/kv-cache.md): a prompt reuses the full pages of live sequences with the
    // same prefix, and prefill starts at the first uncached token. Off by default until the
    // shared-prefix end-to-end test (Day 36) turns it on for the server.
    bool prefix_caching = false;
    // CUDA Graphs for decode (docs/cuda-graphs.md): a decode-only step of up to 16 sequences replays
    // a graph captured for its batch bucket (1, 2, 4, 8, 16) and split count. Off by default until
    // the Day 57 A/B.
    bool cuda_graphs = false;
    // GPU memory manager (Day 58, docs/gpu-memory.md): freed pool memory above this is returned to
    // the driver at the next synchronization.
    std::size_t memory_release_threshold_bytes = std::size_t{64} << 20;
};

// InferenceBackend implemented on top of cuda-llm-engine (paged KV cache, split-KV decode attention).
class CudaLlmEngineBackend final : public InferenceBackend {
public:
    explicit CudaLlmEngineBackend(const CudaLlmEngineConfig& config);
    ~CudaLlmEngineBackend() override;

    std::string_view name() const noexcept override;
    const ModelInfo& model_info() const noexcept override;
    KvCacheStats kv_stats() const override;

    SeqId create_sequence() override;
    TokenId prefill(SeqId seq, std::span<const TokenId> prompt) override;
    // Prefills the items whose new tokens fit the scratch together, in one forward pass with the
    // varlen tiled attention (docs/prefill-attention.md); a lone item, or one longer than a chunk,
    // goes through prefill()'s path. Each item succeeds or fails on its own.
    void prefill_batch(std::span<PrefillItem> items) override;
    // Chunked prefill (Day 52): the next part of a partly prefilled sequence, from its cached
    // offset (prefill from an offset, docs/kv-cache.md). In prefill_batch, such items are packed too.
    TokenId prefill_more(SeqId seq, std::span<const TokenId> input) override;
    bool supports_chunked_prefill() const override { return true; }
    // Mixed steps (Day 53): the prefills packed with the decode rows in one forward pass, the
    // prefill segments' attention through the varlen tiled kernel and the decode rows' through a
    // row-per-sequence kernel. Items that do not fit the pass are prefilled alone first.
    void mixed_step(std::span<PrefillItem> items, DecodeBatch& decode) override;
    bool supports_mixed_steps() const override { return true; }
    std::vector<TokenId> decode_step(std::span<const SeqId> seqs,
                                     std::span<const TokenId> tokens) override;
    void free_sequence(SeqId seq) override;
    int sequence_length(SeqId seq) const override;
    void check_kv_invariants() const override;
    int live_sequences() const override;
    void share_prefix(SeqId dst, SeqId src, int pages) override;
    SequencePages sequence_pages(SeqId seq) const override;

    // What the K/V values depend on besides the tokens: the root of every prefix hash chain
    // (docs/kv-cache.md, "Hashing full blocks"). model_info().kv_cache_key is its hash.
    const core::KvCacheIdentity& kv_identity() const noexcept;

    // Prefix caching on or off (CudaLlmEngineConfig::prefix_caching is the start value). Only with
    // no live sequences: std::logic_error otherwise. Lets one backend compare both, batch for batch.
    // Evicts every CACHED page, so either setting starts from an empty cache.
    void set_prefix_caching(bool on);
    bool prefix_caching() const noexcept;

    // Forgets every CACHED page (released, still indexed); live sequences are untouched. Returns
    // how many pages were evicted.
    int evict_cached_pages();

    // Prefill() takes up to a sequence's capacity (max_pages_per_sequence * page_size_tokens) and
    // computes it in chunks of at most max_prompt_tokens (the scratch size), each chunk attending to
    // all before it: a resumed request's prompt + generated tokens can exceed max_prompt_tokens.
    // Fewer tokens per chunk (1 .. max_prompt_tokens) let tests run the chunked path on short
    // prompts; std::invalid_argument outside that range.
    void set_prefill_chunk_tokens(int tokens);

    // Batched prefill so far: forward passes that prefilled two or more sequences together, and the
    // sequences they prefilled. For tests and metrics.
    std::uint64_t batched_prefills() const noexcept;
    std::uint64_t batched_prefill_sequences() const noexcept;
    std::uint64_t mixed_passes() const noexcept;

    // How a decode step runs (docs/cuda-graphs.md). kEager: the engine's batched decode. kGraph:
    // a step of at most 16 sequences replays a captured graph (captured on its first use).
    // kGraphEager: the graph's work without the graph (same padding, buffers and stream), to compare
    // a replay with bit for bit. Switching is allowed at any time.
    enum class DecodePath { kEager, kGraphEager, kGraph };
    void set_decode_path(DecodePath path);
    DecodePath decode_path() const noexcept;
    int cuda_graphs_captured() const noexcept;        // (bucket, splits) graphs instantiated so far
    std::uint64_t cuda_graph_replays() const noexcept;  // both safe to read from any thread

    // Host time of the decode steps that took a graph path (kGraph or kGraphEager) since the last
    // reset, in milliseconds (Day 57): writing the inputs, issuing the GPU work, waiting for it.
    // Read and reset from the thread that decodes.
    struct DecodeHostTimes {
        double stage_ms = 0, issue_ms = 0, wait_ms = 0;
        std::uint64_t steps = 0;
    };
    DecodeHostTimes decode_host_times() const;
    void reset_decode_host_times();

    // GPU memory by owner (Day 58, docs/gpu-memory.md): NeuroServe's own allocations through the
    // pool, cuda-llm-engine's recorded with their owners, the device's used memory beside them.
    // Safe from any thread.
    GpuMemoryManager::Report memory_report() const;  // mixed steps run as one forward pass (Day 53)

private:
    struct Impl;                  // holds all engine types; defined only in the .cpp
    std::unique_ptr<Impl> impl_;
};

}  // namespace neuroserve