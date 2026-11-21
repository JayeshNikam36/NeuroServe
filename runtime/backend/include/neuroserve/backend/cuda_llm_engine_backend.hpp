#pragma once

#include <memory>
#include <string>
#include <string_view>

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

private:
    struct Impl;                  // holds all engine types; defined only in the .cpp
    std::unique_ptr<Impl> impl_;
};

}  // namespace neuroserve