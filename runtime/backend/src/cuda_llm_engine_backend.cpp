#include "neuroserve/backend/cuda_llm_engine_backend.hpp"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// cuda-llm-engine (global namespace). Kept out of every public NeuroServe header.
#include "batched.h"

#include "neuroserve/core/kv_cache_manager.hpp"
#include "neuroserve/cuda/device_info.hpp"

namespace neuroserve {
namespace {

namespace fs = std::filesystem;

constexpr std::size_t kMiB = std::size_t{1} << 20;
// Room for prefill scratch, logits, cuBLAS workspace and fragmentation.
constexpr std::size_t kVramHeadroomBytes = 512 * kMiB;

void validate_config(const CudaLlmEngineConfig& c) {
    const fs::path dir(c.model_dir);
    for (const char* file : {"weights.bin", "index.tsv"}) {
        if (!fs::exists(dir / file)) {
            throw std::invalid_argument("model_dir '" + c.model_dir + "' is missing " + file);
        }
    }
    if (c.page_size_tokens != 16) {
        throw std::invalid_argument("page_size_tokens must be 16 (the fast decode kernel is specialised to 16-token pages)");
    }
    if (c.num_pages <= 0 || c.max_batch <= 0 || c.max_pages_per_sequence <= 0 || c.max_prompt_tokens <= 0) {
        throw std::invalid_argument("num_pages, max_batch, max_pages_per_sequence and max_prompt_tokens must be positive");
    }
    if (c.max_prompt_tokens > c.max_pages_per_sequence * c.page_size_tokens) {
        throw std::invalid_argument("max_prompt_tokens exceeds max_pages_per_sequence * page_size_tokens");
    }
}

// K + V bytes per token for the engine's model shape (FP16).
std::size_t kv_bytes_per_token(const Model& m) {
    return 2 * static_cast<std::size_t>(m.n_layers) * static_cast<std::size_t>(m.cfg.n_kv_heads) *
           static_cast<std::size_t>(m.cfg.head_dim) * sizeof(__half);
}

// First maximum, like the engine and torch.argmax: ties go to the lowest token id.
TokenId argmax_first(const std::vector<float>& logits) {
    return static_cast<TokenId>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

}  // namespace

struct CudaLlmEngineBackend::Impl {
    CudaLlmEngineConfig config;
    ModelInfo info;
    Model model{};
    std::optional<PagedKVCache> cache;   // the GPU pools; its own allocator is emptied and never used
    std::optional<core::KVCacheManager> kv;  // the single owner of KV block accounting (docs/kv-cache.md)
    BatchState batch{};
    bool model_loaded = false;
    bool batch_allocated = false;
    std::unordered_map<SeqId, ::Sequence> sequences;
    SeqId next_id = 1;

    ~Impl() { release(); }

    void release() noexcept {
        if (cache) {
            for (auto& entry : sequences) {
                entry.second.pages.clear();  // the manager's blocks: never returned to the engine's allocator
                ::free_sequence(*cache, entry.second);
            }
        }
        sequences.clear();
        kv.reset();
        if (batch_allocated) {
            free_batch_state(batch);
            batch_allocated = false;
        }
        if (cache) {
            free_paged_cache(*cache);
            cache.reset();
        }
        if (model_loaded) {
            free_model(model);
            model_loaded = false;
        }
    }

    ::Sequence& get(SeqId id) {
        auto it = sequences.find(id);
        if (it == sequences.end()) {
            throw std::invalid_argument("unknown sequence id " + std::to_string(id));
        }
        return it->second;
    }

    const ::Sequence& get(SeqId id) const {
        auto it = sequences.find(id);
        if (it == sequences.end()) {
            throw std::invalid_argument("unknown sequence id " + std::to_string(id));
        }
        return it->second;
    }

    // Gives the engine sequence exactly the manager's blocks, so the engine's own allocation step
    // (reserve / ensure_pages: "take blocks until they cover the length") finds nothing to take.
    void sync_pages(SeqId id, ::Sequence& seq) const {
        const std::vector<core::BlockId> table = kv->table(id);
        seq.pages.assign(table.begin(), table.end());
    }

    // The engine's and the manager's view of a sequence must agree after every call.
    void check_in_sync(SeqId id, const ::Sequence& seq) const {
        if (seq.len != kv->length(id) || seq.pages.size() != kv->table(id).size()) {
            throw std::logic_error("KV state out of sync for sequence " + std::to_string(id) + ": engine has " +
                                   std::to_string(seq.len) + " tokens in " + std::to_string(seq.pages.size()) +
                                   " pages, the manager " + std::to_string(kv->length(id)) + " in " +
                                   std::to_string(kv->table(id).size()));
        }
    }

    void check_token(TokenId t) const {
        if (t < 0 || t >= info.vocab_size) {
            throw std::invalid_argument("token id " + std::to_string(t) + " is outside the vocabulary");
        }
    }
};

CudaLlmEngineBackend::CudaLlmEngineBackend(const CudaLlmEngineConfig& config)
    : impl_(std::make_unique<Impl>()) {
    // Every check runs before the engine is called: the engine calls exit() on failure.
    validate_config(config);
    impl_->config = config;

    const cuda::InitResult init = cuda::initialize_device(config.device_index);

    const Model shape{};  // engine defaults describe Llama-3.2-1B
    const std::size_t weights_bytes = fs::file_size(fs::path(config.model_dir) / "weights.bin");
    const std::size_t kv_bytes = static_cast<std::size_t>(config.num_pages) *
                                 static_cast<std::size_t>(config.page_size_tokens) * kv_bytes_per_token(shape);
    const std::size_t needed = weights_bytes + kv_bytes + kVramHeadroomBytes;
    if (needed > init.info.free_mem_bytes) {
        throw std::runtime_error(
            "not enough free GPU memory: need " + std::to_string(needed / kMiB) + " MiB (weights " +
            std::to_string(weights_bytes / kMiB) + " + KV " + std::to_string(kv_bytes / kMiB) +
            " + headroom " + std::to_string(kVramHeadroomBytes / kMiB) + "), free " +
            std::to_string(init.info.free_mem_bytes / kMiB) + " MiB; lower num_pages");
    }

    impl_->model = load_model(config.model_dir, config.max_prompt_tokens);
    impl_->model_loaded = true;
    impl_->cache.emplace(alloc_paged_cache(impl_->model, config.num_pages, config.page_size_tokens));
    // KVCacheManager decides which block holds what. The engine's allocator is emptied, so a path
    // that still asked it for a block would stop the process loudly instead of handing out a block
    // the manager has already given to another sequence.
    impl_->cache->alloc.free_list.clear();
    impl_->kv.emplace(core::KvBlockConfig{config.num_pages, config.page_size_tokens, config.max_pages_per_sequence});
    impl_->batch = alloc_batch_state(impl_->model, config.max_batch, config.max_pages_per_sequence);
    impl_->batch_allocated = true;

    ModelInfo& info = impl_->info;
    info.name = config.model_name;
    info.n_layers = impl_->model.n_layers;
    info.n_kv_heads = impl_->model.cfg.n_kv_heads;
    info.head_dim = impl_->model.cfg.head_dim;
    info.vocab_size = impl_->model.vocab;
    info.max_prompt_tokens = config.max_prompt_tokens;
    info.kv_bytes_per_token = kv_bytes_per_token(impl_->model);
}

CudaLlmEngineBackend::~CudaLlmEngineBackend() = default;

std::string_view CudaLlmEngineBackend::name() const noexcept { return "cuda-llm-engine"; }

const ModelInfo& CudaLlmEngineBackend::model_info() const noexcept { return impl_->info; }

KvCacheStats CudaLlmEngineBackend::kv_stats() const {
    KvCacheStats s;
    s.page_size_tokens = impl_->config.page_size_tokens;
    s.total_pages = impl_->kv->config().num_blocks;
    s.free_pages = impl_->kv->free_blocks();
    s.max_pages_per_sequence = impl_->config.max_pages_per_sequence;
    return s;
}

SeqId CudaLlmEngineBackend::create_sequence() {
    const SeqId id = impl_->next_id++;
    impl_->kv->create(id);
    impl_->sequences.emplace(id, ::new_sequence(impl_->config.max_pages_per_sequence));
    return id;
}

TokenId CudaLlmEngineBackend::prefill(SeqId id, std::span<const TokenId> prompt) {
    Impl& im = *impl_;
    ::Sequence& seq = im.get(id);
    if (seq.len != 0) {
        throw std::logic_error("prefill: sequence " + std::to_string(id) + " already holds tokens");
    }
    if (prompt.empty()) {
        throw std::invalid_argument("prefill: empty prompt");
    }
    const int n = static_cast<int>(prompt.size());
    if (n > im.config.max_prompt_tokens) {
        throw std::length_error("prefill: prompt has " + std::to_string(n) + " tokens, limit is " +
                                std::to_string(im.config.max_prompt_tokens));
    }
    for (const TokenId t : prompt) {
        im.check_token(t);
    }
    const int pages = im.kv->blocks_needed(id, n);
    if (pages > im.kv->free_blocks()) {
        throw KvCacheExhausted("prefill: needs " + std::to_string(pages) + " KV pages, " +
                               std::to_string(im.kv->free_blocks()) + " free");
    }

    const std::vector<int> ids(prompt.begin(), prompt.end());
    im.kv->append(id, n);  // the blocks for the prompt, from the manager
    im.sync_pages(id, seq);
    // The engine reports GPU errors by exiting (CUDA_CHECK), never by throwing, so nothing below
    // can leave the manager ahead of the engine.
    std::vector<float> logits;
    model_prefill_paged(im.model, *im.cache, seq, ids, logits);
    im.check_in_sync(id, seq);
    return argmax_first(logits);
}

std::vector<TokenId> CudaLlmEngineBackend::decode_step(std::span<const SeqId> ids,
                                                       std::span<const TokenId> tokens) {
    Impl& im = *impl_;
    if (ids.empty()) {
        throw std::invalid_argument("decode_step: no sequences");
    }
    if (ids.size() != tokens.size()) {
        throw std::invalid_argument("decode_step: " + std::to_string(ids.size()) + " sequences but " +
                                    std::to_string(tokens.size()) + " tokens");
    }
    if (ids.size() > static_cast<std::size_t>(im.config.max_batch)) {
        throw std::invalid_argument("decode_step: batch of " + std::to_string(ids.size()) +
                                    " exceeds max_batch " + std::to_string(im.config.max_batch));
    }

    std::vector<::Sequence*> seqs;
    seqs.reserve(ids.size());
    std::unordered_set<SeqId> seen;
    int new_pages = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (!seen.insert(ids[i]).second) {
            throw std::invalid_argument("decode_step: sequence " + std::to_string(ids[i]) + " listed twice");
        }
        ::Sequence& s = im.get(ids[i]);
        if (s.len == 0) {
            throw std::logic_error("decode_step: sequence " + std::to_string(ids[i]) + " has not been prefilled");
        }
        im.check_token(tokens[i]);
        const int needed = im.kv->blocks_needed(ids[i], 1);
        if (static_cast<int>(s.pages.size()) + needed > im.config.max_pages_per_sequence) {
            throw std::length_error("decode_step: sequence " + std::to_string(ids[i]) + " reached its maximum length");
        }
        new_pages += needed;
        seqs.push_back(&s);
    }
    if (new_pages > im.kv->free_blocks()) {
        throw KvCacheExhausted("decode_step: needs " + std::to_string(new_pages) + " new KV pages, " +
                               std::to_string(im.kv->free_blocks()) + " free");
    }

    // Every check has passed: take the new blocks (at most one per sequence) from the manager.
    for (std::size_t i = 0; i < ids.size(); ++i) {
        im.kv->append(ids[i], 1);
        im.sync_pages(ids[i], *seqs[i]);
    }
    const std::vector<int> toks(tokens.begin(), tokens.end());
    std::vector<int> next;
    model_decode_batch_greedy(im.model, *im.cache, im.batch, seqs, toks, next);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        im.check_in_sync(ids[i], *seqs[i]);
    }
    return std::vector<TokenId>(next.begin(), next.end());
}

void CudaLlmEngineBackend::free_sequence(SeqId id) {
    auto it = impl_->sequences.find(id);
    if (it == impl_->sequences.end()) {
        throw std::invalid_argument("unknown sequence id " + std::to_string(id));
    }
    it->second.pages.clear();  // the blocks go back to the manager, not to the engine's allocator
    ::free_sequence(*impl_->cache, it->second);
    impl_->kv->release(id);
    impl_->sequences.erase(it);
}

int CudaLlmEngineBackend::sequence_length(SeqId id) const { return impl_->get(id).len; }

}  // namespace neuroserve