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
#include "neuroserve/core/kv_prefix_hash.hpp"
#include "neuroserve/cuda/device_info.hpp"
#include "prefill_from_offset.cuh"

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

// The K/V identity: everything besides the tokens that decides the K/V values (docs/kv-cache.md).
// cuda-llm-engine computes K/V in FP16 and hard-codes Llama 3's RoPE in load_model
// (engine/model.cu: compute_rope_inv_freq(head_dim, 500000.0, 32.0, 1.0, 4.0, 8192.0)); the values
// here must stay equal to it.
core::KvCacheIdentity kv_identity_for(const CudaLlmEngineConfig& c) {
    const fs::path dir(c.model_dir);
    core::KvCacheIdentity id;
    id.model_id = c.model_name;
    id.weights_digest = core::file_fingerprint((dir / "weights.bin").string()) + ":" +
                        core::file_fingerprint((dir / "index.tsv").string());
    id.dtype = "fp16";
    id.block_size = c.page_size_tokens;
    id.rope = {500000.0, "llama3", 32.0, 1.0, 4.0, 8192};
    return id;
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
    std::optional<core::PrefixHasher> hasher;  // the K/V identity and the root of every hash chain
    bool model_loaded = false;
    bool batch_allocated = false;
    std::unordered_map<SeqId, ::Sequence> sequences;
    std::unordered_map<SeqId, std::vector<TokenId>> tokens;  // with prefix caching: each sequence's tokens
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
        tokens.clear();
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
    impl_->hasher.emplace(kv_identity_for(config));  // reads about 4 MiB of the weights; may throw

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
    impl_->kv.emplace(core::KvBlockConfig{config.num_pages, config.page_size_tokens, config.max_pages_per_sequence},
                      config.prefix_caching ? std::optional(impl_->hasher->identity()) : std::nullopt);
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
    info.kv_cache_key = core::hash_hex(impl_->hasher->root());
}

CudaLlmEngineBackend::~CudaLlmEngineBackend() = default;

std::string_view CudaLlmEngineBackend::name() const noexcept { return "cuda-llm-engine"; }

const ModelInfo& CudaLlmEngineBackend::model_info() const noexcept { return impl_->info; }

const core::KvCacheIdentity& CudaLlmEngineBackend::kv_identity() const noexcept {
    return impl_->hasher->identity();
}

KvCacheStats CudaLlmEngineBackend::kv_stats() const {
    KvCacheStats s;
    s.page_size_tokens = impl_->config.page_size_tokens;
    s.total_pages = impl_->kv->config().num_blocks;
    s.free_pages = impl_->kv->free_blocks();
    s.max_pages_per_sequence = impl_->config.max_pages_per_sequence;
    const core::KvBlockStats kv = impl_->kv->stats();
    s.shared_pages = kv.shared_blocks;
    s.indexed_pages = kv.indexed_blocks;  // 0: prefix caching starts with prefill from a cached offset
    s.prefix_lookups = kv.prefix_lookups;
    s.prefix_lookup_pages = kv.prefix_lookup_blocks;
    s.prefix_hit_pages = kv.prefix_hit_blocks;
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
    const std::vector<int> ids(prompt.begin(), prompt.end());
    int cached = 0;
    if (im.kv->prefix_caching()) {
        const core::Admission a = im.kv->admit(id, prompt);  // reuses indexed full pages, takes the rest;
                                                             // throws, changing nothing, if pages run out
        cached = a.cached_tokens;
        if (a.copy) {  // copy-on-write: the prefill will write the last token into this page
            backend_detail::copy_page(im.model, *im.cache, a.copy->from, a.copy->to);
        }
    } else {
        const int pages = im.kv->blocks_needed(id, n);
        if (pages > im.kv->free_blocks()) {
            throw KvCacheExhausted("prefill: needs " + std::to_string(pages) + " KV pages, " +
                                   std::to_string(im.kv->free_blocks()) + " free");
        }
        im.kv->append(id, n);  // the blocks for the prompt, from the manager
    }
    im.sync_pages(id, seq);
    // The engine reports GPU errors by exiting (CUDA_CHECK), never by throwing, so nothing below
    // can leave the manager ahead of the engine.
    std::vector<float> logits;
    if (cached == 0) {
        model_prefill_paged(im.model, *im.cache, seq, ids, logits);
    } else {
        backend_detail::model_prefill_from(im.model, *im.cache, seq, ids, cached, logits);
    }
    im.check_in_sync(id, seq);
    if (im.kv->prefix_caching()) {
        std::vector<TokenId>& held = im.tokens[id];
        held.assign(prompt.begin(), prompt.end());
        im.kv->commit(id, held);  // the prompt's full pages are written: index them
    }
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
        if (im.kv->prefix_caching()) {
            std::vector<TokenId>& held = im.tokens[ids[i]];
            held.push_back(tokens[i]);
            im.kv->commit(ids[i], held);  // indexes the page this token filled, if any
        }
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
    impl_->tokens.erase(id);
    impl_->sequences.erase(it);
}

int CudaLlmEngineBackend::sequence_length(SeqId id) const { return impl_->get(id).len; }

void CudaLlmEngineBackend::check_kv_invariants() const {
    const Impl& im = *impl_;
    im.kv->check_invariants();  // I1-I6 inside the manager
    if (!im.cache->alloc.free_list.empty()) {
        throw core::KvInvariantViolation("the engine's own allocator holds " +
                                         std::to_string(im.cache->alloc.free_list.size()) +
                                         " pages: blocks were returned to it instead of to the manager");
    }
    if (im.kv->stats().sequences != static_cast<int>(im.sequences.size())) {
        throw core::KvInvariantViolation("I12: the manager holds " + std::to_string(im.kv->stats().sequences) +
                                         " sequences, the engine " + std::to_string(im.sequences.size()));
    }
    for (const auto& [id, seq] : im.sequences) {
        if (!im.kv->contains(id)) {
            throw core::KvInvariantViolation("I12: the engine holds sequence " + std::to_string(id) +
                                             ", the manager does not");
        }
        const std::vector<core::BlockId> table = im.kv->table(id);
        if (seq.len != im.kv->length(id) || !std::equal(seq.pages.begin(), seq.pages.end(), table.begin(), table.end())) {
            throw core::KvInvariantViolation("sequence " + std::to_string(id) + ": the engine has " +
                                             std::to_string(seq.len) + " tokens in " + std::to_string(seq.pages.size()) +
                                             " pages, the manager " + std::to_string(im.kv->length(id)) + " in " +
                                             std::to_string(table.size()) + " (or other blocks)");
        }
    }
}

int CudaLlmEngineBackend::live_sequences() const { return static_cast<int>(impl_->sequences.size()); }

void CudaLlmEngineBackend::share_prefix(SeqId dst, SeqId src, int pages) {
    Impl& im = *impl_;
    ::Sequence& to = im.get(dst);
    im.get(src);                          // both must exist
    im.kv->share_prefix(dst, src, pages);  // validates (empty dst, full pages) and counts the references
    im.sync_pages(dst, to);
    to.len = pages * im.config.page_size_tokens;  // the shared positions are cached: decode continues after them
    im.check_in_sync(dst, to);
    if (im.kv->prefix_caching()) {
        const std::vector<TokenId>& from = im.tokens[src];
        im.tokens[dst].assign(from.begin(), from.begin() + std::min<std::ptrdiff_t>(to.len, static_cast<std::ptrdiff_t>(from.size())));
    }
}

SequencePages CudaLlmEngineBackend::sequence_pages(SeqId id) const {
    const Impl& im = *impl_;
    im.get(id);
    SequencePages out;
    for (const core::BlockId block : im.kv->table(id)) {
        ++out.held;
        out.exclusive += im.kv->ref_count(block) == 1;
    }
    return out;
}

}  // namespace neuroserve