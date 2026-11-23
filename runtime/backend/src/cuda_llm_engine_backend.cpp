#include "neuroserve/backend/cuda_llm_engine_backend.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <filesystem>
#include <iterator>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// cuda-llm-engine (global namespace). Kept out of every public NeuroServe header.
#include "batched.h"

#include "neuroserve/core/kv_cache_manager.hpp"
#include "neuroserve/core/kv_memory.hpp"
#include "neuroserve/core/kv_prefix_hash.hpp"
#include "neuroserve/cuda/device_info.hpp"
#include "decode_graph.cuh"
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
    bool prefix_caching = false;
    int prefill_chunk = 0;  // tokens per prefill chunk: max_prompt_tokens unless a test sets fewer
    SeqId next_id = 1;
    backend_detail::PrefillBatchScratch prefill_scratch{};  // batched prefill (Day 51)
    bool prefill_scratch_allocated = false;
    std::uint64_t mixed_passes = 0;  // mixed steps run as one forward pass (Day 53)
    // CUDA Graphs for decode (Day 56), made on first use. The KV pool has one page more than the
    // manager knows of: the pad page padding rows write to.
    std::optional<backend_detail::DecodeGraphs> graphs;
    std::atomic<int> graphs_captured{0};  // copies of graphs' counters, for readers on other threads
    std::atomic<std::uint64_t> graph_replays{0};
    CudaLlmEngineBackend::DecodePath decode_path = CudaLlmEngineBackend::DecodePath::kEager;
    std::uint64_t batched_prefills = 0;
    std::uint64_t batched_prefill_sequences = 0;

    ~Impl() { release(); }

    void release() noexcept {
        if (cache) {
            for (auto& entry : sequences) {
                entry.second.pages.clear();  // the manager's blocks: never returned to the engine's allocator
                ::free_sequence(*cache, entry.second);
            }
        }
        graphs.reset();  // before the model and the cache it points into
        sequences.clear();
        tokens.clear();
        kv.reset();
        if (batch_allocated) {
            free_batch_state(batch);
            batch_allocated = false;
        }
        if (prefill_scratch_allocated) {
            backend_detail::free_prefill_batch_scratch(prefill_scratch);
            prefill_scratch_allocated = false;
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

    // prefill()'s checks: an empty sequence, a non-empty prompt that fits, valid tokens.
    ::Sequence& checked_for_prefill(SeqId id, std::span<const TokenId> prompt) {
        ::Sequence& seq = get(id);
        if (seq.len != 0) {
            throw std::logic_error("prefill: sequence " + std::to_string(id) + " already holds tokens");
        }
        if (prompt.empty()) {
            throw std::invalid_argument("prefill: empty prompt");
        }
        const int n = static_cast<int>(prompt.size());
        const int capacity = config.max_pages_per_sequence * config.page_size_tokens;
        if (n > capacity) {  // longer than max_prompt_tokens is fine: it is prefilled in chunks
            throw std::length_error("prefill: " + std::to_string(n) + " tokens, a sequence holds at most " +
                                    std::to_string(capacity));
        }
        for (const TokenId t : prompt) {
            check_token(t);
        }
        return seq;
    }

    // Gives the sequence its blocks: reused prefix blocks and new ones with prefix caching, all new
    // without. Returns how many leading tokens are cached. Throws, changing nothing, if blocks run out.
    int admit(SeqId id, ::Sequence& seq, std::span<const TokenId> prompt) {
        const int n = static_cast<int>(prompt.size());
        int cached = 0;
        if (prefix_caching) {
            const core::Admission a = kv->admit(id, prompt);  // reuses indexed full pages, takes the rest;
                                                              // throws, changing nothing, if pages run out
            cached = a.cached_tokens;
            if (a.copy) {  // copy-on-write: the prefill will write the last token into this page
                backend_detail::copy_page(model, *cache, a.copy->from, a.copy->to);
            }
        } else {
            const int pages = kv->blocks_needed(id, n);
            if (pages > kv->free_blocks()) {
                throw KvCacheExhausted("prefill: needs " + std::to_string(pages) + " KV pages, " +
                                       std::to_string(kv->free_blocks()) + " free");
            }
            kv->append(id, n);  // the blocks for the prompt, from the manager
        }
        sync_pages(id, seq);
        return cached;
    }

    // prefill_more()'s checks: a partly prefilled sequence and an input that extends what it holds.
    ::Sequence& checked_for_more(SeqId id, std::span<const TokenId> input) {
        ::Sequence& seq = get(id);
        const int n = static_cast<int>(input.size());
        if (seq.len == 0 || n <= seq.len) {
            throw std::logic_error("prefill_more: sequence " + std::to_string(id) + " holds " + std::to_string(seq.len) +
                                   " tokens; the input must extend them (" + std::to_string(n) + " given)");
        }
        const int capacity = config.max_pages_per_sequence * config.page_size_tokens;
        if (n > capacity) {
            throw std::length_error("prefill_more: " + std::to_string(n) + " tokens, a sequence holds at most " +
                                    std::to_string(capacity));
        }
        if (prefix_caching) {  // the tokens it holds are known: the input must start with them
            const std::vector<TokenId>& held = tokens[id];
            if (held.size() != static_cast<std::size_t>(seq.len) || !std::equal(held.begin(), held.end(), input.begin())) {
                throw std::logic_error("prefill_more: the input does not extend sequence " + std::to_string(id) +
                                       "'s tokens");
            }
        }
        for (std::size_t i = static_cast<std::size_t>(seq.len); i < input.size(); ++i) {
            check_token(input[i]);
        }
        return seq;
    }

    // The blocks for the new tokens of a partly prefilled sequence. Returns the tokens it already
    // holds (the new part's start). Throws, changing nothing, if blocks run out.
    int admit_more(SeqId id, ::Sequence& seq, std::span<const TokenId> input) {
        const int extra = static_cast<int>(input.size()) - seq.len;
        const int pages = kv->blocks_needed(id, extra);
        if (pages > kv->free_blocks()) {
            throw KvCacheExhausted("prefill_more: needs " + std::to_string(pages) + " KV pages, " +
                                   std::to_string(kv->free_blocks()) + " free");
        }
        if (static_cast<int>(seq.pages.size()) + pages > config.max_pages_per_sequence) {
            throw std::length_error("prefill_more: sequence " + std::to_string(id) + " would exceed its maximum length");
        }
        kv->append(id, extra);
        sync_pages(id, seq);
        return seq.len;
    }

    // After the K/V are written: the engine and the manager agree, and the prompt's full pages are
    // indexed for later hits.
    void finish_prefill(SeqId id, const ::Sequence& seq, std::span<const TokenId> prompt) {
        check_in_sync(id, seq);
        if (prefix_caching) {
            std::vector<TokenId>& held = tokens[id];
            held.assign(prompt.begin(), prompt.end());
            kv->commit(id, held);
        }
    }

    // decode_step()'s checks, which change nothing: returns the sequences and, in `new_pages`, the
    // blocks their new tokens need (at most one each).
    std::vector<::Sequence*> checked_for_decode(std::span<const SeqId> ids, std::span<const TokenId> toks,
                                                int& new_pages) {
        if (ids.empty()) {
            throw std::invalid_argument("decode_step: no sequences");
        }
        if (ids.size() != toks.size()) {
            throw std::invalid_argument("decode_step: " + std::to_string(ids.size()) + " sequences but " +
                                        std::to_string(toks.size()) + " tokens");
        }
        if (ids.size() > static_cast<std::size_t>(config.max_batch)) {
            throw std::invalid_argument("decode_step: batch of " + std::to_string(ids.size()) +
                                        " exceeds max_batch " + std::to_string(config.max_batch));
        }
        std::vector<::Sequence*> seqs;
        seqs.reserve(ids.size());
        std::unordered_set<SeqId> seen;
        new_pages = 0;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (!seen.insert(ids[i]).second) {
                throw std::invalid_argument("decode_step: sequence " + std::to_string(ids[i]) + " listed twice");
            }
            ::Sequence& s = get(ids[i]);
            if (s.len == 0) {
                throw std::logic_error("decode_step: sequence " + std::to_string(ids[i]) + " has not been prefilled");
            }
            check_token(toks[i]);
            const int needed = kv->blocks_needed(ids[i], 1);
            if (static_cast<int>(s.pages.size()) + needed > config.max_pages_per_sequence) {
                throw std::length_error("decode_step: sequence " + std::to_string(ids[i]) +
                                        " reached its maximum length");
            }
            new_pages += needed;
            seqs.push_back(&s);
        }
        return seqs;
    }

    // The decode's new blocks (at most one per sequence), from the manager.
    void append_for_decode(std::span<const SeqId> ids, const std::vector<::Sequence*>& seqs) {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            kv->append(ids[i], 1);
            sync_pages(ids[i], *seqs[i]);
        }
    }

    // After the decode's K/V are written: the engine and the manager agree, and the fed-back tokens
    // are indexed for prefix caching.
    void finish_decode(std::span<const SeqId> ids, const std::vector<::Sequence*>& seqs,
                       std::span<const TokenId> toks) {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            check_in_sync(ids[i], *seqs[i]);
            if (prefix_caching) {
                std::vector<TokenId>& held = tokens[ids[i]];
                held.push_back(toks[i]);
                kv->commit(ids[i], held);  // indexes the page this token filled, if any
            }
        }
    }

    // A prefill item that passed its checks and has its blocks: `cached` tokens of it are already in
    // its pages (a prefix hit, or earlier chunks).
    struct Admitted {
        PrefillItem* item;
        ::Sequence* seq;
        int cached;
    };

    // Every item's checks and blocks, in order; an item that fails them fails alone (`error` set).
    std::vector<Admitted> admit_items(std::span<PrefillItem> items) {
        std::vector<Admitted> admitted;
        for (PrefillItem& item : items) {
            item.error = nullptr;
            try {
                if (get(item.seq).len == 0) {  // a new prefill
                    ::Sequence& seq = checked_for_prefill(item.seq, item.prompt);
                    admitted.push_back({&item, &seq, admit(item.seq, seq, item.prompt)});
                } else {  // the next chunk of one (chunked prefill)
                    ::Sequence& seq = checked_for_more(item.seq, item.prompt);
                    admitted.push_back({&item, &seq, admit_more(item.seq, seq, item.prompt)});
                }
            } catch (...) {
                item.error = std::current_exception();
            }
        }
        return admitted;
    }

    // Which admitted items go in one pass: those whose new tokens fit one chunk, while the pack fits
    // `tokens` rows and `seqs` sequences. The rest go through prefill()'s path, one by one.
    void split_pack(std::vector<Admitted>& admitted, int tokens, int seqs, std::vector<Admitted*>& pack,
                    std::vector<Admitted*>& alone) {
        int packed_tokens = 0;
        for (Admitted& a : admitted) {
            const int fresh = static_cast<int>(a.item->prompt.size()) - a.cached;
            if (fresh <= prefill_chunk && packed_tokens + fresh <= tokens && static_cast<int>(pack.size()) < seqs) {
                pack.push_back(&a);
                packed_tokens += fresh;
            } else {
                alone.push_back(&a);
            }
        }
    }

    void run_alone(const std::vector<Admitted*>& alone) {
        for (Admitted* a : alone) {
            try {
                a->item->first = run_chunked(a->item->seq, *a->seq, a->item->prompt, a->cached);
            } catch (...) {
                a->item->error = std::current_exception();
            }
        }
    }

    // prefill_batch() after the admission: what fits one pass together, the rest (and a pack of one)
    // through prefill()'s path.
    void prefill_admitted(std::vector<Admitted>& admitted) {
        std::vector<Admitted*> pack;
        std::vector<Admitted*> alone;
        split_pack(admitted, config.max_prompt_tokens, prefill_scratch.max_seqs, pack, alone);
        if (pack.size() == 1) {
            alone.insert(alone.begin(), pack.front());
            pack.clear();
        }
        if (!pack.empty()) {
            run_pack(pack, {});
        }
        run_alone(alone);
    }

    // One forward pass over the packed items and the decode rows (none, for a batched prefill).
    // Returns the decode rows' next tokens.
    std::vector<int> run_pack(const std::vector<Admitted*>& pack,
                              const std::vector<backend_detail::DecodeRow>& decodes) {
        std::vector<std::vector<int>> ids;
        ids.reserve(pack.size());
        std::vector<backend_detail::PrefillPart> parts;
        for (Admitted* a : pack) {
            ids.emplace_back(a->item->prompt.begin(), a->item->prompt.end());
        }
        for (std::size_t i = 0; i < pack.size(); ++i) {
            parts.push_back({pack[i]->seq, &ids[i], pack[i]->cached});
        }
        const std::vector<int> next =
            backend_detail::model_mixed_batch(model, *cache, parts, decodes, prefill_scratch);
        for (std::size_t i = 0; i < pack.size(); ++i) {
            PrefillItem& item = *pack[i]->item;
            try {
                finish_prefill(item.seq, *pack[i]->seq, item.prompt);
                item.first = next[i];
            } catch (...) {
                item.error = std::current_exception();
            }
        }
        if (!pack.empty()) {
            ++batched_prefills;
            batched_prefill_sequences += pack.size();
        }
        return std::vector<int>(next.begin() + static_cast<long>(pack.size()), next.end());
    }

    // One greedy decode step of prefilled sequences whose pages cover the new token, by decode_path.
    void run_decode(std::vector<::Sequence*>& seqs, std::span<const TokenId> toks, std::vector<int>& next) {
        const std::vector<int> ids(toks.begin(), toks.end());
        if (decode_path != CudaLlmEngineBackend::DecodePath::kEager) {
            if (!graphs) {
                graphs.emplace(model, *cache, config.max_batch, config.max_pages_per_sequence, config.num_pages);
            }
            if (static_cast<int>(seqs.size()) <= graphs->max_bucket()) {
                graphs->step(seqs, ids, next, decode_path == CudaLlmEngineBackend::DecodePath::kGraph);
                graphs_captured.store(graphs->graphs(), std::memory_order_relaxed);
                graph_replays.store(graphs->replays(), std::memory_order_relaxed);
                return;
            }
        }
        model_decode_batch_greedy(model, *cache, batch, seqs, ids, next);
    }

    // prefill()'s computation for one admitted sequence, in chunks of at most prefill_chunk tokens
    // (the scratch holds max_prompt_tokens): a client's prompt is one chunk; a resumed request's
    // prompt + generated tokens can be several (Day 38).
    TokenId run_chunked(SeqId id, ::Sequence& seq, std::span<const TokenId> prompt, int cached) {
        const std::vector<int> ids(prompt.begin(), prompt.end());
        const int n = static_cast<int>(ids.size());
        // The engine reports GPU errors by exiting (CUDA_CHECK), never by throwing, so nothing below
        // can leave the manager ahead of the engine.
        std::vector<float> logits;
        const int chunk = prefill_chunk;
        int done = cached;
        if (done == 0) {
            const int first = std::min(n, chunk);
            if (first == n) {
                model_prefill_paged(model, *cache, seq, ids, logits);
            } else {
                model_prefill_paged(model, *cache, seq, std::vector<int>(ids.begin(), ids.begin() + first), logits);
            }
            done = first;
        }
        while (done < n) {  // each later chunk attends to everything before it, as a cached prefix
            const int end = std::min(n, done + chunk);
            backend_detail::model_prefill_from(model, *cache, seq,
                                               end == n ? ids : std::vector<int>(ids.begin(), ids.begin() + end), done,
                                               logits);
            done = end;
        }
        finish_prefill(id, seq, prompt);
        return argmax_first(logits);
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
    const std::size_t kv_bytes = static_cast<std::size_t>(config.num_pages + 1) *  // + the pad page
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
    // One page more than the manager's: the pad page of CUDA-Graph decode (page num_pages, never handed out).
    impl_->cache.emplace(alloc_paged_cache(impl_->model, config.num_pages + 1, config.page_size_tokens));
    // KVCacheManager decides which block holds what. The engine's allocator is emptied, so a path
    // that still asked it for a block would stop the process loudly instead of handing out a block
    // the manager has already given to another sequence.
    impl_->cache->alloc.free_list.clear();
    impl_->kv.emplace(core::KvBlockConfig{config.num_pages, config.page_size_tokens, config.max_pages_per_sequence},
                      impl_->hasher->identity());  // the index is there; prefix_caching decides if it is used
    impl_->prefix_caching = config.prefix_caching;
    impl_->decode_path = config.cuda_graphs ? DecodePath::kGraph : DecodePath::kEager;
    impl_->prefill_chunk = config.max_prompt_tokens;
    impl_->batch = alloc_batch_state(impl_->model, config.max_batch, config.max_pages_per_sequence);
    impl_->batch_allocated = true;
    impl_->prefill_scratch =
        backend_detail::alloc_prefill_batch_scratch(impl_->model, config.max_batch, config.max_pages_per_sequence);
    impl_->prefill_scratch_allocated = true;

    ModelInfo& info = impl_->info;
    info.name = config.model_name;
    info.n_layers = impl_->model.n_layers;
    info.n_kv_heads = impl_->model.cfg.n_kv_heads;
    info.head_dim = impl_->model.cfg.head_dim;
    info.vocab_size = impl_->model.vocab;
    info.max_prompt_tokens = config.max_prompt_tokens;
    info.kv_bytes_per_token = kv_bytes_per_token(impl_->model);
    // The model directory's config.json, when the export kept it, must describe the weights just
    // loaded: the KV memory per token (Day 39) comes from the same numbers either way.
    const fs::path config_json = fs::path(config.model_dir) / "config.json";
    if (fs::exists(config_json)) {
        std::ifstream in(config_json);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const core::ModelShape declared = core::parse_model_config(text);  // std::invalid_argument if broken
        const std::int64_t from_config = core::kv_bytes_per_token(declared, "fp16");
        if (declared.num_layers != impl_->model.n_layers || declared.num_kv_heads != impl_->model.cfg.n_kv_heads ||
            declared.head_dim != impl_->model.cfg.head_dim ||
            from_config != static_cast<std::int64_t>(info.kv_bytes_per_token)) {
            throw std::invalid_argument("config.json describes another model than the weights in " + config.model_dir +
                                        " (" + std::to_string(from_config) + " KV bytes per token vs " +
                                        std::to_string(info.kv_bytes_per_token) + ")");
        }
    }
    info.kv_cache_key = core::hash_hex(impl_->hasher->root());
}

CudaLlmEngineBackend::~CudaLlmEngineBackend() = default;

std::string_view CudaLlmEngineBackend::name() const noexcept { return "cuda-llm-engine"; }

const ModelInfo& CudaLlmEngineBackend::model_info() const noexcept { return impl_->info; }

void CudaLlmEngineBackend::set_prefix_caching(bool on) {
    if (!impl_->sequences.empty()) {
        throw std::logic_error("set_prefix_caching: only with no live sequences");
    }
    impl_->kv->evict_cached();  // what was cached under the old setting is forgotten
    impl_->prefix_caching = on;
}

int CudaLlmEngineBackend::evict_cached_pages() { return impl_->kv->evict_cached(); }

void CudaLlmEngineBackend::set_prefill_chunk_tokens(int tokens) {
    if (tokens < 1 || tokens > impl_->config.max_prompt_tokens) {
        throw std::invalid_argument("prefill chunk must be 1 .. max_prompt_tokens (" +
                                    std::to_string(impl_->config.max_prompt_tokens) + ") tokens");
    }
    impl_->prefill_chunk = tokens;
}

bool CudaLlmEngineBackend::prefix_caching() const noexcept { return impl_->prefix_caching; }

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
    s.cached_pages = kv.cached_blocks;
    s.evicted_pages = kv.evicted_blocks;
    s.prefix_reused_tokens = kv.prefix_reused_tokens;
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
    ::Sequence& seq = im.checked_for_prefill(id, prompt);
    const int cached = im.admit(id, seq, prompt);
    return im.run_chunked(id, seq, prompt, cached);
}

TokenId CudaLlmEngineBackend::prefill_more(SeqId id, std::span<const TokenId> input) {
    Impl& im = *impl_;
    ::Sequence& seq = im.checked_for_more(id, input);
    const int start = im.admit_more(id, seq, input);
    return im.run_chunked(id, seq, input, start);
}

void CudaLlmEngineBackend::prefill_batch(std::span<PrefillItem> items) {
    Impl& im = *impl_;
    std::vector<Impl::Admitted> admitted = im.admit_items(items);
    im.prefill_admitted(admitted);
}

void CudaLlmEngineBackend::mixed_step(std::span<PrefillItem> items, DecodeBatch& decode) {
    Impl& im = *impl_;
    if (items.empty() || decode.ids.empty()) {
        InferenceBackend::mixed_step(items, decode);  // nothing to mix
        return;
    }
    decode.error = nullptr;
    decode.next.clear();
    // The decode's checks first: they change nothing. If they fail, the prefills still run.
    std::vector<::Sequence*> seqs;
    int new_pages = 0;
    try {
        seqs = im.checked_for_decode(decode.ids, decode.tokens, new_pages);
        for (const PrefillItem& item : items) {
            if (std::find(decode.ids.begin(), decode.ids.end(), item.seq) != decode.ids.end()) {
                throw std::invalid_argument("mixed_step: sequence " + std::to_string(item.seq) +
                                            " is both prefilled and decoded");
            }
        }
    } catch (...) {
        decode.error = std::current_exception();
        prefill_batch(items);
        return;
    }
    std::vector<Impl::Admitted> admitted = im.admit_items(items);
    if (new_pages > im.kv->free_blocks()) {  // the prefills took the pages the decode needed
        decode.error = std::make_exception_ptr(KvCacheExhausted(
            "decode_step: needs " + std::to_string(new_pages) + " new KV pages, " +
            std::to_string(im.kv->free_blocks()) + " free"));
        im.prefill_admitted(admitted);
        return;
    }
    im.append_for_decode(decode.ids, seqs);
    // The pack shares the scratch with the decode rows: one row and one sequence each.
    const int rows = static_cast<int>(seqs.size());
    std::vector<Impl::Admitted*> pack;
    std::vector<Impl::Admitted*> alone;
    im.split_pack(admitted, im.config.max_prompt_tokens - rows, im.prefill_scratch.max_seqs - rows, pack, alone);
    im.run_alone(alone);  // what does not fit the pass, before it
    std::vector<int> next;
    if (pack.empty()) {
        im.run_decode(seqs, decode.tokens, next);
    } else {
        std::vector<backend_detail::DecodeRow> rows_in;
        for (std::size_t i = 0; i < seqs.size(); ++i) {
            rows_in.push_back({seqs[i], decode.tokens[i]});
        }
        next = im.run_pack(pack, rows_in);
        ++im.mixed_passes;
    }
    im.finish_decode(decode.ids, seqs, decode.tokens);
    decode.next.assign(next.begin(), next.end());
}

std::uint64_t CudaLlmEngineBackend::batched_prefills() const noexcept { return impl_->batched_prefills; }

std::uint64_t CudaLlmEngineBackend::batched_prefill_sequences() const noexcept {
    return impl_->batched_prefill_sequences;
}

std::uint64_t CudaLlmEngineBackend::mixed_passes() const noexcept { return impl_->mixed_passes; }

void CudaLlmEngineBackend::set_decode_path(DecodePath path) { impl_->decode_path = path; }

CudaLlmEngineBackend::DecodePath CudaLlmEngineBackend::decode_path() const noexcept { return impl_->decode_path; }

int CudaLlmEngineBackend::cuda_graphs_captured() const noexcept {
    return impl_->graphs_captured.load(std::memory_order_relaxed);
}

std::uint64_t CudaLlmEngineBackend::cuda_graph_replays() const noexcept {
    return impl_->graph_replays.load(std::memory_order_relaxed);
}

CudaLlmEngineBackend::DecodeHostTimes CudaLlmEngineBackend::decode_host_times() const {
    if (!impl_->graphs) {
        return {};
    }
    const auto& t = impl_->graphs->host_times();
    return {t.stage_ms, t.issue_ms, t.wait_ms, t.steps};
}

void CudaLlmEngineBackend::reset_decode_host_times() {
    if (impl_->graphs) {
        impl_->graphs->reset_host_times();
    }
}

std::vector<TokenId> CudaLlmEngineBackend::decode_step(std::span<const SeqId> ids,
                                                       std::span<const TokenId> tokens) {
    Impl& im = *impl_;
    int new_pages = 0;
    std::vector<::Sequence*> seqs = im.checked_for_decode(ids, tokens, new_pages);
    if (new_pages > im.kv->free_blocks()) {
        throw KvCacheExhausted("decode_step: needs " + std::to_string(new_pages) + " new KV pages, " +
                               std::to_string(im.kv->free_blocks()) + " free");
    }
    im.append_for_decode(ids, seqs);  // every check has passed
    std::vector<int> next;
    im.run_decode(seqs, tokens, next);
    im.finish_decode(ids, seqs, tokens);
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
    if (im.prefix_caching) {
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