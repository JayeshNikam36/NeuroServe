#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "neuroserve/core/inference_backend.hpp"
#include "neuroserve/core/kv_cache_manager.hpp"

namespace neuroserve::test_support {

// Deterministic in-memory backend for CPU tests: the next token is (last token + 1) % vocab.
// Lets generation and scheduling logic be tested in CI without a GPU or a model.
// KV pages are accounted by the real KVCacheManager, exactly as in the CUDA backend: a sequence
// holds ceil(length / page_size) pages, and prefill or decode_step throw KvCacheExhausted
// (changing nothing) when the pages run out.
// With prefix_caching, prefill admits the prompt through the manager's prefix index (reusing the
// full pages of live sequences with the same prefix) and computes only the rest, and every written
// full page is committed to the index, as a prefix-caching GPU backend does (docs/kv-cache.md).
// Like a real backend it is not thread-safe: only the engine thread may call it. Read the
// test helpers only after the engine thread has stopped.
class FakeBackend final : public InferenceBackend {
public:
    explicit FakeBackend(int vocab_size = 1000, int total_pages = 64, int page_size = 16)
        : page_size_(page_size), kv_(core::KvBlockConfig{total_pages, page_size, total_pages}, identity(page_size)) {
        info_.name = "fake";
        info_.vocab_size = vocab_size;
        info_.max_prompt_tokens = 512;
    }

    std::string_view name() const noexcept override { return "fake"; }
    const ModelInfo& model_info() const noexcept override { return info_; }

    KvCacheStats kv_stats() const override {
        KvCacheStats s;
        s.page_size_tokens = page_size_;
        s.total_pages = kv_.config().num_blocks;
        s.free_pages = kv_.free_blocks();
        s.max_pages_per_sequence = kv_.config().max_blocks_per_sequence;
        const core::KvBlockStats kv = kv_.stats();
        s.shared_pages = kv.shared_blocks;
        s.indexed_pages = kv.indexed_blocks;
        s.prefix_lookups = kv.prefix_lookups;
        s.prefix_lookup_pages = kv.prefix_lookup_blocks;
        s.prefix_hit_pages = kv.prefix_hit_blocks;
        s.cached_pages = kv.cached_blocks;
        s.evicted_pages = kv.evicted_blocks;
        return s;
    }

    SeqId create_sequence() override {
        const SeqId id = next_id_++;
        kv_.create(id);
        lengths_[id] = 0;
        return id;
    }

    TokenId prefill(SeqId id, std::span<const TokenId> prompt) override {
        if (on_prefill) {
            on_prefill();
        }
        int& len = length(id);
        if (len != 0) {
            throw std::logic_error("prefill: sequence already holds tokens");
        }
        if (prompt.empty()) {
            throw std::invalid_argument("prefill: empty prompt");
        }
        if (fail_prefill_token_ >= 0 && prompt.front() == fail_prefill_token_) {
            throw std::runtime_error("injected prefill failure");
        }
        const int n = static_cast<int>(prompt.size());
        int cached = 0;
        if (prefix_caching) {
            const core::Admission a = kv_.admit(id, prompt);  // KvCacheExhausted (changing nothing) if short
            cached = a.cached_tokens;
            block_copies_ += a.copy ? 1 : 0;  // a GPU backend copies the block's K/V here
        } else {
            const int pages = (n + page_size_ - 1) / page_size_;
            if (pages > kv_.free_blocks()) {
                throw KvCacheExhausted("prefill: needs " + std::to_string(pages) + " KV pages");
            }
            kv_.append(id, n);
        }
        ++prefill_calls_;
        prefilled_tokens_ += n - cached;  // the fake "computes" only what the cache did not hold
        if (record_batches) {
            prefill_first_tokens.push_back(prompt.front());
        }
        tokens_[id].assign(prompt.begin(), prompt.end());
        if (prefix_caching) {
            kv_.commit(id, tokens_[id]);  // its full pages are written now
        }
        len = n;
        return next(prompt.back());
    }

    std::vector<TokenId> decode_step(std::span<const SeqId> ids,
                                     std::span<const TokenId> tokens) override {
        if (on_decode) {
            on_decode(ids.size());
        }
        if (fail_after_ >= 0 && decode_calls_ >= fail_after_) {
            throw std::runtime_error("injected decode failure");
        }
        if (fail_once_at_ >= 0 && decode_calls_ == fail_once_at_) {
            fail_once_at_ = -1;
            throw std::runtime_error("injected decode failure");
        }
        int new_pages = 0;
        for (const SeqId id : ids) {
            length(id);  // known
            new_pages += kv_.blocks_needed(id, 1);
        }
        if (new_pages > kv_.free_blocks()) {
            throw KvCacheExhausted("decode_step: needs " + std::to_string(new_pages) + " new KV pages");
        }
        ++decode_calls_;
        if (record_batches) {
            decode_batches.emplace_back(ids.begin(), ids.end());
        }
        std::vector<TokenId> out;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            kv_.append(ids[i], 1);
            ++length(ids[i]);
            tokens_[ids[i]].push_back(tokens[i]);
            if (prefix_caching) {
                kv_.commit(ids[i], tokens_[ids[i]]);  // indexes the page this token filled, if any
            }
            out.push_back(next(tokens[i]));
        }
        length(ids[0]) += drift_after_decode;
        return out;
    }

    void free_sequence(SeqId id) override {
        if (lengths_.count(id) == 0) {
            throw std::invalid_argument("unknown sequence id " + std::to_string(id));
        }
        if (!leak_on_free) {
            lengths_.erase(id);
            tokens_.erase(id);
            kv_.release(id);
        }
    }

    int sequence_length(SeqId id) const override {
        auto it = lengths_.find(id);
        if (it == lengths_.end()) {
            throw std::invalid_argument("unknown sequence id " + std::to_string(id));
        }
        return it->second;
    }

    int live_sequences() const override { return static_cast<int>(lengths_.size()); }

    // The manager's own invariants. (Not the fake's lengths against it: drift_after_decode makes
    // them disagree on purpose, for the engine loop to find.)
    void check_kv_invariants() const override { kv_.check_invariants(); }

    void share_prefix(SeqId dst, SeqId src, int pages) override {
        length(dst);
        length(src);
        kv_.share_prefix(dst, src, pages);
        lengths_[dst] = pages * page_size_;
        const std::vector<TokenId>& from = tokens_[src];
        tokens_[dst].assign(from.begin(), from.begin() + std::min<std::ptrdiff_t>(pages * page_size_, static_cast<std::ptrdiff_t>(from.size())));
    }

    SequencePages sequence_pages(SeqId id) const override {
        sequence_length(id);  // known
        SequencePages out;
        for (const core::BlockId block : kv_.table(id)) {
            ++out.held;
            out.exclusive += kv_.ref_count(block) == 1;
        }
        return out;
    }

    // ---- Test helpers ----
    // Prefill looks prompts up in the prefix index and commits written pages to it.
    bool prefix_caching = false;
    std::int64_t prefilled_tokens() const { return prefilled_tokens_; }  // prompt tokens computed
    int block_copies() const { return block_copies_; }                   // copy-on-write copies
    const core::KVCacheManager& kv() const { return kv_; }
    // Injected bookkeeping faults, for the engine loop's invariant checks:
    bool leak_on_free = false;   // free_sequence() succeeds but keeps the sequence (a leak)
    int drift_after_decode = 0;  // each decode_step adds this to its first sequence's length
    int decode_calls() const { return decode_calls_; }
    int prefill_calls() const { return prefill_calls_; }
    void fail_decode_after(int calls) { fail_after_ = calls; }          // every call from `calls` on fails
    void fail_decode_once_at(int call) { fail_once_at_ = call; }        // only that one call fails
    void fail_prefill_when_first_token(TokenId token) { fail_prefill_token_ = token; }

    // Called at the start of every decode_step with the batch size. Set it before the engine
    // thread starts; tests use it to pause the engine at a known point.
    std::function<void(std::size_t batch_size)> on_decode;
    std::function<void()> on_prefill;  // the same, at the start of every prefill

    // When true, every successful decode_step's sequence ids are appended to decode_batches, and
    // the first token of every prefilled prompt to prefill_first_tokens (the admission order).
    bool record_batches = false;
    std::vector<std::vector<SeqId>> decode_batches;
    std::vector<TokenId> prefill_first_tokens;

private:
    int& length(SeqId id) {
        auto it = lengths_.find(id);
        if (it == lengths_.end()) {
            throw std::invalid_argument("unknown sequence id " + std::to_string(id));
        }
        return it->second;
    }

    TokenId next(TokenId t) const { return (t + 1) % info_.vocab_size; }

    static core::KvCacheIdentity identity(int page_size) {
        core::KvCacheIdentity id;
        id.model_id = "fake";
        id.weights_digest = "fake";
        id.dtype = "fp16";
        id.block_size = page_size;
        id.rope.theta = 10000.0;
        return id;
    }

    ModelInfo info_;
    int page_size_;
    core::KVCacheManager kv_;
    std::unordered_map<SeqId, int> lengths_;             // the backend's lengths (drift_after_decode bends them)
    std::unordered_map<SeqId, std::vector<TokenId>> tokens_;  // every token each live sequence holds
    std::int64_t prefilled_tokens_ = 0;
    int block_copies_ = 0;
    SeqId next_id_ = 1;
    int decode_calls_ = 0;
    int prefill_calls_ = 0;
    int fail_after_ = -1;
    int fail_once_at_ = -1;
    TokenId fail_prefill_token_ = -1;
};

}  // namespace neuroserve::test_support
