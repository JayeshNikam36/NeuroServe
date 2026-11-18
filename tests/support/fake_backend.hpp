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
// With share_prompt_prefixes, prefill reuses the full pages of a live sequence whose prompt starts
// the same way (prefix sharing, before the real prefix index exists), so engine tests can run
// requests that share pages.
// Like a real backend it is not thread-safe: only the engine thread may call it. Read the
// test helpers only after the engine thread has stopped.
class FakeBackend final : public InferenceBackend {
public:
    explicit FakeBackend(int vocab_size = 1000, int total_pages = 64, int page_size = 16)
        : page_size_(page_size), kv_(core::KvBlockConfig{total_pages, page_size, total_pages}) {
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
        s.shared_pages = kv_.stats().shared_blocks;
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
        const auto [source, shared] = share_prompt_prefixes ? longest_shared_prefix(prompt) : std::pair<SeqId, int>{0, 0};
        const int n = static_cast<int>(prompt.size());
        const int pages = (n + page_size_ - 1) / page_size_ - shared;
        if (pages > kv_.free_blocks()) {
            throw KvCacheExhausted("prefill: needs " + std::to_string(pages) + " KV pages");
        }
        ++prefill_calls_;
        if (record_batches) {
            prefill_first_tokens.push_back(prompt.front());
        }
        if (shared > 0) {
            kv_.share_prefix(id, source, shared);
        }
        kv_.append(id, n - shared * page_size_);
        prompts_[id].assign(prompt.begin(), prompt.end());
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
            prompts_.erase(id);
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
    // Prefill reuses the full pages of a live sequence whose prompt starts the same way.
    bool share_prompt_prefixes = false;
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

    // The live sequence whose prompt shares the most full pages with this one (ties: lowest id).
    std::pair<SeqId, int> longest_shared_prefix(std::span<const TokenId> prompt) const {
        std::pair<SeqId, int> best{0, 0};
        for (const auto& [id, other] : prompts_) {
            const auto common = static_cast<int>(
                std::mismatch(prompt.begin(), prompt.end(), other.begin(), other.end()).first - prompt.begin());
            const int pages = common / page_size_;
            if (pages > best.second || (pages == best.second && pages > 0 && id < best.first)) {
                best = {id, pages};
            }
        }
        return best;
    }

    ModelInfo info_;
    int page_size_;
    core::KVCacheManager kv_;
    std::unordered_map<SeqId, int> lengths_;             // the backend's lengths (drift_after_decode bends them)
    std::unordered_map<SeqId, std::vector<TokenId>> prompts_;  // of live, prefilled sequences
    SeqId next_id_ = 1;
    int decode_calls_ = 0;
    int prefill_calls_ = 0;
    int fail_after_ = -1;
    int fail_once_at_ = -1;
    TokenId fail_prefill_token_ = -1;
};

}  // namespace neuroserve::test_support
