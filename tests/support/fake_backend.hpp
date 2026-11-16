#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "neuroserve/core/inference_backend.hpp"

namespace neuroserve::test_support {

// Deterministic in-memory backend for CPU tests: the next token is (last token + 1) % vocab.
// Lets generation and scheduling logic be tested in CI without a GPU or a model.
// KV pages are counted like the real backend: a sequence holds ceil(length / page_size) pages, and
// prefill or decode_step throw KvCacheExhausted (changing nothing) when the pages run out.
// Like a real backend it is not thread-safe: only the engine thread may call it. Read the
// test helpers only after the engine thread has stopped.
class FakeBackend final : public InferenceBackend {
public:
    explicit FakeBackend(int vocab_size = 1000, int total_pages = 64, int page_size = 16)
        : total_pages_(total_pages), page_size_(page_size) {
        info_.name = "fake";
        info_.vocab_size = vocab_size;
        info_.max_prompt_tokens = 512;
    }

    std::string_view name() const noexcept override { return "fake"; }
    const ModelInfo& model_info() const noexcept override { return info_; }

    KvCacheStats kv_stats() const override {
        KvCacheStats s;
        s.page_size_tokens = page_size_;
        s.total_pages = total_pages_;
        s.free_pages = total_pages_ - used_pages();
        s.max_pages_per_sequence = total_pages_;
        return s;
    }

    SeqId create_sequence() override {
        const SeqId id = next_id_++;
        lengths_[id] = 0;
        return id;
    }

    TokenId prefill(SeqId id, std::span<const TokenId> prompt) override {
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
        const int pages = pages_for(static_cast<int>(prompt.size()));
        if (pages > total_pages_ - used_pages()) {
            throw KvCacheExhausted("prefill: needs " + std::to_string(pages) + " KV pages");
        }
        ++prefill_calls_;
        len = static_cast<int>(prompt.size());
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
            const int len = length(id);
            new_pages += pages_for(len + 1) - pages_for(len);
        }
        if (new_pages > total_pages_ - used_pages()) {
            throw KvCacheExhausted("decode_step: needs " + std::to_string(new_pages) + " new KV pages");
        }
        ++decode_calls_;
        if (record_batches) {
            decode_batches.emplace_back(ids.begin(), ids.end());
        }
        std::vector<TokenId> out;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            ++length(ids[i]);
            out.push_back(next(tokens[i]));
        }
        return out;
    }

    void free_sequence(SeqId id) override {
        if (lengths_.erase(id) == 0) {
            throw std::invalid_argument("unknown sequence id " + std::to_string(id));
        }
    }

    int sequence_length(SeqId id) const override {
        auto it = lengths_.find(id);
        if (it == lengths_.end()) {
            throw std::invalid_argument("unknown sequence id " + std::to_string(id));
        }
        return it->second;
    }

    // ---- Test helpers ----
    int live_sequences() const { return static_cast<int>(lengths_.size()); }
    int decode_calls() const { return decode_calls_; }
    int prefill_calls() const { return prefill_calls_; }
    void fail_decode_after(int calls) { fail_after_ = calls; }          // every call from `calls` on fails
    void fail_decode_once_at(int call) { fail_once_at_ = call; }        // only that one call fails
    void fail_prefill_when_first_token(TokenId token) { fail_prefill_token_ = token; }

    // Called at the start of every decode_step with the batch size. Set it before the engine
    // thread starts; tests use it to pause the engine at a known point.
    std::function<void(std::size_t batch_size)> on_decode;

    // When true, every successful decode_step's sequence ids are appended to decode_batches.
    bool record_batches = false;
    std::vector<std::vector<SeqId>> decode_batches;

private:
    int& length(SeqId id) {
        auto it = lengths_.find(id);
        if (it == lengths_.end()) {
            throw std::invalid_argument("unknown sequence id " + std::to_string(id));
        }
        return it->second;
    }

    TokenId next(TokenId t) const { return (t + 1) % info_.vocab_size; }

    int pages_for(int tokens) const { return (tokens + page_size_ - 1) / page_size_; }

    int used_pages() const {
        int pages = 0;
        for (const auto& entry : lengths_) {
            pages += pages_for(entry.second);
        }
        return pages;
    }

    ModelInfo info_;
    int total_pages_;
    int page_size_;
    std::unordered_map<SeqId, int> lengths_;
    SeqId next_id_ = 1;
    int decode_calls_ = 0;
    int prefill_calls_ = 0;
    int fail_after_ = -1;
    int fail_once_at_ = -1;
    TokenId fail_prefill_token_ = -1;
};

}  // namespace neuroserve::test_support
