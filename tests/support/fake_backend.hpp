#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "neuroserve/core/inference_backend.hpp"

namespace neuroserve::test_support {

// Deterministic in-memory backend for CPU tests: the next token is (last token + 1) % vocab.
// Lets generation and scheduling logic be tested in CI without a GPU or a model.
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
        len = static_cast<int>(prompt.size());
        return next(prompt.back());
    }

    std::vector<TokenId> decode_step(std::span<const SeqId> ids,
                                     std::span<const TokenId> tokens) override {
        if (fail_after_ >= 0 && decode_calls_ >= fail_after_) {
            throw std::runtime_error("injected decode failure");
        }
        ++decode_calls_;
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

    // Test helpers
    int live_sequences() const { return static_cast<int>(lengths_.size()); }
    int decode_calls() const { return decode_calls_; }
    void fail_decode_after(int calls) { fail_after_ = calls; }

private:
    int& length(SeqId id) {
        auto it = lengths_.find(id);
        if (it == lengths_.end()) {
            throw std::invalid_argument("unknown sequence id " + std::to_string(id));
        }
        return it->second;
    }

    TokenId next(TokenId t) const { return (t + 1) % info_.vocab_size; }

    int used_pages() const {
        int pages = 0;
        for (const auto& entry : lengths_) {
            pages += (entry.second + page_size_ - 1) / page_size_;
        }
        return pages;
    }

    ModelInfo info_;
    int total_pages_;
    int page_size_;
    std::unordered_map<SeqId, int> lengths_;
    SeqId next_id_ = 1;
    int decode_calls_ = 0;
    int fail_after_ = -1;
};

}  // namespace neuroserve::test_support