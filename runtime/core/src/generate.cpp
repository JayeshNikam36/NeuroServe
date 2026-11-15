#include "neuroserve/core/generate.hpp"

#include <stdexcept>
#include <string>

namespace neuroserve {
namespace {

// Owns one backend sequence and frees it on scope exit (RAII), even when an exception is thrown.
class SequenceGuard {
public:
    explicit SequenceGuard(InferenceBackend& backend)
        : backend_(backend), id_(backend.create_sequence()) {}

    ~SequenceGuard() {
        try {
            backend_.free_sequence(id_);
        } catch (...) {
            // A destructor must never throw; the sequence is gone either way.
        }
    }

    SequenceGuard(const SequenceGuard&) = delete;
    SequenceGuard& operator=(const SequenceGuard&) = delete;

    SeqId id() const noexcept { return id_; }

private:
    InferenceBackend& backend_;
    SeqId id_;
};

}  // namespace

std::vector<TokenId> generate_greedy(InferenceBackend& backend,
                                     std::span<const TokenId> prompt,
                                     int max_new_tokens) {
    if (max_new_tokens < 1) {
        throw std::invalid_argument("max_new_tokens must be at least 1, got " +
                                    std::to_string(max_new_tokens));
    }

    SequenceGuard seq(backend);
    std::vector<TokenId> out;
    out.reserve(static_cast<std::size_t>(max_new_tokens));

    out.push_back(backend.prefill(seq.id(), prompt));
    while (static_cast<int>(out.size()) < max_new_tokens) {
        const SeqId ids[] = {seq.id()};
        const TokenId last[] = {out.back()};
        out.push_back(backend.decode_step(ids, last).at(0));
    }
    return out;
}

}  // namespace neuroserve