#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace neuroserve {

using TokenId = std::int32_t;
using SeqId = std::uint64_t;

struct ModelInfo {
    std::string name;
    int n_layers = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int max_prompt_tokens = 0;          // longest prompt prefill() accepts
    std::size_t kv_bytes_per_token = 0; // K + V for all layers
};

struct KvCacheStats {
    int page_size_tokens = 0;
    int total_pages = 0;
    int free_pages = 0;
    int max_pages_per_sequence = 0;
};

// Thrown when KV memory is too short for a request. The scheduler catches this to defer or preempt.
class KvCacheExhausted : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Executes a model and owns its weights and KV memory.
// Not thread-safe: called only from the engine thread. Generation is greedy for now.
// On any exception, the backend's state is unchanged.
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    InferenceBackend(const InferenceBackend&) = delete;
    InferenceBackend& operator=(const InferenceBackend&) = delete;

    virtual std::string_view name() const noexcept = 0;
    virtual const ModelInfo& model_info() const noexcept = 0;
    virtual KvCacheStats kv_stats() const = 0;

    // Creates an empty sequence (no KV pages yet).
    virtual SeqId create_sequence() = 0;

    // Runs the whole prompt on an empty sequence and returns the next token.
    // Throws std::invalid_argument / std::length_error / std::logic_error for bad input,
    // KvCacheExhausted if there are not enough free pages.
    virtual TokenId prefill(SeqId seq, std::span<const TokenId> prompt) = 0;

    // One decode step for several sequences: tokens[i] is appended to seqs[i].
    // Returns the next token for each sequence, in the same order.
    virtual std::vector<TokenId> decode_step(std::span<const SeqId> seqs,
                                             std::span<const TokenId> tokens) = 0;

    // Releases the sequence and returns all its KV pages.
    virtual void free_sequence(SeqId seq) = 0;

    // Number of tokens currently cached for the sequence.
    virtual int sequence_length(SeqId seq) const = 0;

    // Debug check, run after every engine step when EngineLoopOptions::check_invariants is set:
    // throws a std::logic_error (core::KvInvariantViolation) naming what is inconsistent in the
    // backend's KV bookkeeping. The default checks nothing.
    virtual void check_kv_invariants() const {}

    // Sequences the backend currently holds, or -1 if it does not count them.
    virtual int live_sequences() const { return -1; }

protected:
    InferenceBackend() = default;
};

}  // namespace neuroserve