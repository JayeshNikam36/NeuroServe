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
    std::string kv_cache_key;           // root of the KV prefix hash chain (16 hex digits); empty if the
                                        // backend has no prefix hashing. Equal keys: compatible K/V
};

struct KvCacheStats {
    int page_size_tokens = 0;
    int total_pages = 0;
    int free_pages = 0;
    int max_pages_per_sequence = 0;
    int shared_pages = 0;  // pages listed by two or more sequences (prefix sharing)
    // Prefix caching (docs/kv-cache.md); zero while it is off:
    int indexed_pages = 0;               // full, written pages in the prefix index
    std::int64_t prefix_lookups = 0;      // prompts looked up at admission
    std::int64_t prefix_lookup_pages = 0; // full prompt pages they could have reused
    std::int64_t prefix_hit_pages = 0;    // of those, the pages they did reuse
};

// The pages one sequence lists, and how many of them no other sequence lists: the pages that
// releasing it (completion, cancellation, preemption) gives back.
struct SequencePages {
    int held = 0;
    int exclusive = 0;
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

    // Makes the empty sequence `dst` start with the first `pages` pages of `src`, which must be full
    // (prefix sharing; docs/kv-cache.md). Their K/V are reused, not copied: each shared page is
    // freed only when the last sequence listing it is freed. std::invalid_argument for a bad call;
    // std::logic_error if the backend cannot share.
    virtual void share_prefix(SeqId dst, SeqId src, int pages) {
        (void)dst, (void)src, (void)pages;
        throw std::logic_error("this backend cannot share KV pages");
    }

    // The default, for backends that never share: every page is the sequence's own.
    virtual SequencePages sequence_pages(SeqId seq) const {
        const int page = kv_stats().page_size_tokens;
        const int held = page > 0 ? (sequence_length(seq) + page - 1) / page : 0;
        return {held, held};
    }

protected:
    InferenceBackend() = default;
};

}  // namespace neuroserve