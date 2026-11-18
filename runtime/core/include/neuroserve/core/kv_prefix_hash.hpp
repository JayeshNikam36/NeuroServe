#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "neuroserve/core/inference_backend.hpp"  // TokenId

namespace neuroserve::core {

// Prefix hashing for full KV blocks (docs/kv-cache.md, "Hashing full blocks"). Block i of a
// sequence gets h_i = H(h_{i-1}, tokens of block i), and the chain starts at a root that hashes
// everything that changes K/V values for the same tokens. Equal hashes then mean an equal whole
// prefix under an equal configuration. H is 64 bits, for lookup only: a hit is verified against
// the stored tokens before it is reused (Day 33), so a collision can cost a lookup, never a wrong
// output.
//
// The hashes are a stable format: the same inputs give the same hashes in every process, build and
// platform (no std::hash, no pointer values, explicit little-endian encoding). The golden values in
// tests/unit/test_kv_prefix_hash.cpp pin it, so changing the encoding is a visible decision.
using BlockHash = std::uint64_t;

// XXH64 (https://github.com/Cyan4973/xxHash, the 64-bit variant), implemented here so the hash has
// no dependency and cannot change with a library version. Streaming: update() in any pieces gives
// the same digest as one call with all the bytes.
class Xxh64 {
public:
    explicit Xxh64(std::uint64_t seed = 0) noexcept;
    void update(std::span<const std::byte> bytes) noexcept;
    void update(std::string_view bytes) noexcept;
    std::uint64_t digest() const noexcept;

private:
    std::uint64_t acc_[4];
    std::uint64_t seed_;
    std::uint64_t total_ = 0;
    unsigned char buffer_[32] = {};
    std::size_t buffered_ = 0;
};

std::uint64_t xxh64(std::span<const std::byte> bytes, std::uint64_t seed = 0) noexcept;
std::uint64_t xxh64(std::string_view bytes, std::uint64_t seed = 0) noexcept;

// RoPE as the engine applies it. Llama 3 uses scaling "llama3" with the four parameters below; a
// model without scaling has an empty scaling_type and zeros.
struct RopeConfig {
    double theta = 0.0;
    std::string scaling_type;
    double factor = 0.0;
    double low_freq_factor = 0.0;
    double high_freq_factor = 0.0;
    int original_max_position_embeddings = 0;
};

// Everything besides the tokens that decides a block's K/V values. Two caches may share blocks only
// if all of it is equal.
struct KvCacheIdentity {
    std::string model_id;        // the model's name
    std::string weights_digest;  // which weights (file_fingerprint of the weight files)
    std::string dtype;           // of the K/V values, e.g. "fp16"
    int block_size = 0;          // tokens per block
    RopeConfig rope;
};

// "0123456789abcdef": 16 lowercase hex digits.
std::string hash_hex(BlockHash hash);

class PrefixHasher {
public:
    // std::invalid_argument for an empty model_id, weights_digest or dtype, a block_size that is not
    // positive, or a RoPE theta that is not a positive finite number.
    explicit PrefixHasher(KvCacheIdentity identity);

    const KvCacheIdentity& identity() const noexcept { return identity_; }
    int block_size() const noexcept { return identity_.block_size; }

    // h_{-1}: the parent of a sequence's first block.
    BlockHash root() const noexcept { return root_; }

    // The hash of a full block whose parent hash is `parent` (root() for the first block).
    // std::invalid_argument unless block has exactly block_size() tokens.
    BlockHash next(BlockHash parent, std::span<const TokenId> block) const;

    // One hash per full block of `tokens`, in order; a partial last block has none (it is still
    // being written, so it is never shared).
    std::vector<BlockHash> chain(std::span<const TokenId> tokens) const;

    // Appends to `chain` the hashes of the full blocks of `tokens` it does not have yet, and returns
    // how many it added: how a growing sequence hashes each block once, when it fills. `chain` must
    // be the hashes of the first chain.size() blocks of `tokens`; std::invalid_argument if `tokens`
    // has fewer full blocks than that.
    int extend(std::vector<BlockHash>& chain, std::span<const TokenId> tokens) const;

private:
    KvCacheIdentity identity_;
    BlockHash root_;
};

// A content fingerprint of a file, as 16 hex digits: XXH64 of its size and either all of it (up to
// 16 MiB) or 64 chunks of 64 KiB spread evenly from its first byte to its last, about 4 MiB. For
// weight files: another export, model or fine-tune changes nearly every tensor, so it changes the
// sampled bytes, while hashing all 2.5 GB at every start would read the weights a second time. It
// does not detect one changed byte between samples; a checkpoint is never edited in place like that.
// std::runtime_error if the file cannot be read.
std::string file_fingerprint(const std::string& path);

}  // namespace neuroserve::core
