#include "neuroserve/core/kv_prefix_hash.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace neuroserve::core {
namespace {

constexpr std::uint64_t kP1 = 0x9E3779B185EBCA87ULL;
constexpr std::uint64_t kP2 = 0xC2B2AE3D27D4EB4FULL;
constexpr std::uint64_t kP3 = 0x165667B19E3779F9ULL;
constexpr std::uint64_t kP4 = 0x85EBCA77C2B2AE63ULL;
constexpr std::uint64_t kP5 = 0x27D4EB2F165667C5ULL;

// Little-endian reads, whatever the host's byte order.
std::uint64_t read64(const unsigned char* p) noexcept {
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | p[i];
    }
    return v;
}

std::uint32_t read32(const unsigned char* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t xxh_round(std::uint64_t acc, std::uint64_t input) noexcept {
    acc += input * kP2;
    acc = std::rotl(acc, 31);
    return acc * kP1;
}

std::uint64_t merge_round(std::uint64_t acc, std::uint64_t value) noexcept {
    acc ^= xxh_round(0, value);
    return acc * kP1 + kP4;
}

// The canonical byte encoding the hashes are defined over (docs/kv-cache.md): fixed-width
// little-endian integers, doubles by their IEEE-754 bits, strings as a u32 length and their bytes.
// Length prefixes keep fields apart: ("ab", "c") and ("a", "bc") encode differently.
class Encoder {
public:
    explicit Encoder(Xxh64& h) noexcept : h_(h) {}

    void u32(std::uint32_t v) noexcept {
        unsigned char b[4];
        for (int i = 0; i < 4; ++i) {
            b[i] = static_cast<unsigned char>(v >> (8 * i));
        }
        h_.update(std::as_bytes(std::span(b)));
    }
    void u64(std::uint64_t v) noexcept {
        unsigned char b[8];
        for (int i = 0; i < 8; ++i) {
            b[i] = static_cast<unsigned char>(v >> (8 * i));
        }
        h_.update(std::as_bytes(std::span(b)));
    }
    void f64(double v) noexcept { u64(std::bit_cast<std::uint64_t>(v)); }
    void str(std::string_view s) noexcept {
        u32(static_cast<std::uint32_t>(s.size()));
        h_.update(s);
    }

private:
    Xxh64& h_;
};

constexpr std::string_view kRootTag = "neuroserve.kv.root.v1";
constexpr std::string_view kBlockTag = "neuroserve.kv.block.v1";
constexpr std::string_view kFileTag = "neuroserve.file.v1";

void validate(const KvCacheIdentity& id) {
    if (id.model_id.empty() || id.weights_digest.empty() || id.dtype.empty()) {
        throw std::invalid_argument("KV cache identity: model_id, weights_digest and dtype must not be empty");
    }
    if (id.block_size <= 0) {
        throw std::invalid_argument("KV cache identity: block_size must be positive");
    }
    if (!std::isfinite(id.rope.theta) || id.rope.theta <= 0.0) {
        throw std::invalid_argument("KV cache identity: RoPE theta must be a positive finite number");
    }
}

BlockHash root_of(const KvCacheIdentity& id) noexcept {
    Xxh64 h;
    Encoder e(h);
    e.str(kRootTag);
    e.str(id.model_id);
    e.str(id.weights_digest);
    e.str(id.dtype);
    e.u32(static_cast<std::uint32_t>(id.block_size));
    e.f64(id.rope.theta);
    e.str(id.rope.scaling_type);
    e.f64(id.rope.factor);
    e.f64(id.rope.low_freq_factor);
    e.f64(id.rope.high_freq_factor);
    e.u32(static_cast<std::uint32_t>(id.rope.original_max_position_embeddings));
    return h.digest();
}

}  // namespace

Xxh64::Xxh64(std::uint64_t seed) noexcept
    : acc_{seed + kP1 + kP2, seed + kP2, seed, seed - kP1}, seed_(seed) {}

void Xxh64::update(std::span<const std::byte> bytes) noexcept {
    const auto* p = reinterpret_cast<const unsigned char*>(bytes.data());
    std::size_t n = bytes.size();
    total_ += n;

    if (buffered_ > 0) {  // top up a partial stripe first
        const std::size_t take = std::min(n, sizeof(buffer_) - buffered_);
        std::memcpy(buffer_ + buffered_, p, take);
        buffered_ += take;
        p += take;
        n -= take;
        if (buffered_ < sizeof(buffer_)) {
            return;
        }
        for (int i = 0; i < 4; ++i) {
            acc_[i] = xxh_round(acc_[i], read64(buffer_ + 8 * i));
        }
        buffered_ = 0;
    }
    while (n >= 32) {
        for (int i = 0; i < 4; ++i) {
            acc_[i] = xxh_round(acc_[i], read64(p + 8 * i));
        }
        p += 32;
        n -= 32;
    }
    if (n > 0) {
        std::memcpy(buffer_, p, n);
        buffered_ = n;
    }
}

void Xxh64::update(std::string_view bytes) noexcept {
    update(std::as_bytes(std::span(bytes.data(), bytes.size())));
}

std::uint64_t Xxh64::digest() const noexcept {
    std::uint64_t h = 0;
    if (total_ >= 32) {
        h = std::rotl(acc_[0], 1) + std::rotl(acc_[1], 7) + std::rotl(acc_[2], 12) + std::rotl(acc_[3], 18);
        for (const std::uint64_t a : acc_) {
            h = merge_round(h, a);
        }
    } else {
        h = seed_ + kP5;
    }
    h += total_;

    const unsigned char* p = buffer_;
    std::size_t n = buffered_;
    while (n >= 8) {
        h ^= xxh_round(0, read64(p));
        h = std::rotl(h, 27) * kP1 + kP4;
        p += 8;
        n -= 8;
    }
    if (n >= 4) {
        h ^= static_cast<std::uint64_t>(read32(p)) * kP1;
        h = std::rotl(h, 23) * kP2 + kP3;
        p += 4;
        n -= 4;
    }
    while (n > 0) {
        h ^= static_cast<std::uint64_t>(*p) * kP5;
        h = std::rotl(h, 11) * kP1;
        ++p;
        --n;
    }

    h ^= h >> 33;
    h *= kP2;
    h ^= h >> 29;
    h *= kP3;
    h ^= h >> 32;
    return h;
}

std::uint64_t xxh64(std::span<const std::byte> bytes, std::uint64_t seed) noexcept {
    Xxh64 h(seed);
    h.update(bytes);
    return h.digest();
}

std::uint64_t xxh64(std::string_view bytes, std::uint64_t seed) noexcept {
    Xxh64 h(seed);
    h.update(bytes);
    return h.digest();
}

std::string hash_hex(BlockHash hash) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[static_cast<std::size_t>(i)] = kDigits[hash & 0xF];
        hash >>= 4;
    }
    return out;
}

PrefixHasher::PrefixHasher(KvCacheIdentity identity) : identity_(std::move(identity)), root_(0) {
    validate(identity_);
    root_ = root_of(identity_);
}

BlockHash PrefixHasher::next(BlockHash parent, std::span<const TokenId> block) const {
    if (block.size() != static_cast<std::size_t>(identity_.block_size)) {
        throw std::invalid_argument("prefix hash: a block has " + std::to_string(identity_.block_size) +
                                    " tokens, got " + std::to_string(block.size()));
    }
    Xxh64 h;
    Encoder e(h);
    e.str(kBlockTag);
    e.u64(parent);
    e.u32(static_cast<std::uint32_t>(block.size()));
    for (const TokenId t : block) {
        e.u32(static_cast<std::uint32_t>(t));
    }
    return h.digest();
}

std::vector<BlockHash> PrefixHasher::chain(std::span<const TokenId> tokens) const {
    std::vector<BlockHash> out;
    extend(out, tokens);
    return out;
}

int PrefixHasher::extend(std::vector<BlockHash>& chain, std::span<const TokenId> tokens) const {
    const std::size_t bs = static_cast<std::size_t>(identity_.block_size);
    const std::size_t full = tokens.size() / bs;
    if (chain.size() > full) {
        throw std::invalid_argument("prefix hash: the chain has " + std::to_string(chain.size()) +
                                    " blocks, but the tokens fill only " + std::to_string(full));
    }
    const std::size_t had = chain.size();
    chain.reserve(full);  // may throw: before any change
    BlockHash parent = had == 0 ? root_ : chain.back();
    for (std::size_t b = had; b < full; ++b) {
        parent = next(parent, tokens.subspan(b * bs, bs));
        chain.push_back(parent);
    }
    return static_cast<int>(full - had);
}

std::string file_fingerprint(const std::string& path) {
    namespace fs = std::filesystem;
    constexpr std::uint64_t kWholeFileLimit = std::uint64_t{16} << 20;
    constexpr std::uint64_t kChunk = std::uint64_t{64} << 10;
    constexpr std::uint64_t kSamples = 64;

    std::error_code ec;
    const std::uint64_t size = fs::file_size(path, ec);
    std::ifstream in(path, std::ios::binary);
    if (ec || !in) {
        throw std::runtime_error("cannot read '" + path + "' to fingerprint it");
    }

    Xxh64 h;
    Encoder e(h);
    e.str(kFileTag);
    e.u64(size);

    std::vector<char> buf(static_cast<std::size_t>(kChunk));
    auto hash_range = [&](std::uint64_t offset, std::uint64_t length) {
        in.seekg(static_cast<std::streamoff>(offset));
        while (length > 0) {
            const std::uint64_t want = std::min<std::uint64_t>(length, kChunk);
            in.read(buf.data(), static_cast<std::streamsize>(want));
            if (static_cast<std::uint64_t>(in.gcount()) != want) {
                throw std::runtime_error("'" + path + "' was shorter than its size while fingerprinting it");
            }
            h.update(std::as_bytes(std::span(buf.data(), static_cast<std::size_t>(want))));
            length -= want;
        }
    };

    if (size <= kWholeFileLimit) {
        hash_range(0, size);
    } else {
        for (std::uint64_t i = 0; i < kSamples; ++i) {
            const std::uint64_t offset = i * (size - kChunk) / (kSamples - 1);  // first chunk .. last chunk
            e.u64(offset);
            hash_range(offset, kChunk);
        }
    }
    return hash_hex(h.digest());
}

}  // namespace neuroserve::core
