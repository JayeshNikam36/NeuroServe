// Prefix hashing for full KV blocks (docs/kv-cache.md, "Hashing full blocks", Day 32).
//
// The golden values were computed by an independent implementation of the documented encoding
// (Python's xxhash and struct modules), so they check the C++ against the specification, and they
// pin the format: a change to the encoding fails here and must be a visible decision.

#include "neuroserve/core/kv_prefix_hash.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using neuroserve::TokenId;
using neuroserve::core::BlockHash;
using neuroserve::core::file_fingerprint;
using neuroserve::core::hash_hex;
using neuroserve::core::KvCacheIdentity;
using neuroserve::core::PrefixHasher;
using neuroserve::core::Xxh64;
using neuroserve::core::xxh64;

// Llama-3.2-1B as the CUDA backend describes it, with a made-up weights digest.
KvCacheIdentity llama() {
    KvCacheIdentity id;
    id.model_id = "llama-3.2-1b";
    id.weights_digest = "0123456789abcdef:fedcba9876543210";
    id.dtype = "fp16";
    id.block_size = 16;
    id.rope = {500000.0, "llama3", 32.0, 1.0, 4.0, 8192};
    return id;
}

// <|begin_of_text|> then 1, 2, 3, ...: n tokens.
std::vector<TokenId> tokens(int n) {
    std::vector<TokenId> t;
    for (int i = 0; i < n; ++i) {
        t.push_back(i == 0 ? 128000 : i);
    }
    return t;
}

std::vector<std::byte> pattern(std::size_t n) {
    std::vector<std::byte> b(n);
    for (std::size_t i = 0; i < n; ++i) {
        b[i] = static_cast<std::byte>(i % 251);
    }
    return b;
}

bool share_any(const std::vector<BlockHash>& a, const std::vector<BlockHash>& b) {
    const std::set<BlockHash> s(a.begin(), a.end());
    return std::any_of(b.begin(), b.end(), [&](BlockHash h) { return s.count(h) > 0; });
}

// ---------------------------------------------------------------- XXH64

TEST(Xxh64, MatchesTheReferenceImplementation) {
    struct Vector {
        std::size_t length;
        std::uint64_t seed0;
        std::uint64_t seeded;  // seed 0x9E3779B97F4A7C15
    };
    // Every code path: empty, < 4, 4..7, 8..31, exactly one stripe, stripes plus a tail.
    const Vector vectors[] = {
        {0, 0xef46db3751d8e999ULL, 0xc4349fc93c010000ULL},    {1, 0xe934a84adb052768ULL, 0x126bb57a12364aa5ULL},
        {3, 0xe5c7bb4533bc65ddULL, 0x67bc6ed5f6c6e4baULL},    {4, 0xffced8604453cc1eULL, 0xd89842cd31e24e54ULL},
        {7, 0x14cc643f630c72d2ULL, 0xececf5faa8a7490eULL},    {8, 0x884a173614b81b8dULL, 0xd18b6d7a5a668732ULL},
        {15, 0xa948f5f0f6abac2dULL, 0x7804c1fcfe249577ULL},   {31, 0xc346d2b59b4d8ee1ULL, 0xf3da6d05709c035dULL},
        {32, 0xcbf59c5116ff32b4ULL, 0xa1c89217e9d50750ULL},   {33, 0x0c535d1acafb8eadULL, 0xe6a3c00cd6e74075ULL},
        {63, 0xe26aa9e2a95f8e4fULL, 0x26a0acd772de057eULL},   {64, 0xf7c67301db6713f0ULL, 0x2589245e62a1969bULL},
        {100, 0x6ac1e58032166597ULL, 0x3b97d91eba03e785ULL},  {1000, 0xf306f04aa88b54d3ULL, 0xd9b7af87dd48f5b4ULL},
    };
    for (const Vector& v : vectors) {
        const std::vector<std::byte> data = pattern(v.length);
        EXPECT_EQ(xxh64(data), v.seed0) << "length " << v.length;
        EXPECT_EQ(xxh64(data, 0x9E3779B97F4A7C15ULL), v.seeded) << "length " << v.length << ", seeded";
    }
    EXPECT_EQ(xxh64(std::string_view("abc")), 0x44bc2cf5ad770999ULL);
}

TEST(Xxh64, UpdatingInAnyPiecesGivesTheSameDigestAsOneCall) {
    const std::vector<std::byte> data = pattern(1000);
    const std::uint64_t whole = xxh64(data, 7);
    for (std::size_t piece = 1; piece <= 70; ++piece) {
        Xxh64 h(7);
        for (std::size_t at = 0; at < data.size(); at += piece) {
            h.update(std::span(data).subspan(at, std::min(piece, data.size() - at)));
        }
        EXPECT_EQ(h.digest(), whole) << "pieces of " << piece << " bytes";
    }
}

// ---------------------------------------------------------------- the chain

TEST(PrefixHash, GoldenValuesPinTheFormat) {
    const PrefixHasher hasher(llama());
    EXPECT_EQ(hasher.root(), 0xe04dba04f624459cULL);
    const std::vector<BlockHash> chain = hasher.chain(tokens(47));
    ASSERT_EQ(chain.size(), 2U);
    EXPECT_EQ(chain[0], 0xa263c4868ff44244ULL);
    EXPECT_EQ(chain[1], 0x028e380b2911c591ULL);
    EXPECT_EQ(hash_hex(hasher.root()), "e04dba04f624459c");
}

TEST(PrefixHash, TheSamePrefixGivesTheSameChain) {
    const std::vector<TokenId> prompt = tokens(80);
    const std::vector<BlockHash> a = PrefixHasher(llama()).chain(prompt);
    const std::vector<BlockHash> b = PrefixHasher(llama()).chain(prompt);  // another hasher: no shared state
    EXPECT_EQ(a, b);
    ASSERT_EQ(a.size(), 5U);

    // A request that continues the same 40 tokens differently shares exactly its 2 full blocks.
    std::vector<TokenId> other(prompt.begin(), prompt.begin() + 40);
    for (int i = 0; i < 40; ++i) {
        other.push_back(90000 + i);
    }
    const std::vector<BlockHash> c = PrefixHasher(llama()).chain(other);
    ASSERT_EQ(c.size(), 5U);
    EXPECT_EQ(c[0], a[0]);
    EXPECT_EQ(c[1], a[1]);
    for (std::size_t i = 2; i < 5; ++i) {
        EXPECT_NE(c[i], a[i]) << "block " << i;
    }
}

TEST(PrefixHash, OnlyFullBlocksAreHashed) {
    const PrefixHasher hasher(llama());
    for (const auto& [n, blocks] : std::vector<std::pair<int, std::size_t>>{
             {0, 0}, {1, 0}, {15, 0}, {16, 1}, {31, 1}, {32, 2}, {47, 2}, {48, 3}}) {
        EXPECT_EQ(hasher.chain(tokens(n)).size(), blocks) << n << " tokens";
    }
    // The partial last block does not affect the full ones.
    EXPECT_EQ(hasher.chain(tokens(47)), hasher.chain(tokens(32)));
}

TEST(PrefixHash, ChangingOneTokenChangesItsBlockAndEveryBlockAfterIt) {
    const PrefixHasher hasher(llama());
    const std::vector<TokenId> base = tokens(64);
    const std::vector<BlockHash> expected = hasher.chain(base);
    for (int p = 0; p < 64; ++p) {
        std::vector<TokenId> changed = base;
        changed[static_cast<std::size_t>(p)] += 1;
        const std::vector<BlockHash> got = hasher.chain(changed);
        ASSERT_EQ(got.size(), 4U);
        for (int b = 0; b < 4; ++b) {
            if (b < p / 16) {
                EXPECT_EQ(got[b], expected[b]) << "token " << p << ", block " << b << " is before it";
            } else {
                EXPECT_NE(got[b], expected[b]) << "token " << p << ", block " << b;
            }
        }
    }
}

TEST(PrefixHash, TheOrderOfTokensInABlockMatters) {
    const PrefixHasher hasher(llama());
    std::vector<TokenId> t = tokens(16);
    const BlockHash h = hasher.next(hasher.root(), t);
    std::swap(t[3], t[4]);
    EXPECT_NE(hasher.next(hasher.root(), t), h);
}

TEST(PrefixHash, ABlockHashDependsOnItsParent) {
    // The same 16 tokens after a different first block: a block's K/V depend on every token before it.
    const PrefixHasher hasher(llama());
    std::vector<TokenId> a = tokens(32);
    std::vector<TokenId> b = a;
    b[5] = 777;
    EXPECT_NE(hasher.chain(a)[1], hasher.chain(b)[1]);
    EXPECT_NE(hasher.next(hasher.root(), std::span(a).subspan(16, 16)), hasher.chain(a)[1]);
}

TEST(PrefixHash, EachPartOfTheIdentityChangesEveryHash) {
    const std::vector<TokenId> prompt = tokens(64);
    const PrefixHasher base(llama());
    const std::vector<BlockHash> base_chain = base.chain(prompt);

    const std::vector<std::pair<std::string, std::function<void(KvCacheIdentity&)>>> changes = {
        {"model id", [](KvCacheIdentity& i) { i.model_id = "llama-3.2-3b"; }},
        {"weights digest", [](KvCacheIdentity& i) { i.weights_digest = "0123456789abcdef:fedcba9876543211"; }},
        {"dtype", [](KvCacheIdentity& i) { i.dtype = "bf16"; }},
        {"block size 8", [](KvCacheIdentity& i) { i.block_size = 8; }},
        {"block size 32", [](KvCacheIdentity& i) { i.block_size = 32; }},
        {"RoPE theta", [](KvCacheIdentity& i) { i.rope.theta = 10000.0; }},
        {"RoPE scaling type", [](KvCacheIdentity& i) { i.rope.scaling_type = ""; }},
        {"RoPE factor", [](KvCacheIdentity& i) { i.rope.factor = 8.0; }},
        {"RoPE low-frequency factor", [](KvCacheIdentity& i) { i.rope.low_freq_factor = 2.0; }},
        {"RoPE high-frequency factor", [](KvCacheIdentity& i) { i.rope.high_freq_factor = 2.0; }},
        {"RoPE original context", [](KvCacheIdentity& i) { i.rope.original_max_position_embeddings = 4096; }},
    };
    std::set<BlockHash> roots{base.root()};
    for (const auto& [what, change] : changes) {
        KvCacheIdentity id = llama();
        change(id);
        const PrefixHasher other(id);
        EXPECT_NE(other.root(), base.root()) << what;
        const std::vector<BlockHash> chain = other.chain(prompt);
        EXPECT_FALSE(chain.empty()) << what;
        EXPECT_FALSE(share_any(chain, base_chain)) << what << ": a block hash equals one of the base chain's";
        roots.insert(other.root());
    }
    EXPECT_EQ(roots.size(), changes.size() + 1) << "two different identities gave the same root";
}

TEST(PrefixHash, IdentityFieldsDoNotRunTogether) {
    KvCacheIdentity a = llama();
    KvCacheIdentity b = llama();
    a.model_id = "ab";
    a.weights_digest = "c";
    b.model_id = "a";
    b.weights_digest = "bc";
    EXPECT_NE(PrefixHasher(a).root(), PrefixHasher(b).root());
}

TEST(PrefixHash, ExtendHashesEachBlockOnceAsASequenceGrows) {
    const PrefixHasher hasher(llama());
    const std::vector<TokenId> all = tokens(70);
    std::vector<BlockHash> chain;
    for (std::size_t n = 0; n <= all.size(); ++n) {
        const int added = hasher.extend(chain, std::span(all).first(n));
        EXPECT_EQ(added, n > 0 && n % 16 == 0 ? 1 : 0) << n << " tokens";
        EXPECT_EQ(chain.size(), n / 16);
    }
    EXPECT_EQ(chain, hasher.chain(all));

    // A chain longer than the tokens' full blocks is a caller bug, and changes nothing.
    const std::vector<BlockHash> before = chain;
    EXPECT_THROW(hasher.extend(chain, std::span(all).first(47)), std::invalid_argument);
    EXPECT_EQ(chain, before);
}

TEST(PrefixHash, NoCollisionsAmong200000DistinctBlocks) {
    const PrefixHasher hasher(llama());
    std::mt19937 rng(32);
    std::uniform_int_distribution<TokenId> token(0, 128255);
    std::unordered_set<BlockHash> seen;
    std::set<std::vector<TokenId>> blocks;
    std::vector<TokenId> block(16);
    while (blocks.size() < 200000) {
        std::generate(block.begin(), block.end(), [&] { return token(rng); });
        if (blocks.insert(block).second) {
            seen.insert(hasher.next(hasher.root(), block));
        }
    }
    // 64-bit hashes: a collision among 200,000 has a probability of about 1e-9.
    EXPECT_EQ(seen.size(), blocks.size());
}

TEST(PrefixHash, RejectsABadIdentityOrABlockOfTheWrongSize) {
    for (const auto& change : std::vector<std::function<void(KvCacheIdentity&)>>{
             [](KvCacheIdentity& i) { i.model_id.clear(); },
             [](KvCacheIdentity& i) { i.weights_digest.clear(); },
             [](KvCacheIdentity& i) { i.dtype.clear(); },
             [](KvCacheIdentity& i) { i.block_size = 0; },
             [](KvCacheIdentity& i) { i.rope.theta = 0.0; },
             [](KvCacheIdentity& i) { i.rope.theta = std::numeric_limits<double>::infinity(); },
         }) {
        KvCacheIdentity id = llama();
        change(id);
        EXPECT_THROW(PrefixHasher{id}, std::invalid_argument);
    }
    const PrefixHasher hasher(llama());
    EXPECT_THROW((void)hasher.next(hasher.root(), tokens(15)), std::invalid_argument);
    EXPECT_THROW((void)hasher.next(hasher.root(), tokens(17)), std::invalid_argument);
}

TEST(PrefixHash, HashHexIsSixteenLowercaseDigits) {
    EXPECT_EQ(hash_hex(0), "0000000000000000");
    EXPECT_EQ(hash_hex(0x00ABCDEF01234567ULL), "00abcdef01234567");
    EXPECT_EQ(hash_hex(~std::uint64_t{0}), "ffffffffffffffff");
}

// ---------------------------------------------------------------- weights fingerprint

class FileFingerprint : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("ns_fingerprint_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                ::testing::UnitTest::GetInstance()->current_test_info()->name());
        std::filesystem::create_directories(dir_);
    }
    void TearDown() override { std::filesystem::remove_all(dir_); }

    std::string write(const std::string& name, const std::vector<char>& bytes) const {
        const std::string path = (dir_ / name).string();
        std::ofstream(path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        return path;
    }

    static std::vector<char> small() {
        std::vector<char> b(1000);
        for (std::size_t i = 0; i < b.size(); ++i) {
            b[i] = static_cast<char>(i % 251);
        }
        return b;
    }
    static std::vector<char> big() {  // 20 MiB: above the whole-file limit, so sampled
        std::vector<char> b(std::size_t{20} << 20);
        for (std::size_t i = 0; i < b.size(); ++i) {
            b[i] = static_cast<char>((i * 7 + 3) % 256);
        }
        return b;
    }

    std::filesystem::path dir_;
};

TEST_F(FileFingerprint, MatchesTheReferenceImplementation) {
    EXPECT_EQ(file_fingerprint(write("small.bin", small())), "8f9ec2d4580296f2");
    EXPECT_EQ(file_fingerprint(write("big.bin", big())), "355a516b8bd9660c");
}

TEST_F(FileFingerprint, ASmallFileIsHashedWhole) {
    const std::vector<char> bytes = small();
    const std::string base = file_fingerprint(write("base.bin", bytes));
    for (std::size_t i = 0; i < bytes.size(); i += 37) {
        std::vector<char> changed = bytes;
        changed[i] = static_cast<char>(changed[i] ^ 1);
        EXPECT_NE(file_fingerprint(write("changed.bin", changed)), base) << "byte " << i;
    }
    std::vector<char> longer = bytes;
    longer.push_back(0);
    EXPECT_NE(file_fingerprint(write("longer.bin", longer)), base);
}

TEST_F(FileFingerprint, ALargeFileIsSampledFromItsFirstByteToItsLast) {
    const std::vector<char> bytes = big();
    const std::string base = file_fingerprint(write("base.bin", bytes));
    auto with_byte_changed = [&](std::size_t at) {
        std::vector<char> changed = bytes;
        changed[at] = static_cast<char>(changed[at] ^ 1);
        return file_fingerprint(write("changed.bin", changed));
    };
    EXPECT_NE(with_byte_changed(0), base) << "first byte";
    EXPECT_NE(with_byte_changed(bytes.size() - 1), base) << "last byte";
    const std::size_t sample_32 = 32 * (bytes.size() - (std::size_t{64} << 10)) / 63;
    EXPECT_NE(with_byte_changed(sample_32 + 100), base) << "inside a middle sample";

    std::vector<char> shorter(bytes.begin(), bytes.end() - 1);
    EXPECT_NE(file_fingerprint(write("shorter.bin", shorter)), base);

    // The documented limit: one byte between two samples is not seen. Sample 0 covers
    // [0, 64 KiB) and sample 1 starts at (size - 64 KiB) / 63, about 325 KiB.
    EXPECT_EQ(with_byte_changed(200'000), base);
}

TEST_F(FileFingerprint, AMissingFileIsAnError) {
    EXPECT_THROW(file_fingerprint((dir_ / "missing.bin").string()), std::runtime_error);
}

}  // namespace
