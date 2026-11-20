// Property-based tests (Day 29): the KVCacheManager must behave exactly like the CPU reference on
// 10,000 seeded random programs, compared after every operation (KV-004). Since Day 33 both run
// with prefix caching on, and the programs admit, commit and match prompts too. On a divergence the
// failing program is shrunk to a short reproduction and printed with its seed.
//
// NS_KV_PROPERTY_PROGRAMS and NS_KV_PROPERTY_SEED change how many programs run and the first seed,
// e.g. NS_KV_PROPERTY_PROGRAMS=1000000 for a long local run.
#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "kv_ops.hpp"
#include "neuroserve/core/kv_cache_manager.hpp"
#include "neuroserve/core/kv_reference.hpp"

namespace kvt = ns_test::kv;
using neuroserve::core::KVCacheManager;
using neuroserve::core::KvBlockConfig;
using neuroserve::core::KvBlockManager;
using neuroserve::core::ReferenceBlockManager;

namespace {

std::uint64_t env_or(const char* name, std::uint64_t fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? std::strtoull(value, nullptr, 10) : fallback;
}

neuroserve::core::KvCacheIdentity identity(int block_size) {
    neuroserve::core::KvCacheIdentity id;
    id.model_id = "property-test";
    id.weights_digest = "0";
    id.dtype = "fp16";
    id.block_size = block_size;
    id.rope.theta = 10000.0;
    return id;
}

const kvt::Factory kReference = [](KvBlockConfig c) { return std::make_unique<ReferenceBlockManager>(c, identity(c.block_size)); };
const kvt::Factory kManager = [](KvBlockConfig c) { return std::make_unique<KVCacheManager>(c, identity(c.block_size)); };

// A deliberately wrong manager, to prove the harness finds and shrinks a real bug: it shares one
// block fewer than asked whenever two or more are asked for.
class SharesOneBlockTooFew final : public KvBlockManager {
public:
    explicit SharesOneBlockTooFew(KvBlockConfig c) : inner_(c, identity(c.block_size)) {}
    const KvBlockConfig& config() const noexcept override { return inner_.config(); }
    void create(neuroserve::SeqId s) override { inner_.create(s); }
    int blocks_needed(neuroserve::SeqId s, int t) const override { return inner_.blocks_needed(s, t); }
    void append(neuroserve::SeqId s, int t) override { inner_.append(s, t); }
    void share_prefix(neuroserve::SeqId d, neuroserve::SeqId s, int b) override {
        inner_.share_prefix(d, s, b >= 2 ? b - 1 : b);
    }
    void release(neuroserve::SeqId s) override { inner_.release(s); }
    bool contains(neuroserve::SeqId s) const override { return inner_.contains(s); }
    std::vector<neuroserve::core::BlockId> table(neuroserve::SeqId s) const override { return inner_.table(s); }
    int length(neuroserve::SeqId s) const override { return inner_.length(s); }
    int ref_count(neuroserve::core::BlockId b) const override { return inner_.ref_count(b); }
    int free_blocks() const override { return inner_.free_blocks(); }
    neuroserve::core::KvBlockStats stats() const override { return inner_.stats(); }
    bool prefix_caching() const noexcept override { return inner_.prefix_caching(); }
    int admit(neuroserve::SeqId s, std::span<const neuroserve::TokenId> p) override { return inner_.admit(s, p); }
    int commit(neuroserve::SeqId s, std::span<const neuroserve::TokenId> t) override { return inner_.commit(s, t); }
    int match(std::span<const neuroserve::TokenId> p) const override { return inner_.match(p); }
    bool indexed(neuroserve::core::BlockId b) const override { return inner_.indexed(b); }
    void check_invariants() const override { inner_.check_invariants(); }

private:
    KVCacheManager inner_;
};

}  // namespace

TEST(KvProperty, ProgramsAreReproducibleFromTheirSeed) {
    EXPECT_EQ(kvt::describe(kvt::generate(42)), kvt::describe(kvt::generate(42)));
    EXPECT_NE(kvt::describe(kvt::generate(42)), kvt::describe(kvt::generate(43)));
    // Pinned: a change to the generator changes every seed's program, which is allowed, but should
    // be a decision (old failure seeds would no longer reproduce).
    EXPECT_EQ(kvt::describe(kvt::generate(7, 4)),
              "seed 7, config {num_blocks 5, block_size 16, max_blocks_per_sequence 7}:\n"
              "  0: create(0)\n  1: admit(0, [12211221200002120000])\n  2: share_prefix(0, 0, 2)");
}

TEST(KvProperty, ManagerMatchesTheReferenceOnRandomPrograms) {
    const std::uint64_t programs = env_or("NS_KV_PROPERTY_PROGRAMS", 10'000);
    const std::uint64_t first_seed = env_or("NS_KV_PROPERTY_SEED", 1);
    std::map<std::string, long> coverage;  // "<op> -> <outcome>": how often each happened
    std::uint64_t ops = 0;
    const kvt::Observer count = [&](const kvt::Op& op, const std::string& outcome) {
        static const char* names[] = {"create", "append", "share_prefix", "release", "blocks_needed",
                                      "admit", "commit", "match"};
        const std::string result = outcome.rfind("ok", 0) == 0 ? "ok" : outcome;
        const bool sharing = op.kind == kvt::OpKind::kShare && result == "ok" && op.amount > 0;
        const bool counted = (op.kind == kvt::OpKind::kAdmit || op.kind == kvt::OpKind::kCommit ||
                              op.kind == kvt::OpKind::kMatch) && result == "ok";
        const std::string detail = sharing ? " (blocks shared)"
                                   : counted ? (outcome == "ok 0" ? " (none)" : op.kind == kvt::OpKind::kCommit ? " (indexed)" : " (hit)")
                                             : "";
        ++coverage[std::string(names[static_cast<int>(op.kind)]) + detail + " -> " + result];
        ++ops;
    };
    for (std::uint64_t i = 0; i < programs; ++i) {
        const kvt::Program program = kvt::generate(first_seed + i);
        if (const auto d = kvt::find_divergence(kReference, kManager, program, count)) {
            const kvt::Program small = kvt::shrink(kReference, kManager, program);
            FAIL() << "KVCacheManager diverges from the reference: " << d->detail
                   << "\nshortest reproduction (" << small.ops.size() << " of " << program.ops.size()
                   << " ops), " << kvt::describe(small) << "\n" << kvt::find_divergence(kReference, kManager, small)->detail;
        }
    }
    // The programs must actually reach the interesting cases, or agreeing on them proves little.
    std::cout << programs << " programs, " << ops << " operations; outcomes:\n";
    for (const auto& [what, n] : coverage) {
        std::cout << "  " << what << ": " << n << "\n";
    }
    if (programs >= 10'000) {
        EXPECT_GE(coverage["share_prefix (blocks shared) -> ok"], 10'000);
        EXPECT_GE(coverage["append -> ok"], 50'000);
        EXPECT_GE(coverage["append -> KvCacheExhausted"], 5'000);
        EXPECT_GE(coverage["append -> length_error"], 10'000);
        EXPECT_GE(coverage["share_prefix -> invalid_argument"], 5'000);
        EXPECT_GE(coverage["create -> invalid_argument"], 10'000);
        EXPECT_GE(coverage["release -> ok"], 20'000);
        EXPECT_GE(coverage["admit (hit) -> ok"], 5'000);
        EXPECT_GE(coverage["admit (none) -> ok"], 10'000);
        EXPECT_GE(coverage["admit -> KvCacheExhausted"], 1'000);
        EXPECT_GE(coverage["admit -> invalid_argument"], 1'000);
        EXPECT_GE(coverage["commit (indexed) -> ok"], 10'000);
        EXPECT_GE(coverage["commit (none) -> ok"], 10'000);
        EXPECT_GE(coverage["commit -> invalid_argument"], 1'000);
        EXPECT_GE(coverage["match (hit) -> ok"], 5'000);
    }
}

TEST(KvProperty, AWrongManagerIsCaughtAndShrunkToAShortReproduction) {
    const kvt::Factory wrong = [](KvBlockConfig c) { return std::make_unique<SharesOneBlockTooFew>(c); };
    for (std::uint64_t seed = 1; seed < 1'000; ++seed) {
        const kvt::Program program = kvt::generate(seed);
        if (!kvt::find_divergence(kReference, wrong, program)) {
            continue;
        }
        const kvt::Program small = kvt::shrink(kReference, wrong, program);
        std::cout << "found with " << kvt::describe(small) << "\n" << kvt::find_divergence(kReference, wrong, small)->detail << "\n";
        // Shrunk to the essence: create a source, give it tokens, create a destination, share
        // two blocks (the smallest count the bug changes). Nothing else is needed.
        EXPECT_EQ(small.ops.size(), 4u);
        EXPECT_EQ(small.ops.back().kind, kvt::OpKind::kShare);
        EXPECT_EQ(small.ops.back().amount, 2);
        return;
    }
    FAIL() << "no program in 1,000 seeds exposed a manager that shares one block too few";
}
