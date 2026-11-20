// The invariant checker (Day 30, KV-010): every invariant the KVCacheManager checks is broken on
// purpose and must be named, with its numbers; and the engine loop, which runs the checks after
// every step, must stop loudly when the backend's bookkeeping and its own disagree.
#include <gtest/gtest.h>

#include <chrono>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "fake_backend.hpp"
#include "prefill_gate.hpp"
#include "neuroserve/core/engine_loop.hpp"
#include "neuroserve/core/kv_cache_manager.hpp"
#include "neuroserve/core/log.hpp"

namespace neuroserve::core {

// Reaches into the manager to corrupt it the way a bug would.
struct KVCacheManagerTestPeer {
    static void add_ref(KVCacheManager& m, BlockId b, int delta) { m.blocks_[static_cast<std::size_t>(b)].refs += delta; }
    static void put_in_free_set(KVCacheManager& m, BlockId b) { m.free_.insert(b); }
    static void take_from_free_set(KVCacheManager& m, BlockId b) { m.free_.erase(b); }
    static std::vector<BlockId>& table(KVCacheManager& m, SeqId s) { return m.sequences_.at(s).table; }
    static int& length(KVCacheManager& m, SeqId s) { return m.sequences_.at(s).length; }
    static void drift_shared(KVCacheManager& m, int delta) { m.shared_blocks_ += delta; }
    static void drift_tokens(KVCacheManager& m, int delta) { m.cached_tokens_ += delta; }
    static void corrupt_indexed_token(KVCacheManager& m, BlockId b) { m.blocks_[static_cast<std::size_t>(b)].tokens[0] += 1; }
    static void drop_from_index(KVCacheManager& m, BlockId b) { m.index_.erase(m.blocks_[static_cast<std::size_t>(b)].hash); }
};

}  // namespace neuroserve::core

namespace {

using neuroserve::TokenId;
using neuroserve::core::EngineLoop;
using neuroserve::core::EngineLoopOptions;
using neuroserve::core::EngineStopped;
using neuroserve::core::GenerationParams;
using neuroserve::core::KvBlockConfig;
using neuroserve::core::KVCacheManager;
using neuroserve::core::KvInvariantViolation;
using Peer = neuroserve::core::KVCacheManagerTestPeer;
using neuroserve::test_support::FakeBackend;
using namespace std::chrono_literals;

// Sequence 1: 40 tokens in blocks 0, 1, 2 (block 2 partial). Sequence 2 shares blocks 0 and 1.
std::unique_ptr<KVCacheManager> two_sharers() {
    auto m = std::make_unique<KVCacheManager>(KvBlockConfig{8, 16, 4});
    m->create(1);
    m->append(1, 40);
    m->create(2);
    m->share_prefix(2, 1, 2);
    m->check_invariants();  // healthy before the fault
    return m;
}

void expect_violation(const KVCacheManager& m, std::initializer_list<const char*> parts) {
    try {
        m.check_invariants();
        ADD_FAILURE() << "the corrupted state passed check_invariants()";
    } catch (const KvInvariantViolation& e) {
        for (const char* part : parts) {
            EXPECT_NE(std::string(e.what()).find(part), std::string::npos) << "message: " << e.what() << "\nmissing: " << part;
        }
    }
}

TEST(KvInvariants, AnOffByOneReferenceCountIsNamed) {
    auto m = two_sharers();
    Peer::add_ref(*m, 2, +1);
    expect_violation(*m, {"I3", "block 2", "reference count 2", "listed by 1"});
}

TEST(KvInvariants, AFreedBlockStillInUseIsNamed) {
    auto m = two_sharers();
    Peer::put_in_free_set(*m, 0);  // a double free: block 0 is free and still in two tables
    expect_violation(*m, {"I1", "block 0", "reference count 2", "but is in the free set"});
}

TEST(KvInvariants, ALostBlockIsNamed) {
    auto m = two_sharers();
    Peer::take_from_free_set(*m, 5);  // neither free nor used: a leak
    expect_violation(*m, {"I1", "block 5", "reference count 0", "not in the free set"});
}

TEST(KvInvariants, ABlockListedTwiceInOneTableIsNamed) {
    auto m = two_sharers();
    Peer::table(*m, 1).push_back(0);
    Peer::length(*m, 1) = 56;  // keeps the table length consistent, so only the duplicate is wrong
    expect_violation(*m, {"I4", "sequence 1", "lists a block twice"});
}

TEST(KvInvariants, ATableLongerThanItsTokensIsNamed) {
    auto m = two_sharers();
    Peer::table(*m, 2).push_back(7);
    expect_violation(*m, {"I5", "sequence 2", "32 tokens", "3 blocks", "expected 2"});
}

TEST(KvInvariants, ABlockOutsideThePoolIsNamed) {
    auto m = two_sharers();
    Peer::table(*m, 1)[2] = 99;
    expect_violation(*m, {"I2", "sequence 1", "block 99", "outside the pool"});
}

TEST(KvInvariants, SharingAPartialBlockIsNamed) {
    auto m = two_sharers();
    Peer::table(*m, 2).push_back(2);  // sequence 2 now also lists block 2, which is not full
    Peer::length(*m, 2) = 33;
    Peer::add_ref(*m, 2, +1);
    expect_violation(*m, {"I6", "block 2", "shared but not full"});
}

TEST(KvInvariants, RunningCountsThatDriftAreNamed) {
    auto shared = two_sharers();
    Peer::drift_shared(*shared, +1);
    expect_violation(*shared, {"I2", "shared 3 vs 2"});
    auto tokens = two_sharers();
    Peer::drift_tokens(*tokens, +5);
    expect_violation(*tokens, {"I2", "tokens 77 vs 72"});
}

TEST(KvInvariants, ReleasingTwiceThroughTheApiIsRefusedAndHarmless) {
    auto m = two_sharers();
    m->release(2);
    EXPECT_THROW(m->release(2), std::invalid_argument);
    EXPECT_EQ(m->ref_count(0), 1);
    m->check_invariants();
}

// ---- The engine loop: checks after every step --------------------------------------------------

// Collects log lines (the engine thread writes them).
class LogCapture {
public:
    LogCapture() {
        neuroserve::core::log::set_sink([this](std::string_view line) {
            std::lock_guard lock(mu_);
            lines_.emplace_back(line);
        });
    }
    ~LogCapture() { neuroserve::core::log::set_sink(nullptr); }
    bool contains(const std::string& text) const {
        std::lock_guard lock(mu_);
        for (const auto& line : lines_) {
            if (line.find(text) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

private:
    mutable std::mutex mu_;
    std::vector<std::string> lines_;
};

GenerationParams params(std::vector<TokenId> prompt, int max_new_tokens) {
    GenerationParams p;
    p.prompt = std::move(prompt);
    p.max_new_tokens = max_new_tokens;
    return p;
}

EngineLoopOptions checked() {
    EngineLoopOptions o;
    o.check_invariants = true;
    return o;
}

bool stops_within(const EngineLoop& loop, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!loop.stop_requested()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

TEST(EngineInvariants, ALeakedSequenceStopsTheEngineLoudly) {
    LogCapture log;
    FakeBackend backend;
    backend.leak_on_free = true;  // free_sequence() "succeeds" but keeps the sequence
    EngineLoop loop(backend, checked());
    auto handle = loop.submit(params({1, 2, 3}, 3));
    ASSERT_TRUE(handle.wait_for(10s));
    EXPECT_EQ(handle.wait().tokens.size(), 3u) << "the request itself finished before the check";
    ASSERT_TRUE(stops_within(loop, 10s)) << "the leak was not caught";
    EXPECT_TRUE(log.contains("I12: the backend holds 1 sequences for 0 running requests"));
    EXPECT_THROW(loop.submit(params({1}, 1)), EngineStopped) << "a stopped engine refuses work";
}

TEST(EngineInvariants, ALengthTheLoopDoesNotKnowAboutStopsTheEngine) {
    LogCapture log;
    FakeBackend backend;
    backend.drift_after_decode = 1;  // the backend's length runs ahead of the loop's count
    EngineLoop loop(backend, checked());
    auto handle = loop.submit(params({1, 2, 3}, 8));
    ASSERT_TRUE(handle.wait_for(10s));
    EXPECT_THROW(handle.wait(), EngineStopped) << "running requests are cancelled, not served from bad state";
    EXPECT_TRUE(log.contains("I11: request ns-"));
    EXPECT_TRUE(log.contains("4 cached tokens in the loop's view but 5 in the backend"));
}

TEST(EngineInvariants, WithChecksOffTheSameLeakGoesUnnoticed) {
    FakeBackend backend;
    backend.leak_on_free = true;
    EngineLoopOptions options;
    options.check_invariants = false;
    EngineLoop loop(backend, options);
    for (int i = 0; i < 3; ++i) {
        auto handle = loop.submit(params({1, 2, 3}, 2));
        ASSERT_TRUE(handle.wait_for(10s));
        EXPECT_EQ(handle.wait().tokens.size(), 2u);
    }
    EXPECT_FALSE(loop.stop_requested());
}

TEST(EngineInvariants, AHealthyEngineUnderPreemptionPassesEveryCheck) {
    // 8 pages of 16 tokens for 24 requests that each grow to 3 pages: admission, preemption,
    // restarts and completion all happen, and the checks run after every step.
    // All 24 are queued before the engine goes on, so they compete: without the gate a loaded
    // machine let the engine finish requests as they came, and nothing was ever preempted.
    FakeBackend backend(1000, /*total_pages=*/8);
    neuroserve::test_support::FirstPrefillGate gate(backend);
    EngineLoop loop(backend, checked());
    std::vector<neuroserve::core::RequestHandle> handles;
    for (int i = 0; i < 24; ++i) {
        handles.push_back(loop.submit(params(std::vector<TokenId>(20, static_cast<TokenId>(i + 1)), 24)));
    }
    gate.open();
    for (auto& h : handles) {
        ASSERT_TRUE(h.wait_for(30s));
        EXPECT_EQ(h.wait().tokens.size(), 24u);
    }
    EXPECT_FALSE(loop.stop_requested());
    EXPECT_GT(loop.stats().preemptions, 0u) << "the test should exercise preemption";
}

// ---- The prefix index (Day 33): I7 and I8.

std::unique_ptr<KVCacheManager> indexed_prompt(std::vector<neuroserve::TokenId>& p) {
    neuroserve::core::KvCacheIdentity id;
    id.model_id = "m";
    id.weights_digest = "w";
    id.dtype = "fp16";
    id.block_size = 16;
    id.rope.theta = 10000.0;
    auto m = std::make_unique<KVCacheManager>(KvBlockConfig{8, 16, 4}, id);
    for (int i = 0; i < 40; ++i) {
        p.push_back(300 + i);
    }
    m->create(1);
    m->admit(1, p);
    m->commit(1, p);  // blocks 0 and 1 indexed
    m->check_invariants();
    return m;
}

TEST(KvInvariants, ATamperedIndexedBlockIsReportedAndNeverReused) {
    std::vector<neuroserve::TokenId> p;
    auto m = indexed_prompt(p);
    Peer::corrupt_indexed_token(*m, 1);  // its stored tokens no longer give its hash
    expect_violation(*m, {"I8", "block 1"});
    EXPECT_EQ(m->match(p), 1) << "verification refuses the block whose tokens differ";
}

TEST(KvInvariants, AnIndexedBlockMissingFromTheIndexIsReported) {
    std::vector<neuroserve::TokenId> p;
    auto m = indexed_prompt(p);
    Peer::drop_from_index(*m, 0);
    expect_violation(*m, {"I7", "block 0"});
}

}  // namespace
