// GPU memory by owner (Day 58): the ledger's counting, without a GPU.

#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "neuroserve/core/memory_ledger.hpp"

namespace {

using neuroserve::core::MemoryLedger;
using neuroserve::core::MemoryOwner;
using neuroserve::core::MemorySource;

TEST(MemoryLedger, CountsLiveBytesByOwnerAndSource) {
    MemoryLedger ledger;
    ledger.record(MemoryOwner::kWeights, 1000, MemorySource::kExternal, "model weights");
    const auto a = ledger.record(MemoryOwner::kKv, 64, MemorySource::kPool, "block tables");
    ledger.record(MemoryOwner::kKv, 64, MemorySource::kPool, "block tables");
    ledger.record(MemoryOwner::kKv, 5000, MemorySource::kExternal, "KV pool");
    ledger.record(MemoryOwner::kWorkspace, 300, MemorySource::kPool, "prefill scratch");

    auto r = ledger.report();
    EXPECT_EQ(r.bytes, 6428u);
    EXPECT_EQ(r.pool_bytes, 428u);
    EXPECT_EQ(r.external_bytes, 6000u);
    const auto& kv = r.owners[static_cast<std::size_t>(MemoryOwner::kKv)];
    EXPECT_EQ(kv.bytes, 5128u);
    EXPECT_EQ(kv.pool_bytes, 128u);
    EXPECT_EQ(kv.external_bytes, 5000u);
    EXPECT_EQ(kv.allocations, 3u);
    EXPECT_EQ(r.owners[static_cast<std::size_t>(MemoryOwner::kOther)].bytes, 0u);
    // Items: one per (owner, source, label), the block tables summed.
    ASSERT_EQ(r.items.size(), 4u);
    EXPECT_EQ(r.items[0].label, "model weights");
    EXPECT_EQ(r.items[2].label, "block tables");
    EXPECT_EQ(r.items[2].bytes, 128u);
    EXPECT_EQ(r.items[2].allocations, 2u);

    ledger.release(a);
    r = ledger.report();
    EXPECT_EQ(r.owners[static_cast<std::size_t>(MemoryOwner::kKv)].bytes, 5064u);
    EXPECT_EQ(r.owners[static_cast<std::size_t>(MemoryOwner::kKv)].peak_bytes, 5128u);  // the peak stays
    EXPECT_EQ(r.peak_bytes, 6428u);
}

TEST(MemoryLedger, AReleaseMustMatchALiveRecord) {
    MemoryLedger ledger;
    const auto id = ledger.record(MemoryOwner::kOther, 8, MemorySource::kPool, "x");
    ledger.release(id);
    EXPECT_THROW(ledger.release(id), std::invalid_argument);
    EXPECT_THROW(ledger.release(12345), std::invalid_argument);
    EXPECT_EQ(ledger.report().bytes, 0u);
}

TEST(MemoryLedger, OwnersHaveNames) {
    EXPECT_EQ(to_string(MemoryOwner::kWeights), "weights");
    EXPECT_EQ(to_string(MemoryOwner::kWorkspace), "workspace");
    EXPECT_EQ(to_string(MemoryOwner::kKv), "kv");
    EXPECT_EQ(to_string(MemoryOwner::kOther), "other");
}

TEST(MemoryLedger, IsSafeFromManyThreads) {
    MemoryLedger ledger;
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&ledger] {
            for (int i = 0; i < 1000; ++i) {
                ledger.release(ledger.record(MemoryOwner::kKv, 256, MemorySource::kPool, "block tables"));
                (void)ledger.report();
            }
        });
    }
    for (auto& t : threads) t.join();
    const auto r = ledger.report();
    EXPECT_EQ(r.bytes, 0u);
    EXPECT_GE(r.peak_bytes, 256u);
    EXPECT_LE(r.peak_bytes, 8u * 256u);
}

}  // namespace
