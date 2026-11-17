#include "neuroserve/core/decision_log.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using neuroserve::core::DecisionAction;
using neuroserve::core::DecisionLog;
using neuroserve::core::Explanation;

namespace {

Explanation about(const std::string& request_id) {
    Explanation e;
    e.request_id = request_id;
    e.action = DecisionAction::kAdmit;
    e.reason = "admitted";
    return e;
}

std::vector<std::uint64_t> seqs(const neuroserve::core::DecisionPage& page) {
    std::vector<std::uint64_t> out;
    for (const auto& e : page.decisions) {
        out.push_back(e.seq);
    }
    return out;
}

}  // namespace

TEST(DecisionLog, NumbersEntriesFromOne) {
    DecisionLog log(10);
    log.add(about("ns-000001"));
    log.add(about("ns-000002"));
    const auto page = log.since(0, 100);
    EXPECT_EQ(seqs(page), (std::vector<std::uint64_t>{1, 2}));
    EXPECT_EQ(page.decisions[1].request_id, "ns-000002");
    EXPECT_EQ(page.next_since, 2u);
    EXPECT_EQ(page.dropped, 0u);
}

TEST(DecisionLog, TheCursorPagesThroughEverythingOnce) {
    DecisionLog log(100);
    for (int i = 0; i < 7; ++i) {
        log.add(about("r" + std::to_string(i)));
    }
    auto page = log.since(0, 3);
    EXPECT_EQ(seqs(page), (std::vector<std::uint64_t>{1, 2, 3}));
    page = log.since(page.next_since, 3);
    EXPECT_EQ(seqs(page), (std::vector<std::uint64_t>{4, 5, 6}));
    page = log.since(page.next_since, 3);
    EXPECT_EQ(seqs(page), (std::vector<std::uint64_t>{7}));
    page = log.since(page.next_since, 3);
    EXPECT_TRUE(page.decisions.empty());
    EXPECT_EQ(page.next_since, 7u);  // nothing new: ask again later with the same cursor
}

TEST(DecisionLog, KeepsOnlyTheNewestAndReportsWhatAReaderMissed) {
    DecisionLog log(3);
    for (int i = 0; i < 5; ++i) {
        log.add(about("r" + std::to_string(i)));
    }
    EXPECT_EQ(log.size(), 3u);  // bounded memory
    const auto page = log.since(0, 100);
    EXPECT_EQ(seqs(page), (std::vector<std::uint64_t>{3, 4, 5}));
    EXPECT_EQ(page.dropped, 2u);  // seq 1 and 2 were overwritten before this reader got them
    EXPECT_EQ(log.since(3, 100).dropped, 0u);
}

TEST(DecisionLog, ActionsHaveNames) {
    EXPECT_EQ(neuroserve::core::to_string(DecisionAction::kAdmit), "admit");
    EXPECT_EQ(neuroserve::core::to_string(DecisionAction::kDefer), "defer");
    EXPECT_EQ(neuroserve::core::to_string(DecisionAction::kReject), "reject");
    EXPECT_EQ(neuroserve::core::to_string(DecisionAction::kPreempt), "preempt");
    EXPECT_THROW(DecisionLog(0), std::invalid_argument);
}
