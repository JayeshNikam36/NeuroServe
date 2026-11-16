// Request lifecycle: every legal transition succeeds, every illegal one throws, timestamps, ids.

#include <chrono>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "neuroserve/core/request.hpp"

namespace {

using neuroserve::core::IllegalTransition;
using neuroserve::core::is_legal_transition;
using neuroserve::core::is_terminal;
using neuroserve::core::kNumRequestStates;
using neuroserve::core::Request;
using neuroserve::core::RequestIdGenerator;
using S = neuroserve::core::RequestState;
using Clock = Request::Clock;
using namespace std::chrono_literals;

const Clock::time_point kT0 = Clock::time_point{} + 1000ms;

std::vector<S> all_states() {
    std::vector<S> states;
    for (std::size_t i = 0; i < kNumRequestStates; ++i) {
        states.push_back(static_cast<S>(i));
    }
    return states;
}

std::string name(S state) { return std::string(neuroserve::core::to_string(state)); }

// The specification, written independently of the implementation.
const std::set<std::pair<S, S>>& legal() {
    static const std::set<std::pair<S, S>> table = {
        {S::kReceived, S::kValidated},  {S::kReceived, S::kRejected},   {S::kReceived, S::kCancelled},
        {S::kValidated, S::kQueued},    {S::kValidated, S::kRejected},  {S::kValidated, S::kCancelled},
        {S::kQueued, S::kAdmitted},     {S::kQueued, S::kRejected},     {S::kQueued, S::kCancelled},
        {S::kAdmitted, S::kPrefill},    {S::kAdmitted, S::kCancelled},  {S::kAdmitted, S::kFailed},
        {S::kPrefill, S::kDecode},      {S::kPrefill, S::kCompleted},   {S::kPrefill, S::kPreempted},
        {S::kPrefill, S::kCancelled},   {S::kPrefill, S::kFailed},
        {S::kDecode, S::kCompleted},    {S::kDecode, S::kPreempted},    {S::kDecode, S::kCancelled},
        {S::kDecode, S::kFailed},
        {S::kPreempted, S::kQueued},    {S::kPreempted, S::kCancelled}, {S::kPreempted, S::kFailed},
    };
    return table;
}

// A legal path from RECEIVED to each state.
std::vector<S> path_to(S target) {
    switch (target) {
        case S::kReceived:  return {};
        case S::kValidated: return {S::kValidated};
        case S::kQueued:    return {S::kValidated, S::kQueued};
        case S::kAdmitted:  return {S::kValidated, S::kQueued, S::kAdmitted};
        case S::kPrefill:   return {S::kValidated, S::kQueued, S::kAdmitted, S::kPrefill};
        case S::kDecode:    return {S::kValidated, S::kQueued, S::kAdmitted, S::kPrefill, S::kDecode};
        case S::kPreempted: return {S::kValidated, S::kQueued, S::kAdmitted, S::kPrefill, S::kPreempted};
        case S::kCompleted: return {S::kValidated, S::kQueued, S::kAdmitted, S::kPrefill, S::kDecode, S::kCompleted};
        case S::kCancelled: return {S::kCancelled};
        case S::kRejected:  return {S::kRejected};
        case S::kFailed:    return {S::kValidated, S::kQueued, S::kAdmitted, S::kFailed};
    }
    return {};
}

Request request_in(S state) {
    Request request("ns-test");
    for (S step : path_to(state)) {
        request.transition_to(step);
    }
    return request;
}

long long ms(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(t - Clock::time_point{}).count();
}

long long ms(const Request& request, S state) {
    const auto t = request.entered_at(state);
    return t ? ms(*t) : -1;
}

TEST(RequestState, NamesMatchTheSpec) {
    const std::vector<std::string> expected = {"RECEIVED", "VALIDATED", "QUEUED",    "ADMITTED",
                                               "PREFILL",  "DECODE",    "PREEMPTED", "COMPLETED",
                                               "CANCELLED", "REJECTED", "FAILED"};
    ASSERT_EQ(expected.size(), kNumRequestStates);
    for (std::size_t i = 0; i < kNumRequestStates; ++i) {
        EXPECT_EQ(name(static_cast<S>(i)), expected[i]);
    }
}

TEST(RequestState, ExactlyFourTerminalStates) {
    std::set<S> terminal;
    for (S state : all_states()) {
        if (is_terminal(state)) {
            terminal.insert(state);
        }
    }
    EXPECT_EQ(terminal, (std::set<S>{S::kCompleted, S::kCancelled, S::kRejected, S::kFailed}));
}

TEST(RequestState, LegalityTableMatchesTheSpec) {
    EXPECT_EQ(legal().size(), 24u);
    for (S from : all_states()) {
        for (S to : all_states()) {
            EXPECT_EQ(is_legal_transition(from, to), legal().count({from, to}) == 1)
                << name(from) << " -> " << name(to);
        }
    }
}

TEST(RequestState, NoTransitionLeavesATerminalState) {
    for (S from : all_states()) {
        if (!is_terminal(from)) {
            continue;
        }
        for (S to : all_states()) {
            EXPECT_FALSE(is_legal_transition(from, to)) << name(from) << " -> " << name(to);
        }
    }
}

TEST(Request, StartsInReceivedWithTimestamp) {
    Request request("ns-000001", kT0);
    EXPECT_EQ(request.id(), "ns-000001");
    EXPECT_EQ(request.state(), S::kReceived);
    EXPECT_EQ(ms(request, S::kReceived), 1000);
    EXPECT_EQ(ms(request.last_transition_at()), 1000);
    EXPECT_FALSE(request.is_finished());
    EXPECT_EQ(request.preemption_count(), 0u);
}

TEST(Request, EmptyIdIsRejected) {
    EXPECT_THROW((void)Request(""), std::invalid_argument);
}

TEST(Request, EveryLegalTransitionSucceeds) {
    for (const auto& [from, to] : legal()) {
        SCOPED_TRACE(name(from) + " -> " + name(to));
        Request request = request_in(from);
        ASSERT_EQ(request.state(), from);
        EXPECT_NO_THROW(request.transition_to(to));
        EXPECT_EQ(request.state(), to);
    }
}

TEST(Request, EveryIllegalTransitionThrowsAndChangesNothing) {
    int checked = 0;
    for (S from : all_states()) {
        for (S to : all_states()) {
            if (legal().count({from, to}) == 1) {
                continue;
            }
            SCOPED_TRACE(name(from) + " -> " + name(to));
            Request request = request_in(from);
            const auto last = request.last_transition_at();
            const auto entered = request.entered_at(to);
            EXPECT_THROW(request.transition_to(to), IllegalTransition);
            EXPECT_EQ(request.state(), from);
            EXPECT_TRUE(request.last_transition_at() == last);
            EXPECT_TRUE(request.entered_at(to) == entered);
            ++checked;
        }
    }
    EXPECT_EQ(checked, 121 - 24);
}

TEST(Request, IllegalTransitionReportsRequestAndStates) {
    Request request = request_in(S::kDecode);
    try {
        request.transition_to(S::kQueued);
        FAIL() << "expected IllegalTransition";
    } catch (const IllegalTransition& e) {
        EXPECT_EQ(e.from(), S::kDecode);
        EXPECT_EQ(e.to(), S::kQueued);
        EXPECT_STREQ(e.what(), "request ns-test: illegal transition DECODE -> QUEUED");
    }
}

TEST(Request, OnlyTerminalStatesAreFinished) {
    for (S state : all_states()) {
        EXPECT_EQ(request_in(state).is_finished(), is_terminal(state)) << name(state);
    }
}

TEST(Request, HappyPathRecordsWhenEachStateWasEntered) {
    Request request("ns-000001", kT0);
    request.transition_to(S::kValidated, kT0 + 1ms);
    request.transition_to(S::kQueued, kT0 + 2ms);
    request.transition_to(S::kAdmitted, kT0 + 3ms);
    request.transition_to(S::kPrefill, kT0 + 4ms);
    request.transition_to(S::kDecode, kT0 + 5ms);
    request.transition_to(S::kCompleted, kT0 + 9ms);

    EXPECT_EQ(ms(request, S::kReceived), 1000);
    EXPECT_EQ(ms(request, S::kValidated), 1001);
    EXPECT_EQ(ms(request, S::kQueued), 1002);
    EXPECT_EQ(ms(request, S::kAdmitted), 1003);
    EXPECT_EQ(ms(request, S::kPrefill), 1004);
    EXPECT_EQ(ms(request, S::kDecode), 1005);
    EXPECT_EQ(ms(request, S::kCompleted), 1009);
    EXPECT_EQ(ms(request.last_transition_at()), 1009);
    for (S unvisited : {S::kPreempted, S::kCancelled, S::kRejected, S::kFailed}) {
        EXPECT_EQ(ms(request, unvisited), -1) << name(unvisited);
    }
    EXPECT_TRUE(request.is_finished());
}

TEST(Request, PreemptionRequeuesAndKeepsTheFirstQueueTime) {
    Request request("ns-000001", kT0);
    int t = 0;
    auto step = [&](S next) { request.transition_to(next, kT0 + std::chrono::milliseconds(++t)); };
    for (S next : {S::kValidated, S::kQueued, S::kAdmitted, S::kPrefill, S::kDecode, S::kPreempted,  // 1..6
                   S::kQueued, S::kAdmitted, S::kPrefill, S::kDecode, S::kPreempted,                // 7..11
                   S::kQueued}) {                                                                    // 12
        step(next);
    }
    EXPECT_EQ(request.state(), S::kQueued);
    EXPECT_EQ(request.preemption_count(), 2u);
    EXPECT_EQ(ms(request, S::kQueued), 1002);     // first entry kept
    EXPECT_EQ(ms(request, S::kPreempted), 1006);  // first preemption
    EXPECT_EQ(ms(request.last_transition_at()), 1012);
}

TEST(Request, PrefillCanCompleteWithoutDecoding) {
    Request request = request_in(S::kPrefill);
    request.transition_to(S::kCompleted);
    EXPECT_TRUE(request.is_finished());
    EXPECT_FALSE(request.entered_at(S::kDecode).has_value());
}

TEST(Request, TimeCannotGoBackwards) {
    Request request("ns-000001", kT0);
    request.transition_to(S::kValidated, kT0 + 5ms);
    EXPECT_THROW(request.transition_to(S::kQueued, kT0 + 4ms), std::invalid_argument);
    EXPECT_EQ(request.state(), S::kValidated);
    EXPECT_NO_THROW(request.transition_to(S::kQueued, kT0 + 5ms));  // equal time is fine
}

TEST(RequestIdGenerator, IdsAreZeroPaddedAndMonotonic) {
    RequestIdGenerator ids;
    EXPECT_EQ(ids.next(), "ns-000001");
    EXPECT_EQ(ids.next(), "ns-000002");

    RequestIdGenerator late(999999);
    EXPECT_EQ(late.next(), "ns-999999");
    EXPECT_EQ(late.next(), "ns-1000000");  // grows instead of wrapping
}

TEST(RequestIdGenerator, IdsAreUniqueAcrossThreads) {
    constexpr std::size_t kThreads = 8;
    constexpr std::size_t kPerThread = 1000;
    RequestIdGenerator ids;
    std::vector<std::vector<std::string>> results(kThreads);
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&ids, &results, t] {
            auto& mine = results[t];
            mine.reserve(kPerThread);
            for (std::size_t i = 0; i < kPerThread; ++i) {
                mine.push_back(ids.next());
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    std::set<std::string> all;
    for (const auto& mine : results) {
        all.insert(mine.begin(), mine.end());
    }
    EXPECT_EQ(all.size(), kThreads * kPerThread);
    EXPECT_EQ(*all.begin(), "ns-000001");
    EXPECT_EQ(*all.rbegin(), "ns-008000");
}

}  // namespace
