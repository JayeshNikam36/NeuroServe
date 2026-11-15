#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "fake_backend.hpp"
#include "neuroserve/core/generate.hpp"

using neuroserve::TokenId;
using neuroserve::generate_greedy;
using neuroserve::test_support::FakeBackend;

TEST(GenerateGreedy, ReturnsRequestedNumberOfTokens) {
    FakeBackend backend;
    const std::vector<TokenId> prompt{5, 6, 7};
    EXPECT_EQ(generate_greedy(backend, prompt, 4), (std::vector<TokenId>{8, 9, 10, 11}));
    EXPECT_EQ(backend.live_sequences(), 0);
}

TEST(GenerateGreedy, OneTokenUsesOnlyPrefill) {
    FakeBackend backend;
    const std::vector<TokenId> prompt{41};
    EXPECT_EQ(generate_greedy(backend, prompt, 1), (std::vector<TokenId>{42}));
    EXPECT_EQ(backend.decode_calls(), 0);
}

TEST(GenerateGreedy, RejectsNonPositiveMaxTokens) {
    FakeBackend backend;
    const std::vector<TokenId> prompt{1};
    EXPECT_THROW(generate_greedy(backend, prompt, 0), std::invalid_argument);
    EXPECT_EQ(backend.live_sequences(), 0);
}

TEST(GenerateGreedy, FreesSequenceWhenDecodeFails) {
    FakeBackend backend;
    backend.fail_decode_after(2);
    const std::vector<TokenId> prompt{1, 2, 3};
    EXPECT_THROW(generate_greedy(backend, prompt, 10), std::runtime_error);
    EXPECT_EQ(backend.live_sequences(), 0) << "sequence leaked after an exception";
}

TEST(GenerateGreedy, FreesSequenceWhenPrefillFails) {
    FakeBackend backend;
    EXPECT_THROW(generate_greedy(backend, std::vector<TokenId>{}, 3), std::invalid_argument);
    EXPECT_EQ(backend.live_sequences(), 0);
}