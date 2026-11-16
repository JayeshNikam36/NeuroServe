// Structured JSON logging: format, escaping, levels, thread safety, and the engine loop's lines.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "fake_backend.hpp"
#include "neuroserve/core/engine_loop.hpp"
#include "neuroserve/core/log.hpp"

namespace {

namespace log = neuroserve::core::log;
using namespace std::chrono_literals;

// Captures log lines for one test and restores the defaults afterwards.
class LogCapture : public ::testing::Test {
protected:
    void SetUp() override {
        log::set_sink([this](std::string_view line) {
            std::lock_guard lock(mu_);
            lines_.emplace_back(line);
        });
        log::set_level(log::Level::kDebug);
    }
    void TearDown() override {
        log::set_sink(nullptr);
        log::set_level(log::Level::kWarning);
    }
    std::vector<std::string> lines() {
        std::lock_guard lock(mu_);
        return lines_;
    }

private:
    std::mutex mu_;
    std::vector<std::string> lines_;
};

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

const auto kTime = std::chrono::system_clock::time_point{} + 1'763'000'000s + 94ms;  // 2025-11-13T02:13:20.094Z

TEST(Log, FormatsOneJsonObjectWithTheServerSchema) {
    const std::string line = log::format_line(log::Level::kInfo, "neuroserve.engine", "request completed",
                                              {{"engine_request_id", "ns-000001"}, {"completion_tokens", 4}},
                                              kTime);
    EXPECT_EQ(line,
              "{\"ts\": \"2025-11-13T02:13:20.094+00:00\", \"level\": \"INFO\", \"logger\": \"neuroserve.engine\", "
              "\"msg\": \"request completed\", \"engine_request_id\": \"ns-000001\", \"completion_tokens\": 4}");
}

TEST(Log, EscapesTextSoEveryLineIsValidJson) {
    const std::string line = log::format_line(log::Level::kError, "x", "quote \" backslash \\ newline \n tab \t bell \a",
                                              {{"key \"k\"", "v\nw"}}, kTime);
    EXPECT_TRUE(contains(line, R"(quote \" backslash \\ newline \n tab \t bell \u0007)")) << line;
    EXPECT_TRUE(contains(line, R"("key \"k\"": "v\nw")")) << line;
    EXPECT_EQ(std::count(line.begin(), line.end(), '\n'), 0) << "a raw newline would split the log line";
}

TEST(Log, FormatsEveryValueType) {
    const double inf = std::numeric_limits<double>::infinity();
    const std::string line = log::format_line(
        log::Level::kDebug, "x", "m",
        {{"i", -7}, {"u", std::size_t{42}}, {"d", 434.37}, {"b", true}, {"s", std::string("ok")}, {"bad", inf}},
        kTime);
    EXPECT_TRUE(contains(line, R"("i": -7, "u": 42, "d": 434.37, "b": true, "s": "ok", "bad": null})")) << line;
}

TEST(Log, ParsesPythonLevelNames) {
    EXPECT_EQ(log::parse_level("DEBUG"), log::Level::kDebug);
    EXPECT_EQ(log::parse_level("INFO"), log::Level::kInfo);
    EXPECT_EQ(log::parse_level("WARNING"), log::Level::kWarning);
    EXPECT_EQ(log::parse_level("ERROR"), log::Level::kError);
    EXPECT_THROW(log::parse_level("info"), std::invalid_argument);
}

TEST_F(LogCapture, LevelFiltersLines) {
    log::set_level(log::Level::kWarning);
    log::write(log::Level::kInfo, "x", "dropped");
    log::write(log::Level::kWarning, "x", "kept");
    log::write(log::Level::kError, "x", "kept too");
    const auto captured = lines();
    ASSERT_EQ(captured.size(), 2u);
    EXPECT_TRUE(contains(captured[0], "\"msg\": \"kept\""));
}

TEST_F(LogCapture, LinesFromManyThreadsNeverInterleave) {
    constexpr int kThreads = 8;
    constexpr int kPerThread = 200;
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([t] {
            for (int i = 0; i < kPerThread; ++i) {
                log::write(log::Level::kInfo, "x", "line", {{"thread", t}, {"i", i}});
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    const auto captured = lines();
    ASSERT_EQ(captured.size(), static_cast<std::size_t>(kThreads * kPerThread));
    for (const auto& line : captured) {
        ASSERT_TRUE(line.front() == '{' && line.back() == '}' && std::count(line.begin(), line.end(), '{') == 1)
            << line;
    }
}

TEST_F(LogCapture, EngineLoopLogsEachRequestWithBothIdsAndNoTokens) {
    neuroserve::test_support::FakeBackend backend;
    neuroserve::core::EngineLoop loop(backend);
    neuroserve::core::GenerationParams params;
    params.prompt = {771, 772, 773};
    params.max_new_tokens = 4;
    params.client_request_id = "req_abc";
    ASSERT_TRUE(loop.submit(params).wait_for(10s));
    loop.stop();

    const auto captured = lines();
    const auto it = std::find_if(captured.begin(), captured.end(),
                                 [](const std::string& l) { return contains(l, "\"msg\": \"request completed\""); });
    ASSERT_NE(it, captured.end());
    EXPECT_TRUE(contains(*it, "\"logger\": \"neuroserve.engine\""));
    EXPECT_TRUE(contains(*it, "\"engine_request_id\": \"ns-000001\""));
    EXPECT_TRUE(contains(*it, "\"request_id\": \"req_abc\""));  // links to the HTTP access log line
    EXPECT_TRUE(contains(*it, "\"completion_tokens\": 4"));
    EXPECT_TRUE(contains(*it, "\"finish_reason\": \"length\""));
    for (const auto& line : captured) {
        EXPECT_FALSE(contains(line, "771")) << "prompt tokens must never be logged: " << line;
    }
}

TEST(EngineLoopKv, SnapshotShowsPagesReturnedWhenWaitReturns) {
    neuroserve::test_support::FakeBackend backend;
    neuroserve::core::EngineLoop loop(backend);
    const auto before = loop.kv_stats();
    EXPECT_EQ(before.free_pages, before.total_pages);
    neuroserve::core::GenerationParams params;
    params.prompt = std::vector<neuroserve::TokenId>(40, 1);
    params.max_new_tokens = 30;
    ASSERT_TRUE(loop.submit(params).wait_for(10s));
    const auto after = loop.kv_stats();  // read while the engine thread is still running
    EXPECT_EQ(after.free_pages, after.total_pages);
}

}  // namespace
