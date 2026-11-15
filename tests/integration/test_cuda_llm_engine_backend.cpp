#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "neuroserve/backend/cuda_llm_engine_backend.hpp"

namespace fs = std::filesystem;
using neuroserve::CudaLlmEngineBackend;
using neuroserve::CudaLlmEngineConfig;
using neuroserve::KvCacheExhausted;
using neuroserve::SeqId;
using neuroserve::TokenId;

namespace {

constexpr int kNumPrompts = 8;

std::string env_or(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value != nullptr ? value : fallback;
}

std::vector<TokenId> read_i32(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path.string());
    }
    f.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<TokenId> values(bytes / sizeof(TokenId));
    f.read(reinterpret_cast<char*>(values.data()),
           static_cast<std::streamsize>(values.size() * sizeof(TokenId)));
    return values;
}

struct Reference {
    std::vector<TokenId> prompt;
    std::vector<TokenId> greedy;
};

}  // namespace

class CudaLlmEngineBackendTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const std::string model_dir = env_or("NS_MODEL_DIR", "/models/llama-3.2-1b-fp16");
        const std::string ref_dir = env_or("NS_REF_DIR", "/models/llama-3.2-1b-ref");
        if (!fs::exists(fs::path(model_dir) / "weights.bin") ||
            !fs::exists(fs::path(ref_dir) / "prompt_00_ids.i32")) {
            skip_reason_ = "model or reference files not found (NS_MODEL_DIR=" + model_dir +
                           ", NS_REF_DIR=" + ref_dir + ")";
            return;
        }
        CudaLlmEngineConfig config;
        config.model_dir = model_dir;
        config.max_prompt_tokens = 512;
        config.num_pages = 512;  // 8192 tokens, 256 MiB of KV
        config.max_batch = 8;
        config.max_pages_per_sequence = 64;
        backend_ = std::make_unique<CudaLlmEngineBackend>(config);

        for (int i = 0; i < kNumPrompts; ++i) {
            char stem[32];
            std::snprintf(stem, sizeof(stem), "prompt_%02d", i);
            refs_.push_back({read_i32(fs::path(ref_dir) / (std::string(stem) + "_ids.i32")),
                             read_i32(fs::path(ref_dir) / (std::string(stem) + "_greedy.i32"))});
        }
    }

    static void TearDownTestSuite() {
        backend_.reset();
        refs_.clear();
    }

    void SetUp() override {
        if (!backend_) {
            GTEST_SKIP() << skip_reason_;
        }
    }

    void expect_no_leaked_pages() const {
        const auto kv = backend_->kv_stats();
        EXPECT_EQ(kv.free_pages, kv.total_pages) << "KV pages leaked";
    }

    static inline std::unique_ptr<CudaLlmEngineBackend> backend_;
    static inline std::vector<Reference> refs_;
    static inline std::string skip_reason_;
};

TEST_F(CudaLlmEngineBackendTest, ModelInfoMatchesLlama32_1B) {
    const auto& info = backend_->model_info();
    EXPECT_EQ(info.n_layers, 16);
    EXPECT_EQ(info.n_kv_heads, 8);
    EXPECT_EQ(info.head_dim, 64);
    EXPECT_EQ(info.vocab_size, 128256);
    EXPECT_EQ(info.kv_bytes_per_token, 32768u);  // 2 x 16 x 8 x 64 x 2 bytes
    EXPECT_EQ(backend_->kv_stats().page_size_tokens, 16);
    expect_no_leaked_pages();
}

TEST_F(CudaLlmEngineBackendTest, SinglePromptMatchesHuggingFace) {
    const Reference& ref = refs_.at(0);
    const SeqId id = backend_->create_sequence();

    std::vector<TokenId> out{backend_->prefill(id, ref.prompt)};
    while (out.size() < ref.greedy.size()) {
        const SeqId ids[] = {id};
        const TokenId last[] = {out.back()};
        out.push_back(backend_->decode_step(ids, last).at(0));
    }

    EXPECT_EQ(out, ref.greedy);
    EXPECT_EQ(backend_->sequence_length(id),
              static_cast<int>(ref.prompt.size() + ref.greedy.size() - 1));
    backend_->free_sequence(id);
    expect_no_leaked_pages();
}

TEST_F(CudaLlmEngineBackendTest, AllPromptsDecodedTogetherMatchHuggingFace) {
    std::vector<SeqId> ids;
    std::vector<std::vector<TokenId>> outs(kNumPrompts);
    for (int i = 0; i < kNumPrompts; ++i) {
        ids.push_back(backend_->create_sequence());
        outs[i].push_back(backend_->prefill(ids[i], refs_[i].prompt));
    }

    const std::size_t n_new = refs_[0].greedy.size();
    for (std::size_t step = 1; step < n_new; ++step) {
        std::vector<TokenId> last;
        for (const auto& o : outs) {
            last.push_back(o.back());
        }
        const std::vector<TokenId> next = backend_->decode_step(ids, last);
        ASSERT_EQ(next.size(), ids.size());
        for (int i = 0; i < kNumPrompts; ++i) {
            outs[i].push_back(next[i]);
        }
    }

    for (int i = 0; i < kNumPrompts; ++i) {
        EXPECT_EQ(outs[i], refs_[i].greedy) << "prompt " << i << " (" << refs_[i].prompt.size() << " tokens)";
    }
    for (const SeqId id : ids) {
        backend_->free_sequence(id);
    }
    expect_no_leaked_pages();
}

TEST_F(CudaLlmEngineBackendTest, RejectsBadRequestsWithoutSideEffects) {
    const int free_before = backend_->kv_stats().free_pages;
    const std::vector<TokenId>& good = refs_[0].prompt;

    EXPECT_THROW(backend_->prefill(9999, good), std::invalid_argument);  // unknown sequence

    const SeqId id = backend_->create_sequence();
    EXPECT_THROW(backend_->prefill(id, std::vector<TokenId>{}), std::invalid_argument);         // empty
    EXPECT_THROW(backend_->prefill(id, std::vector<TokenId>(600, 1000)), std::length_error);    // too long
    EXPECT_THROW(backend_->prefill(id, std::vector<TokenId>{1, -5}), std::invalid_argument);    // bad token
    {
        const SeqId ids[] = {id};
        const TokenId toks[] = {1};
        EXPECT_THROW(backend_->decode_step(ids, toks), std::logic_error);  // not prefilled yet
    }
    EXPECT_EQ(backend_->sequence_length(id), 0);

    ASSERT_NO_THROW(backend_->prefill(id, good));
    EXPECT_THROW(backend_->prefill(id, good), std::logic_error);  // already prefilled
    {
        const SeqId ids[] = {id, id};
        const TokenId toks[] = {1, 1};
        EXPECT_THROW(backend_->decode_step(ids, toks), std::invalid_argument);  // duplicate
    }
    {
        const SeqId ids[] = {id};
        const TokenId toks[] = {1, 2};
        EXPECT_THROW(backend_->decode_step(ids, toks), std::invalid_argument);  // size mismatch
    }

    backend_->free_sequence(id);
    EXPECT_THROW(backend_->free_sequence(id), std::invalid_argument);  // double free
    EXPECT_EQ(backend_->kv_stats().free_pages, free_before);
}

TEST_F(CudaLlmEngineBackendTest, ReportsKvExhaustionInsteadOfCrashing) {
    const std::vector<TokenId> prompt(400, 1000);  // exactly 25 pages
    const int pages_per_prompt = 25;

    std::vector<SeqId> live;
    while (backend_->kv_stats().free_pages >= pages_per_prompt) {
        const SeqId id = backend_->create_sequence();
        backend_->prefill(id, prompt);
        live.push_back(id);
    }

    const int free_before = backend_->kv_stats().free_pages;
    const SeqId extra = backend_->create_sequence();
    EXPECT_THROW(backend_->prefill(extra, prompt), KvCacheExhausted);
    EXPECT_EQ(backend_->kv_stats().free_pages, free_before);  // nothing allocated
    EXPECT_EQ(backend_->sequence_length(extra), 0);

    backend_->free_sequence(extra);
    for (const SeqId id : live) {
        backend_->free_sequence(id);
    }
    expect_no_leaked_pages();
}