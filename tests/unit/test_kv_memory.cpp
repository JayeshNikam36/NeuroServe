// KV memory from the model config and the admission modes (Day 39, SCH-008).
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "fake_backend.hpp"
#include "prefill_gate.hpp"
#include "neuroserve/core/engine_loop.hpp"
#include "neuroserve/core/kv_memory.hpp"

namespace {

using neuroserve::TokenId;
using neuroserve::core::EngineLoop;
using neuroserve::core::EngineLoopOptions;
using neuroserve::core::GenerationParams;
using neuroserve::core::KvAdmission;
using neuroserve::core::ModelShape;
using neuroserve::test_support::FakeBackend;
using neuroserve::test_support::FirstPrefillGate;
using namespace std::chrono_literals;
namespace core = neuroserve::core;

// meta-llama/Llama-3.2-1B's config.json, as published.
constexpr std::string_view kLlama32_1B = R"({
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 16,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.0.dev0",
  "use_cache": true,
  "vocab_size": 128256
})";

TEST(KvMemory, Llama32_1BInFp16Takes32KiBPerTokenAnd512KiBPerBlock) {
    const ModelShape shape = core::parse_model_config(kLlama32_1B);
    EXPECT_EQ(shape.num_layers, 16);
    EXPECT_EQ(shape.num_kv_heads, 8);
    EXPECT_EQ(shape.head_dim, 64);
    EXPECT_EQ(shape.dtype, "bfloat16");
    EXPECT_EQ(core::kv_bytes_per_token(shape, "fp16"), 32 * 1024) << "2 (K and V) x 16 layers x 8 heads x 64 x 2 bytes";
    EXPECT_EQ(core::kv_bytes_per_block(shape, "fp16", 16), 512 * 1024);
    EXPECT_EQ(core::kv_bytes_per_token(shape, shape.dtype), 32 * 1024) << "bfloat16 is 2 bytes too";
    EXPECT_EQ(core::kv_bytes_per_token(shape, "float32"), 64 * 1024);
}

TEST(KvMemory, MissingOptionalFieldsHaveTheirDefaults) {
    // No head_dim: hidden_size / num_attention_heads. No num_key_value_heads: no grouped-query
    // attention, one KV head per query head.
    const ModelShape shape = core::parse_model_config(
        R"({"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 512})");
    EXPECT_EQ(shape.head_dim, 64);
    EXPECT_EQ(shape.num_kv_heads, 8);
    EXPECT_TRUE(shape.dtype.empty());
}

TEST(KvMemory, OnlyTopLevelKeysAreRead) {
    const ModelShape shape = core::parse_model_config(R"({
        "text_config": {"num_hidden_layers": 99, "nested": [1, {"head_dim": 7}]},
        "name": "with \"quotes\" and {braces}",
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32})");
    EXPECT_EQ(shape.num_layers, 2);
    EXPECT_EQ(shape.head_dim, 32);
}

TEST(KvMemory, ABrokenConfigIsRejectedWithItsReason) {
    for (const std::string_view bad : {
             std::string_view(R"({"num_attention_heads": 8, "hidden_size": 512})"),                           // no layers
             std::string_view(R"({"num_hidden_layers": "16", "num_attention_heads": 8, "head_dim": 64})"),   // a string
             std::string_view(R"({"num_hidden_layers": 0, "num_attention_heads": 8, "head_dim": 64})"),      // zero
             std::string_view(R"({"num_hidden_layers": 2.5, "num_attention_heads": 8, "head_dim": 64})"),    // not whole
             std::string_view(R"({"num_hidden_layers": 2, "num_attention_heads": 8, "num_key_value_heads": 3, "head_dim": 64})"),
             std::string_view(R"([1, 2])"),
             std::string_view(R"({"num_hidden_layers": 2,)"),
         }) {
        EXPECT_THROW(core::parse_model_config(bad), std::invalid_argument) << bad;
    }
    EXPECT_THROW(core::dtype_bytes("int8"), std::invalid_argument);
}

TEST(KvMemory, AdmissionModesHaveNames) {
    EXPECT_EQ(core::parse_admission("on-demand"), KvAdmission::kOnDemand);
    EXPECT_EQ(core::parse_admission("worst-case"), KvAdmission::kWorstCase);
    EXPECT_EQ(core::admission_name(KvAdmission::kWorstCase), "worst-case");
    EXPECT_THROW(core::parse_admission("greedy"), std::invalid_argument);
}

// ---- Both admission modes through the engine: 4 requests of 20 + 60 tokens (5 pages each, 20
// together) on a 10-page pool.

GenerationParams request(int r) {
    GenerationParams p;
    p.prompt = std::vector<TokenId>(20, static_cast<TokenId>(100 * (r + 1)));
    p.max_new_tokens = 60;
    return p;
}

std::vector<TokenId> expected(const GenerationParams& p) {
    std::vector<TokenId> out;
    for (int i = 1; i <= p.max_new_tokens; ++i) {
        out.push_back((p.prompt.back() + i) % 1000);
    }
    return out;
}

struct Outcome {
    std::uint64_t preemptions = 0;
    std::size_t widest_batch = 0;
};

Outcome run(KvAdmission mode, bool prefix_caching = false) {
    FakeBackend backend(1000, 10);
    backend.prefix_caching = prefix_caching;
    Outcome out;
    backend.on_decode = [&](std::size_t batch) { out.widest_batch = std::max(out.widest_batch, batch); };
    EngineLoopOptions options;
    options.admission = mode;
    options.check_invariants = true;
    FirstPrefillGate gate(backend);
    EngineLoop loop(backend, options);
    std::vector<core::RequestHandle> handles;
    for (int r = 0; r < 4; ++r) {
        handles.push_back(loop.submit(request(r)));
    }
    gate.open();
    for (int r = 0; r < 4; ++r) {
        EXPECT_TRUE(handles[static_cast<std::size_t>(r)].wait_for(10s));
        EXPECT_EQ(handles[static_cast<std::size_t>(r)].wait().tokens, expected(request(r))) << "request " << r;
    }
    EXPECT_FALSE(loop.stop_requested()) << "every step's invariant checks passed";
    const auto kv = loop.kv_stats();
    EXPECT_EQ(kv.free_pages, kv.total_pages);
    out.preemptions = loop.stats().preemptions;
    return out;
}

TEST(KvAdmissionModes, OnDemandRunsMoreAtOnceAndPreemptsWhenTheyGrow) {
    const Outcome o = run(KvAdmission::kOnDemand);
    EXPECT_GT(o.widest_batch, 2u) << "prefills need 2 pages each: more than two start together";
    EXPECT_GT(o.preemptions, 0u) << "then they grow past the pool";
}

TEST(KvAdmissionModes, WorstCaseReservesEveryRequestsWholeLengthAndNeverPreempts) {
    for (const bool caching : {false, true}) {
        const Outcome o = run(KvAdmission::kWorstCase, caching);
        EXPECT_EQ(o.preemptions, 0u) << "prefix caching " << caching;
        EXPECT_LE(o.widest_batch, 2u) << "5 pages reserved each: at most two fit in 10";
        EXPECT_EQ(o.widest_batch, 2u) << "and two do run together";
    }
}

}  // namespace
