#pragma once
// KV-cache memory from the model's config (docs/kv-cache.md, "Physical memory"; Day 39, SCH-008):
// how many bytes a token's K and V take, read from a Hugging Face config.json, and the admission
// modes that decide how much of that memory a request must find free before it starts.

#include <cstdint>
#include <string>
#include <string_view>

namespace neuroserve::core {

// What the KV cache's size depends on, as config.json gives it.
struct ModelShape {
    int num_layers = 0;    // num_hidden_layers
    int num_kv_heads = 0;  // num_key_value_heads (num_attention_heads without grouped-query attention)
    int head_dim = 0;      // head_dim, or hidden_size / num_attention_heads
    std::string dtype;     // torch_dtype: float16, bfloat16, float32 (empty: not given)
};

// Reads the fields above from a config.json's text. Only top-level keys are read; nested objects
// (rope_scaling, quantization_config) are skipped. std::invalid_argument if the text is not a JSON
// object or a required field is missing or not a positive integer.
ModelShape parse_model_config(std::string_view json);

// Bytes of one element of `dtype`: 2 for float16 / fp16 / bfloat16 / bf16, 4 for float32 / fp32.
// std::invalid_argument otherwise.
int dtype_bytes(std::string_view dtype);

// K and V for one token in every layer: 2 x layers x kv_heads x head_dim x element bytes. The
// engine stores K/V in `kv_dtype` (FP16 for cuda-llm-engine), whatever dtype the weights came in.
std::int64_t kv_bytes_per_token(const ModelShape& shape, std::string_view kv_dtype);
std::int64_t kv_bytes_per_block(const ModelShape& shape, std::string_view kv_dtype, int block_size);

// How much KV memory a request must find free to be admitted.
enum class KvAdmission {
    // Its prefill's pages now; it grows on demand, and a shortage later is met by preemption
    // (ADR-0004). Fits the most requests; preempts when they all grow at once.
    kOnDemand,
    // Its worst case: the pages for prompt + max_new_tokens, reserved for as long as it runs. A
    // running request can then always grow, so nothing is ever preempted; fewer run at once.
    kWorstCase,
};

std::string_view admission_name(KvAdmission mode) noexcept;  // "on-demand", "worst-case"
KvAdmission parse_admission(std::string_view name);         // std::invalid_argument if unknown

}  // namespace neuroserve::core
