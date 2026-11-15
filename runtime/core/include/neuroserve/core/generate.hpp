#pragma once

#include <span>
#include <vector>

#include "neuroserve/core/inference_backend.hpp"

namespace neuroserve {

// Greedy generation for one prompt. Returns exactly max_new_tokens token ids
// (stop tokens arrive with the scheduler). The sequence is always freed, also on error.
std::vector<TokenId> generate_greedy(InferenceBackend& backend,
                                     std::span<const TokenId> prompt,
                                     int max_new_tokens);

}  // namespace neuroserve