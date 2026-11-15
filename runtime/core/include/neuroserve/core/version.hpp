#pragma once

#include <string_view>

namespace neuroserve {

// Semantic version of this build, e.g. "0.0.1".
std::string_view version() noexcept;

// True when the library was compiled with CUDA support.
bool cuda_enabled() noexcept;

}  // namespace neuroserve