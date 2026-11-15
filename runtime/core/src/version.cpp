#include "neuroserve/core/version.hpp"

namespace neuroserve {

std::string_view version() noexcept { return NS_VERSION_STRING; }

bool cuda_enabled() noexcept {
#ifdef NS_WITH_CUDA
    return true;
#else
    return false;
#endif
}

}  // namespace neuroserve