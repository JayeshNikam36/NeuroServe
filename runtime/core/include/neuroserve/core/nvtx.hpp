#pragma once

// NVTX ranges, so Nsight Systems shows the engine step's phases on its timeline next to the
// kernels they launch (docs/profiling.md). Compiled in only in CUDA builds, where the toolkit's
// header-only NVTX 3 is available (NS_NVTX); elsewhere a range is an empty object.

#if defined(NS_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif

namespace neuroserve::core {

class NvtxRange {
public:
#if defined(NS_NVTX)
    explicit NvtxRange(const char* name) noexcept { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
#else
    explicit NvtxRange(const char* /*name*/) noexcept {}
#endif
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace neuroserve::core
