#include "neuroserve/backend/gpu_memory.hpp"

#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>

#include <cuda_runtime.h>

namespace neuroserve {

struct GpuMemoryManager::Impl {
    int device = 0;
    bool pooled = false;
    cudaMemPool_t pool = nullptr;
    std::size_t threshold = 0;
    core::MemoryLedger ledger;
    std::mutex mu;
    std::unordered_map<void*, core::MemoryLedger::Id> live;  // allocate()'s pointers
};

namespace {
void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("GpuMemoryManager: ") + what + ": " + cudaGetErrorString(err));
    }
}
}  // namespace

GpuMemoryManager::GpuMemoryManager(int device, std::size_t release_threshold_bytes) : impl_(std::make_unique<Impl>()) {
    impl_->device = device;
    impl_->threshold = release_threshold_bytes;
    int supported = 0;
    check(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device), "cudaDeviceGetAttribute");
    if (supported != 0) {
        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = device;
        check(cudaMemPoolCreate(&impl_->pool, &props), "cudaMemPoolCreate");
        std::uint64_t threshold = release_threshold_bytes;  // cuuint64_t
        check(cudaMemPoolSetAttribute(impl_->pool, cudaMemPoolAttrReleaseThreshold, &threshold),
              "cudaMemPoolSetAttribute(release threshold)");
        impl_->pooled = true;
    }
}

GpuMemoryManager::~GpuMemoryManager() {
    {
        std::lock_guard lock(impl_->mu);
        for (const auto& [ptr, id] : impl_->live) {  // a leak: give it back rather than lose it
            if (impl_->pooled) {
                cudaFreeAsync(ptr, nullptr);
            } else {
                cudaFree(ptr);
            }
        }
    }
    if (impl_->pooled) {
        cudaStreamSynchronize(nullptr);
        cudaMemPoolDestroy(impl_->pool);
    }
}

void* GpuMemoryManager::allocate(core::MemoryOwner owner, std::size_t bytes, std::string label) {
    void* ptr = nullptr;
    const cudaError_t err = impl_->pooled ? cudaMallocFromPoolAsync(&ptr, bytes, impl_->pool, nullptr)
                                          : cudaMalloc(&ptr, bytes);
    if (err == cudaErrorMemoryAllocation) {
        cudaGetLastError();  // clear it
        throw std::bad_alloc();
    }
    check(err, "allocate");
    const auto id = impl_->ledger.record(owner, bytes, core::MemorySource::kPool, std::move(label));
    std::lock_guard lock(impl_->mu);
    impl_->live.emplace(ptr, id);
    return ptr;
}

void GpuMemoryManager::free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    core::MemoryLedger::Id id = 0;
    {
        std::lock_guard lock(impl_->mu);
        const auto it = impl_->live.find(ptr);
        if (it == impl_->live.end()) {
            throw std::invalid_argument("GpuMemoryManager::free: a pointer it did not allocate");
        }
        id = it->second;
        impl_->live.erase(it);
    }
    check(impl_->pooled ? cudaFreeAsync(ptr, nullptr) : cudaFree(ptr), "free");
    impl_->ledger.release(id);
}

core::MemoryLedger::Id GpuMemoryManager::record_external(core::MemoryOwner owner, std::size_t bytes,
                                                          std::string label) {
    return impl_->ledger.record(owner, bytes, core::MemorySource::kExternal, std::move(label));
}

void GpuMemoryManager::release_external(core::MemoryLedger::Id id) { impl_->ledger.release(id); }

GpuMemoryManager::Report GpuMemoryManager::report() const {
    Report r;
    r.ledger = impl_->ledger.report();
    r.pool.supported = impl_->pooled;
    r.pool.release_threshold_bytes = impl_->threshold;
    if (impl_->pooled) {
        std::uint64_t v = 0;  // cuuint64_t
        check(cudaMemPoolGetAttribute(impl_->pool, cudaMemPoolAttrReservedMemCurrent, &v), "pool reserved");
        r.pool.reserved_bytes = static_cast<std::size_t>(v);
        check(cudaMemPoolGetAttribute(impl_->pool, cudaMemPoolAttrUsedMemCurrent, &v), "pool used");
        r.pool.used_bytes = static_cast<std::size_t>(v);
        check(cudaMemPoolGetAttribute(impl_->pool, cudaMemPoolAttrReservedMemHigh, &v), "pool peak");
        r.pool.reserved_peak_bytes = static_cast<std::size_t>(v);
    } else {
        r.pool.reserved_bytes = r.pool.used_bytes = r.ledger.pool_bytes;
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    r.device_total_bytes = total_bytes;
    r.device_used_bytes = total_bytes - free_bytes;
    r.unattributed_bytes = static_cast<std::int64_t>(r.device_used_bytes) -
                           static_cast<std::int64_t>(r.ledger.external_bytes + r.pool.reserved_bytes);
    return r;
}

}  // namespace neuroserve
