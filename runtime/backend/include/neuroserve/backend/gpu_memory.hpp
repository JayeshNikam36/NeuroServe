#pragma once
// GPU memory manager with pools and owner tags (Day 58, GPU-004; docs/gpu-memory.md).
//
// Everything NeuroServe allocates on the GPU itself goes through allocate(): a stream-ordered
// memory pool (cudaMemPool_t, cudaMallocFromPoolAsync) with a release threshold, every allocation
// tagged with its owner. What cuda-llm-engine allocates inside its own calls (weights, KV pool,
// activations) cannot go through it without changing the engine; the backend records those with
// record_external(), so the report attributes every byte it knows of to an owner, and shows the
// rest of the device's used memory (the CUDA context, cuBLAS, other processes) as unattributed.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "neuroserve/core/memory_ledger.hpp"

namespace neuroserve {

class GpuMemoryManager {
public:
    // A pool on `device` that keeps up to release_threshold_bytes of freed memory for reuse and
    // returns the rest to the driver at the next synchronization. If the device has no memory
    // pools, allocate() falls back to cudaMalloc (report().pool.supported is false).
    GpuMemoryManager(int device, std::size_t release_threshold_bytes);
    ~GpuMemoryManager();  // every allocate() must have been freed
    GpuMemoryManager(const GpuMemoryManager&) = delete;
    GpuMemoryManager& operator=(const GpuMemoryManager&) = delete;

    // Ordered on the legacy default stream (the backend's work, and blocking streams, follow it).
    // Throws std::bad_alloc if the device is out of memory. Thread-safe.
    void* allocate(core::MemoryOwner owner, std::size_t bytes, std::string label);
    void free(void* ptr);  // nullptr is ignored; std::invalid_argument for a pointer not allocated here

    core::MemoryLedger::Id record_external(core::MemoryOwner owner, std::size_t bytes, std::string label);
    void release_external(core::MemoryLedger::Id id);

    struct Report {
        core::MemoryLedger::Report ledger;
        struct Pool {
            bool supported = false;
            std::size_t release_threshold_bytes = 0;
            std::size_t reserved_bytes = 0;       // held from the driver now (in use + kept for reuse)
            std::size_t used_bytes = 0;           // in use now
            std::size_t reserved_peak_bytes = 0;
        } pool;
        std::size_t device_total_bytes = 0;
        std::size_t device_used_bytes = 0;  // total - free, every process
        // device_used - (external bytes + pool reserved bytes): the CUDA context, cuBLAS's own
        // memory, allocations no one recorded, other processes. Negative if another process freed
        // memory between the readings.
        std::int64_t unattributed_bytes = 0;
    };
    Report report() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace neuroserve
