#pragma once
// GPU memory by owner (Day 58, GPU-004; docs/gpu-memory.md). The ledger only counts: the CUDA
// backend's GpuMemoryManager allocates from a pool and records here, and records the engine's own
// allocations (made inside cuda-llm-engine) as external entries. No CUDA here, so it is tested on
// the CPU.

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace neuroserve::core {

enum class MemoryOwner { kWeights, kWorkspace, kKv, kOther };
inline constexpr std::size_t kMemoryOwners = 4;
std::string_view to_string(MemoryOwner owner);

// kPool: allocated through the manager's pool. kExternal: allocated elsewhere (the engine) and
// recorded so every byte has an owner.
enum class MemorySource { kPool, kExternal };

class MemoryLedger {
public:
    using Id = std::uint64_t;

    // Thread-safe. Returns the id that release() takes.
    Id record(MemoryOwner owner, std::size_t bytes, MemorySource source, std::string label);
    void release(Id id);  // std::invalid_argument for an id not recorded, or released already

    struct OwnerUsage {
        std::size_t bytes = 0;           // live now
        std::size_t pool_bytes = 0;      // of which through the pool
        std::size_t external_bytes = 0;  // of which recorded from elsewhere
        std::size_t peak_bytes = 0;      // the most live at once
        std::size_t allocations = 0;     // live now
    };
    // Live bytes by label: what each owner's bytes are made of.
    struct Item {
        MemoryOwner owner = MemoryOwner::kOther;
        MemorySource source = MemorySource::kPool;
        std::string label;
        std::size_t bytes = 0;
        std::size_t allocations = 0;
    };
    struct Report {
        std::array<OwnerUsage, kMemoryOwners> owners{};
        std::size_t bytes = 0, pool_bytes = 0, external_bytes = 0, peak_bytes = 0;
        std::vector<Item> items;  // ordered by owner, then label
    };
    Report report() const;

private:
    struct Entry {
        MemoryOwner owner;
        MemorySource source;
        std::string label;
        std::size_t bytes;
    };
    mutable std::mutex mu_;
    Id next_id_ = 1;
    std::unordered_map<Id, Entry> live_;
    std::array<OwnerUsage, kMemoryOwners> owners_{};
    std::size_t bytes_ = 0, peak_bytes_ = 0;
};

}  // namespace neuroserve::core
