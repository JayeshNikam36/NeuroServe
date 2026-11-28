#include "neuroserve/core/memory_ledger.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>

namespace neuroserve::core {

std::string_view to_string(MemoryOwner owner) {
    switch (owner) {
        case MemoryOwner::kWeights: return "weights";
        case MemoryOwner::kWorkspace: return "workspace";
        case MemoryOwner::kKv: return "kv";
        case MemoryOwner::kOther: return "other";
    }
    return "other";
}

MemoryLedger::Id MemoryLedger::record(MemoryOwner owner, std::size_t bytes, MemorySource source, std::string label) {
    std::lock_guard lock(mu_);
    const Id id = next_id_++;
    live_.emplace(id, Entry{owner, source, std::move(label), bytes});
    OwnerUsage& u = owners_[static_cast<std::size_t>(owner)];
    u.bytes += bytes;
    (source == MemorySource::kPool ? u.pool_bytes : u.external_bytes) += bytes;
    u.peak_bytes = std::max(u.peak_bytes, u.bytes);
    ++u.allocations;
    bytes_ += bytes;
    peak_bytes_ = std::max(peak_bytes_, bytes_);
    return id;
}

void MemoryLedger::release(Id id) {
    std::lock_guard lock(mu_);
    const auto it = live_.find(id);
    if (it == live_.end()) {
        throw std::invalid_argument("MemoryLedger::release: no live allocation " + std::to_string(id));
    }
    const Entry& e = it->second;
    OwnerUsage& u = owners_[static_cast<std::size_t>(e.owner)];
    u.bytes -= e.bytes;
    (e.source == MemorySource::kPool ? u.pool_bytes : u.external_bytes) -= e.bytes;
    --u.allocations;
    bytes_ -= e.bytes;
    live_.erase(it);
}

MemoryLedger::Report MemoryLedger::report() const {
    std::lock_guard lock(mu_);
    Report r;
    r.owners = owners_;
    r.bytes = bytes_;
    r.peak_bytes = peak_bytes_;
    std::map<std::tuple<int, int, std::string>, Item> items;
    for (const auto& [id, e] : live_) {
        (e.source == MemorySource::kPool ? r.pool_bytes : r.external_bytes) += e.bytes;
        Item& item = items[{static_cast<int>(e.owner), static_cast<int>(e.source), e.label}];
        item.owner = e.owner;
        item.source = e.source;
        item.label = e.label;
        item.bytes += e.bytes;
        ++item.allocations;
    }
    for (auto& [key, item] : items) {
        r.items.push_back(std::move(item));
    }
    return r;
}

}  // namespace neuroserve::core
