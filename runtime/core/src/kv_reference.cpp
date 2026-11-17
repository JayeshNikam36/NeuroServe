#include "neuroserve/core/kv_reference.hpp"

#include <algorithm>
#include <string>

namespace neuroserve::core {

namespace {

std::string seq_name(SeqId seq) { return "sequence " + std::to_string(seq); }

}  // namespace

ReferenceBlockManager::ReferenceBlockManager(KvBlockConfig config) : config_(config) {
    if (config_.num_blocks <= 0 || config_.block_size <= 0 || config_.max_blocks_per_sequence <= 0) {
        throw std::invalid_argument("KV block config: num_blocks, block_size and max_blocks_per_sequence must be positive");
    }
    refs_.assign(static_cast<std::size_t>(config_.num_blocks), 0);
}

const ReferenceBlockManager::Sequence& ReferenceBlockManager::get(SeqId seq) const {
    const auto it = sequences_.find(seq);
    if (it == sequences_.end()) {
        throw std::invalid_argument("unknown " + seq_name(seq));
    }
    return it->second;
}

ReferenceBlockManager::Sequence& ReferenceBlockManager::get(SeqId seq) {
    return const_cast<Sequence&>(static_cast<const ReferenceBlockManager*>(this)->get(seq));
}

int ReferenceBlockManager::blocks_for(long long tokens) const {
    return static_cast<int>((tokens + config_.block_size - 1) / config_.block_size);
}

void ReferenceBlockManager::create(SeqId seq) {
    if (sequences_.count(seq) != 0) {
        throw std::invalid_argument(seq_name(seq) + " already exists");
    }
    sequences_.emplace(seq, Sequence{});
}

int ReferenceBlockManager::blocks_needed(SeqId seq, int tokens) const {
    if (tokens < 0) {
        throw std::invalid_argument("blocks_needed: tokens must not be negative");
    }
    const Sequence& s = get(seq);
    return blocks_for(static_cast<long long>(s.length) + tokens) - static_cast<int>(s.table.size());
}

void ReferenceBlockManager::append(SeqId seq, int tokens) {
    const int needed = blocks_needed(seq, tokens);  // checks seq and tokens
    Sequence& s = get(seq);
    if (static_cast<int>(s.table.size()) + needed > config_.max_blocks_per_sequence) {
        throw std::length_error("append: " + seq_name(seq) + " would need " +
                                std::to_string(s.table.size() + static_cast<std::size_t>(needed)) +
                                " blocks, the limit is " + std::to_string(config_.max_blocks_per_sequence));
    }
    if (needed > free_blocks()) {
        throw KvCacheExhausted("append: " + seq_name(seq) + " needs " + std::to_string(needed) + " blocks, " +
                               std::to_string(free_blocks()) + " free");
    }
    for (int i = 0; i < needed; ++i) {  // lowest free id first; checked above, so this cannot fail
        const auto block = static_cast<BlockId>(std::find(refs_.begin(), refs_.end(), 0) - refs_.begin());
        refs_[static_cast<std::size_t>(block)] = 1;
        s.table.push_back(block);
    }
    s.length += tokens;
}

void ReferenceBlockManager::share_prefix(SeqId dst, SeqId src, int blocks) {
    if (dst == src) {
        throw std::invalid_argument("share_prefix: a sequence cannot share with itself");
    }
    const Sequence& from = get(src);
    Sequence& to = get(dst);
    if (to.length != 0) {
        throw std::invalid_argument("share_prefix: " + seq_name(dst) + " is not empty");
    }
    const int full = from.length / config_.block_size;
    if (blocks < 0 || blocks > full) {
        throw std::invalid_argument("share_prefix: " + seq_name(src) + " has " + std::to_string(full) +
                                    " full blocks, asked for " + std::to_string(blocks));
    }
    if (blocks > config_.max_blocks_per_sequence) {  // cannot happen while src respects the limit
        throw std::length_error("share_prefix: more blocks than a sequence may hold");
    }
    for (int i = 0; i < blocks; ++i) {
        const BlockId block = from.table[static_cast<std::size_t>(i)];
        ++refs_[static_cast<std::size_t>(block)];
        to.table.push_back(block);
    }
    to.length = blocks * config_.block_size;
}

void ReferenceBlockManager::release(SeqId seq) {
    const Sequence& s = get(seq);
    for (const BlockId block : s.table) {
        --refs_[static_cast<std::size_t>(block)];
    }
    sequences_.erase(seq);
}

bool ReferenceBlockManager::contains(SeqId seq) const { return sequences_.count(seq) != 0; }

std::vector<BlockId> ReferenceBlockManager::table(SeqId seq) const { return get(seq).table; }

int ReferenceBlockManager::length(SeqId seq) const { return get(seq).length; }

int ReferenceBlockManager::ref_count(BlockId block) const {
    if (block < 0 || block >= config_.num_blocks) {
        throw std::invalid_argument("block id " + std::to_string(block) + " is outside the pool");
    }
    return refs_[static_cast<std::size_t>(block)];
}

int ReferenceBlockManager::free_blocks() const {
    return static_cast<int>(std::count(refs_.begin(), refs_.end(), 0));
}

KvBlockStats ReferenceBlockManager::stats() const {
    KvBlockStats out;
    out.total_blocks = config_.num_blocks;
    for (const int r : refs_) {
        out.free_blocks += r == 0;
        out.used_blocks += r >= 1;
        out.shared_blocks += r >= 2;
    }
    out.sequences = static_cast<int>(sequences_.size());
    for (const auto& [id, s] : sequences_) {
        out.cached_tokens += s.length;
    }
    return out;
}

void ReferenceBlockManager::check_invariants() const {
    // Recount every block's references from the tables, then compare (I1–I6 of docs/kv-cache.md).
    std::vector<int> listed(refs_.size(), 0);
    std::vector<bool> written_after(refs_.size(), false);  // listed somewhere as a partial block
    for (const auto& [id, s] : sequences_) {
        const int expected_blocks = blocks_for(s.length);
        if (static_cast<int>(s.table.size()) != expected_blocks) {  // I5
            throw KvInvariantViolation("I5: " + seq_name(id) + " has " + std::to_string(s.length) + " tokens and " +
                                       std::to_string(s.table.size()) + " blocks, expected " +
                                       std::to_string(expected_blocks));
        }
        if (expected_blocks > config_.max_blocks_per_sequence) {
            throw KvInvariantViolation("I5: " + seq_name(id) + " holds more blocks than max_blocks_per_sequence");
        }
        std::vector<BlockId> sorted = s.table;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {  // I4
            throw KvInvariantViolation("I4: " + seq_name(id) + " lists a block twice");
        }
        for (std::size_t i = 0; i < s.table.size(); ++i) {
            const BlockId block = s.table[i];
            if (block < 0 || block >= config_.num_blocks) {  // I2
                throw KvInvariantViolation("I2: " + seq_name(id) + " lists block " + std::to_string(block) +
                                           ", outside the pool");
            }
            ++listed[static_cast<std::size_t>(block)];
            const bool full = static_cast<long long>(i + 1) * config_.block_size <= s.length;
            if (!full) {
                written_after[static_cast<std::size_t>(block)] = true;
            }
        }
    }
    int free = 0;
    for (std::size_t b = 0; b < refs_.size(); ++b) {
        if (refs_[b] != listed[b]) {  // I3
            throw KvInvariantViolation("I3: block " + std::to_string(b) + " has reference count " +
                                       std::to_string(refs_[b]) + " but is listed by " + std::to_string(listed[b]) +
                                       " tables");
        }
        if (refs_[b] > 1 && written_after[b]) {  // I6
            throw KvInvariantViolation("I6: block " + std::to_string(b) + " is shared but not full in every table");
        }
        free += refs_[b] == 0;
    }
    const KvBlockStats s = stats();
    if (s.free_blocks + s.used_blocks != s.total_blocks || free != s.free_blocks) {  // I1, I2
        throw KvInvariantViolation("I1/I2: free + used != total");
    }
}

}  // namespace neuroserve::core
