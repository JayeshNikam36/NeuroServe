#include "neuroserve/core/kv_cache_manager.hpp"

#include <algorithm>
#include <map>
#include <string>

namespace neuroserve::core {

namespace {

std::string seq_name(SeqId seq) { return "sequence " + std::to_string(seq); }

}  // namespace

KVCacheManager::KVCacheManager(KvBlockConfig config) : config_(config) {
    if (config_.num_blocks <= 0 || config_.block_size <= 0 || config_.max_blocks_per_sequence <= 0) {
        throw std::invalid_argument("KV block config: num_blocks, block_size and max_blocks_per_sequence must be positive");
    }
    blocks_.resize(static_cast<std::size_t>(config_.num_blocks));
    for (BlockId b = 0; b < config_.num_blocks; ++b) {
        free_.insert(free_.end(), b);
    }
}

const KVCacheManager::Sequence& KVCacheManager::get(SeqId seq) const {
    const auto it = sequences_.find(seq);
    if (it == sequences_.end()) {
        throw std::invalid_argument("unknown " + seq_name(seq));
    }
    return it->second;
}

KVCacheManager::Sequence& KVCacheManager::get(SeqId seq) {
    return const_cast<Sequence&>(static_cast<const KVCacheManager*>(this)->get(seq));
}

int KVCacheManager::blocks_for(long long tokens) const {
    return static_cast<int>((tokens + config_.block_size - 1) / config_.block_size);
}

const KVCacheManager::Block& KVCacheManager::block_at(BlockId block) const {
    if (block < 0 || block >= config_.num_blocks) {
        throw std::invalid_argument("block id " + std::to_string(block) + " is outside the pool");
    }
    return blocks_[static_cast<std::size_t>(block)];
}

void KVCacheManager::create(SeqId seq) {
    if (!sequences_.emplace(seq, Sequence{}).second) {
        throw std::invalid_argument(seq_name(seq) + " already exists");
    }
}

int KVCacheManager::blocks_needed(SeqId seq, int tokens) const {
    if (tokens < 0) {
        throw std::invalid_argument("blocks_needed: tokens must not be negative");
    }
    const Sequence& s = get(seq);
    return blocks_for(static_cast<long long>(s.length) + tokens) - static_cast<int>(s.table.size());
}

void KVCacheManager::append(SeqId seq, int tokens) {
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
    s.table.reserve(s.table.size() + static_cast<std::size_t>(needed));  // may throw: before any change
    for (int i = 0; i < needed; ++i) {
        const BlockId block = *free_.begin();
        free_.erase(free_.begin());
        blocks_[static_cast<std::size_t>(block)] = Block{1, seq};
        s.table.push_back(block);
    }
    s.length += tokens;
    cached_tokens_ += tokens;
}

void KVCacheManager::share_prefix(SeqId dst, SeqId src, int blocks) {
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
    to.table.assign(from.table.begin(), from.table.begin() + blocks);  // may throw: before any change
    for (const BlockId block : to.table) {
        Block& b = blocks_[static_cast<std::size_t>(block)];
        if (++b.refs == 2) {
            ++shared_blocks_;
        }
    }
    to.length = blocks * config_.block_size;
    cached_tokens_ += to.length;
}

void KVCacheManager::release(SeqId seq) {
    const auto it = sequences_.find(seq);
    if (it == sequences_.end()) {
        throw std::invalid_argument("unknown " + seq_name(seq));
    }
    for (const BlockId block : it->second.table) {
        Block& b = blocks_[static_cast<std::size_t>(block)];
        if (--b.refs == 1) {
            --shared_blocks_;
        } else if (b.refs == 0) {
            free_.insert(block);
        }
    }
    cached_tokens_ -= it->second.length;
    sequences_.erase(it);
}

bool KVCacheManager::contains(SeqId seq) const { return sequences_.count(seq) != 0; }

std::vector<BlockId> KVCacheManager::table(SeqId seq) const { return get(seq).table; }

int KVCacheManager::length(SeqId seq) const { return get(seq).length; }

int KVCacheManager::ref_count(BlockId block) const { return block_at(block).refs; }

int KVCacheManager::free_blocks() const { return static_cast<int>(free_.size()); }

SeqId KVCacheManager::allocated_by(BlockId block) const {
    const Block& b = block_at(block);
    if (b.refs == 0) {
        throw std::invalid_argument("block " + std::to_string(block) + " is free");
    }
    return b.allocated_by;
}

KvBlockStats KVCacheManager::stats() const {
    KvBlockStats out;
    out.total_blocks = config_.num_blocks;
    out.free_blocks = free_blocks();
    out.used_blocks = config_.num_blocks - out.free_blocks;
    out.shared_blocks = shared_blocks_;
    out.sequences = static_cast<int>(sequences_.size());
    out.cached_tokens = cached_tokens_;
    return out;
}

void KVCacheManager::check_invariants() const {
    // Everything the running counts claim is recomputed from the tables (docs/kv-cache.md, I1–I6).
    std::vector<int> listed(blocks_.size(), 0);
    std::vector<bool> partial_somewhere(blocks_.size(), false);
    std::int64_t tokens = 0;
    std::map<SeqId, const Sequence*> ordered;  // deterministic order: the first violation is stable
    for (const auto& [id, s] : sequences_) {
        ordered.emplace(id, &s);
    }
    for (const auto& [id, s] : ordered) {
        const int expected_blocks = blocks_for(s->length);
        if (static_cast<int>(s->table.size()) != expected_blocks || expected_blocks > config_.max_blocks_per_sequence) {
            throw KvInvariantViolation("I5: " + seq_name(id) + " has " + std::to_string(s->length) + " tokens and " +
                                       std::to_string(s->table.size()) + " blocks, expected " +
                                       std::to_string(expected_blocks) + " (at most " +
                                       std::to_string(config_.max_blocks_per_sequence) + ")");
        }
        std::vector<BlockId> sorted = s->table;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw KvInvariantViolation("I4: " + seq_name(id) + " lists a block twice");
        }
        for (std::size_t i = 0; i < s->table.size(); ++i) {
            const BlockId block = s->table[i];
            if (block < 0 || block >= config_.num_blocks) {
                throw KvInvariantViolation("I2: " + seq_name(id) + " lists block " + std::to_string(block) +
                                           ", outside the pool");
            }
            ++listed[static_cast<std::size_t>(block)];
            if (static_cast<long long>(i + 1) * config_.block_size > s->length) {
                partial_somewhere[static_cast<std::size_t>(block)] = true;
            }
        }
        tokens += s->length;
    }
    int shared = 0;
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        const int refs = blocks_[b].refs;
        if (refs != listed[b]) {
            throw KvInvariantViolation("I3: block " + std::to_string(b) + " has reference count " +
                                       std::to_string(refs) + " but is listed by " + std::to_string(listed[b]) +
                                       " tables");
        }
        const bool in_free_set = free_.count(static_cast<BlockId>(b)) != 0;
        if (in_free_set != (refs == 0)) {
            throw KvInvariantViolation("I1: block " + std::to_string(b) + " has reference count " +
                                       std::to_string(refs) + (in_free_set ? " but is" : " but is not") +
                                       " in the free set");
        }
        if (refs > 1 && partial_somewhere[b]) {
            throw KvInvariantViolation("I6: block " + std::to_string(b) + " is shared but not full in every table");
        }
        shared += refs >= 2;
    }
    if ((!free_.empty() && (*free_.begin() < 0 || *free_.rbegin() >= config_.num_blocks)) ||
        shared != shared_blocks_ || tokens != cached_tokens_) {
        throw KvInvariantViolation("I2: the free set or the running counts disagree with the tables (shared " + std::to_string(shared_blocks_) +
                                   " vs " + std::to_string(shared) + ", tokens " + std::to_string(cached_tokens_) +
                                   " vs " + std::to_string(tokens) + ")");
    }
}

}  // namespace neuroserve::core
