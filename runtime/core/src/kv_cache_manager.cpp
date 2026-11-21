#include "neuroserve/core/kv_cache_manager.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <utility>

namespace neuroserve::core {

namespace {

std::string seq_name(SeqId seq) { return "sequence " + std::to_string(seq); }

}  // namespace

KVCacheManager::KVCacheManager(KvBlockConfig config, std::optional<KvCacheIdentity> prefix_cache) : config_(config) {
    if (config_.num_blocks <= 0 || config_.block_size <= 0 || config_.max_blocks_per_sequence <= 0) {
        throw std::invalid_argument("KV block config: num_blocks, block_size and max_blocks_per_sequence must be positive");
    }
    if (prefix_cache) {
        if (prefix_cache->block_size != config_.block_size) {
            throw std::invalid_argument("KV block config: the prefix cache identity has another block size");
        }
        hasher_.emplace(std::move(*prefix_cache));
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
        take_free_block(seq, s);
    }
    s.length += tokens;
    cached_tokens_ += tokens;
}

void KVCacheManager::take_free_block(SeqId seq, Sequence& s, BlockId keep) {  // capacity checked, reserved
    BlockId block = -1;
    if (!free_.empty()) {
        block = *free_.begin();
        free_.erase(free_.begin());
    } else {  // evict the least recently released CACHED block (not `keep`: a copy's source)
        auto victim = lru_.begin();
        if (victim->second == keep) {
            ++victim;
        }
        block = victim->second;
        lru_.erase(victim);
        unindex(block);
        ++evicted_blocks_;
    }
    Block& b = blocks_[static_cast<std::size_t>(block)];
    b.refs = 1;
    b.allocated_by = seq;
    s.table.push_back(block);
}

void KVCacheManager::unindex(BlockId block) {
    Block& b = blocks_[static_cast<std::size_t>(block)];
    index_.erase(b.hash);
    b.indexed = false;
    b.tokens.clear();
}

int KVCacheManager::evict_cached() {
    const int n = static_cast<int>(lru_.size());
    for (const auto& [tick, block] : lru_) {
        unindex(block);
        free_.insert(block);
    }
    lru_.clear();
    evicted_blocks_ += n;
    return n;
}

const PrefixHasher& KVCacheManager::hasher(const char* op) const {
    if (!hasher_) {
        throw std::logic_error(std::string(op) + ": prefix caching is off");
    }
    return *hasher_;
}

std::vector<BlockId> KVCacheManager::walk(std::span<const TokenId> prompt) const {
    const std::size_t bs = static_cast<std::size_t>(config_.block_size);
    const std::size_t reusable = prompt.size() / bs;  // every full block (the last one by copy-on-write)
    std::vector<BlockId> hits;
    BlockHash hash = hasher_->root();
    for (std::size_t i = 0; i < reusable; ++i) {
        const auto tokens = prompt.subspan(i * bs, bs);
        hash = hasher_->next(hash, tokens);
        const auto it = index_.find(hash);
        if (it == index_.end()) {
            break;
        }
        // Verified, not trusted: the same tokens, after the block reused just before (a 64-bit
        // collision costs a miss, never a wrong reuse).
        const Block& b = blocks_[static_cast<std::size_t>(it->second)];
        if (b.parent != (hits.empty() ? -1 : hits.back()) ||
            !std::equal(b.tokens.begin(), b.tokens.end(), tokens.begin(), tokens.end())) {
            break;
        }
        hits.push_back(it->second);
    }
    return hits;
}

Admission KVCacheManager::admit(SeqId seq, std::span<const TokenId> prompt) {
    hasher("admit");
    Sequence& s = get(seq);
    if (s.length != 0 || !s.table.empty()) {
        throw std::invalid_argument("admit: " + seq_name(seq) + " is not empty");
    }
    if (prompt.empty()) {
        throw std::invalid_argument("admit: empty prompt");
    }
    const std::vector<BlockId> hits = walk(prompt);
    const int total = blocks_for(static_cast<long long>(prompt.size()));
    const std::size_t bs = static_cast<std::size_t>(config_.block_size);
    const bool copy = hits.size() * bs == prompt.size();  // the last hit holds the last prompt token
    const int shared = static_cast<int>(hits.size()) - (copy ? 1 : 0);
    const int needed = total - shared;
    // CACHED hits are revived, and a CACHED copy source is kept until it is copied: neither can
    // also be evicted to make room.
    int pinned = 0;
    for (const BlockId block : hits) {
        pinned += blocks_[static_cast<std::size_t>(block)].refs == 0;
    }
    if (total > config_.max_blocks_per_sequence) {
        throw std::length_error("admit: " + seq_name(seq) + " would need " + std::to_string(total) +
                                " blocks, the limit is " + std::to_string(config_.max_blocks_per_sequence));
    }
    if (needed > free_blocks() - pinned) {
        throw KvCacheExhausted("admit: " + seq_name(seq) + " needs " + std::to_string(needed) + " blocks, " +
                               std::to_string(free_blocks() - pinned) + " free");
    }
    s.table.reserve(static_cast<std::size_t>(total));  // may throw: before any change
    for (int i = 0; i < shared; ++i) {
        const BlockId block = hits[static_cast<std::size_t>(i)];
        Block& b = blocks_[static_cast<std::size_t>(block)];
        if (b.refs == 0) {  // a CACHED hit is revived: OWNED again, out of the LRU queue
            lru_.erase({b.released_at, block});
        }
        if (++b.refs == 2) {
            ++shared_blocks_;
        }
        s.table.push_back(block);
    }
    const BlockId source = copy ? hits.back() : -1;
    for (int i = 0; i < needed; ++i) {  // with a copy, the first of these takes the last hit's place
        take_free_block(seq, s, source);
    }
    s.length = static_cast<int>(prompt.size());
    cached_tokens_ += s.length;
    ++lookups_;
    lookup_blocks_ += static_cast<std::int64_t>(prompt.size() / bs);
    hit_blocks_ += static_cast<std::int64_t>(hits.size());
    Admission out;
    if (copy) {
        ++copied_blocks_;
        out.copy = BlockCopy{hits.back(), s.table[hits.size() - 1]};
        out.cached_tokens = static_cast<int>(prompt.size()) - 1;
    } else {
        out.cached_tokens = static_cast<int>(hits.size() * bs);
    }
    reused_tokens_ += out.cached_tokens;
    return out;
}

int KVCacheManager::commit(SeqId seq, std::span<const TokenId> tokens) {
    const PrefixHasher& h = hasher("commit");
    const Sequence& s = get(seq);
    if (static_cast<int>(tokens.size()) != s.length) {
        throw std::invalid_argument("commit: " + seq_name(seq) + " has " + std::to_string(s.length) + " tokens, got " +
                                    std::to_string(tokens.size()));
    }
    const std::size_t bs = static_cast<std::size_t>(config_.block_size);
    // Plan first, then change: a bad call changes nothing.
    std::vector<std::pair<std::size_t, BlockHash>> to_index;  // table position, hash
    BlockHash parent_hash = h.root();
    for (std::size_t i = 0; i < tokens.size() / bs; ++i) {
        const auto block_tokens = tokens.subspan(i * bs, bs);
        const Block& b = blocks_[static_cast<std::size_t>(s.table[i])];
        if (b.indexed) {
            // Every table that lists an indexed block lists its parent just before it, so equal
            // tokens here mean an equal hash: no need to hash it again.
            if (!std::equal(b.tokens.begin(), b.tokens.end(), block_tokens.begin(), block_tokens.end())) {
                throw std::invalid_argument("commit: block " + std::to_string(i) + " of " + seq_name(seq) +
                                            " is indexed with other tokens");
            }
            parent_hash = b.hash;
            continue;
        }
        const BlockHash hash = h.next(parent_hash, block_tokens);
        if (index_.count(hash) != 0) {
            break;  // a duplicate: another block holds this prefix; a later commit retries
        }
        to_index.emplace_back(i, hash);
        parent_hash = hash;
    }
    index_.reserve(index_.size() + to_index.size());  // may throw: before any change
    for (const auto& [i, hash] : to_index) {
        const BlockId block = s.table[i];
        Block& b = blocks_[static_cast<std::size_t>(block)];
        index_.emplace(hash, block);
        b.indexed = true;
        b.hash = hash;
        b.parent = i == 0 ? -1 : s.table[i - 1];
        b.tokens.assign(tokens.begin() + static_cast<std::ptrdiff_t>(i * bs),
                        tokens.begin() + static_cast<std::ptrdiff_t>((i + 1) * bs));
    }
    return static_cast<int>(to_index.size());
}

int KVCacheManager::match(std::span<const TokenId> prompt) const {
    hasher("match");
    return static_cast<int>(walk(prompt).size());
}

bool KVCacheManager::indexed(BlockId block) const { return block_at(block).indexed; }

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
    const std::vector<BlockId>& table = it->second.table;
    for (auto block_it = table.rbegin(); block_it != table.rend(); ++block_it) {  // last block first
        const BlockId block = *block_it;
        Block& b = blocks_[static_cast<std::size_t>(block)];
        if (--b.refs == 1) {
            --shared_blocks_;
        } else if (b.refs == 0) {
            if (b.indexed) {  // CACHED: still indexed, reusable until evicted
                b.released_at = ++release_clock_;
                lru_.insert({b.released_at, block});
            } else {
                free_.insert(block);
            }
        }
    }
    cached_tokens_ -= it->second.length;
    sequences_.erase(it);
}

bool KVCacheManager::contains(SeqId seq) const { return sequences_.count(seq) != 0; }

std::vector<BlockId> KVCacheManager::table(SeqId seq) const { return get(seq).table; }

int KVCacheManager::length(SeqId seq) const { return get(seq).length; }

int KVCacheManager::ref_count(BlockId block) const { return block_at(block).refs; }

int KVCacheManager::free_blocks() const { return static_cast<int>(free_.size() + lru_.size()); }

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
    out.indexed_blocks = static_cast<int>(index_.size());
    out.prefix_lookups = lookups_;
    out.prefix_lookup_blocks = lookup_blocks_;
    out.prefix_hit_blocks = hit_blocks_;
    out.copied_blocks = copied_blocks_;
    out.prefix_reused_tokens = reused_tokens_;
    out.cached_blocks = static_cast<int>(lru_.size());
    out.evicted_blocks = evicted_blocks_;
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
        const bool cached = refs == 0 && blocks_[b].indexed;
        const bool in_lru = lru_.count({blocks_[b].released_at, static_cast<BlockId>(b)}) != 0;
        if (in_free_set != (refs == 0 && !cached) || in_lru != cached) {
            throw KvInvariantViolation("I1: block " + std::to_string(b) + " (reference count " + std::to_string(refs) +
                                       (blocks_[b].indexed ? ", indexed" : "") + ") is " +
                                       (in_free_set ? "" : "not ") + "in the free set and " + (in_lru ? "" : "not ") +
                                       "in the LRU queue");
        }
        if (refs > 1 && partial_somewhere[b]) {
            throw KvInvariantViolation("I6: block " + std::to_string(b) + " is shared but not full in every table");
        }
        shared += refs >= 2;
        const Block& blk = blocks_[b];
        if (!blk.indexed) {
            continue;
        }
        const auto it = index_.find(blk.hash);
        if (it == index_.end() || it->second != static_cast<BlockId>(b)) {
            throw KvInvariantViolation("I7: block " + std::to_string(b) + " is marked indexed but the index " +
                                       (it == index_.end() ? "does not hold its hash" : "maps its hash to block " +
                                                                                            std::to_string(it->second)));
        }
        if (partial_somewhere[b]) {
            throw KvInvariantViolation("I7: block " + std::to_string(b) + " is indexed but not full in every table");
        }
        const bool parent_ok = blk.parent == -1 || (blk.parent >= 0 && blk.parent < config_.num_blocks &&
                                                    static_cast<std::size_t>(blk.parent) != b &&
                                                    blocks_[static_cast<std::size_t>(blk.parent)].indexed);
        const BlockHash parent_hash = blk.parent == -1 ? hasher_->root()
                                      : parent_ok      ? blocks_[static_cast<std::size_t>(blk.parent)].hash
                                                       : 0;
        if (!parent_ok || static_cast<int>(blk.tokens.size()) != config_.block_size ||
            hasher_->next(parent_hash, blk.tokens) != blk.hash) {
            throw KvInvariantViolation("I8: block " + std::to_string(b) + "'s hash is not H(its parent's hash, its tokens)" +
                                       (parent_ok ? "" : ": its parent is not an indexed block"));
        }
        if (cached && blk.parent != -1) {  // I10: eviction takes leaves first, so it never orphans a block
            const Block& parent = blocks_[static_cast<std::size_t>(blk.parent)];
            if (parent.refs == 0 && parent.released_at < blk.released_at) {
                throw KvInvariantViolation("I10: cached block " + std::to_string(b) + " is newer in the LRU queue than its parent " +
                                           std::to_string(blk.parent));
            }
        }
    }
    if (lru_.size() != static_cast<std::size_t>(std::count_if(blocks_.begin(), blocks_.end(), [](const Block& x) {
            return x.refs == 0 && x.indexed;
        }))) {
        throw KvInvariantViolation("I10: the LRU queue does not hold exactly the CACHED blocks");
    }
    for (const auto& [hash, block] : index_) {
        if (block < 0 || block >= config_.num_blocks || !blocks_[static_cast<std::size_t>(block)].indexed ||
            blocks_[static_cast<std::size_t>(block)].hash != hash) {
            throw KvInvariantViolation("I7: the index maps a hash to block " + std::to_string(block) +
                                       ", which is not indexed under it");
        }
    }
    if ((!free_.empty() && (*free_.begin() < 0 || *free_.rbegin() >= config_.num_blocks)) ||
        shared != shared_blocks_ || tokens != cached_tokens_) {
        throw KvInvariantViolation("I2: the free set or the running counts disagree with the tables (shared " + std::to_string(shared_blocks_) +
                                   " vs " + std::to_string(shared) + ", tokens " + std::to_string(cached_tokens_) +
                                   " vs " + std::to_string(tokens) + ")");
    }
}

}  // namespace neuroserve::core
