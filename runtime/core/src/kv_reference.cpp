#include "neuroserve/core/kv_reference.hpp"

#include <algorithm>
#include <string>
#include <utility>

namespace neuroserve::core {

namespace {

std::string seq_name(SeqId seq) { return "sequence " + std::to_string(seq); }

}  // namespace

ReferenceBlockManager::ReferenceBlockManager(KvBlockConfig config, std::optional<KvCacheIdentity> prefix_cache)
    : config_(config) {
    if (config_.num_blocks <= 0 || config_.block_size <= 0 || config_.max_blocks_per_sequence <= 0) {
        throw std::invalid_argument("KV block config: num_blocks, block_size and max_blocks_per_sequence must be positive");
    }
    if (prefix_cache) {
        if (prefix_cache->block_size != config_.block_size) {
            throw std::invalid_argument("KV block config: the prefix cache identity has another block size");
        }
        hasher_.emplace(std::move(*prefix_cache));
    }
    refs_.assign(static_cast<std::size_t>(config_.num_blocks), 0);
    records_.resize(static_cast<std::size_t>(config_.num_blocks));
}

const PrefixHasher& ReferenceBlockManager::hasher(const char* op) const {
    if (!hasher_) {
        throw std::logic_error(std::string(op) + ": prefix caching is off");
    }
    return *hasher_;
}

std::vector<BlockId> ReferenceBlockManager::walk(std::span<const TokenId> prompt) const {
    const PrefixHasher& h = *hasher_;
    const std::size_t bs = static_cast<std::size_t>(config_.block_size);
    const std::size_t reusable = prompt.size() / bs;  // every full block (the last one by copy-on-write)
    std::vector<BlockId> hits;
    BlockHash hash = h.root();
    for (std::size_t i = 0; i < reusable; ++i) {
        const auto tokens = prompt.subspan(i * bs, bs);
        hash = h.next(hash, tokens);
        const BlockId parent = hits.empty() ? -1 : hits.back();
        BlockId found = -1;
        for (std::size_t b = 0; b < records_.size(); ++b) {  // a linear scan: obviously right
            const Record& r = records_[b];
            if (r.indexed && r.hash == hash && r.parent == parent &&
                std::equal(r.tokens.begin(), r.tokens.end(), tokens.begin(), tokens.end())) {
                found = static_cast<BlockId>(b);
            }
        }
        if (found < 0) {
            break;
        }
        hits.push_back(found);
    }
    return hits;
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
    for (int i = 0; i < needed; ++i) {  // checked above, so this cannot fail
        s.table.push_back(take_block());
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
    for (std::size_t i = s.table.size(); i-- > 0;) {  // last block first
        const auto b = static_cast<std::size_t>(s.table[i]);
        if (--refs_[b] == 0 && records_[b].indexed) {
            records_[b].released_at = ++release_clock_;  // CACHED
        }
    }
    sequences_.erase(seq);
}

BlockId ReferenceBlockManager::take_block(BlockId keep) {
    for (std::size_t b = 0; b < refs_.size(); ++b) {  // a FREE block, lowest id first
        if (refs_[b] == 0 && !records_[b].indexed) {
            refs_[b] = 1;
            return static_cast<BlockId>(b);
        }
    }
    std::size_t oldest = refs_.size();  // none: the least recently released CACHED block
    for (std::size_t b = 0; b < refs_.size(); ++b) {
        if (refs_[b] == 0 && records_[b].indexed && static_cast<BlockId>(b) != keep &&
            (oldest == refs_.size() || records_[b].released_at < records_[oldest].released_at)) {
            oldest = b;
        }
    }
    records_[oldest] = Record{};  // evicted: unindexed
    ++evicted_blocks_;
    refs_[oldest] = 1;
    return static_cast<BlockId>(oldest);
}

int ReferenceBlockManager::evict_cached() {
    int n = 0;
    for (std::size_t b = 0; b < refs_.size(); ++b) {
        if (refs_[b] == 0 && records_[b].indexed) {
            records_[b] = Record{};
            ++n;
        }
    }
    evicted_blocks_ += n;
    return n;
}

Admission ReferenceBlockManager::admit(SeqId seq, std::span<const TokenId> prompt) {
    hasher("admit");
    Sequence& s = get(seq);
    if (s.length != 0 || !s.table.empty()) {
        throw std::invalid_argument("admit: " + seq_name(seq) + " is not empty");
    }
    if (prompt.empty()) {
        throw std::invalid_argument("admit: empty prompt");
    }
    std::vector<BlockId> hits = walk(prompt);
    const int total = blocks_for(static_cast<long long>(prompt.size()));
    const std::size_t bs = static_cast<std::size_t>(config_.block_size);
    std::optional<BlockId> copied;  // the last hit, if it holds the last prompt token: copied, not shared
    if (!hits.empty() && hits.size() * bs == prompt.size()) {
        copied = hits.back();
        hits.pop_back();
    }
    const int needed = total - static_cast<int>(hits.size());
    if (total > config_.max_blocks_per_sequence) {
        throw std::length_error("admit: " + seq_name(seq) + " would need " + std::to_string(total) +
                                " blocks, the limit is " + std::to_string(config_.max_blocks_per_sequence));
    }
    // Reused CACHED blocks, and a CACHED copy source, cannot also be evicted to make room.
    int pinned = 0;
    for (const BlockId block : hits) {
        pinned += refs_[static_cast<std::size_t>(block)] == 0;
    }
    pinned += copied && refs_[static_cast<std::size_t>(*copied)] == 0;
    if (needed > free_blocks() - pinned) {
        throw KvCacheExhausted("admit: " + seq_name(seq) + " needs " + std::to_string(needed) + " blocks, " +
                               std::to_string(free_blocks() - pinned) + " free");
    }
    for (const BlockId block : hits) {  // a CACHED one is revived: count 1, OWNED again
        ++refs_[static_cast<std::size_t>(block)];
        s.table.push_back(block);
    }
    for (int i = 0; i < needed; ++i) {
        s.table.push_back(take_block(copied ? *copied : -1));
    }
    s.length = static_cast<int>(prompt.size());
    ++lookups_;
    lookup_blocks_ += static_cast<std::int64_t>(prompt.size() / bs);
    hit_blocks_ += static_cast<std::int64_t>(hits.size() + (copied ? 1 : 0));
    Admission out;
    if (copied) {
        ++copied_blocks_;
        out.copy = BlockCopy{*copied, s.table[hits.size()]};  // the first new block, in its place
        out.cached_tokens = static_cast<int>(prompt.size()) - 1;
    } else {
        out.cached_tokens = static_cast<int>(hits.size() * bs);
    }
    return out;
}

int ReferenceBlockManager::commit(SeqId seq, std::span<const TokenId> tokens) {
    const PrefixHasher& h = hasher("commit");
    const Sequence& s = get(seq);
    if (static_cast<int>(tokens.size()) != s.length) {
        throw std::invalid_argument("commit: " + seq_name(seq) + " has " + std::to_string(s.length) + " tokens, got " +
                                    std::to_string(tokens.size()));
    }
    const std::size_t bs = static_cast<std::size_t>(config_.block_size);
    // Plan first, then change: a bad call changes nothing.
    std::vector<std::pair<std::size_t, BlockHash>> to_index;  // table position, hash
    BlockHash hash = h.root();
    for (std::size_t i = 0; i < tokens.size() / bs; ++i) {
        const auto block_tokens = tokens.subspan(i * bs, bs);
        hash = h.next(hash, block_tokens);
        const Record& r = records_[static_cast<std::size_t>(s.table[i])];
        if (r.indexed) {
            if (r.hash != hash || !std::equal(r.tokens.begin(), r.tokens.end(), block_tokens.begin(), block_tokens.end())) {
                throw std::invalid_argument("commit: block " + std::to_string(i) + " of " + seq_name(seq) +
                                            " is indexed with other tokens");
            }
            continue;
        }
        const bool duplicate = std::any_of(records_.begin(), records_.end(),
                                           [&](const Record& other) { return other.indexed && other.hash == hash; });
        if (duplicate) {
            break;
        }
        to_index.emplace_back(i, hash);
    }
    for (const auto& [i, block_hash] : to_index) {
        Record& r = records_[static_cast<std::size_t>(s.table[i])];
        r.indexed = true;
        r.hash = block_hash;
        r.parent = i == 0 ? -1 : s.table[i - 1];
        r.tokens.assign(tokens.begin() + static_cast<std::ptrdiff_t>(i * bs),
                        tokens.begin() + static_cast<std::ptrdiff_t>((i + 1) * bs));
    }
    return static_cast<int>(to_index.size());
}

int ReferenceBlockManager::match(std::span<const TokenId> prompt) const {
    hasher("match");
    return static_cast<int>(walk(prompt).size());
}

bool ReferenceBlockManager::indexed(BlockId block) const {
    ref_count(block);  // checks the id
    return records_[static_cast<std::size_t>(block)].indexed;
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
    out.indexed_blocks = static_cast<int>(std::count_if(records_.begin(), records_.end(), [](const Record& r) { return r.indexed; }));
    out.prefix_lookups = lookups_;
    out.prefix_lookup_blocks = lookup_blocks_;
    out.prefix_hit_blocks = hit_blocks_;
    out.copied_blocks = copied_blocks_;
    for (std::size_t b = 0; b < refs_.size(); ++b) {
        out.cached_blocks += refs_[b] == 0 && records_[b].indexed;
    }
    out.evicted_blocks = evicted_blocks_;
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
        const Record& r = records_[b];
        if (!r.indexed) {
            continue;
        }
        if (written_after[b]) {  // I7
            throw KvInvariantViolation("I7: block " + std::to_string(b) + " is indexed but not full in every table");
        }
        if (refs_[b] == 0 && r.parent >= 0 && r.parent < config_.num_blocks &&
            refs_[static_cast<std::size_t>(r.parent)] == 0 &&
            records_[static_cast<std::size_t>(r.parent)].released_at < r.released_at) {  // I10
            throw KvInvariantViolation("I10: cached block " + std::to_string(b) + " is newer in the LRU queue than its parent " +
                                       std::to_string(r.parent));
        }
        for (std::size_t o = 0; o < b; ++o) {
            if (records_[o].indexed && records_[o].hash == r.hash) {  // I7
                throw KvInvariantViolation("I7: blocks " + std::to_string(o) + " and " + std::to_string(b) +
                                           " are indexed under one hash");
            }
        }
        const bool parent_ok = r.parent == -1 || (r.parent >= 0 && r.parent < config_.num_blocks &&
                                                  static_cast<std::size_t>(r.parent) != b &&
                                                  records_[static_cast<std::size_t>(r.parent)].indexed);
        const BlockHash parent_hash = r.parent == -1 ? hasher_->root() : parent_ok ? records_[static_cast<std::size_t>(r.parent)].hash : 0;
        if (!parent_ok || static_cast<int>(r.tokens.size()) != config_.block_size ||
            hasher_->next(parent_hash, r.tokens) != r.hash) {  // I8
            throw KvInvariantViolation("I8: block " + std::to_string(b) + "'s hash is not H(its parent's hash, its tokens)");
        }
    }
    const KvBlockStats s = stats();
    if (s.free_blocks + s.used_blocks != s.total_blocks || free != s.free_blocks) {  // I1, I2
        throw KvInvariantViolation("I1/I2: free + used != total");
    }
}

}  // namespace neuroserve::core
