#pragma once
// Property-based testing of KV block managers (docs/kv-cache.md, Day 29): seeded random programs of
// operations, run on two implementations side by side and compared after every operation. Header
// only, for the unit tests; later days add operations (prefix lookup, eviction) here.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "neuroserve/core/kv_block_manager.hpp"

namespace ns_test::kv {

using neuroserve::SeqId;
using neuroserve::core::BlockId;
using neuroserve::core::KvBlockConfig;
using neuroserve::core::KvBlockManager;

enum class OpKind { kCreate, kAppend, kShare, kRelease, kBlocksNeeded };

struct Op {
    OpKind kind = OpKind::kCreate;
    SeqId seq = 0;    // the sequence acted on (share: the destination)
    SeqId src = 0;    // share: the source
    int amount = 0;   // append / blocks_needed: tokens; share: blocks (may be invalid on purpose)
};

struct Program {
    std::uint64_t seed = 0;
    KvBlockConfig config;
    std::vector<Op> ops;
};

// A small, portable generator (SplitMix64): the same seed gives the same program with every
// compiler and standard library (std:: distributions are not portable across them).
class Rng {
public:
    explicit Rng(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    int below(int n) { return static_cast<int>(next() % static_cast<std::uint64_t>(n)); }  // [0, n)
    int between(int lo, int hi) { return lo + below(hi - lo + 1); }                         // [lo, hi]

private:
    std::uint64_t state_;
};

// A random program: a random small pool, then ops on a few sequence ids. The generator keeps a model
// of which ids are live and how long they are, so that most calls are valid (they exercise the
// interesting paths: block boundaries, sharing, running out of blocks, the length limit) and about
// one in eight is invalid on purpose (an unknown or live id, a negative count, sharing too much).
inline Program generate(std::uint64_t seed, int max_ops = 120) {
    Rng rng(seed);
    Program p;
    p.seed = seed;
    const int block_sizes[] = {1, 2, 4, 16};
    p.config.block_size = block_sizes[rng.below(4)];
    p.config.num_blocks = rng.between(1, 40);
    p.config.max_blocks_per_sequence = rng.between(1, 12);
    const int bs = p.config.block_size;
    const int ids = rng.between(2, 8);
    std::vector<int> length(static_cast<std::size_t>(ids), -1);  // the model: -1 = not live
    const auto pick = [&](bool live) {  // an id that is (or is not) live, if there is one
        std::vector<int> matching;
        for (int id = 0; id < ids; ++id) {
            if ((length[static_cast<std::size_t>(id)] >= 0) == live) {
                matching.push_back(id);
            }
        }
        return matching.empty() ? rng.below(ids) : matching[static_cast<std::size_t>(rng.below(static_cast<int>(matching.size())))];
    };
    const auto pick_where = [&](auto&& wanted) {  // a random id whose modelled length matches, or -1
        std::vector<int> matching;
        for (int id = 0; id < ids; ++id) {
            if (wanted(length[static_cast<std::size_t>(id)])) {
                matching.push_back(id);
            }
        }
        return matching.empty() ? -1 : matching[static_cast<std::size_t>(rng.below(static_cast<int>(matching.size())))];
    };
    const int count = rng.between(1, max_ops);
    for (int i = 0; i < count; ++i) {
        Op op;
        const bool invalid = rng.below(8) == 0;
        const int roll = rng.below(100);
        if (roll < 20) {
            op.kind = OpKind::kCreate;
            op.seq = static_cast<SeqId>(pick(invalid));
        } else if (roll < 60) {
            op.kind = OpKind::kAppend;
            op.seq = static_cast<SeqId>(pick(!invalid));
            op.amount = invalid && rng.below(2) == 0 ? -1 : rng.between(0, 3 * bs + 1);
        } else if (roll < 78) {
            // A valid share needs an empty destination and a source with full blocks: use them if
            // the model has them; with no empty sequence, create one (the share can come later).
            const int empty = pick_where([&](int len) { return len == 0; });
            const int source = pick_where([&](int len) { return len >= bs; });
            if (!invalid && empty < 0) {
                op.kind = OpKind::kCreate;
                op.seq = static_cast<SeqId>(pick(false));
            } else {
                op.kind = OpKind::kShare;
                op.seq = static_cast<SeqId>(empty >= 0 && !invalid ? empty : pick(true));
                op.src = static_cast<SeqId>(source >= 0 ? source : pick(true));
                const int full = std::max(0, length[op.src] / bs);
                op.amount = invalid ? (rng.below(2) == 0 ? -1 : full + 1) : rng.between(std::min(1, full), full);
            }
        } else if (roll < 95) {
            op.kind = OpKind::kRelease;
            op.seq = static_cast<SeqId>(pick(!invalid));
        } else {
            op.kind = OpKind::kBlocksNeeded;
            op.seq = static_cast<SeqId>(pick(!invalid));
            op.amount = rng.between(0, 3 * bs + 1);
        }
        // Keep the model roughly right (it ignores blocks running out, which only makes an op
        // fail; the managers are the ones compared, not the model).
        int& len = length[op.seq];
        switch (op.kind) {
            case OpKind::kCreate: if (len < 0) len = 0; break;
            case OpKind::kAppend:
                if (len >= 0 && op.amount > 0 && (len + op.amount + bs - 1) / bs <= p.config.max_blocks_per_sequence) {
                    len += op.amount;
                }
                break;
            case OpKind::kShare:
                if (len == 0 && op.seq != op.src && op.amount >= 0 && op.amount <= std::max(0, length[op.src] / bs)) {
                    len = op.amount * bs;
                }
                break;
            case OpKind::kRelease: len = -1; break;
            case OpKind::kBlocksNeeded: break;
        }
        p.ops.push_back(op);
    }
    return p;
}

inline std::string describe(const Op& op) {
    std::ostringstream os;
    switch (op.kind) {
        case OpKind::kCreate: os << "create(" << op.seq << ")"; break;
        case OpKind::kAppend: os << "append(" << op.seq << ", " << op.amount << ")"; break;
        case OpKind::kShare: os << "share_prefix(" << op.seq << ", " << op.src << ", " << op.amount << ")"; break;
        case OpKind::kRelease: os << "release(" << op.seq << ")"; break;
        case OpKind::kBlocksNeeded: os << "blocks_needed(" << op.seq << ", " << op.amount << ")"; break;
    }
    return os.str();
}

inline std::string describe(const Program& p) {
    std::ostringstream os;
    os << "seed " << p.seed << ", config {num_blocks " << p.config.num_blocks << ", block_size "
       << p.config.block_size << ", max_blocks_per_sequence " << p.config.max_blocks_per_sequence << "}:";
    for (std::size_t i = 0; i < p.ops.size(); ++i) {
        os << "\n  " << i << ": " << describe(p.ops[i]);
    }
    return os.str();
}

// What a call did: "ok" plus any result, or the kind of exception it threw.
inline std::string apply(KvBlockManager& m, const Op& op) {
    try {
        switch (op.kind) {
            case OpKind::kCreate: m.create(op.seq); return "ok";
            case OpKind::kAppend: m.append(op.seq, op.amount); return "ok";
            case OpKind::kShare: m.share_prefix(op.seq, op.src, op.amount); return "ok";
            case OpKind::kRelease: m.release(op.seq); return "ok";
            case OpKind::kBlocksNeeded: return "ok " + std::to_string(m.blocks_needed(op.seq, op.amount));
        }
    } catch (const neuroserve::KvCacheExhausted&) {
        return "KvCacheExhausted";
    } catch (const std::length_error&) {
        return "length_error";
    } catch (const std::invalid_argument&) {
        return "invalid_argument";
    } catch (const std::exception& e) {
        return std::string("unexpected exception: ") + e.what();
    }
    return "unreachable";
}

// Everything observable about a manager, as text, so that two states compare with ==.
inline std::string snapshot(const KvBlockManager& m, const std::set<SeqId>& ids) {
    std::ostringstream os;
    const auto s = m.stats();
    os << "free " << m.free_blocks() << ", stats {" << s.total_blocks << " " << s.free_blocks << " " << s.used_blocks
       << " " << s.shared_blocks << " " << s.sequences << " " << s.cached_tokens << "}\n";
    for (const SeqId id : ids) {
        os << "seq " << id << ": ";
        if (!m.contains(id)) {
            os << "-\n";
            continue;
        }
        os << m.length(id) << " tokens [";
        for (const BlockId b : m.table(id)) {
            os << " " << b;
        }
        os << " ]\n";
    }
    os << "refs";
    for (BlockId b = 0; b < m.config().num_blocks; ++b) {
        os << " " << m.ref_count(b);
    }
    return os.str();
}

// The same observable state: compared field by field (fast); snapshot() is for the message.
inline bool same_state(const KvBlockManager& a, const KvBlockManager& b, const std::set<SeqId>& ids) {
    const auto sa = a.stats();
    const auto sb = b.stats();
    if (a.free_blocks() != b.free_blocks() || sa.total_blocks != sb.total_blocks || sa.free_blocks != sb.free_blocks ||
        sa.used_blocks != sb.used_blocks || sa.shared_blocks != sb.shared_blocks || sa.sequences != sb.sequences ||
        sa.cached_tokens != sb.cached_tokens) {
        return false;
    }
    for (const SeqId id : ids) {
        if (a.contains(id) != b.contains(id)) {
            return false;
        }
        if (a.contains(id) && (a.length(id) != b.length(id) || a.table(id) != b.table(id))) {
            return false;
        }
    }
    for (BlockId blk = 0; blk < a.config().num_blocks; ++blk) {
        if (a.ref_count(blk) != b.ref_count(blk)) {
            return false;
        }
    }
    return true;
}

using Factory = std::function<std::unique_ptr<KvBlockManager>(KvBlockConfig)>;

struct Divergence {
    std::size_t step = 0;     // index of the op after which the implementations differ
    std::string detail;
};

// Runs the program on both implementations; the first difference in an op's outcome, in the
// observable state, or an invariant violation in either, is a divergence.
// `observe` (optional) sees every op with its outcome, for coverage counts.
using Observer = std::function<void(const Op&, const std::string&)>;

inline std::optional<Divergence> find_divergence(const Factory& make_a, const Factory& make_b, const Program& p,
                                                 const Observer& observe = {}) {
    auto a = make_a(p.config);
    auto b = make_b(p.config);
    std::set<SeqId> ids;
    for (std::size_t i = 0; i < p.ops.size(); ++i) {
        const Op& op = p.ops[i];
        ids.insert(op.seq);
        const std::string ra = apply(*a, op);
        const std::string rb = apply(*b, op);
        if (observe) {
            observe(op, ra);
        }
        if (ra != rb) {
            return Divergence{i, describe(op) + " returned \"" + ra + "\" vs \"" + rb + "\""};
        }
        for (const KvBlockManager* m : {a.get(), b.get()}) {
            try {
                m->check_invariants();
            } catch (const std::exception& e) {
                return Divergence{i, std::string(m == a.get() ? "first" : "second") +
                                         " implementation broke an invariant: " + e.what()};
            }
        }
        if (!same_state(*a, *b, ids)) {
            return Divergence{i, "states differ after " + describe(op) + ":\n" + snapshot(*a, ids) + "\n--- vs ---\n" +
                                     snapshot(*b, ids)};
        }
    }
    return std::nullopt;
}

// Shortens a failing program until no single simplification keeps the implementations apart:
// removing an op, lowering an op's count (to the smallest that still fails), or merging two appends
// to the same sequence into one. What remains is a short reproduction.
inline Program shrink(const Factory& make_a, const Factory& make_b, Program p) {
    const auto fails = [&](Program& candidate) {
        const auto d = find_divergence(make_a, make_b, candidate);
        if (d) {
            candidate.ops.resize(d->step + 1);  // nothing after the first divergence matters
        }
        return d.has_value();
    };
    if (!fails(p)) {
        return p;
    }
    bool progress = true;
    while (progress) {
        progress = false;
        for (std::size_t i = 0; i < p.ops.size() && !progress; ++i) {  // remove an op
            Program c = p;
            c.ops.erase(c.ops.begin() + static_cast<std::ptrdiff_t>(i));
            if (fails(c)) {
                p = std::move(c);
                progress = true;
            }
        }
        for (std::size_t i = 0; i < p.ops.size() && !progress; ++i) {  // lower a count
            for (int smaller = 0; smaller < p.ops[i].amount && !progress; ++smaller) {
                Program c = p;
                c.ops[i].amount = smaller;
                if (fails(c)) {
                    p = std::move(c);
                    progress = true;
                }
            }
        }
        for (std::size_t i = 0; i < p.ops.size() && !progress; ++i) {  // merge two appends
            for (std::size_t j = i + 1; j < p.ops.size() && !progress; ++j) {
                if (p.ops[i].kind != OpKind::kAppend || p.ops[j].kind != OpKind::kAppend || p.ops[i].seq != p.ops[j].seq ||
                    p.ops[i].amount < 0 || p.ops[j].amount < 0) {
                    continue;
                }
                Program c = p;
                c.ops[i].amount += c.ops[j].amount;
                c.ops.erase(c.ops.begin() + static_cast<std::ptrdiff_t>(j));
                if (fails(c)) {
                    p = std::move(c);
                    progress = true;
                }
            }
        }
    }
    return p;
}

}  // namespace ns_test::kv
