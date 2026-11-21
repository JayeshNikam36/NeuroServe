# KV cache

This is the design of NeuroServe's KV-cache manager for Phase 3 (Days 26–45). It records how KV
memory works today, the design that replaces it, who owns what, and the invariants that the
per-step checker enforces. Each later Phase 3 day implements one part, against this document.

Requirements: KV-001 to KV-010 in the technical specification.

## Status

| Day | Part | State |
| --- | --- | --- |
| 26 | This design | done |
| 27 | `KvBlockManager` contract and `ReferenceBlockManager`, the CPU oracle | done |
| 28 | `KVCacheManager` (ordered free set, running counts, `allocated_by`), with the backend taking every block from it and the engine's allocator emptied | done |
| 29 | Property-based tests (`tests/unit/test_kv_property.cpp`, harness in `tests/support/kv_ops.hpp`): 10,000 seeded random programs, about 600,000 operations, manager vs reference after every operation; a divergence is shrunk to a short reproduction. `NS_KV_PROPERTY_PROGRAMS` / `NS_KV_PROPERTY_SEED` run more or other programs | done |
| 30 | Per-step checks: the manager's I1–I6, the engine's and the manager's views of every sequence, and the loop's I11–I12, after every engine step (`EngineLoopOptions::check_invariants`); fault injection for each (`tests/unit/test_kv_invariants.cpp`) | done |
| 31 | Reference counting through the backend and the engine: `share_prefix` and `sequence_pages` on every backend, shared pages in `/health`, preemption planned with a request's own pages, I11 counting a shared page once; tests in every release order and through completion, cancellation and preemption (`tests/unit/test_kv_sharing.cpp`) | done |
| 32 | Prefix hashing for full blocks (`kv_prefix_hash.hpp`): XXH64, a pinned byte encoding, the chained block hash and its root, the CUDA backend's K/V identity, `kv_cache_key` in `model_info()`; tests against an independent implementation (`tests/unit/test_kv_prefix_hash.cpp`) | done |
| 33 | Prefix index and lookup at admission: `admit`, `commit`, `match` in the `KvBlockManager` contract (reference and manager), verified hits, hit counters in `/health`, I7 and I8 checked; contract, property and engine tests (`test_kv_prefix_index.cpp`, `test_kv_prefix_cache.cpp`). The CUDA backend turns it on with Day 34's prefill | done |
| 34 | Prefill from a cached offset: `attention_prefill_paged` (causal attention over the cached prefix through the block table) and `model_prefill_from` in the CUDA backend; `prefix_caching` in `CudaLlmEngineConfig` and `neuroserve.Engine` (off by default until Day 36); GPU test: the Hugging Face chat references match with and without a prefix hit | done |
| 35 | Copy-on-write: `admit` reuses the page holding the last prompt token as a private copy (`Admission.copy`), the CUDA backend copies its K/V (`copy_page`) and the prefill computes one token; both implementations, property tests; GPU test: the copy's request and the holder both match Hugging Face | done |
| 36 | The shared-prefix end-to-end test: 16 requests with a 512-token system prompt, run without and with the cache on one backend in identically shaped batches; outputs identical, hit rate and reused tokens reported; `set_prefix_caching` on the CUDA backend | done |
| 37 | `CACHED` blocks and LRU eviction: released indexed blocks stay reusable, a new block is a FREE one or the least recently released CACHED one, released leaves first; `evict_cached`; I1, I2 and I10 checked; both implementations, property tests; GPU: a finished request's pages reused, evicted oldest first under pressure, outputs equal Hugging Face | done |
| 38 | Preemption by recompute: a preempted request keeps its tokens and resumes by prefilling prompt + generated tokens (nothing generated twice), reusing its own `CACHED` pages; chunked prefill in the CUDA backend for inputs longer than the scratch; a resumption counts at most one step's token budget; tests on the CPU and the GPU (Hugging Face, with and without the cache, all four policies) | done |
| 39 | Memory-aware admission: `kv_memory.hpp` reads layers, KV heads, head dim and dtype from `config.json` (32 KiB/token, 512 KiB/block for Llama-3.2-1B FP16); `on-demand` and `worst-case` admission (`EngineLoopOptions::admission`, `neuroserve.Engine(kv_admission=...)`); the CUDA backend refuses a `config.json` that disagrees with its weights | done |
| 40 | KV-cache metrics: `GET /v1/runtime/state` (occupancy, cached and free pages, hits, misses, hit rate, reused tokens, evictions, preemptions, admission mode); an exact `prefix_reused_tokens` counter in the contract; `engine.prefix_caching` and `engine.kv_admission` in the server config | done |
| 41 | Compute Sanitizer: `memcheck` and `racecheck` on the unit (KV, scheduler, engine loop), CUDA and GPU integration suites; CMake target `compute_sanitizer`, `scripts/compute_sanitizer.sh`, summary in `tests/logs/compute-sanitizer.md` | done |
| 42 | Block-size sweep: `ns_block_size_sweep` (benchmarks/block_size) computes capacity and waste at 8, 16 and 32 tokens for a seeded length mix and times the paged decode attention kernel at each, interleaved in one process; results in `benchmarks/results/day42-block-size.md` | done |

## Today (v0.1)

### Physical memory

K and V live in **pools of fixed-size blocks**, one K pool and one V pool per layer, each shaped
`[num_blocks, block_size, n_kv_heads, head_dim]` in FP16. A block number names the same slot in
all 32 pools (16 layers × K and V), so one block holds the K/V of 16 consecutive positions of one
sequence in every layer.

| Llama-3.2-1B, FP16 | Value |
| --- | --- |
| K + V per token, all layers | 2 × 16 layers × 8 KV heads × 64 dims × 2 bytes = **32 KiB** |
| Block (16 tokens) | **512 KiB** |
| Pool in `configs/dev.yaml` (2048 blocks) | **1 GiB**: 32,768 tokens of KV |
| Longest sequence (`max_pages_per_sequence` 64) | 1,024 tokens |

The code says "page" for a block (`num_pages`, `free_pages`). This document says **block**; the
two words mean the same thing.

### Who owns it today

- **cuda-llm-engine owns the memory and the allocator.** A `PageAllocator` is a LIFO free list;
  a `Sequence` holds the vector of its block numbers (`pages`), its length, and a device copy of
  its block table. `reserve()` takes blocks from the free list as a sequence grows, and
  `free_sequence()` returns them. The engine `exit(1)`s if the free list runs dry, so it must never
  be asked to.
- **`CudaLlmEngineBackend`** checks, before every engine call, that enough blocks are free, and
  throws `KvCacheExhausted` instead.
- **`EngineLoop`** keeps its own count of each request's blocks (`ceil(cached / 16)`), and
  passes it to the scheduling policy, which decides admissions and preemptions (ADR-0004).

A block belongs to exactly one sequence; nothing is shared, so nothing is counted.

### How a block table is used

```
 sequence R1: 40 tokens cached           K pool, layer l: [num_blocks, 16, 8 heads, 64]
 block table (logical → physical)        ┌─────────┐
   logical 0 ──────────────────────────► │ block 7 │  positions  0–15 of R1
   logical 1 ──────────────────────────► │ block 3 │  positions 16–31 of R1
   logical 2 ──────────────────────────► │ block 12│  positions 32–39 of R1 (8 of 16 slots used)
                                         └─────────┘
 position p lives in block table[p / 16], slot p % 16, in every layer's K and V pool
```

- **Prefill** writes the prompt's K/V through the table (`kv_scatter`). Its attention runs over
  the prompt's own tokens (`attention_naive`), not through the pools.
- **Batched decode** builds one table per sequence into a `[max_batch, max_pages]` device array
  every step. It writes each new token's K/V at position `len` (`kv_scatter_batch`), and the
  split-KV attention kernel reads all positions through the tables.

Two consequences shape the design:
1. **A block is written only at its next free slot.** Positions before `len` are never written
   again. So **a full block is immutable**, which is what makes sharing it safe.
2. **Prefill cannot yet start from a cached offset.** Its attention sees only the new tokens. Prefix
   reuse needs a prefill whose attention also reads the cached positions through the table (see
   "Prefill from a cached offset").

## The design

One **`KVCacheManager`** in `runtime/core` becomes the single owner of KV memory **accounting**.
The GPU pools stay where they are, allocated once by the backend at startup.

| Structure | Holds | Purpose |
| --- | --- | --- |
| Block records | per block: state, reference count, hash, last-release tick | the truth about every block |
| Free list | blocks in state `FREE` | allocation without eviction |
| Block tables | per sequence: physical block ids, in logical order, plus its token count | what the kernels read |
| Prefix index | hash → block id, for full blocks | finding reusable blocks |
| LRU queue | blocks in state `CACHED`, least recently released first | eviction order |
| Statistics | occupancy, hits, misses, reused tokens, evictions, copy-on-write copies, preemptions | metrics (Day 40) |

### Block states

```
            allocate                 last reference released, block not indexed
   FREE ─────────────────► OWNED ───────────────────────────────────────────────► FREE
    ▲                       │  ▲
    │ evict (LRU)           │  │ prefix hit (ref 0 → 1)
    │                       ▼  │
    └────────────────── CACHED ◄───── last reference released, block indexed (full)
```

- **`FREE`:** in the free list, not indexed, reference count 0.
- **`OWNED`:** referenced by at least one block table (count ≥ 1). It may be indexed, if full.
- **`CACHED`:** count 0 but indexed. Its K/V are still valid, so a later request with the same
  prefix can revive it. It sits in the LRU queue and is evicted (unindexed, freed) only when an
  allocation finds the free list empty.

**Capacity for scheduling is `FREE` + `CACHED`.** A cached block is free memory that happens to
hold something useful.

### Ownership and threads

- **The engine thread is the only thread that touches the manager.** Every operation happens
  between GPU calls, in the same step that decides it, so reference counts are plain integers.
  Atomics would only hide a second writer, and a second writer would be a bug (KV-006).
- **Other threads see copies.** `/health`, metrics and the decision log read a snapshot taken
  under the engine loop's mutex, as `stats()` does today.
- **The backend stops allocating.**
  - Before each prefill or decode it receives the manager's tables, and fills each engine
    `Sequence.pages` with them, so the engine's `reserve()` finds enough blocks and allocates
    nothing.
  - On release it clears `pages` before `free_sequence()`, so the engine's allocator is never
    used.
  - The submodule is not modified: the engine API already lets the caller supply the block list.
  - The engine's allocator is left with no blocks, so a missed path fails loudly instead of
    silently double-allocating.
- **Blocks carry their owner.** Each block table records its sequence id, and in debug builds each
  block records the sequence that last wrote it. A leak or a write through a stale table then
  names the request responsible (KV-003).

### Operations

The CPU reference allocator (Day 27) and the manager (Day 28) implement one interface,
`KvBlockManager` (`runtime/core/include/neuroserve/core/kv_block_manager.hpp`). The reference is the
test oracle: the same calls must leave both in the same state (Day 29). So the contract fixes
everything an implementation could otherwise choose freely:

- **New blocks are taken lowest id first.** The engine's free list is LIFO; the manager uses an
  ordered set, so the same operations give the same tables in every implementation.
- **A table has exactly `ceil(length / block_size)` entries:** blocks are taken at boundaries, never
  ahead.
- **Every operation succeeds or throws and changes nothing:**
  - `KvCacheExhausted` when blocks are short;
  - `std::length_error` past `max_blocks_per_sequence`;
  - `std::invalid_argument` for bad calls.

```cpp
class KvBlockManager {  // all calls on the engine thread
public:
    void create(SeqId seq);                               // an empty sequence
    int blocks_needed(SeqId seq, int tokens) const;       // new blocks `tokens` more would take
    void append(SeqId seq, int tokens);                   // grow, taking blocks at boundaries
    void share_prefix(SeqId dst, SeqId src, int blocks);  // empty dst starts with src's first
                                                          // `blocks` blocks, which must be full
    void release(SeqId seq);                              // drop references; count 0 -> free
    std::vector<BlockId> table(SeqId seq) const;          // physical ids, in logical order
    int length(SeqId seq) const;
    int ref_count(BlockId block) const;
    int free_blocks() const;
    KvBlockStats stats() const;                           // free, used, shared, sequences, tokens
    void check_invariants() const;                        // KvInvariantViolation names the broken one
};
```

`share_prefix` is the primitive under prefix caching: on Days 32–33 an `admit(seq, prompt)` call
looks the prompt's full blocks up in the prefix index, shares the hits, and appends the rest. The
`CACHED` state, the LRU queue and `commit` (indexing written blocks) are added with them (Days 33
and 37), to the reference first and then to the manager.

## Reference counting (Day 31)

- **A block's count is the number of block tables that list it.** `admit` increments it for every
  prefix hit; `release` decrements it for every block in the table.
- **A block is freed only when the last owner releases it** (count 1 → 0). Then it becomes
  `CACHED` if it is indexed, or `FREE` otherwise.
- **Preemption is a release.** A preempted request drops its references; blocks other requests
  still use survive. **The memory a preemption reclaims is only the blocks whose count drops to
  0**, so the policy's `pages_held` becomes "blocks this request alone holds". A policy that
  counted shared blocks would expect memory that preemption cannot give back.

**Implemented (Day 31).**
- **Backends:** every backend has `share_prefix(dst, src, pages)` and
  `sequence_pages(seq) -> {held, exclusive}`, and `/health` reports `shared_pages`.
- **The engine loop:** each step it reads every running request's exclusive pages before the policy
  decides, and passes them as `RunningRequest::pages`.
- **I11:** it now checks `used = Σ exclusive + shared`, so each shared page is counted once.
- **The test that proves the policy side:** 16 requests share a two-page prefix on a 10-page pool,
  and the engine must preempt among them. With the old count (all pages held), the policy admits
  requests whose prefill then runs out of pages, and they fail. With the exclusive count none does.

**Why the counts are plain integers, not atomics (KV-006).**
- **Who touches them:** reference counts are only read and changed on the engine thread, between
  GPU calls. That thread is the only one that calls the backend, which owns the manager.
- **What other threads see:** `/health`, metrics and the decision log read copies, made under the
  engine loop's mutex at the end of a step.
- **Why atomics would be wrong here:** there is no second writer, so an atomic would only hide one
  if a bug ever added it. ThreadSanitizer runs every CPU test in CI, and would report that bug as a
  data race instead.
- **When this would change:** only if several threads ever drove one backend (for example a
  separate prefill thread). That would need a design of its own, not just atomic counters, because
  a block's count and its place in the free set must change together.

## Prefix caching

### Hashing full blocks (Day 32)

- **Only full blocks are hashed and shared.** A partial block is still being written.
- **Every block's hash is chained to the one before it:**
  `h_i = H(h_{i-1}, tokens of block i)`, with
  `h_{-1} = H(model id, weights digest, dtype, block size, RoPE theta and scaling)`.
- **Why chained:** a block's K/V depend on every token before it, not only its own 16. The chain
  makes equal hashes mean equal whole prefixes.
- **Why the root carries the configuration:** another model, dtype, block size or RoPE setting
  changes K/V values for the same tokens, so it must change every hash (KV-007).
- **H is a 64-bit hash for lookup.** A hit is then **verified**: the block's stored tokens must
  equal the request's, and its parent must be the block before it in the request's table.
  Collisions can cost a lookup, never a wrong output.

**Implemented (Day 32)** in `runtime/core/include/neuroserve/core/kv_prefix_hash.hpp`.

- **`PrefixHasher(identity)`** gives `root()`, `next(parent, block)`, `chain(tokens)` (one hash per
  full block) and `extend(chain, tokens)`, which hashes each block once, when a growing sequence
  fills it.
- **H is XXH64,** implemented in the file (no dependency, so a library upgrade cannot change it).
- **The hashes are a stable format.** They are the same in every process, build and platform: no
  `std::hash`, no pointers, explicit little-endian bytes. Golden values in the tests pin the format,
  so changing it is a visible decision.

**The encoding.** Integers are fixed-width little-endian, a double is its IEEE-754 bits, and a
string is a `u32` length followed by its bytes. The length prefixes keep fields apart:
`("ab", "c")` and `("a", "bc")` encode differently.

| Hash | XXH64 (seed 0) of |
| --- | --- |
| root | `"neuroserve.kv.root.v1"`, model id, weights digest, dtype, `u32` block size, `f64` RoPE theta, RoPE scaling type, `f64` factor, `f64` low-frequency factor, `f64` high-frequency factor, `u32` original context |
| block | `"neuroserve.kv.block.v1"`, `u64` parent hash, `u32` token count, each token as `i32` |

**The CUDA backend's identity.**

| Field | Value |
| --- | --- |
| model id | the configured model name |
| weights digest | `file_fingerprint(weights.bin) + ":" + file_fingerprint(index.tsv)` |
| dtype | `fp16` |
| block size | the page size, 16 |
| RoPE | Llama 3: theta 500,000, scaling `llama3` (factor 32, low 1, high 4, original context 8,192), the values cuda-llm-engine hard-codes in `load_model` |

- **`model_info()["kv_cache_key"]` is the root, as 16 hex digits.** Two engines with equal keys
  compute equal K/V for equal tokens, so cached blocks may only ever be reused under an equal key.
- **The weights digest is sampled.** `file_fingerprint` hashes a file's size plus either all of it
  (up to 16 MiB) or 64 evenly spaced 64 KiB chunks from its first byte to its last, about 4 MiB.
  - **Why sampled:** hashing all 2.5 GB at every start would read the weights a second time.
  - **What it catches:** another export, model or fine-tune changes nearly every tensor, so it
    changes the samples. The base and instruct Llama-3.2-1B have the same shapes and file size, and
    still get different keys.
  - **What it does not catch:** one byte changed between two samples. A checkpoint is never edited
    in place like that; the test documents the limit.
- **I8** ("an indexed block's hash equals `H(parent, its tokens)`") is checked from Day 33, once
  blocks are indexed.


1. Walk the prompt's full blocks in order, hash by hash. Stop at the first miss: a block can only
   be reused if everything before it is too.
2. **A hit on an `OWNED` block** shares it (count + 1). **A hit on a `CACHED` block** revives it:
   it leaves the LRU queue, count 0 → 1.
3. Allocate blocks for the rest of the prompt.
4. The prefill then starts at the first uncached token.
5. Hits, misses and reused tokens are counted.

**Example.** 16 requests share a 512-token system prompt (32 full blocks). Without the cache they
hold 16 × 32 = 512 blocks for it. With it they hold 32, which frees 480 blocks (**240 MiB**) and
skips 7,680 tokens of prefill. That is the shared-prefix test of Day 36.

**Implemented (Day 36)** as `CudaPrefixCacheTest.SixteenRequestsWithA512TokenSystemPromptMatchTheNoCacheRun`.

- **Why on one backend, batch for batch.** Outputs are deterministic but not batch-invariant
  (ADR-0006): two runs whose batches differ in shape can differ without any bug. So one backend runs
  the 16 requests twice, switching the cache with `set_prefix_caching` (allowed only with no live
  sequence), and both runs have the same steps: 16 prefills in order, then 32 decode steps of one
  batch of 16. The cache is the only difference.
- **What it checks:** every request's 32 tokens equal the no-cache run's; 15 of 16 requests reuse
  the 32 system pages (hit rate 480 / 512 full pages, 93.75%); 7,680 of 8,320 prompt tokens are
  reused, so 640 are prefilled instead of 8,320.
- **The server keeps prefix caching off** for now. Without the `CACHED` state, only requests that
  overlap in time share pages; Day 37 keeps finished requests' pages, and that is when serving
  benefits from turning it on.

**Implemented (Day 33)** in the `KvBlockManager` contract, so the reference and the manager are
compared on it operation by operation (the property tests now admit, commit and match too).

- **`admit(seq, prompt)`** walks, shares the hits and appends the rest, all or nothing. It returns
  the reused tokens: the prefill starts there.
- **`commit(seq, tokens)`** indexes a sequence's full blocks once their K/V are written: after the
  prefill, and after each decode step that fills a block. Generated tokens are indexed too.
- **`match(prompt)`** counts the blocks `admit` would reuse, and changes nothing.
- **A hit is verified, not trusted.** The indexed block's stored tokens must equal the prompt's,
  and its parent must be the block the walk reused just before it. A 64-bit collision then costs a
  miss, never a wrong reuse. A test corrupts an indexed block's tokens: I8 reports it and the
  lookup refuses it.
- **The block holding the last prompt token was not reused on Day 33,** because its logits must be
  computed and that writes its K/V. Since Day 35 it is reused as a private copy (below).
- **A duplicate is not indexed.** Two blocks with the same prefix happen when the last-token rule
  made a request compute a block that another already holds. The index keeps the first; the
  sequence stops indexing there, and a later `commit` retries, so its copy takes over once the
  original is freed.
- **Counted:** lookups, the blocks they could have reused (`prefix_lookup_pages`) and the hits
  (`prefix_hit_pages`), in `KvCacheStats` and `/health`. The hit rate is hits ÷ lookup pages.
- **Day 33 freed an indexed block with its last holder,** so a hit needed a live request with the
  same prefix. Since Day 37 it stays `CACHED` (below).
- **Where it runs today:** the engine tests run a prefix-caching backend (`FakeBackend` on the
  real manager) through the engine loop, with every step's invariant checks on. The CUDA backend
  keeps prefix caching off, and its counters read 0, until its prefill can start at a cached offset
  (Day 34): today's prefill attends only to the tokens it is given.

### Prefill from a cached offset (Day 34)

The suffix must attend to the cached positions, which today's prefill cannot do. There are two
ways:

| Option | How | Cost |
| --- | --- | --- |
| A. Decode the suffix token by token | Feed each uncached token through the existing paged decode path | No new kernel, but one step per suffix token: slow for long suffixes, and the suffix's K/V come from a different code path than a full prefill's |
| B. Paged causal prefill attention | A prefill that writes the suffix's K/V through the table and attends to positions `0 … p` for each query `p` | One new, simple attention kernel. It is the first step toward Phase 4's variable-length prefill (Days 47–49) |

**Decision: B.** The kernel starts simple and correct; Phase 4 makes it fast. Positions and RoPE
start at the cached length, so query `p` uses position id `p`, not `p - cached`.

**A risk, from ADR-0006.** A cached block's K/V were computed inside another request's prefill,
whose GEMMs had a different number of rows. The values can therefore differ in the last bits from
those of a fresh prefill of this prompt, and at a near-tie that can change a greedy choice. The
Day 34 acceptance test, matching Hugging Face exactly, is checked on the reference workloads; on
other inputs any difference from the no-cache run is measured and reported, as the release gate
already does.

**Implemented (Day 34)** in `runtime/backend/src/prefill_from_offset.cu`, without changing the
engine submodule.

- **`model_prefill_from(model, cache, seq, ids, start)`** runs the engine's per-layer math for
  `ids[start ..]` only. Position ids, and so RoPE, start at `start`. The new tokens' K/V are
  scattered into the sequence's pages first, then attention reads the whole prefix through the
  block table.
- **`attention_prefill_paged`** gives each new token `t` causal attention over positions
  `0 … start + t`. Its per-query math is `attention_decode_paged`'s step for step, so a suffix token
  gets the attention the decode path would give it. One block per (head, token): simple and correct
  now, made fast in Phase 4.
- **The prefill picks its path by what was reused.** `start == 0` (no hit) uses the engine's own
  `model_prefill_paged`, unchanged, so outputs without a hit cannot change. A hit uses
  `model_prefill_from`.
- **The CUDA backend admits and commits** like the fake: `admit` before the prefill, `commit` after
  it and after every decode step. It keeps each sequence's tokens for that.
- **Off by default.** `prefix_caching` in `CudaLlmEngineConfig` and `neuroserve.Engine(...,
  prefix_caching=True)`. The server keeps it off until the shared-prefix end-to-end test
  (Day 36) shows outputs equal to the no-cache run under load.
- **The acceptance test** (`CudaPrefixCacheTest`) uses the Hugging Face chat references for
  Llama-3.2-1B-Instruct. Their prompts all start with the same system header. Each case runs at
  batch size 1 twice: with nothing to reuse, and while a holder keeps the header's pages live, so
  its prefill starts at the cached offset. Both runs must equal the Hugging Face greedy tokens. A
  second test reuses every page but the last of an identical prompt.

### Copy-on-write (Day 35)

**The rule: a write goes only into a block with reference count 1 that is not indexed.** Because
writes happen only at a sequence's next free slot, the only time this can fail in v0.1.5 is a
**full prefix hit**:
1. Every prompt token is cached, including the last, whose logits are still needed.
2. Computing them writes that token's K/V again, into a shared, indexed block.
3. The new values may differ in the last bits (another code path), which would corrupt the block
   for every other user.

The manager therefore **copies that block first**: it allocates a new block, copies the 512 KiB
on the GPU, points this request's table at the copy, drops one reference on the original, and
recomputes the last token into the copy. The alternative (never reusing the block that holds the
last prompt token) costs up to 15 tokens of prefill per request and needs no copy. Copy-on-write
also stays the guard for every later feature that writes after sharing (parallel sampling,
forked sequences).

**Implemented (Day 35).** A write can reach a shared block in one place only: decode always writes
into a sequence's own partial block or a new one, so the only shared block written is the one
holding the last prompt token, whose logits the prefill must compute.

- **`admit` now reuses every full block,** the last one too. When the hit covers the last prompt
  token, that block is not shared: the sequence gets a free block in its place, and `admit` returns
  it as `Admission.copy = {from, to}` with `cached_tokens = prompt.size() - 1`.
- **The CUDA backend copies the block** (`copy_page`: K and V of every layer, device to device)
  before the prefill, which then computes the one remaining token into the copy. The original
  keeps its single owner and is never written.
- **The copy needs a free block.** If none is free, `admit` throws `KvCacheExhausted` and changes
  nothing. Hit counters count the copied block as a hit; `copied_blocks` counts copies.
- **The copy is not indexed:** it duplicates an indexed block. It takes over if the original goes
  (the duplicate rule of Day 33).
- **Tested on the GPU** with the Hugging Face chat reference whose prompt fills whole pages
  (48 tokens): the request that prefilled one token into the copy and the holder it was copied from
  both match Hugging Face.

### Eviction (Day 37)

- **`CACHED` blocks are evicted least recently released first,** and only when the free list
  cannot satisfy an allocation: before the request is deferred, and before anything is preempted.
- **Evicting a block** removes it from the index and the LRU queue, and returns it to `FREE`.
- **Only count-0 blocks can be evicted.** An `OWNED` block is never touched (KV-008).
- LRU is the baseline. A cost-aware policy (reuse frequency × recompute cost) is a later
  experiment, measured against it.

**Implemented (Day 37)** in the `KvBlockManager` contract, reference and manager, compared by the
property tests (10,000 programs, invariants checked after every operation).

- **Release:** a block whose count reaches 0 becomes `CACHED` if it is indexed (still in the
  index, reusable by `admit`), `FREE` otherwise. A table is released last block first.
- **Allocation:** a `FREE` block, lowest id first; if none is free, the least recently released
  `CACHED` block, evicted (unindexed) on the spot. A request is deferred only when neither is left.
- **Leaves first.** Every table that lists a block lists its parent before it, and a table is
  released from the end, so a cached block is never older in the LRU queue than a cached block
  after it in a chain. Eviction then takes leaves first and never orphans a cached block. I10
  checks this ordering.
- **A hit on a `CACHED` block revives it** (`OWNED`, out of the LRU queue). Revived hits, and a
  `CACHED` copy-on-write source, are pinned during the admission: they cannot be evicted to make
  room for the same request.
- **`free_blocks` counts `FREE + CACHED`:** a cached block is reclaimable on demand, so admission
  and the scheduler see it as free. `/health` shows `cached_pages` and `evicted_pages`.
- **`evict_cached`** forgets every `CACHED` block (the CUDA backend's `evict_cached_pages`, also
  run by `set_prefix_caching`).
- **Tested on the GPU:** a finished chat request's pages are reused by the next one; 27 prompts of
  640 tokens then overfill the 1,024-page pool, the oldest pages go first, and the chat references
  still equal Hugging Face before and after.
- **The server keeps prefix caching off** until preemption by recompute (Day 38), which changes how
  preempted requests come back and is the last change to how pages are released.

## Preemption (Day 38)

Until Day 38 a preempted request restarted from its prompt and regenerated its tokens. ADR-0006
showed this can diverge from what it already streamed: decoding is deterministic, not
batch-invariant, and the restart runs in another batch.

**With the manager, a preemption becomes cheap and safe:**
- **The release is cheap.** The request's full blocks stay indexed as `CACHED`. Its prompt and
  generated tokens are kept.
- **Resumption reuses what survived.** It looks up **prompt + generated tokens** in the prefix
  index. Every block not evicted in between is reused, and the rest is prefilled from the first
  missing token.
- **Streaming stays consistent.** The tokens the client already has are **input** to the
  recompute, never regenerated, so a stream cannot diverge from itself.
- **The stop rules count every token,** old and new: `max_tokens` and stop tokens apply to the
  whole generation.

So the cache that serves shared prefixes also makes preemption cost close to nothing when memory
is only briefly short.

**Implemented (Day 38)** in the engine loop and the CUDA backend.

- **Preemption keeps the tokens.** `preempt` frees the sequence (its indexed pages become
  `CACHED`), keeps `tokens`, and requeues the request at the front in arrival order. The victim is
  still the policy's choice (lowest priority, latest arrival). `preemptions` counts them.
- **Resumption prefills prompt + tokens,** and the prefill's output is the next new token. Nothing
  is generated twice: the test counts every token the backend produces, and it equals the tokens
  requested. With prefix caching, the request's own pages that survived are reused.
- **How much survives depends on how long memory stays short.** On a pool that stays short, the
  others evict the victim's pages before it returns and it recomputes everything; when the
  shortage eases first, it reuses what is left. Both are correct; the tests cover both.
- **Long resumptions are prefilled in chunks.** Prompt + generated tokens can exceed the prefill
  scratch (`max_prompt_tokens`), so the CUDA backend's `prefill` takes up to a sequence's capacity
  and computes chunks of at most `max_prompt_tokens`, each attending to everything before it
  (`model_prefill_from` from the chunk's start). A GPU test runs every chat reference in 16-token
  chunks and matches Hugging Face.
- **A resumption counts at most one step's token budget,** so a request whose recompute exceeds
  `max_batch_tokens` is still admitted (alone, at worst) instead of waiting forever; that step's
  tokens then exceed the budget by the recompute. Chunked prefill across steps (Days 52–53) removes
  the overshoot.
- **Tested on the GPU** by the existing preemption test (24 concurrent requests on a tiny cache,
  all four policies, every output equal to Hugging Face), now resuming from tokens, and by the same
  test with prefix caching on (`tests/python/test_preemption_cached.py`).

## Memory-aware admission (Day 39)

Bytes per token come from the model's `config.json`:

    2 (K and V) × num_hidden_layers × num_key_value_heads × head_dim × sizeof(dtype)

Two modes, chosen in the config:
- **`on-demand`** (the default): blocks are taken as requests grow, and preemption handles shortage.
- **`worst-case`:** each request reserves `prompt + max_tokens` at admission, so it never preempts,
  but fewer requests run at once.

The Day 39 test pins Llama-3.2-1B FP16 at 32 KiB per token and 512 KiB per block.

**Implemented (Day 39).**

- **`kv_memory.hpp`** parses a Hugging Face `config.json` (top-level keys only; nested objects
  such as `rope_scaling` are skipped): `num_hidden_layers`, `num_key_value_heads` (default:
  `num_attention_heads`), `head_dim` (default: `hidden_size / num_attention_heads`), `torch_dtype`.
  A missing or non-integer field is an error with its name. K/V bytes use the engine's K/V dtype,
  FP16, whatever the weights' dtype: Llama-3.2-1B ships as `bfloat16`, 2 bytes either way.
- **The CUDA backend checks the model directory's `config.json`** when the export kept one: its
  layers, KV heads and head dim must be the loaded weights', or the engine refuses to start.
- **`EngineLoopOptions::admission`** (`neuroserve.Engine(..., kv_admission="worst-case")`) picks
  the mode. Worst case gives each queued request the pages of `prompt + max_new_tokens` and
  subtracts every running request's remaining growth from the free pages the policy sees, so a
  running request can always grow: the policies need no change, and none ever preempts.
- **Tested through the engine:** four requests of 5 pages each on a 10-page pool. On demand, more
  than two start together and they preempt as they grow; worst case runs exactly two at a time and
  never preempts, with and without prefix caching. All outputs are correct in both.
- **The server's config** still uses on-demand; choosing the mode there comes with the KV metrics
  (Day 40), where its effect on preemptions is visible.

## Block size (Day 42)

`ns_block_size_sweep` measures what the block size costs and buys at 8, 16 and 32 tokens, in one
process and on one workload: capacity and padding waste, exactly, for 10,000 sequences of a seeded
chat-like length mix, and decode attention time with the engine's paged kernel
(`attention_decode_paged`, which takes the block size as a parameter) for a batch of 8 whose block
tables point at shuffled pages, the block sizes interleaved in rotating order every round. The
results and their reading are in `benchmarks/results/day42-block-size.md`.

16 stays the default: the engine's fast batched decode kernel is written for 16-token pages, and
the sweep shows what 8 or 32 would change against it (waste, capacity, attention time and how much
of a prefix can be shared, which is whole blocks only).

## Compute Sanitizer (Day 41)

`cmake --build build/dev-cuda --target compute_sanitizer` runs the GPU test binaries under
`memcheck` (out-of-bounds and misaligned device accesses, invalid frees, CUDA API errors) and
`racecheck` (shared-memory races). The unit tests make no CUDA calls, so there is nothing for
Compute Sanitizer to check there; they run under ASan, UBSan and TSan. On WSL the GPU debugger
interface must be enabled once (`EnableDebuggerInterface.bat` as administrator on Windows, then
`wsl --shutdown`), or nothing is instrumented. The GPU integration suite runs whole under `memcheck`;
`racecheck`, which instruments every shared-memory access, runs its smallest test for each kernel
path: the engine's prefill and decode (single and batched), prefill from a cached offset
(`attention_prefill_paged`), copy-on-write (`copy_page`) and chunked prefill. A run passes only with
its tests passing, zero sanitizer errors and at least one test run. The summary,
`tests/logs/compute-sanitizer.md`, is committed; the full logs are not.

## Metrics (Day 40)

`GET /v1/runtime/state` reports the KV cache:

| Field | Meaning |
| --- | --- |
| `kv_cache.used_pages`, `occupancy` | pages held by live requests, and their share of the pool |
| `kv_cache.cached_pages` | released pages still indexed: reusable, reclaimed on demand |
| `kv_cache.free_pages` | pages holding nothing |
| `kv_cache.shared_pages`, `indexed_pages`, `evicted_pages` | pages held by two or more requests; in the prefix index; cached pages reclaimed so far |
| `prefix_cache.lookups`, `lookup_pages`, `hit_pages`, `miss_pages`, `hit_rate` | admissions looked up; full prompt pages they could have reused; reused; not; hits ÷ lookup pages |
| `prefix_cache.reused_tokens` | prompt tokens not prefilled. Exact: a copied last page counts its tokens but the last |
| `admission`, `preemptions`, `policy` | the KV admission mode, preemptions so far, the scheduling policy |

The server config's `engine.prefix_caching` (default off) and `engine.kv_admission`
(`on-demand` or `worst-case`) choose the behaviour these numbers describe. The counters are
cumulative since the engine started. Exporting them to a telemetry backend comes with the metrics
endpoint in Phase 4; until then this is where they are read.

## Invariants

The checks run after every engine step when `EngineLoopOptions::check_invariants` is on. It is on
by default, because the `NS_CHECK_INVARIANTS` build option is on; a release build turns the
option off. Each step's checks cover:

- **in the manager,** I1–I6;
- **in the backend,** that the engine's sequences and the manager's tables agree, block for block,
  and that the engine's own allocator stays empty;
- **in the engine loop,** I11 and I12.

A violation names the invariant, the block or sequence, and both sides of the broken equation. It
also stops the engine: running requests get `EngineStopped`, and the error is logged ("engine loop
crashed"). A server whose KV bookkeeping is wrong must not keep writing K/V into blocks it may no
longer own.

I7–I10 arrive with the prefix index and the LRU queue (Days 33 and 37), and I13 with GPU-side
table checks.

| # | Invariant | Catches |
| --- | --- | --- |
| I1 | Every block is in exactly one of: the free list, the LRU queue, or referenced by at least one table | lost and double-listed blocks |
| I2 | `FREE + CACHED + OWNED = num_blocks`, and every block id is in `[0, num_blocks)` | leaks, stray ids |
| I3 | A block's reference count equals the number of tables listing it | counting bugs, use after free |
| I4 | No table lists a block twice | aliasing within a sequence |
| I5 | A table has exactly `ceil(len / block_size)` blocks, at most `max_pages_per_sequence` | over- and under-allocation |
| I6 | A block with count > 1 is full and indexed | sharing a block that can still change |
| I7 | Every indexed block is full; index and block records agree in both directions | stale index entries |
| I8 | An indexed block's hash equals `H(hash of the block before it in the table, its tokens)` | broken chains, wrong reuse |
| I9 | The block that receives a step's new token has count 1 and is not indexed | writes into shared blocks |
| I10 | The LRU queue holds exactly the `CACHED` blocks, ordered by last release | unevictable leaks, evicting live blocks |
| I11 | The scheduler's view equals the backend's: each running request's cached tokens are its sequence's length and cover its pages, and used pages = the requests' exclusive pages + shared pages | policy deciding on wrong numbers |
| I12 | A released or finished sequence has no table | references from dead requests |
| I13 | (debug) The device block tables equal the host tables after upload | stale tables on the GPU |

**Fault injection (Day 30)** corrupts each of them on purpose (a count off by one, a double
free, a block in two lists, a write into a shared block) and checks that the checker names it.

## Testing

| Day | Test | Proves |
| --- | --- | --- |
| 27 | CPU reference allocator, own unit tests | the oracle behaves as specified |
| 29 | 10,000 seeded random allocate / append / share / release sequences, manager vs reference, after every operation | the manager equals the oracle (KV-004) |
| 30 | Invariant checker and fault injection | every invariant is checked |
| 31 | Shared block freed only by its last owner | reference counting |
| 32 | Same prefix → same chain; any change in model, weights, dtype, block size, RoPE or a token → a different hash, from that block on; XXH64 and the encoding match an independent implementation; no collisions among 200,000 blocks | hash identity |
| 33 | A second request with the same system prompt reuses its full blocks and the hit is counted; never the last token's block; stops at the first miss; verified hits; duplicates; all-or-nothing admission; the manager equals the reference on 10,000 random programs with admit, commit and match | prefix index |
| 34 | With a prefix hit, the chat references still equal Hugging Face's greedy tokens, case by case, at batch size 1; an identical prompt reusing all but its last page too | prefill from a cached offset |
| 35 | An identical whole-page prompt copies its last page and prefills one token; the copy needs a free block or nothing changes; both requests match Hugging Face | copy-on-write |
| 36 | 16 requests, one 512-token system prompt: outputs identical to the no-cache run in identically shaped batches; hit rate and reused tokens reported | shared prefix end to end |
| 37 | Released indexed blocks stay CACHED and are reused; eviction takes the least recently released first, leaves first, only count-0 blocks; a copy's source is not evicted for its copy; outputs stay equal to Hugging Face under memory pressure | CACHED and LRU eviction |
| 38 | A preempted request resumes from prompt + generated tokens: every token generated exactly once, outputs correct; its own cached pages reused when they survive; an over-budget resumption still admitted; chunked prefill equals Hugging Face | preemption by recompute |
| 39 | Llama-3.2-1B FP16 from its `config.json`: 32 KiB per token, 512 KiB per block; broken configs rejected; on-demand preempts, worst-case never does, both correct | memory-aware admission |
| 34–36 | Prefix hit outputs vs Hugging Face; two requests diverging after a shared prefix; 16 requests with a 512-token system prompt vs a no-cache run, hit rate reported | prefix reuse is correct |
| 37 | Memory pressure: only count-0 blocks evicted, oldest first; outputs correct | eviction |
| 38 | Forced shortage: preempted requests resume from their cached tokens, match Hugging Face, and never re-stream a token | preemption |
| 41 | Compute Sanitizer memcheck and racecheck on the KV and scheduler suites | no invalid GPU memory access |
| 44 | 30 minutes of mixed high-concurrency load: no leaks, no invariant failures | stability |

## Open questions

- **Block size.** 16 is fixed by the fast decode kernel. The Day 42 sweep (8 / 16 / 32) needs the
  generic paged kernel for the other sizes, so it measures capacity and waste with it, and
  attention time with both kernels.
- **Hash width.** 64 bits plus verification is enough for correctness. If verification ever shows
  up in profiles, compare against a 128-bit hash without it.
- **Eviction policy.** LRU first. A cost-aware policy is tested only with a workload that can tell
  them apart (Day 43, `shared_prefix` and `memory_pressure`).

## Requirements map

| Requirement | Where |
| --- | --- |
| KV-001 Fixed-size block allocation | Free list and block records (Days 27–28) |
| KV-002 Block tables per sequence | Block tables (Day 28) |
| KV-003 Associate blocks with requests | Owner sequence ids on tables and blocks (Day 28) |
| KV-004 Safe release | I3, I12, property tests (Day 29), Compute Sanitizer (Day 41) |
| KV-005 Free-block tracking | I1, I2 after every step (Day 30) |
| KV-006 Reference counting | Day 31 |
| KV-007 Exact-prefix caching | Days 32–36 |
| KV-008 Eviction (LRU) | Day 37 |
| KV-009 Cache utilization metrics | Day 40 |
| KV-010 Detect invalid block states | The invariant checker and fault injection (Day 30) |
