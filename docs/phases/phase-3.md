# Phase 3 — KV-cache depth (Days 26–45)

**Tag:** `v0.1.5-kv`

## Built

- **A KV-cache manager NeuroServe owns** (`KVCacheManager`): 16-token blocks, block tables per
  sequence, an ordered free set, owners on every block. The CUDA backend takes every block from
  it; the engine's own allocator is gone.
- **A reference implementation** (`ReferenceBlockManager`) with the same contract, written for
  clarity, as the oracle of the property tests.
- **Per-step invariant checks** (I1–I12) in the manager and the engine loop, on by default in
  development builds, with a fault-injection test for each.
- **Reference counting** through the backend and the engine, preemption included.
- **Prefix hashing:** XXH64 implemented in the repository, a pinned byte encoding, chained block
  hashes whose root covers the model id, a sampled weights fingerprint, dtype, block size and
  RoPE; tested against an independent implementation.
- **The prefix index** (`admit`, `commit`, `match`) with every hit verified before use.
- **Prefill from a cached offset** (`attention_prefill_paged`, `model_prefill_from`) and
  **copy-on-write** of the block holding the last prompt token (`copy_page`).
- **`CACHED` blocks and LRU eviction**, leaves first.
- **Preemption by recompute that keeps the tokens**, with chunked prefill for long inputs.
- **Memory-aware admission** from `config.json` (`on-demand`, `worst-case`).
- **Metrics:** `GET /v1/runtime/state`.
- **Measurement tools:** the block-size sweep, the prefix-caching comparison and the stress run.

## Verified

- **Property tests:** 10,000 seeded random programs (about 600,000 operations), manager against
  reference after every operation; a divergence shrinks to a short reproduction.
- **Correctness with the cache:** with a hit, outputs equal those without and equal Hugging Face,
  through prefill from an offset, copy-on-write, eviction and resumption after preemption.
- **Compute Sanitizer:** memcheck and racecheck, 0 errors on the CUDA and GPU integration suites.
- **Prefix caching measured:** TTFT P50 −93% and throughput up to +45% on `shared_prefix`; +17%
  with a quarter of the pool, where LRU keeps about 87% of the possible reuse.
- **Stress:** 30 minutes of mixed high-concurrency load, no failure, no invariant violation, no
  leak, no drift.
- **Release check:** `scripts/release_check.sh` passed.

The exit criteria and their evidence are in [kv-cache.md](../kv-cache.md#measured-results).

## Changed from the plan

- **Preemption keeps the generated tokens** (Day 38) instead of restarting from the prompt. A
  restart could contradict what the request had already streamed (found in Phase 2, ADR-0006);
  resuming from its own tokens cannot.
- **Chunked prefill came early, for resumptions only:** prompt + generated tokens can exceed the
  prefill scratch. Chunked prefill of new prompts stays in Phase 4.
- **Compute Sanitizer ran on WSL** after all, once the WDDM debugger interface was enabled in the
  Windows registry. It runs on the CUDA and GPU integration suites; the unit suite makes no CUDA
  call, so there is nothing in it to instrument.
- **The prefix comparison starts a server per run,** because prefix caching is an engine setting:
  every run starts with an empty cache and the two variants differ in that setting only.

## Found and carried forward

- **A cost-aware eviction policy is not worth it yet:** LRU reached 87% of the no-eviction bound.
- **The stress run did not force preemption:** cached blocks are reclaimed first and running
  requests never filled a quarter of the pool. A stress profile that does is a follow-up.
- **Prefix caching is off by default.** Turning it on in `configs/dev.yaml` waits until the
  release gate's batch-agreement check runs with it (a hit, like batching, can change a greedy
  choice at a near-tie).
- **The engine's paged kernels are the baseline:** at batch 8, decode attention takes about 282
  µs per layer, and without the cache the server saturates at about 3.6 requests/s on
  `shared_prefix`. Phase 4 starts by profiling the step.
- **Timing-sensitive tests** (the load generator's send lag on a loaded machine) flaked once and
  passed on rerun; they need wider margins.

## Lessons

- An oracle pays for itself: the reference manager and the property tests made every later change
  to the manager a diff against known-good behaviour.
- Check invariants at every step, in development builds, and fault-inject each one; then a stress
  run can say "no violation" and mean it.
- Measure the bound, not just the result: computing what a never-evicting cache could reuse turned
  "66–74% hit rate" into "LRU keeps 87% of what is possible", which settled the eviction question.
