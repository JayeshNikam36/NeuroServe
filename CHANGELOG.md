# Changelog

All notable changes are listed here. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0, minor versions may change the API.

## [Unreleased] — Phase 4: GPU optimization

### Added
- Step profile: the engine thread times every step's phases (housekeeping, scheduling,
  preemption, the backend's prefill and decode calls, the rest of the host work, invariant checks)
  and its idle time; `GET /v1/runtime/profile` (`?records=true` for each of the newest 8,192
  steps), `DELETE /v1/runtime/profile` to reset; `neuroserve.Engine.step_profile()`.
- NVTX ranges for the step's phases in CUDA builds, so Nsight Systems shows them next to the
  kernels; `scripts/nsys_profile.sh`.
- `neuroserve.bench.step_profile`: decode, prefill and mixed scenarios; the time split, decode cost
  by batch size, prefill cost per token, and a Chrome / Perfetto trace per scenario.

## [0.1.5] — Phase 3: KV-cache depth (tag `v0.1.5-kv`)

### Added
- KV-cache manager owned by NeuroServe (`KVCacheManager`): fixed 16-token blocks, per-sequence
  block tables, an ordered free set; the CUDA backend takes every block from it. A CPU reference
  implementation (`ReferenceBlockManager`) is the oracle for 10,000 seeded property-test programs.
- Invariant checker: the manager's and the engine loop's invariants (I1–I12) after every engine
  step when built with `NS_CHECK_INVARIANTS` (on by default; `neuroserve._core.check_invariants`
  says which), with fault injection for each.
- Reference-counted sharing of KV blocks between requests.
- Exact-prefix caching: chained XXH64 hashes over full blocks (rooted in the model id, a weights
  fingerprint, dtype, block size and RoPE), a prefix index whose every hit is verified before use,
  prefill from the cached offset (`attention_prefill_paged`, `model_prefill_from`), and
  copy-on-write of the block holding the last prompt token. `engine.prefix_caching` in the server
  config, `neuroserve.Engine(prefix_caching=...)`.
- `CACHED` blocks: released blocks stay reusable until needed; LRU eviction, leaves first.
- Memory-aware admission from the model's `config.json` (KV bytes per token and per block):
  `engine.kv_admission` `on-demand` or `worst-case`.
- `GET /v1/runtime/state`: KV occupancy, used / cached / free / shared pages, prefix lookups,
  hits, misses, hit rate, reused tokens, evictions, preemptions, admission mode.
- Benchmarks: `ns_block_size_sweep` (8 / 16 / 32-token blocks), `neuroserve.bench.prefix_ab`
  (prefix caching on vs off, a fresh server per run), `neuroserve.bench.stress` (sustained mixed
  load with checks for leaks, invariant failures and drift); the `shared_prefix` and
  `memory_pressure` load profiles.
- Compute Sanitizer runs (`scripts/compute_sanitizer.sh`, target `compute_sanitizer`).

### Changed
- Preemption keeps a request's tokens: it resumes by prefilling prompt + generated tokens, reusing
  its own cached blocks, so nothing is generated twice and what it streamed stays valid. Inputs
  longer than the prefill scratch are prefilled in chunks.

### Results (RTX 4050 Laptop GPU, Llama 3.2 1B FP16)
- Prefix caching on `shared_prefix`: TTFT P50 −93% at 4 requests/s, throughput +45% at saturation,
  hit rate 82.9% (the no-eviction bound); on `memory_pressure` with a quarter of the pool TTFT P50
  −63%, throughput +17%, LRU at about 87% of the possible reuse
  ([day43](benchmarks/results/day43-prefix-caching.md)).
- 30-minute stress run: 3,725 requests and 95,752 evictions, no failure, no invariant violation,
  no block held after the drain, GPU and host memory flat, latency within 2% from start to end
  ([day44](benchmarks/results/day44-stress.md)).
- Block size 16 kept: 2.6% waste; 8 gains 1.4% capacity, 32 loses 2.8%, attention time equal
  ([day42](benchmarks/results/day42-block-size.md)).
- Compute Sanitizer memcheck and racecheck: 0 errors on the CUDA and GPU integration suites
  ([tests/logs/compute-sanitizer.md](tests/logs/compute-sanitizer.md)).

### Known limitations
- Prefix caching is off by default in `configs/dev.yaml` (`engine.prefix_caching: true` turns it
  on); a hit can change a greedy choice at a near-tie, like batching (ADR-0006).
- Only full 16-token blocks are shared; a prompt's partial last block is always recomputed.
- New prompts are still prefilled in one step under the token budget; chunked prefill of new
  prompts and mixed prefill/decode steps come in Phase 4.
- The stress run did not force preemption (running requests never filled the pool); preemption is
  covered by unit and GPU tests.

## [0.1.0] — Phase 2: Scheduling and serving

### Added
- Engine loop on a dedicated C++ thread: continuous batching with batched decode, per-step stop,
  per-step token budget (`engine.max_batch_tokens`), thread-safe submit queue.
- Request lifecycle state machine with per-state timestamps.
- Pluggable scheduling policies (`fcfs`, `token-budget`, `decode-first`, `priority` with aging),
  checked decisions, golden-trace tests; `GET`/`PUT /v1/runtime/policy` to switch while serving.
- On-demand KV pages with preemption by recompute (ADR-0004).
- Streaming (`"stream": true`): server-sent events in OpenAI's chunk format, usage chunk,
  per-request timings in the last chunk.
- Cancellation: `DELETE /v1/requests/{X-Request-ID}` and client disconnects.
- Overload protection: `engine.max_queue` (429 `queue_full`, `Retry-After`) and
  `engine.queue_timeout_ms` (503 `queue_timeout`).
- Latency instrumentation: queue / TTFT / ITL / e2e per request; P50/P95/P99 in `/health`.
- Explainable scheduling decision log: `GET /v1/runtime/decisions`.
- C++ JSON logging.
- Benchmarking: `neuroserve.bench.loadgen` (seeded open-loop Poisson load, versioned prompt
  dataset), `neuroserve.bench.compare` (paired, order-balanced policy comparison), NVML GPU sampler.
- First published results: [benchmarks/results/day24-policy-comparison.md](benchmarks/results/day24-policy-comparison.md).
- `scripts/release_check.sh`: the release gate (every test, then the live server through the
  official openai client).

### Changed
- Every request is served through the engine loop (Phase 1 ran one blocking generation per
  request); response and error formats are unchanged.

### Known limitations
- Greedy decoding only, `n=1`.
- A prompt is prefilled in one step (at most `engine.max_batch_tokens` tokens); `decode-first`
  starves long prompts under load until chunked prefill (Phase 4).
- No prefix caching yet (Phase 3).
- Deterministic, not batch-invariant: at a near-tie the batch can change a greedy choice
  ([ADR-0006](docs/adr/0006-determinism-not-batch-invariance.md)).

## [0.0.1] — Phase 1: Foundation

### Added
- OpenAI-compatible HTTP server: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`.
- Greedy generation on the GPU through cuda-llm-engine v1.0.2 (FP16 Llama, paged KV cache).
- Hugging Face-exact chat templating, stop tokens, and detokenization.
- Python package `neuroserve` with pybind11 bindings to the C++ runtime.
- Structured JSON logging with request ids; prompts not logged by default.
- OpenAI-format errors ([docs/errors.md](docs/errors.md)); unhealthy state after GPU errors.
- Docker dev environment, CMake presets, CI, and tests (C++, CUDA, integration, Python).

### Known limitations
- Greedy decoding only, `n=1`, no streaming, no batching across requests.
