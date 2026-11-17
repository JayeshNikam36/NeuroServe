# Changelog

All notable changes are listed here. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0, minor versions may change the API.

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
