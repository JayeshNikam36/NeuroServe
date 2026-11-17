# Phase 2 — Scheduling and serving (Days 11–25)

**Tag:** `v0.1.0`

## Built

- **Engine loop.** It runs on its own C++ thread behind a thread-safe submit queue and owns the
  backend: continuous batching with one batched decode call per step, a per-step stop, and a
  per-step token budget (`max_batch_tokens`, decode tokens first).
- **Request lifecycle.** A state machine with 11 states and timestamps for every state; an
  illegal transition throws.
- **Scheduling.** A policy interface (a pure function whose every decision the loop checks) with
  four policies: `fcfs`, `token-budget`, `decode-first` and `priority` (with aging, so nothing
  starves). Golden-trace tests cover them, and the policy can be switched at runtime
  (`PUT /v1/runtime/policy`).
- **KV memory.** KV pages are allocated on demand, with preemption by recompute (ADR-0004).
- **Streaming** over server-sent events in OpenAI's chunk format, through a per-request token
  buffer in C++. The engine thread never calls into Python.
- **Cancellation** by `DELETE /v1/requests/{id}` or a client disconnect, applied at the next
  step boundary.
- **Overload protection:** a bounded queue (429 with `Retry-After`) and a queue timeout (503).
- **Latency instrumentation:** queue time, TTFT, ITL and e2e for every request, and percentiles
  from fixed geometric buckets (error under one bucket, about 9%).
- **Explainable scheduling:** every admit, defer, reject and preempt is recorded with the numbers
  it was made on (`GET /v1/runtime/decisions`).
- **Benchmarking tools:**
  - an open-loop load generator with seeded Poisson arrivals and a versioned prompt dataset;
  - a policy comparison with paired, order-balanced rounds;
  - an NVML GPU sampler.

## Verified

- **Release check.** `scripts/release_check.sh` passed from a fresh clone that followed the
  README. It covers the build, every ctest suite (CPU, GPU, integration, and Python with 129
  pytest tests), and the live server on the real model. In that last part the official openai
  client streams, and under every policy 24 concurrent streamed requests each return exactly the
  text of the same request sent alone. This also closes the fresh-clone check deferred from
  Phase 1.
- **24-request workload.** 24 concurrent requests, with a KV cache small enough to force
  preemption, match Hugging Face token for token under every policy on the GPU
  (`tests/python/test_preemption.py`). Its CPU counterpart runs in CI.
- **ThreadSanitizer** runs every CPU test in CI.
- **Latency percentiles published.** The first policy comparison measured 6 rounds after a
  discarded warm-up round, with paired and order-balanced runs and GPU clocks recorded:
  [benchmarks/results/day24-policy-comparison.md](../../benchmarks/results/day24-policy-comparison.md).

## Changed from the plan

- **Streaming:** a per-request token buffer in C++, read by a worker thread with the GIL released,
  instead of a C++ callback into an asyncio queue. The engine thread never waits for the GIL.
- **Disconnect detection:** the middleware became plain ASGI, because Starlette's
  `BaseHTTPMiddleware` hid client disconnects from the endpoints.
- **Preemption by recompute** came in Phase 2, with on-demand KV pages (planned for Phase 3). On
  demand, the batch can outgrow the cache, so preemption was needed at once.
- **The runtime policy endpoint** was added so one warm server serves a whole comparison. A switch
  applies at the engine's next step, and the comparison confirms in the decision log which policy
  made every decision of every run.

## Found and carried forward

- **`decode-first` is unstable on mixed prompt lengths at 2 requests/s.** TTFT P50 was 22.3 s, 35
  of 720 requests hit the 30 s queue timeout, and throughput fell by 34%. While anything decodes,
  a 480-token prompt cannot be admitted under its prefill cap, and strict order holds every request
  behind it. In exchange, max ITL P95 fell by 44%. Fix: chunked prefill (Days 52–53).
- **The policy made no practical difference on `short`, and `token-budget` matched `fcfs` on
  `mixed`.** At this load the batch limit (8) binds before the token budget does.
- **The first request after a server start took about 5 s** (the GPU at idle clocks, plus one-time
  CUDA initialisation), and every request that arrived meanwhile queued behind it. Benchmarks
  discard a warm-up round. A warm-up request at startup would hide the delay from users (not yet
  scheduled).
- **Still open:** the client-disconnect check with curl through Docker's port forwarding on WSL.
  Disconnects are tested through a real uvicorn server in the test suite.

## Lessons

- **A fake must model the real component's timing, not only its interface.** The test fake
  switched policy at once; the real engine switches at its next step, and an idle engine takes
  none. The first real benchmark stopped at its first switch.
- **Pair the runs.** On `short`, TTFT P95 varied from about 45 to 300 ms between rounds under every
  policy, far more than any policy effect. Only same-schedule comparisons separate the two.
- **Read the real report before trusting it.** A rounded "0% of samples" warning, a minimum clock
  that was really the idle ramp, and "+38580%" all came from code that had passed its tests.
