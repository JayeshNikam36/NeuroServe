# Engine loop

`neuroserve::core::EngineLoop` (`runtime/core/include/neuroserve/core/engine_loop.hpp`) runs
generation for many requests on one dedicated engine thread.

## Rules

- **One thread owns the backend.** Backends are not thread-safe; callers only enqueue and wait.
- **Validation happens in `submit()`**, on the caller's thread: a bad request is `REJECTED`
  immediately and never reaches the queue.
- **Hand-over through the mutex.** A request is changed by the caller until it is queued under the
  lock, and only by the engine thread afterwards.
- **Batched decode.** Every step decodes all active requests with one `decode_step` call.
- **Per-step stop.** A request completes on the step that produces its stop token or its last
  allowed token; no tokens are generated past the stop.
- **Up-front KV reservation.** Admission reserves pages for prompt + `max_new_tokens`, so the cache
  cannot run out mid-generation. Simple and safe, but it under-uses memory; on-demand allocation
  with preemption replaces it later.
- **Failure isolation.** A prefill error fails one request; a decode error fails that step's
  batch; the loop keeps serving. Sequences and reservations are always released.
- **Clean shutdown.** `stop()` cancels queued and running requests with `EngineStopped`, so no
  caller waits forever.

## Verification

- 12 CPU tests (`tests/unit/test_engine_loop.cpp`), including 100 concurrent submitter threads,
  batch size, KV-limited admission, failure isolation and shutdown. A gate in the fake backend
  pauses the engine at a known point, so the tests do not depend on timing.
- The `tsan-cpu` preset runs every CPU test under ThreadSanitizer, locally and in CI.
