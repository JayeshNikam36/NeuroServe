# Engine loop

`neuroserve::core::EngineLoop` (`runtime/core/include/neuroserve/core/engine_loop.hpp`) runs
generation for many requests on one dedicated engine thread.

## Rules

- **One thread owns the backend.** Backends are not thread-safe; callers only enqueue and wait.
- **Validation happens in `submit()`**, on the caller's thread: a bad request is `REJECTED`
  immediately and never reaches the queue.
- **Hand-over through the mutex.** A request is changed by the caller until it is queued under the
  lock, and only by the engine thread afterwards.
- **Batched decode.** Every step decodes all running requests with one `decode_step` call;
  requests admitted in a step are prefilled in it and join the decode batch from the next step.
  A step's tokens (prefill + decode) are bounded by `max_batch_tokens`.
- **Per-step stop.** A request completes on the step that produces its stop token or its last
  allowed token; no tokens are generated past the stop.
- **On-demand KV pages with preemption.** A request holds pages only for the tokens it has
  cached. Each step the scheduling policy sees how many new pages the running batch needs; if they
  are not free, it preempts the most recently admitted requests, which free their pages, go back
  to the front of the queue and restart from their prompt (see docs/scheduling.md and ADR-0004).
- **Failure isolation.** A prefill error fails one request; a decode error fails that step's
  batch; the loop keeps serving. Sequences and their pages are always released.
- **Clean shutdown.** `stop()` cancels queued and running requests with `EngineStopped`, so no
  caller waits forever.

## Verification

- 12 CPU tests (`tests/unit/test_engine_loop.cpp`), including 100 concurrent submitter threads,
  batch size, KV-limited admission, failure isolation and shutdown. A gate in the fake backend
  pauses the engine at a known point, so the tests do not depend on timing.
- The `tsan-cpu` preset runs every CPU test under ThreadSanitizer, locally and in CI.

## Python and logs

- `neuroserve.Engine` owns the backend and an `EngineLoop`. `Engine.run()` submits one request and
  waits with the GIL released, so concurrent Python threads (the API server's worker threads) are
  batched together. `Engine.generate()` is `run()` without stop tokens.
- `EngineService` has no lock any more, and stop tokens are handled by the engine:
  `generated_tokens` equals `completion_tokens`.
- `Engine.kv_stats()` reads the loop's snapshot, refreshed by the engine thread before a request
  is answered. `/health` therefore never calls the backend while the engine thread is using it.
- The C++ runtime writes JSON lines to stderr (logger `neuroserve.engine`, level from
  `logging.level`). Each finished request logs `engine_request_id` (`ns-…`) and `request_id` (the
  HTTP `X-Request-ID`); the access log carries `engine_request_id` too, so the two lines can be
  joined. Token ids and text are never logged. The logger writes directly rather than through
  Python logging, so the engine thread never needs the GIL (the engine's destructor holds the GIL
  while it joins the engine thread).
- Not yet verified for `engine.device_index` other than 0: the engine thread must use the same
  CUDA device as the thread that loaded the model.
