# Phase 1 — Foundation (Days 1–10)

**Tag:** `v0.0.1-foundation`

## Built

- CMake (presets `dev-cuda`, `ci-cpu`), Docker dev container (CUDA 12.4, non-root), CI with
  warnings as errors.
- cuda-llm-engine integrated as a pinned git submodule (v1.0.2).
- C++ runtime: `InferenceBackend` interface, `CudaLlmEngineBackend`, greedy generation with RAII
  sequence cleanup, CUDA error checking.
- Python bindings (pybind11, GIL released, typed exceptions), packaged with scikit-build-core.
- Hugging Face-exact chat templating and incremental detokenization.
- OpenAI-compatible server: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`;
  explicit 400s for unsupported options; YAML config with strict validation.
- Production basics: request ids, JSON logs, prompt privacy by default, clean 500s,
  unhealthy state after GPU errors.

## Verified

- Tests: 25 ctest (cpu, gpu, integration, python) and 51 pytest, all passing.
- Greedy output token-identical to Hugging Face at every layer, including end to end through
  the HTTP API (Llama-3.2-1B base and Instruct).
- Fresh clone, README only: builds and passes all tests in FILL_FRESH_CLONE_TIME.
- Smoke observation (not a benchmark): 16 generated tokens in about 434 ms end to end on an
  RTX 4050 Laptop GPU (WSL2).

## Changed from the plan

- Serve Llama-3.2-1B-Instruct instead of the base model: the base model's special-token
  embeddings are untrained, so chat output was garbage.
- C++ JSON logger moved from Day 9 to Day 12: no C++ serving path to log yet.
- Compute Sanitizer moved to Day 41: it needs the WDDM debugger interface, unavailable on WSL.

## Found and carried forward

- The engine decodes the full `max_tokens` budget even after the stop token
  (visible as `generated_tokens` vs `completion_tokens` in the access log). Fix: Day 12.
- README gaps found by the fresh-clone check: FILL_GAPS_FOUND.

## Lessons

- FILL_LESSON (one or two sentences in your own words — what you would tell an interviewer
  you learned in Phase 1).
