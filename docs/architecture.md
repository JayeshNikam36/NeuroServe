# NeuroServe architecture

NeuroServe is a GPU-aware LLM inference runtime with an OpenAI-compatible HTTP API.
This document describes the system as of `v0.0.1-foundation` (end of Phase 1).

## Layers

```
 HTTP client (curl, OpenAI SDK)
        │ JSON over HTTP
 ┌──────▼──────────────────────────────────────────────────────────────┐
 │ FastAPI server                  python/neuroserve/server/           │
 │  RequestContextMiddleware   request id, JSON access log, last-resort 500
 │  routes                     /health  /v1/models                     │
 │                             /v1/chat/completions  /v1/completions   │
 │  validation                 pydantic schemas (extra fields rejected)│
 │  _check_request             unsupported options → explicit 400      │
 │  _run                       worker thread + error → status mapping  │
 └──────┬──────────────────────────────────────────────────────────────┘
        │ messages / prompt
 ┌──────▼──────────────────────────────────────────────────────────────┐
 │ EngineService + tokenizer       python/neuroserve/                  │
 │  ChatTokenizer              HF chat template (sandboxed jinja2)     │
 │  stop ids                   from generation_config.json             │
 │  detokenization             incremental, never emits U+FFFD         │
 └──────┬──────────────────────────────────────────────────────────────┘
        │ token ids
 ┌──────▼──────────────────────────────────────────────────────────────┐
 │ neuroserve._core                bindings/ (pybind11)                │
 │  releases the GIL during GPU work; C++ exceptions → Python types    │
 └──────┬──────────────────────────────────────────────────────────────┘
 ┌──────▼──────────────────────────────────────────────────────────────┐
 │ C++ runtime                     runtime/                            │
 │  generate_greedy            prefill + decode loop, RAII sequence guard
 │  InferenceBackend           interface: prefill / decode_step / free │
 │  CudaLlmEngineBackend       pimpl; validates inputs before every    │
 │                             engine call (the engine exit(1)s on errors)
 └──────┬──────────────────────────────────────────────────────────────┘
 ┌──────▼──────────────────────────────────────────────────────────────┐
 │ cuda-llm-engine v1.0.2          third_party/ (git submodule)        │
 │  FP16 Llama kernels, paged KV cache (16-token pages)                │
 └─────────────────────────────────────────────────────────────────────┘
```

## Life of a chat request

1. The middleware assigns a request id (a safe client `X-Request-ID` is kept) and stores it in a
   `contextvars` variable, so every log record of this request carries it.
2. FastAPI validates the body; `_check_request` rejects options not supported yet
   (`stream`, `n>1`, `temperature>0`) and enforces `max_tokens` limits.
3. `_run` copies the request context and runs generation in a worker thread, so the event loop
   stays free and worker-thread logs keep the request id.
4. `ChatTokenizer` renders the model's chat template exactly like Hugging Face and encodes it.
   Prompts longer than `engine.max_prompt_tokens` are rejected with 413.
5. The bindings release the GIL and call the C++ runtime: prefill the prompt into the paged
   KV cache, then decode greedily up to `max_tokens`.
6. Output tokens are cut at the first stop id, detokenized, and returned with usage numbers.
7. The middleware writes one JSON access line (status, duration, token counts) and returns the
   request id in the response header.

## Error model

All errors use OpenAI's error format; the full table is in [errors.md](errors.md). Expected
failures map to 4xx/503. Unexpected exceptions become a 500 with the request id and no internal
detail; the traceback goes to the log only. A CUDA error marks the server unhealthy (`/health`
and every later request return 503) because the GPU context may be corrupted.

## Testing

| Layer | Tests | Label |
| --- | --- | --- |
| C++ runtime logic | GoogleTest with `FakeBackend`, no GPU | `cpu` |
| CUDA helpers | GoogleTest on the GPU | `gpu` |
| Engine backend | GoogleTest against real weights, HF reference outputs | `integration` |
| Bindings, tokenizer, API | pytest, fake engine and real model | `python` |

Correctness is checked against Hugging Face at every layer: token-for-token greedy parity for
the engine, the backend, the bindings, the chat tokenizer, and end to end through the HTTP API.
CI builds with warnings as errors and runs the CPU tests on every pull request.

## Configuration

One YAML file (`configs/dev.yaml`), validated at startup; unknown keys are errors. Sections:
`model`, `engine`, `limits`, `logging`, `server`. Prompt logging is off by default.

## Known limitations (planned)

| Limitation | Plan |
| --- | --- |
| One request's generation is not batched with others | Continuous batching (Phase 2) |
| Decoding runs the full `max_tokens` budget; the stop is applied afterwards | Per-step stop in the C++ engine loop (Day 12) |
| Greedy only, `n=1`, no streaming | Sampling and streaming (Phase 2) |
| No C++-side logging | C++ JSON logger (Day 12) |
| Compute Sanitizer unavailable on WSL | Run on native Linux or a cloud GPU (Day 41) |

Design decisions are recorded in [adr/](adr/).
