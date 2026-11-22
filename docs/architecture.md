# NeuroServe architecture

NeuroServe is a GPU-aware LLM inference runtime with an OpenAI-compatible HTTP API.
This document describes the system as of `v0.1.5-kv` (end of Phase 3: KV-cache depth).

## Layers

```
 HTTP client (curl, the official openai SDK)
        │ JSON over HTTP, server-sent events when streaming
 ┌──────▼──────────────────────────────────────────────────────────────┐
 │ FastAPI server                  python/neuroserve/server/           │
 │  RequestContextMiddleware   request id, JSON access log (plain ASGI,│
 │                             so a client disconnect stays visible)   │
 │  /v1/chat/completions  /v1/completions    blocking or "stream": true│
 │  /health  /v1/models  DELETE /v1/requests/{id}                      │
 │  /v1/runtime/decisions  /v1/runtime/policy                          │
 │  validation                 pydantic schemas; unsupported → 400     │
 │  error mapping              engine exceptions → 400/413/429/499/503 │
 └──────┬──────────────────────────────────────────────────────────────┘
        │ messages / prompt
 ┌──────▼──────────────────────────────────────────────────────────────┐
 │ EngineService + tokenizer       python/neuroserve/                  │
 │  ChatTokenizer              HF chat template (sandboxed jinja2)     │
 │  StreamingGeneration        reads tokens in a worker thread,        │
 │                             detokenizes incrementally (never half   │
 │                             a character)                            │
 └──────┬──────────────────────────────────────────────────────────────┘
        │ token ids
 ┌──────▼──────────────────────────────────────────────────────────────┐
 │ neuroserve._core                bindings/ (pybind11)                │
 │  Engine.run / stream / cancel / stats / latency / decisions,        │
 │  set_scheduling_policy; TokenStream.read(timeout) with the GIL      │
 │  released; C++ exceptions → typed Python exceptions                 │
 └──────┬──────────────────────────────────────────────────────────────┘
 ┌──────▼──────────────────────────────────────────────────────────────┐
 │ C++ runtime                     runtime/core, runtime/backend       │
 │  EngineLoop                 one engine thread owns the backend;     │
 │                             bounded submit queue; one step at a time│
 │  SchedulingPolicy           fcfs | token-budget | decode-first |    │
 │                             priority: admit / preempt, pure function│
 │  Request                    lifecycle state machine, timestamps     │
 │  LatencyHistogram           queue / TTFT / ITL / e2e percentiles    │
 │  DecisionLog                why each request was admitted, deferred,│
 │                             rejected or preempted (ring buffer)     │
 │  CudaLlmEngineBackend       InferenceBackend on cuda-llm-engine     │
 └──────┬──────────────────────────────────────────────────────────────┘
 ┌──────▼──────────────────────────────────────────────────────────────┐
 │ cuda-llm-engine v1.0.2          third_party/ (git submodule)        │
 │  FP16 Llama kernels, paged KV cache (16-token pages), batched decode│
 └─────────────────────────────────────────────────────────────────────┘
```

KV memory is NeuroServe's own since Phase 3: `KVCacheManager` owns every 16-token block, its
reference count and its place in the prefix index, and the CUDA backend reads and writes K/V
through the block tables it is given. Released blocks stay `CACHED` (reusable) until evicted,
least recently used first; a prompt whose leading full blocks are cached is prefilled from the
first uncached token. The design, its invariants and the measurements are in
[kv-cache.md](kv-cache.md).

Outside the request path, `neuroserve.bench` (load generator, policy comparison, prefix-caching
comparison, stress run, NVML GPU sampler) drives servers for measurements; see
[benchmarking.md](benchmarking.md).

## One engine step

The engine thread repeats one step while there is work, and sleeps when there is none:

1. Apply cancellations (`DELETE`, client disconnects) and reject queued requests that have waited
   longer than `engine.queue_timeout_ms` (HTTP 503).
2. Switch to a newly requested scheduling policy, if there is one.
3. Build a `SchedulerInput` snapshot (queued requests with the pages their prefill needs, running
   requests with the pages their next decode needs, free pages, `max_batch`, the token budget)
   and ask the policy for a `ScheduleDecision`: whom to preempt, whom to admit. The loop checks
   the decision against every limit before applying it, and records an explanation with its
   numbers for each request concerned.
4. Preempt (the request keeps its tokens and later resumes by prefilling prompt + generated
   tokens), prefill the admitted
   requests, and decode one token for every request that was already running, in one batched
   call. Prefill and decode tokens together stay within `engine.max_batch_tokens`.
5. Publish each new token to its request's buffer, record timings, and complete the requests
   that produced a stop token or their last allowed token.

Details: [engine-loop.md](engine-loop.md), [scheduling.md](scheduling.md),
[request-lifecycle.md](request-lifecycle.md), ADRs 0003–0005.

## Life of a streamed chat request

1. The middleware assigns a request id (a safe client `X-Request-ID` is kept) and stores it in a
   `contextvars` variable, so every log record of this request carries it.
2. FastAPI validates the body; options not supported yet (`n>1`, `temperature>0`) are explicit
   400s, and `max_tokens` is checked against the limits.
3. `ChatTokenizer` renders the model's chat template exactly like Hugging Face. A prompt longer
   than `engine.max_prompt_tokens` is a 413.
4. `Engine.stream` submits the request on the caller's thread: validation happens there, and a
   full queue (`engine.max_queue`) is a 429 with `Retry-After`.
5. The engine thread admits the request when the policy allows it, prefills it, then decodes it
   in the batch with the other running requests, one token per step.
6. Each new token goes into the request's own buffer in C++; the engine thread never calls into
   Python. A worker thread reads the buffer with the GIL released, detokenizes, and the event
   loop sends one OpenAI-format chunk per read. If the client disconnects, the request is
   cancelled at the next step and its pages are freed.
7. The last chunk carries `finish_reason` and the request's timings (`neuroserve` block); with
   `include_usage`, a usage chunk follows; then `data: [DONE]`. The access log gets one line when
   the stream starts and one, with usage and timings, when it completes.

Details: [streaming.md](streaming.md), [latency.md](latency.md).

## Error model

All errors use OpenAI's error format; the full table is in [errors.md](errors.md). Expected
failures map to 4xx/503 (overload: 429 `queue_full`, 503 `queue_timeout`; cancelled: 499).
Unexpected exceptions become a 500 with the request id and no internal detail; the traceback goes
to the log only. A CUDA error marks the server unhealthy (`/health` and every later request
return 503) because the GPU context may be corrupted. Within the engine, a prefill error fails
one request and a decode error fails that step's batch; the loop keeps serving.

## Testing

| Layer | Tests | Label |
| --- | --- | --- |
| C++ runtime logic | GoogleTest with `FakeBackend`, no GPU: engine loop, request states, scheduler golden traces, histograms, decision log | `cpu` (also under ThreadSanitizer) |
| CUDA helpers | GoogleTest on the GPU | `gpu` |
| Engine backend | GoogleTest against real weights, HF reference outputs | `integration` |
| Bindings, tokenizer, API, benchmarks | pytest, fake engine and real model | `python` |

Correctness is checked against Hugging Face at every layer: token-for-token greedy parity for
the engine, the backend, the bindings, the chat tokenizer, and end to end through the HTTP API,
streaming included. 24 concurrent requests with a KV cache small enough to force preemption match
Hugging Face under every policy. Outputs are deterministic but not batch-invariant: what is and is
not guaranteed is in [ADR-0006](adr/0006-determinism-not-batch-invariance.md). CI builds with warnings as errors and runs the CPU tests, plain
and under ThreadSanitizer, on every pull request. `scripts/release_check.sh` is the release gate:
every test, then the live server through the official openai client.

## Configuration

One YAML file (`configs/dev.yaml`), validated at startup; unknown keys are errors. Sections:
`model`, `engine` (batch size, token budget, KV pages, scheduling policy, queue limits),
`limits`, `logging`, `server`. Prompt logging is off by default; the server listens on localhost.

## Known limitations (planned)

| Limitation | Plan |
| --- | --- |
| A prompt is prefilled in one step, so it must fit the token budget, and `decode-first` starves long prompts under load ([benchmarking.md](benchmarking.md)) | Chunked prefill and mixed prefill/decode steps (Days 52–53) |
| Prefix caching shares full 16-token blocks only, and is off by default in `configs/dev.yaml` | On by default once the release gate covers it; partial-block sharing not scheduled |
| Measured on one laptop GPU; no baseline against another server yet | vLLM baseline and comparison (Days 94–95) |
| No authentication; the runtime policy endpoint is an operator control | Security defaults review (Day 115) |
| Deterministic but not batch-invariant: at a near-tie the batch (or a prefix hit) can change a greedy choice ([ADR-0006](adr/0006-determinism-not-batch-invariance.md)); since Day 38 a preempted request resumes from its own tokens, so it never contradicts what it streamed | The cause is in the kernels' numerics (reduction order across batch sizes): Phase 4 kernel work. Day 46's profile times phases, not numerics |
| Greedy decoding only, `n=1` | Not scheduled |

Design decisions are recorded in [adr/](adr/).
