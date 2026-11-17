# NeuroServe

[![CI](https://github.com/JayeshNikam36/NeuroServe/actions/workflows/ci.yml/badge.svg)](https://github.com/JayeshNikam36/NeuroServe/actions/workflows/ci.yml)

An experimental GPU-aware LLM inference runtime featuring continuous batching,
paged KV-cache management, CUDA execution optimization and real-time
performance observability, measured against vLLM on an NVIDIA RTX 4050 Laptop GPU.

> **Status:** v0.1 — OpenAI-compatible streaming server with continuous batching, four scheduling
> policies and published latency percentiles. Next: KV-cache depth (Phase 3). See the roadmap below.

## What this is

NeuroServe is the serving, scheduling, memory-governance and observability layer
built on top of [cuda-llm-engine](https://github.com/JayeshNikam36/cuda-llm-engine),
a from-scratch C++/CUDA inference engine for Llama-3.2-1B with a paged KV cache
and split-KV decode attention.

## Results (v0.1)

First policy comparison on an RTX 4050 Laptop GPU (Llama 3.2 1B, FP16, `max_batch` 8): medians
over 6 measured rounds of 120 requests, paired and order-balanced, with GPU clocks recorded.

| Profile | Policy | TTFT P50 | TTFT P95 | ITL P50 | max ITL P95 | e2e P95 | output tok/s | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `short` | `fcfs` | 29.8 ms | 149 ms | 16.5 ms | 49.9 ms | 1.25 s | 234 | 0 |
| `short` | `token-budget` | 29.8 ms | 164 ms | 16.6 ms | 49.1 ms | 1.30 s | 234 | 0 |
| `short` | `decode-first` | 30.0 ms | 157 ms | 16.5 ms | 49.5 ms | 1.29 s | 234 | 0 |
| `mixed` | `fcfs` | 51.9 ms | 888 ms | 16.8 ms | 147 ms | 6.75 s | 255 | 0 |
| `mixed` | `token-budget` | 52.1 ms | 890 ms | 16.8 ms | 147 ms | 6.73 s | 255 | 0 |
| `mixed` | `decode-first` | 22.3 s | 29.5 s | 16.1 ms | 82.7 ms | 32.8 s | 180 | 35 of 720 |

`short`: ~40-token chat prompts, 64 output tokens, 4 requests/s. `mixed`: 32/128/480-token
prompts, 32/128/384 output tokens, 2 requests/s.

- **On `short` the policy makes no practical difference** (every paired change within ±2.1%),
  and on `mixed` `token-budget` matches `fcfs`: at this load the batch limit binds first.
- **`decode-first` cuts the worst gaps between tokens by 44%** (max ITL P95, better in 6 of 6
  rounds), but without chunked prefill it starves long prompts: at 2 requests/s its queue grows
  until requests time out.

Analysis: [docs/benchmarking.md](docs/benchmarking.md). Full report:
[benchmarks/results/day24-policy-comparison.md](benchmarks/results/day24-policy-comparison.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `runtime/` | C++ runtime libraries (`neuroserve::core`) |
| `cuda/` | CUDA kernels and GPU utilities (`neuroserve::cuda`) |
| `tests/` | GoogleTest suites, labelled `cpu` and `gpu` |
| `docker/` | Development container |
| `scripts/` | `dev.sh` (container), `build.sh` (configure, build, test) |
| `third_party/cuda-llm-engine` | Execution engine (git submodule, pinned) |
| `docs/adr` | Architecture decision records |
| `tools/` | Command-line utilities (`ns_device_info`) |
| `runtime/backend/` | `InferenceBackend` implementation on cuda-llm-engine (`neuroserve::backend`) |
| `bindings/` | pybind11 module (`neuroserve._core`) |
| `python/neuroserve/` | Python package |
| `python/neuroserve/server/` | OpenAI-compatible API server (FastAPI) |
| `configs/` | Server configuration (YAML) |
| `python/neuroserve/bench/` | Load generator, policy comparison, GPU sampler |
| `benchmarks/results/` | Committed benchmark results (`raw/` is local only) |
| `docs/` | Architecture, engine loop, scheduling, streaming, latency, benchmarking, errors |

## Build

```bash
bash scripts/dev.sh --build bash scripts/build.sh dev-cuda   # GPU build + all tests
bash scripts/dev.sh bash scripts/build.sh ci-cpu             # CPU-only build (same as CI)
```
### Python

```bash
bash scripts/dev.sh bash -c "PYTHONPATH=python python3 scripts/generate_demo.py 'The capital of France is'"
```

```python
import neuroserve
engine = neuroserve.Engine("/models/llama-3.2-1b-fp16")
engine.generate(prompt_ids, max_new_tokens=32)   # greedy, GIL released during generation
```
### API server

```bash
NS_PUBLISH_PORT=8000 bash scripts/dev.sh bash -c "PYTHONPATH=python python3 -m neuroserve.server --config configs/dev.yaml --host 0.0.0.0"
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "llama-3.2-1b-instruct", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 32}'
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`
(prompt as text or token ids), `DELETE /v1/requests/{X-Request-ID}` (cancel a request in progress;
a client disconnect cancels too), `GET /v1/runtime/decisions` (why the scheduler admitted, deferred,
rejected or preempted each request; see [docs/scheduling.md](docs/scheduling.md)),
`GET`/`PUT /v1/runtime/policy` (switch the scheduling policy while serving).

Greedy decoding. `"stream": true` streams the tokens as they are
generated (server-sent events in OpenAI's chunk format, so the official `openai` client works; see
[docs/streaming.md](docs/streaming.md)). The server listens on localhost only; the port is published
to 127.0.0.1.

NeuroServe request options (optional `neuroserve` object): `return_token_ids` adds the generated
token ids to the response; `ignore_eos` generates exactly `max_tokens` (benchmarks, parity tests);
`priority` is 0 (low), 1 (normal, default) or 2 (high), used by the `priority` scheduling policy
([docs/scheduling.md](docs/scheduling.md)).
Responses from both endpoints match Hugging Face token for token (`tests/python/test_api_parity.py`).

Benchmarking: `python -m neuroserve.bench.loadgen` sends a seeded, reproducible workload (Poisson
arrivals at wall-clock times) and reports latency percentiles; `python -m neuroserve.bench.compare`
compares scheduling policies on it (paired, order-balanced rounds, GPU clocks and temperature
recorded). See [docs/benchmarking.md](docs/benchmarking.md).

Every response carries an `X-Request-ID` header (send your own to trace requests end to end).
Logs are JSON lines with the request id on every record; prompts are not logged unless
`logging.log_prompts` is enabled. Error codes: [docs/errors.md](docs/errors.md).

### Model and reference files

Integration tests need the converted Llama-3.2-1B weights and Hugging Face reference outputs:

- `~/models/llama-3.2-1b-fp16/` — `weights.bin`, `index.tsv` (from cuda-llm-engine's export scripts)
- `~/models/llama-3.2-1b-ref/` — `prompt_NN_ids.i32`, `prompt_NN_greedy.i32` (copied from cuda-llm-engine's `tests/reference/bin`)

`scripts/dev.sh` mounts `~/models` read-only at `/models`. Without these files the integration tests are skipped.

- `~/models/llama-3.2-1b-instruct/` — tokenizer and chat template (`hf download meta-llama/Llama-3.2-1B-Instruct tokenizer.json tokenizer_config.json special_tokens_map.json generation_config.json --local-dir ~/models/llama-3.2-1b-instruct`)
- `~/models/llama-3.2-1b-ref/chat_reference.json` — Hugging Face chat-template reference (`scripts/make_chat_reference.py`, run on the host with transformers)

- `~/models/llama-3.2-1b-instruct-fp16/` — served model (Llama-3.2-1B-Instruct), converted with `scripts/export_weights.py`
- `~/models/llama-3.2-1b-ref/instruct_greedy.json` — Hugging Face greedy outputs for chat prompts (`scripts/make_generation_reference.py`)

### Check your GPU

```bash
bash scripts/dev.sh ./build/dev-cuda/tools/ns_device_info          # human-readable
bash scripts/dev.sh ./build/dev-cuda/tools/ns_device_info --json   # machine-readable
```

## Getting the code

```bash
git clone --recurse-submodules git@github.com:JayeshNikam36/NeuroServe.git
```

Release check (every test, then the live server through the official openai client):

```bash
bash scripts/dev.sh --build bash scripts/release_check.sh
```

## Roadmap

| Release | Scope |
| --- | --- |
| v0.1 (released) | OpenAI-compatible streaming server, pluggable scheduling policies, latency percentiles |
| v0.2 | Varlen and chunked prefill, CUDA Graphs, GPU memory pools — each A/B measured |
| v0.3 | Live observability dashboard |
| v1.0 | Benchmarks vs vLLM / TensorRT-LLM, technical report, Docker release |

## License

Apache-2.0

