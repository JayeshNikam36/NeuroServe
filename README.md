# NeuroServe

[![CI](https://github.com/JayeshNikam36/NeuroServe/actions/workflows/ci.yml/badge.svg)](https://github.com/JayeshNikam36/NeuroServe/actions/workflows/ci.yml)

An experimental GPU-aware LLM inference runtime featuring continuous batching,
paged KV-cache management, CUDA execution optimization and real-time
performance observability, measured against vLLM on an NVIDIA RTX 4050 Laptop GPU.

> **Status:** Phase 1 — foundation. Not yet usable. See the roadmap below.

## What this is

NeuroServe is the serving, scheduling, memory-governance and observability layer
built on top of [cuda-llm-engine](https://github.com/JayeshNikam36/cuda-llm-engine),
a from-scratch C++/CUDA inference engine for Llama-3.2-1B with a paged KV cache
and split-KV decode attention.

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

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (greedy, non-streaming for now).
The server listens on localhost only; the port is published to 127.0.0.1.

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`
(prompt as text or token ids). Greedy, non-streaming for now.

NeuroServe request options (optional `neuroserve` object): `return_token_ids` adds the generated
token ids to the response; `ignore_eos` generates exactly `max_tokens` (benchmarks, parity tests).
Responses from both endpoints match Hugging Face token for token (`tests/python/test_api_parity.py`).

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

## Roadmap

| Release | Scope |
| --- | --- |
| v0.1 | OpenAI-compatible streaming server, pluggable scheduling policies, latency percentiles |
| v0.2 | Varlen and chunked prefill, CUDA Graphs, GPU memory pools — each A/B measured |
| v0.3 | Live observability dashboard |
| v1.0 | Benchmarks vs vLLM / TensorRT-LLM, technical report, Docker release |

## License

Apache-2.0

