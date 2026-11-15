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

## Build

```bash
bash scripts/dev.sh --build bash scripts/build.sh dev-cuda   # GPU build + all tests
bash scripts/dev.sh bash scripts/build.sh ci-cpu             # CPU-only build (same as CI)
```
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

