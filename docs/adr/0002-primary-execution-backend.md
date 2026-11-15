# ADR-0002: cuda-llm-engine as the primary execution backend

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
NeuroServe's scheduler, KV-cache manager and memory budget must control real
GPU memory. A paged KV-cache manager only does that if the attention kernel
reads keys and values through the block table. Standard Hugging Face PyTorch
models use contiguous per-sequence caches, so a block manager beside them
would only simulate paging.

## Options considered
1. **PyTorch / Hugging Face as the primary backend.** Fast to start, broad model
   support, but the KV manager would not own the memory the model uses.
2. **vLLM as the backend, NeuroServe as a layer on top.** Real paging, but the
   scheduling and memory work would be vLLM's, not ours.
3. **cuda-llm-engine as the primary backend.** Its paged decode attention
   already reads through block tables; it is verified token-for-token against
   Hugging Face and measured against vLLM.

## Decision
Option 3. Backends sit behind an `InferenceBackend` interface:
- `CudaLlmEngineBackend` — primary execution path.
- `HuggingFaceBackend` — correctness reference (golden outputs) only.
- vLLM and TensorRT-LLM — external comparison baselines, never in-process.

## Consequences
- v0.1 supports one model family (Llama-3.2-1B-Instruct); wider support is
  out of scope.
- Prefill performance is our responsibility (largest measured gap vs vLLM);
  it is scheduled for Phase 4.
- All results are directly comparable to the engine's published v1.0 baseline.