# GPU memory: pools and owners (Day 58)

Every byte NeuroServe knows of on the GPU has an owner: **weights**, **workspace**, **kv** or
**other**. `GET /v1/runtime/state` reports them under `gpu_memory` (MiB), `Engine.memory()` in
bytes, and `CudaLlmEngineBackend::memory_report()` in C++.

## Two ways memory is counted

- **Through the pool.** Everything NeuroServe allocates itself goes through
  `GpuMemoryManager::allocate(owner, bytes, label)` (`runtime/backend/include/neuroserve/backend/gpu_memory.hpp`):
  a stream-ordered memory pool (`cudaMemPoolCreate`, `cudaMallocFromPoolAsync` / `cudaFreeAsync`
  on the legacy default stream) with a **release threshold** (`CudaLlmEngineConfig::memory_release_threshold_bytes`,
  64 MiB): freed memory up to the threshold stays in the pool for the next allocation, the rest
  goes back to the driver at the next synchronization. A device without memory pools falls back
  to `cudaMalloc` (`pool.supported` false).
- **Recorded from the engine.** cuda-llm-engine allocates its weights, KV pool, activations and
  decode batch state inside its own calls, with `cudaMalloc`. Routing those through the pool would
  mean changing the engine, which stays at v1.0.2; the backend records them with
  `record_external(owner, bytes, label)` instead, sized from the shapes (exact) or, where the
  engine's buffers are not all known, from the device's free memory before and after the call
  (measured).

| owner | through the pool (NeuroServe) | recorded (cuda-llm-engine) |
| --- | --- | --- |
| weights | | the model's FP16 weights (exact, from the shapes; the output head is tied to the embedding) |
| workspace | batched-prefill scratch; CUDA Graph decode buffers and their cuBLAS workspace | load_model's activations, scratch and cuBLAS handle (measured); the decode batch state (exact) |
| kv | **each sequence's block table** | the KV pool: K and V of every layer, `num_pages + 1` pages (exact; the extra page is the pad page of CUDA Graph decode) |
| other | | |

`items` lists what each owner's bytes are made of, by label.

## Block tables from the pool

The engine's `new_sequence()` called `cudaMalloc` for every request's block table, and
`free_sequence()` called `cudaFree` when it finished. `cudaFree` can wait for the device. The
backend now builds and frees sequences itself the same way (`engine/paged.cu` does nothing else
in them), with the block table from the pool: an allocation per request is a pool lookup, and
freeing returns it to the pool in stream order without waiting.

## What the report says

- `owners`: per owner, the live bytes, how many came through the pool and how many were recorded,
  the peak, and the live allocations.
- `pool`: the release threshold, what the pool holds from the driver now (`reserved`: in use plus
  kept for reuse), what is in use, and the peak.
- `device`: total and used memory (`cudaMemGetInfo`: every process), and **unattributed**: used
  memory that is neither recorded nor in the pool. That is the CUDA context, cuBLAS's own
  allocations, allocations the measured part missed, and other processes on the GPU.

## Measured (Day 58)

The GPU test's report (RTX 4050 Laptop, 6,140 MiB; Llama 3.2 1B FP16, 512 KV pages, max batch 8):

| | MiB |
| --- | --- |
| weights | 2,357 (1.24 billion FP16 parameters) |
| kv | 256 (513 pages of 16 tokens x 32 KiB) |
| workspace | 52.9, of which 2.0 through the pool |
| other | 0 |
| pool reserved | 32 (threshold 64): the pool takes memory from the driver in large blocks |
| device used | 3,830 |
| **unattributed** | **1,133** |

More than a gigabyte of the device's used memory is nobody's in the ledger: the CUDA context,
the cuBLAS and kernel libraries loaded on the device, and whatever other processes (on Windows,
the display) hold. That is the figure a VRAM budget (Day 59) has to leave as headroom rather than
assume; it is measured here, not estimated.

## Tests

`tests/unit/test_memory_ledger.cpp` (the counting, on the CPU: owners, sources, items by label,
peaks, a release that matches no record, many threads); on the stand-in engine, every owner is
recorded, block tables come from the pool and go back, the CUDA Graph buffers are workspace from
the pool; on the GPU (`CudaLlmEngineBackendTest`), the weights recorded match `weights.bin`, the
KV pool its exact size, 8 sequences' block tables come from the pool and return, and a manager on
its own counts, frees and refuses a pointer it did not allocate.
