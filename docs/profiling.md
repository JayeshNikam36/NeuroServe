# Profiling

Where an engine step's time goes, measured two ways: the engine's own step profile, always
available, and Nsight Systems, when it is installed. Phase 4 (GPU optimization) starts from the
baseline recorded here (Day 46).

## The step profile

The engine thread times every step on the host clock (`steady_clock`), phase by phase:

| Phase | What it covers |
| --- | --- |
| `housekeeping` | applying cancellations, queue timeouts, reading the backend's free pages, each request's own pages |
| `schedule` | the policy's decision, the loop's checks of it, and the explanations for the decision log |
| `preempt` | releasing the sequences of preempted requests |
| `prefill` | the backend's prefill calls, one per admitted request |
| `decode` | the backend's batched decode call |
| `host` | the rest of the step on the host: creating sequences, appending and publishing tokens, completions, statistics, the KV snapshot |
| `invariants` | the KV invariant checks (`check_invariants`, on in development builds) |

`prefill` and `decode` are the backend calls only. They return their tokens to the host, so their
time is the GPU work they launch plus the launches and the copy back: the GPU's share of the step,
seen from the host. The phases add up to the step's total. Time with nothing to do is `idle`.

Every step is kept as a record (the newest 8,192, `EngineLoopOptions::step_profile_capacity`),
with its start time, its phases, and what it did (requests and tokens prefilled, the decode batch).
Recording costs a few clock reads and one copy per step, so the profile is always on.

- `GET /v1/runtime/profile`: totals since the last reset, idle time, step counts.
  `?records=true` adds every kept step.
- `DELETE /v1/runtime/profile`: starts a new profile (before a measurement).
- `neuroserve.Engine.step_profile(records=False)` and `reset_step_profile()` in Python.

### The baseline tool

    python -m neuroserve.bench.step_profile --config configs/dev.yaml --out-dir benchmarks/results/raw/profile-1

Starts a server (or uses one, `--url`), warms it up, and runs three scenarios, resetting the
profile before each:

| Scenario | Load | Shows |
| --- | --- | --- |
| `decode` | 32 requests at once, 64-token prompts, 256 output tokens | the batch filling to 8 and draining: decode time at every batch size |
| `prefill` | 32 requests at once, 480-token prompts, 8 output tokens | prefill cost per token |
| `mixed` | the `mixed` profile at 2 requests/s | a realistic mix of steps |

For each it reports the share of busy time in prefill, decode, host work and invariant checks;
step P50 and P99; decode-only steps grouped by batch size (decode time, per sequence, host time,
invariant checks); and prefill time per token. `steps-<scenario>.csv` has every step, and
`trace-<scenario>.json` is a Chrome trace of them (open it in [ui.perfetto.dev](https://ui.perfetto.dev)
or `chrome://tracing`): one slice per step, with its phases inside it.

## Nsight Systems

    bash scripts/dev.sh bash scripts/nsys_profile.sh            # decode and prefill scenarios

Runs the server under `nsys profile` (CUDA API calls, kernels, memory copies, NVTX ranges), runs
the step-profile tool against it, stops it and exports summaries: kernels by GPU time, CUDA API
calls, memory copies, NVTX ranges (`stats_*.csv`). The report, `neuroserve.nsys-rep`, opens in the
Nsight Systems GUI on Windows. In CUDA builds the engine marks its phases with NVTX ranges
(`ns.step`, `ns.schedule`, `ns.prefill`, `ns.decode`, `ns.invariants`;
`runtime/core/include/neuroserve/core/nvtx.hpp`), so the timeline shows each step's kernels under
the phase that launched them.

`nsys` is in the dev image (`docker/dev.Dockerfile`): the newest 2024 standalone release from
NVIDIA's apt repository (2024.6.2 when added). Not CUDA 12.4's own 2023.4: it records this
multi-threaded server but cannot import the recording ("Wrong event order"). After changing the
Dockerfile, `bash scripts/dev.sh --build` rebuilds the image. CPU sampling and context-switch
tracing are off (`--sample=none --cpuctxsw=none`): they need privileges WSL's containers do not
have, and the step profile already covers the host side.

**What it records on WSL 2.** NVTX ranges and CUDA API calls (launches, copies, synchronisation),
with their timing on the CPU. Not the kernels' own GPU activity: the report has no kernel data on
this setup (a Docker container under WSL 2). Per-kernel GPU time therefore needs another tool on
the kernel days: CUDA events around the kernels under test, or Nsight Compute.

## Baseline (Day 46)

RTX 4050 Laptop GPU, Llama 3.2 1B (FP16), `max_batch` 8, 512-token step budget, prefix caching
off, invariant checks on; SM clock 2655 MHz throughout. Full report:
[benchmarks/results/day46-step-profile.md](../benchmarks/results/day46-step-profile.md).

| Scenario | prefill | decode | host (loop) | invariant checks | step P50 | step P99 |
| --- | --- | --- | --- | --- | --- | --- |
| `decode` | 3.5% | 94.6% | 1.3% | 0.6% | 16.2 ms | 38.5 ms |
| `prefill` | 85.5% | 12.3% | 2.1% | 0.1% | 153.1 ms | 161.9 ms |
| `mixed` | 8.2% | 89.9% | 1.2% | 0.6% | 16.1 ms | 42.5 ms |

Decode-only steps in `mixed`, by batch size:

| batch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode P50 (ms) | 15.09 | 15.60 | 15.71 | 15.81 | 15.91 | 15.99 | 16.07 | 16.16 |
| per sequence (ms) | 15.09 | 7.80 | 5.24 | 3.95 | 3.18 | 2.66 | 2.30 | 2.02 |

- **Decode is bound by reading the weights, and batching it is almost free.** A step costs 15.1
  ms with one sequence and 16.2 ms with eight (+7% for 8x the tokens). Each step reads all the
  weights: about 2.5 GB in FP16, and at the GPU's roughly 192 GB/s that alone takes about 13 ms.
  The decode path is already near that floor, so the lever is the batch: more sequences per step
  (a larger `max_batch`, paid for in KV memory) raise throughput almost linearly.
- **Prefill is compute-bound and blocks the batch.** 0.28–0.30 ms per token (about 3,500 tokens
  per second) for 64- and 480-token prompts alike: the weight matmuls dominate, not attention. A
  480-token prompt is one 135 ms step during which no running request gets a token; that is the
  step P99 of 38–42 ms in `decode` and `mixed`, and the inter-token stalls behind it. The Phase 4
  answers: faster prefill attention and matmuls, and chunked prefill in steps mixed with decode.
- **The host is not the bottleneck.** The loop's own work is about 0.1 ms per step (1–2% of busy
  time) and the invariant checks another 0.1 ms (0.6%), so the checks stay on in development
  builds. A few single steps spent 3–28 ms on the host when many requests arrived at once
  (creating their sequences); rare, but they show up in the P99.
- **Kernel launches: about 325 per step.** Under `nsys` (decode and prefill scenarios: 1,080
  decode and 68 prefill calls) the backend made 351,000 kernel launches (`cudaLaunchKernel` and
  `cuLaunchKernel`, about 7 µs each on the CPU) and waited in `cudaEventSynchronize` for 12.4 ms of
  a 16.4 ms decode step. The launches overlap the GPU work today, since the GPU is the slower
  side, but they cost about 2.5 ms of CPU per step: once decode gets faster, CUDA graphs or fused
  kernels have to cut them. 5,819 synchronous `cudaMemcpy` calls (median 42 µs) are the next host
  cost to remove.
- **The two tools agree.** `ns.decode` under `nsys` (median 16.3 ms over 1,080 calls) matches the
  step profile's decode time; profiling under `nsys` moved the split by under two points.

