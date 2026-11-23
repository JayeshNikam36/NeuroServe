# CUDA Graphs for decode (design, Day 55)

Status: design. Capture and replay are Day 56, the A/B benchmark Day 57.

A decode step launches about 280 small kernels, one batch of new tokens through 16 layers. On
the host, each launch costs a few microseconds. On the GPU, each dependent kernel waits for the
previous one's launch to land. A CUDA Graph records the step's launches once, then replays them
with one `cudaGraphLaunch`. This note covers:

- what the decode step does today, and what in it blocks capture;
- which buffers the graph needs to keep at fixed addresses;
- the batch buckets, and how a step is padded up to its bucket;
- the plan for Days 56 and 57.

It reads the engine's decode path at cuda-llm-engine v1.0.2 (`third_party/cuda-llm-engine`,
`engine/batched.cu`) and NeuroServe's backend (`runtime/backend/src/cuda_llm_engine_backend.cpp`).

## Scope

- **Decode-only steps.** These are steps with no prefill, so the batch has one row per sequence
  and every shape depends only on the batch size. Between arrivals, every step of a serving run
  is one.
- **Not covered:**
  - Prefill and mixed steps (Day 53). Their row count changes with every prompt.
  - The logits path (`model_decode_batch`). The server only uses greedy decoding.

## The decode step today

`CudaLlmEngineBackend::decode_step` runs the checks and KV bookkeeping on the host. It then
calls `model_decode_batch_greedy`, which runs `decode_step_gpu` and then argmax:

| # | Where | What | Blocks capture? |
| --- | --- | --- | --- |
| 1 | backend `checked_for_decode`, `append_for_decode` | input checks; one KV block per sequence that needs one, from the manager; `seq.pages` refreshed | no: host only, stays outside the graph |
| 2 | `batched.cu:112-113`, `ensure_pages` | tops up `seq.pages` from the engine's allocator (a no-op here: the backend gave the pages already) | no: host only |
| 3 | `batched.cu:129-139` | builds `tables` (`n x max_pages`), `lens`, `pos` in `std::vector`s: host allocations every step | no, but moves to pinned staging |
| 4 | `batched.cu:140-143` | **four synchronous `cudaMemcpy` host-to-device from pageable memory** (tables, lens, positions, token ids) | **yes**: a synchronous copy is not allowed in a capture, and each one waits for the GPU |
| 5 | `batched.cu:144` | `choose_num_splits(n, 8, max_len)`: the attention kernel's grid, decided on the host | **yes**: the graph fixes the grid (see buckets) |
| 6 | `batched.cu:148, 189` | `cudaEventRecord` of `ev0` and `ev1` (timing) | moves outside the graph |
| 7 | `batched.cu:149-154`, 16 x `layer_forward_batched` | the kernels: embedding; per layer, rmsnorm, Q/K/V `linear` x 3, RoPE, `kv_scatter_batch` x 2, `attention_decode_fast`, O `linear`, residual, rmsnorm, gate and up `linear`, SwiGLU, down `linear`, residual; final rmsnorm; head `linear` | capturable, **but every call goes to the legacy default stream** (`stream = 0` defaults), which cannot be captured |
| 8 | `linear.cu:7`, `cublasSetStream` + `cublasGemmEx` (`CUBLAS_GEMM_DEFAULT_TENSOR_OP`) | 7 GEMMs per layer and the head | capturable on a stream, **with a workspace set by `cublasSetWorkspace`** (`model.cu:228` only calls `cublasCreate`), so cuBLAS allocates nothing during capture |
| 9 | `batched.cu:187` | `argmax_rows` on the GPU | capturable |
| 10 | `batched.cu:190` | **`cudaEventSynchronize(ev1)`**: the host waits for the step | moves after the launch: one sync per step stays (the scheduler needs the tokens) |
| 11 | `batched.cu:196` | **synchronous `cudaMemcpy` device-to-host** of the `n` next tokens | **yes**: becomes an async copy into pinned memory |
| 12 | `batched.cu:202`; backend `finish_decode` | `seq.len += 1`; manager and engine compared; prefix caching indexes the page a token filled | no: host only |

**Nothing is allocated on the GPU during a step.** `alloc_batch_state` (`batched.cu:59-72`) and
`load_model` (`model.cu:214-236`) allocate every device buffer once, so no pointer changes from
step to step. That is the main thing graphs need.

**The kernel wrappers take a stream.** `embedding_lookup`, `rmsnorm`, `rope_apply`,
`kv_scatter_batch`, `swiglu`, `residual_add`, `linear` and `argmax_rows` all end with
`cudaStream_t stream = 0`. `attention_decode_fast`'s declaration continues past the lines read
today (`kernels/attention_decode_fast.cuh:14-17`); Day 56 checks that it takes a stream too.
Capture therefore needs no change to the engine's kernels, only a forward that passes a stream.

**The kernels read their per-step inputs from the device.** RoPE and `kv_scatter_batch` read
positions from `d_pos`, `kv_scatter_batch` and `attention_decode_fast` read the block tables from
`d_tables`, and attention reads each sequence's length from `d_lens`. A graph captured once
therefore stays correct as sequences grow, as long as those buffers are refreshed before each
launch. Only the batch size and the split count change a grid.

## Batch buckets and padding

There is one graph per bucket of **1, 2, 4, 8 and 16** sequences, capped at `max_batch`. A step of
`n` sequences uses the smallest bucket `b >= n`. Rows `n .. b - 1` are padding, and padding must be
harmless and cheap:

- **Harmless:**
  - A padding row feeds token 0 at position 0 with length 1.
  - Its block table points every entry at a **pad page**, one KV page reserved for this purpose
    and never given to the KV manager. Its K/V writes land there and never in a real sequence's
    pages, and its attention reads one key from that page.
  - Its argmax result is never copied back.
- **Cheap:** decode is bound by reading the weights (Day 46). A GEMM with 4 rows instead of 3
  reads the same weights, so padding adds a little compute and no weight traffic. The Day 57 A/B
  measures that cost.

**The split count is fixed within a bucket.** `choose_num_splits` (`attention_decode_fast.cuh:22`)
returns `64 / (8 n)`, capped at the number of pages of the longest sequence, then at 32, and at
least 1. For the buckets that gives:

| bucket | uncapped splits | split counts a graph may need |
| --- | --- | --- |
| 1 | 8 | 1-8 (fewer than 8 while the longest sequence is under 128 tokens) |
| 2 | 4 | 1-4 (under 64 tokens) |
| 4 | 2 | 1-2 (under 32 tokens) |
| 8 | 1 | 1 |
| 16 | 0, so 1 | 1 |

**Padding does not change the split count.** Integer division gives `64 / (8 n)` the same value
for every `n` in a bucket: 3 rows get 2 splits, like bucket 4, and 5 to 8 rows get 1, like
bucket 8. Padding rows have length 1, so they never raise the longest length either.

Graphs are therefore keyed by **(bucket, splits)**: 8 + 4 + 2 + 1 + 1 = **16 graphs** at most.
One step's graph is chosen with the same function the eager path uses. The pairs that are only
needed for short sequences can be captured on first use. An alternative is one graph per bucket
at its uncapped split count, if Day 56 shows `attention_decode_fast` handles splits with no pages
(today's cap suggests it may not). Day 56 decides between the two with a test, not an assumption.

## Static buffers

A graph bakes in its pointers. It gets its own input and output buffers, sized for the largest
bucket and allocated once, at capture:

| buffer | size | filled |
| --- | --- | --- |
| pinned host staging | ids, positions, lengths: 16 each; tables: `16 x max_pages` ints | by the host, before the launch |
| device inputs `d_ids`, `d_pos`, `d_lens`, `d_tables` | the same | by a memcpy node from the staging, first in the graph |
| device `d_next`, pinned host `next` | 16 ints | argmax, then a memcpy node back, last in the graph |
| attention workspace | `attention_decode_fast_workspace_floats(16, 32 heads, 32 splits)` | inside the graph |
| `d_normed`, `d_logits` | `16 x hidden`, `16 x vocab` halves | inside the graph |
| activations `m.d_x`, `m.scratch` | the model's, sized `max_T` rows (at least 16) | inside the graph; shared with the prefill path, which runs on the same stream order |
| cuBLAS workspace | a few MB, set once with `cublasSetWorkspace` on the model's handle | by cuBLAS |
| KV pools, weights, `d_inv_freq` | already static | |

A step then takes these calls:

1. Write `n` real rows and `b - n` padding rows into the pinned staging.
2. `cudaGraphLaunch` the (bucket, splits) graph.
3. `cudaStreamSynchronize`.
4. Read `n` tokens from the pinned `next`.

That is one host sync per step instead of six: four copies in, the event wait, and the copy out.

**Streams.** The graph is captured on, and launched into, a stream the backend creates as a
*blocking* stream. The prefill and mixed paths keep using the legacy default stream, which
synchronises with blocking streams, so the two stay ordered with no events added. Capture runs
when the engine is idle (at startup, or on first use of a (bucket, splits) pair), in
`cudaStreamCaptureModeThreadLocal`.

## Where the code goes

The capture is done in **NeuroServe's backend, not in the submodule**. A new
`runtime/backend/src/decode_graph.cu` (`backend_detail::DecodeGraphs`) holds a copy of
`layer_forward_batched` that takes a stream, like `prefill_from_offset.cu` copies the engine's
prefill. It calls the engine's kernel wrappers with that stream, so cuda-llm-engine stays at
v1.0.2.

`CudaLlmEngineBackend::decode_step` sends a step of at most 16 sequences to the graph when
graphs are on (a new `cuda_graphs` config option, off until the Day 57 A/B says otherwise). The
eager path remains for everything else and for comparison.

## Tests (Day 56)

1. **Graph against eager, every bucket.** For `n` = 1..16 (padded to its bucket), across lengths
   that hit every split count, a replay and an eager pass give the same next tokens and the same
   K/V in the real sequences' pages.
   - The eager pass for the comparison runs with the same padded row count. cuBLAS may pick a
     different GEMM for 3 rows than for 4, so comparing against an unpadded eager step could
     differ in the last bits.
   - The pad page is the only page the padding rows write to.
2. **Hugging Face.** The existing reference tests (8 prompts, decoded together) run with graphs
   on and still match token for token.
3. **Splits with no pages.** A replay at a split count above the pages of a short sequence,
   compared with the eager pass. This decides "graph per (bucket, splits)" against "graph per
   bucket".
4. **Interleaving.** Prefill, mixed and graph decode steps alternate (the engine loop on the
   GPU), with the KV invariants checked after every step.

## What to measure (Day 57)

- **Decode step time, by batch size,** with and without graphs (the step profile's
  `decode_by_batch`).
- **Per-step host time,** with and without graphs.
- **The cost of padding.** Each `n` against its bucket.
- **The design:** an interleaved A/B, 5 rounds, round 1 discarded.

Expected gain: up to the launch gaps the profile shows between kernels. About 280 launches at a
few microseconds each is an estimate of 1 ms or less against a 16 ms batch-1 step, so single-digit
percent. This is to be measured, not promised. On WSL2, Nsight Systems records no kernel data
(Day 46), so the A/B is the measurement.

Graphs will not touch the decode-only steps with 18-20 ms of host time that Day 54's step profile
found. That time is outside the backend call, in the loop's own work. It stays a separate open
question, and the Day 57 A/B records host time per step so it can be tracked.
