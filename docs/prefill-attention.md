# Variable-length prefill attention: design

Status: v0 (Day 48) and the tiled kernel (Day 49) implemented in `runtime/kernels`, benchmarked
(Day 50); the engine prefills the requests of a step together with it (Day 51). This note was written before the code (Day 47), and each implementation day is
checked against it.

## Why

The Day 46 baseline ([profiling.md](profiling.md#baseline-day-46)) shows what prefill costs today:

- **0.28–0.30 ms per prompt token** (about 3,500 tokens per second), for 64- and 480-token
  prompts alike.
- **A 480-token prompt is one step of about 135 ms.** No running request gets a token during it;
  that is the 38–42 ms step P99 under load.
- **One backend call per admitted request.** Each call runs every layer for that prompt alone:
  three prompts admitted in one step are three full passes over the weights, one after another.

The prefill attention kernel written on Day 34 for prefill from a cached offset
(`attention_prefill_paged_kernel`, `runtime/backend/src/prefill_from_offset.cu`) was written to
be obviously correct, not fast (a prompt prefilled from zero goes through the engine's own prefill
kernels; Day 48 measures both):

- one thread block per (query head, new token), 128 threads;
- every block reads the whole of its sequence's K and V from global memory: for a prompt of T
  new tokens over S keys, `T x S` key reads per head instead of `S`;
- the scores of all S keys sit in shared memory (`S x 4` bytes), so shared memory grows with the
  context;
- one thread sums the softmax, in order.

What this design changes, and what it does not:

1. **Many sequences, one launch.** Every prompt admitted in a step is packed into one batch of
   tokens, and attention runs as one kernel over all of them, each sequence with its own cached
   prefix. This is what lets the engine prefill all admitted prompts in one pass over the weights
   (the matmuls then see `sum(T)` rows instead of `T`), and later run chunked prefill and mixed
   prefill/decode steps.
2. **Tiled attention with online softmax** (FlashAttention-2): K and V are read once per tile of
   queries, not once per query, and shared memory no longer grows with the context.
3. **Not the matmuls.** At these lengths the weight matmuls do most of prefill's arithmetic
   (`2 x 1.24 G` FLOPs per token, against `4 x S x 64 x 32` per token per layer for attention,
   with S averaging half the prompt under the causal mask: about 1–2% for a 480-token prompt). How much of the 135 ms the current attention kernel takes is
   not known yet: Nsight Systems on WSL 2 does not record kernel activity. **Day 48 measures it
   first** (CUDA events around the attention launches of one prefill), so the gain of each step
   is known, not assumed.

## Measured (Day 48)

`ns_attention_bench` (`benchmarks/attention`): one layer's prefill attention, CUDA events, median
of 20 runs, RTX 4050 Laptop GPU. Full table:
[benchmarks/results/day48-attention-v0.md](../benchmarks/results/day48-attention-v0.md).

| Case | today's kernel, ms per layer | v0, ms per layer | x 16 layers |
| --- | --- | --- | --- |
| 1 x 64 tokens | 0.126 | 0.129 | 2.0 ms |
| 1 x 480 | 5.58 | 5.57 | 89 ms |
| 1 x 2,048 | 102.2 | 102.7 | 1,635 ms |
| 4 x 480 | 22.3 | 22.1 | 356 ms |
| 256 new tokens after 1,024 cached | 16.4 | 14.1 | 262 ms |

- **Today's kernel is quadratic, and slow.** Time grows with new tokens x keys: 5.6 ms per layer
  at 480 tokens, 102 ms at 2,048. Every block rereads all of its sequence's K and V, and one
  thread sums each softmax.
- **It runs on prefix hits, chunks and resumptions, not on fresh prompts.** A prompt prefilled from
  zero goes through the engine's own prefill attention: Day 46's cost per token was flat from 64 to
  480 tokens (0.30 and 0.28 ms), which a quadratic term this size would not allow. So the slow
  kernel is not in the 135 ms of a fresh 480-token prompt. It is in every prefill from an offset:
  256 new tokens after a 1,024-token cached prefix spend about 262 ms in attention alone, where the
  rest of that prefill costs about 72 ms. Prefix caching won on Day 43 with prompts of at most 480
  tokens; with long cached prefixes, this kernel would eat the gain.
- **v0 is as fast as today's kernel**, as intended: the same algorithm, one launch for the whole
  pack instead of one per sequence (slightly faster for the pack and the cached case). It is the
  correctness reference.
- **So Day 49's tiled kernel is the win.** Reading K and V once per 64-query tile instead of once
  per query, with tensor cores, should take 480 tokens from 5.6 ms per layer to well under 1 ms.
  It makes prefill from an offset (prefix hits, chunked prefill, resumption) cost about what a
  fresh prompt costs, and it can then replace the engine's own prefill attention as well, so every
  prefill uses one kernel.

## v0 as implemented

`neuroserve::kernels::attention_prefill_varlen_v0` (`runtime/kernels`), with the interface above
in a `VarlenAttentionArgs` struct. One block of 128 threads per (packed token, query head); the
block finds its sequence by binary search over `cu_seqlens_q` (the tile map is for the tiled
kernel); scores for every key in shared memory, so at most 12,288 keys (48 KB). Tests:
`tests/cuda/test_attention_varlen.cpp` (8 tests, every case listed under Tests below).

## The tiled kernel as implemented (Day 49)

`attention_prefill_varlen_tiled(args, tile_map, num_tiles, shape)`, with `varlen_tile_map` building
the work list on the host (longest tiles first) and four tile shapes: 64 x 32 (default, chosen by
the sweep below), 64 x 64, 128 x 64 and 128 x 32 (query rows per block x keys per K/V tile). It follows the design above in
what it computes and in what it guarantees, with two simplifications for this first version:

- **WMMA, with the scores staged in shared memory.** `Q K^T` and `P V` run on tensor cores through
  the WMMA API (16 x 16 x 16 fragments, FP16 in, FP32 accumulate). WMMA does not say which lane
  holds which accumulator element, so each warp stores `S` to its own slice of shared memory, lanes
  0–15 do the online softmax one row each (FP32, `exp2f`, fixed order) and write `P` in FP16, and
  `P V` lands in the same slice before every lane folds its 32 output elements (row `k / 2`,
  columns `lane` and `lane + 32`) into registers. The design's register-resident softmax needs
  `mma.sync` with its documented fragment layout: the next step if the measurement says the
  shared-memory round trip matters.
- **K and V single-buffered**, loaded with 128-bit loads through the block table; double buffering
  with `cp.async` comes with that next step.

The shared-memory scratch for `S` makes blocks larger than the design's 40 KB:

| Shape | per block | blocks per SM (100 KB) | warps per SM |
| --- | --- | --- | --- |
| 64 x 64 | 53.8 KB | 1 | 4 |
| 64 x 32 | 40.8 KB | 2 | 8 |
| 128 x 64 | 89.5 KB | 1 | 8 |
| 128 x 32 | 72.5 KB | 1 | 8 |

Above 48 KB the launcher opts in (`cudaFuncSetAttribute`) after checking the device's limit.
85 registers per thread, no spills (`ptxas -v`, sm_89).

Tests (`tests/cuda/test_attention_varlen.cpp`, `AttentionVarlenTiled`): the reference cases of v0
(every length 1–2,048, the mixed pack with cached prefixes, random packs, a sharp softmax, no GQA)
for the default shape; every shape against v0 within the tolerance on a pack with every tile edge
(lengths 1, 31–33, 64, 65, 127–129, 480, 2,048; starts on and off page and tile boundaries);
batch independence and determinism for every shape; the tile map (every token once, longest
first); refused arguments.

### Measured (Day 49)

`ns_attention_bench`, one layer, CUDA events, median of 20 runs, RTX 4050 Laptop GPU. Full table:
[benchmarks/results/day49-attention-tiled.md](../benchmarks/results/day49-attention-tiled.md).

| Case | today's kernel | tiled, best shape | speed-up | x 16 layers, tiled |
| --- | --- | --- | --- | --- |
| 1 x 64 tokens | 0.13 ms | 0.027 ms (64 x 32) | 4.8x | 0.4 ms |
| 1 x 480 | 5.86 ms | 0.230 ms (64 x 32) | 25.5x | 3.7 ms |
| 1 x 2,048 | 109 ms | 3.10 ms (128 x 64) | 35.1x | 50 ms |
| 4 x 480 | 24.2 ms | 0.867 ms (64 x 32) | 27.9x | 14 ms |
| 256 new after 1,024 cached | 17.0 ms | 0.504 ms (128 x 64) | 33.7x | 8 ms |
| 8 x 32 | 0.33 ms | 0.056 ms (64 x 32) | 5.8x | 0.9 ms |

- **The target is met.** A 480-token prompt's attention went from 5.9 ms per layer to 0.23 ms
  (94 ms to 3.7 ms over 16 layers). The longer the context, the larger the gain (35x at 2,048
  keys), since the tiled kernel reads K and V once per 64 queries instead of once per query.
- **Prefill from an offset is no longer dominated by attention.** 256 new tokens after a 1,024-token
  cached prefix: about 262 ms of attention before, 8 ms now, against about 72 ms for the rest of
  that prefill. Long prefix hits, chunked prefill and resumption now cost about what their new
  tokens cost.
- **64 x 32 is the best shape up to 480 tokens, and the default.** Its 41 KB blocks fit two per SM
  (8 warps); 64 x 64's 54 KB blocks fit one (4 warps), and it is the slowest tiled shape everywhere.
  128 x 64 is 3–9% faster from 1,024 keys, where there are enough blocks to fill the GPU with fewer,
  larger ones. The engine can pick by length later; the gap is small.
- **Room left.** At 2,048 keys the kernel does about 5.5 TFLOPS (`4,096 x T^2` FLOPs per layer under
  the causal mask; 4.1 TFLOPS at 480). Occupancy, set by the shared-memory scratch for the scores,
  is the visible limit: the register-resident softmax (`mma.sync`) and double-buffered K/V
  (`cp.async`) are the next step if attention becomes the bottleneck again.

### Benchmark (Day 50)

`ns_prefill_kernel_bench`: fresh prompts of 64–2,048 tokens at batch 1 and batch 8, one layer, the
L2 flushed before every timed run, NVML clocks logged after every run. RTX 4050 Laptop GPU. Full
report: [benchmarks/results/day50-prefill-kernel.md](../benchmarks/results/day50-prefill-kernel.md).

| Length | batch 1: naive | tiled (best) | TFLOPS | speed-up | batch 8: naive | tiled (best) | TFLOPS | speed-up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.132 ms | 0.032 ms | 0.54 | 4.2x | 0.98 ms | 0.118 ms | 1.16 | 8.3x |
| 256 | 1.56 ms | 0.097 ms | 2.77 | 16.1x | 13.1 ms | 0.581 ms | 3.71 | 22.6x |
| 512 | 6.32 ms | 0.257 ms | 4.19 | 24.6x | 50.9 ms | 1.82 ms | 4.74 | 28.0x |
| 1,024 | 24.9 ms | 0.826 ms | 5.20 | 30.1x | 202 ms | 6.90 ms | 4.98 | 29.3x |
| 2,048 | 102 ms | 2.92 ms | 5.88 | 34.8x | 814 ms | 26.0 ms | 5.29 | 31.3x |

- **The naive kernel stays at 0.13–0.17 TFLOPS at every size**: it does not use the tensor cores
  and rereads K and V for every query, so more work only takes longer. The tiled kernel climbs from
  0.5 TFLOPS at 64 tokens to 5–5.9 TFLOPS from 1,024: 4x to 35x faster.
- **Against the hardware:** about 43% of the tensor cores' FP16 peak with FP32 accumulation, which
  is roughly 13.6 TFLOPS here (20 SMs x 2,655 MHz x 256 FLOPs per SM per clock, the rate per SM
  and clock implied by NVIDIA's published figures for GeForce Ada). The plateau at about 5 TFLOPS
  from 512 tokens at batch 8 is the occupancy limit noted above.
- **Short prompts are launch- and occupancy-bound.** At 64 tokens one prompt is 32 blocks, too few
  for 20 SMs; batch 8 doubles the TFLOPS (0.54 to 1.16). Packing the prompts of a step into one
  launch, which the varlen layout allows, is what helps short prompts most.
- **A warm L2 flatters the short cases.** Without the flush, 64–256 tokens run 12–46% faster (K and
  V of the last run are still in the 24 MB L2), 512 tokens up to 10%; from 1,024 tokens the
  difference is under 5%. Every number in this table is with the flush.
- **The clocks held:** SM at 2,655 MHz in every one of 1,175 runs, 51–70 °C, no thermal events.
  NVML reported its power-cap reason on 32 runs (3%) without the clock dropping below 2,655 MHz.
- **Shapes:** 64 x 32 is best or within 4% everywhere up to 1,024 tokens; 128 x 64 is 4–8% faster at
  2,048. 64 x 32 stays the default.
- **Correctness held through the sweep:** the tiled outputs differ from the naive ones by at most
  0.0005 in any element.

## Batched prefill in the engine (Day 51)

The requests admitted in one engine step are prefilled together:

- **The engine** creates a sequence for each and makes one `InferenceBackend::prefill_batch` call
  with all of them (`PrefillItem`: sequence, prompt, then the first token or the error). Each item
  succeeds or fails on its own, as with `prefill()`: a request whose prefill fails fails alone and
  its sequence is freed. The default `prefill_batch` prefills the items one by one, so any backend
  keeps working unchanged.
- **The CUDA backend** checks every item and gives it its KV blocks first (prefix-cache hits
  included), then packs the items whose new tokens fit one chunk into one forward pass while the
  pack fits the 512-token prefill scratch and `max_batch` sequences
  (`backend_detail::model_prefill_batch`): the embedding, every projection, RoPE (with each token's
  own position) and the MLP run once on the stacked tokens; each sequence's new K/V are scattered
  into its own pages; attention is `attention_prefill_varlen_tiled` (64 x 32) over the pack; the
  last row of every sequence goes through the final norm and the head together, one
  `[sequences x hidden]` by `[hidden x vocab]` product. A pack of one, or an item longer than a
  chunk (a resumed request), takes `prefill()`'s path.
- **Counted:** `batched_prefills()` and `batched_prefill_sequences()` on the backend.

Tested: the engine (`tests/unit/test_engine_prefill_batch.cpp`: requests admitted in one step go
in one call; a failing request in the batch fails alone), the backend's packing on a stand-in
engine (what is packed and what goes alone, errors, prefix hits packed from their cached offset),
and on the GPU (`CudaLlmEngineBackendTest`): the 8 Hugging Face reference prompts prefilled all
together, in reverse order and in pairs, 24 sequences, each equal to Hugging Face token for token;
and a prompt with a 48-token cached prefix packed next to a fresh prompt (equal to the same prompt
without caching, and to Hugging Face). The release check passed with batched prefill serving the
live server (concurrent streams equal to the same requests alone).

Measured with the step profile (Day 46's scenarios, RTX 4050 Laptop GPU; full report
[benchmarks/results/day51-step-profile.md](../benchmarks/results/day51-step-profile.md)):

| Scenario | prefill before (Day 46) | with batched prefill | per request (P50) |
| --- | --- | --- | --- |
| `decode`: 32 x 64-token prompts at once | 3,314 tokens/s | **6,165 tokens/s (1.86x)** | 19.0 -> 9.8 ms |
| `prefill`: 32 x 480-token prompts at once | 3,569 tokens/s | 3,585 tokens/s | 135 -> 134 ms |
| `mixed`: 2 requests/s | 3,461 tokens/s | 3,393 tokens/s | 25.3 -> 25.3 ms |

- **Short prompts are bound by reading the weights, like decode.** A 64-token prompt alone takes
  19 ms, most of it reading the 2.5 GB of weights once. Two or more prompts in one pass share that
  read, so prefill throughput nearly doubled and its share of the `decode` scenario fell from 3.5%
  to 1.9%.
- **Long prompts gain nothing yet,** as expected: two 480-token prompts do not fit one 512-token
  step, so each is still prefilled alone, and at 480 tokens the matmuls are compute-bound anyway.
- **Paced arrivals rarely share a step.** At 2 requests/s with 16 ms steps, requests seldom arrive
  together; batching helps bursts. Chunked prefill (several requests' chunks per step) and mixed
  prefill/decode steps are what move `mixed`.

## Chunked prefill (Day 52)

With `prefill_chunk_tokens` set (engine option and server `engine.prefill_chunk_tokens`; 0, the
default, is off), a prompt is prefilled at most that many tokens per step, and the steps in between
carry the other requests' decodes. A long prompt no longer stalls every running request for the
whole of its prefill, and a prompt may now be longer than the step budget (`max_batch_tokens`):
it takes several steps.

- **The engine** plans the step's chunks first (`EngineLoop::plan_chunks_locked`), in admission
  order: each prefilling request gets `min(remaining, prefill_chunk_tokens, budget left)` tokens,
  at least one, so a request always moves. Admission counts a queued request's first chunk, not
  its whole prompt, against the budget and against on-demand KV pages. The step's continuing
  chunks and its newly admitted first chunks go in **one** `prefill_batch` call; a `PrefillItem`'s
  prompt is then the input up to the chunk's end, and its sequence already holds the earlier
  chunks. A request joins the decode batch the step after its last chunk; decodes run only for
  requests that are not prefilling. Cancelling a request mid-prefill frees its pages at once.
- **The backend contract:** `supports_chunked_prefill()` (false by default: the engine then
  prefills each prompt whole, as before) and `prefill_more(seq, input)`, the next part of a partly
  prefilled sequence, which returns the first generated token when `input` is the whole prompt.
  The default `prefill_batch` calls `prefill_more` for items whose sequence is not empty.
- **The CUDA backend** checks that the input extends what the sequence holds (with prefix caching,
  token for token), gives the new tokens their pages, and prefills them from the sequence's
  length as a cached offset: the same path a prefix-cache hit takes (`model_prefill_from`, and in a
  pack `model_prefill_batch`, which now accepts `seq.len == start`). Continuation chunks and new
  prompts are packed together like Day 51's items.
- **Limitation:** with prefix caching on, a prompt's cached prefix is found only within its first
  chunk, since that is when its pages are allocated. Choose a chunk at least as long as the shared
  prefix (or leave chunking off) when prefixes matter more than the step tail.

Tested: the engine on the fake backend (`tests/unit/test_engine_chunked_prefill.cpp`: a long
prompt prefilled in chunks while another request decodes; the same outputs for chunk sizes 64,
128, 256 and 512 as unchunked; the step budget bounds chunks and decodes together; a prompt longer
than a step accepted only with chunking; a backend that cannot chunk; cancelling mid-prefill);
the backend's checks and packing on the stand-in engine; on the GPU, the instruct references (a
512-token system prompt and a user turn) prefilled in chunks of 64, 128, 256 and 512 with caching
off and on, and in packed rounds of 128, all equal to Hugging Face token for token; and through the
Python API, concurrent requests with a step budget shorter than the longest prompt
(`tests/python/test_chunked_prefill.py`). 303/303 tests pass.

Measured with the step profile (RTX 4050 Laptop GPU; full report
[benchmarks/results/day52-chunked-prefill.md](../benchmarks/results/day52-chunked-prefill.md)):

| Scenario | | chunking off (budget 512) | chunks of 128 (budget 256) |
| --- | --- | --- | --- |
| `interference`: 48 requests at 3/s, 1 in 4 with a 480-token prompt | step P99 | 150.5 ms | **69.1 ms (-54%)** |
| | step P50 | 16.3 ms | 16.4 ms |
| | TTFT P50 / P99 | 589 / 1,505 ms | 800 / 2,019 ms (+36% / +34%) |
| | time between tokens P50 / P99 (per-request mean) | 19.4 / 20.6 ms | 19.8 / 21.5 ms |
| `prefill`: 32 x 480-token prompts at once, 8 output tokens | step P99 | 165.5 ms | **72.6 ms (-56%)** |
| | elapsed | 5.1 s | 5.7 s (+12%) |
| | time between tokens P50 | 156.7 ms | **60.0 ms (-62%)** |
| | TTFT P50 / P99 | 2,410 / 4,866 ms | 2,664 / 5,460 ms (+11% / +12%) |

- **The worst step halves.** Without chunking, a step that prefills a 480-token prompt takes about
  150 ms, and every running request waits that long for its next token. With 128-token chunks and
  a 256-token budget no step holds a whole long prefill: the step P99 falls from 150 to 69 ms, and
  in the prefill burst the time between tokens from 157 to 60 ms.
- **Time to first token pays for it.** A long prompt now takes four steps instead of one and shares
  them with decodes, so TTFT rises by about a third under `interference` and the burst finishes
  12% later (fixed per-pass costs, such as reading the weights, are paid once per chunk).
- **The mean time between tokens hides the stalls:** it is a per-request average, and a 150 ms
  stall every few hundred tokens barely moves it; the step P99 is what a streaming client feels as
  a hitch.
- So chunking stays **off by default**: it is a knob for latency-sensitive streaming next to long
  prompts, traded against TTFT and throughput. Smaller chunks bound the step tail further at more
  cost; mixed steps that decode inside the prefill's pass (one forward pass for both) are what
  would cut that cost.

## Mixed steps (Day 53)

A step that prefills (new prompts, or the next chunks of long ones) while other requests decode
now runs as **one forward pass**: the weights are read once for the prefill tokens and the decode
tokens together, instead of once by `prefill_batch` and again by `decode_step`.

- **The engine** plans the step's prefill items and its decode batch, then makes one
  `InferenceBackend::mixed_step(items, decode)` call (`EngineLoopOptions::mixed_steps`, on by
  default; server `engine.mixed_steps`; `Engine(mixed_steps=...)`). Each prefill item and the
  decode still succeed or fail on their own: a failed decode fails only the decoding requests, as
  before. The default `mixed_step` makes the two calls, so any backend works unchanged; one whose
  `supports_mixed_steps()` is true runs them together. `stats()["mixed_steps"]` counts them.
- **The CUDA backend** checks the decode first (changing nothing), admits the prefill items and
  gives the decode its pages, then packs the items that fit with the decode rows after them
  (`backend_detail::model_mixed_batch`; at most 512 rows and `max_batch` sequences in all). The
  embedding, every projection, RoPE and the MLP run once on all rows; each row's K/V go to its own
  pages; **attention is dispatched per segment**: the prefill segments through the varlen tiled
  kernel (64 x 32), the decode rows through a row-per-sequence kernel (one block per head and row,
  the decode attention's math); the head runs once on every sequence's last row. Items that do not
  fit the pass are prefilled alone before it. If the prefills take the pages the decode needed, the
  decode fails with `KvCacheExhausted` (as a decode after them would have) and the prefills stand.
  `mixed_passes()` counts the steps run as one pass.

Tested: the engine on the fake backend (`tests/unit/test_engine_mixed_steps.cpp`: a long prompt's
chunks and a running request's decode in one call per step; the same outputs, prefill sizes and
tokens as with two calls; a failed decode fails only the decode batch; 24 requests of mixed
lengths match the two-call run); the backend's packing on the stand-in engine (prefills and
decodes in one pass, equal to prefill then decode, a failed decode check still prefills, a
sequence in both is refused, pages taken by the prefills fail only the decode, what does not fit
is prefilled first); and on the GPU: the 8 Hugging Face references, four decoding while the other
four are prefilled 4 tokens per step, both ways round, with caching off and on (32 sequences), and
the instruct references (512-token system prompt), half decoding while half are prefilled 128
tokens per step, all equal to Hugging Face token for token. Every existing suite, the Python API
tests and the release check now run with mixed steps on. 309/309 tests pass.

Measured with the step profile (RTX 4050 Laptop GPU, chunking off, step budget 512; full report
[benchmarks/results/day53-mixed-steps.md](../benchmarks/results/day53-mixed-steps.md)):

| Scenario | | two passes | mixed steps |
| --- | --- | --- | --- |
| `interference`: 48 requests at 3/s, 1 in 4 with a 480-token prompt | TTFT P50 / P99 | 560 / 1,507 ms | **73 / 834 ms** |
| | step P99 | 150.0 ms | **57.2 ms** |
| | time between tokens P50 / P99 | 19.1 / 20.4 ms | 17.4 / 17.8 ms |
| `mixed`: the mixed profile at 2 requests/s | TTFT P50 / P99 | 71.6 / 1,015 ms | **54.3 / 824 ms** |
| | step P99 | 42.1 ms | 29.1 ms |
| | time between tokens P50 / P99 | 17.6 / 22.7 ms | 16.7 / 18.2 ms |

- **Better on every latency figure,** and more than reading the weights once can explain. Most of
  the gain on long prompts is the attention kernel: a prompt prefilled alone (a pack of one, Day
  51) still took the engine's own prefill path with its per-query attention, about 150 ms at 480
  tokens; in a mixed step the same prompt goes through the varlen tiled kernel (Day 49). The
  weights read once for prefill and decode is the rest, the part that moves `mixed`, whose prompts
  are short.
- **Unlike chunking, nothing is traded:** TTFT falls instead of rising, and the step tail falls
  almost as far (150 -> 57 ms here, 69 ms with 128-token chunks on Day 52).
- **Follow-up:** a prompt prefilled alone with no decode running still takes the old path; sending
  a pack of one through `model_mixed_batch` too would give it the tiled kernel.

## The model it serves

Llama 3.2 1B, FP16:

| | |
| --- | --- |
| layers | 16 |
| query heads | 32 |
| KV heads | 8 (grouped-query attention: 4 query heads share each KV head) |
| head dimension `d` | 64 |
| KV row (one token, all KV heads) | 8 x 64 = 512 halves = 1 KB, for K and for V |
| KV page | 16 tokens (Day 42) |
| scale | `1 / sqrt(64)` = 0.125 |

The kernel takes these as runtime arguments (`n_heads`, `n_kv_heads`) and `d` as a template
parameter (64 instantiated; 128 is the next model size's).

## Interface

```cpp
// Causal attention for B sequences whose new tokens are packed one after another.
// Sequence b has start[b] cached tokens (already in the KV pages) and
// T[b] = cu_seqlens_q[b + 1] - cu_seqlens_q[b] new tokens, whose K and V the caller has already
// written to its pages (as kv_scatter does today). Its new token i is at position start[b] + i and
// attends to keys 0 .. start[b] + i.
void attention_prefill_varlen(
    const __half* q,               // [total_q, n_heads, d], token-major: the layout the QKV matmul produces
    const __half* k_pool,          // [num_pages * page_size, n_kv_heads, d]
    const __half* v_pool,          // same
    const int32_t* cu_seqlens_q,   // [B + 1], cu_seqlens_q[0] = 0, cu_seqlens_q[B] = total_q
    const int32_t* start,          // [B], cached tokens per sequence (0 for a fresh prompt)
    const int32_t* block_tables,   // [B, max_pages], page ids per sequence, row-major
    int32_t max_pages,             // block_tables' row length
    const int32_t* tile_map,       // [num_q_tiles, 2]: (b, first token of the tile within b); see Scheduling
    int32_t num_q_tiles,
    __half* out,                   // [total_q, n_heads, d], same layout as q
    int n_heads, int n_kv_heads, int page_size, float scale, cudaStream_t stream);
```

- **Packed, token-major `q` and `out`** (`cu_seqlens_q`, the FlashAttention varlen convention):
  the projections and the MLP already work on rows of tokens, so packing B prompts is "stack their
  rows", with no padding and no per-sequence launch.
- **K and V stay in the pages.** Cached and new keys are read the same way, through the block
  table: there is no contiguous copy of the context. A sequence with `start[b] = 0` is a fresh
  prompt; `start[b] > 0` is a prefix-cache hit or the next chunk of a chunked prefill. One kernel
  serves both, which is why the interface takes `start`.
- **`tile_map` is built on the host** with the other per-step metadata (one small copy per step,
  not per layer). Blocks then find their work with one load, not a search over `cu_seqlens_q`.
- **Every integer is 32-bit.** `total_q` and page ids stay far below 2^31 at this scale; offsets
  into the pools are computed in 64-bit (`size_t`).

## The causal mask

Query `i` of sequence `b` is at position `p = start[b] + i` and sees keys `0 .. p`. For a tile of
queries at positions `[p0, p0 + Br)` and a tile of keys `[k0, k0 + Bc)`:

- `k0 + Bc - 1 <= p0`: every key is visible to every query. **No mask.** This is almost every tile
  of a long context.
- `k0 > p0 + Br - 1`: no key is visible. **Not loaded at all**: the key loop for a query tile ends
  at `p0 + Br - 1`.
- otherwise, the diagonal: mask `key > query` to `-inf` before the softmax.

The loop over key tiles therefore runs over `ceil((p0 + Br) / Bc)` tiles, and only the last one
or two need the mask. Every query sees at least itself, so no row is ever fully masked and the
softmax never divides by zero.

## Grouped-query attention

Query head `h` reads KV head `g = h / (n_heads / n_kv_heads)` (4 query heads per KV head here).

- **v1 (Day 49): one query head per thread block, GQA by indexing.** The 4 blocks of a group load
  the same K/V tiles; they are scheduled next to each other (head is the fastest-varying grid
  index within a group), so the second to fourth loads come from L2, not DRAM.
- **Considered, later:** packing the 4 query heads of a group into one block's rows (a tile of
  `4 x Br/4` rows), so each K/V tile is read from L2 once per group. It needs a wider tile for the
  same number of tokens and a different masking pattern per row group. Worth it only if Day 49's
  measurement shows the kernel bound by L2 traffic.

## Tiling and the online softmax

FlashAttention-2 (Dao, 2023, arXiv:2307.08691): one thread block owns a tile of `Br` queries of
one head, keeps it in shared memory, and streams the key tiles past it. For each key tile `j`,
per query row (all in FP32):

```
S     = (Q K_j^T) * scale            (masked to -inf on the diagonal tile)
m_new = max(m, rowmax(S))
P     = exp(S - m_new)
l     = l * exp(m - m_new) + rowsum(P)
O     = O * exp(m - m_new) + P V_j
m     = m_new
```

and at the end `out = O / l`. `exp` is `exp2f` on scores pre-multiplied by `log2(e)`, folded into
`scale`. `m` starts at `-inf`, `l` and `O` at 0.

- **Work split:** 4 warps per block, each warp owns 16 query rows of the `Br = 64` tile and does
  its own rows' softmax: no shared-memory reduction across warps, no `__syncthreads` inside the
  softmax (the FlashAttention-2 split, as opposed to splitting the keys across warps).
- **Matmuls on tensor cores:** `Q K^T` and `P V` with warp-level `mma.sync` (m16n8k16, FP16 in,
  FP32 accumulate), available on sm_80 and later. `P` is converted to FP16 for `P V`, as in
  FlashAttention.
- **Loads:** 128-bit (`uint4`, 8 halves). A K or V row of one head is 64 halves = 128 bytes =
  8 loads; a warp loads 4 rows per instruction, fully coalesced within each row. Rows of a tile
  come from up to `Bc / page_size + 1` pages (4 or 5 for `Bc = 64`), each row's address through
  the block table.
- **Double-buffered K/V:** tile `j + 1` is loaded (`cp.async`, sm_80+) while tile `j` is
  computed.

## Shared memory and registers

RTX 4050 Laptop GPU: AD107, compute capability 8.9, 20 SMs. Per SM: 100 KB shared memory, 64 K
32-bit registers, at most 1,536 threads and 24 blocks. A block can use up to 99 KB of shared
memory, above 48 KB only by opting in (`cudaFuncSetAttribute`). The kernel checks these with
`cudaGetDeviceProperties` at startup rather than assuming them.

Starting point, `Br = 64`, `Bc = 64`, `d = 64`, 128 threads:

| Buffer | Size |
| --- | --- |
| Q tile, 64 x 64 FP16 | 8 KB |
| K tile, double-buffered | 2 x 8 KB |
| V tile, double-buffered | 2 x 8 KB |
| **per block** | **40 KB** |

- **2 blocks per SM** (80 of 100 KB): 8 warps per SM. Scores `S` and `P` never touch shared
  memory: they live in the `mma` accumulator registers of the warp that owns the rows.
- **Registers, estimated per thread:** O accumulator (16 x 64 FP32 over 32 lanes) 32, S tile 32,
  P as FP16 16, m and l 4, addresses and loop state about 30: roughly 120. 128 threads x 2 blocks
  x 128 registers = 32 K of the SM's 64 K. A third block would need 120 KB of shared memory, so
  shared memory, not registers, sets occupancy.
- **Shared memory is independent of the context length.** Today's kernel needs `S x 4` bytes;
  this one needs 40 KB at any length.
- **To sweep on Day 49:** `Br` 64 or 128 (128: 8 warps, a 16 KB Q tile, 48 KB per block, still
  2 blocks per SM), `Bc` 32 or 64, single or double buffering. Chosen by measurement.

## Scheduling the work

- **Grid:** `num_q_tiles x n_heads` blocks, with `num_q_tiles = sum over b of ceil(T[b] / Br)`.
  The head index varies fastest so a group's 4 query heads run together (GQA through L2).
- **Longest work first.** A tile's cost grows with its last position (the number of key tiles it
  reads). The host orders `tile_map` by descending end position, so the long tiles start first
  and the short ones fill the gaps at the end (the "tail" of a launch).
- **Sizes at this scale:** one 480-token prompt is 8 query tiles x 32 heads = 256 blocks, about 6
  waves of 40 (2 per SM x 20 SMs). Four 128-token prompts are the same count. A single short
  prompt (T < 64) is 32 blocks: fewer than the 40 the GPU can run at once, which is fine for a
  launch that short.

## Numerics and determinism

- **FP16 inputs, FP32 accumulation and softmax, FP16 output**, like today's kernel.
- **Deterministic:** every block computes its rows with a fixed order of key tiles (ascending),
  no atomics and no split of the keys across blocks, so the same inputs give the same bits.
- **Independent of the batch.** A sequence's output does not depend on what else is packed with
  it, or where: its tiles start at its own first token, the key loop depends only on its own
  positions, and nothing is reduced across sequences. Packed with other prompts or alone, a
  prompt's attention output is bit-identical. This is a stated goal (ADR-0006: today's outputs are
  deterministic but not batch-invariant), tested directly.
- **Not bit-identical to today's kernel.** The order of the softmax sums and the FP16 rounding of
  `P` differ. Both are checked against an FP32 reference within a tolerance; at a near-tie a
  greedy token can change, as with any change of kernel.

## Tests

Kernel level, against a reference that computes each sequence alone in FP32 on the host (Day 48
is "matches per-sequence naive attention for mixed lengths 1–2,048"):

- **Lengths:** packs of sequences with `T` in {1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 129, 480,
  1000, 2047, 2048} and random lengths, seeded; packs of 1 to 16 sequences.
- **Cached prefixes:** `start` of 0, of whole pages and of partial pages (17, 100), up to a total
  of 2,048 keys.
- **Paging:** shuffled page ids; unused pool rows filled with NaN, so a read outside a sequence's
  pages poisons its output and fails the test.
- **GQA:** 32 query heads over 8 KV heads, and 1:1 (`n_kv_heads = n_heads`).
- **Tolerance:** `|out - ref| <= 2e-3 + 1e-2 x |ref|` per element (FP16 output, FP32 reference);
  the report gives the largest error found.
- **Batch independence:** every sequence's output equal, bit for bit, whether it runs alone or
  packed with others, in any position of the pack.
- **Determinism:** the same pack twice gives the same bits.
- **Compute Sanitizer:** memcheck and racecheck on the kernel tests (`scripts/compute_sanitizer.sh`).

Engine level, once it serves prefill: the existing GPU integration tests (outputs against
Hugging Face, prefix hits, copy-on-write, resumption) unchanged, and the step profile's
`prefill` scenario for the gain.

## Plan

| Day | Step | Done when |
| --- | --- | --- |
| 48 | Measure today's attention share of prefill (CUDA events); v0 varlen kernel, simple: one block per (query, head), the interface above, the tests above | matches the per-sequence reference at every length 1–2,048 |
| 49 | Tiled kernel: Q tile in shared memory, streamed K/V tiles, online softmax, `mma.sync`, 128-bit loads; the tile sweep | matches v0 within the tolerance at every tested length; faster than v0 and than today's kernel, measured |
| 51 | The engine prefills all admitted prompts in one batched pass using it | step profile and outputs |
| 52 | Chunked prefill: long prompts split by the step budget, resumed from the cached offset each step, packed with other prefills and sharing steps with decodes | outputs equal for chunk sizes 64, 128, 256, 512 |
| 53 | Mixed steps: the step's prefill chunks and decode tokens in one forward pass, attention dispatched per segment | every correctness suite passes with mixed steps on |

## Out of scope

- **Decode.** Decode keeps its own kernel (one query per sequence); Day 46 showed decode is bound
  by reading the weights, not by attention.
- **Hopper features** (TMA, `wgmma`, FlashAttention-3): not on sm_89.
- **Splitting keys across blocks** (flash-decoding style): for long contexts with few queries;
  not needed below 2,048 keys, and it would break the fixed reduction order.
- **FP8 and quantized KV.**

## Open questions

- How much of the 135 ms is attention today? Day 48's measurement decides whether the tiled
  kernel (Day 49) or the batched pass over the weights matters more for the step's length.
- Does GQA through L2 suffice, or is the kernel L2-bound (the packed-heads variant)?
- Does the tolerance hold for the longest contexts (2,048 keys), where FP16 `P` loses the most?
