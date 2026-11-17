# ADR-0006: Deterministic, not batch-invariant: what NeuroServe guarantees about outputs

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

The docs said that greedy decoding on this engine is *batch-invariant*: a request produces the same
tokens whatever else is in its batch. The GPU tests seemed to support it: 100 concurrent requests,
and 24 with forced preemption under every policy, match Hugging Face token for token. But they all
use the same 8 reference prompts.

The v0.1 release gate (`scripts/release_smoke.py`) used 8 other chat prompts, and found a
counter-example (Day 25):

| Experiment | Result |
| --- | --- |
| The same request alone, twice | identical (8/8 prompts) |
| Streamed alone vs not streamed alone | identical tokens and text (8/8) |
| 2 or 4 requests at once | identical to alone |
| 8 or 24 at once | 7 of 8 prompts identical to alone; "Suggest a name for a friendly robot." takes another token at position 14 of 48 and continues differently |
| 8 identical copies of each prompt at once | the copies always agree with each other; for the robot prompt, all 8 differ from alone, in the same way |
| Preemptions during the experiments | 0 |

So the output is deterministic: the same work in the same batch gives the same tokens. But it is
not batch-invariant: the floating-point result of a decode step depends on the batch size, and
when two tokens are almost tied, the batch decides which one greedy decoding picks. The cause
inside the kernels is not isolated yet. The likely candidates are a GEMM kernel chosen by batch
size and a batch-dependent split in the attention kernel. Batched inference servers commonly
behave this way unless they use batch-invariant kernels.

## Options considered

1. **Batch-invariant kernels now:** a fixed reduction order and a fixed GEMM algorithm for every
   batch size. This is the only way to batch invariance, but it costs throughput, it belongs with
   the kernel work of Phase 4, and it would delay v0.1 for a property no client is promised.
2. **Run each request alone.** Invariant by construction, but it gives up batching, which is the
   point of the runtime.
3. **Guarantee what is true, test exactly that, and measure the rest.** Chosen.

## Decision

NeuroServe guarantees, and tests:

- **Determinism:** the same requests in the same batches give the same tokens. Identical requests
  sent at once agree.
- **Streaming changes nothing:** a streamed response has the same tokens and text as the same
  request without streaming.
- **Hugging Face parity for the reference workloads,** alone and batched, under every policy and
  with forced preemption.

NeuroServe does not guarantee that a request returns the same tokens in every batch. The release
gate reports how many concurrent responses match the same prompt sent alone, but it does not fail
on that number.

## Consequences

- **A policy can change outputs at a near-tie,** because it changes batch compositions. "Every
  policy produces the same tokens" holds for the reference workloads, not for every input.
- **Preemption by recompute (ADR-0004) restarts from the prompt in a different batch,** so a
  restarted request can, at a near-tie, generate a different token than it already streamed. The
  engine detects this and logs an error ("restarted request generated a different token than it
  streamed"), but the client keeps the earlier tokens. It has not been seen yet: 0 preemptions in
  the Day 24 benchmark, and the forced-preemption GPU test passes. The fix is Day 38's recompute
  from the prompt plus the tokens already generated, so what the client has is kept as input,
  never regenerated.
- **Finding the cause is part of the Phase 4 profiling** (Day 46). Batch-invariant kernels would be
  an A/B-measured option there, with their throughput cost.
