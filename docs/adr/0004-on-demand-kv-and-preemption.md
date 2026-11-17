# ADR-0004: On-demand KV pages with preemption by recompute

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
Admission reserved KV pages for prompt + `max_new_tokens` up front. That can never run out
mid-generation, but most reserved pages sit empty for most of a request's life, so far fewer
requests run at once than memory allows. Paged KV memory (the engine's block tables) makes
allocation per page possible.

## Options considered
1. **Keep up-front reservation.** Simple and safe; wastes the memory that batching needs.
2. **On-demand pages, preempt by swapping** KV pages to host memory and back. Keeps progress;
   needs pinned host buffers, copy streams and engine support the backend does not have.
3. **On-demand pages, preempt by recompute from the prompt.** Free the pages; the request restarts
   from its prompt later. Wastes the discarded work, needs nothing new from the backend.
4. **On-demand pages, recompute from prompt + generated tokens** (what vLLM does). Less wasted
   work, but the resumed tokens come from a prefill pass instead of decode steps, which can differ
   in the last floating-point bits and change a greedy choice.

## Decision
Option 3. The engine computes each step's page needs exactly (the same `ceil(len / page)` rule as
the backend), the policy decides preemption before any GPU call, and FCFS preempts the most
recently admitted requests first and never the oldest. Every request must fit in the cache alone
(validated at submit), so the oldest running request can always finish: no livelock.

## Consequences
- More concurrent requests per GB of KV memory; preemption is counted (`stats.preemptions`) and
  logged per request.
- Restarted requests regenerate identical tokens only if decoding is greedy and batch-invariant.
  Day 25 showed it is not batch-invariant (ADR-0006): at a near-tie a restart in another batch can
  diverge from what was already streamed (logged as an error). Recompute from the prompt plus the
  generated tokens (option 4) removes the problem; planned for Day 38, with its own parity tests.
- The work done before a preemption is lost; swap or prefix reuse (Phase 3) can recover it.
