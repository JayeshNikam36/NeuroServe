# ADR-0005: Priority scheduling with aging from arrival

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
Some requests are more urgent than others (interactive vs batch). Strict priorities let a steady
stream of urgent requests starve low-priority ones forever. The scheduler needs priorities with a
provable bound on how long any request waits, and it must stay a pure, testable policy.

## Options considered
1. **Strict priority classes.** Simple; starvation is possible, which the plan rules out.
2. **Aging by engine steps** (like `token-budget`'s `max_wait_steps`). Deterministic, but a step
   can take 5 ms or 500 ms, so the bound would mean nothing to an API client.
3. **Aging by wall-clock time since arrival.** The engine puts each request's age in the policy's
   input; the policy computes `min(2, priority + age / aging_ms)`. The bound is in milliseconds.
4. **Weighted fair queueing.** Fairer shares per class, but more state and harder to explain.

## Decision
Option 3. Priorities 0–2 on `GenerationParams`, `PriorityPolicy(aging_ms)` from
`engine.priority_aging_ms`. Age is measured from arrival and not reset by preemption, so repeated
preemption cannot starve a request either. Admission is strict in effective-priority order (stop
at the first that does not fit), which is what makes the bound hold. Preemption picks the lowest
effective priority first and never the oldest running request.

## Consequences
- Bound: a request of priority `p` waits at most `(2 - p) × aging_ms` plus the time for the
  requests ahead of it, tested in the simulator against a stream of urgent requests.
- The policy is still a pure function: the engine supplies the age, the simulator supplies
  simulated time (10 ms per step), so the golden traces stay deterministic.
- A high-priority request does not preempt running work just to start sooner; it only goes first
  in the queue. Preempting for priority (not only for memory) is a possible later change.
- Strict ordering means head-of-line blocking within the priority order, as in FCFS.
