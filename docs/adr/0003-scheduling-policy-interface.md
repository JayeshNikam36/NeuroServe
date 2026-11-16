# ADR-0003: Scheduling as a pluggable, pure policy

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
Admission was hard-coded FCFS inside the engine loop. Phase 2 compares several policies
(FCFS, token-budgeted, decode-first, with preemption), so the decision must be swappable, and each
policy's behaviour must be testable precisely and without a GPU.

## Options considered
1. **Policy logic inside the engine loop, switched by flags.** Least code now; every new policy
   touches the loop, and decisions can only be tested through threads and timing.
2. **Policy as an object with access to the engine's state.** Flexible, but a policy could change
   state behind the loop's back, and tests need a running engine.
3. **Policy as a pure function of a snapshot.** `schedule(SchedulerInput) -> ScheduleDecision`;
   the loop owns all mechanism and checks every decision.

## Decision
Option 3. `SchedulingPolicy` with `FcfsPolicy` as the baseline, created by name from config.
Decisions are checked by `check_decision` before they are applied. Policies are tested with
golden traces replayed by a deterministic simulator of the engine loop's rules.

## Consequences
- New policies add a class and traces, not changes to the loop.
- The snapshot is rebuilt each step while requests are queued (O(queue) per step); fine at current
  queue sizes, revisit if profiling shows otherwise.
- The simulator must follow the engine loop's rules. When the loop changes (e.g. on-demand KV
  allocation, Day 14), the simulator and the traces change in the same PR.
