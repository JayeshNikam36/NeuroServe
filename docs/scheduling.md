# Scheduling

Which queued requests the engine admits next is decided by a **scheduling policy**
(`runtime/core/include/neuroserve/core/scheduler.hpp`), chosen with `engine.scheduling_policy`.

## Policy and mechanism

Once per engine step, the loop builds a `SchedulerInput` snapshot (queued requests in arrival
order, the running batch, `max_batch`, free KV pages) and asks the policy for a
`ScheduleDecision`: the queue indices to admit, in order. The policy is a pure function: it never
touches the backend, the KV cache or a request. The loop checks every decision with
`check_decision` (indices valid and unique, `max_batch` and free pages respected) before applying
it; a decision that breaks a limit stops the engine with an error rather than corrupting state.

## Policies

| Name | Admits | Trade-off |
| --- | --- | --- |
| `fcfs` | In arrival order while requests fit; stops at the first that does not | No starvation; head-of-line blocking leaves capacity idle |

## Golden traces

`tests/data/scheduler_traces/*.trace` each describe a workload (limits, arrivals) and the
step-by-step log the policy must produce. `tests/support/scheduler_sim.hpp` replays the workload
with the engine loop's rules, without threads or a GPU, and the `GoldenTrace.*` tests compare the
log line by line.

After an intended policy change, regenerate the expected logs and review the diff in the PR:

```bash
bash scripts/dev.sh bash -c "NS_UPDATE_GOLDEN=1 ./build/ci-cpu/tests/ns_unit_tests --gtest_filter='GoldenTrace.*'"
git diff tests/data/scheduler_traces
```
