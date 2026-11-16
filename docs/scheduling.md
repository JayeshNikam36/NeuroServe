# Scheduling

Once per engine step a **scheduling policy**
(`runtime/core/include/neuroserve/core/scheduler.hpp`, chosen with `engine.scheduling_policy`)
decides which running requests to preempt and which queued requests to admit.

## KV pages on demand

A request holds KV pages only for the tokens it has cached (`ceil(cached / 16)`) and takes one
more page each time it crosses a page boundary. Nothing is reserved up front, so more requests fit
at once; in exchange the running batch can outgrow the cache, and the policy must preempt.

## Policy and mechanism

The loop builds a `SchedulerInput` snapshot: queued requests in arrival order with the pages their
first step needs (`ceil((prompt + 1) / 16)`), running requests in admission order with the pages
they hold and the new pages their next decode needs (0 or 1), `max_batch`, and free pages. The
policy returns a `ScheduleDecision`: running indices to preempt, queued indices to admit. The
policy is a pure function; the loop checks every decision with `check_decision` (indices valid
and unique, `max_batch` respected, the step's pages fit after preemption) before applying it, and
a decision that breaks a limit stops the engine with an error.

A preempted request frees its sequence and pages, goes back to the front of the queue in arrival
order, and later restarts from its prompt. Greedy decoding is deterministic and batch-invariant on
this engine, so it regenerates exactly the same tokens (tested on the GPU against Hugging Face).

## Policies

| Name | Preempts | Admits | Trade-off |
| --- | --- | --- | --- |
| `fcfs` | Most recently admitted first, never the oldest, only when the batch cannot grow | In arrival order while requests fit; stops at the first that does not | No starvation or livelock; head-of-line blocking leaves capacity idle |

## Golden traces

`tests/data/scheduler_traces/*.trace` each describe a workload (limits, arrivals) and the
step-by-step log the policy must produce. `tests/support/scheduler_sim.hpp` replays the workload
with the engine loop's rules, without threads or a GPU, and the `GoldenTrace.*` tests compare the
log line by line.

After an intended change to a policy or to the loop's rules, regenerate the expected logs and
review the diff in the PR:

```bash
bash scripts/dev.sh bash -c "NS_UPDATE_GOLDEN=1 ./build/ci-cpu/tests/ns_unit_tests --gtest_filter='GoldenTrace.*'"
git diff tests/data/scheduler_traces
```
