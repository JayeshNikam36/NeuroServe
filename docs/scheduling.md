# Scheduling

Once per engine step a **scheduling policy**
(`runtime/core/include/neuroserve/core/scheduler.hpp`, chosen with `engine.scheduling_policy`)
decides which running requests to preempt and which queued requests to admit.

## KV pages on demand

A request holds KV pages only for the tokens it has cached (`ceil(cached / 16)`) and takes one
more page each time it crosses a page boundary. Nothing is reserved up front, so more requests fit
at once; in exchange the running batch can outgrow the cache, and the policy must preempt.

## Steps and the token budget

One engine step prefills the requests admitted in it and decodes one token for every request that
was already running; a newly admitted request decodes from the next step on. The step's tokens
(admitted prompt tokens + one per running request) must fit `engine.max_batch_tokens` (default
512). Decode tokens are counted first, so running requests always make progress and one long
prompt cannot stall the batch for more than the budget allows. A prompt longer than the budget is
rejected at submit (the server config requires `max_prompt_tokens <= max_batch_tokens`). Splitting
long prompts across steps (chunked prefill) is a later step.

## Policy and mechanism

The loop builds a `SchedulerInput` snapshot: queued requests in arrival order with the pages their
prefill needs (`ceil(prompt / 16)`), running requests in admission order with the pages they hold
and the new pages their next decode needs (0 or 1), `max_batch`, free pages and the token budget. The
policy returns a `ScheduleDecision`: running indices to preempt, queued indices to admit. The
policy is a pure function; the loop checks every decision with `check_decision` (indices valid
and unique, `max_batch` respected, the step's pages and tokens fit after preemption) before
applying it, and
a decision that breaks a limit stops the engine with an error.

A preempted request frees its sequence and pages, goes back to the front of the queue in arrival
order, and later restarts from its prompt. Greedy decoding is deterministic and batch-invariant on
this engine, so it regenerates exactly the same tokens (tested on the GPU against Hugging Face).

## Policies

| Name | Preempts | Admits | Trade-off |
| --- | --- | --- | --- |
| `fcfs` | Most recently admitted first, never the oldest, only when the batch cannot grow | In arrival order while slots, pages and the token budget last; stops at the first that does not fit | No starvation or livelock; head-of-line blocking leaves capacity idle |
| `token-budget` | As `fcfs` | In arrival order, skipping requests that do not fit so smaller ones can use the room; a request that has waited `max_wait_steps` (16) steps can no longer be overtaken | Higher utilisation and less queueing for short prompts; long prompts wait longer, bounded by the aging rule |
| `decode-first` | As `fcfs` | In strict arrival order, but while requests are decoding a step's prefills are capped at a quarter of the token budget | Smoother time between tokens for running requests; new requests start later |

Every policy must produce exactly the same tokens for every request; only the order and timing of
the work differ. The engine-loop correctness tests run once per policy, and so does the GPU
preemption test (token-exact against Hugging Face).

The policy can be switched on a running engine (`Engine.set_scheduling_policy(name)` in Python,
`EngineLoop::set_policy` in C++); the switch takes effect at the next step, which allows A/B
comparisons on the same loaded model.

## Golden traces

`tests/data/scheduler_traces/*.trace` each describe a workload (policy, limits, arrivals) and the
step-by-step log the policy must produce, including each step's tokens as `prefill+decode`. `tests/support/scheduler_sim.hpp` replays the workload
with the engine loop's rules, without threads or a GPU, and the `GoldenTrace.*` tests compare the
log line by line.

After an intended change to a policy or to the loop's rules, regenerate the expected logs and
review the diff in the PR:

```bash
bash scripts/dev.sh bash -c "NS_UPDATE_GOLDEN=1 ./build/ci-cpu/tests/ns_unit_tests --gtest_filter='GoldenTrace.*'"
git diff tests/data/scheduler_traces
```
