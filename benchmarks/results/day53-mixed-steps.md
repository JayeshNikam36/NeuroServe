# Day 53: mixed steps, measured

- **GPU:** RTX 4050 Laptop GPU (sm_89) · **model:** Llama 3.2 1B FP16 · config `configs/dev.yaml`,
  prefix caching off, chunked prefill off, step budget 512.
- **Runs:** `neuroserve.bench.step_profile --scenarios interference,mixed`, once with
  `--no-mixed-steps` (prefill and decode in two passes) and once with mixed steps on (the
  default). Raw output in `benchmarks/results/raw/day53-two-pass/` and `raw/day53-mixed/` (not
  committed). A mixed step's one backend call is counted as prefill (docs/profiling.md).

## Where the time went

| Scenario | run | busy / elapsed | prefill | decode | host (loop) | invariant checks | steps | step P50 (ms) | step P99 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `interference` | two passes | 96% of 17.5 s | 14.3% | 83.2% | 1.9% | 0.6% | 872 | 16.25 | 150.04 |
| `interference` | mixed | 95% of 17.0 s | 10.1% | 87.5% | 1.8% | 0.7% | 935 | 16.26 | 57.17 |
| `mixed` | two passes | 95% of 65.9 s | 8.2% | 89.8% | 1.3% | 0.7% | 3548 | 16.05 | 42.08 |
| `mixed` | mixed | 95% of 65.9 s | 6.3% | 91.7% | 1.3% | 0.7% | 3750 | 15.98 | 29.07 |

## Latency

| Scenario | run | TTFT P50 (ms) | TTFT P99 (ms) | time between tokens P50 (ms) | P99 (ms) |
| --- | --- | --- | --- | --- | --- |
| `interference` | two passes | 560.20 | 1507.46 | 19.10 | 20.37 |
| `interference` | mixed | 73.12 | 834.00 | 17.36 | 17.80 |
| `mixed` | two passes | 71.64 | 1014.50 | 17.56 | 22.70 |
| `mixed` | mixed | 54.27 | 823.75 | 16.68 | 18.19 |

TTFT is client-side; the time between tokens is each request's mean, from the server, its P50 and
P99 over requests.

## Reading

Mixed steps improve every latency figure. Next to 480-token prompts, TTFT P50 falls from 560 to
73 ms and the step P99 from 150 to 57 ms. Most of that is the attention kernel: a prompt
prefilled alone used the engine's own prefill path (per-query attention), while in a mixed step
it goes through the varlen tiled kernel. Reading the weights once for prefill and decode is the
rest, which is what moves the `mixed` profile (short prompts): TTFT P50 72 -> 54 ms, time between
tokens P99 22.7 -> 18.2 ms. One run each; Day 54 repeats these order-balanced.
