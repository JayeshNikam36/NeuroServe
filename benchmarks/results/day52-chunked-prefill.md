# Day 52: chunked prefill, measured

- **GPU:** RTX 4050 Laptop GPU (sm_89) · **model:** Llama 3.2 1B FP16 · config `configs/dev.yaml`,
  prefix caching off.
- **Runs:** `neuroserve.bench.step_profile --scenarios interference,prefill`, once with chunking off
  (step budget 512) and once with `--prefill-chunk-tokens 128 --max-batch-tokens 256`. Raw output in
  `benchmarks/results/raw/day52-off/` and `raw/day52-chunk128/` (not committed).

## Where the time went

| Scenario | run | busy / elapsed | prefill | decode | host (loop) | invariant checks | steps | step P50 (ms) | step P99 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `interference` | off | 95% of 17.5 s | 14.3% | 83.5% | 1.6% | 0.6% | 869 | 16.33 | 150.52 |
| `interference` | chunks of 128 | 96% of 17.9 s | 16.2% | 81.2% | 2.0% | 0.6% | 866 | 16.38 | 69.08 |
| `prefill` | off | 98% of 5.1 s | 85.3% | 12.4% | 2.3% | 0.1% | 39 | 153.36 | 165.47 |
| `prefill` | chunks of 128 | 99% of 5.7 s | 69.5% | 28.5% | 1.7% | 0.2% | 103 | 58.26 | 72.61 |

## Latency

| Scenario | run | TTFT P50 (ms) | TTFT P99 (ms) | time between tokens P50 (ms) | P99 (ms) |
| --- | --- | --- | --- | --- | --- |
| `interference` | off | 589.33 | 1504.53 | 19.38 | 20.56 |
| `interference` | chunks of 128 | 799.56 | 2019.03 | 19.81 | 21.50 |
| `prefill` | off | 2410.45 | 4866.39 | 156.71 | 158.80 |
| `prefill` | chunks of 128 | 2664.34 | 5460.43 | 60.04 | 61.80 |

TTFT is client-side; the time between tokens is each request's mean, from the server, its P50 and
P99 over requests.

## Reading

The step tail halves (150 -> 69 ms, 165 -> 73 ms): no step holds a whole 480-token prefill, so the
running requests' tokens are no longer held back by one. Time to first token rises by about a
third under `interference`, and the burst takes 12% longer: each long prompt is prefilled over four
steps, and every chunk pays the pass's fixed costs. Chunking stays off by default; see
docs/prefill-attention.md, "Chunked prefill (Day 52)".
