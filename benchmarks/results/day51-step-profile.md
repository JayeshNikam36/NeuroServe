# Engine step profile

- **When:** 2026-11-22T19:27:34+00:00 (5daa8ee-dirty)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU · **model:** llama-3.2-1b-instruct · **config:** `configs/dev.yaml`; prefix caching off; invariant checks on
- **Method:** the engine times every step's phases on its own thread (host clock). The backend's prefill and decode calls return tokens to the host, so their time includes the GPU work they launch and wait for; everything else in the step is host work. Idle = the engine thread waiting for requests.

## Where the time went

| Scenario | busy / elapsed | prefill | decode | host (loop) | invariant checks | steps | step P50 (ms) | step P99 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode` | 99% of 17.3 s | 1.9% | 95.8% | 1.6% | 0.6% | 1030 | 16.11 | 35.58 |
| `prefill` | 98% of 5.1 s | 84.6% | 12.3% | 3.0% | 0.1% | 39 | 152.18 | 163.25 |
| `mixed` | 95% of 65.9 s | 8.4% | 89.8% | 1.2% | 0.7% | 3552 | 16.00 | 42.23 |

Shares are of the busy time (the sum of every step). Host (loop) = housekeeping, scheduling, preemption and the rest of the step outside the backend calls and the invariant checks.

## `decode`: 32 requests at once, 64-token prompts, 256 output tokens each

The batch fills to max_batch and drains: decode time at every batch size.

Decode-only steps by batch size (ms):

| batch | steps | decode P50 | decode P95 | per sequence | host P50 | invariants P50 | step P50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 15.42 | 15.69 | 15.42 | 0.012 | 0.101 | 15.53 |
| 2 | 1 | 15.61 | 15.61 | 7.80 | 5.519 | 0.103 | 21.23 |
| 3 | 1 | 15.53 | 15.53 | 5.18 | 0.012 | 0.098 | 15.64 |
| 5 | 2 | 15.77 | 15.88 | 3.15 | 4.087 | 0.100 | 19.96 |
| 7 | 3 | 16.05 | 16.54 | 2.29 | 0.113 | 0.098 | 16.75 |
| 8 | 1005 | 15.87 | 16.52 | 1.98 | 0.110 | 0.102 | 16.10 |

Prefill: 2048 tokens in 16 steps; 0.1529 ms per token (P50 over steps), 6165 tokens/s overall; 9.79 ms per request (P50).

GPU: SM clock median 2655 MHz under load, max 60 °C, mean power 56.3 W.

Requests: 32 of 32 succeeded; trace: `trace-decode.json`, steps: `steps-decode.csv`.

## `prefill`: 32 requests at once, 480-token prompts, 8 output tokens each

Mostly prefill: its cost per token.

Decode-only steps by batch size (ms):

| batch | steps | decode P50 | decode P95 | per sequence | host P50 | invariants P50 | step P50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 15.61 | 15.61 | 15.61 | 2.019 | 0.106 | 17.74 |
| 2 | 1 | 15.57 | 15.57 | 7.79 | 12.327 | 0.102 | 28.00 |
| 3 | 1 | 15.71 | 15.71 | 5.24 | 2.912 | 0.098 | 18.72 |
| 4 | 1 | 16.30 | 16.30 | 4.08 | 5.845 | 0.103 | 22.25 |
| 5 | 1 | 16.84 | 16.84 | 3.37 | 15.218 | 0.097 | 32.15 |
| 6 | 1 | 16.48 | 16.48 | 2.75 | 6.035 | 0.103 | 22.62 |
| 7 | 1 | 16.34 | 16.34 | 2.33 | 6.697 | 0.103 | 23.14 |

Prefill: 15360 tokens in 32 steps; 0.2790 ms per token (P50 over steps), 3585 tokens/s overall; 133.91 ms per request (P50).

GPU: SM clock median 2655 MHz under load, max 64 °C, mean power 57.3 W.

Requests: 32 of 32 succeeded; trace: `trace-prefill.json`, steps: `steps-prefill.csv`.

## `mixed`: synthetic prompts of 32/128/480 tokens with 32/128/384 output tokens, 2 requests/s

The mixed profile at its own rate: a realistic step mix.

Decode-only steps by batch size (ms):

| batch | steps | decode P50 | decode P95 | per sequence | host P50 | invariants P50 | step P50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 585 | 15.08 | 15.62 | 15.08 | 0.026 | 0.109 | 15.22 |
| 2 | 400 | 15.54 | 16.14 | 7.77 | 0.039 | 0.112 | 15.69 |
| 3 | 209 | 15.70 | 16.30 | 5.23 | 0.052 | 0.112 | 15.86 |
| 4 | 304 | 15.71 | 16.34 | 3.93 | 0.064 | 0.111 | 15.90 |
| 5 | 349 | 15.80 | 16.42 | 3.16 | 0.075 | 0.110 | 16.01 |
| 6 | 589 | 15.93 | 16.52 | 2.65 | 0.089 | 0.109 | 16.15 |
| 7 | 465 | 15.96 | 16.66 | 2.28 | 0.101 | 0.107 | 16.23 |
| 8 | 531 | 16.03 | 16.67 | 2.00 | 0.113 | 0.107 | 16.30 |

Prefill: 17792 tokens in 120 steps; 0.2826 ms per token (P50 over steps), 3393 tokens/s overall; 25.30 ms per request (P50).

GPU: SM clock median 2640 MHz under load, max 68 °C, mean power 56.4 W.

Requests: 120 of 120 succeeded; trace: `trace-mixed.json`, steps: `steps-mixed.csv`.
