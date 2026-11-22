# Engine step profile

- **When:** 2026-11-22T03:02:41+00:00 (580ec25-dirty)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU · **model:** llama-3.2-1b-instruct · **config:** `configs/dev.yaml`; prefix caching off; invariant checks on
- **Method:** the engine times every step's phases on its own thread (host clock). The backend's prefill and decode calls return tokens to the host, so their time includes the GPU work they launch and wait for; everything else in the step is host work. Idle = the engine thread waiting for requests.

## Where the time went

| Scenario | busy / elapsed | prefill | decode | host (loop) | invariant checks | steps | step P50 (ms) | step P99 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `decode` | 99% of 17.7 s | 3.5% | 94.6% | 1.3% | 0.6% | 1028 | 16.19 | 38.54 |
| `prefill` | 97% of 5.2 s | 85.5% | 12.3% | 2.1% | 0.1% | 39 | 153.14 | 161.87 |
| `mixed` | 95% of 66.1 s | 8.2% | 89.9% | 1.2% | 0.6% | 3535 | 16.10 | 42.54 |

Shares are of the busy time (the sum of every step). Host (loop) = housekeeping, scheduling, preemption and the rest of the step outside the backend calls and the invariant checks.

## `decode`: 32 requests at once, 64-token prompts, 256 output tokens each

The batch fills to max_batch and drains: decode time at every batch size.

Decode-only steps by batch size (ms):

| batch | steps | decode P50 | decode P95 | per sequence | host P50 | invariants P50 | step P50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 15.47 | 15.47 | 15.47 | 0.011 | 0.090 | 15.58 |
| 4 | 1 | 15.83 | 15.83 | 3.96 | 27.662 | 0.178 | 43.67 |
| 6 | 1 | 15.73 | 15.73 | 2.62 | 6.517 | 0.094 | 22.34 |
| 7 | 2 | 16.14 | 16.56 | 2.31 | 3.046 | 0.094 | 19.28 |
| 8 | 1007 | 15.97 | 17.30 | 2.00 | 0.109 | 0.092 | 16.19 |

Prefill: 2048 tokens in 16 steps; 0.2972 ms per token (P50 over steps), 3314 tokens/s overall; 19.02 ms per request (P50).

GPU: SM clock median 2655 MHz under load, max 63 °C, mean power 56.3 W.

Requests: 32 of 32 succeeded; trace: `trace-decode.json`, steps: `steps-decode.csv`.

## `prefill`: 32 requests at once, 480-token prompts, 8 output tokens each

Mostly prefill: its cost per token.

Decode-only steps by batch size (ms):

| batch | steps | decode P50 | decode P95 | per sequence | host P50 | invariants P50 | step P50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 15.45 | 15.45 | 15.45 | 3.044 | 0.099 | 18.59 |
| 2 | 1 | 15.39 | 15.39 | 7.69 | 2.742 | 0.101 | 18.23 |
| 3 | 1 | 16.21 | 16.21 | 5.40 | 2.234 | 0.095 | 18.54 |
| 4 | 1 | 15.61 | 15.61 | 3.90 | 2.872 | 0.097 | 18.58 |
| 5 | 1 | 16.64 | 16.64 | 3.33 | 1.480 | 0.094 | 18.21 |
| 6 | 1 | 16.03 | 16.03 | 2.67 | 2.064 | 0.097 | 18.19 |
| 7 | 1 | 16.14 | 16.14 | 2.31 | 3.571 | 0.151 | 19.87 |

Prefill: 15360 tokens in 32 steps; 0.2805 ms per token (P50 over steps), 3569 tokens/s overall; 134.66 ms per request (P50).

GPU: SM clock median 2655 MHz under load, max 68 °C, mean power 56.2 W.

Requests: 32 of 32 succeeded; trace: `trace-prefill.json`, steps: `steps-prefill.csv`.

## `mixed`: synthetic prompts of 32/128/480 tokens with 32/128/384 output tokens, 2 requests/s

The mixed profile at its own rate: a realistic step mix.

Decode-only steps by batch size (ms):

| batch | steps | decode P50 | decode P95 | per sequence | host P50 | invariants P50 | step P50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 586 | 15.09 | 17.06 | 15.09 | 0.026 | 0.097 | 15.21 |
| 2 | 391 | 15.60 | 16.46 | 7.80 | 0.039 | 0.099 | 15.77 |
| 3 | 207 | 15.71 | 16.41 | 5.24 | 0.054 | 0.100 | 15.87 |
| 4 | 305 | 15.81 | 16.42 | 3.95 | 0.064 | 0.100 | 15.99 |
| 5 | 321 | 15.91 | 16.59 | 3.18 | 0.078 | 0.101 | 16.14 |
| 6 | 590 | 15.99 | 17.00 | 2.66 | 0.089 | 0.098 | 16.20 |
| 7 | 472 | 16.07 | 17.25 | 2.30 | 0.099 | 0.096 | 16.30 |
| 8 | 543 | 16.16 | 17.54 | 2.02 | 0.118 | 0.098 | 16.43 |

Prefill: 17792 tokens in 120 steps; 0.2855 ms per token (P50 over steps), 3461 tokens/s overall; 25.32 ms per request (P50).

GPU: SM clock median 2655 MHz under load, max 73 °C, mean power 56.9 W.

Requests: 120 of 120 succeeded; trace: `trace-mixed.json`, steps: `steps-mixed.csv`.
