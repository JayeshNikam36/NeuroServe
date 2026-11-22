# Stress run

**Result: PASS** (10 of 10 checks)

- **When:** 2026-11-22T00:56:40+00:00 to 2026-11-22T01:27:08+00:00 (1bde7f2-dirty)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU · **model:** llama-3.2-1b-instruct · **config:** `configs/dev.yaml` with prefix caching on and 512 KV pages (25% of 2048)
- **Load:** 24 clients in a closed loop for 30 min (each sends its next request when its last one ends); mix `mixed` 50%, `shared_prefix` 30%, `memory_pressure` 20%; 5% of requests cancelled after 0.05–1.5 s; seed 1
- **Sampling:** every 10 s; start and end compared over the first and last third of the samples after a 5-minute warm-up

| Check | Result | Measured |
| --- | --- | --- |
| no failed requests | pass | 3725 finished, 0 failed (none), 205 cancelled by the client on purpose |
| invariant checks on | pass | every engine step checks the KV invariants |
| engine alive, no invariant violation | pass | /health ok; probe request ok; server log: no violation, no traceback |
| no KV pages held after the drain | pass | used 0, shared 0, cached 509, free 3 of 512; engine active 0, pending 0 |
| GPU memory flat | pass | start median 3092 MiB, end peak 3092 MiB (allowed +64) |
| server memory flat | pass | resident memory start median 898 MiB, end median 896 MiB (allowed +64) |
| TTFT P50 stable | pass | median of the sample windows: start 7831.0 ms, end 7899.1 ms (+0.9%; allowed +25%) |
| TTFT P95 stable | pass | median of the sample windows: start 9641.2 ms, end 9792.7 ms (+1.6%; allowed +50%) |
| ITL P50 stable | pass | median of the sample windows: start 19.8 ms, end 19.9 ms (+0.1%; allowed +25%) |
| throughput stable | pass | output tok/s start 401.5, end 396.1 (-1.3%; allowed -20%); start = 301–794 s, end = 1316–1800 s |

Totals: 3725 requests finished, 719104 output tokens, 205 cancelled by the client, 0 failed; 95752 pages evicted, 0 preemptions, prefix hit rate 50.2%.

## Per minute

| Minute | finished | TTFT P50 | TTFT P95 | ITL P50 | output tok/s | used pages (max) | cached pages | evicted | preempted | GPU MiB | server RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 104 | 6997 | 7351 | 18.0 | 408 | 259 | 294 | 797 | 0 | 3092 | 863 |
| 2 | 133 | 7202 | 8774 | 18.5 | 422 | 266 | 312 | 2435 | 0 | 3092 | 888 |
| 3 | 136 | 7290 | 9251 | 18.3 | 422 | 268 | 292 | 2394 | 0 | 3092 | 889 |
| 4 | 120 | 7831 | 9597 | 19.6 | 407 | 298 | 268 | 3269 | 0 | 3092 | 890 |
| 5 | 128 | 7666 | 8555 | 19.4 | 408 | 284 | 256 | 3043 | 0 | 3092 | 898 |
| 6 | 128 | 7552 | 9522 | 19.3 | 409 | 260 | 292 | 3008 | 0 | 3092 | 897 |
| 7 | 126 | 7728 | 9756 | 19.7 | 421 | 298 | 286 | 3309 | 0 | 3092 | 897 |
| 8 | 121 | 7882 | 9893 | 19.7 | 389 | 313 | 270 | 3206 | 0 | 3092 | 897 |
| 9 | 130 | 7692 | 8762 | 20.1 | 408 | 320 | 260 | 3172 | 0 | 3092 | 898 |
| 10 | 126 | 7674 | 8133 | 19.8 | 402 | 293 | 249 | 3223 | 0 | 3092 | 898 |
| 11 | 123 | 8024 | 9081 | 20.3 | 401 | 289 | 242 | 3533 | 0 | 3092 | 898 |
| 12 | 125 | 7818 | 9766 | 19.8 | 397 | 278 | 282 | 3208 | 0 | 3092 | 898 |
| 13 | 122 | 7957 | 9812 | 20.3 | 396 | 306 | 249 | 3489 | 0 | 3092 | 898 |
| 14 | 117 | 7992 | 10090 | 19.9 | 402 | 320 | 266 | 3336 | 0 | 3092 | 898 |
| 15 | 125 | 7896 | 9018 | 20.4 | 389 | 267 | 257 | 3432 | 0 | 3092 | 897 |
| 16 | 121 | 7932 | 9998 | 20.0 | 407 | 312 | 236 | 3362 | 0 | 3092 | 897 |
| 17 | 130 | 7684 | 8312 | 19.5 | 414 | 292 | 270 | 3211 | 0 | 3092 | 897 |
| 18 | 121 | 7745 | 9716 | 19.5 | 407 | 276 | 272 | 3150 | 0 | 3092 | 896 |
| 19 | 122 | 8027 | 9972 | 20.2 | 384 | 311 | 244 | 3365 | 0 | 3092 | 896 |
| 20 | 125 | 7875 | 8063 | 20.0 | 402 | 325 | 238 | 3454 | 0 | 3092 | 898 |
| 21 | 117 | 8095 | 10168 | 20.0 | 388 | 296 | 254 | 3375 | 0 | 3092 | 898 |
| 22 | 126 | 7943 | 9840 | 20.3 | 412 | 288 | 269 | 3569 | 0 | 3092 | 897 |
| 23 | 123 | 7847 | 8992 | 19.5 | 396 | 258 | 268 | 3074 | 0 | 3092 | 897 |
| 24 | 127 | 7851 | 8966 | 20.0 | 394 | 272 | 247 | 3451 | 0 | 3092 | 897 |
| 25 | 120 | 8023 | 9132 | 20.6 | 388 | 344 | 234 | 3657 | 0 | 3092 | 896 |
| 26 | 116 | 8004 | 10141 | 20.0 | 403 | 262 | 256 | 3369 | 0 | 3092 | 896 |
| 27 | 125 | 7852 | 10014 | 19.6 | 420 | 318 | 261 | 3312 | 0 | 3092 | 896 |
| 28 | 124 | 7702 | 9031 | 20.0 | 382 | 293 | 275 | 3425 | 0 | 3092 | 896 |
| 29 | 117 | 7924 | 10048 | 19.7 | 402 | 292 | 224 | 3228 | 0 | 3092 | 896 |
| 30 | 128 | 7835 | 9754 | 19.8 | 394 | 278 | 264 | 3309 | 0 | 3092 | 896 |

Latency cells are medians of the 10 s windows in that minute (ms). `timeline.csv` has every sample.
