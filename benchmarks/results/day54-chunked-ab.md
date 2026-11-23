# Chunked prefill against unchunked, long prompts arriving

- **When:** 2026-11-23T02:10:00+00:00 (af7ae45-dirty)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU · **model:** llama-3.2-1b-instruct · **base config:** `configs/dev.yaml`
- **Variants:** `unchunked` = {"prefill_chunk_tokens": 0, "mixed_steps": true}; `chunked` = {"prefill_chunk_tokens": 128, "max_batch_tokens": 256, "mixed_steps": true}
- **Design:** 4 measured rounds after 1 discarded warm-up round(s); a fresh server per run, warmed with 4 unrelated requests; every variant runs the same seeded schedule within a round, the order balanced across rounds (seeds 1–5); paced at each profile's own rate
- **Cells:** median over the measured rounds (min–max). Times in ms, measured by the client. TTFT long / short: prompts of at least / under 256 tokens. ITL: the engine's time between consecutive tokens of a request, over every token of the run; ITL max: the longest; worst gap P99: the P99 over requests of each request's longest gap (engine). Text gap P99: the client's gaps between streamed text chunks, for reference (an output can stream many tokens as one chunk).

## `long_prompt`: synthetic prompts, 7 in 10 of 32 tokens and 3 in 10 of 480, 128 output tokens each, 3 requests/s

| Variant | TTFT P50 | TTFT P99 | TTFT long P50 | TTFT long P99 | TTFT short P50 | TTFT short P99 | ITL P50 | ITL P99 | ITL max | text gap P99 | worst gap P99 | e2e P50 | output tok/s | failed | mixed steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `unchunked` | 365 (63.4–853) | 1760 (1157–2194) | 424 (86.3–819) | 1689 (1157–2194) | 353 (52.5–957) | 1697 (1127–2123) | 16.5 (16.4–16.6) | 54.5 (53.9–59.4) | 65.9 (59.4–79.2) | 56.0 (54.7–59.1) | 65.9 (59.4–79.2) | 2545 (2258–3032) | 366 (301–411) | 0 | 251 |
| `chunked` | 478 (60.7–909) | 1804 (1290–2406) | 494 (140–1013) | 1766 (1290–2406) | 435 (51.9–941) | 1721 (1199–2153) | 16.6 (16.5–16.6) | 31.1 (31.0–31.4) | 42.8 (35.2–49.1) | 38.2 (34.8–43.6) | 42.8 (35.2–49.1) | 2672 (2310–3088) | 362 (301–411) | 0 | 434 |

Paired with `unchunked` (same round, same requests): median change, and in how many rounds the variant was better.

| Variant | TTFT P50 | TTFT P99 | TTFT long P50 | TTFT short P50 | ITL P99 | worst gap P99 | e2e P50 | output tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `chunked` | +16.6% (1/4) | +6.2% (0/4) | +41.3% (0/4) | -1.4% (3/4) | -42.8% (4/4) | -39.0% (4/4) | +2.7% (0/4) | -0.4% (0/4) |

## `mixed`: synthetic prompts of 32/128/480 tokens with 32/128/384 output tokens, 2 requests/s

| Variant | TTFT P50 | TTFT P99 | TTFT long P50 | TTFT long P99 | TTFT short P50 | TTFT short P99 | ITL P50 | ITL P99 | ITL max | text gap P99 | worst gap P99 | e2e P50 | output tok/s | failed | mixed steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `unchunked` | 56.9 (54.5–57.1) | 1113 (214–1779) | 84.3 (81.4–88.1) | 962 (164–1037) | 52.6 (51.4–54.6) | 1815 (257–2307) | 16.3 (16.3–16.5) | 30.8 (28.0–36.9) | 69.9 (60.6–84.7) | 41.0 (37.9–50.6) | 69.9 (60.5–76.4) | 2150 (2120–2190) | 248 (215–320) | 0 | 466 |
| `chunked` | 57.1 (54.9–59.4) | 1137 (228–1780) | 143 (127–146) | 1052 (224–1078) | 54.0 (50.6–54.4) | 1828 (263–2331) | 16.4 (16.3–16.5) | 30.1 (28.8–31.6) | 99.3 (73.9–122) | 36.9 (34.1–42.9) | 99.3 (73.9–122) | 2159 (2125–2197) | 248 (215–320) | 0 | 721 |

Paired with `unchunked` (same round, same requests): median change, and in how many rounds the variant was better.

| Variant | TTFT P50 | TTFT P99 | TTFT long P50 | TTFT short P50 | ITL P99 | worst gap P99 | e2e P50 | output tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `chunked` | +0.6% (0/4) | +3.3% (1/4) | +65.0% (0/4) | +0.2% (2/4) | -2.1% (3/4) | +43.5% (1/4) | +0.3% (0/4) | +0.0% (1/4) |

## GPU

Warnings: 

- round 2 long_prompt unchunked: thermal slowdown in 2 of 104 samples
- round 2 long_prompt chunked: thermal slowdown in 4 of 105 samples
- round 3 long_prompt chunked: thermal slowdown in 2 of 133 samples
- round 4 long_prompt unchunked: thermal slowdown in 2 of 115 samples
- round 4 long_prompt chunked: thermal slowdown in 2 of 117 samples
- round 5 long_prompt chunked: thermal slowdown in 1 of 98 samples
- round 5 long_prompt unchunked: thermal slowdown in 1 of 98 samples
- round 2 mixed unchunked: thermal slowdown in 1 of 353 samples
- round 3 mixed chunked: thermal slowdown in 4 of 296 samples
- round 3 mixed unchunked: thermal slowdown in 4 of 296 samples
- round 4 mixed unchunked: thermal slowdown in 2 of 359 samples
- round 4 mixed chunked: thermal slowdown in 2 of 354 samples
- round 5 mixed chunked: thermal slowdown in 1 of 249 samples
- round 5 mixed unchunked: thermal slowdown in 2 of 254 samples
