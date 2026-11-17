# Scheduling policies compared

- **When:** 2026-11-17T04:13:00+00:00 (2be6a36)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU · **model:** llama-3.2-1b-instruct · **dataset:** prompts-v1
- **Design:** 6 measured rounds after 1 discarded warm-up round(s); policy order balanced across rounds; every policy runs the same seeded schedule within a round (seeds 1–7)
- **Cells:** median over the measured rounds (min–max). Times in ms. TTFT and e2e are measured by the client; ITL is each request's mean time between tokens and max ITL its longest gap (server).

## `short`: short chat questions from prompts-v1, 64 output tokens each, 4 requests/s

| Policy | TTFT P50 | TTFT P95 | TTFT P99 | ITL P50 | ITL P95 | max ITL P95 | e2e P50 | e2e P95 | e2e P99 | queue P95 | output tok/s | failed | preempted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fcfs` | 29.8 (29.0–30.8) | 149 (43.9–293) | 282 (93.0–422) | 16.5 (16.4–17.0) | 17.6 (17.4–17.8) | 49.9 (48.0–50.7) | 1076 (1068–1110) | 1249 (1128–1409) | 1362 (1135–1547) | 125 (18.1–274) | 234 (223–278) | 0 | 0 |
| `token-budget` | 29.8 (28.7–30.7) | 164 (44.6–301) | 311 (50.5–431) | 16.6 (16.4–16.9) | 17.6 (17.4–17.7) | 49.1 (47.6–50.3) | 1078 (1066–1104) | 1302 (1130–1424) | 1513 (1140–1598) | 125 (22.8–279) | 234 (223–278) | 0 | 0 |
| `decode-first` | 30.0 (27.9–31.0) | 157 (46.2–301) | 314 (169–429) | 16.5 (16.5–16.9) | 17.6 (17.4–17.6) | 49.5 (48.1–50.8) | 1077 (1067–1109) | 1292 (1134–1384) | 1502 (1183–1546) | 127 (25.3–280) | 234 (223–278) | 0 | 0 |

Paired with `fcfs` (same round, same requests): median change, and in how many rounds the policy was better.

| Policy | TTFT P50 | TTFT P95 | ITL P50 | max ITL P95 | e2e P50 | e2e P95 | output tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `token-budget` | -0.6% (5/6) | +2.1% (1/6) | +0.3% (2/6) | -0.5% (4/6) | -0.1% (4/6) | -0.1% (3/6) | +0.0% (1/6) |
| `decode-first` | -1.8% (4/6) | +1.6% (1/6) | +0.1% (1/6) | -0.8% (4/6) | -0.1% (4/6) | -0.1% (3/6) | +0.0% (2/6) |

## `mixed`: synthetic prompts of 32/128/480 tokens with 32/128/384 output tokens, 2 requests/s

| Policy | TTFT P50 | TTFT P95 | TTFT P99 | ITL P50 | ITL P95 | max ITL P95 | e2e P50 | e2e P95 | e2e P99 | queue P95 | output tok/s | failed | preempted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fcfs` | 51.9 (48.5–69.1) | 888 (187–1030) | 1181 (301–1868) | 16.8 (16.6–17.1) | 20.6 (18.3–21.5) | 147 (146–148) | 2155 (2064–2251) | 6745 (6505–7025) | 7045 (6740–7481) | 766 (130–990) | 255 (216–322) | 0 | 0 |
| `token-budget` | 52.1 (49.2–65.2) | 890 (201–1020) | 1185 (335–1852) | 16.8 (16.6–17.2) | 20.6 (18.2–21.3) | 147 (146–147) | 2159 (2070–2254) | 6734 (6507–7026) | 7111 (6795–7465) | 767 (116–979) | 255 (216–322) | 0 | 0 |
| `decode-first` | 22300 (13676–27565) | 29515 (27992–29981) | 29955 (29013–30040) | 16.1 (16.0–16.3) | 19.0 (18.5–19.8) | 82.7 (73.1–98.9) | 23773 (16864–29017) | 32805 (29440–35501) | 35402 (30882–36085) | 29406 (27498–29827) | 180 (153–200) | 35 (queue_timeout 35) | 0 |

Paired with `fcfs` (same round, same requests): median change, and in how many rounds the policy was better.

| Policy | TTFT P50 | TTFT P95 | ITL P50 | max ITL P95 | e2e P50 | e2e P95 | output tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `token-budget` | -1.2% (3/6) | +0.6% (1/6) | +0.1% (2/6) | +0.1% (3/6) | +0.2% (0/6) | -0.0% (3/6) | +0.0% (3/6) |
| `decode-first` | ×387 (0/6) | ×33 (0/6) | -3.8% (6/6) | -43.6% (6/6) | ×11 (0/6) | ×4.8 (0/6) | -34.2% (0/6) |

## GPU

Clock figures use the samples taken under load (GPU utilisation of 50% or more).

| Profile | Policy | SM clock median (MHz) | SM clock P5 | max temp (°C) | mean power (W) | power-capped | thermal slowdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `short` | `fcfs` | 2745 | 2745 | 69 | 58.8 | 0% | 0% |
| `short` | `token-budget` | 2745 | 2745 | 68 | 58.5 | 0% | 0% |
| `short` | `decode-first` | 2745 | 2745 | 68 | 59.0 | 0% | 0% |
| `mixed` | `fcfs` | 2730 | 2730 | 70 | 60.1 | 0% | 0% |
| `mixed` | `token-budget` | 2730 | 2730 | 70 | 60.0 | 0% | 0% |
| `mixed` | `decode-first` | 2730 | 2730 | 71 | 62.8 | 0% | 0% |

Warnings: 

- round 4 mixed token-budget: thermal slowdown in 1 of 358 samples
