# Prefix caching on vs off

- **When:** 2026-11-21T22:16:20+00:00 (0f9d71a-dirty)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU · **model:** llama-3.2-1b-instruct · **base config:** `configs/dev.yaml` (2048 KV pages)
- **Design:** 4 measured rounds after 1 discarded warm-up round(s); a fresh server per run (empty cache), warmed with 4 unrelated requests; both variants run the same seeded schedule within a round, in alternating order (seeds 1–5)
- **Modes:** paced = the profile's own Poisson rate; burst = every request sent at once (1000/s), no queue timeout: output tok/s is then the server's throughput
- **Cells:** median over the measured rounds (min–max). Times in ms, measured by the client; ITL is each request's mean gap (server). Prefilled tokens = prompt tokens admitted minus those reused from the cache. Hit rate = reused pages ÷ full prompt pages looked up.

## `shared_prefix`, paced: 4 shared 384-token prefixes (uniform) + a 64-token suffix each, 64 output tokens, 4 requests/s

KV pool: 2048 pages (100% of the base config).

| Variant | TTFT P50 | TTFT P95 | e2e P50 | e2e P95 | ITL P50 | output tok/s | prefilled tokens | hit rate | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cache-off` | 1846 (1172–4992) | 5915 (4057–6750) | 3893 (3334–7120) | 8007 (5659–8797) | 33.1 (32.1–34.0) | 224 (196–232) | 53760 (53760–53760) | 0.0% (0.0%–0.0%) | 0 |
| `cache-on` | 123 (78.6–459) | 1055 (787–1246) | 1572 (1486–1966) | 2449 (2272–2767) | 23.0 (21.9–24.1) | 258 (228–262) | 9216 (9216–9216) | 82.9% (82.9%–82.9%) | 0 |

`cache-on` paired with `cache-off` (same round, same requests): median change, and in how many rounds it was better.

| TTFT P50 | TTFT P95 | e2e P50 | e2e P95 | output tok/s | prefilled tokens |
| --- | --- | --- | --- | --- | --- |
| -92.8% (4/4) | -81.8% (4/4) | -60.3% (4/4) | -68.7% (4/4) | +14.3% (4/4) | -82.9% (4/4) |

Cache, summed over the measured `cache-on` runs: 480 lookups; 11136 of 13440 full prompt pages reused (82.9%); a cache that never evicted could have reused 11136 (100% of that reached); 178176 prompt tokens not prefilled; 0 pages evicted; 0 preemptions.

## `shared_prefix`, burst: 4 shared 384-token prefixes (uniform) + a 64-token suffix each, 64 output tokens, 4 requests/s

KV pool: 2048 pages (100% of the base config). Rate: 1000/s.

| Variant | TTFT P50 | TTFT P95 | e2e P50 | e2e P95 | ITL P50 | output tok/s | prefilled tokens | hit rate | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cache-off` | 15769 (15548–16459) | 30777 (30217–31324) | 17820 (17632–18674) | 32724 (32181–33221) | 32.4 (32.1–33.6) | 233 (230–237) | 53760 (53760–53760) | 0.0% (0.0%–0.0%) | 0 |
| `cache-on` | 10833 (10248–10844) | 20989 (20543–21197) | 12317 (11840–12357) | 22410 (21965–22624) | 22.8 (22.7–23.0) | 339 (336–340) | 9216 (9216–9216) | 82.9% (82.9%–82.9%) | 0 |

`cache-on` paired with `cache-off` (same round, same requests): median change, and in how many rounds it was better.

| TTFT P50 | TTFT P95 | e2e P50 | e2e P95 | output tok/s | prefilled tokens |
| --- | --- | --- | --- | --- | --- |
| -31.2% (4/4) | -31.5% (4/4) | -30.9% (4/4) | -31.2% (4/4) | +45.0% (4/4) | -82.9% (4/4) |

Cache, summed over the measured `cache-on` runs: 480 lookups; 11136 of 13440 full prompt pages reused (82.9%); a cache that never evicted could have reused 11136 (100% of that reached); 178176 prompt tokens not prefilled; 0 pages evicted; 0 preemptions.

## `memory_pressure`, paced: 16 shared 448-token prefixes (Zipf, s = 1) + a 32-token suffix each, 128 / 256 output tokens, 1.5 requests/s

KV pool: 512 pages (25% of the base config).

| Variant | TTFT P50 | TTFT P95 | e2e P50 | e2e P95 | ITL P50 | output tok/s | prefilled tokens | hit rate | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cache-off` | 325 (173–1362) | 3820 (1649–4435) | 5661 (5102–5845) | 8939 (6720–9613) | 22.4 (22.2–22.6) | 277 (267–288) | 57600 (57600–57600) | 0.0% (0.0%–0.0%) | 0 |
| `cache-on` | 148 (60.6–160) | 1929 (565–2597) | 4643 (4622–4876) | 6276 (5189–7109) | 19.3 (19.0–19.5) | 280 (268–293) | 16304 (15232–19440) | 71.7% (66.2%–73.6%) | 0 |

`cache-on` paired with `cache-off` (same round, same requests): median change, and in how many rounds it was better.

| TTFT P50 | TTFT P95 | e2e P50 | e2e P95 | output tok/s | prefilled tokens |
| --- | --- | --- | --- | --- | --- |
| -63.4% (4/4) | -50.8% (4/4) | -16.7% (4/4) | -27.2% (4/4) | +1.1% (4/4) | -71.7% (4/4) |

Cache, summed over the measured `cache-on` runs: 480 lookups; 10195 of 14400 full prompt pages reused (70.8%); a cache that never evicted could have reused 11732 (87% of that reached); 163120 prompt tokens not prefilled; 7473 pages evicted; 0 preemptions.

## `memory_pressure`, burst: 16 shared 448-token prefixes (Zipf, s = 1) + a 32-token suffix each, 128 / 256 output tokens, 1.5 requests/s

KV pool: 512 pages (25% of the base config). Rate: 1000/s.

| Variant | TTFT P50 | TTFT P95 | e2e P50 | e2e P95 | ITL P50 | output tok/s | prefilled tokens | hit rate | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cache-off` | 33167 (31407–33668) | 62806 (61540–64614) | 37811 (36905–39165) | 67087 (64772–70224) | 23.2 (23.0–24.2) | 322 (309–333) | 57600 (57600–57600) | 0.0% (0.0%–0.0%) | 0 |
| `cache-on` | 27913 (26833–28502) | 53163 (52078–54131) | 31547 (31050–32551) | 56974 (54950–59259) | 19.9 (19.8–20.8) | 375 (374–391) | 16448 (14736–17984) | 71.5% (68.8%–74.4%) | 0 |

`cache-on` paired with `cache-off` (same round, same requests): median change, and in how many rounds it was better.

| TTFT P50 | TTFT P95 | e2e P50 | e2e P95 | output tok/s | prefilled tokens |
| --- | --- | --- | --- | --- | --- |
| -15.3% (4/4) | -15.4% (4/4) | -15.9% (4/4) | -15.2% (4/4) | +17.1% (4/4) | -71.4% (4/4) |

Cache, summed over the measured `cache-on` runs: 480 lookups; 10299 of 14400 full prompt pages reused (71.5%); a cache that never evicted could have reused 11732 (88% of that reached); 164784 prompt tokens not prefilled; 7377 pages evicted; 0 preemptions.

## GPU

Clock figures use the samples taken under load (GPU utilisation of 50% or more).

| Profile, mode | Variant | SM clock median (MHz) | SM clock P5 | max temp (°C) | mean power (W) | power-capped | thermal slowdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `shared_prefix`, paced | `cache-off` | 2640 | 2640 | 77 | 58.4 | 1% | 3% |
| `shared_prefix`, paced | `cache-on` | 2640 | 2625 | 75 | 57.2 | 5% | 1% |
| `shared_prefix`, burst | `cache-off` | 2640 | 2610 | 77 | 58.8 | 1% | 1% |
| `shared_prefix`, burst | `cache-on` | 2640 | 2640 | 73 | 58.5 | 0% | 3% |
| `memory_pressure`, paced | `cache-off` | 2640 | 2640 | 74 | 58.3 | 0% | 0% |
| `memory_pressure`, paced | `cache-on` | 2640 | 2640 | 74 | 57.6 | 1% | 1% |
| `memory_pressure`, burst | `cache-off` | 2640 | 2640 | 75 | 59.1 | 1% | 3% |
| `memory_pressure`, burst | `cache-on` | 2640 | 2640 | 74 | 58.6 | 1% | 3% |

Warnings: 

- round 2 shared_prefix/paced cache-on: thermal slowdown in 2 of 142 samples
- round 2 shared_prefix/paced cache-off: thermal slowdown in 5 of 171 samples
- round 4 shared_prefix/paced cache-on: thermal slowdown in 1 of 146 samples
- round 2 shared_prefix/burst cache-on: thermal slowdown in 3 of 112 samples
- round 2 shared_prefix/burst cache-off: thermal slowdown in 2 of 159 samples
- round 3 shared_prefix/burst cache-on: thermal slowdown in 1 of 110 samples
- round 4 shared_prefix/burst cache-off: thermal slowdown in 1 of 159 samples
- round 5 shared_prefix/burst cache-off: thermal slowdown in 2 of 162 samples
- round 2 memory_pressure/paced cache-on: thermal slowdown in 2 of 396 samples
- round 2 memory_pressure/paced cache-off: thermal slowdown in 2 of 395 samples
- round 3 memory_pressure/paced cache-on: thermal slowdown in 4 of 411 samples
- round 4 memory_pressure/paced cache-on: thermal slowdown in 3 of 384 samples
- round 4 memory_pressure/paced cache-off: thermal slowdown in 2 of 387 samples
- round 5 memory_pressure/paced cache-off: thermal slowdown in 2 of 400 samples
- round 5 memory_pressure/paced cache-on: thermal slowdown in 1 of 402 samples
- round 2 memory_pressure/burst cache-on: thermal slowdown in 8 of 300 samples
- round 2 memory_pressure/burst cache-off: thermal slowdown in 11 of 364 samples
- round 3 memory_pressure/burst cache-off: thermal slowdown in 2 of 338 samples
- round 3 memory_pressure/burst cache-on: thermal slowdown in 3 of 289 samples
- round 4 memory_pressure/burst cache-on: thermal slowdown in 1 of 300 samples
- round 4 memory_pressure/burst cache-off: thermal slowdown in 2 of 347 samples
- round 5 memory_pressure/burst cache-off: thermal slowdown in 1 of 336 samples
