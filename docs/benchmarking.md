# Benchmarking

## Load generator

`python -m neuroserve.bench.loadgen` sends a seeded workload to a running server and reports
latency percentiles, throughput and errors:

    python -m neuroserve.bench.loadgen --url http://127.0.0.1:8000 --profile short --seed 1 --out results/short.json
    python -m neuroserve.bench.loadgen --profile mixed --seed 1 --dry-run     # the schedule only

- **Open loop, wall-clock arrivals.** Each request is sent at its planned time whether or not
  earlier ones have finished, as real traffic would. Arrival gaps are exponential (a Poisson
  process) at the profile's rate. `max_send_lag_ms` in the summary shows how closely the planned
  times were kept.
- **Reproducible.** The schedule (arrival times, prompts, output lengths) is derived from the
  profile and the seed alone, with separate seeded streams for arrivals and content: the same seed
  gives the same schedule on every machine (`schedule_sha256` in the results proves it), and a
  shorter run is a prefix of a longer one.
- **Controlled lengths.** Every request sets `neuroserve.ignore_eos`, so it produces exactly its
  `max_tokens`; the `mixed` profile sends prompts as token ids of exact lengths.
- **Measured on both sides.** Requests are streamed: the client records its own time to first
  token and end-to-end time; the server's timings (queue, TTFT, ITL, e2e) come from the
  `neuroserve` block of the last chunk.

| Profile | Prompts | Output tokens | Rate | Requests |
| --- | --- | --- | --- | --- |
| `short` | the 32 short questions of `prompts-v1` (chat) | 64 | 4/s | 120 |
| `mixed` | synthetic, 32 / 128 / 480 tokens (50 / 30 / 20%) | 32 / 128 / 384 (40 / 40 / 20%) | 2/s | 120 |
| `shared_prefix` | one of 4 shared 384-token prefixes (uniform) + 64 tokens of its own | 64 | 4/s | 120 |
| `memory_pressure` | one of 16 shared 448-token prefixes (Zipf, s = 1) + 32 tokens of its own | 128 / 256 (50 / 50%) | 1.5/s | 120 |

The two prefix profiles model requests that start with a shared system prompt. Their prefixes are
synthetic token ids drawn from a seeded stream of their own (other prefixes for another seed) and
are whole 16-token pages long, so a cache hit can reuse every prefix page. In `memory_pressure`
prefix k is picked with weight 1 / (k + 1): a few prefixes are popular and the rest rare, which is
what decides how well an eviction policy keeps the useful ones.

`--requests` and `--rate` override a profile's size and rate.

## The prompt dataset

`python/neuroserve/bench/data/prompts-v1.jsonl` is versioned: its digest is pinned in
`workload.py` and checked whenever it is loaded. Results measured on different prompts are not
comparable, so a change means a new file (`prompts-v2.jsonl`), never an edit.

## Results

`--out` writes one JSON file: `meta` (profile, seed, schedule and dataset digests, the server's
policy, GPU and model), `summary` (succeeded, errors by code, duration, throughput, output tokens
per second, send lag, and count/mean/P50/P95/P99 of client TTFT and e2e and server queue time and
ITL), and every request's own record.

## Comparing scheduling policies

    python -m neuroserve.bench.compare --out-dir benchmarks/results/raw/compare-1

Runs every profile (`--profiles`, default `short,mixed`) under every policy (`--policies`, default
`fcfs,token-budget,decode-first`; the first is the baseline) for `--rounds` measured rounds
(default 6), after `--warmup-rounds` (default 1) that are discarded. The server switches policy
between runs (`PUT /v1/runtime/policy`), so one server process on one warm GPU serves the whole
comparison. After every run the decision log is read back: every decision of the run must have
been made by the requested policy, or the comparison stops.

- **Paired.** Within a round every policy runs the same seeded schedule (round r uses seed + r - 1):
  the policies are compared on identical requests. Across rounds the requests change.
- **Balanced order.** The order of the policies follows a Williams design. Over a full cycle (6
  rounds for 3 policies) each policy runs first, second and last equally often, and follows each
  other policy equally often, so a GPU that heats up during a round, or a run that leaves state
  behind, cannot favour one policy.
- **Warm-up discarded.** The first round absorbs CUDA and allocator warm-up and the low clocks of
  an idle GPU.
- **Idle between runs.** A run starts only when nothing is queued or running, and is followed by
  `--cooldown-s` (5 s).
- **GPU state recorded.** `neuroserve.bench.gpu_monitor` samples the GPU with NVML (nvidia-ml-py)
  every 0.2 s: SM and memory clocks, temperature, power, utilisation and the reasons the clocks are
  being held down. The report flags any run whose median SM clock is more than 5% from the typical
  one for its profile, and any thermal slowdown. Power capping is reported too, but it is normal for
  a laptop GPU under sustained load.

`summary.md` gives, per profile and policy, the median over the measured rounds of each run's
percentiles, with their range across rounds; then the paired comparison with the baseline: the
median per-round change, and in how many rounds the policy was better. `summary.json` holds all of
it, and `runs/` has one file per run with every request and every GPU sample.
`--rerender DIR` recomputes `summary.json` and `summary.md` from those saved runs after a change to
the analysis, without repeating a run.

## Prefix caching on vs off

    python -m neuroserve.bench.prefix_ab --config configs/dev.yaml --out-dir benchmarks/results/raw/prefix-ab-1

Runs `shared_prefix` and `memory_pressure` (`--profiles`) with prefix caching off and on, for
`--rounds` measured rounds (default 4) after `--warmup-rounds` (default 1) that are discarded.

- **A fresh server per run.** Prefix caching is an engine setting, so every run starts its own
  server process (`python -m neuroserve.server`, on `--port`, default 8100) from the base config
  with `engine.prefix_caching` set, and stops it afterwards. Every run therefore starts with an
  empty cache, and the two variants differ in that one setting. Before measuring, each server
  serves 4 unrelated requests (CUDA and allocator warm-up); the counters are read after them.
- **The KV pool.** `memory_pressure` runs with 25% of the base config's pages (512 of 2048): every
  running request still fits, but there is room to keep only a few of the 16 prefixes cached, so
  the cache has to evict. `shared_prefix` runs with the whole pool.
- **Paced and burst** (`--modes`). Paced sends the profile's schedule at its own Poisson rate:
  latency at a realistic load, where throughput simply follows the offered load. Burst sends the
  same requests all at once with no queue timeout: the run's output tokens per second are then the
  server's throughput, and how long the queue takes to drain shows the work the cache saves.
- **Paired, alternating order.** Within a round both variants run the same seeded schedule; the
  order alternates from round to round (off first, then on first), so neither always runs on a
  GPU the other has just heated.
- **What is recorded.** Client TTFT and e2e, server ITL, output tokens per second; from
  `GET /v1/runtime/state` before and after the run: lookups, prompt pages looked up and reused
  (hit rate), reused tokens, evicted pages and preemptions; the prompt tokens actually prefilled
  (admitted minus reused); the GPU's clocks, temperature, power and throttle reasons, as in the
  policy comparison, with the same warnings. For each run the report also gives the pages a cache
  that never evicted could have reused, an upper bound on the hit rate.
- **Checked.** A run whose server reports a different prefix-caching setting or pool size than
  requested, or that looked up prefixes with the cache off, stops the comparison.

`summary.md` gives, per profile and mode, the median over the measured rounds (and range) of each
metric for both variants, the paired change of `cache-on` against `cache-off` and in how many
rounds it was better, the cache counters, and the GPU table. `runs/` holds every request and GPU
sample, `configs/` the config each server ran with, and `server-logs/` its log; `--rerender DIR`
recomputes the report from them.

## Stress run

    python -m neuroserve.bench.stress --config configs/dev.yaml --out-dir benchmarks/results/raw/stress-1

Sustained load for `--duration-min` minutes (default 30), to find what only shows over time:
leaked KV pages, growing memory, an invariant that breaks after thousands of steps, latency that
drifts.

- **The server.** Started by the run (as in the prefix comparison) with prefix caching on and 25%
  of the KV pool (`--pool-fraction`), so sharing, copy-on-write, eviction and preemption all happen
  under load, not only the plain path.
- **The load.** `--concurrency` clients (default 24, three times `max_batch`) in a closed loop:
  each sends its next request as soon as its last one ends, so the queue never empties. Requests
  are drawn from `mixed` (50%), `shared_prefix` (30%) and `memory_pressure` (20%), seeded; each
  profile's schedules change seed every 200 requests, so new prefixes keep arriving and old ones
  age out of the cache. `--cancel-fraction` (default 5%) of the requests are cancelled by the
  client 0.05–1.5 s after they are sent, mid-prefill or mid-stream.
- **Every `--interval-s` (10 s)** it records the requests that finished in that interval (TTFT
  P50 / P95, ITL P50, output tokens per second), the KV pages (used, cached, free, shared), the
  evictions, preemptions and prefix hit rate so far, the engine's queue, the GPU's memory in use
  (NVML) and the server process's resident memory. `timeline.csv` has every sample.
- **The drain.** At the end the clients stop sending and finish what they have in flight; the run
  then waits until the engine is idle and sends one probe request.

The checks, all in `summary.md` (exit code 0 only if every one passes):

| Check | Passes when |
| --- | --- |
| no failed requests | every request not cancelled on purpose succeeded |
| invariant checks on | the build checks the KV invariants after every engine step (`NS_CHECK_INVARIANTS`, reported by `neuroserve._core.check_invariants`) |
| engine alive, no invariant violation | `/health` ok and the probe request succeeds after the drain; no violation and no traceback in the server log (a violation stops the engine loop) |
| no KV pages held after the drain | 0 used and 0 shared pages; free + cached = all pages |
| GPU memory flat | the end's peak is within 64 MiB of the start's median |
| server memory flat | the end's median resident memory is within 64 MiB or 5% of the start's |
| TTFT P50, TTFT P95, ITL P50 stable | the end's median is at most 25% (P95: 50%) above the start's |
| throughput stable | the end's median output tokens per second is at most 20% below the start's |

"Start" and "end" are the first and last third of the samples after a `--warmup-min` (5 min)
warm-up; every threshold is a flag.

## First results: scheduling policies (Day 24)

Full report: [benchmarks/results/day24-policy-comparison.md](../benchmarks/results/day24-policy-comparison.md).
RTX 4050 Laptop GPU, Llama 3.2 1B (FP16), `max_batch` 8, 512-token step budget; 6 measured rounds
of 120 requests per profile and policy.

- **`short`: the policy makes no practical difference.** Every paired change is within ±2.1%. The
  only change in a consistent direction is `token-budget`'s TTFT P50, lower in 5 of 6 rounds, by
  0.6% (about 0.2 ms). With ~40-token prompts the token budget never binds, and the batch limit
  applies to every policy alike. The round-to-round spread (TTFT P95 from about 45 to 300 ms under
  every policy) comes from the schedules themselves: some seeds produce bursts. Without pairing,
  that spread would swamp any policy effect.
- **`mixed`: `token-budget` behaves like `fcfs`** (all paired changes within ±1.2%). At this load a
  waiting request is held back by the batch limit (8), not by the token budget, so letting smaller
  prompts skip ahead never comes into play.
- **`mixed`: `decode-first` is unstable at 2 requests/s.**
  - **TTFT P50:** 22.3 s against 52 ms.
  - **Timeouts:** 35 of 720 requests timed out after 30 s in the queue (all 480-token prompts).
  - **Throughput:** 34% lower.
  - **The mechanism:** while anything decodes, a step's prefill is capped at 128 tokens (a quarter
    of the budget), so a 480-token prompt at the head of the queue waits until the batch has fully
    drained. Strict arrival order then holds every request behind it.
  - **Why the queue grows:** every long prompt forces a drain, so capacity falls below the offered
    load and the queue grows until requests time out. Short prompts wait as long as long ones (TTFT
    P50 about 20 s for 32-token prompts).
- **`decode-first` still does what it is for.** Max ITL P95 is 44% lower (147 → 83 ms) and ITL P50
  3.8% lower, in 6 of 6 rounds: running requests are no longer stalled by steps that also run large
  prefills. Keeping that benefit without starving long prompts needs chunked prefill (a long
  prompt's prefill split across steps).
- **The GPU was stable:** the median SM clock under load was 2730–2745 MHz in every run, the
  temperature stayed at or below 71 °C, there was no power capping, and thermal slowdown appeared in
  a single sample of one run.

## Results: prefix caching (Day 43)

Full report: [benchmarks/results/day43-prefix-caching.md](../benchmarks/results/day43-prefix-caching.md).
RTX 4050 Laptop GPU, Llama 3.2 1B (FP16), `max_batch` 8, 512-token step budget; 4 measured rounds
of 120 requests per profile, mode and variant, after one warm-up round; every request succeeded.
Medians over the rounds; paired change = cache-on against cache-off in the same round.

| Profile, mode | TTFT P50 off → on | TTFT P95 off → on | output tok/s off → on | hit rate |
| --- | --- | --- | --- | --- |
| `shared_prefix`, paced | 1846 → 123 ms (−93%, 4/4) | 5915 → 1055 ms (−82%, 4/4) | 224 → 259 (+14%) | 82.9% |
| `shared_prefix`, burst | 15.8 → 10.8 s (−31%, 4/4) | 30.8 → 21.0 s (−32%, 4/4) | 233 → 339 (**+45%**, 4/4) | 82.9% |
| `memory_pressure`, paced | 325 → 149 ms (−63%, 4/4) | 3820 → 1929 ms (−51%, 4/4) | 277 → 280 (+1%) | 66–74% |
| `memory_pressure`, burst | 33.2 → 27.9 s (−15%, 4/4) | 62.8 → 53.2 s (−15%, 4/4) | 323 → 375 (**+17%**, 4/4) | 69–74% |

- **The cache raises capacity, and that is what moves latency.** In the burst runs throughput is
  the server's capacity: for `shared_prefix` it rises from 233 to 339 output tokens per second,
  from about 3.6 to 5.3 requests per second at 64 tokens each. Paced at 4 requests per second,
  the server without the cache is therefore overloaded (TTFT P50 1.8 s and growing queues, output
  below the offered 256 tok/s); with the cache it keeps up (TTFT P50 0.12 s). The cache skips 384
  of each 448 prompt tokens, and here prefill is most of the work.
- **`shared_prefix` reached its upper bound.** 82.9% is exactly what a cache that never evicted
  could do here: the first request with each of the 4 prefixes misses, every later one reuses all
  24 prefix pages of its 28. Nothing was evicted.
- **Under memory pressure the gain is smaller, for two reasons.** Outputs are longer (128 or 256
  tokens), so decode is a larger share of the work the cache cannot skip; and with 512 pages the
  cache cannot hold all 16 prefixes, so about 1,800 pages were evicted per run. LRU still kept
  66–74% hit rate against a bound of 81–82%, about 87% of the possible reuse (the Zipf weights
  keep the popular prefixes recently used). Paced throughput is unchanged because the server keeps
  up with the offered 1.5 requests per second either way; the latency gain is the shorter prefills.
- **Nothing was traded for it.** No request failed in any run, with the cache on or off, on the
  full pool or on a quarter of it. Evictions and preemptions per variant are in the full report.
- **The GPU was stable:** the median SM clock under load was 2640 MHz in every run (40 of 40), and
  the temperature peaked between 70 and 77 °C; power and throttle shares are in the full report.
- **Absolute numbers are this engine's baseline kernels.** The decode and prefill kernels are
  rewritten in Phase 4; the comparison (same engine, cache off vs on) is what this measures.
