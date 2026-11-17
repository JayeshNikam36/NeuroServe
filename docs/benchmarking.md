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
comparison.

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

Committed results: [benchmarks/results/day24-policy-comparison.md](../benchmarks/results/day24-policy-comparison.md).
