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
