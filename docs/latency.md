# Latency metrics

Every response carries its own timings in the `neuroserve` block (the last chunk before
`[DONE]` when streaming), and the engine keeps percentiles over all requests, reported by
`/health` under `latency` and by `Engine.latency()` in Python.

## What is measured

All times come from one monotonic clock (`std::chrono::steady_clock`) in the C++ engine, so they
do not include HTTP or JSON overhead, and a wall-clock change cannot distort them.

| Metric | From | To | Samples |
| --- | --- | --- | --- |
| `queue_ms` | first `QUEUED` | first `ADMITTED` | one per completed request |
| `ttft_ms` | `RECEIVED` | first token (time to first token) | one per completed request |
| `itl_ms` | one new token | the next new token (inter-token latency) | one per token after the first |
| `e2e_ms` | `RECEIVED` | `COMPLETED` | one per completed request |

Per request, `itl_ms` is the mean gap between consecutive tokens and `max_itl_ms` the longest one.
After a preemption the engine regenerates tokens the client already has; those are not timed again,
but the stall before the next new token is, so a preemption shows up in `max_itl_ms` and in the
tail of the ITL distribution, as the client experiences it.

## Percentiles from fixed buckets

`LatencyHistogram` (runtime/core) counts samples in fixed, geometric buckets: 8 per factor of
two, from 0.01 ms to 10 minutes, plus one bucket below and one above. Memory and recording cost
stay constant however many requests the server handles.

P50, P95 and P99 are estimated by linear interpolation inside the bucket that holds the requested
rank. That bucket also holds the true value (nearest-rank definition), so the error is below one
bucket width: 2^(1/8) - 1, about 9% of the value. `tests/unit/test_latency_histogram.cpp` checks
that bound against known samples: uniform, a slow 2% tail, and 100,000 log-normal samples. The
mean, the count and the maximum are exact.

Failed, cancelled and rejected requests are not included: these are latencies of served requests.
