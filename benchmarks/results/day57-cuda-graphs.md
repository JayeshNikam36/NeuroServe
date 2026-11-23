# CUDA Graphs for decode: A/B

- **What:** the wall time of one `decode_step()` call (P50 over 64 timed steps after 8 warm-up steps), each batch of sequences prefilled to 512 tokens; median over 4 measured rounds (round 1 discarded), the path order rotating each round.
- **Paths:** `eager` = the engine's batched decode (today); `graph-eager` = the graph's work issued kernel by kernel (padded to the bucket); `graph` = replayed from a CUDA Graph.
- **GPU:** SM clock 2640-2640 MHz after the measured runs; power-capped in 0, thermally capped in 0 of 96 runs.

## Step time

| batch | bucket | eager (ms) | graph-eager (ms) | graph (ms) | graph vs eager | graph vs graph-eager |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 15.12 | 14.94 | 14.77 | -2.3% (4/4) | -1.2% (4/4) |
| 2 | 2 | 15.62 | 15.40 | 14.88 | -4.5% (4/4) | -3.2% (4/4) |
| 3 | 4 | 15.76 | 15.58 | 15.12 | -3.8% (4/4) | -2.9% (4/4) |
| 4 | 4 | 15.95 | 15.70 | 15.21 | -4.3% (4/4) | -3.0% (3/4) |
| 5 | 8 | 15.99 | 15.94 | 15.40 | -3.5% (4/4) | -3.5% (4/4) |
| 6 | 8 | 16.19 | 16.06 | 15.53 | -3.4% (4/4) | -2.8% (4/4) |
| 7 | 8 | 16.28 | 16.15 | 15.66 | -3.6% (4/4) | -2.8% (4/4) |
| 8 | 8 | 16.40 | 16.19 | 15.73 | -3.9% (4/4) | -2.8% (4/4) |

Paired with the same round's run: median change of the step time, and in how many rounds the graph was faster.

## Host time per step (graph paths)

| batch | graph-eager: stage / issue / wait (ms) | graph: stage / issue / wait (ms) |
| --- | --- | --- |
| 1 | 0.001 / 2.002 / 13.20 | 0.001 / 0.171 / 14.60 |
| 2 | 0.001 / 2.197 / 13.40 | 0.001 / 0.150 / 14.76 |
| 3 | 0.001 / 2.259 / 13.45 | 0.000 / 0.183 / 15.01 |
| 4 | 0.001 / 2.301 / 13.59 | 0.000 / 0.148 / 15.08 |
| 5 | 0.001 / 2.187 / 13.83 | 0.001 / 0.185 / 15.28 |
| 6 | 0.001 / 2.325 / 13.82 | 0.001 / 0.187 / 15.40 |
| 7 | 0.001 / 2.287 / 14.02 | 0.001 / 0.174 / 15.52 |
| 8 | 0.001 / 2.240 / 14.17 | 0.001 / 0.168 / 15.57 |

Stage: writing the pinned inputs; issue: launching the GPU work (about 280 kernels kernel by kernel, one `cudaGraphLaunch` from a graph); wait: `cudaStreamSynchronize` until it is done.

## The cost of padding

A batch that is not a bucket size runs padded to its bucket. The graph's step time at each such batch against the graph's at its bucket, and against the engine's unpadded eager step:

| batch | bucket | graph (ms) | graph at the bucket (ms) | vs the bucket | eager, unpadded (ms) |
| --- | --- | --- | --- | --- | --- |
| 3 | 4 | 15.12 | 15.21 | -0.6% | 15.76 |
| 5 | 8 | 15.40 | 15.73 | -2.1% | 15.99 |
| 6 | 8 | 15.53 | 15.73 | -1.2% | 16.19 |
| 7 | 8 | 15.66 | 15.73 | -0.4% | 16.28 |

Graphs captured: 4; replays: 2876.
