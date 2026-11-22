# Prefill attention kernel: time and TFLOPS by length

- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU, 20 SMs, L2 24 MB, compute capability 8.9
- **Shapes:** Llama 3.2 1B attention: 32 query heads, 8 KV heads, head dimension 64; fresh prompts (no cached prefix), 16-token pages in shuffled order; one layer
- **Kernels:** naive = Day 34's `attention_prefill_paged`, one launch per sequence; tiled = `attention_prefill_varlen_tiled`, one launch for the batch, at 64 x 32 (the default) and 128 x 64
- **Timing:** CUDA events; 3 warm-up runs, then up to 25 timed runs (at least 5, about 4 s per kernel and case); median, with P10–P90. Every timed run starts with the L2 flushed: a 96 MB memset outside the timed interval. `warm` = the default shape without flushing.
- **TFLOPS:** the attention matmuls under the causal mask, 4 x 32 heads x 64 x (pairs of query and visible key) per layer, over the median time
- **Clocks:** NVML after every timed run: SM clock median (minimum), temperature, power, and clock-event reasons. Power or thermal capping in 32 and 0 of 1175 runs.

## Batch 1

| Length | GFLOP | naive (ms) | tiled 64x32 (ms) | P10–P90 | TFLOPS | tiled 128x64 (ms) | TFLOPS | speed-up (best) | warm L2 64x32 (ms) | SM MHz, 64x32 | max °C | W | x 16 layers, best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.02 | 0.132 | 0.032 | 0.019–0.033 | 0.54 | 0.040 | 0.43 | 4.2x | 0.027 | 2655 (2655) | 51 | 18.7 | 0.51 ms |
| 128 | 0.07 | 0.427 | 0.053 | 0.041–0.054 | 1.27 | 0.059 | 1.14 | 8.0x | 0.042 | 2655 (2655) | 51 | 21.0 | 0.85 ms |
| 256 | 0.27 | 1.564 | 0.097 | 0.085–0.098 | 2.77 | 0.108 | 2.51 | 16.1x | 0.082 | 2655 (2655) | 52 | 24.2 | 1.56 ms |
| 512 | 1.08 | 6.318 | 0.257 | 0.244–0.258 | 4.19 | 0.270 | 3.98 | 24.6x | 0.233 | 2655 (2655) | 52 | 32.7 | 4.11 ms |
| 1024 | 4.30 | 24.894 | 0.835 | 0.822–1.039 | 5.15 | 0.826 | 5.20 | 30.1x | 0.797 | 2655 (2655) | 58 | 47.0 | 13.22 ms |
| 2048 | 17.19 | 101.570 | 3.045 | 3.041–3.266 | 5.64 | 2.922 | 5.88 | 34.8x | 3.109 | 2655 (2655) | 65 | 69.1 | 46.76 ms |

## Batch 8

| Length | GFLOP | naive (ms) | tiled 64x32 (ms) | P10–P90 | TFLOPS | tiled 128x64 (ms) | TFLOPS | speed-up (best) | warm L2 64x32 (ms) | SM MHz, 64x32 | max °C | W | x 16 layers, best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.14 | 0.979 | 0.118 | 0.105–0.119 | 1.16 | 0.159 | 0.86 | 8.3x | 0.081 | 2655 (2655) | 65 | 44.7 | 1.88 ms |
| 128 | 0.54 | 3.614 | 0.232 | 0.219–0.236 | 2.33 | 0.273 | 1.98 | 15.5x | 0.182 | 2655 (2655) | 59 | 33.0 | 3.72 ms |
| 256 | 2.16 | 13.136 | 0.581 | 0.566–0.778 | 3.71 | 0.657 | 3.28 | 22.6x | 0.520 | 2655 (2655) | 59 | 39.3 | 9.29 ms |
| 512 | 8.61 | 50.917 | 1.818 | 1.810–2.035 | 4.74 | 1.903 | 4.52 | 28.0x | 1.793 | 2655 (2655) | 64 | 58.4 | 29.08 ms |
| 1024 | 34.39 | 201.916 | 6.903 | 6.446–7.556 | 4.98 | 7.465 | 4.61 | 29.3x | 6.857 | 2655 (2655) | 66 | 54.6 | 110.44 ms |
| 2048 | 137.51 | 814.446 | 28.046 | 27.731–28.625 | 4.90 | 25.991 | 5.29 | 31.3x | 27.954 | 2655 (2655) | 70 | 52.4 | 415.86 ms |

Naive TFLOPS for reference: batch 1 x 64 0.129; batch 1 x 128 0.158; batch 1 x 256 0.172; batch 1 x 512 0.170; batch 1 x 1024 0.173; batch 1 x 2048 0.169; batch 8 x 64 0.139; batch 8 x 128 0.150; batch 8 x 256 0.164; batch 8 x 512 0.169; batch 8 x 1024 0.170; batch 8 x 2048 0.169.

Tiled 64 x 32 against naive, largest absolute difference in any output element: 0.0005 (the tests check every case against an FP32 reference).
