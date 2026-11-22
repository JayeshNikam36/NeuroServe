# Prefill attention: today's kernel, varlen v0 and the tiled kernel

- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU, 20 SMs, compute capability 8.9
- **Shapes:** Llama 3.2 1B: 32 query heads, 8 KV heads, head dimension 64, 16-token pages (shuffled)
- **Timing:** CUDA events, median of 20 runs after 3 warm-up runs, one layer's attention. `today` is Day 34's `attention_prefill_paged`, one launch per sequence (what the backend runs for a prefill from a cached offset); `v0` is `attention_prefill_varlen_v0`, one launch for the pack; `tiled RxC` is `attention_prefill_varlen_tiled` with R query rows per block and C keys per K/V tile.
- **Against the step:** 16 layers of today's kernel against the whole prefill step at Day 46's 0.28 ms per token (measured up to 480 tokens; longer prompts extrapolated, since the matmuls scale linearly). A prompt prefilled from zero does not run this kernel (the engine's own prefill attention does; Day 46's flat cost per token shows it is cheap); a prefix hit, a chunk or a resumed request does. The last column is what this kernel would cost such a prefill.

| Case | new tokens | keys (max) | today, ms per layer | v0, ms per layer | today x 16 layers | prefill step (Day 46) | this kernel against the step |
|---|---|---|---|---|---|---|---|
| 1 x 64 | 64 | 64 | 0.129 | 0.129 | 2.1 ms | 17.9 ms | 12% |
| 1 x 128 | 128 | 128 | 0.426 | 0.422 | 6.8 ms | 35.8 ms | 19% |
| 1 x 256 | 256 | 256 | 1.568 | 1.559 | 25.1 ms | 71.7 ms | 35% |
| 1 x 480 | 480 | 480 | 5.869 | 5.813 | 93.9 ms | 134.4 ms | 70% |
| 1 x 1024 | 1024 | 1024 | 26.857 | 26.729 | 429.7 ms | 286.7 ms | 150% |
| 1 x 2048 | 2048 | 2048 | 108.923 | 110.145 | 1742.8 ms | 573.4 ms | 304% |
| 4 x 480 | 1920 | 480 | 24.150 | 23.704 | 386.4 ms | 537.6 ms | 72% |
| cached 1024 + 256 | 256 | 1280 | 16.961 | 14.983 | 271.4 ms | 71.7 ms | 379% |
| 8 x 32 (short prompts) | 256 | 32 | 0.329 | 0.263 | 5.3 ms | 71.7 ms | 7% |

## The tiled kernel, ms per layer

| Case | v0 | tiled 64x64 | tiled 64x32 | tiled 128x64 | tiled 128x32 | best | speed-up on today's | best x 16 layers |
|---|---|---|---|---|---|---|---|---|
| 1 x 64 | 0.129 | 0.033 | 0.027 | 0.034 | 0.035 | 64x32 | 4.8x | 0.43 ms |
| 1 x 128 | 0.422 | 0.055 | 0.042 | 0.047 | 0.048 | 64x32 | 10.1x | 0.67 ms |
| 1 x 256 | 1.559 | 0.128 | 0.083 | 0.090 | 0.093 | 64x32 | 18.9x | 1.33 ms |
| 1 x 480 | 5.813 | 0.388 | 0.230 | 0.239 | 0.245 | 64x32 | 25.5x | 3.69 ms |
| 1 x 1024 | 26.729 | 1.375 | 0.800 | 0.790 | 0.823 | 128x64 | 34.0x | 12.63 ms |
| 1 x 2048 | 110.145 | 5.632 | 3.398 | 3.102 | 3.391 | 128x64 | 35.1x | 49.63 ms |
| 4 x 480 | 23.704 | 1.514 | 0.867 | 0.906 | 0.923 | 64x32 | 27.9x | 13.87 ms |
| cached 1024 + 256 | 14.983 | 0.804 | 0.529 | 0.504 | 0.524 | 128x64 | 33.7x | 8.06 ms |
| 8 x 32 (short prompts) | 0.263 | 0.111 | 0.056 | 0.122 | 0.088 | 64x32 | 5.8x | 0.90 ms |
