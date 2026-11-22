# Prefill attention: today's kernel and varlen v0

- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU, 20 SMs, compute capability 8.9
- **Shapes:** Llama 3.2 1B: 32 query heads, 8 KV heads, head dimension 64, 16-token pages (shuffled)
- **Timing:** CUDA events, median of 20 runs after 3 warm-up runs, one layer's attention. `today` is Day 34's `attention_prefill_paged`, one launch per sequence (what the backend runs for a prefill from a cached offset); `v0` is `attention_prefill_varlen_v0`, one launch for the pack.
- **Against the step:** 16 layers of today's kernel against the whole prefill step at Day 46's 0.28 ms per token (measured up to 480 tokens; longer prompts extrapolated, since the matmuls scale linearly). A prompt prefilled from zero does not run this kernel (the engine's own prefill attention does; Day 46's flat cost per token shows it is cheap); a prefix hit, a chunk or a resumed request does. The last column is what this kernel would cost such a prefill.

| Case | new tokens | keys (max) | today, ms per layer | v0, ms per layer | today x 16 layers | prefill step (Day 46) | this kernel against the step |
|---|---|---|---|---|---|---|---|
| 1 x 64 | 64 | 64 | 0.126 | 0.129 | 2.0 ms | 17.9 ms | 11% |
| 1 x 128 | 128 | 128 | 0.425 | 0.422 | 6.8 ms | 35.8 ms | 19% |
| 1 x 256 | 256 | 256 | 1.569 | 1.557 | 25.1 ms | 71.7 ms | 35% |
| 1 x 480 | 480 | 480 | 5.583 | 5.573 | 89.3 ms | 134.4 ms | 66% |
| 1 x 1024 | 1024 | 1024 | 24.890 | 24.778 | 398.2 ms | 286.7 ms | 139% |
| 1 x 2048 | 2048 | 2048 | 102.179 | 102.731 | 1634.9 ms | 573.4 ms | 285% |
| 4 x 480 | 1920 | 480 | 22.262 | 22.103 | 356.2 ms | 537.6 ms | 66% |
| cached 1024 + 256 | 256 | 1280 | 16.370 | 14.103 | 261.9 ms | 71.7 ms | 365% |
| 8 x 32 (short prompts) | 256 | 32 | 0.327 | 0.264 | 5.2 ms | 71.7 ms | 7% |
