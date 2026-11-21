# Block-size sweep (Day 42)

NVIDIA GeForce RTX 4050 Laptop GPU, compute 8.9, CUDA runtime 12.4, driver API 13.0; 2026-11-21T20:37Z.
Llama-3.2-1B shape (16 layers, 8 KV heads of 64, FP16): 32 KiB of K/V per token.

## Capacity and waste

10,000 sequences, prompt log-uniform in [16, 512] tokens plus output uniform in [16, 256] (seed 42).

| Block (tokens) | Waste (padding / allocated) | Blocks per sequence | Tokens allocated per sequence | Sequences per GiB of K/V |
| --- | --- | --- | --- | --- |
| 8 | 1.25% | 35.0 | 280.2 | 116.9 |
| 16 | 2.63% | 17.8 | 284.2 | 115.3 |
| 32 | 5.35% | 9.1 | 292.3 | 112.1 |

## Decode attention time

`attention_decode_paged`, one layer, a batch of 8 per step (one launch per sequence), pages shuffled; 30 interleaved rounds of 20 steps per block size. Per step, microseconds.

| Batch | Block | Median | Range | vs 16 | Median x 16 layers |
| --- | --- | --- | --- | --- | --- |
| mix (238 324 329 182 55 60 279 354) | 8 | 282.1 | 258.3 to 361.6 | -0.1% | 4514 |
| mix (238 324 329 182 55 60 279 354) | 16 | 282.4 | 257.2 to 355.4 | +0.0% | 4518 |
| mix (238 324 329 182 55 60 279 354) | 32 | 284.9 | 257.5 to 384.5 | +0.9% | 4558 |
| 8 x 1000 tokens | 8 | 1161.2 | 1120.9 to 1189.8 | +0.3% | 18580 |
| 8 x 1000 tokens | 16 | 1157.6 | 1130.9 to 1184.5 | +0.0% | 18522 |
| 8 x 1000 tokens | 32 | 1146.1 | 1103.5 to 1187.4 | -1.0% | 18338 |

The engine's fast batched decode kernel, the one serving uses, is written for 16-token pages and is not swept; this kernel isolates what the block size changes, where each position's K and V are found.

## Reading

- **Waste:** 1.25% of allocated slots at 8 tokens, 2.63% at 16, 5.35% at 32.
- **Capacity:** 116.9, 115.3 and 112.1 sequences of the mix per GiB of K/V at 8, 16 and 32.
- **Decode attention vs 16:** -0.1% at 8 and +0.9% at 32 for the mixed batch; +0.3% and -1.0% for 8 x 1000 tokens.
- **Block-table entries:** 35.0, 17.8 and 9.1 per sequence; a prefix is shared in whole blocks, so a smaller block shares more of a prompt and a larger one less.
