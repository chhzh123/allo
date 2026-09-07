
### gpt2-medium (/scratch/hc676/e3_block_timed2, 4 reps, seed 0)

624 launches a block; mismatched values across all reps: 0.

| stage | where | launches | kernel ms | load ms | read ms | pack ms | host ms |
|---|---|---:|---:|---:|---:|---:|---:|
| ln1 | host | 0 | 0.00 | 0.00 | 0.00 | 0.0 | 1.46 |
| Q | device | 16 | 3.51 | 6.88 | 3.95 | 171.4 | 0.00 |
| K | device | 16 | 3.46 | 6.92 | 3.73 | 131.0 | 0.00 |
| V | device | 16 | 3.45 | 6.71 | 3.71 | 125.2 | 0.00 |
| score | device | 32 | 3.17 | 12.71 | 6.57 | 237.4 | 0.00 |
| mask | host | 0 | 0.00 | 0.00 | 0.00 | 0.0 | 1.42 |
| smax | device | 128 | 8.57 | 50.55 | 23.43 | 934.7 | 0.00 |
| ssum | device | 128 | 9.77 | 50.77 | 22.97 | 931.2 | 0.00 |
| snorm | device | 128 | 10.21 | 50.83 | 23.12 | 933.0 | 0.00 |
| ctx | device | 16 | 1.72 | 6.35 | 3.15 | 118.6 | 0.00 |
| O | device | 16 | 3.36 | 6.40 | 3.33 | 107.8 | 0.00 |
| res1 | host | 0 | 0.00 | 0.00 | 0.00 | 0.0 | 0.30 |
| ln2 | host | 0 | 0.00 | 0.00 | 0.00 | 0.0 | 0.89 |
| FFN1 | device | 64 | 13.64 | 26.64 | 14.53 | 460.9 | 0.00 |
| FFN2 | device | 64 | 11.86 | 25.97 | 12.59 | 492.1 | 0.00 |
| res2 | host | 0 | 0.00 | 0.00 | 0.00 | 0.0 | 0.30 |

Means over reps: device kernel 72.7 ms, transfers 371.8 ms, packing 4.64 s, host math 53.0 ms, wall 13.08 s (best kernel 72.3 ms).

### llama-7b (/scratch/hc676/e3_llama_dev, 2 reps, seed 0)

4064 launches a block; mismatched values across all reps: 0.

| stage | where | launches | kernel ms | load ms | read ms | pack ms | host ms |
|---|---|---:|---:|---:|---:|---:|---:|
| ln1 | host | 0 | 0.00 | 0.00 | 0.00 | 0.0 | 6.40 |
| Q | device | 256 | 55.39 | 95.67 | 55.28 | 1815.8 | 0.00 |
| K | device | 256 | 55.21 | 97.08 | 55.53 | 1826.1 | 0.00 |
| V | device | 256 | 55.20 | 95.31 | 56.42 | 1828.5 | 0.00 |
| score | device | 64 | 6.76 | 22.76 | 12.76 | 457.4 | 0.00 |
| mask | host | 0 | 0.00 | 0.00 | 0.00 | 0.0 | 2.52 |
| smax | device | 256 | 16.84 | 90.48 | 45.43 | 1806.1 | 0.00 |
| ssum | device | 256 | 19.30 | 90.77 | 45.32 | 1798.5 | 0.00 |
| snorm | device | 256 | 20.13 | 92.10 | 45.46 | 1806.2 | 0.00 |
| ctx | device | 64 | 6.68 | 22.97 | 13.20 | 457.5 | 0.00 |
| O | device | 256 | 54.39 | 93.38 | 55.05 | 1862.5 | 0.00 |
| res1 | host | 0 | 0.00 | 0.00 | 0.00 | 0.0 | 1.46 |
| ln2 | host | 0 | 0.00 | 0.00 | 0.00 | 0.0 | 3.30 |
| FFN1 | device | 688 | 148.08 | 285.10 | 156.18 | 5536.9 | 0.00 |
| FFN3 | device | 688 | 147.53 | 263.18 | 136.04 | 4598.5 | 0.00 |
| FFN2 | device | 768 | 129.72 | 292.70 | 163.34 | 5057.5 | 0.00 |
| res2 | host | 0 | 0.00 | 0.00 | 0.00 | 0.0 | 1.20 |

Means over reps: device kernel 715.2 ms, transfers 2381.5 ms, packing 28.85 s, host math 38.8 ms, wall 138.06 s (best kernel 704.6 ms).
