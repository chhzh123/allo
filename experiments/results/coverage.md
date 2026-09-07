# Coverage: what the plan asked for, and what this bundle holds

Assembled 2026-09-07T13:32:31 on brg-zhang-xcel.ece.cornell.edu. 341 rows, 282 of them passing. Every row is as its package wrote it; a status other than `pass`/`ok` carries its own `failure_reason`, and no value here is estimated.

| Experiment | Asked for | Package | Rows | Passing | Statuses |
|---|---|---|---:|---:|---|
| E1 | int8 GEMM at 4/8/16/32 for SPMW, AutoSA and Allo: cosimulation cycles, routed resources, timing | `e1_gemm/spmw` | 12 | 8 | pass=8, unsupported=4 |
| E1 | int8 GEMM at 4/8/16/32 for SPMW, AutoSA and Allo: cosimulation cycles, routed resources, timing | `e1_gemm/spmw_mem` | 12 | 8 | pass=8, unsupported=4 |
| E1 | int8 GEMM at 4/8/16/32 for SPMW, AutoSA and Allo: cosimulation cycles, routed resources, timing | `e1_gemm/autosa` | 8 | 8 | pass=8 |
| E1 | int8 GEMM at 4/8/16/32 for SPMW, AutoSA and Allo: cosimulation cycles, routed resources, timing | `e1_gemm/allo` | 16 | 8 | not_run_impl=2, pass=8, timeout=4, timeout_cosim=2 |
| E2 | radix-2 FFT at 128/256/512/1024 for SPMW, HP-FFT and Allo: latency, interval, routed resources | `e2_fft/spmw` | 8 | 8 | pass=8 |
| E2 | radix-2 FFT at 128/256/512/1024 for SPMW, HP-FFT and Allo: latency, interval, routed resources | `e2_fft/hpfft` | 73 | 49 | not_run=22, ok=49, timing_fail=2 |
| E2 | radix-2 FFT at 128/256/512/1024 for SPMW, HP-FFT and Allo: latency, interval, routed resources | `e2_fft/allo` | 20 | 20 | ok=20 |
| E3 | complete GPT-2 medium and LLaMA-7B blocks chained on one bitstream, with its resources and timing | `e3_tpu` | 2 | 2 | pass=2 |
| E4 | FEATHER GEMM and convolution with general weights, SPMW against the original RTL, plus P&R | `e4_feather` | 66 | 60 | fail=4, not_run=2, pass=60 |
| E5 | compilation time: shared/serial, shared/parallel and per-instance, three repetitions, randomised order | `e5_compile` | 24 | 23 | pass=23, timeout=1 |
| E6 | grouped versus conventional attention on the same hardware budget and the same workload | `e6_attention` | 80 | 80 | ok=80 |
| E7 | language-aware recount of design-only source size on both sides | `e7_loc` | 20 | 8 | environment_blocked=1, pass=8, unsupported=11 |

Each package directory holds its own `README.md` (what was run, with the exact commands, how each metric was measured, and its caveats), `results.csv`/`results.json`, and `reports/` with the tool reports and logs the rows were read from.
