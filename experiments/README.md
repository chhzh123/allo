# Experiments

One folder per experiment, holding the sources that produced its numbers, and
one folder holding the numbers themselves.

| Folder | What it holds |
|---|---|
| `e1_gemm/` | int8 GEMM arrays: the SPMW designs, AutoSA's input programs with the generated code and exact commands, and Allo's generator with its emitted kernels. **Read its README first**: the three systems do not compute the same workload. |
| `e2_fft/` | the folded radix-2 FFT, a single-path delay-feedback pipeline |
| `e3_tpu/` | the mini-TPU stage engine, and the host orchestrator that chains a whole GPT-2 medium or LLaMA-7B decoder layer on one bitstream |
| `e4_feather/` | the FEATHER port, and the bench and runner that drive FEATHER's own RTL for comparison |
| `e5_compile/` | the compilation-time harness: one project per role against one per site, with the worker cap and child CPU time |
| `e6_attention/` | grouped against conventional attention on the same hardware budget |
| `e7_loc/` | the language-aware source counter |
| `agentstudy/` | the agentic-design study: task, testbench, vectors, three documentation packs, three build-and-grade arms, the agent loop, and the pre-registration |
| `results/` | every package's `results.csv`, `results.json` and `README.md`, the bundle's `coverage.md`, `manifest.json` and `REPORT.md`, and the collectors that produced them |

## What is not here

The tool reports themselves, meaning the Vivado utilisation and timing reports,
the simulation waveforms and the build logs each row was read from. They are
752 MB and live on brg-zhang-xcel at
`/scratch/hc676/spmw_eval_remaining_2026-09-06`, under each package's
`reports/` and `validation/`. Every row in a `results.csv` names its report
directory in the `report_paths` column, so a number can be traced back to the
file it came from.

## Reproducing

Each package's README gives the exact commands. They all need the environment
on that machine: Vitis HLS and Vivado 2023.2, the U280 part, and for the board
runs an installed XRT with the packaged bitstream.
