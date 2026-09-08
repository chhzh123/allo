# Experiments

One folder per experiment. Each now holds the code and reports that reproduce
its own table, in a common layout, plus the `results.csv` that table comes
from.

## The common layout

    <experiment>/
        README.md          what the table means, and what it does not
        results.csv        the table
        <framework>/<point>/
            source/        the design input, and the exact command or tcl
            generated/     what the compiler emitted for it
            report/        synthesis, simulation and place-and-route reports
        scripts/           what drove the runs

`<point>` is whatever axis the experiment varies: `S<n>` for square arrays
(E1, E4), `N<n>` for transform lengths (E2, E5's FFT), `M<n>` for sequence
lengths (E6). Three experiments deviate, each for a stated reason in its own
README:

- **E3** has one bitstream running two workloads, so it has one design folder
  rather than one per results row.
- **E5** changes only how a fixed design is compiled -- the generated C++ is
  byte identical in all three modes -- so its `generated/` points at E1's and
  E2's rather than duplicating them.
- **E7** builds nothing; it counts source lines, so it keeps count records and
  a manifest instead of source/generated/report.

Where a framework is hand-written rather than generated (HP-FFT's HLS,
FEATHER's RTL), `generated/` holds a note saying so rather than a copy of the
source. Where generated code exists but was not kept by the original runs, it
was re-staged from the same entry point the measured builds used, and the
README says how it was checked -- E2 and E4 match their `results.csv` role
counts at every size. **E6 is the one gap**: its variants are separate drivers
rather than one registry design, so its `generated/` is empty and says why.

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

The reports each table is read from are now here, selected rather than
mirrored: 24 MB across the seven experiments, against 752 MB of full build
trees. What was left behind is named in each README -- multi-megabyte bus
traces, cosimulation `hls.log` files, tile dumps, and superseded attempts. The
full trees remain on brg-zhang-xcel at
`/scratch/hc676/spmw_eval_remaining_2026-09-06`, under each package's
`reports/` and `validation/`. Every row in a `results.csv` names its report
directory in the `report_paths` column, so a number can be traced back to the
file it came from.

## Reproducing

Each package's README gives the exact commands. They all need the environment
on that machine: Vitis HLS and Vivado 2023.2, the U280 part, and for the board
runs an installed XRT with the packaged bitstream.
