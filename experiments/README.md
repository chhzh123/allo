# Experiments

The evaluation of *SPMW: A Grid-Based Programming Model for Scalable
Accelerator Design* (Section 6 of the arXiv version), one folder a figure or
table, laid out as the paper is. Each folder holds what its numbers were read
from: the design source, what the compilers generated, the logs and reports,
the table of values the figure or table uses (`results.csv`, or the record
files its README names), and the scripts that drove the runs. Its README says what the numbers mean and what they do
not.

| Paper | Folder | What is compared |
|---|---|---|
| **6.1.1 Compute and connectivity** | | |
| Fig. 9(a) | [`performance/gemm_workloads/`](performance/gemm_workloads/) | one 16x16 array on 64^3 to 512^3 int8 GEMMs: SPMW, AutoSA, Allo |
| Fig. 9(b), Table 1 (GEMM) | [`performance/gemm_arrays/`](performance/gemm_arrays/) | D^3 GEMMs on D x D arrays, 4x4 to 32x32: SPMW, AutoSA, Allo |
| Fig. 9(c,d), Table 1 (FFT) | [`performance/fft/`](performance/fft/) | radix-2 FP32 FFTs at 256 and 1,024 points, 2 to 16 samples a cycle: SPMW, HP-FFT, Spiral |
| **6.1.2 Composability** | | |
| Fig. 10 | [`performance/mini_tpu/`](performance/mini_tpu/) | whole programmable engines on LLaMA and DeepSeek gate/up slices: SPMW mini-TPU, Gemmini, VTA |
| Fig. 12 | [`performance/feather/`](performance/feather/) | FEATHER's GEMM and convolution: its RTL, the RTL with a row-wise weight loader, SPMW |
| **6.2 Compilation and simulation time** | | |
| Fig. 13(a,b) | [`compilation/compile_time/`](compilation/compile_time/) | build time of the GEMM arrays; source to RTL: SPMW, AutoSA (TAPA and Vitis HLS), Allo, SYCL |
| Fig. 13(c) | [`compilation/simulation_time/`](compilation/simulation_time/) | the 16x16 mini-TPU: SPMW's simulator, Vitis HLS C simulation, xsim |
| **6.3 Architectural exploration** | | |
| Table 2 | [`exploration/attention/`](exploration/attention/) | a whole attention head's P V, original against grouped columns |
| **6.4 Productivity** | | |
| Table 3 | [`productivity/code_size/`](productivity/code_size/) | design-only source lines |
| Fig. 14 | [`productivity/agent_study/`](productivity/agent_study/) | five coding models writing an 8x8 array in SPMW, C++ HLS and SystemVerilog |

## Layout

    <folder>/
        README.md          what the numbers mean, and what they do not
        results.csv        the rows the figure or table uses, where there is one
        <framework>/<point>/
            source/        the design input, and the exact command or tcl
            generated/     what the compiler emitted for it
            report/        synthesis, simulation and place-and-route reports
        scripts/           what drove the runs

`<point>` is the axis a folder varies: `S<n>` for an n x n array, `S16_W<side>`
for a GEMM of that side on the 16x16 array, `N<n>` or `W<w>` for a transform
length or width. A folder that changes only how a fixed design is compiled or
simulated keeps its timing records rather than copies of the design. Where a
design is hand-written rather than generated (HP-FFT's HLS, FEATHER's RTL),
`generated/` is absent or holds a note.

## What is not here

- **Missing builds**, named in their folders' READMEs. These are the wide AutoSA 16x16 and 32x32
  builds of Fig. 9(b) and Table 1 (`results.csv` rows only), the raw
  reports of SPMW's wider 256-point FFT lanes and of HP-FFT UF2 at 256
  points, FEATHER's corrected RTL (only the patches are here), and the
  attention runs' generated code.
- **The Discussion's floorplanner result** (a 32x32 matrix-vector engine,
  346 to 457 MHz) has no folder.
- **Everything the paper does not use**: runs, variants and experiments
  that were explored and left out. They are in the history: commit `c20b2ee8` is the tree before this
  layout.

## Reproducing

Everything was built on brg-zhang-xcel (two Xeon Silver 4214, 48 threads,
shared) with Vitis HLS and Vivado 2023.2 for the Alveo U280. The SYCL compile
times use oneAPI DPC++ 2025.0 for Agilex 7. Each README gives the commands.
Most drivers name their working directories under `/scratch/hc676` on that
machine and the repository checkout they build from, so they are a record of
what ran as much as a recipe.
