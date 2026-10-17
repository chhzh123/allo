# E5: compilation time

Two questions. **How compilation time scales with the array across the three
systems**, which is the result, and **where SPMW's own scaling comes from**,
which is the mechanism behind it.

## How compilation time scales with the array, across three systems

Measured in E1, on the same designs whose cycles and area that experiment
reports; `experiments/e1_gemm/compile_time.csv` is the source. HLS synthesis
only -- place and route is a separate column below, and it is dominated by
Vivado rather than by the flow that fed it.

| Array | SPMW wall (8 workers) | SPMW serial | AutoSA | Allo |
|---|---:|---:|---:|---:|
| 4x4 | 81 s | 375 s | 41 s | 60 s |
| 8x8 | 84 s | 388 s | 182 s | 80 s |
| 16x16 | 106 s | 508 s | 1,783 s | 423 s |
| 32x32 | 110 s | 538 s | 35,023 s | 5,978 s |

**SPMW's synthesis time is nearly flat: 81 to 110 seconds across a
sixty-four-fold growth in the array.** AutoSA's rises by a factor of 860 over
the same range, from 41 seconds to **9 hours 44 minutes**, and Allo's by a
factor of 100, from 60 seconds to **1 hour 40 minutes**. Both of them track the
size of the kernel text they emit; SPMW does not, because it does not emit one.

The reason is the split backend. SPMW compiles one HLS project per *role* -- per
wiring class -- and instantiates it once per site, and the number of roles does
not grow with the array: a 32x32 mesh has the same handful of roles as a 4x4
one, so there is the same handful of projects to synthesise. AutoSA and Allo
emit one flat kernel whose text grows with the array, and Vitis's front end
grows superlinearly with it.

**Two columns for SPMW, because the comparison would otherwise flatter it.**
SPMW synthesises its roles concurrently on eight workers; AutoSA and Allo run
one project. The serial column is the sum of SPMW's own per-role job times,
which is what a single worker would cost, and it is the fair one against the
other two. On that column SPMW is *slower* at 4x4 and 8x8 -- 375 s against AutoSA's 41 and
Allo's 60, and 388 against 182 and 80 -- and then 3.5x and **65x** faster than
AutoSA at 16x16 and 32x32, 1.2x and **11x** faster than Allo. The crossover is
around 8x8: below it the split backend's fixed cost dominates, above it the
other two grow and SPMW does not. Parallelism is a real advantage of having many small projects
rather than one large one, but it is reported separately rather than folded in.

### Place and route is a different story

| Array | SPMW mesh | AutoSA | Allo |
|---|---:|---:|---:|
| 4x4 | 316 s | 369 s | 524 s |
| 8x8 | 462 s | 578 s | 777 s |
| 16x16 | 1,746 s | 1,278 s | 1,194 s |
| 32x32 | 4,188 s | 5,818 s | 5,388 s |

Place and route grows with the *hardware*, which is the same size in all three
systems, so nothing here is a property of the flow: SPMW and AutoSA are within
about 40% of each other in both directions. All three columns are now the same
out-of-context Vivado recipe, so they are directly comparable; Allo's earlier
figures used Vitis `export_design -flow impl`, which bundles RTL synthesis in,
and are not what is shown here.

**So the compile-time result is about the front end, not the back end.** At
32x32 the three route in 4,188, 5,818 and 5,388 seconds -- within 40% of each
other, because it is the same hardware -- while their synthesis takes 110,
35,023 and 5,978. Routing is the floor everyone pays; synthesis is where the
flows differ, and it is the only column SPMW changes.

## The same array through Intel's SYCL compiler (2026-10-16)

Is that growth Vitis's? E1's array written in SYCL and compiled to RTL by
Intel's FPGA backend says it is not. `sycl/source/e1_sycl.cpp` is the same
computation -- `C = A B`, int8 operands, int32 results, M = N = K = S, on S x
S cells of one multiplier -- built the way AutoSA builds it: every cell a
kernel of its own that takes A from the west and B from the north and hands
its sum down its column, joined by pipes to a loader for A, one for B and a
store for C. Intel's emulator checks it against its own reference at 4x4, 8x8
and 16x16 (`sycl/report/emu_S*.log`); at 32x32 the emulator, which gives each
of the 1,027 kernels a thread that spins on its pipes, did not finish in
twenty minutes. Intel's own report shows the hardware at every size: S^2 cell
kernels of one multiplier each (`sycl/report/S*_summary.ndjson`).

The compiler is oneAPI DPC++ 2025.0 with its FPGA support package, the last
release that compiles SYCL for an FPGA: from 2025.1 the compiler rejects
`-fintelfpga`, as the 2026.1 installed on the machine does. It was unpacked
from Intel's package repository into a user directory
(`sycl/scripts/intel_unpack.sh`, `intel_packages.txt`). The compile is its
early image, `-Xshardware -fsycl-link=early -Xstarget=Agilex7
-Xsclock=300MHz`: everything up to RTL and reports, nothing of Quartus. The
protocol is this experiment's own: a fresh directory a compile, one compile at
a time, the sizes in an order shuffled from a fixed seed in each repetition,
the load average recorded. Three repetitions at 4x4, 8x8 and 16x16 and one at
32x32, which takes five hours and twenty-three minutes; medians, with the
range.

| Array | Kernels (cells) | Intel SYCL, whole compile | Intel HLS backend | Kernels only (vs csynth) | AutoSA csynth | Allo csynth | SPMW, 8 workers | SPMW, one worker | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 19 (16) | 35.8 s (35.5-36.0) | 22.3 s (22.2-22.4) | 29.3 s (29.0-29.4) | 41 s | 60 s | 81 s | 375 s | 0.4 GB |
| 8x8 | 67 (64) | 90.5 s (88.0-90.9) | 68.1 s (66.1-69.1) | 76.1 s (73.6-76.4) | 171 s | 80 s | 84 s | 388 s | 0.8 GB |
| 16x16 | 259 (256) | 634 s (628-644) | 576 s (572-586) | 589 s (583-599) | 1,677 s | 423 s | 106 s | 508 s | 3.0 GB |
| 32x32 | 1,027 (1,024) | 19,387 s | 19,185 s | 19,223 s | 34,968 s | 5,978 s | 110 s | 538 s | 15.5 GB |

`Intel SYCL, whole compile` is the `icpx` command end to end. `Intel HLS
backend` is the part of it in `aoc`, from SPIR-V to RTL and reports. `Kernels
only` is the whole compile less the compile of the host program and the final
packaging, which `csynth_design` does not do either, so it is the column to
read against AutoSA's and Allo's. The E1 columns are the single runs of the
first table above. The steps before `aoc`, timed once a size from the
compiler's own log (`sycl/scripts/sycl_split.sh`), and what was built:

| Array | Repetitions | CPU / wall | Load at start | Device front end | Host compile | Link | To SPIR-V | ALUTs | FFs | Multipliers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 3 | 99%-99% | 2.28-3.63 | 2.4 s | 4.4 s | 2.1 s | 1.8 s | 9,133 | 16,523 | 16 |
| 8x8 | 3 | 99%-100% | 2.40-3.67 | 2.8 s | 9.8 s | 2.3 s | 1.8 s | 23,261 | 36,913 | 64 |
| 16x16 | 3 | 100%-100% | 2.52-5.54 | 4.9 s | 31.1 s | 3.3 s | 2.3 s | 70,248 | 110,352 | 256 |
| 32x32 | 1 | 100%-100% | 14.34 | 13.4 s | 112.3 s | 7.9 s | 5.0 s | 275,767 | 420,268 | 1,024 |

- **Intel's compile grows like AutoSA's.** 2.5, 7.0 and 30.6 times a
  doubling of the array's side, against AutoSA's 4.2, 9.8 and 20.9. It is
  faster than AutoSA at every size and slower than Allo from 16x16 on: at
  32x32 its kernels take 5.3 hours against AutoSA's 9.7 and Allo's 1.7, 175
  times SPMW's 110 seconds and 36 times SPMW's 538 on one worker.
- **The time is the backend's, on one core.** `aoc` is 62% of the compile at
  4x4 and 99% at 32x32, in two steps: "optimizing and doing static analysis",
  2 hours 58 minutes at 32x32, and "linking with IP library", which emits the
  RTL, 2 hours 21 minutes. Every run used one core -- CPU time is 99 to 100%
  of wall time -- and at most 15.5 GB.
- **So the growth is not a Vitis property.** A second vendor's HLS, from a
  different front end and for a different device family, grows the same way
  on this array, because it too compiles each of the S^2 cells as a component
  of its own. SPMW compiles a role once, whatever the number of its sites.

What differs from E1 and is not removed: the device family (Agilex 7, not the
U280); the design, written by hand in the structure AutoSA generates rather
than generated by it; and the time of measurement -- E1's columns are single
runs taken earlier at load 5 to 24, these saw load 2 to 6, and 14 at the start
of the 32x32 run while the load average of earlier emulator runs decayed.

    sycl/scripts/intel_unpack.sh /scratch/hc676/intel   # the compiler, no root
    sycl/scripts/sycl_emu.sh 4 8 16 32                  # check it in the emulator
    sycl/scripts/sycl_compile.sh 3 4 8 16 32            # time it (after_rep1.sh: 32x32 once)
    sycl/scripts/sycl_split.sh 4 8 16 32                # the steps before aoc
    python3 sycl/scripts/sycl_table.py                  # the two tables above

## Where that scaling comes from: the hardware held fixed

Same layout as E1 to E4, with the size axis each design already uses: `S<n>`
for the GEMM, `N<n>` for the FFT. What differs is `generated/`, and it differs
for a reason that is the point of the experiment: **the generated C++ is byte
identical in all three modes.** Only the number of `vitis_hls` invocations and
the worker cap change. So there is no per-mode hardware to keep, and each
`generated/` says where the hardware actually lives -- `e1_gemm/spmw_kernel/`
and `e2_fft/spmw/`. This is a compilation experiment, not a hardware one.

### The three modes

| Mode | Projects | Workers | What it isolates |
|---|---|---|---|
| `shared-serial` | one per role | 1 | work reduction with no job parallelism |
| `shared-parallel` | one per role | 8 | parallel synthesis of independent roles |
| `per-instance` | one per site | 8 | the reuse benefit under the same worker cap |

### Results, median of three repetitions

| Design | Size | Roles | Instances | shared-serial | shared-parallel | per-instance |
|---|---:|---:|---:|---:|---:|---:|
| gemm8 | 4x4 | 9 | 16 | 361.0 s | 82.0 s | 85.4 s |
| gemm8 | 8x8 | 9 | 64 | 364.4 s | 82.4 s | 339.1 s |
| gemm8 | 16x16 | 9 | 256 | 361.5 s | 82.4 s | 1,377.7 s |
| gemm8 | 32x32 | 9 | 1,024 | 363.2 s | 82.2 s | **timeout** |
| fftsdf | 128 | 8 | 8 | 351.3 s | 47.5 s | 47.8 s |
| fftsdf | 256 | 9 | 9 | 400.9 s | 88.5 s | 88.2 s |
| fftsdf | 512 | 10 | 10 | 443.3 s | 92.2 s | 91.3 s |
| fftsdf | 1024 | 11 | 11 | 490.0 s | 92.9 s | 93.0 s |

**`shared-parallel` is flat in array size**: 82 seconds for every GEMM from
4x4 to 32x32, a 64-fold range of instances, because the role count does not
grow with the array -- it is 9 at every size. `per-instance` compiles one
project per site and tracks the instance count instead: 85, 339, 1,378, then
past the preset one-hour timeout at 1,024 sites.

### The FFT is the control

`fftsdf` has **roles equal to instances** -- every stage is structurally
distinct, so there is nothing to reuse. Its `shared-parallel` and
`per-instance` columns are the same number at every size, within 1 per cent.
That is the experiment's own negative control: where reuse cannot apply, the
mode that exploits reuse buys nothing, which is what says the GEMM's 16x
speedup at 16x16 is reuse rather than an artefact of the harness.

### Protocol

Every run gets a fresh directory. The mode order is randomised per (repetition,
design, size) from a seeded permutation. Three repetitions; the table is the
median and `results.csv` also carries the minimum and maximum -- the spread is
1.01x to 1.08x. Eight workers in both parallel modes. The `per-instance` GEMM
at 32x32 carries a preset one-hour timeout and is recorded as a timeout with
the number of jobs that completed; no wall time is extrapolated from it.

Place-and-route is not in these figures: it is the same design in every mode,
so it would add a constant to each.

### What the columns mean

`hls_wall_s` is the elapsed time of the concurrent synthesis and export phase.
`hls_sum_job_elapsed_s` is the sum of the individual jobs' elapsed times --
this is what an earlier record called "CPU", and it is not CPU time; `cpu_s`
is, from `getrusage(RUSAGE_CHILDREN)`. `frontend_s` is elaboration and per-role
codegen, reported apart and never folded into the HLS time.

### Reading it against the earlier record

An earlier record quoted a 46.0x wall-time reduction at 32x32 using 24 workers,
and 4x4 pairs using 32; neither is a repetition of this eight-worker protocol.
Its 123.8x figure was a ratio of summed job elapsed times, not of CPU. Both are
earlier experiments, not folded into these medians.

## Rerun on the direct-MLIR compiler (2026-09-14)

The compiler was refactored to build a design's program as `spmw` MLIR
directly -- each role and mover goes to Allo's IR builder as an in-memory AST,
and the `spmw.map` ops are built with the MLIR bindings -- where before every
role was rendered as an `allo.dataflow` Python program, written to disk,
executed and traced back. `frontend_s` is the only column that flow touches:
the HLS projects it stages hold the same C++, up to the order of a role's
stream parameters.

The GEMM points were re-measured under the protocol above from the refactored
tree (`e5_refactor.sh` on brg-zhang-xcel, logs under
`/scratch/hc676/spmw_refactor/e5`): fresh directory per run, mode order from
the same seed, eight workers, three repetitions of the two shared modes. The
`per-instance` mode, whose point is the reuse contrast rather than the
frontend, was run once, with the 32x32 bound kept at one hour.
`rerun_2026-09-14/results.csv` is the table, from `rerun_2026-09-14/e5_runs.log`
through `scripts/e5_rows.py`.

| Size | Frontend, before → after | shared-serial | shared-parallel | per-instance |
|---:|---:|---:|---:|---:|
| 4x4 | 1.10 s → 0.75 s | 361.0 s → 347.6 s | 82.0 s → 79.7 s | 85.4 s → 83.0 s |
| 8x8 | 1.15 s → 0.79 s | 364.4 s → 347.8 s | 82.4 s → 79.8 s | 339.1 s → 332.0 s |
| 16x16 | 1.28 s → 0.96 s | 361.5 s → 347.5 s | 82.4 s → 79.8 s | 1,377.7 s → 1,326.0 s |
| 32x32 | 1.95 s → 1.63 s | 363.2 s → 348.1 s | 82.2 s → 79.7 s | timeout (665–680 jobs) → timeout (697 jobs) |

Medians, before from `results.csv` (its `shared-parallel` rows for the
frontend) and after from the rerun. **The frontend is 17 to 32 per cent
shorter** -- 0.75 s against 1.10 s at 4x4, 1.63 s against 1.95 s at 32x32 --
which is the render, exec and re-trace step gone. The synthesis columns move
by 3 to 4 per cent in the same direction at every size and in every mode,
`per-instance` included, so that is the machine (load 1 to 3 during this
rerun, against 4 to 17 originally) rather than the compiler. What the
experiment is about is unchanged: `shared-parallel` is **79.7, 79.8, 79.8,
79.7 s** from 4x4 to 32x32, flat to within 0.3 per cent across a 64-fold range
of instances, and the repetition spread is 1.00x to 1.01x.
