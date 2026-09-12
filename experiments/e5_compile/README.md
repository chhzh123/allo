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
| 16x16 | 106 s | 508 s | **1,783 s** | 423 s |
| 32x32 | 110 s | 538 s | **35,023 s** | not measured |

**SPMW's synthesis time is nearly flat: 81 to 110 seconds across a
sixty-four-fold growth in the array.** AutoSA's rises by a factor of 860 over
the same range, from 41 seconds to **9 hours 44 minutes**. Allo sits between
them where it was measured.

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
other two. On that column SPMW is *slower* at 4x4 and 8x8 -- 375 against 41 and
388 against 182 -- and 3.5x then **65x faster** at 16x16 and 32x32. The crossover
is around 8x8. Parallelism is a real advantage of having many small projects
rather than one large one, but it is reported separately rather than folded in.

### Place and route is a different story

| Array | SPMW mesh | AutoSA | Allo |
|---|---:|---:|---:|
| 4x4 | 316 s | 369 s | 1,124 s |
| 8x8 | 462 s | 578 s | 1,280 s |
| 16x16 | 1,746 s | 1,278 s | not measured |
| 32x32 | 4,188 s | not measured | not measured |

Place and route grows with the *hardware*, which is the same size in all three
systems, so nothing here is a property of the flow: SPMW and AutoSA are within
about 40% of each other in both directions. Allo's column is Vitis
`export_design -flow impl`, which includes RTL synthesis, so it is not directly
comparable to the other two and is shown for completeness.

**So the compile-time result is about the front end, not the back end.** At
32x32 SPMW spends 110 seconds of synthesis against 4,188 of routing; AutoSA
spends 35,023 against a routing run that was never completed.

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
