# E5: compilation time, with the hardware held fixed

What this isolates: the split backend compiles one Vitis HLS project per
*role* (per wiring class) and instantiates it once per site. A backend
without that reuse would compile one project per *site*. This measures the
difference with the architecture unchanged: the generated C++ is byte
identical in every mode, and only the number of `vitis_hls` invocations and
the worker cap change. It is a compilation experiment, not a hardware one;
the hardware these projects describe is the hardware of E1 and E2.

## Modes

| Mode | Projects | Workers | What it isolates |
|---|---|---|---|
| `shared-serial` | one per role | 1 | work reduction with no job parallelism |
| `shared-parallel` | one per role | 8 | parallel synthesis of independent roles |
| `per-instance` | one per site | 8 | the reuse benefit under the same worker cap |

## Protocol

`scripts/spmw_ablate_compile.py` measures one (design, size, mode) point and
writes one `E5 key=value ...` line. `/scratch/hc676/e5_run.sh` drives it:

- Every run gets a **fresh directory**, removed afterwards, so no run reuses
  another's staged projects or HLS caches.
- The **mode order is randomised** per (repetition, design, size) from a
  seeded permutation, because the earlier ablation always ran reuse first.
- **Three repetitions** of every point; `results.csv` reports the median with
  the minimum and maximum, never a single run.
- Eight workers throughout, the same cap in every parallel mode.
- The `per-instance` mode at the largest GEMM carries a **preset one-hour
  timeout**; a point that hits it is recorded as a timeout with the number of
  jobs that completed. No wall time is ever extrapolated from a partial run.
- The driver waits for the E1 and E2 sweeps to finish before starting, so it
  does not share the machine with its own designs' place-and-route runs. The
  load average at the start and end of each run is recorded, because the
  machine is shared and this is elapsed time.

## What each number means

- `frontend_s` -- elaboration and per-role code generation, before any tool
  runs. Reported apart, never folded into the HLS time.
- `hls_wall_s` -- elapsed time of the concurrent `csynth_design` and
  `export_design` phase. IP export is inside this figure, as it is in E1's.
- `hls_sum_job_elapsed_s` -- the sum of the individual jobs' own elapsed
  times. This is what the earlier record called "CPU"; it is not CPU time,
  and the two are reported separately here for that reason.
- `cpu_s` -- real processor time of the tool processes, user plus system,
  from `getrusage(RUSAGE_CHILDREN)` around the phase.
- `projects`, `roles`, `instances` -- the number of HLS projects the mode
  compiled, the number of distinct kernels, and the number of hardware
  instances they stand for.

Place-and-route time is not in these figures. It is the same design in every
mode, so it would add a constant to each; E1 and E2's `results.csv` carry it
per stage for the designs measured here.

## Reading it against the earlier record

The earlier 46.0x wall-time reduction at 32x32 used 24 workers, and the
earlier 4x4 pairs used 32; neither is a repetition of this eight-worker
protocol. The 123.8x figure was a ratio of summed job elapsed times, not of
CPU consumption. Both are kept as separate, earlier experiments rather than
folded into these medians.

## Files

`results.csv` and `results.json` (one row per design, size and mode, with the
median and range over three repetitions), `reports/<design>_<size>_<mode>_r<n>/`
(each run's stdout, its `e5.json`, and one role's synthesis report as
evidence that the projects really were built), and `e5_runs.log`, the raw
lines every run wrote.
