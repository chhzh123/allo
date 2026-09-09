# E1 compile time: what was recorded, and what it does not support

`compile_time.csv` collects every wall-clock figure the E1 builds happen to
have written, from `cost.json` (SPMW), `stages.txt` (Allo) and the build logs
(AutoSA). It is a record, not a controlled comparison, and the difference
matters: **do not quote a cross-framework ratio from this file.**

## Why it is not a comparison

- **The stages are not the same work.** SPMW's figure is `csynth` plus
  `export_design`; Allo's `csynth` excludes both its cosimulation and its
  implementation, which are separate rows; AutoSA's single figure is `csynth`
  and `cosim` together. Three different quantities in one column.
- **The concurrency is not the same.** SPMW compiles one project per role and
  runs up to eight of them at once, which is a property of the backend rather
  than of the measurement; Allo and AutoSA are one `vitis_hls` each. The
  `hls_sum_of_jobs` rows give SPMW's work-neutral figure for that reason.
- **They were taken at different times on a shared machine**, over two days, at
  load averages between about 5 and 24, one measurement each with no
  repetitions. E5 measured this properly for SPMW alone -- fresh directories,
  randomised mode order, three repetitions, median with range -- and found a
  spread of only 1.01x to 1.08x, but it deliberately waited for a quiet
  machine, so that stability does not transfer to these numbers.

## What it does show

One effect is large enough to survive all of that in direction, though not in
its exact ratio:

| Array | SPMW kernel, HLS | SPMW, sum of jobs | Allo csynth | AutoSA csynth+cosim |
|---|---:|---:|---:|---:|
| 4x4 | 91.8 s | 687 s | 60 s | 87 s |
| 8x8 | 99.0 s | 737 s | 80 s | 284 s |
| 16x16 | 105.4 s | 789 s | 423 s | 2,533 s |
| 32x32 | 110.6 s | 821 s | -- | -- |

SPMW's synthesis time is nearly flat as the array grows -- 1.2x across a 64x
increase in elements -- because it compiles one project per *role* and the role
count does not grow with the array: 15 roles at every size. The work-neutral
column is flat for the same reason, so this is not an artefact of the eight
workers. Allo and AutoSA compile the whole array as one program, and their
times grow with it.

At 4x4 SPMW is the *slowest* of the three by this measure, and it stays that
way work-neutrally until the array is large. The crossover, not the ratio, is
the claim these numbers can support.

Place-and-route grows for everyone, because Vivado sees a netlist that grows
whatever the frontend did: the SPMW kernel goes 335 s to 5,247 s from 4x4 to
32x32.

## What a real measurement would need

The E5 protocol, applied across frameworks rather than across SPMW's own modes:
identical stage coverage per row (frontend, csynth, cosim, P&R reported
separately), a fixed worker cap declared per framework, fresh directories,
randomised order, at least three repetitions, medians with ranges, and the load
average recorded at the start and end of each run. None of that can be
reconstructed from the logs here; it has to be re-run.
