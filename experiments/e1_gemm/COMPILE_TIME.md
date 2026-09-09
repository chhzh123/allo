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

AutoSA's `csynth` and `cosim` are now recorded **apart**, from each build's own
phase timings, because the combined figure was not comparable with SPMW's:
SPMW's number excludes cosimulation, and at 32x32 cosimulation is 13,886 s of
AutoSA's 48,868. The column below is csynth against csynth.

| Array | SPMW csynth+export (8 jobs) | SPMW sum of jobs | AutoSA csynth | AutoSA cosim |
|---|---:|---:|---:|---:|
| 4x4 | 91.8 s | 687 s | **40.8 s** | 42 s |
| 8x8 | 99.0 s | 737 s | 171 s | 108 s |
| 16x16 | 105.4 s | 789 s | 1,677 s | 851 s |
| 32x32 | 110.6 s | 821 s | **34,968 s** | 13,886 s |

Two things are true at once and both belong in any statement of this.

**SPMW's synthesis time is flat**: 91.8 to 110.6 seconds, 1.2x across a 64-fold
increase in elements, because it compiles one project per *role* and the role
count does not grow with the array -- 9 at every size. The work-neutral column
is flat for the same reason, so this is not an artefact of the eight workers.

**AutoSA is faster at 4x4 and superlinear after it**: 40.8, 171, 1,677, 34,968
seconds, roughly 4x then 10x then 21x per doubling of the array side. It starts
ahead and ends 316 times behind on wall time, 43 times behind work-neutrally.

So the honest claim is a crossover between 4x4 and 8x8, not a constant ratio,
and the size at which it happens is part of the result. The wide-interface
builds cost essentially the same as the narrow ones (182 vs 171 s at 8x8,
35,023 vs 34,968 at 32x32), so the port width is not what makes them slow.

One asymmetry remains and is not removed: SPMW's figure includes
`export_design` and AutoSA's csynth does not. That inflates SPMW's side, which
is the conservative direction for the claim above.

## What a real measurement would need

The E5 protocol, applied across frameworks rather than across SPMW's own modes:
identical stage coverage per row (frontend, csynth, cosim, P&R reported
separately), a fixed worker cap declared per framework, fresh directories,
randomised order, at least three repetitions, medians with ranges, and the load
average recorded at the start and end of each run. None of that can be
reconstructed from the logs here; it has to be re-run.
