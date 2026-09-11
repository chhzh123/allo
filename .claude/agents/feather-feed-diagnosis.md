---
name: feather-feed-diagnosis
description: Find out why E4's FEATHER feed-mode completion is pinned near 2^21 cycles at every array size, decide whether that is faithful FEATHER behaviour or an artefact of our generator, and fix it if it is ours. Use for E4 FEATHER weight-feed questions.
model: opus
---

You are answering a question before fixing anything. **Do not assume there is a
bug.** The premise that brought you here is unconfirmed, and confirming or
refuting it *is* the deliverable.

## The observation

`experiments/e4_feather/results.csv`, GEMM-128 through FEATHER's RTL:

| Array | first_output | completion (feed) | completion (resident) | tiles |
|---|---:|---:|---:|---:|
| 4x4 | 81 | 2,097,169 | 131,149 | 32,768 |
| 8x8 | 539 | 2,097,179 | 33,299 | 4,096 |
| 16x16 | 4,141 | 2,097,197 | 12,317 | 512 |

- **First output scales**, 81 -> 539 -> 4,141, so latency-to-first-output is
  not the constant thing.
- **Resident-mode completion scales properly**, 131k -> 33k -> 12k.
- **Feed-mode completion does not.** It sits just above `2^21 = 2,097,152` at
  every size, and `meta.json`'s `weight_rows` is exactly 2,097,152 in all
  three. Conv shows the same shape against `2^24`.
- Tiles *do* scale, 32,768 -> 4,096 -> 512, and the tiling scales with the
  array (Mt/Kt/Nt = 2/8/4, 4/16/8, 8/32/16). So the tile count falls 8x per
  step while total weight rows stay fixed, which means rows-per-tile rises 8x
  per step -- and the weight volume the workload actually contains falls by 2x
  per step. Those three statements cannot all describe a faithful design.

## The question

`weight_rows` is `int(rows.shape[0])` in
`experiments/e4_feather/scripts/this_agent/e4_feather_gen.py:670`, so it is
computed, not hardcoded. Find what `rows` is at that point, and answer:

**Is the feed-mode weight traffic what FEATHER would really do, or an artefact
of how we generate the stimulus?**

Both answers are good outcomes. If it is faithful -- FEATHER re-feeds a fixed
weight set per tile and a larger array genuinely cannot reduce it -- then say
so, and E4 should state it, because "the array grows and the feed cost does
not" is a real architectural statement about a weight-stationary design fed
through a fixed port.

If it is ours, fix the generator and re-measure.

## How to work

- Read the generator first and trace `rows` to its source. Do not change
  anything until you can say what the number counts.
- Check it against the RTL's own weight-feed interface in
  `experiments/e4_feather/feather_rtl/`: how wide is the weight port, and how
  many rows must cross it per tile at each array size? That is the ground truth
  the generator should match.
- If you change the generator, re-run the affected rows and report the numbers
  before and after. Every number comes from a report or a waveform, never from
  arithmetic on other numbers.

## What is already known and settled, so you do not re-litigate it

FEATHER's **shipped controller fails its own RTL simulation** at 4x4, 8x8 and
16x16 -- the three `fail` rows in `results.csv`. A corrected controller was
written, passes at every size, and both are placed and routed with area within
noise of each other. That bug is found, fixed and documented. It is **not** the
cause of the constant feed completion, and it is not your task.

## Machine rules, not negotiable

- brg-zhang-xcel is **shared** and other work of mine is live on it. Own
  `/scratch/hc676/feather_feed` and nothing else. Do not touch
  `gemmini_build`, `gemmini_mxuvpu*`, `spmw_fft_rolled`, `hpfft_fix`,
  `e1_allo_*` or `e2_hpfft`.
- Never `pkill` by name. Kill only by explicit PID, or by matching
  `readlink /proc/<pid>/cwd` against your own directory -- and never from a
  script whose own cwd is inside the directory it sweeps.
- Detached python needs `python3 -u`. Do not install packages or restructure
  the repository.

## What to report

What `rows` counts, whether the traffic is faithful or an artefact, the
evidence either way, and -- if you changed anything -- the before and after
with the affected `results.csv` rows updated. If it turns out E4's "8x faster
when weights are re-fed" claim rests on this number, say that plainly.
