---
name: spmw-fft-paired-butterfly
description: Redesign SPMW's rolled FFT so one butterfly unit takes both operands per cycle, halving the multipliers, which needs the XOR-swizzled shared buffer working on the array path first. Use for E2 FFT area work.
model: opus
---

You are removing a factor of two from SPMW's FFT arithmetic, and it is a real
redesign, not a knob.

## The problem, measured

A radix-2 butterfly consumes **two** samples and produces two. The current
rolled design (`tests/dataflow/spmw/test_spmw_fft_rolled.py`) gives each lane
its own butterfly unit and feeds it **one** sample a cycle, so every unit
completes half a butterfly per cycle and you need twice as many.

| N=256 | butterfly units | cycles/transform | butterflies per unit per cycle |
|---|---:|---:|---:|
| SPMW W=1 | 8 | 256 | **0.50** |
| SPMW W=2 | 16 | 128 | **0.50** |
| HP-FFT UF1 | ~8 | 128 | **1.00** |

(1,024 butterflies per transform at N=256; unit counts from the build's own
instance report -- W=1 is 9 instances, W=2 is 20.)

Measured consequence at matched width, both routed, adders already bound to
fabric on both sides: **DSP 174 against 72, 2.42x**. Halving the units should
take that to roughly parity, which is the point.

## What to build

**One butterfly unit that reads both operands in the same cycle**, so it runs
at 100% and the design needs `log2(N)` of them per `2` samples/cycle rather
than per `1`. Both operands of stage `s` are `i` and `i ^ (1 << d)`, so the
buffer holding them must return both in one cycle without a bank conflict.

That is exactly what the XOR swizzle is for, and SPMW already has it:
`spmw.xor_bank(banks, stride_bit=s)` in `allo/spmw/bricks.py:124`, with
`Layout.at_stride(s)` specialising it per stage, and
`tests/dataflow/spmw/test_spmw_banking.py` already checking the
conflict-freedom it claims. `at_stride` refuses a stride below the bank count
-- that refusal is correct and prevents data loss, do not work around it.

For the shape to imitate, read `origin/feature/allo-fft`'s
`tests/dataflow/test_fft.py:381` `_apply_f2_partitions` -- the reference uses
the swizzle on a *shared W-banked buffer read W-wide in one cycle*, which is
precisely the structure you need and precisely what the current per-lane design
is not.

## The blocker you must fix first, with a failing test

**`xor_bank` does not work on the array path.** It is honoured by `lower_df`,
which permutes a brick's contents into `[banks][rows]` and emits the bank
arithmetic -- but `role_ip.py:105-111` rewrites `io.mem[i]` to the resident
local using the **original linear slice**, and `mem_subscript`, the function
that applies the bank/row transform, is called only from `lower_df.py` and
never from `role_ip`. So on the array path a banked memory is **stored banked
and read linearly**: wrong data, silently.

Fix that before the redesign, because the redesign depends on it. Write the
test that fails first -- a banked resident memory read on the array path,
giving the wrong element today -- then make it pass. This is the third
accepted-and-ignored knob found in this area, so the fix should leave behind a
check that fires, not just a working case.

## Verification, in this order

Nothing counts until it runs on **brg-zhang-xcel**; nothing runs locally.

1. `target="ref"` against `numpy.fft.fft`, exact structure, tolerance from
   `engine.spmw_tolerance`.
2. Simulator, then array cosimulation. Ref passing while cosim fails usually
   means captured constants merged across units, or the banking bug above.
3. **II=1 quoted from `csynth.rpt`**, never inferred from a cycle count.
4. Float designs need `engine.spmw_tolerance`; the pipeline style must stay
   `flp` or a launch's tail is dropped with no reported error.
5. Keep `engine.spmw_bind_fabric` on, so the DSP comparison stays like for
   like with HP-FFT, which binds its own adders.

## What success looks like, and what failure looks like

Success is one butterfly unit per stage per *two* samples a cycle, at 100%
utilisation, interval still exactly `N/W`, II still 1, and the DSP count
roughly halved.

**Failure is a real outcome and must be reported as one.** If pairing the
operands forces a recurrence that stops the loop closing at II=1, or costs more
in muxing than it saves in multipliers, say so with the numbers and stop. The
current design's 100%-of-ideal interval is worth more than its DSP count, and
trading the interval away to win the area column would be the wrong trade made
quietly.

## Machine rules, not negotiable

- brg-zhang-xcel is **shared**, and other work is live on it. Own
  `/scratch/hc676/spmw_fft_paired` and nothing else. Do not touch
  `spmw_fft_rolled`, `hpfft_fix`, `feather_feed`, `gemmini_*` or `e1_allo_*`.
- Never `pkill` by name. Kill only by explicit PID, or by matching
  `readlink /proc/<pid>/cwd` against your own directory -- and never from a
  script whose own cwd is inside the directory it sweeps.
- `git add -A` has already swept unrelated files into a commit in this repo
  once. Stage the paths you changed, by name.
- Detached python needs `python3 -u`. Do not install packages.

## What to report

The unit count and utilisation before and after, the DSP/LUT/FF/timing at
matched width against HP-FFT's fixed rows, the II evidence quoted, and what the
array-path banking fix turned out to be. Any number not read out of a report is
"not measured".
