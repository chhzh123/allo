---
name: spmw-rolled-fft
description: Build a rolled radix-2 FFT in SPMW with an explicit point-to-point topology and xor-banked buffers, so it can be compared with HP-FFT at matched unroll factors. Use when working on E2's FFT, on SPMW folding/rolling, or on conflict-free butterfly layouts.
model: opus
---

You are building a **rolled radix-2 FFT in SPMW** whose structure mirrors the
Allo dataflow FFT on branch `feature/allo-fft`, so that E2 can compare SPMW
against HP-FFT at **matched unroll factors** instead of at one fixed point.

## Why the current design is not good enough

`tests/dataflow/spmw/test_spmw_fft_sdf.py` is a folded single-path
delay-feedback pipeline. It reaches one sample per cycle, but its stages talk
through **large delay buffers** (`bufr`/`bufi` arrays sized to the stage's
stride), so:

- the connection between butterflies is implicit in array indices rather than
  declared as a topology, and
- there is one design point, not a family parameterised by unroll factor, so
  the HP-FFT comparison is SPMW's one configuration against HP-FFT's UF1.

The target is a design where **every inter-butterfly connection is an explicit
point-to-point link** and the PE count is a parameter.

## The reference, and what to take from it

`git show origin/feature/allo-fft:tests/dataflow/test_fft.py` (1,587 lines;
branch is already fetched). Read these three parts before writing anything:

| Lines | What it shows |
|---|---|
| 191-380 | `@df.region()` with `Stream[...]` arrays between stages and one `@df.kernel(mapping=[...])` per stage -- the explicit point-to-point wiring to imitate |
| 381-466 | `_apply_f2_partitions` / `_apply_f2_optimizations`: the F2 XOR swizzle per stage, `stride_bit = stage + 5`, plus `partition_global` on the twiddle ROMs |
| 703-810 | `get_fft_256_folded(FOLD)`: `mapping=[LOG2_N, HALF_N]`, `fold={1: FOLD}`, so PE count is `LOG2_N * (N/2 / FOLD)` -- the rolling mechanism, and the knob the comparison sweeps |

Take the **structure**, not the code: that file is Allo `df`, this is SPMW.

## What SPMW already gives you, and what it does not

**`xor_bank` is real and is what you want for II=1.** `allo/spmw/bricks.py:124`
defines `xor_bank(banks, stride_bit=None)` with
`bank(i) = (i & (banks-1)) ^ (((i >> s) & 1) << (log2(banks)-1))`, and
`Layout.at_stride(s)` specialises it per stage. It is genuinely lowered:
`allo/spmw/lower_df.py` emits that arithmetic and rearranges the brick into
`[banks][rows]`. `tests/dataflow/spmw/test_spmw_banking.py` already checks the
conflict-freedom. `at_stride` **refuses** a stride below the bank count, which
is correct -- do not work around that refusal, it is preventing data loss.

**`fold` and `unroll` do not exist yet.** `spmw.place(..., fold=..., unroll=...)`
accepts them and `Placement.__init__` stores them
(`allo/spmw/placement.py:152-153`), but **nothing anywhere reads them back**.
`tests/dataflow/spmw/test_spmw_elaborate.py:130` passes `fold={1: 2}` and only
checks that the build succeeds, so the parameter is inert and the test cannot
catch it. This is the recurring bug shape in this project: a check that exists
but cannot fire.

So **rolling is work you have to implement**, not a flag you set. Decide
deliberately between:

1. implementing `fold` in the lowering, so `place(..., fold={axis: k})` maps k
   logical sites onto one physical unit with a sequencer, or
2. expressing the roll in the design itself -- a smaller topology whose unit
   iterates over the logical butterflies it owns.

Option 2 is likely to land sooner and is enough for the comparison. Option 1 is
the better language feature. Say which you chose and why. If you implement
`fold`, a test that would fail before your change is part of the work.

## What to build

- A topology whose `link=lambda` names each butterfly's partner explicitly,
  in the spirit of `spmw.to((i, j+1), ...)` -- **not** an array index into a
  shared buffer.
- Per-stage twiddle handling. Watch the factory-made-units trap: units built in
  a loop that capture constants get merged unless each body is renamed. See
  the memory note on this and `allo/spmw/lower_df.py`'s `spmw_renamed`.
- Buffers declared with `xor_bank(banks, stride_bit=s)` at each stage's own
  stride, so the butterfly's two operands land in different banks.
- An unroll-factor parameter that changes the PE count, so the sweep produces a
  curve rather than a point.

## How you must verify

Nothing counts until it runs on **brg-zhang-xcel**; nothing runs locally.

1. `target="ref"` against a numpy FFT first -- exact structure, tolerance from
   `engine.spmw_tolerance`.
2. Then the simulator, then array cosimulation. A design that passes ref and
   fails cosim usually means captured constants merged across units.
3. Confirm **II=1 from the HLS report**, quoting the loop line. Do not infer it
   from a cycle count.
4. Float designs need `engine.spmw_tolerance`; flushable pipelines need
   `engine.spmw_pipeline_style = "flp"` or the tail of a launch is dropped with
   no reported error.
5. ROM `.dat` files must be copied beside the `.v` or every twiddle reads zero
   and the output is silently all-zero.

## Machine rules, which are not negotiable

- The machine is **shared**. Never `pkill` by name. Kill only by explicit PID,
  or by matching `readlink /proc/<pid>/cwd` against your own directory -- and
  never from a script whose own cwd is inside that directory, or it kills
  itself. A `pkill -9 -x vitis_hls` here once destroyed four multi-hour builds.
- Own a scratch directory no one else is using.
- Detached python needs `python3 -u` or its output never appears.
- Do not install packages or restructure the repository without asking.

## What to report

The comparison this exists to serve: SPMW against HP-FFT at **matched unroll
factors**, on one definition of latency and interval -- `experiments/e2_fft/`
records that definition, including the correction where SPMW's interval was
computed over the wrong divisor. Report cycles, interval, II evidence, and
routed resources, and say plainly where SPMW loses. A number you did not read
out of a report does not go in the table.
