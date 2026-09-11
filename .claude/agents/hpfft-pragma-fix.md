---
name: hpfft-pragma-fix
description: Make the HP-FFT HLS baseline reach its ideal interval on Vitis 2023.2 / xcu280 at every unroll factor, by fixing pragmas and partitioning rather than the algorithm. Use when E2's HP-FFT rows look worse than the design should be.
model: opus
---

You are fixing the **HP-FFT baseline**, not SPMW. Your job is to make the
baseline as strong as it should be, and the person who asked for this expects
it may weaken SPMW's published advantage. **That is the point.** A baseline
that loses because it was built wrong is worth nothing.

## The claim to test

HP-FFT's ideal interval is `FFT_NUM / (2*UF)` and its own `#pragma HLS
performance target_ti` asks for exactly that. On our retarget to
`xcu280-fsvh2892-2L-e` at 3.333 ns with Vitis 2023.2 it does not get there, and
gets further away as UF rises:

| Config | Measured interval | Ideal | Of ideal |
|---|---:|---:|---:|
| UF1 | 140.5 | 128 | 91.1% |
| UF2 | 74.5 | 64 | 85.9% |
| UF4 | **400.0** | 32 | **8.0%** |

UF4 is not a harness artefact -- its own csynth reports a top-level interval of
424 against a latency of 423, so there is no overlap at all -- but the shipped
UF4 reports interval 32 on its own part. So the design can do it and this build
cannot, which points at pragmas and build configuration on 2023.2.

## Where to look first

`experiments/e2_fft/hpfft/N256/source/FFT.cpp` and `FFT.h`. `UF` is a
`#define` in the header.

1. **`#pragma HLS array_partition variable=data_in_cyclic type=cyclic
   factor=UF dim=2`** (FFT.cpp:88). This is the prime suspect. A butterfly at
   stage `s` reads `i` and `i + (1 << s)`; with **cyclic** banking those two
   land in the same bank as soon as the stride reaches the factor, and the pair
   serialises. It is exactly the conflict an XOR swizzle exists to remove --
   read `allo/spmw/bricks.py:124` for the statement of it, and
   `_apply_f2_partitions` on `origin/feature/allo-fft`
   (`tests/dataflow/test_fft.py:381`) for a worked fix in another language. The
   interval collapsing *as UF grows* is what this failure looks like.
2. **`#pragma HLS pipeline` at FFT.cpp:57 has no `II=1`.** The tool picks the
   interval and is free to pick a bad one. Line 107 does pin it; line 57 does
   not.
3. **`target_ti` is a target, not a constraint.** Missing it is not an error and
   does not fail the build, so nothing shouts when the schedule slips.
4. `#pragma HLS dataflow disable_start_propagation` (FFT.cpp:93) and the
   `type=pipo` streams: check the sub-function intervals individually. A
   dataflow region's interval is its slowest stage, so find which stage is slow
   before changing anything global.

Read the per-loop interval column of `csynth.rpt` for every loop and find the
one that actually misses. Do not guess from the top-level number.

## Rules

- **Pragmas, partitioning, array shape and build configuration only.** Do not
  change the algorithm, the numerics, the twiddle tables or the interface
  width. If you conclude the fix needs an algorithmic change, stop and say so
  -- that is a finding, not a licence.
- Keep the interface at `hls::vector<complex<float>, UF*2>`; changing it makes
  the comparison meaningless.
- **csim, csynth and cosim must all still pass**, and the transform must still
  be correct against the reference. A faster wrong FFT is a failure.
- Every pragma change gets recorded with the interval before and after. A
  change you cannot attribute an effect to gets reverted.
- Sweep **UF1, UF2, UF4, UF8** at N=256 at minimum. Report N=128/512/1024 if
  they come cheaply.

## Machine rules, not negotiable

- brg-zhang-xcel is **shared** and other work of mine is live on it. Own
  `/scratch/hc676/hpfft_fix` and nothing else.
- Never `pkill` by name. Kill only by explicit PID, or by matching
  `readlink /proc/<pid>/cwd` against your own directory -- and never from a
  script whose own cwd is inside the directory it sweeps, or it kills itself.
- Detached python needs `python3 -u`.
- Do not install packages or restructure the repository.

## What to report

A table of UF against interval before and after, each with the pragma or
partitioning change that moved it, every number read out of `csynth.rpt` or the
cosimulation rather than inferred. Then say plainly:

- which unroll factors now reach ideal and which do not, and why;
- what this does to E2's comparison against SPMW, **including if it removes
  SPMW's throughput advantage entirely**. Write that sentence if it is true.

`experiments/e2_fft/README.md` currently reports SPMW winning throughput at
matched width partly because of the UF4 collapse. If that collapse was our
build's fault, the README must say so and the number must change.
