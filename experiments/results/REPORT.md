# SPMW evaluation: the remaining experiments

All seven experiments of `SPMW_REMAINING_EXPERIMENTS.md` ran on brg-zhang-xcel.
Six are complete; one baseline is short two array sizes because the vendor tool
cannot cosimulate them. Every number below was measured on hardware or in RTL
simulation, and nothing is extrapolated.

| | |
|---|---|
| Board | AMD Alveo U280, `xcu280-fsvh2892-2L-e` |
| Tools | Vitis HLS 2023.2, Vivado 2023.2; XRT for the board runs |
| Target | 300 MHz, a 3.333 ns period, unless a row says otherwise |
| Bundle | `/scratch/hc676/spmw_eval_remaining_2026-09-06`, 12 packages, 341 rows, 282 passing |

| | Experiment | State |
|---|---|---|
| E1 | Output-stationary int8 GEMM at four array sizes: SPMW, AutoSA, Allo | Allo short two sizes |
| E2 | Radix-2 FFT, 128 to 1,024 points: SPMW, HP-FFT, Allo | complete |
| E3 | Complete GPT-2 medium and LLaMA-7B blocks on the board | complete |
| E4 | FEATHER with general weights: the port against the original RTL | complete |
| E5 | Compilation time with the hardware held fixed | complete |
| E6 | Grouped attention on an equal hardware budget | complete |
| E7 | Design-only source size, counted language-aware | complete |

---

## E1. Output-stationary GEMM

Every design in this comparison places exactly one multiplier per processing
element, so the arithmetic is identical and any difference is in the logic
around it. Comparing like scopes at 32x32, the SPMW kernel uses 19% fewer
lookup tables and 37% fewer registers than AutoSA, with ten times the slack.

### Routed, out of context at 3.333 ns

| Array | Design | LUT | FF | DSP | BRAM18 | Slack |
|---|---|---:|---:|---:|---:|---:|
| 4x4 | SPMW mesh | 560 | 1,021 | 16 | 0 | +1.167 ns |
| | SPMW kernel | 1,867 | 2,516 | 16 | 0 | +1.301 ns |
| | AutoSA | 8,129 | 17,481 | 16 | 9 | +0.536 ns |
| | Allo | 3,693 | 4,459 | 16 | 3 | +0.625 ns |
| 8x8 | SPMW mesh | 2,225 | 4,301 | 64 | 0 | +0.996 ns |
| | SPMW kernel | 8,015 | 10,220 | 64 | 0 | +1.232 ns |
| | AutoSA | 19,755 | 38,762 | 64 | 9 | +0.331 ns |
| | Allo | 8,419 | -- | 64 | -- | +0.598 ns |
| 16x16 | SPMW mesh | 9,026 | 18,048 | 256 | 0 | +0.428 ns |
| | SPMW kernel | 32,756 | 40,732 | 256 | 0 | +0.541 ns |
| | AutoSA | 47,434 | 81,265 | 256 | 9 | +0.455 ns |
| | Allo | cosimulation stalls | | | | |
| 32x32 | SPMW mesh | 37,917 | 74,945 | 1,024 | 0 | +0.431 ns |
| | SPMW kernel | 137,741 | 163,708 | 1,024 | 0 | +0.529 ns |
| | AutoSA | 169,841 | 262,017 | 1,024 | 11 | +0.048 ns |
| | Allo | cosimulation stalls | | | | |

No unrouted nets anywhere. *Mesh* is the array with its boundary buffers;
*kernel* is the same array with its memory loaders and drain, which is the
scope AutoSA and Allo also report.

### Cycles, against the work each launch does

The three flows do not launch the same thing, so their cycle counts are not
directly comparable and each row names its workload. One SPMW launch computes a
single tile; AutoSA and Allo fold the tiles of a fixed problem inside the launch.

| Array | System | First out | Completion | Workload of one launch |
|---|---|---:|---:|---|
| 4x4 | SPMW | 10 | 66 | one 4^3 tile |
| | AutoSA | 63 | 788 | 16^3, 16 tiles folded |
| | Allo | 112 | 136 | one 4^3 tile |
| | Allo | 316,471 | 334,909 | 128^3, 1,024 tiles folded |
| 8x8 | SPMW | 14 | 100 | one 8^3 tile |
| | AutoSA | 69 | 795 | 16^3, 4 tiles folded |
| | Allo | -- | 306 | one 8^3 tile |
| | Allo | 106,807 | 125,245 | 128^3, 256 tiles folded |
| 16x16 | SPMW | 22 | 152 | one 16^3 tile |
| | AutoSA | 71 | 797 | 16^3, one tile |
| 32x32 | SPMW | 38 | 298 | one 32^3 tile |
| | AutoSA | 87 | 2,989 | 32x32x16, one tile |

SPMW figures are the packaged kernel against a behavioural AXI memory, start bit
to last drain beat. Its cost per tile grows as roughly 2S, so the abstraction
adds no term that scales with the array.

### Three findings about the baselines

- **AutoSA cannot build a 32x32 array for a 16-cubed problem.** It silently
  clamps the partition and emits a byte-identical 16x16 design, so that row uses
  a 32x32x16 problem instead. The clamped attempts are kept as unsupported rows
  with the diff as evidence.
- **Its kernel as specified is functionally wrong.** With no accumulator reset in
  the scope, 4x4 fails 240 of 256 outputs. The measured rows use AutoSA's own
  documented corrected form.
- **A Vitis cosimulation pass verdict means nothing here**, because AutoSA's
  generated testbench returns success unconditionally, and its reported latency
  measures the testbench's control transactions rather than the kernel. Both were
  replaced with a checking testbench and waveform analysis.

### Where Allo stops

Its library systolic array simulates and synthesises correctly at every size,
but from 16x16 up, Vitis stops writing files inside RTL testbench generation and
spins at full processor load indefinitely. Two 32x32 builds held for twelve hours
with no output before being stopped; a watchdog now catches the condition in 35
minutes. Synthesis before it succeeds normally, in 423 seconds at 16x16, so this
is a limit on cosimulating a 256-process design rather than a property of the
design. Implementation-only retries were still running when this was written.

---

## E2. Radix-2 FFT

The folded single-path delay-feedback pipeline sustains an interval of exactly N
plus 0.09 cycles per transform at every size from 128 to 1,024 points. Its cost
grows with the logarithm of N, because a size doubling adds one stage unit:
twenty more multipliers and about 1,500 more lookup tables.

| Points | System | Latency | Interval | LUT | FF | DSP | BRAM18 | Slack |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | SPMW | 522 | 132 | 10,930 | 25,425 | 139 | 12 | +0.461 ns |
| | HP-FFT | 679 | 76 | 19,874 | 19,159 | 60 | 68 | +0.313 ns |
| | Allo | 11,586 | 11,545 | -- | -- | 16 | 8 | -- |
| 256 | SPMW | 1,035 | 264 | 12,361 | 28,865 | 159 | 18 | +0.388 ns |
| | HP-FFT | 1,400 | 143 | 22,403 | 22,244 | 72 | 82 | +0.283 ns |
| | Allo | 26,261 | 26,220 | -- | -- | 16 | 10 | -- |
| 512 | SPMW | 2,059 | 528 | 13,864 | 32,222 | 179 | 24 | +0.129 ns |
| | HP-FFT | 2,953 | 269 | 25,881 | 25,518 | 84 | 92 | +0.179 ns |
| | Allo | 58,808 | 58,767 | -- | -- | 16 | -- | -- |
| 1,024 | SPMW | 4,107 | 1,056 | 15,423 | 35,587 | 199 | 31 | +0.431 ns |
| | HP-FFT | 6,298 | 529 | 29,316 | 28,835 | 96 | 104 | +0.261 ns |
| | Allo | 130,299 | 130,258 | -- | -- | 16 | 14 | -- |

Complex FP32. Latency is one transform; interval is the steady-state spacing
when transforms stream back to back. **HP-FFT carries two samples per beat**, so
its interval near N/2 is the same one-sample-per-cycle rate at twice the port
width. Allo runs one transform per call, so its interval is its latency, and its
resources are synthesis estimates.

Every SPMW cosimulation matched the reference on every token: 4,224 tokens at
128 points through 33,792 at 1,024.

### Four defects this design surfaced

Every one was silent, and all four are fixed and committed.

- **Missing ROM data files** meant every twiddle factor read as zero, because the
  simulator copied only the Verilog and not the `.dat` files it reads at load.
- **The floating-point tolerance was too tight** for a pipeline that cancels
  order-N intermediates; a design now declares its own.
- **The lowering merged same-named constants across units**, so stages built by a
  factory all used the first stage's parameters. Correct at two points, wrong
  from four up, and only in the compiled targets.
- **Pipelines did not drain** when their loop ended, losing 17 of 4,352 tokens
  with zero reported errors. A design can now ask for a flushable pipeline style.

---

## E3. Complete transformer blocks

One GPT-2 medium layer and one LLaMA-7B layer run end to end on the same 300 MHz
design, each stage's device output feeding the next. Across six repetitions and
5,776 device launches, not one value differed from the integer reference.

| Block | Launches | Device | Transfers | Host math | Packing | Reps | Mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 medium | 624 | 72.7 ms | 371.8 ms | 53.0 ms | 4.64 s | 4 | 0 |
| LLaMA-7B | 4,064 | 715.2 ms | 2,381.5 ms | 38.8 ms | 28.85 s | 2 | 0 |

Per layer, prefill of 128 tokens. The bitstream uses 132,922 LUT, 116,913 FF,
304 DSP and 187 BRAM18 with +0.016 ns of slack and zero total negative slack.

The blocks are **hybrid**: the matrix and softmax stages run on the device, while
normalisation, rotary embedding, masking, the activation functions, residual
additions and all requantisation run on the host. These are therefore not
end-to-end serving latencies. LLaMA's wider projections, head width and gated
feed-forward are decomposed on the host into this engine's existing launch
shapes, which is why it needs 4,064 launches where GPT-2 medium needs 624.

Device time is a fifth of the transfer time for the GPT-2 layer and a third for
LLaMA. On this engine a complete block is bound by the data movement its launch
granularity implies, not by the array.

---

## E4. FEATHER

Whole workloads with general weights, mixed-sign across the full int8 range and
real reordering programs, checked tile by tile against the original RTL's own
arithmetic and against a host reference. Reaching this needed one correction to
the shipped controller, which its own drivers work around.

| Array | Weights | Tiles | SPMW port | FEATHER RTL | Cycles per tile |
|---|---|---:|---:|---:|---|
| 4x4 | resident | 32,768 | 131,118 | 131,149 | 4 and 4 |
| | fed per tile | 32,768 | 262,182 | 2,097,169 | 8 and 64 |
| 8x8 | resident | 4,096 | 32,856 | 33,299 | 8 and 8 |
| | fed per tile | 4,096 | 65,644 | 2,097,179 | 16 and 512 |
| 16x16 | resident | 512 | 8,340 | 12,317 | 16 and 16 |
| | fed per tile | 512 | 16,720 | 2,097,197 | 32 and 4,096 |

A 128-cubed int8 GEMM under the drivers' own tiling; a 16x16x64 convolution with
3x3 kernels gives the same picture. Cycles are counted in the same simulator on
both sides.

With weights resident the two agree to within 0.4% at 4x4 and 8x8, at identical
cycles per tile. When a workload has to re-feed weights, the port holds its
interval while the original RTL takes up to 128 times longer, because its shipped
weight loader admits one processing element per cycle. That is a property of the
reference implementation's loader, not of the FEATHER architecture.

Out of context the two are comparable in area at 4x4: 2,305 lookup tables and
3,639 registers for the port against 2,309 and 3,378 for FEATHER's own. The
difference is that the port places its multipliers in 16 digital signal
processing slices while the RTL builds them from logic.

---

## E5. Compilation time

With the hardware held fixed and the generated code identical, compiling one
project per role instead of one per site saves nothing at 4x4 and 16.7x at
16x16. The folded FFT is the control: it has one site per role, and the
experiment correctly finds no gain there.

| Design | Roles / sites | Shared, 8 workers | Shared, serial | Per-instance | Reuse gain |
|---|---|---:|---:|---:|---:|
| GEMM 4x4 | 9 / 16 | 82.0 s | 361.0 s | 85.4 s | 1.0x |
| GEMM 8x8 | 9 / 64 | 82.4 s | 364.4 s | 339.1 s | 4.1x |
| GEMM 16x16 | 9 / 256 | 82.4 s | 361.5 s | 1,377.7 s | 16.7x |
| GEMM 32x32 | 9 / 1,024 | 82.2 s | 363.2 s | capped at 3,600 s | over 43.8x |
| FFT 128 | 8 / 8 | 47.5 s | 351.3 s | 47.8 s | 1.0x |
| FFT 256 | 9 / 9 | 88.5 s | 400.9 s | 88.2 s | 1.0x |
| FFT 512 | 10 / 10 | 92.2 s | 443.3 s | 91.3 s | 1.0x |
| FFT 1,024 | 11 / 11 | 92.9 s | 490.0 s | 93.0 s | 1.0x |

Median of three repetitions, mode order randomised per point, eight workers
throughout, a fresh directory per run, spread under 2%. The 32x32 per-instance
figure is a bound: all three repetitions hit the preset one-hour cap after
roughly 660 of 1,024 projects.

The shared build stays flat near 82 seconds from 4x4 to 32x32 because it always
compiles the same nine roles, while the per-instance build tracks the site count.
Parallelism is a separate and much smaller effect, worth 4.4x, and it saturates
once the roles fit the worker count. That saturation is also why the FFT jumps
from 47.5 seconds at eight roles to 88.5 at nine: one round of eight workers
against two.

The earlier 46.0x at 32x32 used 24 workers and the earlier 4x4 pairs used 32; the
123.8x was a ratio of summed job elapsed times. Neither is a repetition of this
protocol, and both stay as their own experiments.

---

## E6. Grouped attention

The earlier 2x figure was an estimate derived from arrays of different sizes. Run
properly, with the same sixteen processing elements and the same workload carried
to completion, the grouped design wins by 1.50x to 2.00x depending on how the
seed reaches the array.

| Design | PEs | M = 6 | M = 64 | M = 4,096 | Cycles per row |
|---|---:|---:|---:|---:|---:|
| Grouped, constant seed | 16 | 45 | 123 | 5,499 | 1.343 |
| Grouped, streamed seed | 16 | 46 | 104 | 4,136 | 1.010 |
| Conventional, two launches | 16 | 76 | 192 | 8,256 | 2.016 |
| HLS grouped | 16 | 360 | 430 | 4,470 | 1.091 |
| HLS conventional | 16 | 794 | 1,418 | 37,706 | 9.206 |

Completion cycles; the conventional design needs two launches where the grouped
design needs one, and its figure is the sum of both. Three seeds give identical
counts.

Both route at 300 MHz with the same sixteen multipliers and no block RAM: 1,037
lookup tables and 2,321 registers for the grouped design against 1,207 and 2,997
for the conventional one. The same pair written by hand in HLS shows 8.44x, but
only because its conventional kernel serialises on unwidened lane ports; packing
those ports brings it back to 1.11x, which is why the fabric-level comparison is
the meaningful one.

---

## E7. Source size

The archived counts applied the C comment rule to Python, so docstrings and
comment-only lines inflated the SPMW side. Recounting language-aware moved every
SPMW figure down and every ratio up.

| Design | HLS | SPMW | Ratio |
|---|---:|---:|---:|
| Systolic GEMM | 94 | 24 | 3.92x |
| Multi-cache GEMM | 113 | 40 | 2.83x |
| Tiled GEMM | 153 | 32 | 4.78x |
| FFT, folded radix-2 | 299 | 106 | 2.82x |
| Mini-TPU matrix unit | 113 | 40 | 2.83x |
| Grouped attention | 112 | 53 | 2.11x |

Design lines only: unit bodies, interfaces, topology rules and boundary bindings.
Imports, type aliases, sizes and instantiations count separately as
configuration; tests and reusable framework code are excluded on both sides.

The FFT row could not be filled before because the two sides implemented
different architectures. It is fillable now that both are folded: HP-FFT's
one-sample-per-cycle kernel against the delay-feedback pipeline of E2. Source
size measures description effort, not development time, and the tiled GEMM pair
differs in element type and loader structure.

---

## Method

- **Pass** means a cosimulation matched the design's own reference on every
  output token, or an implementation routed with no unrouted nets and
  non-negative worst slack. Nothing is marked pass on a tool's own verdict alone.
- **Bounded** marks a figure that hit a preset cap, recorded with the work
  completed at that point. The 32x32 per-instance compile is the only one, and it
  is reported as a bound rather than extrapolated.
- **Still open**: Allo at 16x16 and 32x32, whose implementation retries were
  running when this was written. Its cosimulation cannot complete in this tool
  version and is recorded with that reason.

Every package in the bundle carries its own README with the exact commands, how
each metric was measured and its caveats, alongside the tool reports each row was
read from. Read a package's README before its numbers: the packages measure
different things and are not interchangeable.
