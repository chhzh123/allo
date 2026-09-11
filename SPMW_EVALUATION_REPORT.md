# SPMW evaluation: the remaining experiments

All seven experiments of `SPMW_REMAINING_EXPERIMENTS.md` ran on brg-zhang-xcel.
Every number below was measured on hardware or in RTL simulation, and nothing
is extrapolated.

**E1 and E2 have been re-measured since the first version of this report, and
their sections say so where a figure changed.** E1's four systems were computing
different amounts of work; they now all compute the same S-cubed problem on one
cycle definition, and Gemmini joins as a second RTL baseline. E2's SPMW interval
was computed by a formula that was wrong twice. Sections E3 to E7 are unchanged.

| | |
|---|---|
| Board | AMD Alveo U280, `xcu280-fsvh2892-2L-e` |
| Tools | Vitis HLS 2023.2, Vivado 2023.2; XRT for the board runs |
| Target | 300 MHz, a 3.333 ns period, unless a row says otherwise |
| Bundle | `/scratch/hc676/spmw_eval_remaining_2026-09-06`, 12 packages, 341 rows, 282 passing |

| | Experiment | State |
|---|---|---|
| E1 | Output-stationary int8 GEMM at four array sizes: SPMW, AutoSA, Allo, Gemmini | re-measured; two cells open |
| E2 | Radix-2 FFT, 128 to 1,024 points: SPMW, HP-FFT, Allo | re-measured, complete |
| E3 | Complete GPT-2 medium and LLaMA-7B blocks on the board | complete |
| E4 | FEATHER with general weights: the port against the original RTL | complete |
| E5 | Compilation time with the hardware held fixed | complete |
| E6 | Grouped attention on an equal hardware budget | complete |
| E7 | Design-only source size, counted language-aware | complete |

---

## E1. Output-stationary GEMM

**These numbers supersede the first version of this section.** That version
compared launches that did different amounts of work: AutoSA was given a fixed
16-cubed problem while the others computed one S-cubed tile, so at 4x4 its
launch did sixty-four times the arithmetic of the row beside it. Every design
here now computes the same thing -- `C = A B`, int8 in and int32 out, with
M = N = K = S on an S x S array holding one multiplier per element -- and every
cycle figure is on one definition: **first memory beat to last memory beat**,
with control writes, the start bit and each flow's own polling outside it. The
package is `experiments/e1_gemm/`.

### Routed, out of context at 3.333 ns

| Array | Design | Port in/out | Cycles | (flow reports) | LUT | FF | DSP | Slack | Clock |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | SPMW mesh | streams | 16 | | 560 | 1,021 | 16 | +1.167 ns | 462 MHz |
| | SPMW kernel | 512/512 | 47 | 66 | 1,867 | 2,516 | 16 | +1.301 ns | 492 MHz |
| | AutoSA | 32/32 | 78 | 148 | 4,974 | 8,171 | 16 | +0.512 ns | 354 MHz |
| | Allo | 32/32 | 118 | 167 | 3,693 | 4,459 | 16 | +0.625 ns | 369 MHz |
| | Gemmini WS mesh | streams | 17 | | 1,592 | 1,064 | **0** | +0.590 ns | 365 MHz |
| 8x8 | SPMW mesh | streams | 28 | | 2,225 | 4,301 | 64 | +0.996 ns | 428 MHz |
| | SPMW kernel | 512/512 | 82 | 100 | 8,015 | 10,220 | 64 | +1.232 ns | 476 MHz |
| | AutoSA | 64/32 | 222 | 292 | 13,393 | 22,924 | 64 | +0.690 ns | 378 MHz |
| | AutoSA, wide | 512/512 | 130 | 202 | 15,343 | 25,536 | 64 | +0.738 ns | 385 MHz |
| | Allo | 32/32 | 288 | 337 | 8,419 | 8,220 | 64 | +0.598 ns | 366 MHz |
| | Gemmini WS mesh | streams | 33 | | 6,457 | 4,128 | **0** | +0.418 ns | 343 MHz |
| 16x16 | SPMW mesh | streams | 52 | | 9,026 | 18,048 | 256 | +0.428 ns | 344 MHz |
| | SPMW kernel | 512/512 | 133 | 152 | 32,756 | 40,732 | 256 | +0.541 ns | 358 MHz |
| | AutoSA | 128/32 | 782 | 859 | 47,390 | 81,265 | 256 | +0.410 ns | 342 MHz |
| | AutoSA, wide | 512/512 | 374 | 445 | 48,957 | 84,062 | 256 | +0.380 ns | 339 MHz |
| | Allo | 32/32 | 928 | 977 | -- | -- | 256 | -- | -- |
| | Gemmini WS mesh | streams | -- | | 26,847 | 17,024 | **0** | +0.306 ns | 330 MHz |
| 32x32 | SPMW mesh | streams | 100 | | 37,917 | 74,945 | 1,024 | +0.431 ns | 345 MHz |
| | SPMW kernel | 512/512 | 280 | 298 | 137,741 | 163,708 | 1,024 | +0.529 ns | 357 MHz |
| | AutoSA | 256/32 | 2,990 | 3,064 | 185,634 | 317,967 | 1,024 | +0.111 ns | 310 MHz |
| | AutoSA, wide | 512/512 | 1,222 | 1,300 | 186,369 | 320,772 | 1,024 | +0.101 ns | 309 MHz |
| | Allo | 32/32 | 3,408 | 3,457 | -- | -- | 1,024 | -- | -- |
| | Gemmini WS mesh | streams | -- | | 111,787 | 71,380 | **0** | +0.083 ns | 308 MHz |

No unrouted nets anywhere. *Mesh* is the array with its boundary buffers, the
only array-scope row; every other row is kernel scope and includes that
system's own memory interface. Port width is read off each design's synthesised
RTL, never assumed from the source. Allo's routed resources at 16x16 and 32x32,
and Gemmini's cycles at those two sizes, were still measuring when this was
written.

### Two things to keep attached to this table

**Do not read the lookup-table column without the DSP column.** Gemmini routes
with **zero** DSP blocks at every size, and so does FEATHER in E4: Vivado leaves
a signed 8x8 multiply written in plain RTL below its inference threshold, while
the HLS path binds it to a DSP. So SPMW's lookup-table advantage over the RTL
baselines is substantially a mapping difference, not a logic-efficiency
difference. What supports the "same array, different mapping" reading is that
the register counts agree within 6 per cent at every size (1.04, 0.96, 0.94,
0.95) while the lookup-table ratio stays near constant (2.84, 2.90, 2.97, 2.95).
Against AutoSA and Allo, which do use DSPs, the comparison is like for like: at
32x32 the SPMW kernel uses 26% fewer lookup tables and 48% fewer registers than
AutoSA, with 4.8 times the slack.

**Gemmini's output-stationary rows are not comparable to SPMW's** and are
excluded above. Its OS element performs per-PE output requantisation -- four
32-bit variable shifts and round-to-nearest -- that neither its own WS element
nor SPMW's array contains; SPMW requantises on the host. That single difference
accounts for 4.2x to 4.5x of area at every size, and it costs the clock: the OS
mesh misses timing at 32x32 at -0.065 ns where the WS mesh still closes.
Gemmini's WS figures are also a single-matmul latency from a driver written for
this experiment, which issues preload and compute serially where Gemmini's own
controller can overlap them, so they understate its back-to-back throughput.

### Three findings about the baselines

- **AutoSA cannot build a 32x32 array for a 16-cubed problem.** It silently
  clamps the partition and emits a byte-identical 16x16 design. The clamped
  attempts are kept as unsupported rows with the diff as evidence.
- **Its kernel as specified is functionally wrong.** With no accumulator reset in
  the scope, 4x4 fails 240 of 256 outputs. The measured rows use AutoSA's own
  documented corrected form.
- **A Vitis cosimulation pass verdict means nothing here**, because AutoSA's
  generated testbench returns success unconditionally, and its reported latency
  measures the testbench's control transactions rather than the kernel. Both were
  replaced with a checking testbench and waveform analysis.

### Where Allo stopped, and what unblocked it

Its library systolic array simulates and synthesises correctly at every size,
but from 16x16 up Vitis stopped writing files inside RTL testbench generation
and spun at full processor load indefinitely; two 32x32 builds held twelve
hours before being stopped. Synthesis itself succeeds normally, in 423 seconds
at 16x16, so this is a limit on cosimulating a 256-process design rather than a
property of the design. Cycle counts at both sizes were recovered afterwards
and are in the table; the routed resources at those sizes are still missing.

One further correction: Allo's 4x4 and 8x8 cycle figures in the first version
of this section, 136 and 306, were the minimum over three launches rather than
a representative one. The figures above, 167 and 337, are on the same
first-beat-to-last-beat definition as every other row.

---

## E2. Radix-2 FFT

**These numbers supersede the first version of this section**, which reported
SPMW's interval as 132.1 cycles at 128 points (96.9% of ideal) rising to
1,056.1 at 1,024. That came from `(launch total - first output) / 32`, which is
wrong twice: it starts at the first *output* beat, so the pipeline fill sits
inside the interval, and it divides by 32 where 33 transforms are emitted. The
HP-FFT column was not affected by either error; only SPMW's was.

Measured properly -- the median of consecutive per-transform completions, which
the cosimulation now reports directly -- the folded single-path delay-feedback
pipeline sustains an interval of **exactly N**, one complex sample per cycle,
100.0% of ideal at every size. Its cost grows with the logarithm of N, because
a size doubling adds one stage unit: twenty more multipliers and about 1,500
more lookup tables.

| Points | System | Samples/cyc | Latency | Interval | Of ideal | LUT | FF | DSP | BRAM18 | Slack |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | SPMW | 1 | **581** | 128.0 | **100.0%** | 10,930 | 25,425 | 139 | 12 | +0.461 ns |
| | HP-FFT UF1 | 2 | 742 | 76.0 | 84.2% | 19,874 | 19,159 | 60 | 68 | +0.313 ns |
| | Allo | -- | 11,545 | 11,545 | -- | 4,340 | 7,622 | 16 | 10 | +0.697 ns |
| 256 | SPMW | 1 | **1,158** | 256.0 | **100.0%** | 12,361 | 28,865 | 159 | 18 | +0.388 ns |
| | HP-FFT UF1 | 2 | 1,527 | 140.5 | 91.1% | 22,403 | 22,244 | 72 | 82 | +0.283 ns |
| | Allo | -- | 26,220 | 26,220 | -- | 4,422 | 7,646 | 16 | 10 | +0.671 ns |
| 512 | SPMW | 1 | **2,310** | 512.0 | **100.0%** | 13,864 | 32,222 | 179 | 24 | +0.129 ns |
| | HP-FFT UF1 | 2 | 3,208 | 269.0 | 95.2% | 25,881 | 25,518 | 84 | 92 | +0.179 ns |
| | Allo | -- | 58,767 | 58,767 | -- | 4,316 | 7,542 | 16 | 12 | +0.694 ns |
| 1,024 | SPMW | 1 | **4,614** | 1,024.0 | **100.0%** | 15,423 | 35,587 | 199 | 31 | +0.431 ns |
| | HP-FFT UF1 | 2 | 6,809 | 525.5 | 97.4% | 29,316 | 28,835 | 96 | 104 | +0.261 ns |
| | Allo | -- | 130,258 | 130,258 | -- | 4,320 | 7,561 | 16 | 12 | +0.438 ns |

Complex FP32. Both flows are read on one definition, the one HP-FFT's own
harness uses: cycles are rising clock edges relative to the launch's first
input beat, a transform's latency is its last output beat minus that, and the
interval is the median of consecutive completion differences.

**The two systems are not at the same port width, and this one cannot be
flagged away.** SPMW's boundary carries one complex sample a beat; HP-FFT's
narrowest shipped configuration, UF1, carries two, so their ideal intervals
differ by 2x by construction. HP-FFT is the faster streamer in absolute terms
and SPMW is the closer to its own boundary's limit; the "of ideal" column is
what makes those two statements comparable. SPMW's latency is lower at every
size, 581 against 742 up to 4,614 against 6,809. Allo runs one transform per
call, so its interval is its latency.

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
cycles per tile. When a workload has to re-feed weights, the port's total falls
with the array while the original RTL's does not move at all: 2,097,169,
2,097,179 and 2,097,197 are the same number three times. Its loader admits one
processing element per cycle -- probed in simulation, the array absorbs exactly
one weight a cycle at every size, 64 writes in 64 cycles at 4x4 and 4,096 in
4,096 at 16x16 -- and a tile holds as many weights as it performs
multiply-accumulates, so re-feeding every tile costs this GEMM one cycle per MAC,
128-cubed, whatever the array is. Only the first output (81, 539, 4,141) and the
resident rows scale.

Half of that is the reference implementation's loader and half is architectural:
the drivers' own layout replicates each weight across N/2 processing-element
rows, so a loader using the whole N-byte row instead of one byte of it would
still leave the feed falling 2x per doubling against 4x the arithmetic. A
weight-stationary array fed through a port that does not widen with it does not
get faster by growing.

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
