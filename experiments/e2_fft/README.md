# E2: a radix-2 complex FP32 FFT, in three systems

Same layout as E1: `<framework>/N<size>/{source,generated,report}`, one
`results.csv` that the folders reproduce, and the scripts that drove the runs.
Sizes are transform lengths, N = 128, 256, 512, 1024.

## The datapaths are not the same width, and that has to be read first

SPMW's boundary is **one complex sample per cycle**. HP-FFT's smallest shipped
configuration, UF1, has a boundary of
`hls::stream<hls::vector<complex<float>,2>>` -- 128 bits, **two complex samples
a beat**. So their ideal intervals differ by 2x by construction, and unlike
E1's memory-port mismatch this one cannot be flagged away: UF1 is the narrowest
configuration the repository ships.

Both are therefore measured on one definition, the one HP-FFT's own harness
records: **cycles are rising clock edges; a transform's latency is its last
output beat minus the launch's first input beat; the steady interval is the
median of consecutive completion differences.** SPMW's cosimulation reports
per-transform completions for this (`spmw_tokens_per_transform` on the fabric),
so the two sides are read the same way rather than compared across two
different quantities.

## Latency and throughput point in opposite directions

| N | System | Full-transform latency | Steady interval | Ideal | Of ideal | Samples/cycle |
|---|---|---:|---:|---:|---:|---:|
| 128 | SPMW | **581** | 128.0 | 128 | **100.0%** | 1.000 |
| | HP-FFT UF1 | 742 | 76.0 | 64 | 84.2% | **1.684** |
| 256 | SPMW | **1,158** | 256.0 | 256 | **100.0%** | 1.000 |
| | HP-FFT UF1 | 1,527 | 140.5 | 128 | 91.1% | **1.822** |
| 512 | SPMW | **2,310** | 512.0 | 512 | **100.0%** | 1.000 |
| | HP-FFT UF1 | 3,208 | 269.0 | 256 | 95.2% | **1.903** |
| 1024 | SPMW | **4,614** | 1024.0 | 1024 | **100.0%** | 1.000 |
| | HP-FFT UF1 | 6,809 | 525.5 | 512 | 97.4% | **1.949** |

**SPMW has the lower full-transform latency at every size, despite half the
datapath**, and its advantage grows with N: 1.28x at 128 to 1.48x at 1024. Its
latency is a flat 4.5N; HP-FFT's is 5.8N rising to 6.7N.

**HP-FFT has the higher throughput at every size**, and its advantage also
grows, 1.68x to 1.95x, as its fixed per-transform overhead amortises and it
approaches the 2x its datapath allows.

The two results have one cause. SPMW sustains **exactly N cycles a transform at
every size** -- 100% of its ideal, measured as a median of 32 consecutive
completions with only the final drain differing (min 128, max 132 at N=128).
HP-FFT runs at 84% of its own ideal at 128 and only reaches 97% at 1024. A
wider datapath that is not kept full buys throughput and costs latency.

## A correction

An earlier version of this table reported SPMW's interval as 132.1 at N=128
(96.9% of ideal), rising to 1,056.1 at N=1024. Those figures came from
`(launch total - first output) / 32`, which is wrong twice: it starts at the
first *output* beat, so the pipeline fill sits inside the interval, and it
divides by 32 when 33 transforms are emitted. At N=128 that turns a true 128.0
into 128 + 127/32 = 132.1. The `completion_cycles` column was worse: SPMW's
held the span of all 33 transforms and HP-FFT's held one transform, a factor of
thirty apart in the same column. **The HP-FFT column was not affected by
either error**; only SPMW's side was wrong. `results.csv` now carries
`full_transform_latency` on one definition for both, and each transform's
completion cycle together with the launch's first input beat is in
`spmw/N*/report/cosim_transforms.log`.

## Results

| N | System | Samples/cyc | First out | Full xform | Interval | LUT | FF | DSP | BRAM18 | Slack | Clock |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | SPMW folded SDF | 1 | 454 | 581 | 128.0 | 10,930 | 25,425 | 139 | 12 | +0.461 ns | 348 MHz |
| | HP-FFT UF1 | 2 | 679 | 742 | 76.0 | 19,874 | 19,159 | 60 | 68 | +0.313 ns | 331 MHz |
| | Allo strided | -- | | 11,545 | 11,545 | 4,340 | 7,622 | 16 | 10 | +0.697 ns | 379 MHz |
| 256 | SPMW folded SDF | 1 | 903 | 1,158 | 256.0 | 12,361 | 28,865 | 159 | 18 | +0.388 ns | 340 MHz |
| | HP-FFT UF1 | 2 | 1,400 | 1,527 | 140.5 | 22,403 | 22,244 | 72 | 82 | +0.283 ns | 328 MHz |
| | Allo strided | -- | | 26,220 | 26,220 | 4,422 | 7,646 | 16 | 10 | +0.671 ns | 376 MHz |
| 512 | SPMW folded SDF | 1 | 1,799 | 2,310 | 512.0 | 13,864 | 32,222 | 179 | 24 | +0.129 ns | 312 MHz |
| | HP-FFT UF1 | 2 | 2,953 | 3,208 | 269.0 | 25,881 | 25,518 | 84 | 92 | +0.179 ns | 317 MHz |
| | Allo strided | -- | | 58,767 | 58,767 | 4,316 | 7,542 | 16 | 12 | +0.694 ns | 379 MHz |
| 1024 | SPMW folded SDF | 1 | 3,591 | 4,614 | 1024.0 | 15,423 | 35,587 | 199 | 31 | +0.431 ns | 345 MHz |
| | HP-FFT UF1 | 2 | 6,298 | 6,809 | 525.5 | 29,316 | 28,835 | 96 | 104 | +0.261 ns | 326 MHz |
| | Allo strided | -- | | 130,258 | 130,258 | 4,320 | 7,561 | 16 | 12 | +0.438 ns | 345 MHz |

Every cycle figure is relative to the launch's **first input beat**, so "first
out" and "full xform" are latencies rather than absolute cycles. Interval is
the median of consecutive transform completions. Allo is not a streaming
design, so it has no first-output beat and its per-call latency is both its
transform latency and its interval. Slack is against a 3.333 ns target, routed
out of context on `xcu280-fsvh2892-2L-e` with nothing unrouted; the clock is
the period that slack implies.

## What each system is

- **SPMW**, `spmw/`: the folded single-path delay-feedback pipeline. log2(N)
  `stage` units in a chain, each with a delay line of N/2^(s+1) complex samples
  and its twiddles as a resident ROM, plus one `reorder` unit that undoes the
  bit-reversal from a double buffer. 8 roles at N=128 rising to 11 at N=1024,
  which is log2(N)+1: the role count is the stage count.
- **HP-FFT**, `hpfft/`: UCLA-VAST/HP-FFT-HLS at `c4611b8`, hand-written HLS.
  256 and 1024 are shipped configurations; **128 and 512 are derived** from the
  shipped sources by changing the size macros, and are labelled that way in
  every path. The design sources are never modified apart from those macros.
- **Allo**, `allo/`: `examples/machsuite/fft/strided`, a sequential in-place
  radix-2 DIF behind four `m_axi` ports. It is not a streaming architecture and
  is two orders of magnitude slower per transform; it is here as the
  general-purpose baseline, not as a competing FFT datapath.

## Where the differences come from

**Area against throughput.** HP-FFT buys its 2x datapath with roughly 1.8x
SPMW's lookup tables and 3.4 to 5.7x its block RAM, while using *fewer* DSPs
(60 to 96 against 139 to 199). The DSP direction is the one that is not
self-explanatory from the port width and is not explained here; it would need
reading how each maps a complex float multiply, which has not been done.

**Allo's cost is a memory dependence, not a width.** Its csynth reports
`HLS 200-880` on the in-place `real[]`/`img[]` arrays and auto-pipelines the
inner loop at II=25, so a butterfly costs tens of cycles rather than one. That
is visible in `report/csynth.rpt` at every size.

## The comparison above is one point against one point

SPMW's rows are a single configuration -- the folded single-path
delay-feedback pipeline, one complex sample a cycle -- and HP-FFT ships six
(UF1..UF32). So the table compares SPMW's only design against HP-FFT's
narrowest. `tests/dataflow/spmw/test_spmw_fft_rolled.py` is the family that
closes that: **`W` lanes, `W` complex samples a cycle, `W` a parameter**, and
`W = 1` is the design already measured -- at N=256 it reproduces its latency
(1,158) and its interval (256.0, 100% of ideal) to the cycle, which is the
check that the sweep extends this row rather than replacing it with something
else.

`scripts/spmw_build_array.py --design fftrolled --size N --lanes W` builds it;
`scripts/spmw/rolled_sweep.py` reads the numbers back out of the reports.

### The unroll factor is the space/time split of the butterfly partners

The `N` points of a block are spread over `W` lanes by the lane law
`lane(i) = i & (W-1)`, `row(i) = i >> log2(W)` -- `spmw.banked(banks=W)`'s own
`bank_of`/`row_of`, the same at every stage. Stage `s` of the
decimation-in-frequency recursion pairs `i` with `i ^ (1 << d)`, `d = S-1-s`,
and the law says where that partner is:

- `d >= log2(W)`: **the same lane, `1 << (d - log2 W)` rows back.** There is no
  wire to name; the partner is a delay line. This is the folded pipeline's
  stage, once per lane.
- `d < log2(W)`: **another lane at the same time, `l ^ (1 << d)`.** That is a
  wire, and the topology names it as one.

So `log2(W)` of the `log2(N)` stages have their partner in space and the rest
in time. At `W = 1` every partner is a delay line; at `W = N/2` every partner
is a wire. That is the whole knob, and
`test_the_unroll_factor_is_where_the_partner_lives` checks it rather than
asserting it in prose.

**The role count does not follow the grid.** Across the sweep at N=256 the
instance count goes 9, 20, 40, 80, 160 and the role count stays at 9, 10, 10,
9, 8 -- so the HLS cost is flat while the array grows 17.8x. (It *falls*
slightly because a delay stage is one role each, having its own literal span,
while all the cross stages together are three.)

### A swap deadlocks; a fan-out does not

The first cross stage had partner lanes trading operands on a pair of streams
inside one row -- the shape the algebra suggests. It passed `ref` and the
simulator and then produced **0 of 1,056 tokens with 0 errors** in the array
cosimulation: a failure that looks like nothing happening.

The generated Verilog says why outright. In the cross unit's pipelined body
both stream reads are enabled at `iter0` and both writes at `iter19`
(`cross0_r0_0_..._Pipeline_l_S_r_0_r.v`), so each lane blocks reading its
partner's token nineteen pipeline stages before either lane can write one, and
neither ever reaches its write. `W = 1`, which has no cross stages, passed the
same cosimulation at 1056/1056, which isolated it.

The operand is therefore fanned out from *upstream*: each unit emits its result
twice, once down its own lane and once across to the lane that will need it
next, and reads its two operands from two producers that are not waiting on it.
Every edge goes from row `t` to row `t+1`, so the graph is a DAG.
`test_the_exchange_is_a_fan_out_not_a_swap` asserts that property.

### II is 1, from the reports

Every pipelined loop of every role reports `Interval = 1`, `yes(flp)`. The two
that carry the work, at N=256:

```
W=1  delay stage, 9 roles
  o l_S__b_0__b_l_S_h_0_h_l_S_c_0_c | - | 2.43 | 8726 | ... | 22 | 1 | 8704 | yes(flp)
W=2  cross butterfly / cross fork
  o l_S__r_0__r | - | 2.43 | 4370 | 1.455e+04 | 20 | 1 | 4352 | yes(flp)
  o l_S__r_0__r | - | 2.43 | 4352 | 1.449e+04 |  2 | 1 | 4352 | yes(flp)
```

The columns are latency, latency(ns), **iteration latency**, **initiation
interval**, trip count, pipelined. Trip counts are `(BATCH+1) * N / W`: 8,704
at one lane and 4,352 at two. This is read out of `csynth.rpt`, not inferred
from a cycle count.

### Matched unroll factors

HP-FFT's boundary is `hls::stream<hls::vector<complex<float>, UF*2>>` -- `2*UF`
complex samples a beat -- and the rolled design's is `W` lanes of one complex
sample a cycle. So **`W = 2*UF`** is the matched point, and both sides then
share an ideal interval of `N / (2*UF)`. Same definition as the table above,
same part, same 3.333 ns target, 33 transforms a launch.

The HP-FFT columns below are the **fixed** baseline. The build first measured
here was wrong at UF4 and UF8; what was wrong with it, and the pragmas that
repair it, are in `hpfft/fix/README.md`. The interval as first built is kept in
the last column so the correction is visible. UF8 was previously not measured at
all; both of its numbers are new.

| N=256, matched | samples/cyc | SPMW latency | SPMW interval | SPMW of ideal | HP-FFT latency | HP-FFT interval | HP-FFT of ideal | (interval as first built) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (below UF1) | 1 | 1,158 | 256.0 | **100.0%** | -- | -- | -- | -- |
| UF1 | 2 | **581** | **128.0** | **100.0%** | 1,527 | 140.5 | 91.1% | 140.5 |
| UF2 | 4 | **368** | **64.0** | **100.0%** | 766 | 74.5 | 85.9% | 74.5 |
| UF4 | 8 | **271** | **32.0** | **100.0%** | 408 | 42.0 | 76.2% | *400.0* |
| UF8 | 16 | **222** | **16.0** | **100.0%** | 269 | 26.0 | 61.5% | *57.0* |

**At matched width SPMW still wins both axes, but by far less than this table
first said.** Latency 2.63x, 2.08x, 1.51x, 1.21x; throughput 1.10x, 1.16x,
1.31x, 1.63x. The "HP-FFT has the higher throughput at every size, and its
advantage grows" conclusion above is an artefact of comparing 1 sample a cycle
against 2: at equal width it reverses. But the **12.5x** at UF4 that this
section previously reported was almost entirely our own build's fault, and the
honest figure is **1.31x**.

**The two systems diverge as they widen.** SPMW holds *exactly* 100% of its
ideal at every width -- the interval is `N/W` to the cycle, with `min` equal to
ideal and `max` four higher on the final drain -- while HP-FFT slides 91.1%,
85.9%, 76.2%, 61.5%. The SPMW column is measured at **every** width from 1 to 16
samples a cycle -- 1,158 / 581 / 368 / 271 / 222 cycles of latency against
intervals of 256 / 128 / 64 / 32 / 16 -- and the interval halves exactly on
each doubling. Latency does not: the delay lines shorten with `W` but the
float pipelines in them do not, so the return diminishes (2.0x from one lane
to two, 1.22x from eight to sixteen).

**HP-FFT's slide is real but it is not the collapse first reported here.** The
8.0% at UF4, and the 28.1% the unfixed UF8 measures, were defects in our build;
both are fixed by pragmas alone and both fixes are verified by cosimulation
against `numpy.fft.fft` (`hpfft/fix/README.md`). What remains is structural: each butterfly stage is a
non-pipelined process in a dataflow region, so its interval is its latency,
`trip count + iteration latency`. The trip count *is* the ideal interval and the
iteration latency is a near-constant ~19 cycles of float pipeline, so the
shortfall is a fixed additive term and its share necessarily grows as the
transform is spread over fewer beats. After the fix all four unroll factors sit
within one cycle of that floor.

The claim this section previously made -- **"on its own part the shipped UF4
reports an interval of 32 ... the collapse is what retargeting to the U280 does
to it, not a property of the source"** -- was wrong twice over, and both halves
are worth correcting. The collapse was a source defect (non-`static` scratch
arrays whose constructors cost 312 of the 423 cycles) that Vitis 2023.2 exposes
and 2024.2 hides; and that upstream "32" is not a sustainable interval. The same
upstream report gives its own dataflow processes intervals of 41, 42, 42, 42,
43, 43 and flags `output_result_array_to_stream` as `Achieved TI 34, TI met:
no`. A dataflow region cannot run at 32 when six of its processes each need
41-43 cycles to restart; the 32 is the rate at which the first, pipelined
process accepts input. Our fixed build measures 42.0 in cosimulation, which is
at or better than upstream's own per-stage floor.

### Where SPMW loses: multipliers and registers

| N=256, matched width | LUT | FF | DSP | BRAM18 | clock |
|---|---:|---:|---:|---:|---:|
| SPMW W=2 (2/cyc) | 24,318 | 55,109 | **318** | **12** | 334 MHz |
| HP-FFT UF1 (2/cyc) | 22,403 | 22,244 | **72** | **82** | 328 MHz |
| ratio | 1.09x | **2.48x** | **4.42x** | **0.15x** | 1.02x |
| SPMW W=4 (4/cyc) | 45,150 | 104,820 | **604** | **24** | 329 MHz |
| HP-FFT UF2 (4/cyc) | 45,482 | 37,809 | **132** | **104** | 332 MHz |
| ratio | **0.99x** | **2.77x** | **4.58x** | **0.23x** | 0.99x |

At the same datapath width SPMW spends **4.4 to 4.6x the DSPs and 2.5 to 2.8x
the registers**, and buys with them 4.3 to 6.8x less block RAM and the interval
above. Lookup tables and clock are a wash, and the LUT side *improves* with
width -- 1.09x HP-FFT's at two samples a cycle, 0.99x at four. The direction is structural rather than mysterious: this design
is spatial in the stages -- `log2(N) * W` physical butterflies, each with its
own complex multiplier and its own delay line -- whereas HP-FFT folds more of
the transform onto shared arithmetic. The registers follow from the same place:
every unit is a deep flushable pipeline and the array is all of them at once.

The E1 table's remark that the DSP direction "is not explained here" is
answered by the matched comparison: it is the price of one butterfly per stage
per lane at II=1, and it grows linearly with `W` (159 DSPs at one lane, 318 at
two).

The routed rows for W=8 and 16 are **not measured** at the time of writing;
`scripts/spmw/rolled_sweep.py` prints them from the reports as they land.

### One cost that is a compiler gap, not a design choice

`spmw.stationary(brick, at=..., index=...)` accepts a per-site index map,
checks its arity against the port, and then **no path slices by it** --
`refsim._memory_port` hands back the whole `init` and
`Lowering.stationary_locals` declares the resident local at the brick's full
shape. (It works for a `Tensor`; only a `Brick` is dropped.) So a ROM whose
contents differ per lane is not expressible, and the delay stage holds its
whole stage's twiddle table and indexes it by lane. At N=256 that is:

| W | delay sites | ROM held | ROM read | replication | held |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 255 | 255 | 1x | 2.0 KB |
| 2 | 14 | 508 | 254 | 2x | 4.0 KB |
| 4 | 24 | 1,008 | 252 | 4x | 7.9 KB |
| 8 | 40 | 1,984 | 248 | 8x | 15.5 KB |
| 16 | 64 | 3,840 | 240 | 16x | 30.0 KB |

(complex entries.) The cross stage's table is genuinely lane-independent -- its
stride is below the lane count, so a lane's twiddle is a function of
`l & (stride-1)` alone -- so it costs nothing. `link(out, to, index=)` has the
same shape of gap: the pairing is stored on the binding and never read, both
`lower_df._plan_link` and refsim pairing sites by positional `zip`. That is why
the lane permutation lives in a topology rule and not in a binding.
`test_stationary_index_on_a_brick_is_ignored` pins the first of these.

## A stale summary, kept

`experiments/results/e2_fft/allo/summary.md` records every P&R row as
`not_run`. That is out of date: `results.csv` beside it has routed results for
all four sizes, and this table uses them. The stale file is left as it is
rather than edited, because it is the earlier agents' record.

## Reproducing

`scripts/` holds what drove these runs. Each `source/` has the design input and
the exact command or tcl; each `generated/` has what the compiler emitted, and
for HP-FFT a note that there is no generation step; each `report/` has the
synthesis, cosimulation and place-and-route reports the numbers come from,
prefixed by stage. SPMW's `generated/` was re-staged from the same
`scripts/spmw_build_array.py` entry point the measured builds used -- the role
counts match its `results.csv` at every size, which is the check that it is the
same code.

## Binding the butterfly's adders to fabric

HP-FFT carries six `#pragma HLS bind_op ... impl=fabric` in its butterfly and
keeps only its multiplies in DSP blocks. SPMW emitted no binding at all, so
part of the DSP gap was a comparison of who remembered the pragma rather than
of the two designs. `spmw_bind_fabric` on a fabric now emits the same pragma
for every feed-forward float add and subtract in every role; multiplies are
left alone, as HP-FFT leaves its own.

N=256, both routed out of context at 3.333 ns, nothing unrouted:

| W | | LUT | FF | DSP | BRAM18 | WNS |
|---:|---|---:|---:|---:|---:|---:|
| 1 | in DSPs | 12,304 | 29,792 | 159 | 12 | +0.411 ns |
| | in fabric | 18,496 | 25,143 | **87** | 4 | +0.253 ns |
| 2 | in DSPs | 24,318 | 55,109 | 318 | 12 | +0.335 ns |
| | in fabric | 36,567 | 46,704 | **174** | 6 | +0.345 ns |

**It is a trade, not a saving.** The multiplier count falls 45% at both widths
-- 159 to 87 and 318 to 174, almost exactly the four adds per butterfly the
decomposition predicted -- and lookup tables rise 50% to pay for it. Registers
and block RAM both *fall*, which was not predicted: a DSP-implemented float
adder carries its own pipeline registers, and moving it into fabric gives those
up along with the DSP.

Timing survives at both widths. W=2 is unchanged within noise, +0.335 to
+0.345 ns; W=1 loses 0.158 ns and still closes with a quarter-nanosecond to
spare.

### What it does to the comparison

At matched width, two samples a cycle, against HP-FFT UF1:

| | LUT | FF | DSP |
|---|---:|---:|---:|
| SPMW, adds in DSPs | 1.09x | 2.48x | **4.42x** |
| SPMW, adds in fabric | **1.63x** | 2.10x | **2.42x** |

Binding removes the part of the DSP gap that was ours to remove: 4.42x becomes
**2.42x**, and what is left is architectural -- one physical butterfly per
stage per lane, each with its own complex multiplier, which is what buys the
exact interval. But it does not make the two designs similar. It moves the
disagreement from the DSP column to the lookup-table column, where SPMW goes
from parity to 1.63x.

Both variants are kept in `results.csv`. The bound one is the like-for-like
configuration, because it is the choice HP-FFT itself makes; the unbound one is
what SPMW produces if a design says nothing, and the gap between them is the
cost of that silence.
