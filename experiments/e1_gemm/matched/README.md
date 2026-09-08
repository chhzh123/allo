# E1 matched: an S x S array computing exactly S x S x S, in every system

The original E1 rows were not comparable. AutoSA's input pinned the problem at
16x16x16 and varied only the array, so its launch iterated a whole problem
while SPMW and Allo computed one S-cubed tile. These runs fix that.

## What changed

`gen_matched.sh` generates AutoSA designs whose problem is `I = J = K = S` with
`array_part[S,S,S]`, `latency[1,1]`, `simd[1]`. The array is `array_part /
latency`, so that is an S x S array, and the problem is exactly S cubed. The
input program is the working `kernel.c` verbatim; only `kernel.h` changes.

| Array | PE instances | Workload | Generated |
|---|---:|---|---|
| 4x4 | 16 | 4³ | yes |
| 8x8 | 64 | 8³ | yes |
| 16x16 | 256 | 16³ | yes |
| 32x32 | **1,024** | 32³ | yes |

**This overturns an earlier finding.** The record previously said AutoSA cannot
build a 32x32 array and silently clamps to 16x16. It clamps only when the tile
is larger than the loop bounds, which is what asking for a 32x32 tile on a
16-cubed problem does. Given a 32-cubed problem it builds all 1,024 elements.

Two failures on the way were mine, not AutoSA's, and are worth recording
because they produce misleading errors. Writing a `kernel.c` in which `C` is
never read after the scop makes the computation dead code, and AutoSA reports
`Single outermost permutable band not found`, which reads like a scheduling
limitation. And its wrapper writes into `out/src`, `out/latency_est`,
`out/resource_est` and `out/tuning` before checking that they exist, so all
four must be created first or it reports that the output directory is not
specified.

## The result: the element is identical in all three systems

The reduction loop is what has to pipeline, and every system pipelines it the
same way.

| System | Loop | Latency | Iteration latency | Interval | Trip |
|---|---|---:|---:|---:|---:|
| SPMW | `l_S_k_0_k` | 11 | 5 | 1 | 8 |
| AutoSA | `VITIS_LOOP_540_1` | 11 | 5 | 1 | 8 |
| Allo | `l_reduction_k` | 11 | 5 | 1 | 8 |

Eight iterations, one per cycle, five cycles deep, eleven end to end, one
multiplier. Nothing separates the three inside the processing element, so any
difference at the array or kernel level comes from how operands are delivered
and how results are drained, not from the arithmetic.

## Results on the matched workload

Every row computes S x S x S on an S x S array with one multiplier per element.
Cycles are one launch; slack is against a 3.333 ns target, out of context;
frequency is the period that slack implies. `results_matched.csv` is the same
table with its sources.

| Array | Design | Cycles | LUT | FF | DSP | BRAM | Slack | Clock |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | SPMW mesh | 16 | 560 | 1,021 | 16 | 0 | +1.167 ns | 462 MHz |
| | SPMW kernel | 66 | 1,867 | 2,516 | 16 | 0 | +1.301 ns | 492 MHz |
| | AutoSA | 148 | 4,974 | 8,171 | 16 | 1.5 | +0.512 ns | 354 MHz |
| 8x8 | SPMW mesh | 28 | 2,225 | 4,301 | 64 | 0 | +0.996 ns | 428 MHz |
| | SPMW kernel | 100 | 8,015 | 10,220 | 64 | 0 | +1.232 ns | 476 MHz |
| | AutoSA | 292 | 13,393 | 22,924 | 64 | 2.5 | +0.690 ns | 378 MHz |
| 16x16 | SPMW mesh | 52 | 9,026 | 18,048 | 256 | 0 | +0.428 ns | 344 MHz |
| | SPMW kernel | 152 | 32,756 | 40,732 | 256 | 0 | +0.541 ns | 358 MHz |
| | AutoSA | 859 | building | | 256 | | | |
| 32x32 | SPMW mesh | 100 | 37,917 | 74,945 | 1,024 | 0 | +0.431 ns | 345 MHz |
| | SPMW kernel | 298 | 137,741 | 163,708 | 1,024 | 0 | +0.529 ns | 357 MHz |
| | AutoSA | building | | | 1,024 | | | |

*mesh* is the array with its boundary buffers and no memory interface; *kernel*
is the same array with its memory loaders and drain, the scope AutoSA also
reports.

## Why AutoSA takes more cycles, and more area, for the same arithmetic

The processing element is identical in all three systems, so the difference is
entirely in how operands reach the elements and how results leave. AutoSA
builds an explicit multi-level input and output network; SPMW moves results
through the array itself. At 8x8 the generated designs contain:

| Module class | AutoSA instances | What it does |
|---|---:|---|
| `PE_wrapper` | 64 | the elements |
| `C_drain_IO_L1_out` | 64 | one drain module per element |
| `C_drain_IO_L2_out` | 7 | one drain aggregator per row |
| `A_IO_L2_in`, `B_IO_L2_in` | 14 | operand distribution per row and column |
| `A_IO_L3_in`, `B_IO_L3_in` | 2 | the readers that touch memory |

So an operand crosses two levels before it reaches an element, and a result
crosses two more on the way out, where SPMW's mover writes straight into the
array's edge buffer and its results leave along a chain through the elements.
That network is what lets AutoSA scale its off-chip bandwidth independently of
the array, and it costs latency and area: at 8x8 AutoSA uses 1.7 times the
lookup tables and 2.2 times the registers of the SPMW kernel for the same 64
multipliers, needs block RAM where SPMW needs none, and closes at 378 MHz
against 476.

Three further differences are worth separating from the architecture, because
they are measurement or interface rather than design:

- **The memory port.** AutoSA fetches S int8 per beat and drains one int32 at a
  time; SPMW's feeders are generated with a 512-bit port. At 8x8 SPMW fetches an
  operand in one beat where AutoSA takes eight. This is a real difference
  between the two kernels but it is not an array property, and it accounts for
  part of the cycle gap.
- **How the cycles are counted.** SPMW's figure is the start bit to the last
  drain beat; AutoSA's is Vitis' cosimulation latency, which includes the
  testbench's own control transactions. On the fixed-problem runs that
  difference was measured at roughly 60 cycles.
- **Scale.** AutoSA's cycles grow faster than SPMW's with the array: 148, 292,
  859 against 66, 100, 152. Its per-element drain network grows as S squared,
  and its operand distribution as S, so the fixed cost of the network grows
  with the array while the arithmetic per element stays at eight cycles.

## Files

`gen_matched.sh` generates, `hls_matched.sh` synthesises and cosimulates,
`spmw_matched.sh` rebuilds the SPMW kernel at a chosen port width.
`autosa_S4/` holds one matched input and what AutoSA produced from it;
`reports_autosa_S<S>/` the element, kernel and cosimulation reports the numbers
above were read from.
