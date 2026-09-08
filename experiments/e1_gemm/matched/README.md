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

## Kernel-level cycles on the matched workload

| S | SPMW kernel | AutoSA kernel |
|---|---:|---:|
| 4 | 66 | 148 |
| 8 | 100 | 292 |

**Do not read these as an architecture comparison.** They include each system's
own memory interface, and the two interfaces differ:

- AutoSA feeds S int8 elements per beat for each operand (`A_t8` at S=8) and
  drains one int32 at a time.
- SPMW's feeders are generated with a 512-bit port by construction. `BEAT = 512`
  in `allo/spmw/shell.py` is a module constant, not a synthesis ceiling, so
  setting `-m_axi_max_widen_bitwidth` lower does not shrink it: a run with the
  ceiling at 32 still synthesised `C_M_AXI_GMEM_DATA_WIDTH = 512`. Matching the
  width properly means changing how the feeder is generated, which also moves
  the host buffer layout, and was not attempted here.

So at S=8 SPMW fetches each operand in one 512-bit beat where AutoSA takes
eight 64-bit beats. That is a real difference between the two kernels, and it
is most of the gap in the table above. The measurements are also defined
slightly differently: SPMW's is the start bit to the last drain beat, AutoSA's
is Vitis' cosimulation latency, which includes the testbench's own control
transactions.

## Where the systems are already matched

At the **array boundary** both consume S int8 per cycle for each operand, so a
comparison there needs no interface at all. SPMW's stream-fed mesh measures
exactly that: 16, 28, 52 and 100 cycles for one S-cubed tile at S = 4, 8, 16
and 32. The equivalent number for AutoSA is not separable from its report
without instrumenting its processing-element array, and is left unmeasured
rather than estimated.

## Files

`gen_matched.sh` generates, `hls_matched.sh` synthesises and cosimulates,
`spmw_matched.sh` rebuilds the SPMW kernel at a chosen port width.
`autosa_S4/` holds one matched input and what AutoSA produced from it;
`reports_autosa_S<S>/` the element, kernel and cosimulation reports the numbers
above were read from.
