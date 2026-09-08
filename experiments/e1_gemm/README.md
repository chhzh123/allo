# E1: an S x S array computing S x S x S, in four designs

Every design here computes the same thing: `C = A B` for 8-bit signed inputs
and 32-bit signed outputs, with `M = N = K = S`, on an array of S x S elements
holding one multiplier each. Earlier runs did not do this; AutoSA was given a
fixed 16-cubed problem while the others computed one S-cubed tile, so its
launch did up to sixty-four times the arithmetic of the row beside it. These
runs fix that, and `results.csv` is the table.

## Layout

Every framework has the same shape, one directory per array size:

    <framework>/S<size>/source/      what the framework was given
    <framework>/S<size>/generated/   what it emitted
    <framework>/S<size>/report/      the reports each number was read from

| Folder | Design |
|---|---|
| `spmw_mesh/` | the SPMW array with its boundary buffers, no memory interface |
| `spmw_kernel/` | the same array packaged with its memory loaders and drain |
| `autosa/` | AutoSA's generated design, its own memory interface included |
| `allo/` | Allo's library systolic array, its own memory interface included |

`spmw_mesh` is the only **array-scope** row. The other three are **kernel
scope**: they include each system's own memory interface, and those interfaces
differ, which the last section explains.

## Results

| Array | Design | Cycles | LUT | FF | DSP | BRAM18 | Slack | Clock |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | SPMW mesh | 16 | 560 | 1,021 | 16 | 0 | +1.167 ns | 462 MHz |
| | SPMW kernel | 66 | 1,867 | 2,516 | 16 | 0 | +1.301 ns | 492 MHz |
| | Allo | 136 | 3,693 | 4,459 | 16 | 3 | +0.625 ns | 369 MHz |
| | AutoSA | 148 | 4,974 | 8,171 | 16 | 3 | +0.512 ns | 354 MHz |
| 8x8 | SPMW mesh | 28 | 2,225 | 4,301 | 64 | 0 | +0.996 ns | 428 MHz |
| | SPMW kernel | 100 | 8,015 | 10,220 | 64 | 0 | +1.232 ns | 476 MHz |
| | AutoSA | 292 | 13,393 | 22,924 | 64 | 5 | +0.690 ns | 378 MHz |
| | Allo | 306 | 8,419 | 8,220 | 64 | 3 | +0.598 ns | 366 MHz |
| 16x16 | SPMW mesh | 52 | 9,026 | 18,048 | 256 | 0 | +0.428 ns | 344 MHz |
| | SPMW kernel | 152 | 32,756 | 40,732 | 256 | 0 | +0.541 ns | 358 MHz |
| | AutoSA | 859 | routing | | 256 | | | |
| 32x32 | SPMW mesh | 100 | 37,917 | 74,945 | 1,024 | 0 | +0.431 ns | 345 MHz |
| | SPMW kernel | 298 | 137,741 | 163,708 | 1,024 | 0 | +0.529 ns | 357 MHz |

Cycles are one launch. Slack is against a 3.333 ns target, routed out of
context; the clock is the period that slack implies. Blank cells are builds
still running, never estimates. Allo's 16x16 cosimulation does not complete in
this tool version: Vitis stops writing files inside testbench generation and
spins, which is recorded in its package README.

## The element is identical in all four

The reduction loop is what has to pipeline, and every system pipelines it the
same way. Read from the reports in each `report/` directory:

| System | Loop | Latency | Iteration latency | Interval | Trip |
|---|---|---:|---:|---:|---:|
| SPMW | `l_S_k_0_k` | 11 | 5 | 1 | 8 |
| AutoSA | `VITIS_LOOP_540_1` | 11 | 5 | 1 | 8 |
| Allo | `l_reduction_k` | 11 | 5 | 1 | 8 |

Eight iterations, one per cycle, five cycles deep, one multiplier. So no
difference in the table above comes from the arithmetic.

## Where the differences do come from

**Operand delivery and result draining.** AutoSA builds an explicit multi-level
network; SPMW moves results through the array itself. At 8x8 AutoSA instantiates
64 elements, 64 per-element drain modules, 7 row drain aggregators, 14 operand
distributors and 2 memory readers, so an operand crosses two levels to reach an
element and a result crosses two more leaving. That is what lets AutoSA scale
off-chip bandwidth independently of the array, and it costs latency, area and
clock: 1.7 times the lookup tables and 2.2 times the registers of the SPMW
kernel for the same 64 multipliers, block RAM where SPMW needs none, and 378 MHz
against 476. Its cost also grows with the array, because the drain network grows
as S squared while the arithmetic per element stays at eight cycles: 148, 292,
859 against 66, 100, 152.

**The memory port, which is not an array property.** AutoSA fetches S int8 per
beat and drains one int32 at a time; SPMW's feeders are generated with a 512-bit
port, because `BEAT = 512` in `allo/spmw/shell.py` is a constant rather than a
synthesis ceiling. Setting `-m_axi_max_widen_bitwidth` to 32 still produced a
512-bit port. So at 8x8 SPMW reads an operand in one beat where AutoSA takes
eight, and part of the cycle gap is that rather than the design.

**How the cycles are counted.** SPMW's figure is the start bit to the last drain
beat. AutoSA's and Allo's are their own flows' cosimulation latency, which
includes the testbench's control transactions; on earlier runs that was worth
about 60 cycles.

## Reproducing

Each `source/` holds the input and, for AutoSA, the exact command in
`autosa_command.txt`, including the space-time transform and the array
partition. The three scripts that drove these runs are in `scripts/`.
