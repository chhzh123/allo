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

Cycles are **first memory beat to last memory beat**, one definition for every
row: the control writes, the start bit and the polling or handshake that each
flow's own testbench performs are all outside it. SPMW's testbench reports it
directly; AutoSA's comes from a waveform of the same cosimulation snapshot,
logging the same handshakes. Where a flow's own reported figure differs it is
kept beside it, because the gap is the control traffic and is worth seeing.

| Array | Design | Port in/out | Cycles | (flow reports) | LUT | FF | DSP | BRAM18 | Slack | Clock |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | SPMW mesh | streams | 16 | | 560 | 1,021 | 16 | 0 | +1.167 ns | 462 MHz |
| | SPMW kernel | 512/512 | 47 | 66 | 1,867 | 2,516 | 16 | 0 | +1.301 ns | 492 MHz |
| | AutoSA | 32/32 | 78 | 148 | 4,974 | 8,171 | 16 | 3 | +0.512 ns | 354 MHz |
| | Allo | 32/32 | 118 | 167 | 3,693 | 4,459 | 16 | 3 | +0.625 ns | 369 MHz |
| 8x8 | SPMW mesh | streams | 28 | | 2,225 | 4,301 | 64 | 0 | +0.996 ns | 428 MHz |
| | SPMW kernel | 512/512 | 82 | 100 | 8,015 | 10,220 | 64 | 0 | +1.232 ns | 476 MHz |
| | AutoSA | 64/32 | 222 | 292 | 13,393 | 22,924 | 64 | 5 | +0.690 ns | 378 MHz |
| | **AutoSA, wide** | **512/512** | **130** | **202** | 15,343 | 25,536 | 64 | 23 | +0.738 ns | 385 MHz |
| | Allo | 32/32 | 288 | 337 | 8,419 | 8,220 | 64 | 3 | +0.598 ns | 366 MHz |
| 16x16 | SPMW mesh | streams | 52 | | 9,026 | 18,048 | 256 | 0 | +0.428 ns | 344 MHz |
| | SPMW kernel | 512/512 | 133 | 152 | 32,756 | 40,732 | 256 | 0 | +0.541 ns | 358 MHz |
| | AutoSA | 128/32 | 782 | 859 | 47,390 | 81,265 | 256 | 9 | +0.410 ns | 342 MHz |
| | **AutoSA, wide** | **512/512** | **374** | **445** | 48,957 | 84,062 | 256 | 23 | +0.380 ns | 339 MHz |
| 32x32 | SPMW mesh | streams | 100 | | 37,917 | 74,945 | 1,024 | 0 | +0.431 ns | 345 MHz |
| | SPMW kernel | 512/512 | 280 | 298 | 137,741 | 163,708 | 1,024 | 0 | +0.529 ns | 357 MHz |

Port width is read off each design's synthesised RTL
(`C_M_AXI_*_DATA_WIDTH`), never assumed from the source.

Slack is against a 3.333 ns target, routed out of context with nothing
unrouted; the clock is the period that slack implies. `spmw_mesh` is the array
alone and moves no memory beats, so its cycles are the array cosimulation's own
count from first input token to last output token, which is the same idea one
level in. Blank cells are builds still running or, for Allo, a cosimulation
that does not complete in this tool version; none is an estimate.

**Allo, re-measured.** Its rows now carry the same definition, from the same
waveform method, at the two sizes where its cosimulation completes at all;
16x16 and 32x32 produce synthesis reports and no cosimulation report in this
tool version, so those cells stay empty.

Two differences had to be removed to get there, and both had been inflating
Allo's standing in the earlier table:

- Its testbench runs **three seeds per launch**, and the shipped figure was the
  *minimum* of the three (min 136, avg 146, max 167 at 4x4), where SPMW and
  AutoSA each report a single launch. Later launches are faster because the
  cosimulation's memory model is warm. The measurement build runs one seed;
  4x4's flow-reported figure is therefore 167 rather than the 136 previously
  recorded, and 8x8's is 337 rather than 306 (min 306, avg 316, max 337). The
  single-launch figure equals that run's own maximum at both sizes, which is
  the first launch: the later ones are the fast ones. The three-seed run
  remains the correctness evidence.
- First beat to last across three launches is not the quantity the other two
  report, so the window had to come from a single-launch build regardless.

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

**The memory port, which is not an array property.** The three systems reach
DRAM through very different ports, and none of that is a property of the array:

| | in | out |
|---|---:|---:|
| SPMW kernel | 512 b | 512 b |
| AutoSA, as generated | S x 8 b | 32 b |
| AutoSA, `--host-serialize` | 512 b | 512 b |
| Allo | 32 b | 32 b |

SPMW's width is structural: `BEAT = 512` in `allo/spmw/shell.py` is a constant
that becomes the `ap_uint<512>` feeder port, not a synthesis ceiling -- setting
`-m_axi_max_widen_bitwidth` to 32 still produced a 512-bit port. AutoSA's is a
command-line knob: its `data_pack` bound is already 64 bytes, and it reaches it
once `--host-serialize` lays the operands out in the order the array consumes
them. That is the same assumption SPMW's shell already makes, in its own words:
"The host lays each family's tokens out in the order its channels consume
them." So the match is symmetric, and neither figure includes host-side layout.

Widening AutoSA at 8x8 takes it from 8 + 8 + 64 beats to 1 + 1 + 4 -- the same
six beats SPMW moves -- and from 222 cycles to 130, against SPMW's 82. Roughly
half the gap was the port. The half that remains is the network above.

4x4 cannot be widened at all: a 4x4 int8 operand is 16 bytes, less than one
beat, so the packed loop's trip count is zero and AutoSA's bit-width shrinker
divides by zero. It fails the same way at a 16- and a 32-byte bound, so it is
the serializer rather than the pack size, and that row stays narrow.

Allo is the narrowest of the three at 32 bits on all three ports, and its beat
census is what that predicts: 4 + 4 + 16 at 4x4 and 16 + 16 + 64 at 8x8, so 96
transactions where SPMW moves 6. It has not been rebuilt wide; unlike AutoSA it
has no flag for this, because its ports come from `int8_t *` scalars and the
width would have to come from Vitis's own widening.

Its waveform also shows the operands loaded **sequentially** rather than
together: A's beats land at 79-94 and B's at 157-172 at 8x8, `load_buf0`
draining fully before `load_buf1` starts, where AutoSA fetches both at once.
That serialization is a dataflow opportunity the generated code does not take,
and it is a large part of the 288.

The wide port is not free, which is the part worth reporting. Buying those 92
cycles cost AutoSA 15 per cent more lookup tables, 11 per cent more registers
and **4.6 times the block RAM** -- 23 tiles against 5 -- because the wider
masters need the staging to match. Its clock moved the right way but barely,
378 to 385 MHz. So at a matched 512-bit interface the comparison at 8x8 is:

| Both at 512 bits | Cycles | LUT | FF | BRAM18 | Clock |
|---|---:|---:|---:|---:|---:|
| 8x8 SPMW kernel | 82 | 8,015 | 10,220 | 0 | 476 MHz |
| 8x8 AutoSA, wide | 130 | 15,343 | 25,536 | 23 | 385 MHz |
| 16x16 SPMW kernel | 133 | 32,756 | 40,732 | 0 | 358 MHz |
| 16x16 AutoSA, wide | 374 | 48,957 | 84,062 | 23 | 339 MHz |

The wide port's own overhead is mostly fixed rather than proportional, which is
visible once there are two sizes: it costs 15 per cent of the lookup tables at
8x8 but 3 per cent at 16x16, and the same 23 block RAM tiles at both, against 5
and 9 narrow. The staging it adds is sized by the 512-bit port, not by the
array. Its effect on the clock is small and not consistently signed: 378 to 385
MHz at 8x8, 342 to 339 at 16x16.

At 16x16 the same change takes AutoSA from 782 beats-cycles to 374 and from
859 flow-reported to 445, again on exactly the arithmetic: 4 + 4 + 16 beats,
the 24 SPMW moves. But the gap that survives the match *grows* with the array,
1.6x at 8x8 and 2.8x at 16x16 against SPMW's 82 and 133, which is what a drain
network growing as S squared predicts and a port width does not.

Matching the port narrowed the cycle gap at 8x8 from 2.7x to 1.6x and widened
the area gap: 1.9 times the lookup tables and 2.5 times the registers for the same 64
multipliers, and block RAM where SPMW uses none. Both directions are the same
cause -- SPMW's feeders write into the array's edge FIFOs, so a 512-bit port
needs no staging behind it, while AutoSA's has to feed a level that then feeds
another.

**How the cycles are counted, now settled.** The table's first cycle column is
one definition on both sides, so the control traffic that used to differ is no
longer in it. Measuring it cost SPMW 19 cycles of apparent advantage at every
size and AutoSA between 71 and 78, which is roughly what the earlier record
guessed for AutoSA and had not accounted for at all on the SPMW side. The
remaining gap is design and interface, not measurement.

## Reproducing

Each `source/` holds the input and, for AutoSA, the exact command in
`autosa_command.txt`, including the space-time transform and the array
partition. The three scripts that drove these runs are in `scripts/`.
