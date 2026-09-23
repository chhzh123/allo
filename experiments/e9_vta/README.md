# E9: VTA at 16x16, against Gemmini and SPMW

Same workload as E8, same device, same measurement discipline. VTA's default
configuration already matches the other two where it counts -- `batch = 1`,
`blockIn = blockOut = 16`, int8 operands into an int32 accumulator -- so this
is three 16x16 engines, 256 multiply-accumulates each, on one transformer
block.

## The answer, on two workloads

**One workload cannot settle this, because the three engines do not have the
same ISA scope.** So there are two, and they say different things.

### 1. The intersection, where the comparison is exact

E3's microbenchmark -- tiled int8 GEMM with bias, ReLU, requantise and clip,
all inside every one of the three ISAs. All bit-exact against one golden, at
16x16, routed at 3.333 ns, counting the design and not the harness around it:

| | cycles / tile | LUT | FF | DSP | clock | ns / tile | LUT x ns |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPMW, E3's fixed cell | 16 | 92,880 | 130,776 | 0 | 349 MHz | 45.8 | 4.26 M |
| **SPMW, 4x4 cells a unit** | **16** | **23,605** | **17,772** | 0 | **333 MHz** | 48.0 | **1.13 M** |
| SPMW, the array one unit | 16 | 22,572 | 11,118 | 0 | 311 MHz | 51.4 | 1.16 M |
| Gemmini `MxuVpu` | 18 | 31,932 | 18,675 | 0 | 324 MHz | 55.6 | 1.77 M |
| VTA | 65 | 26,403 | 5,991 | 0 | 300 MHz | 216.3 | 5.71 M |

**Written the way Gemmini writes its processing element, SPMW is smaller than
Gemmini and faster on every column**: 26% fewer lookup tables, 5% fewer
registers, two fewer cycles a tile, a 3% faster clock, and 14% less time a
tile. At the other end of the same program -- the whole array one unit -- it
has 15% fewer lookup tables than VTA and a quarter of its cycles; its
registers are 1.9x VTA's, 4,096 of them the double-buffered weights that VTA
keeps in block RAM.

The first version of this table had SPMW at 2.9x Gemmini's lookup tables and
7x its registers. That was how the SPMW design was written and scheduled, not
what SPMW is: [Closing the gap](#closing-the-gap) takes it apart one measured
step at a time.

### 2. The transformer block, where scope decides it

| | cycles | on-chip coverage | LUT | FF | DSP | clock | time |
|---|---:|---|---:|---:|---:|---:|---:|
| VTA | 210,944 | GEMM + requantise only | **26,403** | **5,991** | **0** | 300.5 MHz | 0.702 ms* |
| Gemmini | 267,776 | **all of it** | 71,389 | 21,474 | 500 | 34.8 MHz | 7.687 ms |
| SPMW, file | 278,912 | **all of it** | 113,341 | 153,869 | 755 | **306.7 MHz** | 0.910 ms |
| SPMW, reload | **220,032** | **all of it** | 144,734 | 186,540 | 1,011 | 304.5 MHz | **0.723 ms** |

\* VTA's time is for 46% of the scale path; 114,688 elements have no unit.

**Only Gemmini and SPMW run the whole block.** VTA's ALU is `min`, `max`,
`add`, `shift` and no multiplier, so softmax and IGELU are impossible on it
and LayerNorm is expressible only at a large penalty. That is a scope
difference, not a defect -- VTA is a quantised-CNN accelerator and these
operations postdate it.

SPMW is shown twice because the weight discipline is a real choice:
**reload** is Gemmini's own -- the next tile's weights shift in behind the
current tile's arithmetic -- and buys 4 cycles a tile (20.6 -> 16.0, a 21%
shorter block) for **28% more lookup tables, 21% more registers and 256 more
multipliers**. Both route with zero unrouted nets. It is a real choice, not a
free win, and both rows are kept so it stays one.

### Reading the two together

- **Mesh throughput:** SPMW `S`, Gemmini `S + 2`, VTA `4S + 1` on the
  microbenchmark; on the GEMM alone, SPMW and VTA tie at 16.0 and Gemmini is
  18.0.
- **Area:** on the microbenchmark, SPMW with 4x4 cells a unit is below
  Gemmini -- 23,605 lookup tables against 31,932 -- and VTA is below both.
  The transformer-block engine still uses E3's cell: 113,341 against 71,389.
- **Efficiency:** SPMW with 4x4 cells a unit, on lookup tables times time a
  tile, by 1.6x over Gemmini and 5.0x over VTA.
- **Coverage:** Gemmini and SPMW all of it, VTA 46% of the scale path.
- **Clock on this FPGA:** SPMW and VTA both near 300 MHz; Gemmini 34.8 MHz,
  which is a porting artifact of an unpipelined float scale path and not an
  architectural result.

### How far out of reach, exactly

Not equally, and the earlier version of this file was too blunt in saying
114,688 elements "go to the host". Separating them:

| pass | elements | on VTA |
|---|---:|---|
| 4 requantisations | 98,304 | **runs**, `shift` |
| 4 softmaxes | 16,384 | **impossible** |
| 1 IGELU | 65,536 | **impossible** |
| 2 LayerNorms | 32,768 | expressible at a penalty |

`iexp` and `igelu` are second-order polynomials **of each element**, so every
element needs its own square. The GEMM cannot square a vector elementwise --
that needs a diagonal built from the data, and building it needs a multiply
VTA does not have. Those two are genuinely out of reach: **81,920 elements**.

LayerNorm is different. Its sum is an ALU `add`; its sum of squared
deviations is a vector dotted with itself, which is exactly what the GEMM
computes if the deviations are int8; its two per-row scalars are 128 values
for the whole block, cheap anywhere; and the final elementwise scale can go
through the GEMM as a diagonal matrix, at roughly 16x the work. So it is
awkward and slow rather than impossible, and this experiment did not build
it -- the row is left empty rather than guessed at.

Where the unsupported work actually runs is outside this experiment. VTA's
own stack partitions the graph and gives unsupported operators to the host,
but nothing here measures that, so no transfer cost is claimed.

## The fair comparison: one workload all three natively run

Comparing on a transformer block favours the two engines built for one.
E3's microbenchmark does not: a tiled int8 GEMM with bias, ReLU, a
requantising shift and a clip to int8, all on device. Every operation in it
is inside VTA's `minpool / maxpool / add / shift` ALU, so nobody is asked to
do something their ISA has no word for -- and Gemmini and SPMW were already
measured on it, against a shared stimulus file and a shared golden.

VTA runs it **bit-exact at every width** -- 16 tiles, zero errors against the
same `stim_S*.txt` the other two were checked on.

| S | | cycles / tile | LUT | FF | DSP | slack |
|---:|---|---:|---:|---:|---:|---:|
| 4 | SPMW, fixed | **4** | 5,818 | 7,568 | 0 | +0.834 |
| | Gemmini | 6 | **2,132** | 1,378 | 0 | +0.591 |
| | VTA | 17 | 3,121 | **1,283** | 0 | **+0.956** |
| 8 | SPMW, fixed | **8** | 22,922 | 32,116 | 0 | +0.328 |
| | Gemmini | 10 | **8,042** | 4,818 | 0 | +0.456 |
| | VTA | 33 | 8,326 | **2,979** | 0 | **+0.457** |
| 16 | SPMW, fixed | **16** | 92,880 | 130,776 | 0 | **+0.468** |
| | Gemmini | 18 | 31,932 | 18,675 | 0 | +0.245 |
| | VTA | 65 | **26,403** | **5,991** | 0 | +0.005 |

Each is an exact law, which is what a clean measurement looks like:

| | cycles a tile | what it is |
|---|---|---|
| SPMW, fixed | **`S`** | one output row a cycle, epilogue fused into the lane |
| Gemmini | **`S + 2`** | `S` rows plus a two-cycle request handshake |
| VTA | **`4S + 1`** | one GEMM pass and **three** ALU passes |

**SPMW is fastest at every width, VTA is smallest at 16 and has the fewest
registers at every width, Gemmini has the best of both.** On area-delay
product at 16x16 -- lookup tables times cycles a tile -- Gemmini wins by 2.6x
over SPMW and 3.0x over VTA:

| | LUT x cycles/tile |
|---|---:|
| Gemmini | **0.57 M** |
| SPMW, fixed | 1.49 M |
| VTA | 1.72 M |

### Why VTA is `4S + 1` on a workload it fully supports

Not the mesh. On the GEMM alone VTA and SPMW **tie at 16.0 cycles a tile**,
which is the arithmetic floor -- a 16x16x16 tile is 4,096 multiplies and the
array does 256 a cycle -- and Gemmini is 18.0, two over, for its per-tile
request handshake. VTA has no throughput advantage over a systolic array; it
has no weight-load fill and no inter-tile handshake, which is worth 0 and 2
cycles respectively. What costs it `4S + 1` is the epilogue:

    16  GEMM
  + 16  ALU pass: max, the ReLU
  + 16  ALU pass: shr, the requantise
  + 16  ALU pass: min, the clip to 127
  ----
    65  measured as 1050 cycles for 16 tiles at S = 16

**VTA's ALU is a load-store unit over the accumulator scratchpad: one opcode
per instruction, one pass over the data each.** Gemmini folds bias, ReLU,
shift and clip into `AccumulatorScale` on the output path and SPMW folds them
into its scale lane, so for both the epilogue is *free* -- it happens as the
results drain. VTA reads the accumulator back three times.

Nor can the passes hide behind the matmul. `Compute.scala` asserts
`!tensorGemm.io.uop.idx.valid || !tensorAlu.io.uop.idx.valid`: the two units
share the micro-op port and never run in the same cycle. Measured directly,
chained ALU instructions are strictly additive -- 258, 521 and 784 cycles for
one, two and three passes over 256 rows, 1.03 cycles a row each.

The bias is free on all three and nobody is charged for it: Gemmini and SPMW
fold it into the epilogue, VTA preloads the accumulator.

## Where the area gap comes from

A 3.5x lookup-table and 22x register gap between E3's SPMW cell and VTA
needs an account. Normalising by the 256 multiply-accumulates each of them performs:

| S | | LUT / MAC | | | FF / MAC | |
|---:|---|---:|---:|---:|---:|---:|
| | | VTA | Gemmini | SPMW | | |
| 4 | | 195 | 133 | 364 | 80 / 86 / 473 | |
| 8 | | 130 | 126 | 358 | 47 / 75 / 502 | |
| 16 | | 103 | 125 | 363 | **23 / 73 / 511** | |

Two different effects, and only one of them is about tooling.

**VTA against Gemmini is a dataflow difference, and it grows with size.**
VTA's per-MAC registers *fall* -- 80, 47, 23 -- because its
`MatrixVectorMultiplication` is a **combinational adder tree between two
memories**: no MAC holds state, and what registers exist are a fixed overhead
(accumulator pipes, the index generator) amortised over `S^2` multipliers.
Gemmini's stay flat at 73-86 because it is a **systolic array**: every PE
holds its own weight and partial sum, so the cost is per cell and never
amortises. At S=4 they are within 1.1x; at S=16 it is 3.1x, and it would keep
growing.

**Gemmini against SPMW was how the SPMW design was written, and it closes.**
Both are systolic and both pay per cell, and E3's SPMW cell was 511 flip-flops
against Gemmini's 73. Routed at 16x16 and split by hierarchy:

| a cell | LUT | FF | |
|---|---:|---:|---|
| E3's cell body | 309 | 362 | a 16-tile packed weight file, a barrel shifter unpacking it every beat, three loops with 32-bit counts, and HLS's four-stage pipeline |
| its three links | 50 | 141 | `a` 18, `p` 62, `w` 66: two entries each -- `q0` and the skid `q1` -- and their valids |
| Gemmini's whole PE, for scale | 119 | 73 | two weight registers, a multiply-add, a mux |

**A correction.** The previous version of this file said the skid cost 5
flip-flops a cell because Vivado had already trimmed it. It had not. The
experiment behind that claim -- every link rebuilt one register deep --
changed only the 16 lane links: a link's depth was `max(Out.depth,
In.depth)`, `Out` defaulted to 2, and so `In(depth=1)` on a mesh link was
silently raised back to 2. The 84 flip-flops "saved" at 4x4 were exactly four
lane links going from 42 to 21. A link's depth is now the deepest any end
*asked* for (`allo.spmw.ports.link_depth`, tested in `test_spmw_rtl.py`), and
every int8 link measures 18 flip-flops, skid included.

## Closing the gap

Three changes, in order. Each row is a separate build of the same workload,
with the same operands and golden, cosimulated clean -- 4,096 of 4,096 tokens
-- and routed at 3.333 ns.

| 16x16 | cycles / tile | first out | LUT | FF | clock |
|---|---:|---:|---:|---:|---:|
| E3's fixed cell, rebuilt | 16.0 | 157 | 92,961 | 130,928 | 343 MHz |
| **1.** Gemmini's weight discipline | 16.1 | 72 | 47,495 | 61,604 | 348 MHz |
| **2.** ...scheduled between its links | 16.0 | 40 | 47,682 | 38,684 | 320 MHz |
| **3.** ...2x2 cells a unit | 16.0 | 40 | 34,770 | 29,128 | 337 MHz |
| **3.** ...4x4 cells a unit | 16.0 | 32 | **23,605** | **17,772** | **333 MHz** |
| **3.** ...8x8 cells a unit | 16.0 | 32 | 24,647 | 14,287 | 322 MHz |
| **3.** ...the whole array one unit | 16.0 | 28 | 22,572 | 11,118 | 311 MHz |
| Gemmini `MxuVpu` | 18 | | 31,932 | 18,675 | 324 MHz |
| VTA | 65 | | 26,403 | 5,991 | 300 MHz |

**1. The weight, held the way Gemmini holds it.** E3's cell keeps all sixteen
tiles resident, four to a packed word, and unpacks a byte out of them every
beat. Gemmini's PE holds one int8 and a second behind it: the next tile's
weight shifts down the row *through* the cells while this tile computes, and a
flag swaps them. The lean cell (`test_spmw_tpu_micro_lean.py`) does the same:
every beat it passes its `nxt` on and takes a new one, so after `S` beats
cell `j` holds the row's `(S-1-j)`-th token, and at the tile boundary
`cur = nxt`. No file, no count, no knowledge of its position -- every cell is
identical, and its weight link is an int8 rather than an int32. That halves
both columns.

**2. Not pipelining on top of the links.** Vitis charges every FIFO access
1.42 ns, so read, multiply, add and write -- 5.06 ns in its model -- cannot
share one 3.33 ns stage, and it registers `a`, `p`, the product and the sum.
But an SPMW link *is* a register: its `dout` is a flop and its `din` lands in
one. `spmw.pipeline(P, ii=1, combinational=True)` credits each stage the link
accesses it touches, and the cell becomes one multiply-add between two link
registers -- 143 flip-flops to 52 -- which is Gemmini's `tile_latency = 0`.
The routed critical path is Gemmini's as well: `a`'s link register, the LUT
multiplier, four CARRY8, `p`'s link register; 3.0 ns, two-thirds of it
routing. First output falls from 72 cycles to 40.

**3. Fewer, larger units.** After step 2 the links are the largest item --
22,524 of the array's 38,684 flip-flops, more than all of Gemmini. An
`f x f` block of lean cells in one unit has `f` links of each kind where its
cells had `f^2`, and one loop and one pipeline's control where they had
`f^2`. `fused_engine` generates the block for any `f`, its column sums a
balanced tree.

A block needs more than one stage, and that changes the credit. HLS
schedules every stage against one clock, so the credit has to be what
*every* stage touches: both accesses for a one-stage body, one when there are
two stages -- the first reads, the last writes -- and none once there is a
middle stage that touches neither. Measured:

| 16x16 | both | one | none |
|---|---:|---:|---:|
| the lean cell, one stage | **320 MHz** | | 348 MHz, 23k more FF |
| 4x4 blocks, two stages | 260 MHz | **333 MHz** | |
| 8x8 blocks, three stages | | 258 MHz | **322 MHz** |
| 16x16 in one unit, three stages | | 262 MHz | **311 MHz** |

Over-crediting is not a subtle failure: it packs a multiply and two adder
levels into one stage, 3.7-3.9 ns routed. `combinational=True` asks for
both, `registered_links=True` for one, and the default for none.

Four by four cells a unit is the headline: **smaller than Gemmini by 26% in
lookup tables and 5% in registers, two cycles a tile fewer, a 3% faster clock
-- 48.0 ns a tile against 55.6.**

### What did not work, and why

| 16x16 | clock | LUT | FF | |
|---|---:|---:|---:|---|
| lean, links one register deep | **63.7 MHz** | 41,435 | 27,565 | the ready is combinational through the array |
| 2x2 blocks, one register deep | 94.5 MHz | 30,893 | 20,590 | the same, both accesses credited |
| 4x4 blocks, one register deep | 207.7 MHz | 20,686 | 12,504 | the same, both accesses credited |
| lean, Gemmini's epilogue shape | 326.9 MHz | 50,547 | 39,392 | the bias un-folds row 0 |
| 4x4 blocks, both accesses credited | 259.5 MHz | 22,748 | 15,483 | a 3.85 ns first stage |
| ...and Vivado retiming | 266.0 MHz | 22,763 | 15,699 | the stage register did not move |
| 8x8 blocks, one access credited | 258.1 MHz | 23,062 | 12,669 | a middle stage got the credit |
| 16x16 in one unit, one access credited | 261.6 MHz | 22,846 | 9,544 | the same |

**A link cannot be one register while it can say "stop".** Without the skid,
`full_n` is `~v0 | read`, and `read` is the consumer's pipeline enable, which
depends on the consumer's own outputs' `full_n`: a combinational chain through
every cell a stall can cross. At 4x4 it is short, and the one-register array
closes at 334 MHz with the same cycle count and half the link registers. At
16x16 the worst path is 80 LUT levels and 15.6 ns. Gemmini's mesh has no such
chain because it has no backpressure: data moves every cycle and its validity
travels with it. A one-register SPMW link needs the same thing -- a static
schedule, with the edges skewed and one stall for the whole array -- which
SPMW does not have. **The skid is what backpressure costs at this clock.**

**Gemmini's epilogue shape does not transfer.** Feeding the bias in at the
top of the array, as Gemmini's `b` input does, saves each lane an adder and
costs 2,865 lookup tables: SPMW's top row reads a constant-zero stream that
Vivado folds away, and the bias un-folds it.

### The three are points on one spectrum -- and SPMW now spans it

    VTA        f = S    one unit, a combinational tree, weights in a scratchpad
    Gemmini    f = 1    per-cell state, a bare register between cells, no stall
    SPMW       f = 1..S one program, with the block size a parameter

| 16x16 | LUT / MAC | FF / MAC |
|---|---:|---:|
| SPMW, E3's cell | 363 | 511 |
| SPMW, lean cell | 186 | 151 |
| SPMW, 4x4 cells a unit | 92 | 69 |
| SPMW, the array one unit | 88 | 43 |
| Gemmini | 125 | 73 |
| VTA | 103 | 23 |

Two things remain. The fusion is written by the design, not the compiler:
`fused_engine` generates the block's source, while `spmw.place` declares
`unroll` for exactly this and `driver.py` still refuses it, so SPMW cannot yet
take a 1x1 cell and a block size and produce that unit itself. And VTA's
register column stays out of reach while SPMW keeps its weights in the array:
two int8 a cell, double-buffered, are 4,096 flip-flops that VTA holds in 70
block RAM tiles outside the measured scope.

## What the missing capability actually costs

"VTA cannot" is not a useful end point. The useful question is what it would
cost VTA to be able to, and that is measurable: VTA already has a **variable**
shift -- `io.a >> n`, shifting by the operand, not an immediate -- so for
I-BERT's `iexp` and `igelu` the **only** missing primitive is a multiply.

`scripts/patch_vta_alu.py` adds one opcode to `TensorAlu`, guarded by
`VTA_ALU_MUL` so the baseline elaborates untouched, and routes it the same way:

| | LUT | FF | DSP | CLB | WNS | clock |
|---|---:|---:|---:|---:|---:|---:|
| `TensorAlu` | 4,805 | 1,681 | 0 | 852 | +0.346 | 334.8 MHz |
| ...with a multiply | 5,763 | 1,681 | **48** | 1,048 | **+1.127** | **453.2 MHz** |
| cost | **+958** | 0 | +48 | +196 | | |

**One opcode -- 958 lookup tables and 48 multipliers, 3.6% of VTA's
26,403-lookup-table datapath -- is what stands between it and computing
softmax and GELU on chip.** Not a register more, and the clock *improves*,
because the multiply lands in DSPs and takes work off the lookup-table path
that was setting the critical path before. Both variants route with zero
unrouted nets.

Set against what the other two spend for the same capability -- Gemmini's 500
DSPs and its 34.8 MHz clock, SPMW's extra 87,000 lookup tables -- that is the
number this experiment is actually for. It does not make VTA able to run the
block: LayerNorm still needs a square root and a reciprocal, and nothing here
builds the micro-op sequences. But it prices the gap instead of asserting it.

## The mesh, measured on all three

| engine | cycles per 16x16x16 tile | why |
|---|---:|---|
| **VTA** | **16.0** | reads the whole 16x16 weight matrix from a scratchpad every cycle -- no load to amortise |
| **SPMW**, reload | **16.0** | 16 steps, the next tile's weights shifted in behind the arithmetic |
| Gemmini | 18.0 | 16 rows plus a two-cycle request handshake |
| SPMW, file | 20.6 | 16 steps plus a serial weight-file load, proportional to the tile count |

16.0 is the floor -- 4,096 multiplies a tile at 256 a cycle -- so VTA and
SPMW's reload form are both *at* it and Gemmini is two over. Nobody is beating
a systolic array here.

VTA's is exactly 16.0, at 16, 32 and 64 tiles -- 258, 514, 1026 cycles, two of
fill. It wins because it is **not weight-stationary**: `TensorGemm` has 256
weight input ports and re-reads the entire matrix each cycle, so there is no
load to amortise and no handshake between tiles. What it spends instead is
**2048 bits a cycle of weight bandwidth** from a scratchpad that this scope
does not count, where Gemmini and SPMW hold one weight per cell inside the
array that this scope does count.

That is the honest reading of VTA's 26,403 lookup tables: some of the
difference is that it does less, and some is that its weight storage is
outside the module being measured. `Core` puts a number on the second part --
**70 block RAM tiles and 12 URAM** of scratchpad, against zero block RAM on
either of the other two.

## Scope

`TensorGemm` + `TensorAlu` is VTA's datapath, and it is the counterpart of
Gemmini's `MxuVpuNorm` and SPMW's block engine -- arithmetic, no scratchpads,
no instruction fetch. `Core`, which adds fetch, load, store and every
scratchpad, is routed separately for context and is a larger scope than
either of the others.

All three VTA modules route with **zero unrouted nets**, and none of them
spends a DSP -- Vivado maps all 256 int8 multiplies to logic, as it does for
Gemmini's mesh.

| module | LUT | FF | DSP | BRAM | URAM | WNS | clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| `TensorGemm` | 21,598 | 4,310 | 0 | 0 | 0 | +0.005 | 300.5 MHz |
| `TensorAlu` | 4,805 | 1,681 | 0 | 0 | 0 | +0.346 | 334.8 MHz |
| datapath, the two together | **26,403** | **5,991** | **0** | **0** | 0 | | 300.5 MHz |
| `Core`, the whole engine | 34,174 | 6,963 | 0 | **70** | 12 | -1.515 | 206.3 MHz |

`Core` is worth reading twice. **VTA's entire engine -- fetch, load, store,
compute and every scratchpad -- is 34,174 lookup tables, less than half
Gemmini's datapath alone.** The scratchpads it adds over the datapath cost
**70 block RAM tiles and 12 URAM**, and that is exactly the storage Gemmini
and SPMW put in registers inside the array: both of those use *zero* block
RAM and pay for it in the logic columns.

The whole engine also misses 300 MHz where the datapath makes it: `Core`
closes at 206.3 MHz against `TensorGemm`'s 300.5, the critical path being
two-thirds routing.

## Two things that cost time

**The HLS path does not build under Vitis 2023.2.** `hardware/xilinx/src/vta.cc`
parses, but *no* function in it registers as a valid top -- `vta`, `fetch`,
`load`, `compute`, `store`, `gemm` and `alu` all report "Could not apply TOP
directive", with empty clang logs and no diagnostic. The Chisel path is the
one used here, and it is VTA's maintained implementation anyway.

**`CoreConfig` alone does not elaborate.** `TensorParams` reads the memory
interface width from `ShellKey`, so it needs `new CoreConfig ++ new F1Config`.
F1 is VTA's Xilinx PCIe card -- 64-bit AXI address and data -- and the closest
of its three shells to a U280; `PynqConfig` is a 32-bit Zynq and `De10Config`
is Intel.

## Files

- `source/ElaborateVTA.scala` -- emits `TensorGemm`, `TensorAlu` and `Core`.
- `scripts/elab_vta.sh` -- Chisel to Verilog, own ivy cache.
- `scripts/pnr_vta.sh` -- out-of-context route, the same recipe as E8's.
- `scripts/vta_gemm_bench.py` -- the xsim bench for the GEMM core; it models
  all four scratchpads, which is why it is 657 port connections.
- `scripts/vta_alu_bench.py` -- the same for the tensor ALU.
- `scripts/vta_totals.py` -- the three-way block table; imports E8's workload
  so all three engines are counted against the same tiles and passes.
- `tests/dataflow/spmw/test_spmw_tpu_micro_lean.py` -- the lean cell, its
  Gemmini-shaped epilogue option, and `fused_engine`, the `f x f` block
  generator; designs `tpumicro-lean`, `-lean-hls`, `-lean1`, `-lean-g`,
  `-fused<f>[-d<depth>]` in `scripts/spmw_build_array.py`.
- `scripts/lean_collect.py` -- one row per build: the `dut` instance's
  lookup tables and registers, the steady interval from tiles 2 to 15, the
  first output, the clock.
