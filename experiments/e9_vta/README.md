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

| 16x16 | units | cycles / tile | LUT | FF | DSP | clock | ns / tile | LUT x ns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SPMW, E3's fixed cell | 256 | 16 | 92,880 | 130,776 | 0 | 349 MHz | 45.8 | 4.26 M |
| **SPMW, Gemmini's PE** | **256** | **16** | **31,515** | 21,584 | 0 | **337 MHz** | **47.4** | **1.50 M** |
| Gemmini `MxuVpu` | 256 PEs | 18 | 31,932 | **18,675** | 0 | 324 MHz | 55.6 | 1.77 M |
| VTA | -- | 65 | 26,403 | 5,991 | 0 | 300 MHz | 216.3 | 5.71 M |

**Like for like -- 256 units, one multiply-accumulate each, as Gemmini has 256
PEs -- SPMW is at Gemmini's lookup tables (31,515 against 31,932) and 1.16x
its registers, two cycles a tile faster with a 4% faster clock: 15% less
time a tile, and a 58-cycle first tile against Gemmini's 70.** The first
version of this table had SPMW at 2.9x Gemmini's lookup tables and 7x its
registers. That was how the SPMW design was written and compiled, not what
SPMW is: [Closing the gap](#closing-the-gap) takes it apart one measured
step at a time.

Fusing cells into fewer, larger units goes further -- 16 units of 4x4 cells
are 23,605 lookup tables and 17,772 registers at 333 MHz, and the whole array
as one unit is 22,572 and 11,118, below VTA's lookup tables at a quarter of
its cycles -- but that is a different design point, and one Gemmini's own
`tileRows`/`tileColumns` can take too; see
[Beyond like for like](#beyond-like-for-like-fewer-larger-units).

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
free win, and both rows are kept so it stays one. The reload build's route
reports and its runs at 8 to 32 tiles are
`../e8_block/report/sw_block*_rl1*`.

### Reading the two together

- **Gemmini's and VTA's rows are their datapaths under this experiment's
  testbenches**, everywhere but in
  [Whole engines, run by their own instructions](#whole-engines-run-by-their-own-instructions).
  That section builds Gemmini's whole accelerator and runs both baselines
  through their own instructions. Gemmini's `S + 2` turns out to be the
  request handshake of the testbench that drove its mesh: its own execute
  controller runs the two real layers within 1.5-18% of the floor.
- **Mesh throughput:** SPMW `S`, Gemmini `S + 2`, VTA `4S + 1` on the
  microbenchmark; on the GEMM alone, SPMW and VTA tie at 16.0 and Gemmini is
  18.0.
- **Area:** on the microbenchmark with 256 units each, SPMW is at Gemmini's
  lookup tables and 1.16x its registers, and VTA is below both. The
  transformer-block engine still uses E3's cell: 113,341 against 71,389.
- **Efficiency:** SPMW with 256 units, on lookup tables times time a tile, by
  1.2x over Gemmini and 3.8x over VTA.
- **On a real layer, VTA's datapath alone looks most efficient, and its
  whole datapath does not.** On LLaMA-3.2-1B's gate and up projections, `K`
  is 2,048 rather than 16, and VTA's epilogue falls to 3% of its time. Its
  `TensorGemm` and `TensorAlu` alone then have the best lookup tables times
  time. But they read a whole weight block and an accumulator row every
  cycle, from scratchpads that scope leaves out. Counted with its
  scratchpads, VTA runs at 209-235 MHz and is 1.5-1.7x slower than SPMW on
  every real layer and array size. See
  [VTA with its memory counted](#vta-with-its-memory-counted).
- **Programmability is free on SPMW.** The 256-unit array driven by a
  program, one 64-bit instruction per GEMM, is the size of the fixed arrays
  it replaces, 32,048 lookup tables at 16x16, at 329-355 MHz, and it runs
  every workload here at the floor plus `3S + 2` cycles. Its whole dispatch
  path is 564 lookup tables. See
  [A programmable SPMW engine](#a-programmable-spmw-engine).
- **Programmable against programmable, SPMW is 1.54-1.62x faster than
  VTA and 2.31-2.47x faster than Gemmini on the two real layers**, each
  run through its own instructions, in the scope all three have: what
  decodes and executes, with its storage. Nearly all of it is clock,
  343-378 MHz against 224-235 and 154-164, since all three are within 18%
  of the floor in cycles.
- **As whole engines, from a memory port in to a memory port out, it is
  1.55-1.73x faster than VTA and 2.37-2.61x faster than Gemmini
  on them.** SPMW's whole engine is 4,860-35,544 lookup tables at
  334-367 MHz, VTA's 10,099-33,923 at 213-224 MHz and Gemmini's
  31,545-65,529 at 147-153 MHz with matmul only, 44,036-89,053 at 64-68 MHz
  as shipped. SPMW's memory system is about 1,400 lookup tables and 11 to 14
  block RAMs: it runs a GEMM a K block at a time, so that every weight is
  used as it arrives, and keeps one activation chunk and the partial sums.
  See [SPMW with a memory system](#spmw-with-a-memory-system) and
  [Whole engines, run by their own instructions](#whole-engines-run-by-their-own-instructions).
- **Coverage:** Gemmini and SPMW all of it, VTA 46% of the scale path.
- **Clock on this FPGA:** SPMW and VTA's datapath both near 300 MHz. The
  transformer block's Gemmini is at 34.8 MHz, which is a porting artifact of
  an unpipelined float scale path and not an architectural result. Its whole
  accelerator routes at 64-68 MHz as shipped and at 147-153 MHz without the
  float scale and the convolution support.

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

## One workload, three array sizes

The table above changes two things at once, since each size runs its own
`S x S x S` tiles. Here the workload is held fixed and only the array
shrinks. The workload is E3's `S = 16` microbenchmark, the same operands:

    Y_t[m, n] = min(127, max(0, sum_k A_t[m, k] B_t[k, n] + b[n]) >> 8)

for sixteen tiles `t` of `M x K x N = 16 x 16 x 16`. `A_t` and `B_t` are
int8, `b` is an int8-valued int32 and every sum is int32. That is 65,536
multiply-accumulates. An `S x S` array takes a tile as `(16/S)^2` weight blocks
and adds `16/S` partial sums into each output before the epilogue, and each
engine does that its own way:

- **SPMW** (`tests/dataflow/spmw/test_spmw_tpu_micro_blocked.py`, design
  `tpumicro-blocked`) is the 256-unit lean cell of
  [Closing the gap](#closing-the-gap), unchanged except that a weight block
  lasts 16 rows. Each lane holds a row's running sum in a 16-deep delay line
  and emits on the last block. At 16x16 there is one block, so there is no delay line.
- **Gemmini** (`../e3_tpu/micro/gemmini-acc/source/MxuAccVpu.scala`) is its
  mesh, its `AccumulatorMem` (two banks, read-modify-write through
  `AccPipeShared`) and its `AccumulatorScale`, wired as Gemmini's
  `Scratchpad` wires them. E3's `MxuVpu` has no accumulator because a
  one-block tile never needs one.
- **VTA** (`scripts/vta_micro_bench.py --width S`) is the same `TensorGemm`
  and `TensorAlu`. The GEMM accumulates the K blocks in the acc scratchpad,
  then three ALU passes follow. It runs in two instruction orders: *batched*
  (every GEMM, then each pass over everything) and *tile by tile*.

All twelve runs are bit-exact against one golden. Each routes at 3.333 ns
with zero unrouted nets, and none uses a DSP or a block RAM:

| array | engine | cycles / tile | first tile | total | clock | time | LUT | FF | LUT x ns / tile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | **SPMW** | **256** | **281** | **4,121** | 368 MHz | **11.2 us** | **2,731** | 1,980 | **1.90 M** |
| | Gemmini | 384 | 416 | 6,176 | 321 MHz | 19.2 us | 3,167 | 2,058 | 3.79 M |
| | VTA, batched | 449 | 6,235 | 7,194 | 421 MHz | 17.1 us | 3,121 | **1,283** | 3.33 M |
| | VTA, tile by tile | 478 | 475 | 7,659 | 421 MHz | 18.2 us | | | 3.55 M |
| 8x8 | **SPMW** | **64** | **97** | **1,057** | 341 MHz | **3.10 us** | 8,426 | 6,504 | **1.58 M** |
| | Gemmini | 80 | 124 | 1,324 | 316 MHz | 4.19 us | 10,017 | 6,130 | 2.54 M |
| | VTA, batched | 161 | 2,107 | 2,586 | 348 MHz | 7.44 us | **8,326** | **2,979** | 3.86 M |
| | VTA, tile by tile | 190 | 187 | 3,051 | 348 MHz | 8.77 us | | | 4.55 M |
| 16x16 | **SPMW** | **16** | **58** | **305** | 339 MHz | **0.90 us** | 31,200 | 21,236 | **1.47 M** |
| | Gemmini | 18 | 86 | 356 | 308 MHz | 1.15 us | 35,052 | 21,342 | 2.05 M |
| | VTA, batched | 65 | 811 | 1,050 | 301 MHz | 3.49 us | **26,403** | **5,991** | 5.71 M |
| | VTA, tile by tile | 94 | 91 | 1,515 | 301 MHz | 5.04 us | | | 8.26 M |

Each engine again follows an exact law, with `M = K = N = 16`:

| | cycles a tile | 4x4 | 8x8 | 16x16 |
|---|---|---:|---:|---:|
| SPMW | `MKN / S^2` | 256 | 64 | 16 |
| Gemmini | `MKN / S^2 x (S + 2) / S` | 384 | 80 | 18 |
| VTA, batched | `MKN / S^2 + 3 MN / S + 1` | 449 | 161 | 65 |

- **SPMW is at the floor at every size.** The next block's weights shift in
  behind the current block, and the partial sums are added in the lane as
  they drain.
- **Gemmini's two-cycle handshake is paid per request, and a request carries
  at most `S` rows** (`MeshWithDelays` requires `total_rows <= block_size`).
  A 16-row block is therefore `16/S` requests, and the overhead is `2/S`:
  50% at 4x4, 25% at 8x8, 12.5% at 16x16. This is the handshake of the
  testbench that drives the mesh here. Gemmini's own controller does not
  pay it per request; see
  [Whole engines, run by their own instructions](#whole-engines-run-by-their-own-instructions).
- **VTA's GEMM is also at the floor, but its epilogue is not.** Three ALU
  passes over `MN/S` accumulator rows shrink only as `1/S`, while the GEMM
  shrinks as `1/S^2`. The epilogue is 43% of the time at 4x4, 60% at 8x8 and
  74% at 16x16. Tile by tile costs 29 more cycles a tile at every size, in
  instruction overhead, and buys a first tile after 91 cycles instead of 811.

From 4x4 to 16x16 each engine gets 16x the multipliers:

| | cycles a tile | whole workload, in time | LUT |
|---|---:|---:|---:|
| SPMW | 16x fewer | 12.4x faster | 11.4x more |
| Gemmini | 21.3x fewer | 16.7x faster | 11.1x more |
| VTA, batched | 6.9x fewer | 4.9x faster | 8.5x more |

**SPMW is fastest at every size, and it has the best lookup-tables-times-time
at every size, improving as the array grows (1.90 M, 1.58 M, 1.47 M).**
Gemmini scales best in cycles, but only because its overhead shrinks: from
1.5x SPMW at 4x4 to 1.125x at 16x16. VTA has the fewest registers at every
size and the fewest lookup tables at 8x8 and 16x16, but its time scales
worst, 4.9x for 16x the multipliers. On a fixed workload the smallest VTA is
its most efficient (3.33 M at 4x4 against 5.71 M at 16x16), which is the
opposite of the other two. At 4x4, VTA's 421 MHz clock puts it ahead of
Gemmini in time, 17.1 us against 19.2 us.

The scope is the same as the rest of this file, and it cuts one way.
**VTA's partial sums live in its acc scratchpad, block RAM outside the
measured modules.** Gemmini's accumulator is inside the scope: 1,196 lookup
tables at 4x4, and 4,620 at 16x16, where this workload never adds a partial
sum. At 16x16, `MxuAccVpu` is 35,052 lookup tables against 31,932 for E3's
`MxuVpu`. SPMW's delay lines are inside the scope too: its four lanes are 1,021
lookup tables and 876 registers of the 4x4 array. SPMW's generator builds a
delay line only where the shape needs one, so its 16x16 array has none.
SPMW's mesh links are bare registers, which are correct only while the array
is fed without gaps. Cosimulation feeds it that way, and all three runs report
no overwritten value.

## A LLaMA layer: the FFN's gate and up projections

LLaMA-3.2-1B's feed-forward block is `down(SiLU(x W_gate) * (x W_up))`, with
`d_model = 2048`, `d_ff = 8192` and no bias. Over a 64-token prompt
(prefill), each of its two up projections is a 64 x 2048 x 8192 matrix
product. That gives two workloads:

- **Gate and up, requantised.** All three engines run this on chip:

      G = clip(x W_gate >> s, -128, 127)     U = clip(x W_up >> s, -128, 127)

  It is E3's microbenchmark without the bias and the ReLU. The differences
  are that `K` is 2,048 instead of 16, and one weight matrix serves all 64
  tokens instead of one per 16 rows.
- **SwiGLU fused.** Only SPMW runs this: `H = clip(isilu(G) * U >> s2, -128,
  127)`, computed in the lanes as the sums drain. `isilu` is an integer SiLU,
  I-BERT's clipped second-order polynomial. Over int8 read as `g/16` its
  error against SiLU is at most 0.090, with an RMS of 0.034. Gemmini's scale
  unit has ReLU, I-GELU, LayerNorm and softmax, but no SiLU, and neither
  Gemmini nor VTA can multiply two results element by element.

The simulated slice is all 64 tokens and the full `K = 2048`, over 64 columns
of each projection, which is 128 of the 16,384. That is 16.8 million
multiply-accumulates, and the full pair of projections is 128 such slices.
Every engine's time is linear in the number of slices, so the table's
full-pair time is 128 times the slice's. The operands are random int8
(`test_spmw_llama_ffn.dump_llama`), since no engine's timing depends on the
values. All runs are bit-exact, and every design routes at 3.333 ns with zero
unrouted nets, no DSP and no block RAM.

- **SPMW** is `blocked_engine` with a block of `M = 64` rows, so each weight
  block is loaded once and all 64 tokens stream through it. For gate and up,
  each lane keeps a row's running sum in a 64-deep delay line
  (`epilogue="requant"`). For SwiGLU, each lane keeps the gate and up sums
  side by side in a 128-deep line (`"swiglu"`).
- **Gemmini** is `MxuAccVpu` with 64-row accumulator groups and no activation
  (`ACC_ROWS=64 ACT=none`).
- **VTA** is `vta_llama_bench.py`. The layer does not fit VTA's scratchpads,
  which hold 2,048 input rows, 1,024 weight blocks and 2,048 accumulator
  rows, so the program is paged the way a real VTA program is. For each
  column chunk, a GEMM with `reset` set zeroes the accumulator, one GEMM per
  `K` page accumulates into it, and three ALU passes requantise it. The pages
  swap in at no cost. That is the work of VTA's load unit, which is outside
  the measured modules, just as the other two engines get their operands at
  no cost.

| array | engine | slice cycles | x floor | clock | gate + up, 64 tokens | LUT | FF | LUT x time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | SPMW | **1,048,649** | **1.000** | 349 MHz | 384.3 ms | **3,019** | 2,044 | 1,160 K |
| | Gemmini | 1,572,944 | 1.500 | 311 MHz | 647.7 ms | 3,303 | 2,128 | 2,139 K |
| | VTA | 1,056,922 | 1.008 | 421 MHz | **321.6 ms** | 3,121 | **1,283** | **1,004 K** |
| | *SPMW + SwiGLU* | *1,048,658* | *1.000* | *356 MHz* | *377.4 ms* | *5,528* | *3,296* | *2,087 K* |
| 8x8 | SPMW | **262,225** | **1.000** | 359 MHz | **93.4 ms** | 9,860 | 6,616 | 921 K |
| | Gemmini | 327,772 | 1.250 | 319 MHz | 131.6 ms | 10,373 | 6,264 | 1,365 K |
| | VTA | 266,330 | 1.016 | 348 MHz | 98.0 ms | **8,326** | **2,979** | **816 K** |
| | *SPMW + SwiGLU* | *262,234* | *1.000* | *345 MHz* | *97.3 ms* | *14,737* | *9,120* | *1,434 K* |
| 16x16 | SPMW | **65,633** | **1.001** | 343 MHz | **24.5 ms** | 36,471 | 24,361 | 893 K |
| | Gemmini | 73,844 | 1.127 | 305 MHz | 31.0 ms | 35,875 | 21,619 | 1,111 K |
| | VTA | 67,642 | 1.032 | 300 MHz | 28.8 ms | **26,403** | **5,991** | **761 K** |
| | *SPMW + SwiGLU* | *65,642* | *1.002* | *315 MHz* | *26.7 ms* | *46,806* | *29,708* | *1,250 K* |

The rows in italics do more work than the others: they also compute SwiGLU,
which the others leave for a host. The laws are those of E3's microbenchmark,
with this layer's `K`:

- **SPMW** runs at the floor, `L K N / S^2`, plus under 100 cycles of fill.
- **Gemmini** pays `(S + 2) / S` over the floor for its per-request
  handshake, as before. That is this testbench's handshake: under its own
  controller Gemmini pays 2-18% on this layer.
- **VTA** pays `1 + 4S/K` over the floor. The three ALU passes and the reset
  each cover the `L N / S` accumulator rows, against the GEMM's
  `L K N / S^2`. That predicts 1.008, 1.016 and 1.031, and the measurements
  are 1.008, 1.016 and 1.032.

What this says:

- **A deep layer removes VTA's epilogue penalty.** VTA runs at 1.03 times the
  floor at 16x16, against 4.06 on E3's microbenchmark. So the engines are
  separated by their clocks and their areas, not by cycles.
- **Time.** SPMW is fastest at 16x16, 1.18x ahead of VTA and 1.26x ahead of
  Gemmini, and at 8x8, 1.05x and 1.41x ahead. VTA is fastest at 4x4, taking
  16% less time than SPMW, on its 421 MHz clock.
- **Lookup tables times time.** For VTA's datapath alone, VTA is best at
  every size, by 13-17% over SPMW, and Gemmini is last, at 1.5-2.1 times VTA.
  SPMW beats Gemmini by 1.2-1.8x. With VTA's scratchpads counted, VTA is
  last; see [VTA with its memory counted](#vta-with-its-memory-counted).
- **SPMW's area grew from the microbenchmark**, to 36,471 lookup tables at
  16x16 against 31,200. There are two causes, and both come from this
  workload's length. First, each cell's loop counter now counts a whole
  layer, 17 to 21 bits instead of 9 to 13, which costs about 14 lookup tables
  and 8 registers a cell. A free-running cell would not need the counter.
  Second, each lane holds 64 running sums, 265 lookup tables against 121.
  That is the accumulation Gemmini keeps in `AccumulatorMem`, at 5,435 lookup
  tables at 16x16, and VTA keeps in its uncounted scratchpad.
- **Fusing SwiGLU costs no cycles.** It costs about 605-630 lookup tables and
  313 registers a lane, for three multiplies bound to fabric and the 128-deep
  line. At 16x16 the fused design takes 26.7 ms, less than VTA and Gemmini
  take to produce `G` and `U` alone. Its clock drops to 315 MHz there, and the
  cause is placement, not logic: a lane's pipeline start fans out to the
  resets of its 128 delay-line registers, and the path is 95% route.
- **The SwiGLU lane takes no link credit.** Its body has stages between its
  reads and its writes, so by `spmw.pipeline`'s credit rule it gets no credit
  (see [Closing the gap](#closing-the-gap)). The first build gave it one
  credit and closed at 301-308 MHz, because HLS had packed 3.1 ns of
  multiply and carry into one stage. Those builds are kept as `c1_`.

The scope is the same as the rest of this file. VTA's scratchpads and its
load and store units are outside it. SPMW streams its weights in from the
testbench at `S` bytes a cycle, and Gemmini takes its operands from the
testbench too.

## One DeepSeek-V4-Pro expert

These are the same two projections, with the same engines, at the shape of
one DeepSeek-V4-Pro routed expert. It has a hidden size of `K = 7168`, and
its gate and up projections are 3,072 wide each. DeepSeek-V4-Pro has 384 such
experts, and each token goes to six of them plus a shared expert. The expert
here sees 64 tokens, which is what a 4,096-token prompt gives each routed
expert on average, and the same row count as the LLaMA run.

The slice is all 64 tokens, the full `K`, and 64 columns of each projection.
That is 58.7 million multiply-accumulates, and the expert's whole gate and up
pair is 48 slices. Weight bandwidth is not modelled, as in the rest of this
file.

DeepSeek-V4 clamps the gate below 10, and the up projection to +-10, before
its SwiGLU. Here both are requantised to int8 and read as `g/16`, so they
saturate at 7.94 before the clamp could act.

The engines are the LLaMA section's, and only the workload changes:

- **Gemmini and VTA** run the same routed hardware, so their area and clock
  are the LLaMA section's. Only the simulation is new.
- **SPMW** is rebuilt, because its array is compiled for the shape. `K` is
  now 448 blocks at 16x16, which is not a power of two, so the lanes count
  their blocks instead of reading them off the step (`_counts` in
  `test_spmw_tpu_micro_blocked.py`). For power-of-two shapes they generate
  exactly the code they did before.

All runs are bit-exact, and every SPMW array routes at 3.333 ns with zero
unrouted nets and no link overwrite.

| array | engine | slice cycles | x floor | clock | expert gate + up, 64 tokens | LUT | FF | LUT x time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | SPMW | **3,670,089** | **1.000** | 361 MHz | 487.6 ms | **3,052** | 2,196 | 1,488 K |
| | Gemmini | 5,505,104 | 1.500 | 311 MHz | 850.1 ms | 3,303 | 2,128 | 2,808 K |
| | VTA | 3,678,682 | 1.002 | 421 MHz | **419.7 ms** | 3,121 | **1,283** | **1,310 K** |
| | *SPMW + SwiGLU* | *3,670,096* | *1.000* | *352 MHz* | *500.0 ms* | *5,183* | *3,296* | *2,591 K* |
| 8x8 | SPMW | **917,585** | **1.000** | 330 MHz | 133.4 ms | 9,951 | 6,952 | 1,328 K |
| | Gemmini | 1,146,972 | 1.250 | 319 MHz | 172.7 ms | 10,373 | 6,264 | 1,791 K |
| | VTA | 921,850 | 1.005 | 348 MHz | **127.3 ms** | **8,326** | **2,979** | **1,060 K** |
| | *SPMW + SwiGLU* | *917,592* | *1.000* | *353 MHz* | *124.9 ms* | *14,174* | *9,184* | *1,770 K* |
| 16x16 | SPMW | **229,473** | **1.000** | 337 MHz | **32.7 ms** | 37,088 | 25,283 | 1,211 K |
| | Gemmini | 258,164 | 1.126 | 305 MHz | 40.6 ms | 35,875 | 21,619 | 1,456 K |
| | VTA | 231,562 | 1.010 | 300 MHz | 37.0 ms | **26,403** | **5,991** | **977 K** |
| | *SPMW + SwiGLU* | *229,480* | *1.000* | *332 MHz* | *33.2 ms* | *45,999* | *30,234* | *1,528 K* |

The laws hold unchanged. SPMW runs at the floor plus under 100 cycles.
Gemmini pays `(S + 2) / S`. VTA pays `1 + 4S/K`, which predicts 1.002, 1.004
and 1.009, and it measures 1.002, 1.005 and 1.010, the difference being its
14 to 56 page GEMMs' instruction overhead.

- **The deeper `K` moves VTA closer still to the floor, and nothing else.**
  SPMW's and Gemmini's cycle counts grow by exactly the ratio of the `K`s,
  3.5x. VTA's grow by 3.42x, because its epilogue does not grow with `K`.
- **Time.** SPMW is fastest at 16x16, 1.13x ahead of VTA and 1.24x ahead of
  Gemmini. At 8x8 VTA is 5% ahead of SPMW, because this SPMW build closed at
  330 MHz against the LLaMA build's 359 MHz. Both builds' worst path is the
  bottom row's multiply-add into the lane link, 61% of it route, so that is
  placement variance and not the new block counter. At 4x4 VTA is 14% ahead,
  on its 421 MHz clock.
- **Lookup tables times time.** For VTA's datapath alone, VTA is best at
  every size, by 14-25% over SPMW, and Gemmini is last, at 1.5-2.1 times VTA.
  With VTA's scratchpads counted, the order reverses, as the next section
  shows.
- **SwiGLU still costs no cycles.** It adds 505-535 lookup tables and about
  280 registers a lane.

## VTA with its memory counted

Every VTA row above counts `TensorGemm` and `TensorAlu` alone. That datapath
holds no weights and no partial sums. Each cycle it reads a whole weight
block, 256 bytes at 16x16, and an accumulator row, and it writes the row
back, all through VTA's scratchpads. Those scratchpads are block RAM and
URAM outside the two modules. SPMW and Gemmini keep their weights and
partial sums inside their arrays, where they are counted. So those rows
compare VTA's arithmetic against the other two's arithmetic plus storage.

This section counts the storage. VTA's whole engine, `Core`, is elaborated
at each width with VTA's shipped buffer depths, and only `blockIn` and
`blockOut` move. It is routed with VTA's own recipe, whose retiming the other
two did not get. All three widths route with zero unrouted nets. The routed
engine is then read two ways:

- **Whole engine.** Everything: fetch, the instruction queues, the load and
  store units with their DMA, compute, and every scratchpad. Its clock is
  set everywhere by the load unit's DMA address generation. The worst path
  runs from the load instruction queue to `vmeCmd`, and SPMW and Gemmini
  have no counterpart of it.
- **Datapath + scratchpads.** The whole engine minus the control that SPMW
  and Gemmini have no counterpart of: fetch, the three instruction queues,
  the semaphores and the event counters. Its clock is the worst path whose
  two ends both lie in this scope (`vta_core_scope_timing.tcl`). The DMA
  command logic is outside the scope, but the input and output buffers are
  in it. Removing those two buffers as well moves the clock by at most
  0.07 ns.

| array | VTA scope | LUT | FF | BRAM | URAM | clock | limited by |
|---|---|---:|---:|---:|---:|---:|---|
| 4x4 | datapath only (above) | 3,121 | 1,283 | 0 | 0 | 421 MHz | the datapath |
| | **datapath + scratchpads** | **5,136** | **2,948** | **10** | **2** | **235 MHz** | the accumulator's read-modify-write |
| | whole engine | 10,311 | 3,456 | 14 | 2 | 205 MHz | load DMA addressing |
| 8x8 | datapath only (above) | 8,326 | 2,979 | 0 | 0 | 348 MHz | the datapath |
| | **datapath + scratchpads** | **10,285** | **4,208** | **18** | **6** | **223 MHz** | the accumulator's read-modify-write |
| | whole engine | 15,710 | 4,747 | 22 | 6 | 210 MHz | load DMA addressing |
| 16x16 | datapath only (above) | 26,403 | 5,991 | 0 | 0 | 300 MHz | the datapath |
| | **datapath + scratchpads** | **28,580** | **6,450** | **66** | **12** | **209 MHz** | a URAM feeding the GEMM |
| | whole engine | 34,174 | 6,963 | 70 | 12 | 206 MHz | load DMA addressing |

The scratchpads cost VTA about 2,000 lookup tables at every width, plus the
RAM, and 90-190 MHz of clock. At every width the accumulator's read-modify-write
path fails 300 MHz on its own. It reads a URAM, adds, and writes the result
back to the same URAM in one clock. At 16x16 that is a 4.13 ns path, 62% of
it logic and mostly the URAM's clock-to-output, because VTA's RTL does not
use the URAM's output register. At 16x16 the input buffer's URAM feeding the
GEMM is 0.07 ns worse still. Even the datapath's own paths miss 3.333 ns once
they sit among the scratchpads, by 0.1-0.5 ns.

Counted this way, VTA falls behind on every workload. The cycle counts are
unchanged, since the hardware is the same:

| workload | array | SPMW | Gemmini | VTA + scratchpads | VTA / SPMW, time | VTA / SPMW, LUT x time |
|---|---|---:|---:|---:|---:|---:|
| E3 micro, 16 tiles | 4x4 | **11.2 us** | 19.2 us | 30.6 us | 2.73x | 5.1x |
| | 8x8 | **3.10 us** | 4.19 us | 11.6 us | 3.74x | 4.6x |
| | 16x16 | **0.90 us** | 1.15 us | 5.02 us | 5.57x | 5.1x |
| LLaMA gate + up, 64 tokens | 4x4 | **384 ms** | 648 ms | 576 ms | 1.50x | 2.5x |
| | 8x8 | **93.4 ms** | 132 ms | 153 ms | 1.64x | 1.7x |
| | 16x16 | **24.5 ms** | 31.0 ms | 41.4 ms | 1.69x | 1.3x |
| DeepSeek-V4 expert, 64 tokens | 4x4 | **488 ms** | 850 ms | 752 ms | 1.54x | 2.6x |
| | 8x8 | **133 ms** | 173 ms | 199 ms | 1.49x | 1.5x |
| | 16x16 | **32.7 ms** | 40.6 ms | 53.1 ms | 1.63x | 1.3x |

On lookup tables times time VTA is now last at every size on every workload,
before any of its RAM is counted. On the DeepSeek expert it is 3.86 M, 2.04 M
and 1.52 M lookup-table milliseconds at 4x4, 8x8 and 16x16. SPMW is at
1.49 M, 1.33 M and 1.21 M, and Gemmini at 2.81 M, 1.79 M and 1.46 M. On time,
VTA stays ahead of Gemmini only at 4x4 on the two real layers, where this
testbench's handshake costs Gemmini 50%.

Three things bound this reading:

- **The area includes VTA's input and output buffers**, which the other two
  do not have in scope, because they take their operands from the testbench.
  They cannot be subtracted reliably. Retiming moves logic across the
  hierarchy, so inside the engine `TensorGemm` reports 7,563 lookup tables
  against 21,598 routed alone. Keeping the buffers counts slightly against
  VTA.
- **SPMW and Gemmini would need a memory system too**, and none is counted
  in this section. Both are built and counted further on: Gemmini's in
  [Whole engines, run by their own instructions](#whole-engines-run-by-their-own-instructions)
  and SPMW's in [SPMW with a memory system](#spmw-with-a-memory-system).
  What SPMW needs differs in kind. VTA's datapath reads `S^2` weight bytes a
  cycle, but from its scratchpad, and memory supplies each byte once. The
  SPMW arrays of this section keep no operand outside the array, so memory
  has to supply `2S` bytes every cycle and send each operand more than once;
  the whole engine reorders the GEMM so that it does not.
- **This is VTA's RTL as it ships.** Pipelining its accumulator path would
  lift its clock, as the link credits lifted SPMW's.

## A programmable SPMW engine

Every SPMW row above is fixed-function. The workload's shape and epilogue
are compiled into the array, so the microbenchmark, the LLaMA layer and the
DeepSeek expert are three different builds at each size. VTA is programmed
at run time, and so is Gemmini in its full system, though the `MxuAccVpu`
measured here has no controller. E3's programmable SPMW engine spent 41
cycles a row. This section builds the programmable engine on the 256-unit
array: one build per size, running any program, at a row a cycle.

**The program** is a stream of 64-bit instructions, one per GEMM:

    (REP - 1) << 32 | (KB - 1) << 16 | FINAL << 8 | BIAS | RAW | RELU | SHIFT

`REP` accumulations, each the sum of `KB` K blocks of `S` rows. `BIAS`,
`RELU`, the shift and the clip are the epilogue, and `RAW` emits the int32
sum. The microbenchmark, the LLaMA slice and the DeepSeek slice are one
instruction each.

**The dispatch is a three-stage pipeline** (`tests/dataflow/spmw/test_spmw_ptpu.py`):

- **A sequencer** fetches and decodes. It is one unit and one flat loop. It
  holds the array's only block counters and issues one 12-bit micro-op per
  output row. The next instruction is read and decoded in the first row of
  the GEMM it describes.
- **A tap per lane** distributes. The micro-ops pass along a row of taps at
  a lane a cycle, which is the array's own skew, and each tap hands its lane
  a copy.
- **The lanes** execute. A lane has no counter and decodes nothing. It reads
  one micro-op with each partial sum.

The cells run no program. An activation token is the int8 value and two
framing bits, `SWAP` on every `S`-th beat and `LAST` on the final one, as a
bus carries TLAST. A cell is two weight registers, a multiply-add and two
flags.

**One build runs every program.** At each size the microbenchmark, the LLaMA
slice, the DeepSeek slice and a five-GEMM program were built separately, and
their 45 generated files -- every role's HLS C++, wrapper and Vitis script,
and the fabric -- are byte-identical (`ptpu/S<n>/report/same_hardware.txt`).
The microbenchmark build is routed, and each of the four programs is
cosimulated on that same generated hardware. Every output token is right,
and no link is overwritten.

| array | clock | LUT | FF | E3 micro, 16 tiles | LLaMA slice | DeepSeek slice | five GEMMs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 355 MHz | 2,803 | 2,204 | 4,110 | 1,048,590 | 3,670,030 | 1,326 |
| 8x8 | 337 MHz | 8,665 | 6,416 | 1,050 | 262,170 | 917,530 | 410 |
| 16x16 | 329 MHz | 32,048 | 22,306 | 306 | 65,586 | 229,426 | 258 |

No DSP, block RAM or URAM at any size. Every cycle count is

    cycles = M K N / S^2 + 3S + 2

The first term is the floor, a row a cycle. The rest is the fill: `S` beats
to shift the first weights in, then the last row's passage down the rows,
across the columns and through a lane. The five-GEMM program has four shapes
and three epilogues, and two of its GEMMs are a single block of `S` rows. It
crosses four instruction boundaries and lands on the same formula, so **an
instruction boundary costs no cycle**.

### What programmability costs

Against the three fixed-function arrays it replaces:

| array | | fixed, micro | fixed, LLaMA | fixed, DeepSeek | programmable |
|---|---|---:|---:|---:|---:|
| 4x4 | LUT | 2,731 | 3,019 | 3,052 | **2,803** |
| | FF | 1,980 | 2,044 | 2,196 | **2,204** |
| | clock | 368 MHz | 349 MHz | 361 MHz | **355 MHz** |
| | cycles | 4,121 | 1,048,649 | 3,670,089 | **4,110 / 1,048,590 / 3,670,030** |
| 8x8 | LUT | 8,426 | 9,860 | 9,951 | **8,665** |
| | FF | 6,504 | 6,616 | 6,952 | **6,416** |
| | clock | 341 MHz | 359 MHz | 330 MHz | **337 MHz** |
| | cycles | 1,057 | 262,225 | 917,585 | **1,050 / 262,170 / 917,530** |
| 16x16 | LUT | 31,200 | 36,471 | 37,088 | **32,048** |
| | FF | 21,236 | 24,361 | 25,283 | **22,306** |
| | clock | 339 MHz | 343 MHz | 337 MHz | **329 MHz** |
| | cycles | 305 | 65,633 | 229,473 | **306 / 65,586 / 229,426** |

**It costs nothing in area.** The programmable engine is within 3% of the
smallest fixed array at every size, and 12-14% smaller than the fixed arrays
for the two real layers at 8x8 and 16x16. Its clock is 1-3% below the fixed
arrays' average. At 4x4 and 8x8 that is inside the spread of the three fixed
builds, and at 16x16 it is 8 MHz below the slowest of them. On time it is
between 2% ahead of the fixed array and 4% behind on eight of the nine runs.
The ninth is the LLaMA layer at 8x8, where the fixed build routed at 359 MHz
and the programmable engine is 6% behind.

The area goes where the split by unit shows (`ptpu/S16/report/area_by_unit.txt`):

| 16x16 | units | LUT | FF | each |
|---|---:|---:|---:|---|
| cells | 256 | 17,493 | 4,736 | 68 LUT, 19 FF |
| links between cells | 720 | 9,837 | 11,194 | |
| lanes, with their partial-sum links | 16 | 4,205 | 5,424 | 234 LUT, 273 FF a lane |
| sequencer | 1 | 209 | 88 | |
| taps and micro-op links | 16 | 355 | 864 | |

The registers sum to the array's 22,306. The lookup tables sum to about a
hundred more than its 32,048, because Vivado reports a lookup table packed
from two units under both.

- **The whole dispatch path is 564 lookup tables and 952 registers**: the
  sequencer, the taps and the micro-op links. That is 1.8% of the array's
  lookup tables at 16x16, 4.4% at 8x8 and 10.6% at 4x4, because the
  sequencer does not grow with the array.
- **The cell is smaller than the fixed one**, 68 lookup tables and 19
  registers against 91 and 37 in the fixed LLaMA array. A fixed cell counts
  the launch's rows, 17 bits for the LLaMA slice at 16x16, in each of 256
  cells. A programmed cell stops on a token and swaps weights on a token,
  so it counts nothing.
- **A lane is 234 lookup tables and 273 registers against 265 and 199.** It
  has a run-time shift where the fixed lane's is a constant and no block
  counters, and it is two stages deeper.
- The clock is set at every size by a cell's multiply-add into its
  partial-sum link, 11 or 12 logic levels. The fixed arrays are limited by
  that same path or by one inside a lane. No dispatch path is near it.

### Against the other two

Gemmini's and VTA's rows so far are their datapaths with a testbench playing
the controller, which is not what this engine should be measured against.
The next section gives this engine a memory system, and the one after it
builds Gemmini's whole accelerator and runs all three whole engines through
their own instructions.

Three things bound what this engine is, whatever it is compared with:

- **The instruction set is one instruction.** A GEMM with a bias, a ReLU, a
  shift and a clip, or the raw sums. VTA's and Gemmini's are wider: a
  general ALU, loads and stores, convolution and pooling. The fused SwiGLU
  lane of the LLaMA section is not a program of this engine; it would be a
  second lane.
- **`S` is hardware.** A block is `S` rows and `S` columns, so `M`, `K` and
  `N` are multiples of `S`, as they are for Gemmini's `DIM`.
- **The array cannot be stalled.** Its links are bare registers, so every
  stream has to keep up with it, as in the fixed arrays. The next section's
  engine waits at the array's edge instead, where a pause reaches every row
  at once.

### What it took

The first version that worked in RTL was too large. Its cells counted the
launch's rows and its lanes each decoded the program and kept their own
block counters. At 16x16 it was 51,226 lookup tables and 47,250 registers at
308 MHz, with 32-bit comparisons in every lane on the worst path. Moving the
counters into one sequencer and the row count onto the stream gave the
engine above, at 63% of the lookup tables and 47% of the registers. Those
builds were not kept.

A unit that stops on a token is a `while`, and Vitis HLS got it wrong twice
in ways the two simulators cannot see:

- **It split the cell across two states.** The loop's exit depends on the
  token just read, so Vitis scheduled that read first and everything else a
  state later, at any clock. The cell then read its weight a cycle after its
  activation, and its neighbour's next weight overwrote the one it had not
  read. A `combinational` unit's `while` is now held to one state, and the
  array build stops if Vitis schedules it in more.
- **It kept the lanes' last three rows.** Vitis's default pipeline moves
  only while its first stage does. After the last micro-op a lane restarts,
  waits for one that never comes, and holds the rows still in flight. A
  `while` deeper than one state is now built as a flushing pipeline.

Both are in `allo/spmw/schedule.py` (`pipeline_whiles`) and in
`docs/source/dive/spmw.rst`.

## SPMW with a memory system

The engine above is fed by streams. It takes `2S` operand bytes every cycle,
each sent again for every block that uses it, in an order a testbench works
out. Gemmini and VTA fetch their operands from memory, once, into
scratchpads, and that machinery is most of what they are. This section gives
the array one (`tests/dataflow/spmw/test_spmw_ptpu_mem.py`). The engine
fetches its program and its operands through one 64-bit read port and writes
its results through one 64-bit write port, a request and then its beats, as
VTA's `Core` does through its VME.

**The order of the GEMM is what keeps it small.** The engine above ran a
GEMM an output block at a time, `S` rows by `S` columns, and so took each
activation block once per column block and each weight block once per row
block. This one runs it a K block at a time: for each `S` of `K`, for each
`S` columns, the tile's `L` rows. Then

- **a weight block is used once, as it arrives**, and is never stored
  outside the array;
- **an activation chunk**, `L` rows of `S` values, serves the column blocks
  of its K block and is done. One chunk of up to 64 rows is all that is
  kept, in distributed RAM;
- **the partial sums persist across K blocks**, one a lane for every output
  of the tile. A lane keeps them in a RAM and adds into it, as VTA's
  accumulator scratchpad does and Gemmini's `AccumulatorMem`: 8,192 sums in
  all, which is the 64 rows by 128 columns of the two layers.

Within a tile no operand crosses the port twice. The LLaMA slice is 49,156
beats in, its 393 KB and one 32-byte instruction, and 1,024 out. The
stream-fed engine took 8.4, 4.2 and 2.1 MB at the three sizes.

**An instruction** is four 64-bit words, a GEMM of `T` tiles of up to 64 rows:

    (T - 1) << 52 | (L - 1) << 44 | (NB - 1) << 32 | (KB - 1) << 16
        | FINAL << 8 | BIAS | REUSE | RELU | SHIFT
    weights << 32 | activations
    results << 32 | biases
    chunk beats << 16 | rows of results a tile

`REUSE` reads every tile's weights from the same address, a layer's tokens
taken 64 at a time, and the biases are read again for each tile. Memory
holds each tensor in the order the array takes it, and the results are
written in the layout the next layer's activations are read in. The
microbenchmark, the LLaMA slice and the DeepSeek slice are one instruction
each.

**Around the cells are ten kinds of unit.** The cells are those of the
engine above, unchanged, and so are the taps that pass a micro-op along the
lanes. The rest:

- **A requester** owns the read port's requests. It walks the GEMM a request
  at a time: per K block a weight block, the activation chunk and the rest
  of the weight blocks, and a block's biases before a block of the first
  K block. Nothing it does waits on a beat of data but an instruction's four
  words, so it runs ahead of memory and a burst follows the one before it
  with no cycle between.
- **A dealer** owns the read port's beats. Each request sends it a tag, and
  it hands the beats that come back to the streams that want them, as far
  ahead of the array as their queues let it: a chunk, some weight blocks, a
  block's biases. That is what hides a load behind the arithmetic.
- **The head** is the sequencer: one iteration a row. It takes the row's
  activation word from the dealer on a K block's first column block and from
  its chunk buffer after; a weight word on a block's last `S` rows, which is
  the next block's weights shifting in; and it issues the row's micro-op,
  which waits in a queue until the row's sums have come down the array.
- **An edge tap per array row** hands its row a byte of each and passes the
  rest down, a row a cycle, which is the array's own skew.
- **A lane** adds a row's partial sum into its accumulator and applies the
  epilogue on a GEMM's last K block. Its bias comes down the micro-op chain,
  a lane at a time on the `S` rows before the block that needs it.
- **A result tap per lane** gathers a row's `S` results into one word, **a
  packer** packs the words into beats, and **a write requester** asks for a
  tile's results a burst at a time, each burst before the one before it is
  acknowledged. The launch is done when the last is.

**The array can wait now, and no cell changed.** A cell's links are still
bare registers. But a row's activations and weights enter the array as one
token, so a row the head does not issue is a pause in every row of cells at
once, and a bare register carries a pause without loss. The head waits
whenever a word has not arrived. On the way out the lanes must never be
held, so the head issues a row of results only into a place the row buffer
is known to have: it counts the buffer's places down, and when they are gone
it takes one of the packer's credits before each such row.

**One build runs every program.** The microbenchmark and a five-GEMM program
were built separately at each size, and their 78 generated files are
byte-identical (`ptpu_mem/S<n>/report/same_hardware.txt`). The
microbenchmark's build is routed. `scripts/spmw_mem_bench.py` puts the
memory of VTA's bench behind its two ports -- a request taken whenever there
is room, a beat a cycle from the next, a write acknowledged the cycle after
its last beat -- and runs the stimulus files the other two engines' benches
read. Every run leaves the golden result in memory, compared word for word
over the whole image, and no link is overwritten.

| array | clock | LUT | FF | block RAM | E3 micro, 16 tiles | LLaMA slice | DeepSeek slice | five GEMMs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 354 MHz | 4,860 | 5,582 | 11 | 4,149 | 1,048,625 | 3,670,065 | 1,561 |
| 8x8 | 367 MHz | 10,872 | 11,799 | 12 | 1,216 | 262,211 | 917,571 | 760 |
| 16x16 | 334 MHz | 35,544 | 33,603 | 14 | 1,221 | 66,246 | 230,086 | 1,496 |

Cycles are from the launch to `done`, which follows the last write's
acknowledgement.

- **On the two layers the engine is at the floor**: 49, 67 and 710 cycles
  over it on either slice, which is the instruction and the first operands
  in, the fill, and the last results out. At 16x16 the last is most of it.
  The final K block's 512 rows of results are 1,024 beats, and the port
  carries one a cycle.
- **The loads hide.** The LLaMA slice's 49,156 beats arrive during
  1,048,576, 262,144 and 65,536 cycles of arithmetic, so the read port is
  busy 5%, 19% and 74% of the time. In the whole run the head waits for an
  operand in 4, 8 and 111 cycles.
- **The microbenchmark is as fast as its port.** Sixteen 16x16x16 matmuls
  are 1,156 beats in. At 4x4 the array is the limit, 4,149 cycles for a
  floor of 4,096. At 8x8 and 16x16 it is the read port: 1,216 and 1,221
  cycles for 1,156 beats, where the array alone would take 1,024 and 256.

### What the memory system costs

Against the stream-fed engine, both routed once at 3.333 ns:

| array | | stream-fed | whole engine | |
|---|---|---:|---:|---:|
| 4x4 | LUT | 2,803 | 4,786 | +71% |
| | FF | 2,204 | 5,582 | +153% |
| | block RAM | 0 | 11 | |
| | clock | 355 MHz | 348 MHz | -2.1% |
| 8x8 | LUT | 8,665 | 10,845 | +25% |
| | FF | 6,416 | 11,799 | +84% |
| | block RAM | 0 | 12 | |
| | clock | 337 MHz | 336 MHz | -0.4% |
| 16x16 | LUT | 32,048 | 35,391 | +10% |
| | FF | 22,306 | 33,601 | +51% |
| | block RAM | 0 | 14 | |
| | clock | 329 MHz | 308 MHz | -6.3% |

**The clock is the cells'.** At every size and at both targets the worst
path is a cell's multiply-add into its partial-sum link, as in the
stream-fed engine. On the 3.333 ns route at 16x16 that path has 0.09 ns of
slack, and no path through a unit of the memory system has less than 0.32.
Routed again at 3.0 ns the whole engine closes at 354, 367 and 334 MHz and the
stream-fed one at 378, 347 and 343: a route moves a clock by more than the memory
system does.

**Its own logic is about 1,400 lookup tables at every size**, and the rest
of what it adds is at the array's edge and in the lanes. By unit, on the
faster route (`ptpu_mem/S<n>/report/p30_area_by_unit.txt`):

| unit | 4x4 LUT | FF | 16x16 LUT | FF | block RAM tiles, 4x4 / 8x8 / 16x16 |
|---|---:|---:|---:|---:|---|
| cells | 1,041 | 272 | 17,362 | 4,738 |  |
| links between cells | 406 | 464 | 9,799 | 11,102 |  |
| edge taps and their links | 88 | 254 | 349 | 2,550 |  |
| head | 527 | 564 | 785 | 1,142 |  |
| micro-op queue, taps and links | 331 | 931 | 1,082 | 3,419 |  |
| lanes, with their accumulators | 924 | 1,524 | 3,832 | 6,768 | 8 / 8 / 8 |
| result taps and their links | 118 | 210 | 868 | 2,382 |  |
| requester | 647 | 579 | 704 | 635 |  |
| dealer | 168 | 98 | 116 | 164 |  |
| operand queues to the head | 209 | 295 | 212 | 294 | 1 / 2 / 4 |
| row buffer and its credits | 143 | 110 | 116 | 94 | 2 / 2 / 2 |
| packer | 13 | 38 | 45 | 71 |  |
| write requester | 268 | 243 | 270 | 244 |  |

- **The read and write sides are 1,448, 1,308 and 1,463 lookup tables**: the requester,
  the dealer, the operand queues, the row buffer, the packer and the write
  requester. They do not grow with the array. Gemmini's DMA, TLB and load
  and store controllers are 10,435 to 12,168, with 12,744 to 14,227 more in
  its command path. VTA's `Core` outside its datapath and scratchpads is
  4,963 to 5,343.
- **The head is 527 to 785 lookup tables** where the stream-fed sequencer
  was 209: it sequences the operands as well as the micro-ops.
- **A lane is 240 lookup tables and 423 registers at 16x16**,
  against 234 and 273 without an accumulator.
- **The registers are in the tokens.** A micro-op is 51 bits where it was
  12, because a lane's 32-bit bias rides it: its queue, taps and links are
  3,419 registers at 16x16 where they were 864. The array's operands enter
  as one token of `2S + 1` bytes, passed down `S` edge taps, and a row of
  results is gathered through `S` result taps: 2,550 and 2,382 more.
- **The RAM is 11, 12 and 14 block RAM tiles**: the lanes' accumulators, 8,192
  sums of 32 bits, in 8 at every size; the row buffer in 2; the two operand
  queues in 1, 2 and 4. VTA's scratchpads are 14, 22 and 70 with 2, 6 and
  12 URAMs, and Gemmini's 64, 80 and 16 with 4, 0 and 8.

### A memory that is not ideal

The same bench with every one of the memory's handshakes withheld a quarter
of the time, each on its own coin: requests, read beats, write requests,
write beats and acknowledgements.

| workload | array | floor | ideal memory | stalling memory | change |
|---|---|---:|---:|---:|---:|
| E3 micro, 16 tiles | 4x4 | 4,096 | 4,149 | 4,169 | +0.5% |
|  | 8x8 | 1,024 | 1,216 | 1,580 | +29.9% |
|  | 16x16 | 256 | 1,221 | 1,599 | +31.0% |
| LLaMA slice | 4x4 | 1,048,576 | 1,048,625 | 1,048,977 | +0.0% |
|  | 8x8 | 262,144 | 262,211 | 262,606 | +0.2% |
|  | 16x16 | 65,536 | 66,246 | 66,776 | +0.8% |
| five GEMMs | 4x4 |  | 1,561 | 1,582 | +1.3% |
|  | 8x8 |  | 760 | 969 | +27.5% |
|  | 16x16 |  | 1,496 | 1,943 | +29.9% |

Every run is right and no link is overwritten, which is the claim above put
to the test: the array waits for its operands and for room for its results,
in RTL, at every size. On a layer the slow memory costs under 1%, because
the loads were hidden with room to spare. Where the port was the limit it
costs what a quarter fewer beats cost.

### What it took

- **The first version passed every test and was twice too slow.** One unit
  read memory and one wrote it, each written as the software it is: ask,
  take the beats, work out what comes next. Both simulators ran its
  programs. Vitis scheduled every loop at II=1 and estimated the head at
  7.2 ns, the writer at 6.1 and the loader at 4.0. The head's path ran from
  the row counter through two subtractions and a comparison to the
  instruction stream's read, from what was read to whether the bias stream
  is read, and back to the counter. So the units that count are now written
  as a hardware designer writes them. Counters run down to zero, with a flag
  set as each one moves, so that a row is decided by flags alone. A read
  that decides anything takes an iteration of its own. Requests are one unit
  and beats another. The estimates are 2.8 to 2.9 ns, and on the 16x16
  engine routed at 3.333 ns no path of the memory system is within 0.23 ns
  of the cells'.
- **A request that waits for its data costs cycles every burst.** With one
  unit asking and dealing, a burst was two cycles longer than its beats. At 8x8
  the microbenchmark is eight bursts a tile for 72 beats. The requester asks
  ahead instead, and the dealer takes the next burst's tag with a burst's
  last beat.
- **A declared array is a loop before it is a memory.** Allo zeroes one with
  a loop ahead of the body. As emitted, a lane spent its first 2,050 cycles
  zeroing its accumulator, and a lane that has not started would lose the
  rows the array hands it. Its read-modify-write also held it at II=2. A unit now
  says an array is a memory: `spmw.ram(P, "buf", distance=4)` makes it
  static with its zeros, and promises that an element written is not read
  again for four iterations, which lets Vitis take the stages it needs at
  II=1. The array build stops if the unit declares no such array.
- **A credit a tile was too coarse.** The head first waited for the tile
  before to leave the row buffer before starting a tile's results. At 16x16
  that round trip is longer than a tile of the microbenchmark, and it ran in
  1,350 cycles. With a credit a row and a count of the buffer's places it
  runs in 1,221.
- **A one-state loop behind a read is reported elsewhere.** A unit that
  reads its site's coordinate ahead of its loop has the loop outlined into a
  module of its own, and the check that a `combinational` body was scheduled
  in one state read only the top module's report. It reads them all now.

`spmw.ram` is in `allo/spmw/schedule.py` and `docs/source/dive/spmw.rst`.

## Whole engines, run by their own instructions

Every Gemmini and VTA cycle count above comes from a datapath under one of
this experiment's testbenches. That is not programmable against
programmable. This section builds Gemmini's whole accelerator, and runs it,
VTA's whole engine and SPMW's on the same stimulus files through their own
instructions, so that each is measured from its first instruction to its
last result in memory.

**Gemmini's whole accelerator** is elaborated as it sits in a Rocket tile,
from its own Chisel at the commits its Chipyard pin names
(`gemmini_full/pins.txt`): the reservation station, the loop unrollers, the
load, store and execute controllers, the scratchpad and accumulator with
their DMA and TLB, and the mesh. Three configurations are routed:

- **As shipped.** Gemmini's own `leanConfig`, which is leaner than its
  default: weight-stationary only, with a float32 scale on load and on
  accumulator read-out.
- **Integer shift.** The accumulator's float scale replaced by a right
  shift, and no scale on load. That is the epilogue SPMW and VTA have.
- **Matmul only.** Integer shift, with what a GEMM never uses switched off
  through Gemmini's own options: the convolution loop unroller, pooling,
  training and depthwise convolutions, first-layer optimisations. This is
  the like-for-like engine, and it is Gemmini's row wherever no other is
  named.

**VTA's whole engine** is the `Core` of
[VTA with its memory counted](#vta-with-its-memory-counted).

**SPMW's whole engine** is the one of
[SPMW with a memory system](#spmw-with-a-memory-system): the array, its
dispatch, and what fetches its operands and writes its results.

Each is measured in two scopes:

- **The whole engine**, from a memory port in to a memory port out.
- **The execute scope**: what decodes and executes instructions, with the
  storage it works from. For Gemmini it is the execute controller, the mesh,
  the scratchpad and accumulator memories and the scale units. For VTA it is
  the datapath and scratchpads of the earlier section. For SPMW it is
  measured twice: as the stream-fed engine of
  [A programmable SPMW engine](#a-programmable-spmw-engine), which has no
  storage at all, and as the whole engine less its read and write sides,
  which has the lanes' accumulators and the head's chunk buffer.

**The programs** are each engine's own:

- **Gemmini** runs what its library issues (`scripts/gemmini_rocc_bench.py`):
  `tiled_matmul_auto`'s tiling, and per tile one `loop_ws`, six commands,
  which its hardware loop unroller expands into the tile's move-ins,
  preloads, computes and move-outs. For the sixteen small matmuls of E3's
  microbenchmark a hand-scheduled program is run as well, and the faster of
  the two is reported.
- **VTA** runs a 128-bit instruction stream launched as its runtime launches
  one (`scripts/vta_core_bench.py`): loads, a GEMM that clears the
  accumulator, GEMMs, ALU passes, a store and a finish, kept in step by its
  dependency tokens. A layer is paged through its scratchpads' two halves,
  so that loads hide behind compute.
- **SPMW** runs four-word GEMM instructions fetched from memory
  (`scripts/spmw_mem_bench.py`).

The LLaMA slice is one instruction on SPMW, 41 commands on Gemmini and 103,
55 and 31 instructions on VTA at the three sizes. The DeepSeek slice is one,
113 to 125, and 343, 175 and 91.

The CPU and the memory are ideal. Gemmini is offered a command the cycle the
last was taken, and every engine's memory answers every request with a beat
a cycle, through a 64-bit port. All 84 runs produce the right result: 36 on
Gemmini's three configurations, 18 more on the matmul-only one, 9 on VTA and
21 on SPMW, nine of those against a memory that stalls. Gemmini as shipped
rounds where the other two shift, so each configuration is checked against
its own arithmetic.

Gemmini is routed out of context with the recipe VTA's `Core` was routed
with, retiming on, at 3.333 ns. VTA is also routed at 4.3 and 4.6 ns, the
matmul-only Gemmini at 6.0 ns and both SPMW engines at 3.0 ns, and an
engine's clock is the best of its routes.
`scripts/whole_engine_tables.py` prints every table below from the result
files.

### The hardware

| array | engine | LUT | FF | BRAM | URAM | DSP | clock |
|---|---|---:|---:|---:|---:|---:|---:|
| 4x4 | SPMW | 4,860 | 5,582 | 11 | 0 | 0 | 354 MHz |
| | Gemmini, matmul only | 31,545 | 21,320 | 64 | 4 | 27 | 152 MHz |
| | Gemmini, integer shift | 37,513 | 26,079 | 64 | 4 | 161 | 94 MHz |
| | Gemmini, as shipped | 44,036 | 29,579 | 64 | 4 | 177 | 68 MHz |
| | VTA | 10,099 | 3,441 | 14 | 2 | 0 | 213 MHz |
| 8x8 | SPMW | 10,872 | 11,799 | 12 | 0 | 0 | 367 MHz |
| | Gemmini, matmul only | 38,941 | 25,648 | 80 | 0 | 25 | 147 MHz |
| | Gemmini, integer shift | 45,362 | 30,560 | 80 | 0 | 159 | 95 MHz |
| | Gemmini, as shipped | 56,329 | 34,389 | 80 | 0 | 183 | 64 MHz |
| | VTA | 15,403 | 4,747 | 22 | 6 | 0 | 218 MHz |
| 16x16 | SPMW | 35,544 | 33,603 | 14 | 0 | 0 | 334 MHz |
| | Gemmini, matmul only | 65,529 | 39,989 | 16 | 8 | 29 | 153 MHz |
| | Gemmini, integer shift | 72,215 | 45,052 | 16 | 8 | 163 | 92 MHz |
| | Gemmini, as shipped | 89,053 | 49,866 | 16 | 8 | 203 | 64 MHz |
| | VTA | 33,923 | 7,387 | 70 | 12 | 0 | 224 MHz |

- **SPMW's whole engine is the smallest at 4x4 and 8x8 and VTA's size at
  16x16.** It has 0.48, 0.71 and 1.05 of VTA's lookup tables and 0.15, 0.28 and 0.54 of the
  matmul-only Gemmini's. It has 1.6, 2.5 and 4.5 times VTA's registers, because
  its weights, its links and its tokens are registers, and 0.26, 0.46 and 0.84 of
  Gemmini's.
- **Gemmini's whole accelerator is 10, 3.9 and 1.9 times its datapath.**
  `MxuAccVpu` above is 3,167, 10,017 and 35,052 lookup tables, and the
  matmul-only engine around it is 31,545, 38,941 and 65,529. What it has
  outside the execute scope hardly grows with the array: 24,518, 24,897 and
  24,875 lookup tables, which is 78% of the engine at 4x4 and 38% at 16x16.
- **Convolution support is 6,000 to 6,700 lookup tables and 134 DSPs, and
  the float scale 6,500 to 17,000 more.** The integer-shift engine is
  37,513, 45,362 and 72,215 lookup tables, and the shipped one 44,036,
  56,329 and 89,053.
- **VTA's whole engine is a third to a half of Gemmini's.** It has 0.32, 0.40
  and 0.52 of the matmul-only engine's lookup tables and 0.16-0.19 of its
  registers.
- **Gemmini and VTA hold their operands in RAM, and SPMW only what it has to
  keep.** Gemmini's is a 256 KB scratchpad and a 64 KB accumulator at every
  size, which Vivado maps to different RAMs as the rows widen. VTA's
  scratchpads grow with the array. SPMW keeps a tile's 8,192 partial sums,
  one activation chunk and its queues, 11 to 14 block RAMs, because it uses
  each weight as it arrives.
- **SPMW routes at 334-367 MHz, VTA at 213-224 MHz, and Gemmini at
  147-153 MHz with matmul only, 92-95 MHz with the integer shift and
  64-68 MHz as shipped.** SPMW is limited by a cell's multiply-add into its
  partial-sum link at every size, as its array alone is. Each Gemmini
  configuration is limited by a different block:
  - *As shipped*, by the accumulator's float32 scale, 55 to 57 logic levels
    in one stage. The scale unit's pipeline registers sit after its logic,
    to be retimed into it, and Vivado's retiming does not move them.
  - *With the integer shift*, by the convolution loop unroller's address
    arithmetic, 37 to 39 levels through three DSP multipliers.
  - *With matmul only*, by the store controller at 4x4 and 8x8, from its
    counters through a multiplier to the scratchpad's write queue in 25
    levels, and by the DMA's writer at 16x16, in 31.

The matmul-only engine by block (`gemmini_full/route/matmul_<n>/split.txt`):

| block | 4x4 LUT | FF | 8x8 LUT | FF | 16x16 LUT | FF |
|---|---:|---:|---:|---:|---:|---:|
| execute controller and mesh | 3,301 | 2,531 | 9,024 | 5,769 | 31,095 | 18,044 |
| &nbsp;&nbsp;of which the mesh | 2,114 | 1,548 | 7,863 | 4,777 | 29,028 | 17,050 |
| scratchpad, accumulator, scale | 3,726 | 1,591 | 5,020 | 2,909 | 9,559 | 5,491 |
| DMA | 7,761 | 6,577 | 8,704 | 6,678 | 9,963 | 6,532 |
| TLB | 926 | 674 | 933 | 664 | 926 | 660 |
| load controller | 730 | 428 | 643 | 322 | 485 | 264 |
| store controller | 1,018 | 484 | 955 | 315 | 794 | 225 |
| command queues | 6,485 | 6 | 5,826 | 6 | 5,511 | 6 |
| reservation station | 4,613 | 6,228 | 4,993 | 6,162 | 4,253 | 5,973 |
| loop unrollers | 3,129 | 1,650 | 3,082 | 1,659 | 2,980 | 1,643 |
| counters | 764 | 1,151 | 764 | 1,151 | 762 | 1,151 |
| whole accelerator | **31,545** | **21,320** | **38,941** | **25,648** | **65,529** | **39,989** |

The blocks sum to 1-3% more than the engine, because Vivado counts a lookup
table shared by two blocks under both.

- **The memory system** -- the DMA, the TLB and the load and store
  controllers -- is 10,435, 11,235 and 12,168 lookup tables. SPMW's read and
  write sides are 1,448, 1,308 and 1,463, and VTA's `Core` outside its datapath and
  scratchpads 4,963, 5,118 and 5,343.
- **The command path** -- the two command queues, the reservation station
  and the matmul loop unroller -- is 14,227, 13,901 and 12,744. The queues
  alone are 5,500 to 6,500, in distributed RAM.
- **The mesh** is 2,114, 7,863 and 29,028 lookup tables: 7%, 20% and 44% of
  the engine.

In the execute scope:

| array | engine, execute scope | LUT | FF | BRAM | URAM | clock |
|---|---|---:|---:|---:|---:|---:|
| 4x4 | SPMW, the array and its dispatch, fed by streams | 2,828 | 2,204 | 0 | 0 | 378 MHz |
| | SPMW, the whole engine less its read and write sides | 3,435 | 4,219 | 8 | 0 | 354 MHz |
| | Gemmini, execute controller, mesh and storage | 7,027 | 4,122 | 64 | 4 | 158 MHz |
| | VTA, datapath and scratchpads | 5,136 | 2,948 | 10 | 2 | 235 MHz |
| 8x8 | SPMW, the array and its dispatch, fed by streams | 8,726 | 6,416 | 0 | 0 | 347 MHz |
| | SPMW, the whole engine less its read and write sides | 9,510 | 10,507 | 8 | 0 | 367 MHz |
| | Gemmini, execute controller, mesh and storage | 14,053 | 8,655 | 80 | 0 | 154 MHz |
| | VTA, datapath and scratchpads | 10,285 | 4,208 | 18 | 6 | 224 MHz |
| 16x16 | SPMW, the array and its dispatch, fed by streams | 32,441 | 22,306 | 0 | 0 | 343 MHz |
| | SPMW, the whole engine less its read and write sides | 34,077 | 32,101 | 8 | 0 | 334 MHz |
| | Gemmini, execute controller, mesh and storage | 40,654 | 23,535 | 16 | 8 | 164 MHz |
| | VTA, datapath and scratchpads | 28,580 | 6,450 | 66 | 12 | 224 MHz |

- **Lookup tables.** SPMW's stream-fed engine has 0.40, 0.62 and 0.80 of
  Gemmini's execute scope, and 0.55, 0.85 and 1.14 of VTA's. The whole
  engine's scope has 0.49, 0.68 and 0.84 and 0.67, 0.92 and 1.19.
- **Registers.** The stream-fed engine has 0.53, 0.74 and 0.95 of Gemmini's,
  and 0.75, 1.5 and 3.5 times VTA's, because it keeps weights and partial
  sums in registers where VTA uses RAM. The whole engine's scope has
  1.4, 2.5 and 5.0 times VTA's: its operands and micro-ops travel as wide
  tokens through register links.
- **RAM.** The stream-fed engine has none: it stores no operand and takes
  everything from its streams. The whole engine's scope has 8 block RAM
  tiles at every size, the lanes' accumulators. Gemmini's and VTA's hold the
  scratchpads their operands sit in.
- **Clock.** SPMW closes at 334-378 MHz, VTA's scope at 224-235 MHz
  and Gemmini's at 154-164 MHz. Gemmini's worst path in the scope runs from
  the execute controller's command queue to the scratchpad's address and
  enable pins, 13 or 14 logic levels. VTA's is its accumulator URAM at every
  size: the read-modify-write, or the GEMM's write into it.

### Cycles

End to end is from the first instruction to the last result in memory.
SPMW's array only is the stream-fed engine, whose operands are always there.
Execute only is the cycles Gemmini's execute controller is busy, and compute
only the cycles VTA's `Compute` spends in GEMM and ALU instructions.

| workload | array | floor | SPMW | Gemmini | VTA | SPMW, array only | Gemmini, execute only | VTA, compute only |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E3 micro, 16 tiles | 4x4 | 4,096 | **4,149** | 7,310 | 11,035 | 4,110 | 4,220 | 10,276 |
|  | 8x8 | 1,024 | **1,216** | 1,944 | 5,198 | 1,050 | 1,123 | 4,132 |
|  | 16x16 | 256 | **1,221** | 1,499 | 3,405 | 306 | 380 | 1,828 |
| LLaMA slice | 4x4 | 1,048,576 | **1,048,625** | 1,073,610 | 1,058,331 | 1,048,590 | 1,073,085 | 1,056,984 |
|  | 8x8 | 262,144 | **262,211** | 274,359 | 269,578 | 262,170 | 273,985 | 266,360 |
|  | 16x16 | 65,536 | **66,246** | 77,605 | 74,911 | 65,586 | 77,176 | 67,656 |
| DeepSeek slice | 4x4 | 3,670,016 | **3,670,065** | 3,723,974 | 3,680,381 | 3,670,030 | 3,723,449 | 3,678,904 |
|  | 8x8 | 917,504 | **917,571** | 945,033 | 925,364 | 917,530 | 944,659 | 921,960 |
|  | 16x16 | 229,376 | **230,086** | 271,099 | 239,011 | 229,426 | 258,774 | 231,616 |

- **On the two layers every engine is within 18% of the floor, and SPMW is
  at it.** It is 49 and 67 cycles over at 4x4 and 8x8 and 710, or 1.1% and
  0.3%, at 16x16. VTA is 0.3-0.9% over at 4x4, 0.9-2.8% at 8x8 and 4.2-14.3%
  at 16x16. Gemmini is 1.5-2.4%, 3.0-4.7% and 18%.
- **SPMW's memory system costs it 35, 41 and 660 cycles a layer.** That is
  end to end against the array alone. Its head pauses 4, 8 and 111 times in
  a whole layer; the rest is the instruction fetch at the start and the
  results at the end, where 512 rows of 16 bytes are 1,024 beats.
- **Gemmini's loads hide behind its compute, and its cost is the weight
  change.** End to end is within 0.6% of execute only, except on the
  DeepSeek slice at 16x16. A weight block serves the 64 rows as 16, 8 and 4
  blocks of `S`, and each change of weights costs the execute controller
  cycles that grow with `S`, so its overhead is 1.5-2.3%, 3.0-4.5% and
  13-18%. On the DeepSeek slice at 16x16 the port adds 4.8%: Gemmini's
  tiling brings in the activations twice, 1.83 MB in all, which is as many
  beats as the floor has cycles, so the mesh waits for them.
- **VTA's compute is where the datapath benches put it.** Compute only is
  within 0.03% of `vta_llama_bench.py`'s count on both layers at every size.
  End to end adds about 1,400, 3,300 and 7,300 cycles to either layer, for
  the first pages in and the last results out. That is at most 0.1% at 4x4
  and 3-11% at 16x16.
- **On the microbenchmark a whole engine mostly moves data.** Sixteen
  16x16x16 matmuls are 8 KB of operands in and 4 KB of results out for
  65,536 multiply-accumulates. Through a 64-bit port the operands are 1,024
  beats, which is the 8x8 floor and four times the 16x16 one. SPMW runs at
  its array's rate at 4x4 and at its port's at 8x8 and 16x16: 4,149, 1,216
  and 1,221 cycles for 1,156 beats in, which are the operands, a bias a tile
  and the instruction. Gemmini takes 1.7, 1.7 and 3.9 times its execute
  only, and VTA 1.07, 1.26 and 1.86 times its compute only. Execute only and
  compute only take the memory out:
  Gemmini's execute controller is busy for 1.03, 1.10 and 1.48 times the
  floor, in the run with every load issued first, and VTA's compute for 2.5,
  4.0 and 7.1 times: a pass that clears the accumulator, the GEMM, and four
  ALU passes at 1.26 cycles a row.

### Time

End to end, at each whole engine's routed clock:

| workload | array | SPMW | Gemmini, matmul only | Gemmini, as shipped | VTA | Gemmini / SPMW | VTA / SPMW |
|---|---|---:|---:|---:|---:|---:|---:|
| E3 micro, 16 tiles | 4x4 | **11.7 us** | 48.2 us | 108 us | 51.7 us | 4.12x | 4.42x |
|  | 8x8 | **3.32 us** | 13.2 us | 31.2 us | 23.9 us | 3.98x | 7.20x |
|  | 16x16 | **3.66 us** | 9.82 us | 23.6 us | 15.2 us | 2.69x | 4.15x |
| LLaMA slice | 4x4 | **2.96 ms** | 7.08 ms | 15.7 ms | 4.96 ms | 2.39x | 1.68x |
|  | 8x8 | **0.715 ms** | 1.86 ms | 4.28 ms | 1.24 ms | 2.61x | 1.73x |
|  | 16x16 | **0.198 ms** | 0.509 ms | 1.21 ms | 0.334 ms | 2.56x | 1.68x |
| DeepSeek slice | 4x4 | **10.4 ms** | 24.6 ms | 54.5 ms | 17.2 ms | 2.37x | 1.67x |
|  | 8x8 | **2.50 ms** | 6.42 ms | 14.8 ms | 4.25 ms | 2.57x | 1.70x |
|  | 16x16 | **0.689 ms** | 1.78 ms | 4.23 ms | 1.07 ms | 2.58x | 1.55x |

- **Whole engine against whole engine, SPMW is 1.55-1.73 times faster than
  VTA and 2.37-2.61 times faster than Gemmini on the two layers.** Nearly
  all of it is clock, 334-367 MHz against 213-224 and 147-153, since
  the three are within 18% of each other in cycles. It has 0.48, 0.71 and 1.05 of
  VTA's lookup tables, so on lookup tables times time it is ahead of VTA by
  1.5-3.5 times and of Gemmini by 4.7-15.5.
- **On the microbenchmark it is 2.7-4.1 times faster than Gemmini and
  4.1-7.2 times faster than VTA.** All three are moving data there. SPMW
  waits for nothing but its port, and VTA also runs four ALU passes.
- **VTA is 1.4-1.7 times faster than Gemmini on the layers**, with a third
  to a half of the lookup tables. Its clock is 1.4-1.5 times Gemmini's and
  it takes 1-12% fewer cycles. Gemmini is the faster on the microbenchmark,
  by 1.1-1.8 times, because it has no ALU passes.

In the execute scope: each engine's execute or compute cycles at its scope's
clock, and its lookup tables times that time, relative to SPMW's. SPMW's row
is the stream-fed engine:

| workload | array | SPMW | Gemmini | VTA | Gemmini / SPMW | VTA / SPMW | Gemmini, LUT x time | VTA, LUT x time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E3 micro, 16 tiles | 4x4 | **10.9 us** | 26.7 us | 43.7 us | 2.46x | 4.03x | 6.1x | 7.3x |
|  | 8x8 | **3.03 us** | 7.28 us | 18.4 us | 2.40x | 6.09x | 3.9x | 7.2x |
|  | 16x16 | **0.893 us** | 2.32 us | 8.15 us | 2.60x | 9.12x | 3.3x | 8.0x |
| LLaMA slice | 4x4 | **2.77 ms** | 6.80 ms | 4.50 ms | 2.45x | 1.62x | 6.1x | 2.9x |
|  | 8x8 | **0.756 ms** | 1.78 ms | 1.19 ms | 2.35x | 1.57x | 3.8x | 1.9x |
|  | 16x16 | **0.191 ms** | 0.472 ms | 0.301 ms | 2.47x | 1.58x | 3.1x | 1.4x |
| DeepSeek slice | 4x4 | **9.70 ms** | 23.6 ms | 15.7 ms | 2.43x | 1.61x | 6.0x | 2.9x |
|  | 8x8 | **2.65 ms** | 6.12 ms | 4.11 ms | 2.31x | 1.55x | 3.7x | 1.8x |
|  | 16x16 | **0.669 ms** | 1.58 ms | 1.03 ms | 2.36x | 1.54x | 3.0x | 1.4x |

- **In the execute scope SPMW is 1.54-1.62 times faster than VTA and
  2.31-2.47 times faster than Gemmini on the two layers.** On lookup tables
  times time it is ahead of VTA by 1.4-2.9 times and of Gemmini by
  3.0-6.1, and neither figure counts the baselines' RAM.
- **On the microbenchmark, in the execute scope, it is 2.4-2.6 times
  faster than Gemmini and 4.0-9.1 times faster than VTA.**
- **The lead survives the memory system.** Whole engine against whole
  engine the ratios are 1.55-1.73 and 2.37-2.61, where the execute scope's
  are 1.54-1.62 and 2.31-2.47.

### What this changes above

- **Gemmini's `S + 2` was this experiment's handshake.** The datapath rows
  have Gemmini at 1.50, 1.25 and 1.13 times the floor on both layers. Its
  own execute controller runs them at 1.01-1.02, 1.03-1.05 and 1.13-1.18.
  The datapath rows overstate its cycles by 47-48% at 4x4 and 20-21% at 8x8,
  and understate them by up to 4% at 16x16.
- **VTA's microbenchmark was `4S + 1` a tile only with a free bias and a
  free clear.** The datapath bench preloaded the bias, cleared nothing and
  ran its ALU passes at 1.03 cycles a row: 7,194, 2,586 and 1,050 cycles. As
  a VTA program the compute alone is 10,276, 4,132 and 1,828, with a
  clearing pass, a fourth ALU pass for the bias, and 1.26 cycles a row.
- **VTA's layer cycles stand**, within 0.03%, with the load and store costs
  above added to them.
- **VTA's clocks rise a little with more routes.** They are 213-224 MHz for
  the whole engine and 224-235 MHz for the datapath and scratchpads, against
  205-210 and 209-235 from the 3.333 ns route alone.
- **Gemmini's float scale stays slow with retiming.** E8's 34.8 MHz was
  `MxuVpuNorm` routed as written. The whole accelerator as shipped, retimed,
  reaches 64-68 MHz, and its worst path is the same float32 scale.
- **SPMW's microbenchmark was an array's, not an engine's.** The stream-fed
  306 cycles at 16x16 need four beats of a 64-bit port every cycle. Through
  one port it is 1,221, which is still ahead of Gemmini's 1,499 and VTA's
  3,405.
- **SPMW's lead on the layers stands against VTA and grows against
  Gemmini.** As whole engines it is 1.55-1.73 times against VTA, where it
  was 1.5-1.7 with VTA's scratchpads counted and SPMW's memory system not
  built. Against Gemmini it is 2.37-2.61 times, where the datapath rows had
  1.2-1.7: Gemmini's datapath routes at 305-321 MHz alone, and its whole
  accelerator at 147-153 MHz.

### What bounds it

- **SPMW's memory system is sized for these layers, and it is the smallest
  that runs them.** A tile is at most 64 rows and 8,192 results, the head's
  chunk buffer and the lanes' accumulators. A layer of more rows runs 64 at
  a time and reads its weights again for each tile, where a scratchpad would
  keep them. It has no TLB and takes physical addresses, as VTA does.
  Memory holds its operands in the order it takes them, the weights a block
  at a time in shift order, as VTA's are blocked by its compiler; Gemmini
  takes plain matrices.
- **The memory is ideal and the port is 64 bits.** SPMW has one read channel
  and everything comes through it. VTA's `Core` is driven above its memory
  arbiter, so each of its five read channels is answered at a beat a cycle;
  its load unit runs one load at a time, so only instruction fetch and the
  micro-op load ever overlap a data load. Gemmini's DMA is 128 bits wide
  inside, and Chipyard widens the system bus to match. Here it sits on
  rocket-chip's 64-bit bus behind its own width adapter, 167 lookup tables
  at 4x4, so that all three have the same port. The matmul-only engine rerun
  on a 128-bit bus, end to end:

  | workload | array | 64-bit bus | 128-bit bus | change |
  |---|---|---:|---:|---:|
  | E3 micro, 16 tiles | 4x4 | 7,310 | 7,308 | 0.0% |
  |  | 8x8 | 1,944 | 1,923 | -1.1% |
  |  | 16x16 | 1,499 | 1,015 | -32.3% |
  | LLaMA slice | 4x4 | 1,073,610 | 1,073,607 | 0.0% |
  |  | 8x8 | 274,359 | 274,884 | +0.2% |
  |  | 16x16 | 77,605 | 78,458 | +1.1% |
  | DeepSeek slice | 4x4 | 3,723,974 | 3,723,971 | 0.0% |
  |  | 8x8 | 945,033 | 964,222 | +2.0% |
  |  | 16x16 | 271,099 | 259,074 | -4.4% |

  It helps where the port was the limit, on the microbenchmark and the
  DeepSeek slice at 16x16, and moves the rest by -1.1% to +2.0%. SPMW's
  ports are 64 bits and are not rerun wider: at 8x8 and 16x16 its
  microbenchmark would gain as Gemmini's does.
- **These are this FPGA's clocks.** Gemmini is written for an ASIC flow: its
  controllers put 13 to 31 logic levels between registers and its scale unit
  leaves its pipeline to retiming. The cycle counts carry to any target and
  the clocks do not.
- **One change to Gemmini's RTL.** At 8x8 and 16x16 Vivado built its
  accumulator banks, 1,024 x 256 and 512 x 512 bits with a write enable a
  byte, from about 525,000 registers. `ramfix.py` rewrites each as 64-bit
  columns, which map to block RAM. The 4x4 engine needs no change.
- **The routes are single runs at a few targets.** VTA has three and the
  matmul-only Gemmini two, whose 6.0 ns routes are within 3% of its
  3.333 ns ones: slower at 4x4 and 8x8, faster at 16x16. Each SPMW engine
  has two:

  | engine | target | 4x4 | 8x8 | 16x16 |
  |---|---:|---:|---:|---:|
  | stream-fed | 3.333 ns | 2.813 ns | 2.963 ns | 3.043 ns |
  | | 3.0 ns | 2.643 ns | 2.885 ns | 2.918 ns |
  | whole | 3.333 ns | 2.872 ns | 2.975 ns | 3.246 ns |
  | | 3.0 ns | 2.821 ns | 2.727 ns | 2.996 ns |

  A route moves SPMW's clock by more than its memory system does.
- **The baselines' instruction sets do more.** The matmul-only
  configuration removes what Gemmini's options can remove. Its reservation
  station, its command queues and its loop unroller stay general, as do
  VTA's loads and its ALU. SPMW's engine has one instruction.

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
against Gemmini's 73. Routed at 16x16 and split by hierarchy
(`../e3_tpu/micro/spmw-fixed/S16/report/rebuild_util_hier.rpt`):

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
lane links going from 42 to 21 -- `../e3_tpu/micro/spmw-fixed/S4/` has both
hierarchies (`report/rebuild_*`, `report/fixed1_*`) and the fabric that was
built (`generated_fixed1/spmw_top.sv`). A link's depth is now the deepest any end
*asked* for (`allo.spmw.ports.link_depth`, tested in `test_spmw_rtl.py`), and
every int8 link measures 18 flip-flops, skid included.

## Closing the gap

Four changes, in order, none of which changes the array: it stays 256 units,
one multiply-accumulate each, as Gemmini's mesh is 256 PEs. Each row is a
separate build of the same workload, with the same operands and golden,
cosimulated clean -- 4,096 of 4,096 tokens -- and routed at 3.333 ns.

| 16x16, 256 units | cycles / tile | first out | first tile | LUT | FF | clock |
|---|---:|---:|---:|---:|---:|---:|
| E3's fixed cell, rebuilt | 16.0 | 157 | 194 | 92,961 | 130,928 | 343 MHz |
| **1.** Gemmini's weight discipline | 16.1 | 72 | 109 | 47,495 | 61,604 | 348 MHz |
| **2.** ...scheduled between its links | 16.0 | 40 | 63 | 47,682 | 38,684 | 320 MHz |
| **3.** ...bare-register links | 16.0 | 40 | 63 | 39,981 | 28,264 | 343 MHz |
| **4.** ...one loop a cell | 16.0 | 36 | 59 | 31,337 | 22,058 | 330 MHz |
| **4.** ...and the lanes credited | 16.0 | 35 | **58** | **31,515** | **21,584** | **337 MHz** |
| Gemmini `MxuVpu` | 18 | | 70 | 31,932 | 18,675 | 324 MHz |
| VTA | 65 | | | 26,403 | 5,991 | 300 MHz |

"First out" is the first output token; "first tile" is E3's measure, the
cycle the first tile's last row leaves, weight load included on both sides.

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

**2. Not pipelining on top of the links.** At a 3.33 ns target Vitis charges
every FIFO access 1.21 ns, so read, multiply, add and write -- 4.64 ns in its
model, against a 2.43 ns budget -- cannot share a stage, and it registers `a`,
`p`, the product and the sum, four stages in all.
But an SPMW link *is* a register: its `dout` is a flop and its `din` lands in
one. `spmw.pipeline(P, ii=1, combinational=True)` credits each stage the link
accesses it touches, and the cell becomes one multiply-add between two link
registers -- 143 flip-flops to 52 -- which is Gemmini's `tile_latency = 0`.
The routed critical path is Gemmini's as well: `a`'s link register, the LUT
multiplier, four CARRY8, `p`'s link register; 3.0 ns, two-thirds of it
routing. First output falls from 72 cycles to 40.

**3. Bare-register links.** After step 2 the links are the largest item --
22,524 of the array's 38,684 flip-flops, more than all of Gemmini. Each one is
a two-entry slice, because it can say "stop". Gemmini's are a bare register:
its mesh has no backpressure, data moves every cycle and its validity travels
with it. SPMW's `spmw_fifo` at depth 0 is the same register and its valid,
with `full_n` tied high. That is safe here and only here: the array is fed
without gaps, and once the wavefront has formed every cell takes a token from
each input every cycle, so no value is replaced before it is read.
Simulation checks exactly that -- a depth-0 link that is written while still
holding an unread value prints `SPMW OVERWRITE` -- and every build here
reports none. The links fall from 22,524 flip-flops to 12,104, and the clock
rises: the skid's input mux is gone from the multiply-add's path.

**4. One loop a cell.** The lean cell shifted tile 0's weights in with a loop
of its own before the step loop, and that second loop was a counter, an FSM
and a hand-off between them in every one of 256 cells. Shifting them in on
the step loop's first `dim` iterations instead -- the tile boundary at
`dim - 1` hands tile 0's weight to `cur` exactly as every later one hands
over the next tile's -- takes the cell from 107 lookup tables and 52
registers to 76 and 29. Giving the lanes one link credit, which their
two-stage body takes (see the credit table below), saves another 560.

**That is Gemmini's resource level with 256 units:** 31,515 lookup tables
against 31,932 and 21,584 registers against 18,675, at 16 cycles a tile
against 18 and 337 MHz against 324 -- 47.4 ns a tile against 55.6, and a
first tile 20% sooner in time. The routed critical path is Gemmini's PE path:
`a`'s bare register, the LUT multiplier, five CARRY8, the next register;
2.85 ns. What SPMW still spends that Gemmini does not is the cell's 9-bit
step counter -- Gemmini's control rides along with the data -- and the lanes,
which keep a 32-bit barrel shifter where Gemmini's takes a five-bit shift.

### Beyond like for like: fewer, larger units

An `f x f` block of lean cells in one unit has `f` links of each kind where
its cells had `f^2`, and one loop and one pipeline's control where they had
`f^2`. `fused_engine` generates the block for any `f`, its column sums a
balanced tree. These builds use step 2's cell with two-entry links, not steps
3 and 4, and Gemmini's `MeshWithDelays` has the same knob -- `tileRows` and
`tileColumns` chain PEs combinationally inside a tile -- which was not built
for this comparison, so they are a design-space result rather than a
like-for-like one.

| 16x16 | units | cycles / tile | first tile | LUT | FF | clock |
|---|---:|---:|---:|---:|---:|---:|
| 2x2 cells a unit | 64 | 16.0 | 62 | 34,770 | 29,128 | 337 MHz |
| 4x4 cells a unit | 16 | 16.0 | 50 | 23,605 | 17,772 | 333 MHz |
| 8x8 cells a unit | 4 | 16.0 | 49 | 24,647 | 14,287 | 322 MHz |
| the whole array one unit | 1 | 16.0 | 43 | 22,572 | 11,118 | 311 MHz |

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
16x16 the worst path is 80 LUT levels and 15.6 ns. **The skid is what
backpressure costs at this clock** -- which is why step 3 removes the
backpressure rather than the skid: a depth-0 link has no ready at all, so
there is no chain to be long.

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
| **SPMW, Gemmini's PE, 256 units** | **123** | **84** |
| SPMW, 4x4 cells a unit | 92 | 69 |
| SPMW, the array one unit | 88 | 43 |
| Gemmini | 125 | 73 |
| VTA | 103 | 23 |

Three things remain. The bare links are asserted by the design, not proven
by the compiler: nothing checks that a depth-0 link's array is fed without
gaps, and an array whose inputs can pause mid-stream needs what Gemmini's
controller provides -- one stall for the whole array -- which SPMW does not
generate yet; simulation catches a violation, and synthesis cannot. The
fusion is written by the design, not the compiler:
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
  first output, the clock. `lean_build.sh` and `lean_hier.sh` are the build
  and the hierarchical split it reads.
- `../e3_tpu/micro/spmw-lean/`, `../e3_tpu/micro/spmw-fused<f>/` -- every
  build in "Closing the gap": the design source, the generated per-role HLS
  C++, wrappers and Vitis scripts, the fabric, and the reports. They sit with
  E3's other microbenchmark engines because they are that workload.
- `scripts/vta_micro_bench.py` -- E3's microbenchmark on VTA's `TensorGemm`
  and `TensorAlu` under xsim. `--width S` blocks the tile onto a narrower
  VTA, and `--order tiled` issues the program tile by tile.
- `fixed_workload/` -- VTA's rows in
  [One workload, three array sizes](#one-workload-three-array-sizes). There
  is one log and one testbench per width and order, and `results.txt`
  collects their result lines. `run_S4_w4_batched` and `run_S8_w8_batched`
  rerun the square workload with the blocked bench, and they reproduce the
  17 and 33 cycles a tile of the fair-comparison table.
- `tests/dataflow/spmw/test_spmw_tpu_micro_blocked.py` -- SPMW's engine for
  the same section, design `tpumicro-blocked`. Its builds are in
  `../e3_tpu/micro/spmw-blocked/S<n>/`, and Gemmini's `MxuAccVpu`, with its
  xsim driver and route reports, is in `../e3_tpu/micro/gemmini-acc/`.
- `tests/dataflow/spmw/test_spmw_llama_ffn.py` -- the LLaMA layer: its
  stimulus and golden, the integer SiLU, and `llama_of`, the designs
  `llama-gateup` and `llama-swiglu` in `scripts/spmw_build_array.py`. Run as
  a script, it writes the stimulus the Gemmini and VTA benches read
  (`--out llama_slice.txt`; 1.4 MB, so it is not committed).
- `scripts/vta_llama_bench.py` -- VTA's paged program for the layer, over the
  same units as `vta_micro_bench.py`.
- `llama/` -- the section's runs. `spmw-gateup/S<n>/` and `spmw-swiglu/S<n>/`
  are laid out like E3's SPMW engines, and a `c1_` report is the SwiGLU lane
  built with one link credit. `gemmini/S<n>/` holds the xsim testbench and
  the route reports of `MxuAccVpu` elaborated with `ACC_ROWS=64 ACT=none`
  (`../e3_tpu/micro/scripts/run_acc_xsim.sh` and `pnr_acc.sh` with variant
  `r64_noact`). `vta/w<n>/` holds each width's testbench and log, and
  `vta/results.txt` their result lines.
- `core/w<n>/` -- VTA's whole engine routed at each width: the Vivado script,
  the flat and hierarchical utilisation, the whole engine's timing, and the
  two scope timings (`scope_timing.rpt`, with the input and output buffers,
  and `scope2_timing.rpt`, without). `core/split.txt` is each engine's split
  into control and datapath + scratchpads, from
  `scripts/vta_core_split.py`. `scripts/elab_core_widths.sh` elaborates the
  engine at widths 4 and 8, `scripts/pnr_vta_rtl.sh` routes it, and
  `scripts/vta_core_scope_timing.tcl` and `vta_core_scope2_timing.tcl` time
  the two scopes.
- `tests/dataflow/spmw/test_spmw_ptpu.py` -- the programmable engine: the
  cell, the sequencer, the tap and the lane, the assembler that turns GEMMs
  into a program and its streams, and the four workloads, designs
  `ptpu-micro`, `ptpu-llama`, `ptpu-dsv4` and `ptpu-mixed` in
  `scripts/spmw_build_array.py`.
- `ptpu/S<n>/` -- its builds, laid out like E3's SPMW engines. The routed
  build is the microbenchmark's. A `mixed_`, `llama_` or `dsv4_` report is
  the same hardware built and cosimulated with that program, and
  `same_hardware.txt` is the comparison of their generated files.
  `area_by_unit.txt` splits the routed array by kind of unit.
  `ptpu/scripts/` has the build wrapper, `same_hw.sh`, the hierarchical
  split and `probe_hls.sh`, which synthesises each role once and prints the
  state every link access was scheduled in.
- `tests/dataflow/spmw/test_spmw_ptpu_mem.py` -- the whole engine: the
  requester, the dealer, the head, the edge and result taps, the lane with
  its accumulator, the packer and the write requester, the model that lays a
  program and its operands out in memory, and the four workloads, designs
  `ptpumem-micro`, `ptpumem-llama`, `ptpumem-dsv4` and `ptpumem-mixed` in
  `scripts/spmw_build_array.py`.
- `scripts/spmw_mem_bench.py` -- its bench: a build's units against a memory
  behind the two ports, a stimulus file or the five-GEMM program as the
  launch, and `--stall` for a memory that is not always ready.
- `ptpu_mem/S<n>/` -- its builds, laid out like `ptpu/`. The routed build is
  the microbenchmark's, a `mixed_` report is the five-GEMM program's build
  of the same hardware, and a `p30_` report is the same HLS output routed
  again at 3.0 ns. `area_by_unit.txt` splits a route by what each unit is
  for and gives the worst path through each kind. `ptpu_mem/sim/<run>/` is
  each bench run's testbench, its instruction words and the simulator's log,
  and `ptpu_mem/results.txt` their result lines, `_stall25` the runs on the
  stalling memory. `ptpu_mem/routes.txt` is every route. `ptpu_mem/scripts/`
  has the build and bench wrappers, the second route, the split and the
  export. `ptpu/routes.txt` and `ptpu/S<n>/report/p30_*` are the stream-fed
  engine's second route, and `r2_build.log` the rebuild it was routed from,
  whose first route reproduces the recorded one.
- `gemmini_full/` -- Gemmini's whole accelerator. `pins.txt` is every
  repository's commit. `source/` is the sbt build and
  `ElaborateGemmini.scala`, which elaborates rocket-chip's example system
  with Gemmini as its RoCC accelerator in the three configurations.
  `scripts/` elaborates (`elab.sh`), takes the `Gemmini` module's files out
  (`closure.py`), rewrites the accumulator banks (`ramfix.py`), routes
  (`pnr_gemmini.sh`), splits a route by block (`split.py`), times the
  execute scope (`gemmini_scope_timing.tcl`) and runs the simulations.
  `route/<configuration>_<n>[_p6]/` holds each route's script, utilisation,
  timing, split and scope timing, and `routes.json` collects them with
  VTA's. `sim/` holds every run's testbench and command list, and
  `sim/results.txt` their result lines: `micro` is the library's program,
  `microb` the hand-scheduled one, `microp` that one with every load issued
  first, and `_b128` the 128-bit bus. The generated SystemVerilog is not
  kept; the pins and the harness rebuild it.
- `scripts/gemmini_rocc_bench.py` -- the bench those runs use: Gemmini's
  tiling and RoCC commands from a stimulus file, the golden in each
  configuration's arithmetic, and a testbench with an ideal CPU and a
  TileLink memory.
- `vta_core/` -- VTA's `Core` run through its own instructions
  (`scripts/vta_core_bench.py`): each run's testbench, instruction and
  micro-op images and log, `results.txt`, and the two relaxed-target routes
  of each width with their scope timings.
- `scripts/whole_engine_tables.py` -- prints every table of
  [Whole engines, run by their own instructions](#whole-engines-run-by-their-own-instructions)
  and the generated ones of
  [SPMW with a memory system](#spmw-with-a-memory-system) from those files,
  `ptpu_mem/`, `ptpu/routes.txt`, `results.csv` and `core/split.txt`.
- `deepseek_v4/` -- the same layout for one DeepSeek-V4-Pro expert (designs
  `dsv4-gateup`, `dsv4-swiglu`; `test_spmw_llama_ffn.deepseek_of`). Gemmini
  runs the LLaMA section's routed hardware, so `gemmini/S<n>/` has only the
  testbench and the simulation, and its route reports are the ones in
  `llama/gemmini/`. The stimulus is
  `test_spmw_llama_ffn.py --k 7168 --out dsv4_slice.txt`.
