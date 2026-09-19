# E9: VTA at 16x16, against Gemmini and SPMW

Same workload as E8, same device, same measurement discipline. VTA's default
configuration already matches the other two where it counts -- `batch = 1`,
`blockIn = blockOut = 16`, int8 operands into an int32 accumulator -- so this
is three 16x16 engines, 256 multiply-accumulates each, on one transformer
block.

## The answer

**On the part of this block VTA was built for, it wins.** It is the smallest
of the three and its GEMM is the fastest, and the comparison should lead with
that rather than with what it lacks.

| | VTA | Gemmini | SPMW |
|---|---:|---:|---:|
| cycles, GEMM + requantisation | **210,944** | 236,544 | 210,944 |
| lookup tables | **26,403** | 71,389 | 113,341 |
| registers | **5,991** | 21,474 | 153,869 |
| multipliers | **0** | 500 | 755 |
| clock | 300.5 MHz | 34.8 MHz | **306.7 MHz** |

The block then also wants two LayerNorms, four softmaxes and an IGELU, and
**VTA has no datapath for them.** That is a scope difference and not a
defect: VTA is a quantised-CNN accelerator, and its four ALU opcodes are
named `minpool`, `maxpool`, `add` and `shift` -- the operation set a
convolution network needs. LayerNorm, softmax and GELU postdate it.

So the honest reading of this experiment is not "VTA is worse". It is:
**the nonlinearities are what Gemmini's 500 multipliers and 34.8 MHz clock,
and SPMW's extra 15,232 cycles and 87,000 lookup tables, are buying** -- and
VTA is what the same mesh costs without them.

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
is inside VTA's `minpool / maxpool / add / shift` ALU, so nobody is being
asked to do something their ISA has no word for -- and Gemmini and SPMW were
already measured on it, against a shared stimulus file and a shared golden.

VTA runs it **bit-exact** (`scripts/vta_micro_bench.py`, 16 tiles, 0 errors
against the same `stim_S16.txt`), and here is the three-way result at 16x16:

| | cycles / tile | LUT | FF | DSP | slack |
|---|---:|---:|---:|---:|---:|
| SPMW, fixed | **16** | 92,880 | 130,776 | 0 | +0.468 |
| Gemmini `MxuVpu` | 18 | 31,932 | 18,675 | 0 | +0.245 |
| VTA | 65 | **26,403** | **5,991** | 0 | +0.005 |

**A clean performance/area trade, and the three engines sit on it in order.**
SPMW is fastest and largest, VTA is smallest and slowest by 4x, Gemmini is
between them on both.

### Why VTA is 4x slower on a workload it fully supports

Not the mesh -- its GEMM is the *fastest* of the three at 16.0 cycles a
tile. It is the epilogue, and the reason is architectural:

    VTA, per tile   16  GEMM
                  + 16  ALU pass: max, the ReLU
                  + 16  ALU pass: shr, the requantise
                  + 16  ALU pass: min, the clip to 127
                  ----
                    65  measured 1050 cycles for 16 tiles

**VTA's ALU is a load-store unit over the accumulator scratchpad: one opcode
per instruction, one pass over the data each.** Gemmini folds bias, ReLU,
shift and clip into `AccumulatorScale` on the output path and SPMW folds them
into its scale lane, so for both the epilogue is *free* -- it happens as the
results drain. VTA has to read the accumulator back three times.

Nor can the passes hide behind the matmul. `Compute.scala` asserts
`!tensorGemm.io.uop.idx.valid || !tensorAlu.io.uop.idx.valid`: the two units
share the micro-op port and never run in the same cycle. Measured directly,
chained ALU instructions are strictly additive -- 258, 521 and 784 cycles for
one, two and three passes over 256 rows, 1.03 cycles a row each.

The bias is free on all three and nobody is charged for it: Gemmini and SPMW
fold it into the epilogue, VTA preloads the accumulator.

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
| **VTA** | **16.0** | reads the whole 16x16 weight matrix from a scratchpad every cycle |
| Gemmini | 18.0 | 16 activation rows plus a two-cycle request handshake, weights double-buffered into the PEs |
| SPMW | 20.6 | 16 steps plus a serial weight-file load down each row |

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
