# E9: VTA at 16x16, against Gemmini and SPMW

Same workload as E8, same device, same measurement discipline. VTA's default
configuration already matches the other two where it counts -- `batch = 1`,
`blockIn = blockOut = 16`, int8 operands into an int32 accumulator -- so this
is three 16x16 engines, 256 multiply-accumulates each, on one transformer
block.

## The answer

| | VTA | Gemmini | SPMW |
|---|---:|---:|---:|
| **cycles for the block** | **210,944** | 267,776 | 278,912 |
| ...but elements sent to the host | **114,688** | 0 | 0 |
| lookup tables | **26,403** | 71,389 | 113,341 |
| registers | **5,991** | 21,474 | 153,869 |
| multipliers | **0** | 500 | 755 |
| block RAM | 0 | 0 | 0 |
| clock | 300.5 MHz | 34.8 MHz | **306.7 MHz** |
| time for what it runs | **0.702 ms** | 7.687 ms | 0.910 ms |

**VTA is the smallest and its GEMM is the fastest, and it cannot run more than
half of this block.** Those are the same fact: what it leaves out is what
makes it small.

## What VTA cannot do, and why it is structural

`TensorAlu`'s entire operation set is **`min`, `max`, `add`, `shr`** -- five
opcodes in the Chisel, four in the HLS. There is **no multiplier in the unit
at all**, no divide, no square root, no exponential.

So of the block's eleven scale-path passes:

| pass | needs | VTA |
|---|---|---|
| 4 requantisations | an arithmetic shift | **yes**, `shr` |
| (a ReLU, if the block had one) | max against zero | **yes**, `max` |
| 2 LayerNorms | a divide, an integer square root, a reciprocal, a multiply | no |
| 4 softmaxes | `iexp` -- three multiplies and a variable shift -- and a reciprocal | no |
| 1 IGELU | `igelu` -- three multiplies | no |

That is **7 of 11 passes and 114,688 of 212,992 elements**, 54% of the scale
path, that has to cross to the host and back. This experiment does not
estimate what that costs, because the cost depends on an interconnect nothing
here measures; it reports the volume and stops.

The comparison is therefore not "VTA is faster". It is: **VTA finishes the
GEMM and the requantisations in 0.702 ms and then stops, and the rest of the
block is somewhere else.** Gemmini and SPMW finish the whole thing.

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
outside the module being measured.

## Scope

`TensorGemm` + `TensorAlu` is VTA's datapath, and it is the counterpart of
Gemmini's `MxuVpuNorm` and SPMW's block engine -- arithmetic, no scratchpads,
no instruction fetch. `Core`, which adds fetch, load, store and every
scratchpad, is routed separately for context and is a larger scope than
either of the others.

Both VTA modules route with **zero unrouted nets** and both meet timing:
`TensorGemm` at +0.005 ns (300.5 MHz) and `TensorAlu` at +0.346 ns
(334.8 MHz). Neither uses a DSP -- Vivado maps all 256 int8 multiplies to
logic, as it does for Gemmini's mesh.

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
