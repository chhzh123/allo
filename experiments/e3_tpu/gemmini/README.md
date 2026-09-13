# Gemmini as a TPU baseline: MXU + VPU

E1 uses Gemmini's mesh as a GEMM baseline. That is the **MXU alone**:
`MeshWithDelays` wrapping `Mesh`, with the skew registers and the req/resp
interface and nothing else -- no scratchpad, no accumulator, no controller, no
DMA. E3 did not use Gemmini at all.

This directory adds the **VPU** half so the two systems can be compared as
TPUs rather than as bare arrays.

## Scope, and why this is the boundary

Gemmini's post-MXU datapath is `AccumulatorScale`, whose activation set is
`NONE, RELU, LAYERNORM, IGELU, SOFTMAX`. The last three are gated behind
`has_normalizations`, which pulls in `Normalizer`. With `has_normalizations =
false` -- Gemmini's own default -- the unit is exactly **scale + ReLU**, and
that is the scope chosen for this comparison.

`num_scale_units = -1` selects Gemmini's simple combinational path: activation
then `scale_func`. The other setting builds a multi-unit scheduler whose
static-assignment policy is only meaningful when some units carry
normalization hardware, which at this scope none do.

## The exact configuration

Everything below is read off `source/MxuVpu.scala` and the two scripts beside
it; nothing is a default left implicit.

**The source.** `ucb-bar/gemmini` at `8c3f992` ("Merge pull request #392 from
ucb-bar/modular"). Only the array files are taken -- `setup_mxuvpu.sh` lists
them -- so there is no RoCC, no scratchpad, no DMA and no controller anywhere
in the netlist.

**The mesh (the MXU).** `MeshWithDelays`, instantiated with exactly E1's
parameters, so the MXU here is the same hardware as E1's Gemmini WS rows:

| Parameter | Value | Why |
|---|---|---|
| `Dataflow` | `WS` | weight-stationary; E1's comparable rows |
| `inputType` / `weightType` | `SInt(8.W)` | int8 operands |
| `accType` / `outputType` | `SInt(32.W)` | int32 accumulate |
| `tileRows`, `tileColumns` | 1, 1 | one PE a tile, so `meshRows` is the array side |
| `meshRows`, `meshColumns` | `dim`, `dim` | 4, 8, 16 -- `MESH_DIM` at elaboration |
| `tree_reduction` | `false` | the systolic chain, not an adder tree |
| `tile_latency` | 0 | no extra pipeline inside a tile |
| `output_delay` | 1 | Gemmini's default output register |
| `leftBanks`, `upBanks`, `outBanks` | 1, 1, 1 | one bank a side |
| `TagQueueTag` | `SimpleTag`, `UInt(8.W)` | the smallest tag the interface accepts |

**The scale path (the VPU).** `AccumulatorScale`:

| Parameter | Value | Why |
|---|---|---|
| `has_normalizations` | `false` | Gemmini's own default; gates LAYERNORM/IGELU/SOFTMAX out and leaves scale + ReLU |
| `has_nonlinear_activations` | `true` | keeps `Activation.RELU` |
| `num_scale_units` | -1 | the simple combinational activate-then-scale path; see the quirks below |
| `latency` | 1 | one pipeline stage |
| `read_small_data` / `read_full_data` | `true` / `false` | the narrow int8 output is what is read |
| `fullDataType` | `Vec(dim, Vec(1, SInt(32.W)))` | one row of accumulator values a beat |
| `rDataType` | `Vec(dim, Vec(1, SInt(8.W)))` | one row of int8 results a beat |
| `scale_t` | `SInt(32.W)` | the scale, a runtime input |
| `scale_func` | `(v, sc) => (v >> sc(4,0).asUInt).asSInt` | `SCALE_MODE=shift`, the form SPMW's `SHR` matches |

**One register between them.** Gemmini's real datapath is
`Mesh -> AccumulatorMem -> AccumulatorScale`, and this scope excludes the
accumulator memory. Wiring the mesh straight into the scale path put the scale
in the same cycle as the mesh's output logic -- 4.694 ns and 20 logic levels at
dim 4, against a 3.333 ns target. That is an artefact of leaving the
accumulator out, not a property of Gemmini, so the boundary it would provide is
modelled by one `RegNext` stage. It is in the netlist that is routed and in the
one that is simulated.

**Toolchain.** Chisel 3.6.0 with the 3.6.0 compiler plugin, Scala 2.13.10,
chiseltest 0.6.2 (0.6.x is what is built against Chisel 3.6; iotesters 2.5.6 is
pinned to 3.5.6 and drags in an incompatible json4s), scalatest 3.2.18,
OpenJDK 11.0.25, sbt at `-Xmx4G` by default and raised by the elaboration
scripts (8 GB at 4 to 16, 16 GB at 32). Vivado 2023.2 for synthesis and for
xsim.

**Route recipe** (`source/mxuvpu_pnr.sh`), the same one every E1 design uses:

    MESH_DIM=<S> SCALE_MODE=shift sbt -batch "runMain gen.ElaborateMxuVpu"
    synth_design -top MxuVpu -part xcu280-fsvh2892-2L-e -mode out_of_context
    create_clock -period 3.333 -name clk [get_ports clock]
    opt_design; place_design; phys_opt_design; route_design

Out of context and with no shell, so the numbers are the datapath's and not a
platform's. `PNR_UNROUTED` is printed from the routed design and is 0 in every
row of `results.csv`.

## What SPMW has to match

SPMW's E3 VPU is a programmable lane, so matching is a program rather than new
hardware. Its ISA already carries every opcode this needs, and the tests
already exercise them:

| Gemmini | SPMW equivalent | Already used in |
|---|---|---|
| `scale_func` (multiply then arithmetic shift) | `MUL` then `SHR` | `spmw_stage_engine.py:92`, `ACCN -> SHR -> STORE` |
| `Activation.RELU` | `MAX` against zero | `spmw_stage_engine.py:392`, `ACCN -> MAX -> STORE` |

The work on the SPMW side is therefore to make the block **use** the on-device
requantise-and-activate path, not to add hardware for it.

## A correction to E3's description

E3's README says the device does the matrix multiplies and the host does
everything else. That understates it: **softmax runs on the device**, through
the `row_max`, `row_sum` and `normalise` VPU programs. What runs on the host is
LayerNorm/RMSNorm, RoPE, the causal mask, GELU/SiLU, the residual adds and the
int32-to-int8 requantisation. The requantisation is the part this comparison
moves onto the device, and the ISA already supports it.

## Results: what the VPU costs on top of the MXU

Both routed out of context at 3.333 ns on `xcu280-fsvh2892-2L-e`, the recipe
every E1 design uses, nothing unrouted anywhere. `results.csv` is the table and
`report/S<n>/` holds the utilisation and timing each number was read from.

| Array | MXU only | MXU + VPU | VPU cost | Clock |
|---|---:|---:|---:|---|
| 4x4 | 1,932 LUT / 1,372 FF | 2,132 / 1,378 | **+200 LUT (10.4%)** | +0.592 -> +0.591 ns |
| 8x8 | 7,535 / 4,759 | 8,042 / 4,818 | **+507 LUT (6.7%)** | +0.432 -> +0.456 ns |
| 16x16 | 30,357 / 18,536 | 31,932 / 18,675 | **+1,575 LUT (5.2%)** | +0.207 -> +0.245 ns |

**The VPU is cheap and gets cheaper.** Scale-plus-ReLU costs a tenth of the
array at 4x4 and a twentieth at 16x16, and it does not cost clock: slack is
unchanged at 4x4 and slightly better at the two larger sizes, which is
place-and-route variance rather than a real gain.

The trend is structural, not incidental. The MXU grows as the square of the
array -- 1,932, 7,535, 30,357 is almost exactly 4x per size step -- while the
VPU processes one row per beat and grows far more slowly, 200, 507, 1,575. A
vector unit of this shape amortises as the array scales.

## The baseline had to be built, not borrowed

The first attempt differenced against **E1's** Gemmini rows and was wrong. E1
routes `Mesh`; this design contains `MeshWithDelays`, which wraps Mesh with the
skew registers, the tag queue and the req/resp control. Charging that wrapper
to the VPU overstated it by 2.7x on lookup tables and 52x on registers. The
`MxuOnly` top above is the matched baseline: the same `MeshWithDelays` with the
same parameters, routed the same way.

**Read the register column with care even so.** The two tops differ in what
they expose at the boundary -- `MxuOnly` drives the mesh's result bus to a
top-level port, where out-of-context synthesis turns it into I/O, while
`MxuVpu` consumes it internally. The lookup-table comparison is sound; the
register deltas are small enough that this difference is a material part of
them.

## Two scale forms, and why the shift one is the comparison

| Array | Form | LUT | FF | DSP | WNS |
|---|---|---:|---:|---:|---:|
| 4x4 | shift | 2,132 | 1,378 | 0 | **+0.591 ns** |
| 4x4 | multiply | 2,436 | 1,403 | 16 | **-1.373 ns** |
| 8x8 | shift | 8,042 | 4,818 | 0 | **+0.456 ns** |
| 8x8 | multiply | 8,585 | 4,882 | 32 | **-1.280 ns** |

SPMW's VPU requantises with `SHR`, an arithmetic shift, so the shift form is
the like-for-like one and it is what the table above uses. Gemmini's more
general multiply-then-shift scale is kept as a variant because it says
something real: a 32x32 signed multiply, shift and clip in one cycle **does not
close at 300 MHz**, and it pulls DSP blocks into a design whose mesh uses none.

## Two Gemmini quirks worth knowing

- `num_units_with_norm` is hardcoded to 4 in `AccumulatorScale.scala`, with a
  `TODO: move to configs` beside it, and is not derived from
  `has_normalizations`. So the multi-unit scale path asserts for any
  `num_scale_units` below 4 even when no unit carries normalization hardware.
  `num_scale_units = -1`, the simple combinational path, avoids it.
- `MeshWithDelays`'s IO bundle cannot be exposed by flipping it wholesale:
  `tags_in_progress` is already an Output, so flipping the aggregate makes both
  sides drivers of it.

## Cycles

There are two drivers, and both check every result against a golden model
rather than only counting.

`MxuVpuDriver` runs **one** matmul: `correct=true` at dim 4 and 8, latency 20
and 36 against the mesh alone's 17 and 33.

`MxuVpuStream` runs the **tiled, streaming** workload the comparison needs --
sixteen `S x S x S` int8 tiles back to back, bias, requantise, ReLU and clip,
all on device -- against the stimulus
`tests/dataflow/spmw/test_spmw_tpu_micro.py` generates. It reads that file
rather than making its own, so it and SPMW are checked against the same bytes.
Weights are double-buffered: `MeshWithDelays` toggles its propagate on every
request and the PE writes `d` into whichever register it is not multiplying by,
so a pass computes with last pass's weights while shifting in the next, and
sixteen tiles take seventeen passes.

| Array | latency | interval | array busy |
|---|---:|---:|---:|
| 4x4 | 22 | 6 | 66.7% |
| 8x8 | 38 | 10 | 80.0% |
| 16x16 | 70 | 18 | 88.9% |

Latency is the first input beat to the last output beat of the first tile;
interval is the gap between tile completions, which is identical for every tile
at every size. The interval is `S + 2` cycles: `S` rows of activations and a
two-cycle request handshake between passes.

**These are xsim's numbers, not chiseltest's.** There is no verilator on the
machine, so chiseltest falls back to a Scala interpreter: 63 seconds at 4x4, 25
minutes at 8x8, and E1 measured six hours for a 16x16 mesh that way. An xsim
testbench on the emitted Verilog does all three in 43 seconds
(`../micro/scripts/gen_mxuvpu_tb.py`). The two agree that the design is correct
and differ by a constant one cycle of interval and two of latency -- 5 against
6 at 4x4, 9 against 10 at 8x8 -- because the xsim driver's request handshake
takes two cycles where chiseltest's takes one. `results.csv` records which
harness each row came from.

### How the two drivers differ, exactly

`MxuVpuStream` (chiseltest) and the xsim testbench
(`../micro/scripts/gen_mxuvpu_tb.py`) drive the same protocol:

1. one `req` per tile carrying the tag, `pe_latency` and the propagate bit;
2. `d` shifts the next tile's weights in `S` beats while `a` streams the
   current tile's `S` activation rows;
3. `out` is read one row a beat, requantised, ReLU'd and clipped by the scale
   path, and checked against the `C` block of `stimulus/stim_S<n>.txt`.

They differ in one place. The xsim driver's request handshake takes two cycles
where chiseltest's takes one, so xsim reports one more cycle of interval and
two more of latency at every size. The table quotes **xsim, the slower of the
two**, which is the reading that flatters SPMW rather than Gemmini.

### The result against SPMW

`../micro/` runs this workload on three designs. Against SPMW's programmable
stage engine Gemmini wins by 27x to 36x. Against a **fixed-function** SPMW
datapath -- the same arithmetic with the instruction fetch, the program and the
`MUL` opcode removed -- it loses the throughput column and wins the area one:

| S | | interval / tile | LUT | FF | DSP | slack |
|---:|---|---:|---:|---:|---:|---:|
| 4 | Gemmini | 6 | **2,132** | **1,378** | **0** | +0.591 |
| | SPMW, fixed-function | **4** | 5,362 | 7,740 | 16 | **+0.927** |
| 8 | Gemmini | 10 | **8,042** | **4,818** | **0** | +0.456 |
| | SPMW, fixed-function | **8** | 20,979 | 30,632 | 64 | **+0.492** |
| 16 | Gemmini | 18 | **31,932** | **18,675** | **0** | +0.245 |
| | SPMW, fixed-function | **16** | 82,858 | 121,960 | 256 | **+0.284** |

Gemmini's interval is `S + 2` and the fixed datapath's is `S`, so SPMW is
1.50x, 1.25x and 1.125x faster per tile and is the only design of the three
that keeps its multiply array busy every cycle. Over exactly sixteen tiles
Gemmini still finishes first, by 1.19x to 1.52x, because SPMW loads all sixteen
tiles' weights up front where Gemmini overlaps a reload with the previous
tile's compute. The full account is in `../micro/README.md`.

### What is not measured

**32x32 cycles.** Area and timing at 32x32 exist for E1's `Mesh` rows but no
Gemmini design here has a cycle count at that size: chiseltest falls back to a
Scala interpreter without verilator, and E1 measured six hours for a 16x16
mesh that way. The xsim route above has no interpreter in it and does not care
about mesh size, so this is a matter of elaborating `MESH_DIM=32` rather than
of a missing tool.

**E1's GEMM workload, on any Gemmini row.** E1 reports Gemmini area and timing
and deliberately no cycles: its shipped `MeshWithDelaysUnitTest` does not
compile against its own HEAD, so a cycle count would come from a driver written
here, and an untuned driver understates the design it drives. The drivers in
*this* directory measure a different workload under a different definition --
first input beat to last output beat of a tile, not first to last memory beat
-- so they do not fill that column and are not quoted into it.
