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

## Status

Feasibility is settled and nothing else here is measured yet.

- **Done:** the scale path elaborates standalone. `Gate.scala` emits 60,518
  characters of Verilog from `AccumulatorScale` with no scratchpad, controller
  or RoCC. `setup_mxuvpu.sh` assembles the build from `gemmini_src` at
  `8c3f992`; the added files are `AccumulatorMem`, `AccumulatorScale`,
  `Activation`, `Normalizer` (types only), `NormCmd`, `Pipeline` and
  `SharedExtMem`, all of which import chisel3, hardfloat and `Util` alone.
- **Next:** wire `MeshWithDelays` to `AccumulatorScale` as one top; a driver
  that runs matmul, accumulate, scale and ReLU against a golden model; then
  cycles and place-and-route, and the matching SPMW program.

No cycle or area number for this baseline exists yet, and none should be
quoted until it is measured.
