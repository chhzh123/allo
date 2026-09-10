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

## Still missing

**No cycle measurement exists for this baseline.** Everything above is area and
timing. A cycle count needs a driver that runs matmul, accumulate, scale and
ReLU against a golden model, which is the next piece of work, and the matching
SPMW program after it. Nothing here should be quoted as a throughput result.
