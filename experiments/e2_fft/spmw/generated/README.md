# The two SPMW FFT designs: source and generated hardware

`source/` holds the designs; this directory holds what the compiler emitted
from them and the reports every claim is read out of.

| Directory | Design | Build |
|---|---|---|
| `rolled/` | rolled W=2, adders in fabric | `n256w2_bind_p` |
| `rolled_w1/` | rolled W=1 (the folded SDF point) | `n256w1_p` |
| `paired/` | paired-operand W=2 | `n256w2_triv_p` |

Each holds `roles/<unit>.cpp` (the HLS C++ the frontend emitted per unit),
`fabric/*.sv` (`spmw_top` and the register slices that wire the units), and
`reports/*_csynth.rpt` (where `Interval = 1` is read; look for `yes(flp)`).

Vitis project internals are **not** here -- they are hundreds of megabytes per
build and no reported number comes from them. The builds themselves stay on
brg-zhang-xcel under `/scratch/hc676/spmw_fft_rolled` and `spmw_fft_paired`.

## The two designs differ in structure, not just in width

This is worth reading before comparing their resource columns.

**`rolled/` encodes the topology explicitly.** Its cross stages name each
butterfly's partner as a link between sites:

```python
CrossIO.a_out: spmw.to((t + 1, ell), CrossIO.a_in),
CrossIO.b_out: spmw.to((t + 1, ell ^ nxt), CrossIO.b_in),
```

one unit per lane per stage, placed on a 2-D grid, with the partner either a
wire to lane `ell ^ nxt` or a delay line in the same lane. That is the
point-to-point form: nothing is a shared buffer.

**`paired/` does not.** Every stage is placed on `spmw.Grid((1,))` -- a single
site -- and there is **no `spmw.to` anywhere in the design**. Stages are chained
through the fabric's stream ports and each stage holds its own buffer
internally. That is what lets one unit see both operands of a butterfly in the
same cycle, and it is how the multiplier count halves; it is also a step back
towards per-stage buffers, and the block RAM column shows it: **6 BRAM18 for
rolled against 44 for paired**.

So the paired design buys its DSP parity with structure the rolled design was
written to avoid. Both are kept because that trade is the finding, and neither
supersedes the other.
