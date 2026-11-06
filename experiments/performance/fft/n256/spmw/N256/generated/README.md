# The three SPMW FFT designs: source and generated hardware

`source/` holds the designs; this directory holds what the compiler emitted
from them and the reports every claim is read out of.

| Directory | Design | Build |
|---|---|---|
| `rolled/` | rolled W=2, adders in fabric | `n256w2_bind_p` |
| `rolled_w1/` | rolled W=1 (the folded SDF point) | `n256w1_p` |
| `paired/` | paired-operand W=2 | `n256w2_triv_p` |
| `lanes/` | lane-unrolled W=2 | `n256w2_p` |

Each holds `roles/<unit>.cpp` (the HLS C++ the frontend emitted per unit),
`fabric/*.sv` (`spmw_top` and the register slices that wire the units), and
`reports/*_csynth.rpt` (where `Interval = 1` is read; look for `yes(flp)`).

Vitis project internals are **not** here -- they are hundreds of megabytes per
build and no reported number comes from them. The builds themselves stay on
brg-zhang-xcel under `/scratch/hc676/spmw_fft_rolled`, `spmw_fft_paired` and
`spmw_fft_lanes`.

## The three designs differ in structure, not just in width

This is worth reading before comparing their resource columns.

**`rolled/` encodes the topology explicitly.** Its cross stages name each
butterfly's partner as a link between sites:

```python
CrossIO.a_out: spmw.to((t + 1, ell), CrossIO.a_in),
CrossIO.b_out: spmw.to((t + 1, ell ^ nxt), CrossIO.b_in),
```

one unit per lane per stage, placed on a 2-D grid, with the partner either a
wire to lane `ell ^ nxt` or a delay line in the same lane. That is the
point-to-point form: nothing is a shared buffer. But a unit is fed **one**
sample a cycle and a butterfly consumes two, so every multiplier idles half the
time and the design needs twice as many of them.

**`paired/` does not.** Every stage is placed on `spmw.Grid((1,))` -- a single
site -- and there is **no `spmw.to` anywhere in the design**. Stages are chained
through the fabric's stream ports and each stage holds its own buffer
internally. That is what lets one unit see both operands of a butterfly in the
same cycle, and it is how the multiplier count halves; it is also a step back
towards per-stage buffers, and the block RAM column shows it.

**`lanes/` keeps both.** The `W` lanes are `W` separate streams and a stage is
`W/2` sites, so a lane index is a placement coordinate -- a compile-time
constant, and no bank arithmetic exists to be proven conflict-free. A site
reads its two operands on two ports. What makes that possible is a
delay-switch-delay permutation in front of every stage whose butterfly distance
spans beats: its output pair at any beat is one lane's samples `d` beats apart,
so choosing `d` to match the stage's distance turns a time separation into a
lane separation. There is no shared buffer at all; the only memory is the
permutation delays, about `1.5 n` complex samples for the whole pipeline
against the paired design's `2n` *per stage*.

At W=2 the design is a chain, because the crossings are a function of the
width: only stages whose distance is below the lane count cross lanes, and
there are `log2(W) - 1` of those. At W >= 4 they go on one `spmw.Topology`
whose `link` rule names each crossing with `spmw.to`, as `rolled/` does.

## Read the block RAM column carefully -- it is two different units

`results.csv` records rolled's block RAM as **6** and paired's as **44**, and
those are not the same measurement. From each build's own `util.rpt`:

| design | Block RAM Tile | RAMB36/FIFO | RAMB18 | BRAM18-equivalent |
|---|---:|---:|---:|---:|
| rolled W=2, bound | 6 | 0 | 12 | **12** |
| paired W=2 | 22 | 0 | 44 | **44** |
| lanes W=2 | 5 | 0 | 10 | **10** |
| HP-FFT UF1 | 41 | 36 | 10 | **82** |

So rolled's 6 is its *tile* count and paired's 44 is its *RAMB18* count. The
earlier version of this file said "6 BRAM18 for rolled against 44 for paired",
which compares one against the other; in one unit it is 12 against 44, or 6
against 22. `rolled_sweep.py` computes `2*RAMB36 + RAMB18` and is right.

## What the three trade

Measured at N=256, W=2 on xcu280-fsvh2892-2L-e at a 3.333 ns target, adders
bound to fabric on every SPMW row so the DSP column is like for like with
HP-FFT, which binds its own:

| | interval | % of ideal | DSP | LUT | FF | BRAM18 | WNS ns | MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HP-FFT UF1 | 140.5 | 91.1% | 72 | 22,403 | 22,244 | 82 | +0.283 | 328 |
| rolled W=2 | **128.0** | **100%** | 174 | 36,567 | 46,704 | 12 | +0.345 | 335 |
| paired W=2 | **128.0** | **100%** | **72** | **21,266** | **24,582** | 44 | +0.432 | 345 |
| lanes W=2 | **128.0** | **100%** | **72** | 24,012 | 28,614 | **10** | **+0.443** | **346** |

`lanes/` is the row that reaches the paired design's DSP count without its
block RAM: it is the only one of the three that is at or below every other
SPMW row on both columns at once. It pays about 13% more lookup tables and 16%
more flip-flops than `paired/` for the permutation network that replaces the
buffer.

None of the three supersedes the others. `rolled/` is the simplest and the one
whose topology is most explicit at W=1; `paired/` is the smallest in lookup
tables; `lanes/` is the one that gets DSP parity and low block RAM together.
