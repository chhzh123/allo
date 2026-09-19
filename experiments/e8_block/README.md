# E8: a transformer block on Gemmini and on SPMW, with the nonlinearities on device

Both engines are 16x16 int8 systolic arrays with a statistics unit and a
scale/activation path, both run the same transformer block, and both are
checked bit-for-bit against the same third thing. The question is what the two
composition models cost when the workload includes the parts a matrix engine
is usually excused from: LayerNorm, softmax and GELU.

## The workload

A small BERT layer -- 64 tokens, 256 model width, 4 heads, a feed-forward of
1024 -- so every part of a block is present and both engines can be simulated
cycle-accurately end to end.

```
h = x + attn(layernorm(x))          softmax inside the attention
y = h + ffn(layernorm(h))           GELU inside the feed-forward
```

That is **52.4M MACs in 12,800 mesh operations of 16x16x16**, and **212,992
elements through the scale path** in eleven passes: two LayerNorms, four
softmaxes, one GELU and four plain requantisations.

## What each engine is, and what is outside both

| | Gemmini | SPMW |
|---|---|---|
| mesh | `MeshWithDelays`, WS, 16x16 | 256 `mac` cells, E3's verbatim |
| statistics | `Normalizer`: 16 accumulation lanes, 16 max lanes, **one** divider, **one** `IntSqrt`, **one** reciprocal, a 15-state machine over two `Stats` banks | `red1`/`red2` (16 lanes each), `sum1`/`sca1`, `sum2`/`sca2` -- six sites, no state machine |
| scale | `AccumulatorScale`, 16 units, float32 `scale_func` | 16 `scale` lanes, the same arithmetic |
| activations | none, relu, layernorm, igelu, softmax | the same five, same encoding |

**The accumulator is outside both.** Gemmini's normaliser is driven by
accumulator *reads* -- three passes over a row, not three matmuls -- so the
top level here exposes `acc_in` and `mesh_out` separately and the host holds
the accumulator. SPMW reads its accumulator three times in the same places,
through `Acc1`/`Acc2`/`Acc3`. Neither side counts memory the other lacks.

**The scale is float32 on both, and that is forced.** Enabling normalizations
in Gemmini is not a flag: `Normalizer` drives `inv_stddev` through `MulPipe`,
which matches only `case Float(expWidth, sigWidth, false)`, as does
`Arithmetic.reciprocal`. So E3's `SInt(32)`-with-a-shift numbers are a
different design point, not a baseline for this.

## Functional equivalence

Not "the two agree" -- both agree with `spmw_block_ref`, a transcription of
Gemmini's source at commit 8c3f992, so a pass is evidence about a
specification. All five activations, bit-exact, on the RTL of both engines.

Four things in the arithmetic would have been wrong if assumed:

- **`mean` is integer division.** `Arithmetic.divider` looks like a float
  divide but is `Float(expWidth = log2Up(32)+1, sigWidth = 32)` with
  `round_minMag` throughout. A 32-bit significand holds any int32 exactly and
  `round_minMag` truncates, so the round trip is C's `/`.
- **The square root is integer** -- `IntSqrt`, restoring, two bits a step. A
  float `sqrtf` of a 32-bit variance is not the same integer.
- **Softmax divides 127**, not 1, by the sum of exponentials. Gemmini's own
  comment: "softmax maximum is 127 for signed int8".
- **The scale rounds half to even.** C truncates. Not a corner case: the
  requantisation scale is a power of two, so an exact half falls on one value
  in 256.

And one the RTL caught that the reference had wrong: **the variance is the
mean of the squared deviations, not their sum.** Gemmini runs its one divider
a second time, `get_sum` -> `get_variance`, so the square root sees
`sum/count`. My reference took the sum, which makes every output `sqrt(len)`
too large -- exactly the factor of 8 at `len = 64` that the first comparison
showed. The reference and the SPMW engine had agreed with each other
perfectly, because the same mistake was written twice.

`igelu`'s requantisation scale is **negative**, and derived rather than
tuned: I-BERT's output scale is `a*S^2/2` and `qc = round(1/(a*S^2))`, so the
scale is `1/(2*qc)` with `qc = -3847`. A least-squares fit against the true
GELU over the whole int8 range gives -1.2878e-4 against the derived
-1.2997e-4, inside the polynomial's own 2.2% error.

## Cycles

Marginal cost of one normalisation row -- two row counts run, the difference
taken, at the exact beat count the block uses. SPMW from RTL cosimulation of
the assembled array, Gemmini from xsim on its own generated Verilog.

| activation | row | beats | SPMW | Gemmini | |
|---|---:|---:|---:|---:|---|
| layernorm | 256 | 16 | **31** | 130 | 4.2x |
| softmax | 64 | 4 | **4** | 41 | 10.3x |
| igelu | 256 | 16 | 16 | 16 | 1.0x |
| igelu | 1024 | 64 | 64 | 64 | 1.0x |
| none | 256 | 16 | 16 | 16 | 1.0x |
| none | 768 | 48 | 48 | 48 | 1.0x |

**The pointwise passes are exactly one cycle a beat on both.** The two that
reduce are where the models differ, and the reason is structural rather than
an oversight on either side: Gemmini's divider, square root and reciprocal are
one each, shared across the whole accumulator and arbitrated by the `Stats`
state machine, so its rows serialise on them -- 130 cycles for a row of which
about 84 is scalar latency. Two statistics banks give it two-way overlap and
no more. SPMW's are unrolled into separate sites that pipeline, which is the
trade the fabric makes everywhere: more of the thing, running at once.

Summed over the block, with the mesh at E3's measured rates for the same two
meshes at the same width -- the one pair of numbers E8 did not re-measure:

| | SPMW | Gemmini | |
|---|---:|---:|---|
| scale path | 15,232 | 37,376 | 2.45x |
| mesh (12,800 tiles) | 204,800 | 230,400 | 1.12x |
| **total** | **220,032** | **267,776** | **1.22x** |

The mesh dominates by 14x, so the scale path's 2.45x becomes 1.22x overall.

## What it took to get SPMW there

Three measurements, in order, each of which changed the design:

1. **The reduction and the scalar arithmetic had to be separate units.** In
   one unit the body is a per-row loop wrapping a reduce whose trip count is a
   runtime field, and HLS reports that outer loop `Pipelined no` -- E3's
   finding in a place I had not applied it. Split, every loop in the design
   reports II=1.
2. **That alone barely helped**: 94 cycles a row became 91. Every unit at
   II=1 and the fabric still at 5.7 cycles a beat says the limit is *between*
   units. Varying the beats per row separated it: `cost = 75 + 1.0 * nbeat`,
   the beats exactly one cycle each and 75 of fixed cost, the same at 4x4,
   8x8 and 16x16. A stage whose iteration takes 65 cycles has to run three or
   four rows ahead of the next one for that latency to hide, and SPMW's
   default depth-2 register slice holds two tokens. Giving the links that
   carry **one token a row** their own depth -- 16 -- takes it to 31. Depth 64
   also gives 31 and costs more, so 16 is the depth that ships.
3. **Pointwise activations were paying for a mean nothing reads.** `sca1`
   divided for every activation except softmax; dividing only for LayerNorm
   took IGELU from 31 cycles a row to 16.

## Resources and clock

Out-of-context on `xcu280-fsvh2892-2L-e`.

*(pending: the SPMW routes)*

**Gemmini closes at 29.0 ns -- 34 MHz -- and retiming does not move it.**
The critical path is 107 logic levels and four chained DSP multiplies, from
`norm/stats_0_state_reg` to the scale's output pipe, and it measured 29.7 ns
at `latency = 1`, 29.0 ns at `latency = 4` with `synth_design -retiming` and
`phys_opt_design -retime` on both sides of routing, and 29.0 ns again at
`latency = 8`. Two constraint periods, 4 ns and 5 ns, landed within 0.5 ns of
each other, which is what a fixed combinational path looks like.

The reason is exact. `AccumulatorScale` computes the activation, the
int-to-float, the float multiply-add, the float-to-int and the clip in **one
combinational cloud**, in both of its branches, and `latency` is a `Pipeline`
on its output. Gemmini's `Pipeline` takes a `comb: Seq[T => T]` of length
`latency + 1` -- combinational functions *interleaved between* its register
stages, which is exactly the hook needed to split this path -- and
`AccumulatorScale` leaves it at the default identity. Its stages also carry
per-stage enables from a stall chain, which is why Vivado declines to retime
across them.

So this is a property of Gemmini's code rather than of its architecture, and
it is fixable in about ten lines of Scala -- but fixing it means changing the
baseline, so it is reported instead. E3's `MxuVpu`, the same design with
normalizations off, closes at +0.245 ns at 3.333 ns: **the mesh is fine, and
the clock is the price of putting the nonlinearities on device in Gemmini's
own code.**

## Honest scope

- The accumulator memory is outside both engines. Its bandwidth is not
  counted, and SPMW reads it three times **concurrently** where Gemmini reads
  it three times in sequence -- SPMW needs three times the accumulator read
  ports for the same passes, which this scope does not charge it for.
- The mesh rates (16 and 18 cycles a tile) are E3's, not re-measured here.
  Both meshes are unchanged: SPMW's cell is E3's verbatim and Gemmini's is
  the same `MeshWithDelays` at the same width.
- `iexp` is restored from the commented-out exponential in Gemmini's source.
  Shipped, its `iexp` is a copy of `igelu` and SOFTMAX is not a softmax;
  matching that bit-for-bit would make the workload not a transformer.
  `scripts/restore_iexp.py` does the restoring and checks brace balance,
  having once swallowed the brace that closes the enclosing object.
- The LLVM-JIT target cannot resolve `math.sqrt`, so `test_spmw_block.py`
  runs the reference simulator and the RTL cosimulation rather than the JIT.
  This design's square root is integer and does not need it; the gap is real
  for any SPMW design that wants a float one.

## Files

- `gemmini/source/MxuVpuNorm.scala` -- the Gemmini top: mesh, `Normalizer`,
  `AccumulatorScale`, `acc_in`/`mesh_out`, and Gemmini's own `scale_func`
  copied verbatim rather than reimplemented.
- `scripts/gem_norm_bench.py` -- the xsim bench. The `NormCmd` schedule is
  built in Python and the Verilog is a player with no policy in it.
- `scripts/pnr_norm.sh` -- the out-of-context route, with retiming.
- `scripts/block_workload.py` -- the block as tiles and rows; no measurements.
- `scripts/block_totals.py` -- the totals, summed from measured rates.
- `scripts/collect.py` -- every log and report into one table.
- `../../tests/dataflow/spmw/spmw_block_engine.py` -- the SPMW engine.
- `../../tests/dataflow/spmw/spmw_block_ref.py` -- the shared reference.
- `../../tests/dataflow/spmw/test_spmw_block.py` -- twelve tests.
