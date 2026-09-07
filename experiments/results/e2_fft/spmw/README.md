# E2, SPMW side: the folded radix-2 FFT at 128/256/512/1024 points

`tests/dataflow/spmw/test_spmw_fft_sdf.py` (registry name `fftsdf`,
`--size N`): a single-path delay-feedback (SDF) radix-2 DIF pipeline, the
architecture HP-FFT folds to. log2(N) `stage` units in a chain, each holding
a delay line of N / 2^(s+1) complex float32 samples and its twiddles as a
resident ROM (`spmw.mem(init=twiddles(N), layout=replicate)`, stationary at
the unit's `tw` port), and one `reorder` unit that writes each block back in
natural order from a double buffer (bit-reversal permutation as a resident
ROM). One sample (re, im) per token, one token per cycle at the boundary;
the launch carries 33 transforms (one extra block flushes the delay lines).
Each stage is its own unit made by a factory (`stage_unit(s)`), so its
half-block `span`, its block count and its twiddle stride are Python
constants: Allo labels the loops and the schedule pipelines and flattens the
block / half / sample nest at one token a cycle. A stage keeps both halves
of the current block (a first, b second, N samples a stage, the same as
HP-FFT's double buffers) and computes on the way out -- (a - b) w leaves
during the next block's first half, a + b during the second half -- so the
float pipeline is feed-forward: a delay line that stores the twiddled
difference has a read-modify-write recurrence that the last stages (span 1,
2, 4) cannot close in one cycle. The reorder unit's single loop lands block b
bit-reversed on one side of its double buffer while block b - 1 streams out
of the other, so a block costs N cycles.

Three earlier versions are recorded in the build logs' history: one loop
with a running counter and a data-dependent address (HLS serialised the
read-modify-write at ~22 cycles a token: 94,295 cycles for 33 transforms at
N=128); two pipelined loops per block with the stage index specialised into
the role (1.8 cycles a token: the large-span stages paid a pipeline fill
per loop entry, 8,194 then 7,935 cycles); and the feed-forward form above.
Bringing the factory-made units up also fixed a framework bug: the lowering
merged same-named captured constants across units, so every stage silently
used the first stage's `span` in the simulator and HLS targets (right at
N=2, wrong from N=4); `allo/spmw/lower_df.py` now renames a conflicting
captured name per body.

Tests: the reference and the OpenMP simulator at N = 8 and 16 against
`numpy.fft` (3 and 4 transforms, atol/rtol 1e-4); the array cosim here at
each N compares every output token of the 33 transforms with the design's
own reference under the design's float tolerance
(`engine.spmw_tolerance = (1e-4, 1e-4)`: the butterflies cancel O(N)
intermediates, so the reference's numpy float32 and the HLS float units
differ by ~1e-5 absolute on a few tokens).

Metrics: `completion_cycles` = first input to last output for the 33
transforms; `first_output_cycles` = latency of the first transform (the
pipeline fill: the first N-1 outputs are the delay lines' initial contents
and are skipped by the reorder); per-transform interval = (completion -
first) / 32 in steady state, which is N cycles when the pipeline sustains one
sample per cycle.

## Two things found on the way

- The first cosims failed with every output zero: Vitis HLS writes ROM
  contents to `.dat` files beside the Verilog, read by `$readmemh` from the
  simulation's working directory, and the array builder copied only the
  `.v` files, so every twiddle read as zero (a - b) * 0. Fixed in
  `spmw_build_array.py` (the `.dat` files travel with the `.v`); P&R was
  never affected.
- With the ROMs in place 11 of 4224 tokens at N=128 differed by rounding
  under the cosim's old class-wide tolerance (rel 1e-5, abs 1e-6); the
  tolerance is now a per-design attribute.

## Files

`results.csv`/`results.json` (one row per (N, mode)); `reports/N<n>_cosim/`
(build log, xsim log), `reports/N<n>_pnr/` (build log, `cost.json`,
`util.rpt`, `timing.rpt`, `route.rpt`, `util_synth.rpt`). P&R is
out of context at 3.333 ns on xcu280-fsvh2892-2L-e, Vivado 2023.2.
