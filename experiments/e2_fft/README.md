# E2: a radix-2 complex FP32 FFT, in three systems

Same layout as E1: `<framework>/N<size>/{source,generated,report}`, one
`results.csv` that the folders reproduce, and the scripts that drove the runs.
Sizes are transform lengths, N = 128, 256, 512, 1024.

## The datapaths are not the same width, and that has to be read first

SPMW's boundary is **one complex sample per cycle**. HP-FFT's smallest shipped
configuration, UF1, has a boundary of
`hls::stream<hls::vector<complex<float>,2>>` -- 128 bits, **two complex samples
a beat**, N/2 beats a transform. So their ideal intervals differ by 2x by
construction, and comparing the interval columns directly credits HP-FFT with a
factor that is its port rather than its architecture. This is E1's memory-port
problem in another form; unlike E1's it cannot be removed by a flag, because
UF1 is the narrowest configuration the repository ships.

Read the last two columns instead. Each system is close to *its own* ideal:

| N | SPMW cyc/transform | ideal | of ideal | HP-FFT cyc/transform | ideal | of ideal |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 132.1 | 128 | 96.9% | 76.0 | 64 | 84.2% |
| 256 | 264.1 | 256 | 96.9% | 140.5 | 128 | 91.1% |
| 512 | 528.1 | 512 | 97.0% | 269.0 | 256 | 95.2% |
| 1024 | 1056.1 | 1024 | 97.0% | 525.5 | 512 | 97.4% |

SPMW holds 97% of its ideal at every size. HP-FFT climbs from 84% to 97% as the
transform grows, because its per-transform overhead is roughly fixed and is
being amortised over more beats. Neither result is about the other; the honest
throughput statement is samples a cycle, where SPMW sustains 0.97 of its
1-sample datapath and HP-FFT 1.68 to 1.95 of its 2-sample one.

## Results

| N | System | Samples/cyc | First out | Interval | LUT | FF | DSP | BRAM18 | Slack | Clock |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | SPMW folded SDF | 1 | 522 | 132.1 | 10,930 | 25,425 | 139 | 12 | +0.461 ns | 348 MHz |
| | HP-FFT UF1 | 2 | 679 | 76.0 | 19,874 | 19,159 | 60 | 68 | +0.313 ns | 331 MHz |
| | Allo strided | -- | | 11,545 | 4,340 | 7,622 | 16 | 10 | +0.697 ns | 379 MHz |
| 256 | SPMW folded SDF | 1 | 1,035 | 264.1 | 12,361 | 28,865 | 159 | 18 | +0.388 ns | 340 MHz |
| | HP-FFT UF1 | 2 | 1,400 | 140.5 | 22,403 | 22,244 | 72 | 82 | +0.283 ns | 328 MHz |
| | Allo strided | -- | | 26,220 | 4,422 | 7,646 | 16 | 10 | +0.671 ns | 376 MHz |
| 512 | SPMW folded SDF | 1 | 2,059 | 528.1 | 13,864 | 32,222 | 179 | 24 | +0.129 ns | 312 MHz |
| | HP-FFT UF1 | 2 | 2,953 | 269.0 | 25,881 | 25,518 | 84 | 92 | +0.179 ns | 317 MHz |
| | Allo strided | -- | | 58,767 | 4,316 | 7,542 | 16 | 12 | +0.694 ns | 379 MHz |
| 1024 | SPMW folded SDF | 1 | 4,107 | 1,056.1 | 15,423 | 35,587 | 199 | 31 | +0.431 ns | 345 MHz |
| | HP-FFT UF1 | 2 | 6,298 | 525.5 | 29,316 | 28,835 | 96 | 104 | +0.261 ns | 326 MHz |
| | Allo strided | -- | | 130,258 | 4,320 | 7,561 | 16 | 12 | +0.438 ns | 345 MHz |

Interval is cycles a transform in steady state. SPMW's is
`(completion - first_output) / 32` over the 33 transforms of one launch, which
is what its own record defines; HP-FFT's is the mean of the last 24
completion-to-completion distances of its 33; Allo's is its per-call latency,
because it is not a streaming design. Slack is against a 3.333 ns target,
routed out of context on `xcu280-fsvh2892-2L-e` with nothing unrouted, and the
clock is the period that slack implies.

## What each system is

- **SPMW**, `spmw/`: the folded single-path delay-feedback pipeline. log2(N)
  `stage` units in a chain, each with a delay line of N/2^(s+1) complex samples
  and its twiddles as a resident ROM, plus one `reorder` unit that undoes the
  bit-reversal from a double buffer. 8 roles at N=128 rising to 11 at N=1024,
  which is log2(N)+1: the role count is the stage count.
- **HP-FFT**, `hpfft/`: UCLA-VAST/HP-FFT-HLS at `c4611b8`, hand-written HLS.
  256 and 1024 are shipped configurations; **128 and 512 are derived** from the
  shipped sources by changing the size macros, and are labelled that way in
  every path. The design sources are never modified apart from those macros.
- **Allo**, `allo/`: `examples/machsuite/fft/strided`, a sequential in-place
  radix-2 DIF behind four `m_axi` ports. It is not a streaming architecture and
  is two orders of magnitude slower per transform; it is here as the
  general-purpose baseline, not as a competing FFT datapath.

## Where the differences come from

**Area against throughput.** HP-FFT buys its 2x datapath with roughly 1.8x
SPMW's lookup tables and 3.4 to 5.7x its block RAM, while using *fewer* DSPs
(60 to 96 against 139 to 199). The DSP direction is the one that is not
self-explanatory from the port width and is not explained here; it would need
reading how each maps a complex float multiply, which has not been done.

**Allo's cost is a memory dependence, not a width.** Its csynth reports
`HLS 200-880` on the in-place `real[]`/`img[]` arrays and auto-pipelines the
inner loop at II=25, so a butterfly costs tens of cycles rather than one. That
is visible in `report/csynth.rpt` at every size.

## A stale summary, kept

`experiments/results/e2_fft/allo/summary.md` records every P&R row as
`not_run`. That is out of date: `results.csv` beside it has routed results for
all four sizes, and this table uses them. The stale file is left as it is
rather than edited, because it is the earlier agents' record.

## Reproducing

`scripts/` holds what drove these runs. Each `source/` has the design input and
the exact command or tcl; each `generated/` has what the compiler emitted, and
for HP-FFT a note that there is no generation step; each `report/` has the
synthesis, cosimulation and place-and-route reports the numbers come from,
prefixed by stage. SPMW's `generated/` was re-staged from the same
`scripts/spmw_build_array.py` entry point the measured builds used -- the role
counts match its `results.csv` at every size, which is the check that it is the
same code.
