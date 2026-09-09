# E2: a radix-2 complex FP32 FFT, in three systems

Same layout as E1: `<framework>/N<size>/{source,generated,report}`, one
`results.csv` that the folders reproduce, and the scripts that drove the runs.
Sizes are transform lengths, N = 128, 256, 512, 1024.

## The datapaths are not the same width, and that has to be read first

SPMW's boundary is **one complex sample per cycle**. HP-FFT's smallest shipped
configuration, UF1, has a boundary of
`hls::stream<hls::vector<complex<float>,2>>` -- 128 bits, **two complex samples
a beat**. So their ideal intervals differ by 2x by construction, and unlike
E1's memory-port mismatch this one cannot be flagged away: UF1 is the narrowest
configuration the repository ships.

Both are therefore measured on one definition, the one HP-FFT's own harness
records: **cycles are rising clock edges; a transform's latency is its last
output beat minus the launch's first input beat; the steady interval is the
median of consecutive completion differences.** SPMW's cosimulation reports
per-transform completions for this (`spmw_tokens_per_transform` on the fabric),
so the two sides are read the same way rather than compared across two
different quantities.

## Latency and throughput point in opposite directions

| N | System | Full-transform latency | Steady interval | Ideal | Of ideal | Samples/cycle |
|---|---|---:|---:|---:|---:|---:|
| 128 | SPMW | **581** | 128.0 | 128 | **100.0%** | 1.000 |
| | HP-FFT UF1 | 742 | 76.0 | 64 | 84.2% | **1.684** |
| 256 | SPMW | **1,158** | 256.0 | 256 | **100.0%** | 1.000 |
| | HP-FFT UF1 | 1,527 | 140.5 | 128 | 91.1% | **1.822** |
| 512 | SPMW | **2,310** | 512.0 | 512 | **100.0%** | 1.000 |
| | HP-FFT UF1 | 3,208 | 269.0 | 256 | 95.2% | **1.903** |
| 1024 | SPMW | **4,614** | 1024.0 | 1024 | **100.0%** | 1.000 |
| | HP-FFT UF1 | 6,809 | 525.5 | 512 | 97.4% | **1.949** |

**SPMW has the lower full-transform latency at every size, despite half the
datapath**, and its advantage grows with N: 1.28x at 128 to 1.48x at 1024. Its
latency is a flat 4.5N; HP-FFT's is 5.8N rising to 6.7N.

**HP-FFT has the higher throughput at every size**, and its advantage also
grows, 1.68x to 1.95x, as its fixed per-transform overhead amortises and it
approaches the 2x its datapath allows.

The two results have one cause. SPMW sustains **exactly N cycles a transform at
every size** -- 100% of its ideal, measured as a median of 32 consecutive
completions with only the final drain differing (min 128, max 132 at N=128).
HP-FFT runs at 84% of its own ideal at 128 and only reaches 97% at 1024. A
wider datapath that is not kept full buys throughput and costs latency.

## A correction

An earlier version of this table reported SPMW's interval as 132.1 at N=128
(96.9% of ideal), rising to 1,056.1 at N=1024. Those figures came from
`(launch total - first output) / 32`, which is wrong twice: it starts at the
first *output* beat, so the pipeline fill sits inside the interval, and it
divides by 32 when 33 transforms are emitted. At N=128 that turns a true 128.0
into 128 + 127/32 = 132.1. The `completion_cycles` column was worse: SPMW's
held the span of all 33 transforms and HP-FFT's held one transform, a factor of
thirty apart in the same column. **The HP-FFT column was not affected by
either error**; only SPMW's side was wrong. `results.csv` now carries
`full_transform_latency` on one definition for both, and each transform's
completion cycle together with the launch's first input beat is in
`spmw/N*/report/cosim_transforms.log`.

## Results

| N | System | Samples/cyc | First out | Full xform | Interval | LUT | FF | DSP | BRAM18 | Slack | Clock |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | SPMW folded SDF | 1 | 454 | 581 | 128.0 | 10,930 | 25,425 | 139 | 12 | +0.461 ns | 348 MHz |
| | HP-FFT UF1 | 2 | 679 | 742 | 76.0 | 19,874 | 19,159 | 60 | 68 | +0.313 ns | 331 MHz |
| | Allo strided | -- | | 11,545 | 11,545 | 4,340 | 7,622 | 16 | 10 | +0.697 ns | 379 MHz |
| 256 | SPMW folded SDF | 1 | 903 | 1,158 | 256.0 | 12,361 | 28,865 | 159 | 18 | +0.388 ns | 340 MHz |
| | HP-FFT UF1 | 2 | 1,400 | 1,527 | 140.5 | 22,403 | 22,244 | 72 | 82 | +0.283 ns | 328 MHz |
| | Allo strided | -- | | 26,220 | 26,220 | 4,422 | 7,646 | 16 | 10 | +0.671 ns | 376 MHz |
| 512 | SPMW folded SDF | 1 | 1,799 | 2,310 | 512.0 | 13,864 | 32,222 | 179 | 24 | +0.129 ns | 312 MHz |
| | HP-FFT UF1 | 2 | 2,953 | 3,208 | 269.0 | 25,881 | 25,518 | 84 | 92 | +0.179 ns | 317 MHz |
| | Allo strided | -- | | 58,767 | 58,767 | 4,316 | 7,542 | 16 | 12 | +0.694 ns | 379 MHz |
| 1024 | SPMW folded SDF | 1 | 3,591 | 4,614 | 1024.0 | 15,423 | 35,587 | 199 | 31 | +0.431 ns | 345 MHz |
| | HP-FFT UF1 | 2 | 6,298 | 6,809 | 525.5 | 29,316 | 28,835 | 96 | 104 | +0.261 ns | 326 MHz |
| | Allo strided | -- | | 130,258 | 130,258 | 4,320 | 7,561 | 16 | 12 | +0.438 ns | 345 MHz |

Every cycle figure is relative to the launch's **first input beat**, so "first
out" and "full xform" are latencies rather than absolute cycles. Interval is
the median of consecutive transform completions. Allo is not a streaming
design, so it has no first-output beat and its per-call latency is both its
transform latency and its interval. Slack is against a 3.333 ns target, routed
out of context on `xcu280-fsvh2892-2L-e` with nothing unrouted; the clock is
the period that slack implies.

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
