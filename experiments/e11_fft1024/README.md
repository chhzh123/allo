# E11: the radix-2 complex FP32 FFT at N = 1,024

E2's comparison, four times longer, with a third streaming design beside it.
Everything below is routed out of context at 3.333 ns on
`xcu280-fsvh2892-2L-e` with nothing unrouted, and every cycle figure comes
from RTL simulation of transforms streamed back to back, every output checked
-- SPMW's token by token against its reference simulation, HP-FFT's and
Spiral's against `numpy.fft.fft`. `results.csv` is the table;
`scripts/make_results.py` writes it from the reports in this folder.

## The result

SPMW's `lanes` design against HP-FFT and a core of the Spiral DFT generator at
each datapath width. HP-FFT's boundary carries `2*UF` complex samples a beat,
SPMW's and Spiral's carry `W`, so `W = 2*UF` is the matched pairing and all
three share the ideal interval `N / (samples per cycle)`.

| samples/cyc | system | latency | interval | of ideal | LUT | FF | DSP | BRAM18 | slack | MHz | transforms/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | HP-FFT UF1 | 6,809 | 525.5 | 97.4% | 29,316 | 28,835 | 96 | 104 | +0.261 | 326 | 619,450 |
|  | Spiral W=2 | 2,128 | 512.0 | 100.0% | 23,696 | 34,845 | 72 | 72 | +0.978 | 425 | 829,352 |
|  | SPMW lanes W=2 | 2,566 | 512.0 | 100.0% | 30,139 | 36,338 | 96 | 29 | +0.303 | 330 | 644,596 |
| 4 | HP-FFT UF2 | 3,234 | 266.0 | 96.2% | 60,287 | 46,729 | 180 | 164 | +0.150 | 314 | 1,181,087 |
|  | Spiral W=4 | 1,223 | 256.0 | 100.0% | 45,040 | 66,455 | 128 | 108 | +0.844 | 402 | 1,569,405 |
|  | SPMW lanes W=4 | 1,286 | 256.0 | 100.0% | 59,879 | 69,633 | 192 | 40 | +0.214 | 321 | 1,252,405 |
| 8 | HP-FFT UF4 | 1,560 | 138.0 | 92.8% | 148,561 | 177,781 | 342 | 277 | -2.049 | 186 | 1,346,410 |
|  | Spiral W=8 | 777 | 128.0 | 100.0% | 87,540 | 129,368 | 232 | 192 | +0.292 | 329 | 2,569,056 |
|  | SPMW lanes W=8 | 645 | 128.0 | 100.0% | 115,007 | 133,537 | 354 | 56 | +0.249 | 324 | 2,533,236 |
| 16 | HP-FFT UF8 | 937 | 74.0 | 86.5% | 263,862 | 177,909 | 666 | 404 | -0.849 | 239 | 3,231,352 |
|  | Spiral W=16 | 551 | 64.0 | 100.0% | 172,301 | 255,005 | 440 | 336 | +0.275 | 327 | 5,109,549 |
|  | SPMW lanes W=16 | 410 | 64.0 | 100.0% | 220,111 | 257,729 | 678 | 96 | +0.065 | 306 | 4,781,212 |

Latency, interval and BRAM18 are E2's definitions (below). The last column is
transforms a second at the clock each design routed to: `1 / (interval x
achieved period)`.

Five things to take from it.

**SPMW and Spiral hold exactly 100% of the ideal interval at every width.**
HP-FFT is close at this length and slips as it widens: 97.4%, 96.2%, 92.8%,
86.5%. The refill it pays once a transform is the same ten to thirteen cycles
E2 found, on an interval four times longer.

**Spiral's first result arrives soonest at 2 and 4 samples a cycle, SPMW's at
8 and 16, and HP-FFT's last everywhere.** In cycles, for Spiral, SPMW and
HP-FFT: 2,128, 2,566 and 6,809 at 2; 1,223, 1,286 and 3,234 at 4; 777, 645 and
1,560 at 8; 551, 410 and 937 at 16 samples a cycle. In intervals that is five
to six for SPMW, four to nine for Spiral, which grows as it widens, and eleven
to thirteen for HP-FFT. The reason for HP-FFT is E2's: its stages hand each
other whole transforms through ping-pong buffers, so a transform's latency is
the sum of whole-stage runtimes -- ten butterfly stages and the bit reversal's
passes -- where the other two start a stage on its first sample.

**Spiral is the smallest of the three and has the fastest clock.** It has 21%
to 25% fewer lookup tables than SPMW, two thirds to three quarters of its DSP
blocks and registers within 5%, and it closes at 425, 402, 329 and 327 MHz
against SPMW's 330, 321, 324 and 306. At the routed clocks it delivers the
most transforms a second at every width: 29% and 25% more than SPMW at 2 and 4
samples a cycle, 1% and 7% more at 8 and 16. Part of that is arithmetic the
other two do not use; the section on Spiral below says what.

**SPMW uses the least block RAM**: 29, 40, 56 and 96 against Spiral's 72, 108,
192 and 336 and HP-FFT's 104, 164, 277 and 404.

**Against HP-FFT, SPMW is the same size at 2 and 4 samples a cycle and smaller
and faster at 8 and 16.** At 2 and 4 the two are within 3% in lookup tables
and 7% in multipliers, SPMW has 26% and 49% more registers, and both close
timing. At 8 HP-FFT's fixed UF4 has 29% more lookup tables and 33% more
registers than SPMW (148,561 and 177,781 against 115,007 and 133,537), five
times the block RAM, and misses the target by 2.049 ns where SPMW has 0.249 ns
to spare: 1.35 million transforms a second against 2.53 million. At 16 the
fixed UF8 has 20% more lookup tables (263,862 against 220,111) and four times
the block RAM, but 31% fewer registers (177,909 against 257,729); it misses by
0.849 ns and SPMW closes with 0.065 ns, 3.23 million transforms a second
against 4.78 million. In cycles the two are closer than that, 138 against 128
and 74 against 64: most of the gap in rate is the clock.

## HP-FFT at this length

UF1 is E2's row (`experiments/e2_fft/hpfft/N1024`). UF2 is the shipped source
and was built for this experiment. UF4 and UF8 are E2's two fixes
(`experiments/e2_fft/hpfft/README.md`) applied to the shipped N = 1,024
sources by the same `mkvariant.py`:

    python3 mkvariant.py HP-FFT-HLS/n1024/UF4 UF4_a8 static fnpipe
    python3 mkvariant.py HP-FFT-HLS/n1024/UF8 UF8_b4 "repart:data_2:cyclic factor=UF*4"

Those two had already been built, cosimulated and routed on 2026-10-03 with
the scripts this experiment uses, and were not built again:
`hpfft/verify_hpfft.txt` records that a fresh `mkvariant.py` reproduces their
`FFT.cpp`, `FFT.h` and `project.tcl` byte for byte and that the build scripts
differ only in their root directory.

The fixes do what they did at N = 256 to the interval -- the shipped UF4
reports 1,576 cycles against an ideal of 128 in csynth, the fixed one
cosimulates at 138 -- but the UF4 fix does not survive routing at this
length. It pipelines the bit-reversal stage as one function at an interval of
N/8 beats, which unrolls a 128-trip loop over the whole transform:

| N = 1,024 | bit-reversal stage | whole design | worst path |
|---|---:|---:|---|
| UF2 | 1,020 LUT, 494 FF | 60,287 LUT, 46,729 FF | +0.150 ns: a float multiplier (3.0 ns, 50% routing) |
| UF4, fixed | 33,999 LUT, 92,617 FF | 148,561 LUT, 177,781 FF | -2.049 ns: a 512-bit FIFO's full flag, at the stage's boundary (5.4 ns, 85% routing) |
| UF8, fixed | 41,210 LUT, 16,425 FF | 263,862 LUT, 177,909 FF | -0.849 ns: the stage's control (4.0 ns, 85% routing) |

At UF4 the stage is over half the design's registers, and the worst path runs
from the full flag of the 512-bit FIFO at its boundary back into that FIFO's
data register: 5.4 ns, 85% of it routing. At UF8 the worst path is again in
the stage's control. At UF2 it is a float multiplier, which is where it should
be.

E2 named a fallback for a UF4 fix that would not close: `static` alone,
without the function pipeline. Built and routed here (`hpfft/UF4_a1`), it does
close, at 312 MHz with 117,955 lookup tables and 88,365 registers, but its
interval is 391 cycles, 32.7% of ideal: 0.80 million transforms a second,
which is less than the function-pipelined build's 1.35 million at 186 MHz and
less than UF2's 1.18 million at half the area. So the function-pipelined build
is the UF4 row above, and at this length there is no UF4 that reaches its
interval and its clock together.

## Spiral

The cores come from the Spiral DFT/FFT IP generator
(https://www.spiral.net/hardware/dftgen.html), which SwizzleFFT's notes
identify as the streaming baseline of the HP-FFT paper's Table II: forward
transform, single-precision floating point, fully streaming, radix 2, W
complex words a cycle, natural input and output order, the generator's default
block RAM budget. They were not built here. The SwizzleFFT project generated,
simulated and routed them on this part, at this clock, with the out-of-context
script the HP-FFT rows use (its `baselines/spiral_dft/` and
`experiments/spiral_point.sh`, repository version `6d1a2af`), and
`spiral/W*/report/` holds copies of those runs' reports: `sim.json` (the numpy
check and the timing), the testbench, and the routed utilisation, timing and
route status. `make_results.py` reads the rows from those files, and they
reproduce SwizzleFFT's `results/spiral_u280.csv` to the digit. The generated
Verilog is "for internal, non-commercial research use only and is not for
redistribution" and is not in this folder.

Three things differ from the other two rows and should be read with the
table.

- **The arithmetic is Spiral's own.** Its floating-point adders are in logic
  and its multipliers are built on one inferred 24x24 product, two DSP blocks
  each, and both round toward zero. SPMW and HP-FFT use Vitis HLS's IEEE
  units, three DSP blocks a multiplier. So the DSP column is two blocks a
  multiplier against three (Spiral's four cores have 36, 64, 116 and 220
  multipliers), and the lookup tables and the clock are not of the same
  adders. Its largest error against numpy is 4.7e-7 of the largest output,
  HP-FFT's 2.1e-7, on different stimuli. How much of Spiral's lead the
  arithmetic accounts for is not measured here: SPMW was not rebuilt with
  truncating adders or a two-DSP multiplier.
- **It is generated RTL, not HLS.** The other two rows are C++ through Vitis
  HLS, with the handshakes and the float cores that brings.
- **Eight transforms, not 32.** Its simulation streams eight; every
  completion interval in it is the same, so the median is not in doubt.

The generator reports latency to the first output beat (1,617, 968, 650 and
488 cycles here). The table's runs to the last, as for the other rows.

## Allo

Allo's FFT here is E2's: the strided in-place transform, one transform per
call, so its interval is its latency. It is not a streaming pipeline and is
listed for completeness, read from E2's `results.csv`.

| samples/cyc | system | latency | interval | of ideal | LUT | FF | DSP | BRAM18 | slack | MHz | transforms/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | Allo, strided | 130,299 | 130,258.0 |  | 4,320 | 7,561 | 16 | 12 | +0.438 | 345 | 2,652 |

The dataflow FFT on the `feature/allo-fft` branch
(`tests/dataflow/test_fft.py`) was not run: its vectorised design is written
for N = 256 with 32 lanes and needs that branch's layout pass, and its scalar
design is one unit a butterfly, 5,120 of them at this length.

## How the numbers are defined

As in E2.

- **Latency** is a transform's last output beat minus the launch's first input
  beat.
- **Interval** is the median of the last three quarters of the differences
  between consecutive transform completions. Spiral's differences are all
  equal; HP-FFT's are all equal in every row but UF1, where they alternate
  between 522 and 529; SPMW's are all equal but the last of a launch, which
  is four cycles longer (`results.csv`).
- **BRAM18** is twice the block RAM tiles.
- **Adders in fabric on both sides**, as in E2.

## Layout

    spmw/W<lanes>/source/       the design and the command
    spmw/W<lanes>/generated/    the fabric and each role's HLS C++
    spmw/W<lanes>/report/       routed utilisation and timing, the build log,
                                each transform's completion cycle (xform.txt)
    hpfft/<config>/source/      FFT.cpp, FFT.h and the project
    hpfft/<config>/report/      csynth, the cosimulation's events and numpy
                                check, routed utilisation and timing
    spiral/W<lanes>/report/     SwizzleFFT's run of the Spiral core: numpy check
                                and timing, testbench, routed reports
    scripts/                    what drove the runs

## Reproducing

    scripts/fft_build.sh 1024 <lanes> --jobs 6     # SPMW: HLS, cosimulation, route
    scripts/hp_run.sh n1024 <config>               # HP-FFT: csynth, cosimulation, route
    scripts/collect_fft.sh                         # stages this folder from the builds
    scripts/collect_spiral.sh                      # copies SwizzleFFT's Spiral reports
    python3 scripts/make_results.py                # results.csv and the tables
