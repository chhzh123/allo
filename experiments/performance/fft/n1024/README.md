# Fig. 9(d) and Table 1 (FFT rows), arXiv version: the 1,024-point FFT

SPMW's lane design, HP-FFT and a Spiral core at each width; Table 1 is the
w = 16 group. Definitions, the Spiral caveats and the throughput rule are in
[`../README.md`](../README.md). `results.csv` is the table, and
`scripts/make_results.py` writes it from the reports in this folder (rerunning
it reproduces the file byte for byte). Every row has its raw reports; only
HP-FFT UF1 lacks a `validate_rtl_vcd.json`.

| w | system | latency | interval | of ideal | MS/s | LUT | FF | DSP | BRAM18 | slack | MHz |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | HP-FFT UF1 | 6,809 | 525.5 | 97.4% | 584.59 | 29,316 | 28,835 | 96 | 104 | +0.261 | 326 |
|  | Spiral W=2 | 2,128 | 512.0 | 100% | 600.00 | 23,696 | 34,845 | 72 | 72 | +0.978 | 425 |
|  | SPMW W=2 | 2,566 | 512.0 | 100% | 600.00 | 30,139 | 36,338 | 96 | 29 | +0.303 | 330 |
| 4 | HP-FFT UF2 | 3,234 | 266.0 | 96.2% | 1,154.89 | 60,287 | 46,729 | 180 | 164 | +0.150 | 314 |
|  | Spiral W=4 | 1,223 | 256.0 | 100% | 1,200.00 | 45,040 | 66,455 | 128 | 108 | +0.844 | 402 |
|  | SPMW W=4 | 1,286 | 256.0 | 100% | 1,200.00 | 59,879 | 69,633 | 192 | 40 | +0.214 | 321 |
| 8 | HP-FFT UF4, fixed | 1,560 | 138.0 | 92.8% | 1,378.72 | 148,561 | 177,781 | 342 | 277 | -2.049 | 186 |
|  | Spiral W=8 | 777 | 128.0 | 100% | 2,400.00 | 87,540 | 129,368 | 232 | 192 | +0.292 | 329 |
|  | SPMW W=8 | 645 | 128.0 | 100% | 2,400.00 | 115,007 | 133,537 | 354 | 56 | +0.249 | 324 |
| 16 | HP-FFT UF8, fixed | 937 | 74.0 | 86.5% | 3,308.90 | 263,862 | 177,909 | 666 | 404 | -0.849 | 239 |
|  | Spiral W=16 | 551 | 64.0 | 100% | 4,800.00 | 172,301 | 255,005 | 440 | 336 | +0.275 | 327 |
|  | SPMW W=16 | 410 | 64.0 | 100% | 4,800.00 | 220,111 | 257,729 | 678 | 96 | +0.065 | 306 |

`make_results.py` prints this table with transforms a second at the uncapped
clock in place of MS/s.

- **SPMW and Spiral reach exactly N/w at every width.** HP-FFT is close at this
  length and slips as it widens: 97.4%, 96.2%, 92.8%, 86.5%. Its refill is 10
  to 13.5 cycles a transform.
- **SPMW answers 2.29x to 2.65x sooner than HP-FFT.** Against Spiral it is
  later at w = 2 and 4 (2,566 and 1,286 against 2,128 and 1,223) and sooner at
  8 and 16 (645 and 410 against 777 and 551): the permutation and reorder
  delays grow with N/w, and wider lanes shorten them.
- **SPMW holds the least block RAM**: 29, 40, 56, 96 against Spiral's 72, 108,
  192, 336 and HP-FFT's 104, 164, 277, 404.
- **HP-FFT's fixed UF4 and UF8 miss 300 MHz** (-2.049 and -0.849 ns; next
  section). SPMW and Spiral close at every width and, at the 300 MHz cap,
  deliver the same throughput. Spiral is smaller and clocks higher.

## HP-FFT at this length

- **UF1** was measured with the 256-point experiment's flow; its reports are in
  `hpfft/UF1/report/` and its row is `hpfft/UF1/report/e2_row.csv`, which
  `make_results.py` reads.
- **UF2** is the shipped source.
- **UF4_a8 and UF8_b4** are the 256-point fixes
  ([`../n256/hpfft/README.md`](../n256/hpfft/README.md)) applied to the shipped
  1,024-point sources by the same `mkvariant.py`:

      python3 ../n256/scripts/hpfft/mkvariant.py HP-FFT-HLS/n1024/UF4 UF4_a8 static fnpipe
      python3 ../n256/scripts/hpfft/mkvariant.py HP-FFT-HLS/n1024/UF8 UF8_b4 "repart:data_2:cyclic factor=UF*4"

  They were built, cosimulated and routed on 2026-10-03 under
  `/scratch/hc676/hpfft_fix` and `/scratch/hc676/hpfft_pnr`.
  `hpfft/verify_hpfft.txt` records that a fresh `mkvariant.py` reproduces their
  `FFT.cpp`, `FFT.h` and `project.tcl` byte for byte and that the build scripts
  differ from this experiment's only in their root directory.

The fixes recover the interval here as at 256 points (138 and 74 against
ideals of 128 and 64), but the UF4 fix does not survive routing. It pipelines
the bit-reversal stage as one function at an interval of N/8 beats, which
unrolls a 128-trip loop over the whole transform. From `pnr_util_hier.rpt` and
`pnr_timing.rpt`:

| | bit-reversal stage | whole design | worst path |
|---|---:|---:|---|
| UF2 | 1,020 LUT, 494 FF | 60,287 LUT, 46,729 FF | +0.150 ns: a float multiplier (3.0 ns, 50% routing) |
| UF4, fixed | 33,999 LUT, 92,617 FF | 148,561 LUT, 177,781 FF | -2.049 ns: a 512-bit FIFO's full flag at the stage's boundary (5.4 ns, 85% routing) |
| UF8, fixed | 41,210 LUT, 16,425 FF | 263,862 LUT, 177,909 FF | -0.849 ns: the stage's control (4.0 ns, 85% routing) |

At UF4 the stage is over half the design's registers. At UF2 the worst path is
a float multiplier, which is where it should be. So at this length no HP-FFT
UF4 here reaches its interval and its clock together; the fixed one is the row
the paper uses.

## Layout

    results.csv                  written by scripts/make_results.py
    spmw/W<w>/source/            lanes.py and command.txt
    spmw/W<w>/generated/         fabric/*.sv and roles/*.cpp, each unit's HLS C++
    spmw/W<w>/report/            routed util, timing and route reports, cost.json,
                                 the build log, each role's csynth, and xform.txt
                                 (each transform's completion cycle)
    hpfft/<cfg>/source/          FFT.cpp, FFT.h, project.tcl, common.tcl
    hpfft/<cfg>/report/          csynth, cosimulation events and stimulus, numpy
                                 check, routed reports, build log
    hpfft/UF1/generated/README.md   a note: HP-FFT has no generation step
    hpfft/verify_hpfft.txt       the reused builds against a fresh mkvariant.py
    spiral/W<w>/report/          SwizzleFFT's run of the Spiral core
    scripts/                     what drove the runs

## Reproducing

    scripts/fft_build.sh 1024 <w> --jobs 6      # SPMW: HLS, cosimulation, route
    scripts/hp_run.sh n1024 <cfg>               # HP-FFT: csynth, cosimulation, route
    scripts/collect_fft.sh                      # stages SPMW and HP-FFT UF2/UF4/UF8 from the builds
    scripts/collect_spiral.sh                   # copies SwizzleFFT's Spiral reports, both lengths
    python3 scripts/make_results.py             # results.csv and the table

`fft_build.sh` wraps the command in `spmw/W<w>/source/command.txt`.
`hp_run.sh` runs copies of `../n256/scripts/hpfft/run_hls.sh` and
`run_cosim5.sh` (as `run_cosim.sh`) under `/scratch/hc676/e14_fft/hpfft`, then
routes with the same out-of-context steps as `run_pnr.sh`. HP-FFT UF1's files
were staged by `../n256/scripts/collect_e2.sh`.
