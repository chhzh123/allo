# Fig. 9(c,d) and Table 1 (FFT rows), arXiv version: the radix-2 FFT

Latency against sustained throughput for forward complex FP32 FFTs of 256 and
1,024 points at w = 2, 4, 8 and 16 complex samples a cycle, in three systems:

| System | What it is | At width w |
|---|---|---|
| SPMW | the lane design (`lanes.py`, a copy of `tests/dataflow/spmw/test_spmw_fft_lanes.py`): HLS C++ per unit, SystemVerilog fabric | `W = w` lanes |
| HP-FFT | hand-written HLS C++, UCLA-VAST/HP-FFT-HLS @ `c4611b8` | `UF = w/2` butterfly lanes a stage; its boundary carries `2 UF` samples a beat |
| Spiral | a core of the Spiral DFT generator, RTL | fully streaming, radix 2, natural input and output order |

| Paper | Folder | Rows |
|---|---|---|
| Fig. 9(c) | [`n256/`](n256/) | all 12 of `n256/results.csv` |
| Fig. 9(d) | [`n1024/`](n1024/) | all 12 of `n1024/results.csv` |
| Table 1, FFT | [`n1024/`](n1024/) | the three w = 16 rows of `n1024/results.csv` |

Table 1's rows:

| | latency | interval | MS/s | LUT | FF | DSP | BRAM18 | MHz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SPMW W=16 | 410 | 64 | 4,800.00 | 220,111 | 257,729 | 678 | 96 | 306 |
| HP-FFT UF8 | 937 | 74 | 3,308.90 | 263,862 | 177,909 | 666 | 404 | 239 |
| Spiral W=16 | 551 | 64 | 4,800.00 | 172,301 | 255,005 | 440 | 336 | 327 |

## Definitions, for both sizes

- **Flow.** Vitis HLS (SPMW, HP-FFT) and Vivado 2023.2, `xcu280-fsvh2892-2L-e`,
  out-of-context place and route at 3.333 ns; no routed report here has a
  routing error.
  Cycle counts come from RTL simulation of transforms streamed back to back.
- **Latency** (`full_transform_latency`): first accepted input beat to the last
  output beat of the first transform.
- **Interval** (`steady_interval_cycles`): the median of the last three quarters
  of the differences between consecutive transform completions.
- **Ideal interval** is `N / w`; **% of ideal** is `100 (N/w) / interval`.
- **Throughput** is `N f / interval` MS/s with `f = min(300, 1000 / (3.333 - WNS))`
  MHz, unrounded. The CSVs' `fmax_mhz` is that clock rounded and uncapped.
  `n1024/results.csv`'s `transforms_per_s` uses the uncapped clock; it is not
  what the paper plots.
- **BRAM18** is `2 x Block RAM Tile` from the routed `util.rpt`, for every row.
  The `RAMB18` line alone undercounts designs built of RAMB36: HP-FFT UF1 at 256
  points has 36 RAMB36 and 10 RAMB18, 41 tiles, so 82.
- **Adders in fabric on both HLS sides**: HP-FFT binds its butterfly's float adds
  with `bind_op ... impl=fabric`; `lanes.py` sets `spmw_bind_fabric` (on unless
  `SPMW_BIND_FABRIC=0`). Without it the DSP column compares who wrote the pragma.
- **Checks.** SPMW streams 32 transforms and compares every output token with
  its reference simulation (`SPMW COSIM PASS`). HP-FFT feeds 33 transforms of
  seeded numpy stimulus and completes 32 (under 2023.2 the last is never
  flushed; `n256/scripts/hpfft/run_cosim5.sh` says why); `validate_rtl_vcd.json`
  checks them against `numpy.fft.fft` (UF1 has none here). `results.csv`
  records 33 for UF1 and 32 for the other HP-FFT rows. Spiral streams 8, checked against numpy; its
  completion intervals are all equal.

## Spiral, at both sizes

The cores come from the Spiral DFT/FFT IP generator
(https://www.spiral.net/hardware/dftgen.html): forward, single-precision
floating point, fully streaming, radix 2, W complex words a cycle, natural
order, the generator's default block RAM budget. They were not built here. The
SwizzleFFT project (version `6d1a2af`) generated, simulated and routed them on
this part at this clock with the same out-of-context flow, and
`n256/spiral/N256/report/W*/` and `n1024/spiral/W*/report/` are copies of those
runs' reports: `sim.json` (numpy check and timing), `sim.log`, the testbench,
and routed utilisation, timing and route status. Both `results.csv` files match
them. The generated Verilog is "for internal, non-commercial research use only
and is not for redistribution" and is not here.

Read the Spiral rows with three differences in mind.

- **Its arithmetic is its own.** Adders in logic; multipliers on one inferred
  24x24 product, two DSP blocks each; both round toward zero. SPMW and HP-FFT
  use Vitis HLS's IEEE units, three DSP blocks a multiplier. So Spiral's DSP
  column is twice its multiplier count (28, 48, 84, 156 at 256 points; 36, 64,
  116, 220 at 1,024), and its LUTs and clock are not of the same adders. Its
  largest error against numpy is 4.6e-7 of the largest output at 256 points and
  4.7e-7 at 1,024, HP-FFT's 1.8e-7 and 2.1e-7, on different stimuli. How much of
  Spiral's lead the arithmetic explains is not measured.
- **It is generated RTL**, not HLS with its handshakes and float cores.
- **Its own latency figure is to the first output beat** (`first_output` in
  `sim.json`: 595, 418, 336, 292 at 256 points; 1,617, 968, 650, 488 at 1,024).
  The tables use the last beat, as for the other rows.

## HP-FFT's corrected configurations

At w = 8 and 16 both sizes use corrected HP-FFT sources, not the shipped ones:
UF4 with static scratch arrays and a function-pipelined bit-reversal stage, UF8
with `data_2` repartitioned. [`n256/hpfft/README.md`](n256/hpfft/README.md) says
what was wrong and what was changed; [`n1024/README.md`](n1024/README.md) says
what the UF4 fix costs in timing at 1,024 points.

## What is not here

- Some raw reports at 256 points: SPMW W=4, 8, 16 and HP-FFT UF2 are
  `results.csv` rows only, and SPMW W=2 and HP-FFT's fixed UF4 and UF8 have no
  routed reports; [`n256/README.md`](n256/README.md) lists what each row has.
  Every 1,024-point row has its reports.
- Runs the paper does not use (Allo's FFT, the other SPMW FFT designs, other
  lengths, the shipped HP-FFT UF4/UF8 builds) are in commit `c20b2ee8`. Table
  3's FFT row counts the folded SPMW design,
  [`../../productivity/code_size/sources/fft_folded_spmw.py`](../../productivity/code_size/sources/fft_folded_spmw.py).
