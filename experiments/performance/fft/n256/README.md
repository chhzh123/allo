# Fig. 9(c), arXiv version: the 256-point FFT

SPMW's lane design, HP-FFT and a Spiral core at each width. Definitions, the
Spiral caveats and the throughput rule are in [`../README.md`](../README.md).
`results.csv` is the table; the paper reads `full_transform_latency`,
`steady_interval_cycles`, the resource columns and `achieved_ns`. Ignore
`completion_cycles` on the SPMW rows: it is the same 5,147 at every width and
is not a transform's latency.

| w | system | latency | interval | of ideal | MS/s | LUT | FF | DSP | BRAM18 | slack | MHz |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | HP-FFT UF1 | 1,527 | 140.5 | 91.1% | 546.62 | 22,403 | 22,244 | 72 | 82 | +0.283 | 328 |
|  | Spiral W=2 | 722 | 128.0 | 100% | 600.00 | 18,735 | 27,419 | 56 | 48 | +1.263 | 483 |
|  | SPMW W=2 | 645 | 128.0 | 100% | 600.00 | 24,012 | 28,614 | 72 | 10 | +0.443 | 346 |
| 4 | HP-FFT UF2 | 766 | 74.5 | 85.9% | 1,030.87 | 45,482 | 37,809 | 132 | 104 | +0.317 | 332 |
|  | Spiral W=4 | 481 | 64.0 | 100% | 1,200.00 | 35,212 | 51,742 | 96 | 84 | +0.868 | 406 |
|  | SPMW W=4 | 376 | 64.0 | 100% | 1,200.00 | 45,711 | 54,192 | 144 | 20 | +0.241 | 323 |
| 8 | HP-FFT UF4, fixed | 408 | 42.0 | 76.2% | 1,828.57 | 91,524 | 90,201 | 246 | 166 | +0.119 | 311 |
|  | Spiral W=8 | 367 | 32.0 | 100% | 2,400.00 | 67,969 | 100,075 | 168 | 144 | +0.427 | 344 |
|  | SPMW W=8 | 260 | 32.0 | 100% | 2,400.00 | 87,021 | 102,464 | 258 | 40 | +0.292 | 329 |
| 16 | HP-FFT UF8, fixed | 269 | 26.0 | 61.5% | 2,873.95 | 179,275 | 148,768 | 474 | 288 | -0.093 | 292 |
|  | Spiral W=16 | 307 | 16.0 | 100% | 4,800.00 | 133,242 | 196,566 | 312 | 240 | +0.375 | 338 |
|  | SPMW W=16 | 200 | 16.0 | 100% | 4,800.00 | 167,847 | 195,716 | 486 | 64 | +0.079 | 307 |

- **SPMW and Spiral reach exactly N/w at every width; HP-FFT does not, and
  falls further behind as it widens**: 91.1%, 85.9%, 76.2%, 61.5%.
- **SPMW answers first at every width**: 1.35x to 2.37x sooner than HP-FFT,
  and before Spiral (645, 376, 260, 200 cycles against 722, 481, 367, 307).
- **SPMW holds the least block RAM**: 10, 20, 40, 64 against Spiral's 48, 84,
  144, 240 and HP-FFT's 82, 104, 166, 288. It has more registers than HP-FFT.
- **HP-FFT UF8 does not close at 300 MHz** (-0.093 ns). SPMW and Spiral close at
  every width; Spiral is smaller and clocks higher (its arithmetic differs).
- **The HP-FFT rows at 8 and 16 are the fixed builds**
  ([`hpfft/README.md`](hpfft/README.md)), not the shipped sources.

## Why HP-FFT trails

HP-FFT puts transform-sized arrays between its stages (`static
complex<float> data_1[FFT_NUM]` and on, inside `#pragma HLS dataflow`). Array
channels in a dataflow region are ping-pong buffers: a stage starts only when
its producer has finished the whole transform, so stages overlap across
transforms but not within one. That is the latency gap. The interval gap is a
refill: every stage process restarts from empty each transform, a near-constant
overhead on an ideal that halves with each step in width:

| config | ideal | measured | overhead |
|---|---:|---:|---:|
| UF1 | 128 | 140.5 | +12.5 |
| UF2 | 64 | 74.5 | +10.5 |
| UF4, fixed | 32 | 42.0 | +10.0 |
| UF8, fixed | 16 | 26.0 | +10.0 |

[`hpfft/README.md`](hpfft/README.md) traces it in csynth. Partitioning the
ping-pong arrays into banks so the unrolled butterflies can read them in
parallel makes each bank its own block RAM, which is HP-FFT's BRAM18 column.

SPMW's units are persistent II=1 processes connected by depth-8 FIFOs of one
complex sample each (`spmw_fifo #(.DW(64), .DEPTH(8))`, 32 of them in
`spmw/N256/generated/lanes_W2/fabric/spmw_top.sv`). A consumer starts on the
first token. The `W` lanes are `W` separate streams and a stage is `W/2` sites,
so a lane index is a placement coordinate and the wiring between lanes is a
constant. Where a butterfly's partner is in the same lane some beats back, a
delay-switch-delay permutation in front of the stage turns the beat distance
into a lane distance; a final reorder buffers one transform for natural output
order. The only block RAM is the twiddle tables; the delays are registers.

## What each row has here

| rows | in this folder | `results.csv` only |
|---|---|---|
| HP-FFT UF1 | source, csynth, cosimulation events, stimulus and testbench, routed reports | the numpy check (no `validate_rtl_vcd.json`) |
| HP-FFT UF2 | nothing | every value |
| HP-FFT UF4, UF8 fixed | `FFT.cpp`, csynth, cosimulation events, numpy check | routed resources and slack |
| SPMW W=2 | the design, its generated HLS C++ and fabric, per-unit csynth | cosimulation cycles, routed resources and slack |
| SPMW W=4, 8, 16 | the design (one source for every width) | every value |
| Spiral W=2..16 | simulation, numpy check, testbench, routed reports | nothing |

The paper's R11 note records the wide-lane SPMW reports as absent upstream.

## Layout

    results.csv
    spmw/N256/source/lanes.py                 the design, every width
    spmw/N256/generated/lanes_W2/roles/       HLS C++ per unit, W=2
    spmw/N256/generated/lanes_W2/fabric/      spmw_top and the FIFOs
    spmw/N256/generated/lanes_W2/reports/     csynth per unit (II=1: `yes(flp)`)
    hpfft/N256/source/UF1/                    FFT.cpp, FFT.h, project.tcl as shipped; common.tcl
    hpfft/N256/source/fix_UF{4_a8,8_b4}/      the fixed FFT.cpp
    hpfft/N256/generated/README.md            a note: HP-FFT has no generation step
    hpfft/N256/report/UF1/                    csynth, cosimulation, routed reports
    hpfft/N256/report/fix_UF{4_a8,8_b4}/      csynth, cosimulation events, numpy check
    spiral/N256/report/W{2,4,8,16}/           SwizzleFFT's run of each Spiral core
    scripts/hpfft/                            HP-FFT's flow, also used at 1,024 points
    scripts/collect_e2.sh                     staged HP-FFT UF1's files from the builds

## Scripts

| script | does |
|---|---|
| `scripts/hpfft/mkvariant.py` | derives a fixed `FFT.cpp` by named edits ([`hpfft/README.md`](hpfft/README.md)) |
| `scripts/hpfft/common.tcl` | HP-FFT's own `common.tcl` with solution name, part and period fixed and `export_design` removed |
| `scripts/hpfft/run_hls.sh <size> <cfg>` | csim and csynth |
| `scripts/hpfft/testbench_check.cpp` | csim gate: 3 transforms against a double-precision DFT |
| `scripts/hpfft/gen_stimulus.py`, `testbench_e2s.cpp` | 33 transforms of seeded numpy stimulus, and the testbench that streams them |
| `scripts/hpfft/run_cosim5.sh <size> <cfg>` | RTL cosimulation with a port-level VCD; runs the next two |
| `scripts/hpfft/cosim_events.py` | `cosim_events.json`: per-transform cycles from the VCD |
| `scripts/hpfft/validate_fft2.py` | `validate_rtl_vcd.json`: RTL outputs against `numpy.fft.fft` |
| `scripts/hpfft/run_pnr.sh <size> <cfg>`, `pnr_ooc.sh` | out-of-context route at 3.333 ns, float IP included |
| `scripts/hpfft/harvest_hls.py` | `harvest.json`: csynth metrics and warnings |

The shell scripts name their build roots under `/scratch/hc676` on
brg-zhang-xcel.

## Reproducing

    # SPMW, from the repository root
    python3 scripts/spmw_build_array.py --design fftlanes --size 256 --lanes <W> \
        --frequency 300 --cosim --pnr --out <dir>

    # HP-FFT, from this folder: the scripts build in $E/<size>/<cfg>
    # (E=/scratch/hc676/e2_hpfft), with scripts/hpfft/* copied to $E
    python3 scripts/hpfft/mkvariant.py HP-FFT-HLS/n256/UF4 $E/n256/UF4_a8 static fnpipe
    python3 scripts/hpfft/mkvariant.py HP-FFT-HLS/n256/UF8 $E/n256/UF8_b4 "repart:data_2:cyclic factor=UF*4"
    bash scripts/hpfft/run_hls.sh n256 UF4_a8       # csim + csynth
    bash scripts/hpfft/run_cosim5.sh n256 <copy>    # on a fresh copy of the config, per its usage line
    bash scripts/hpfft/run_pnr.sh n256 UF4_a8       # route

Spiral's reports are copied by `../n1024/scripts/collect_spiral.sh`.
