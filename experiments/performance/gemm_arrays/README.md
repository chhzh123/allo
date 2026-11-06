# GEMM on D x D arrays, 4x4 to 32x32

Fig. 9(b) and Table 1 (GEMM rows) of the paper (arXiv version).

Every design computes `C = A B` with int8 A and B and int32 C, `M = N = K = D`,
on a D x D output-stationary array with one multiplier, one DSP, a cell, in
one launch. Three systems: SPMW, AutoSA's generated design and Allo's library
systolic array. Every row is **kernel scope**, memory port to memory port:
each design has its own AXI masters and control interface; the host's layout
of the operands, host communication and control traffic are outside it. All
are built with Vitis HLS and Vivado 2023.2 for `xcu280-fsvh2892-2L-e` and
routed out of context at 3.333 ns.

## Layout

    <system>/S<D>/source/      what the framework was given
    <system>/S<D>/generated/   what it emitted
    <system>/S<D>/report/      the reports the numbers are read from

| Folder | Design |
|---|---|
| `spmw_kernel/S<D>/` | SPMW's array (`source/test_spmw_autosa_match.py`, registry name `autosa` in the repository's `scripts/spmw_build_array.py`) with a 512-bit memory mover on each of its three edge bindings and a control register file. `report/full_kernel/` is that kernel simulated and routed whole |
| `autosa/S<D>/` | AutoSA as generated, narrow ports |
| `autosa/S8_wide/` | AutoSA with `--host-serialize`, 512-bit ports, 8x8 |
| `allo/S<D>/` | Allo's library systolic array (`source/allo_library_systolic.py`), wrapped and emitted by `source/e1_allo_gen.py`; `allo/S16/source/e1_allo_gen.py` also generates `../gemm_workloads/allo/` |
| `results.csv` | one row a design and size (15 rows) |
| `ALLO_PNR.md` | how Allo was routed, and its resources and timing at every size |

## Fig. 9(b): latency

Cycles from the first memory beat to the last (definitions below). AutoSA is
its lower-latency build at each size: narrow at 4x4, wide at 8x8 and up.

| Array | SPMW | AutoSA | Allo | AutoSA / SPMW | Allo / SPMW |
|---|---:|---:|---:|---:|---:|
| 4x4 | **47** | 78 (narrow) | 118 | 1.66x | 2.51x |
| 8x8 | **82** | 130 (wide) | 288 | 1.59x | 3.51x |
| 16x16 | **133** | 374 (wide) | 928 | 2.81x | 6.98x |
| 32x32 | **280** | 1,222 (wide) | 3,408 | 4.36x | 12.17x |

## Table 1: 32x32

Throughput is `2 D^3 f / (1000 L)` GOP/s at `f = min(300, f_routed)` MHz.

| Design | Latency (cycles) | Throughput (GOP/s) | LUT | FF | DSP | BRAM18 | Freq. (MHz) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPMW | **280** | **70.22** | 147,551 | 193,953 | 1,024 | 46 | 330 |
| AutoSA, wide | 1,222 | 16.09 | 186,369 | 320,772 | 1,024 | 48 | 309 |
| Allo | 3,408 | 4.82 | 151,663 | 109,251 | 1,024 | 3 | 251 |

SPMW's row is `spmw_kernel/S32/report/full_kernel/` (cycles in `cycles.txt`,
resources in `util.rpt`, slack +0.302 ns in `timing.rpt`). AutoSA's and Allo's
are `results.csv`; Allo's reports are `allo/S32/report/ooc_*`. Allo misses
timing at WNS -0.657 ns: its worst path starts at the result-store loop's
start register and fans out over a 1,077-load net into a LUTRAM result buffer
(`buf2`), 3.890 ns with 90% in routing.

## Every row

| Array | Design | Port in/out | Cycles | Flow reports | LUT | FF | DSP | BRAM18 | Slack | Clock |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | SPMW | 512/512 | 47 | 66 | 6,548 | 11,104 | 16 | 46 | +0.313 ns | 331 MHz |
| | AutoSA | 32/32 | 78 | 148 | 4,974 | 8,171 | 16 | 3 | +0.512 ns | 354 MHz |
| | Allo | 32/32 | 118 | 167 | 3,703 | 4,459 | 16 | 3 | +0.720 ns | 383 MHz |
| 8x8 | SPMW | 512/512 | 82 | 100 | 12,833 | 19,944 | 64 | 46 | +0.475 ns | 350 MHz |
| | AutoSA | 64/32 | 222 | 292 | 13,393 | 22,924 | 64 | 5 | +0.690 ns | 378 MHz |
| | AutoSA, wide | 512/512 | 130 | 202 | 15,343 | 25,536 | 64 | 46 | +0.738 ns | 385 MHz |
| | Allo | 32/32 | 288 | 337 | 8,425 | 8,220 | 64 | 3 | +0.719 ns | 383 MHz |
| 16x16 | SPMW | 512/512 | 133 | 152 | 38,507 | 53,682 | 256 | 46 | +0.500 ns | 353 MHz |
| | AutoSA | 128/32 | 782 | 859 | 47,390 | 81,265 | 256 | 18 | +0.410 ns | 342 MHz |
| | AutoSA, wide | 512/512 | 374 | 445 | 48,957 | 84,062 | 256 | 46 | +0.380 ns | 339 MHz |
| | Allo | 32/32 | 928 | 977 | 32,480 | 28,780 | 256 | 3 | +0.343 ns | 334 MHz |
| 32x32 | SPMW | 512/512 | 280 | 298 | 147,551 | 193,953 | 1,024 | 46 | +0.302 ns | 330 MHz |
| | AutoSA | 256/32 | 2,990 | 3,064 | 185,634 | 317,967 | 1,024 | 19 | +0.111 ns | 310 MHz |
| | AutoSA, wide | 512/512 | 1,222 | 1,300 | 186,369 | 320,772 | 1,024 | 48 | +0.101 ns | 309 MHz |
| | Allo | 32/32 | 3,408 | 3,457 | 151,663 | 109,251 | 1,024 | 3 | -0.657 ns | 251 MHz |

Port widths are each design's synthesised `C_M_AXI_*_DATA_WIDTH`, widest read
master and the write master. SPMW's kernel routes report no routing errors
(`full_kernel/route_status.rpt`).

**SPMW's resources here are the kernel's, not `results.csv`'s.** The
`spmw_kernel` rows of `results.csv` carry the kernel's cycles but the route of
the array and its distribution chains alone, without the movers and control
(`spmw_kernel/S<D>/report/util.rpt`, `timing.rpt`; `cost.json` lists 15 roles
and no mover): 1,867 to 137,741 LUTs, no block RAM, 357 to 492 MHz. That
route's `cost.json` is also where Fig. 13(a)'s SPMW build times come from
(`../../compilation/compile_time/build_time/`). Routed whole, the memory side
adds 4,681, 4,818, 5,751 and 9,810 LUTs and 8,588, 9,724, 12,950 and 30,245
FFs from 4x4 to 32x32, and 46 BRAM18 at every size, all of it in the three
movers (`full_kernel/util_hier.rpt`: 8, 7 and 7 RAMB36 and two RAMB18). Wide
AutoSA's 46 to 48 are its 512-bit masters too.

**AutoSA's narrow BRAM18 at 4x4 and 8x8** are read from `report/util.rpt`:
1.5 and 2.5 tiles, 3 and 5 BRAM18. `results.csv` holds 6 and 10 for them,
doubled once more than the reports say. The paper does not use these cells.

## Definitions

- **Cycles** are first memory beat to last memory beat of one launch, any
  master, counted on rising clock edges where VALID and READY are both high.
  The paper calls it latency from the first input-memory transfer through the
  final result write. SPMW's kernel testbench reports it
  (`full_kernel/cycles.txt`, `SPMW TB DATA`); a baseline's comes from a
  waveform of the same cosimulation snapshot (`scripts/beats.py`). The
  baselines' beat counts are in `results.csv`; the waveforms were not kept.
- **Flow reports** is each flow's own start-to-done figure: SPMW's testbench
  start to its done poll at a 4 ns simulation clock, the baselines' Vitis
  cosimulation report (`kernel0_cosim.rpt`, Allo's `gemm_cosim_1launch.rpt`).
  The difference is control traffic. The paper uses the first column.
- **Clock** is `1000 / (3.333 - WNS)` MHz, from the out-of-context route.
- **BRAM18** is twice the Block RAM tiles of `report_utilization`.

## The designs

- **SPMW.** A and B each enter, one packed vector a step, at the head of a
  chain of feed units, each of which keeps its row's or column's lane; each
  cell's result leaves down its column, its own first and then those from
  above, to a third mover. The kernel bench memory returns a read beat every other cycle within a burst
  (the repository's `scripts/spmw_kernel_sim.py`, default `--beat 2`), where the baselines'
  cosimulation memories return one a cycle.
- **AutoSA**, `space_time[3]`, `array_part[D,D,D]`, `latency[1,1]`, `simd[1]`
  (`scripts/gen_matched.sh`; `source/kernel.h` sets `I = J = K = D`). As
  generated its read ports are D bytes wide and its write port 32 bits.
  `--host-serialize` (`scripts/gen_wide.sh`) lays the operands out in the
  order the array reads them and widens all three to 512 bits. 4x4 cannot be
  widened: its 16-byte operand is less than one beat, and AutoSA's serializer
  fails on it, so 4x4 stays narrow.
- **Allo.** The library unfolds the PE grid and streams A and B between PEs,
  but the top function calls `load_buf0`, `load_buf1`, `systolic_gemm` and
  `store_res2` one after another with no dataflow region around them
  (`generated/kernel.cpp`): all of A, then all of B, then the compute, then
  all of C, through 32-bit ports. Cycles are from a one-seed launch
  (`source/tb_1launch.cpp`, `report/gemm_cosim_1launch.rpt`); the three-seed
  run (`report/gemm_cosim.rpt`) is the correctness check.

AutoSA's and Allo's reduction loop is 8 steps at II = 1, 11 cycles at 8x8
(`autosa/S8/report/PE_csynth.rpt`, `allo/S8/report/PE_kernel_gemm_0_0_csynth.rpt`);
the paper states the same for SPMW, whose cell report is not kept here.

## Not in this folder

- **AutoSA's wide 16x16 and 32x32 builds**, the AutoSA points of Fig. 9(b) at
  those sizes and Table 1's AutoSA row. Only their `results.csv` rows are
  here; the builds are on brg-zhang-xcel in `/scratch/hc676/e1_autosa_wide`.
- **Some AutoSA reports**: `autosa/S8_wide` and `autosa/S16` have HLS and
  cosimulation reports but no routed ones, and `autosa/S32` has its source and
  generated code only. Their resources are `results.csv`'s.

The tree before this folder was cut to what the paper uses is commit
`c20b2ee8` (`experiments/e1_gemm/`).

## Reproducing

On brg-zhang-xcel; the drivers name their working directories under
`/scratch/hc676`.

    # SPMW: package, simulate one launch, route the kernel whole (report/full_kernel/)
    scripts/spmw_ws_kernel.sh <D> 4 autosa e1kernel
    # AutoSA, narrow
    scripts/gen_matched.sh          # generate D = 4, 8, 16, 32
    scripts/hls_matched.sh <D>      # csynth and cosimulation
    scripts/vcd_matched.sh <D>      # waveform, first and last beat (beats.py)
    scripts/pnr_matched.sh <D>      # out-of-context route at 3.333 ns
    # AutoSA, wide (D = 8, 16, 32; 4 fails in generation)
    scripts/gen_wide.sh
    scripts/hls_wide.sh <D>
    # Allo
    scripts/allo_retry.sh <D>       # allo_beats.sh (one-seed cosim) under a hang watchdog, then vcd_allo.sh
    scripts/allo_ooc.sh <D>         # out-of-context route
    scripts/allo_pnr.sh <D>         # Vitis export -flow impl; completes at 4x4 and 8x8 only (ALLO_PNR.md)
    scripts/collect_e1.sh           # stages source/, generated/, report/ from the builds

`vcd_matched.sh` and `pnr_matched.sh` name the narrow builds' directory
(`/scratch/hc676/e1_autosa_matched`); the wide builds were measured by the
same recipe on theirs, and no separate driver for that is kept. `collect_e1.sh`
does not stage `report/full_kernel/`, `autosa/S8_wide/`, Allo's `ooc_*` or
one-launch reports.
