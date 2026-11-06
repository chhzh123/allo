# Allo's routed numbers: the flow, and what they say

The Allo rows of `README.md`'s tables, Fig. 9(b) and Table 1.

## The flow

Allo's design is routed the way the SPMW kernel and AutoSA are: Vivado
directly, out of context, on the RTL Vitis HLS synthesised --
`synth_design -mode out_of_context`, `opt_design`, `place_design`,
`phys_opt_design`, `route_design` at 3.333 ns (`scripts/allo_ooc.sh`; the
script each run used is `allo/S<D>/report/ooc_pnr.tcl`, its reports
`ooc_util.rpt`, `ooc_timing.rpt` and, at 16x16, `ooc_util_hier.rpt`).
`scripts/pnr_matched.sh` routes AutoSA by the same recipe.

Vitis's own route, `export_design -flow impl` (`scripts/allo_pnr.sh`), is
not used. It regenerates the RTL testbench before handing off to Vivado, and
at 16x16 and 32x32 that step spins without writing anything; it completes at
4x4 and 8x8 only (`allo/S4/report/export_impl.rpt`, `allo/S8/report/export_impl.rpt`).

## Does the flow change the numbers? Only the clock

At the two sizes routed both ways:

| Allo | Vitis `export -flow impl` | Vivado out of context |
|---|---:|---:|
| **4x4** LUT | 3,693 | 3,703 |
| FF | 4,459 | 4,459 |
| DSP | 16 | 16 |
| BRAM | 3 x 18K | 1.5 x 36K |
| WNS | +0.625 ns | +0.720 ns |
| Clock | 369 MHz | 383 MHz |
| **8x8** LUT | 8,419 | 8,425 |
| FF | 8,220 | 8,220 |
| DSP | 64 | 64 |
| BRAM | 3 x 18K | 1.5 x 36K |
| WNS | +0.598 ns | +0.719 ns |
| Clock | 366 MHz | 383 MHz |

The logic is the same design either way: 10 LUTs apart at 4x4 and 6 at 8x8,
registers identical. The block RAM is the same memory in different units,
below. The clock moves about 4%, which is the two flows' directives, not a
different circuit.

## Block RAM units

**Vitis reports BRAM in 18K blocks; Vivado reports Block RAM Tiles, which are
36K.** Allo 4x4 is `BRAM: 3` from Vitis and `Block RAM Tile 1.5` from Vivado,
the same memory. Every BRAM18 figure in this folder is twice the tiles of
Vivado's `report_utilization`: Allo's `ooc_util.rpt` reads `Block RAM Tile
1.5` and `RAMB18E2 3` at every size, so its BRAM18 is 3.

## The Allo column

| Array | Cycles | LUT | of which LUTRAM | FF | DSP | BRAM18 | WNS | Clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 118 | 3,703 | 574 | 4,459 | 16 | 3 | +0.720 ns | 383 MHz |
| 8x8 | 288 | 8,425 | 1,590 | 8,220 | 64 | 3 | +0.719 ns | 383 MHz |
| 16x16 | 928 | 32,480 | 5,450 | 28,780 | 256 | 3 | +0.343 ns | 334 MHz |
| 32x32 | 3,408 | 151,663 | 36,858 | 109,251 | 1,024 | 3 | **-0.657 ns** | **251 MHz** |

No URAM at any size.

## Why the block RAM is flat

Block RAM stays at 1.5 tiles from 4x4 to 32x32 while the array grows
sixty-four-fold, because **Allo's storage is in LUTs**, and that scales:

| Array | LUT as Logic | LUT as Memory | of which shift registers | of which distributed RAM | Block RAM tiles |
|---|---:|---:|---:|---:|---:|
| 4x4 | 3,129 | 574 | 474 | 100 | 1.5 |
| 8x8 | 6,835 | 1,590 | 1,386 | 204 | 1.5 |
| 16x16 | 27,030 | 5,450 | 5,034 | 416 | 1.5 |
| 32x32 | 114,805 | 36,858 | 36,010 | 848 | 1.5 |

The distributed RAM is the operand and result buffers `buf0`, `buf1` and
`buf2`; the shift registers are mostly the A and B FIFOs between PEs (4,096
of the 5,034 at 16x16, `allo/S16/report/ooc_util_hier.rpt`). At 32x32, 24% of
Allo's LUTs are memory rather than logic, so its LUT column is not all logic.
The three RAMB18 are the data buffers of its three 32-bit AXI masters, one
each, which do not scale with the array.

## 16x16, kernel scope

| System | Cycles | LUT | FF | DSP | Clock |
|---|---:|---:|---:|---:|---:|
| SPMW | **133** | 38,507 | 53,682 | 256 | **353 MHz** |
| Allo | 928 | **32,480** | **28,780** | 256 | 334 MHz |
| AutoSA, wide | 374 | 48,957 | 84,062 | 256 | 339 MHz |

**Allo is the smallest design at this size.** It uses 16% fewer LUTs and 46%
fewer registers than the SPMW kernel; SPMW's area advantage is over AutoSA,
21% fewer LUTs and 36% fewer registers. What SPMW buys with its extra area is
latency: the same 16-cubed GEMM in 133 cycles against Allo's 928, 6.98 times
fewer. Part of both gaps is the memory side: SPMW's kernel has three 512-bit
movers, Allo 32-bit ports and load, compute and store one after another.

## 32x32 misses timing

Allo routes at 32x32 but does not close: **WNS -0.657 ns, TNS -688.463 ns
over 4,244 failing endpoints** of 465,809; hold is clean (+0.019 ns). The
clock is 251 MHz against the 300 MHz target. The worst path (`ooc_timing.rpt`)
runs from the result-store loop's start register over a net of 1,077 loads
into a LUTRAM bank of the result buffer `buf2`: 3.890 ns, 90% of it routing.

| System | Cycles | LUT | FF | WNS | Clock |
|---|---:|---:|---:|---:|---:|
| SPMW | **280** | **147,551** | 193,953 | **+0.302 ns** | **330 MHz** |
| Allo | 3,408 | 151,663 | **109,251** | -0.657 ns | 251 MHz |
| AutoSA, wide | 1,222 | 186,369 | 320,772 | +0.101 ns | 309 MHz |

At 32x32 Allo is 2.8% larger than SPMW in LUTs, still 44% smaller in
registers, and the only one of the three that misses 300 MHz.
