# Allo's routed numbers: the flow, and what 16x16 says

## Why this flow, and not the one the first two sizes used

Allo's 4x4 and 8x8 rows were routed by Vitis, `export_design -flow impl`, and
16x16 and 32x32 were never routed at all. Running that same stage at those
sizes hangs: `export_design` regenerates the RTL testbench before handing off
to Vivado, and that step spins at 100% CPU writing nothing. It was allowed
three independent attempts and stalled every time, at 4 seconds, at 961
seconds and at 7.4 hours.

Checking how the other systems were routed made the fix obvious.
`scripts/pnr_matched.sh` routes AutoSA with **Vivado directly, out of
context** -- `synth_design -mode out_of_context`, `opt_design`,
`place_design`, `phys_opt_design`, `route_design` -- and its own comment says
"same recipe as the SPMW arrays". So SPMW and AutoSA already share one flow,
and Allo's two published rows were the only ones in E1 that did not.

`scripts/allo_ooc.sh` therefore routes Allo the same way, on the RTL Vitis had
already synthesised. Vitis never runs, so the hang cannot occur, and Allo
joins the recipe every other row uses. **16x16 routed in 1,194 seconds** after
the export path had failed for hours.

## 16x16, routed

| | Allo 16x16 |
|---|---:|
| CLB LUTs | 32,480 |
| CLB Registers | 28,780 |
| DSPs | 256 |
| Block RAM tiles | 1.5 |
| URAM | 0 |
| Worst negative slack | +0.343 ns |
| Implied clock | 334 MHz |
| Unrouted nets | 0 |

`route_design completed successfully`; evidence in `allo/S16/report/ooc_*`
(utilisation, hierarchical utilisation, timing summary, the Vivado log and the
exact script). Slack is against the 3.333 ns target, so the implied period is
2.990 ns.

## What it says, including the part that does not favour SPMW

At 16x16, kernel scope, all three on the same out-of-context recipe:

| System | Cycles | LUT | FF | DSP | Clock |
|---|---:|---:|---:|---:|---:|
| SPMW kernel | **133** | 32,756 | 40,732 | 256 | **358 MHz** |
| Allo | 928 | **32,480** | **28,780** | 256 | 334 MHz |
| AutoSA | 782 | 47,390 | 81,265 | 256 | 342 MHz |

**Allo is the smallest design at this size, not SPMW.** It uses 0.8% fewer
lookup tables and 29% fewer registers than the SPMW kernel. Any claim that
SPMW's array is smaller than Allo's at 16x16 is not supported by these
numbers. SPMW's area advantage in E1 is over AutoSA -- 31% fewer lookup tables
and 50% fewer registers -- and Allo sits below both.

**Where SPMW wins here is throughput at equal area.** It computes the same
16-cubed problem in 133 cycles against Allo's 928, a factor of seven, for
essentially the same lookup-table count and 41% more registers. Much of that
gap is the memory boundary rather than the array: SPMW's kernel is fed through
512-bit ports and Allo's through 32-bit ports, which is also part of why Allo
is smaller. The honest one-line summary is that at 16x16 SPMW buys seven times
the throughput with more registers and a wider port, not with less logic.

## Does the flow change move the numbers? Barely, except the clock

4x4 and 8x8 have now been routed both ways, which answers it directly:

| Allo | Vitis `export -flow impl` | Vivado out of context |
|---|---:|---:|
| **4x4** LUT | 3,693 | 3,703 |
| FF | 4,459 | 4,459 |
| DSP | 16 | 16 |
| BRAM | 3 x 18K | 1.5 x 36K |
| WNS | +0.625 ns | +0.720 ns |
| Implied clock | 369 MHz | 383 MHz |
| **8x8** LUT | 8,419 | 8,425 |
| FF | 8,220 | 8,220 |
| DSP | 64 | 64 |
| BRAM | 3 x 18K | 1.5 x 36K |
| WNS | +0.598 ns | +0.719 ns |
| Implied clock | 366 MHz | 383 MHz |

Logic is the same design either way: ten lookup tables apart at 4x4 and six at
8x8, 0.3% and 0.07%, with the register count identical to the unit at both
sizes. The block RAM figures are the same
memory in different units, which is the trap below. What does move is the
clock, 369 to 383 MHz, about 4%, which is the two flows' synthesis and
implementation directives rather than a different circuit.

So the area numbers published for 4x4 and 8x8 were sound; only their clocks
were on a different basis from every other row in the table. Both sizes land
at the same 383 MHz under the common recipe, against 369 and 366 before.

## A units trap in the `bram_18k_equiv` column

**Vitis reports BRAM in 18K blocks; Vivado reports Block RAM Tiles, which are
36K.** Allo 4x4 is `BRAM: 3` from Vitis and `Block RAM Tile 1.5` from Vivado
-- the same memory, a factor of two apart.

The column is named `bram_18k_equiv`, but every row produced by the
out-of-context recipe -- SPMW, AutoSA, Gemmini and Allo -- stored Block RAM
Tiles, so those values were **half** what the column name said. AutoSA's 9.5
at 32x32 was the giveaway: 36K tiles come in halves, 18K blocks do not.

**Now corrected.** Every non-zero cell has been doubled, so the column holds
what its name claims. Verified against the reports rather than assumed: Allo
16x16's `ooc_util.rpt` reads `Block RAM Tile 1.5` and `RAMB18E2 3`, and both
`pnr_matched.sh` and `allo_ooc.sh` read the tile line. Allo is 3 RAMB18 at
every size, AutoSA 6/10/18/19 narrow and 46/46/48 wide, SPMW and Gemmini
zero throughout.

E2's table had the same trap in a worse form -- one unit in some rows and the
other in the rest -- and is corrected too. Anyone quoting a block RAM figure
from before these two commits is quoting half of one.

## The Allo column, complete

| Array | Cycles | LUT | of which LUTRAM | FF | DSP | BRAM tiles | WNS | Clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 118 | 3,703 | 574 | 4,459 | 16 | 1.5 | +0.720 ns | 383 MHz |
| 8x8 | 288 | 8,425 | 1,590 | 8,220 | 64 | 1.5 | +0.719 ns | 383 MHz |
| 16x16 | 928 | 32,480 | 5,450 | 28,780 | 256 | 1.5 | +0.343 ns | 334 MHz |
| 32x32 | 3,408 | 151,663 | 36,858 | 109,251 | 1,024 | 1.5 | **-0.657 ns** | **251 MHz** |

All four on the recipe AutoSA and the SPMW arrays use, all routing with
nothing unrouted.

## 32x32 misses timing, and not narrowly

Allo routes at 32x32 but does not close: **WNS -0.657 ns, TNS -688.5 ns over
4,244 failing endpoints** of 465,809. Hold is clean (+0.019 ns). The implied
clock is 251 MHz against the 300 MHz target. It is the second design in E1 to
miss at this size -- Gemmini's output-stationary mesh misses too, at -0.065 ns
-- and the only one to miss by more than a rounding margin.

At 32x32, kernel scope:

| System | Cycles | LUT | FF | WNS | Clock |
|---|---:|---:|---:|---:|---:|
| SPMW kernel | **280** | 137,741 | 163,708 | **+0.529 ns** | **357 MHz** |
| Allo | 3,408 | 151,663 | **109,251** | -0.657 ns | 251 MHz |
| AutoSA | 2,990 | 185,634 | 317,967 | +0.111 ns | 310 MHz |

**The two sizes tell opposite stories, and both belong in the paper.** At
16x16 Allo is the smallest design and beats the SPMW kernel on both lookup
tables and registers. At 32x32 it is larger than SPMW on lookup tables, still
much smaller on registers, and it stops meeting timing while SPMW keeps half a
nanosecond of margin. SPMW is the only design in E1 with comfortable slack at
32x32; AutoSA scrapes in at +0.111 and Allo and Gemmini-OS fail.

## Why the block RAM figure is flat, resolved

Block RAM stays at 1.5 tiles from 4x4 to 32x32 while the array grows
sixty-four-fold. That is not an error and not an artefact of the flow:
**Allo puts its tiles in distributed LUT RAM instead**, and that does scale.

| Array | LUT as Logic | LUT as Memory | Block RAM tiles |
|---|---:|---:|---:|
| 4x4 | 3,129 | 574 | 1.5 |
| 8x8 | 6,835 | 1,590 | 1.5 |
| 16x16 | 27,030 | 5,450 | 1.5 |
| 32x32 | 114,805 | 36,858 | 1.5 |

At 32x32, 24% of Allo's lookup tables are memory rather than logic. So its
lookup-table column is not comparable to SPMW's as "logic": part of it is
storage that SPMW places in registers and Gemmini would place elsewhere. The
block RAM left at 1.5 tiles is interface buffering that does not scale with
the mesh.
