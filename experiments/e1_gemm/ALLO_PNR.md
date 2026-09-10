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

## Still open

- 4x4 and 8x8 are being re-routed on this same recipe so the Allo column is on
  one flow rather than two. Until they land, the published 3,693/4,459 and
  8,419/8,220 remain Vitis export numbers and are **not** directly comparable
  to the 16x16 row above.
- 32x32 is running.
