# E4: FEATHER, as published RTL and as an SPMW port

Same layout as E1 to E3: `<framework>/S<n>/{source,generated,report}`.
`feather_rtl/` is the published Verilog in its two controller variants
(shipped and corrected); `spmw/` is the port. Report files are prefixed by
variant and mode, because one array size carries several of both.

## The compute is identical; the difference is a one-off weight load

**`cycles_per_tile` is the same on both sides at every size: 4.0, 8.0, 16.0.**
Once loaded, the two arrays compute at exactly the same rate, which is what one
would expect of the same architecture expressed twice.

Everything that separates them is the initial weight load, and it accounts for
the end-to-end gap almost exactly:

| Array | SPMW total | FEATHER total | difference | `N^3` | RTL slower by |
|---|---:|---:|---:|---:|---:|
| 4x4 | 131,118 | 131,149 | 31 | 64 | 0.0% |
| 8x8 | 32,856 | 33,299 | 443 | 512 | 1.3% |
| 16x16 | 8,340 | 12,317 | 3,977 | 4,096 | 47.7% |

FEATHER's first output lands at `N^3` plus a small constant -- 64+17, 512+27,
4096+45 -- because its controller admits one processing element per cycle
(`r_pe_sel <= r_pe_sel + 1`, free-running) and the array holds `N^2` elements
with an `N`-deep weight file each.

**This is a controller limitation, not an architectural one, and it should not
be read as a structural advantage for SPMW.** The weight port is already `N`
bytes wide -- `WEIGHTS_DATA_WIDTH = 8*DPE_COL_NUM` in `feather_top.v` -- and
the loader uses one byte of it per cycle. A controller that drove the full
width would load a column at a time, `N^2` cycles rather than `N^3`, an `N`-fold
reduction: 4,096 to 256 at 16x16, which is well below SPMW's own 164-cycle
first output on the same row.

So the fair statement is:

- **Compute rate: a tie.** Identical cycles per tile at every size.
- **End-to-end: SPMW ahead by the load**, which is 0.0%, 1.3% and 47.7% as the
  array grows and the tile count falls -- it is a fixed cost amortised over
  32,768, 4,096 and 512 tiles.
- **The load itself is not a property of the architecture.** It is what the
  published controller does with a port that could carry `N` times more.

An earlier version of this section reported the first-output ratios -- 1.6x,
5.6x, 25.2x -- as SPMW's latency advantage. That overstated it: it compared a
startup phase that FEATHER's own datapath could shorten by `N`, and it ignored
that the compute rates are equal.

## The shipped RTL is broken, and every number here uses the corrected one

FEATHER's published controller **fails its own RTL simulation** at 4x4, 8x8 and
16x16 -- the three `rtl_orig_*` `fail` rows in `results.csv`. The fault is in
the PEs' ping/pong select: it toggles once per weight per visit, and the
shipped toggle puts a PE's even-index weights in the wrong half, so the array
computes against stale weights. `scripts/harness/apply_patch.py` carries the
one-line correction and the reasoning.

Every `rtl_fixed_*` row -- which is every FEATHER number quoted anywhere in
this experiment -- uses the corrected controller. Both variants are placed and
routed, and their area and timing are within noise of each other, so the
correction costs nothing and is not a thumb on the scale. The broken rows are
kept rather than deleted, because a baseline that does not pass its own
testbench is a finding about the baseline.

## The comparable pair, and the thing that has to be read with it

Two SPMW variants were built. Only **`feather` (single tile, resident
weights)** is a port of what the RTL does; `feather_stream` streams every
operand instead and is a different design, three to ten times larger, failing
timing at 16x16. The table below is the comparable pair.

| Array | System | LUT | FF | DSP | Slack |
|---|---|---:|---:|---:|---:|
| 4x4 | SPMW port | 2,305 | 3,639 | **16** | +1.200 ns |
| | FEATHER RTL, corrected | 2,309 | 3,378 | **0** | +1.174 ns |
| 8x8 | SPMW port | 9,974 | 14,989 | **64** | +0.866 ns |
| | FEATHER RTL, corrected | 9,694 | 15,332 | **0** | +0.355 ns |
| 16x16 | SPMW port | 31,019 | 49,673 | **256** | +0.338 ns |
| | FEATHER RTL, corrected | 57,499 | 91,251 | **0** | +0.120 ns |
| 32x32 | SPMW port | not run | | | |
| | FEATHER RTL, corrected | 338,694 | 624,270 | **0** | **-0.227 ns** |

**The DSP column is the caveat.** FEATHER's RTL routes with **zero DSPs** at
every size; the SPMW port uses one per element -- 16, 64, 256. So the LUT
comparison is not like for like: at 16x16 the port's 26,000 fewer lookup tables
and 41,000 fewer registers are bought with 256 DSP blocks the RTL does not use.
Vivado is inferring DSPs from the port's multiply and implementing FEATHER's in
fabric. Anyone reading the LUT column as an area win should read the DSP column
in the same row; the honest statement is that the two designs land in different
parts of the device, not that one is half the size of the other.

At 4x4 and 8x8 the lookup-table counts are within 3 per cent of each other,
which is the more informative result: at those sizes the port is neither
cheaper nor dearer, it just moves the multipliers.

**Both 32x32 builds fail timing** (-0.227 ns shipped, -0.206 ns corrected), and
the SPMW port was never run at that size, so 32x32 supports no comparison at
all.

## Two controller variants

`results.csv` carries the published controller and a corrected one. The shipped
controller **fails its own RTL simulation** at 4x4, 8x8 and 16x16 (three `fail`
rows); the corrected one passes at every size. Both are kept and both are
placed and routed, because the area and timing numbers are within noise of each
other and the correction is a control-path fix, not an architectural change.

## Files

- `source/` -- the SPMW port (`spmw_feather.py`) on one side; the RTL driver
  and its testbench (`spmw_feather_rtl.py`, `tb_feather_rtl.sv`) on the other.
- `generated/` -- what the split backend emitted for the port, one `.cpp` and
  one `.sv` per role, re-staged from the same `scripts/spmw_build_array.py`
  entry point the measured builds used (9 roles at 4x4, 11 at 8x8 and 16x16).
  `feather_rtl/*/generated/` holds a note instead: hand-written RTL has no
  generation step.
- `report/` -- routed utilisation, hierarchy, timing and route status per
  variant, and the simulation results per workload and mode.
- `scripts/` -- what drove the runs, split by which agent produced them.

Left behind: each RTL simulation's `bus.log` is 2.8 MB of bus trace and
`tiles.log.gz` its tile dump; nothing in the table reads them. The full tree is
33 MB on the machine.
