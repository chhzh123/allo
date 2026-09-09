# E4: FEATHER, as published RTL and as an SPMW port

Same layout as E1 to E3: `<framework>/S<n>/{source,generated,report}`.
`feather_rtl/` is the published Verilog in its two controller variants
(shipped and corrected); `spmw/` is the port. Report files are prefixed by
variant and mode, because one array size carries several of both.

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
