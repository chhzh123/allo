# SPMW rolled synthesis-time-win — measurement report

This report is the M2 measurement artifact for the plan's synthesis-time-win claim (AC-4): the rolled
`spmw.map` lets HLS synthesize **O(#roles)** function bodies, not **O(P0·P1)**. It records the
role-body-count metric across grid sizes and the real Vitis csim/csynth evidence for the flagship
systolic twin.

## The metric — role-body count is constant as the grid scales

Emitted-HLS body count (from `spmw.emit_hls` / `spmw.role_body_count`, a 2-D nearest-neighbor mesh):

| grid    | distinct role bodies | per-PID cloned bodies |
|---------|----------------------|-----------------------|
| 8 × 8   | 9                    | 0                     |
| 16 × 16 | 9                    | 0                     |
| 32 × 32 | 9                    | 0                     |

The nine bodies are the link-presence classes (interior, four edges, four corners). The named MLIR
pass `spmw-role-partition` computes the same classification on the rolled IR and attaches it as
`spmw.link_classes` (a `DenseI64ArrayAttr` whose length is the class count and whose entries are the
per-class instance counts). At 4×4 it is `[4,2,1,1,2,1,2,1,2]` (9 classes, sum 16); at 8×8 it is
`[36,6,1,1,6,1,6,1,6]` (9 classes, sum 64). The class count stays 9; only the interior/edge instance
counts grow. Tests: `test_role_body_count_is_nine_for_any_mesh_size`,
`test_link_presence_classes_are_nine_for_any_mesh`.

## Real Vitis 2023.2 / u280 evidence (flagship systolic twin, IR-driven rolled emitter)

The rolled top emitted by `emit_rolled_hls_ir` (which consumes the rolled `spmw.map` + its
`spmw.partition` / `spmw.channel_families`) was run through the real toolchain:

**L2 — csim (correctness):** the rolled O(#roles) top csim-**MATCHES** `A @ B`
(`test_rolled_top_csim_matches_reference`, `target="rolled"`, IR-driven).

**L3 — csynth (synthesis), rolled top at two grid sizes:** both grids csynth (`CSYNTH_OK`). The
**distinct synthesized role bodies stay 4** (`pe_interior`, `load_a`, `load_b`, `drain`); only the
instance counts scale, exactly as the emitted `top` loops instantiate them (`load_a` once per row
`M`, `load_b` once per column `N`, `pe_interior` once per grid point `M·N`, `drain` `M+N`):

| grid  | distinct role bodies | load_a | load_b | pe_interior | drain |
|-------|----------------------|--------|--------|-------------|-------|
| 4 × 4 | 4                    | 4      | 4      | 16          | 8     |
| 8 × 8 | 4                    | 8      | 8      | 64          | 16    |

`pe_interior` is scheduled **once** (one body) and uses **5 DSP at both grid sizes** (its FF/LUT grow
only with the contraction depth K 4→8, not with grid replication). So the **body count is constant
(4)** as the grid grows, while instances scale O(P) for the boundary roles and O(P²) copies of the
single interior body — i.e. "O(#roles) function bodies, not O(P0·P1)". (A wall-clock csynth-time trend
across 8/16/32 is future work; the body-count and per-role-resource invariance are the load-bearing
metrics.)

## Reproduce

- Body-count metric (no toolchain): `pytest tests/dataflow/spmw/test_spmw_hls.py
  tests/dataflow/spmw/test_spmw_role_partition_pass.py`.
- Vitis csim (toolchain-guarded): `pytest tests/dataflow/spmw/test_spmw_csim.py` after sourcing the
  Vitis env; `test_rolled_top_csim_matches_reference` exercises the IR-driven rolled path.
- csynth scaling: `spmw.build(twin, target="rolled", project=dir)` emits `kernel.cpp` + `run.tcl`;
  `vitis_hls -f run.tcl` produces the per-module report under `rolled.prj/solution1/syn/report/`.
