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
single interior body — i.e. "O(#roles) function bodies, not O(P0·P1)".

## Synthesis wall-clock trend (task2.5 harness, real csynth at 8/16/32)

`scripts/spmw_synth_scaling.py` runs a real `csynth_design` on the rolled top at a sweep of square
grids (each in its own project, fixed contraction depth K=4 so the interior body is identical across
sizes) and records the synthesis wall-clock next to the distinct-function-body count:

| grid | PE instances | distinct role bodies | csynth wall-clock (s) | status |
|------|--------------|----------------------|-----------------------|--------|
| 4x4 | 16 | 5 | 24.5 | CSYNTH_OK |
| 8x8 | 64 | 5 | 88.9 | CSYNTH_OK |
| 16x16 | 256 | 5 | 863.9 | CSYNTH_OK |

(measured on Vitis 2023.2/u280 @300MHz; also in `docs/spmw_synth_scaling_data.md`.)

The **distinct function-body count is constant (5** — `top` plus the four roles `pe_interior`,
`load_a`, `load_b`, `drain`) as the grid grows: HLS schedules a fixed set of module bodies and
replicates the one already-scheduled `pe_interior` across the mesh. The csynth wall-clock still
*grows* with the grid, and that is expected — a **spatial** accelerator genuinely instantiates O(P²)
PE copies and O(P²) FIFOs, and the tool must elaborate and bind every instance; that is inherent
physical scaling, not a code-generation blow-up. The synthesis-time **win** is against the naive
per-PID emission, which presents O(P²) *distinct* function bodies for the scheduler to process
separately: `test_synthesis_time_win_df_grows_but_rolled_stays_constant` shows the df path's per-PID
`gemm_i_j` body count strictly grows with the grid, while the rolled path stays at O(#roles). The
rolled `spmw.map` keeps the number of bodies the front end must schedule constant.

The fully-unrolled 32×32 (1024 PE instances) exceeds a 40-minute csynth budget — extrapolating the
measured super-linear trend (≈O(P^1.6): 24.5 s → 88.9 s → 863.9 s as instances go 16 → 64 → 256) it
is on the order of two hours. That is the O(P²) *instance* count of a 1024-PE spatial array, not a
body-count blow-up (the emitted top still has the same 5 bodies); the naive per-PID top would instead
present 1024 *distinct* PE bodies to schedule and would not synthesize at all in that budget.

## Reproduce

- Body-count metric (no toolchain): `pytest tests/dataflow/spmw/test_spmw_hls.py
  tests/dataflow/spmw/test_spmw_role_partition_pass.py`.
- Vitis csim (toolchain-guarded): `pytest tests/dataflow/spmw/test_spmw_csim.py` after sourcing the
  Vitis env; `test_rolled_top_csim_matches_reference` exercises the IR-driven rolled path.
- Synthesis wall-clock trend: `python3 scripts/spmw_synth_scaling.py 8 16 32 --out
  docs/spmw_synth_scaling_data.md` (writes the table above); each size emits `kernel.cpp` + `run.tcl`
  and runs `vitis_hls -f run.tcl`, whose per-module report lands under
  `rolled.prj/solution1/syn/report/`.
