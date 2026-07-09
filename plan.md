# SPMW: A Single-Program Multiple-Work-Unit Frontend for Allo

## Goal Description

Build **SPMW**, a Python frontend (`allo.spmw`) plus a new `spmw` MLIR dialect, passes, and
backend machinery, that makes the three sources of spatial-accelerator regularity —
**replication**, **interconnect topology**, and **boundary specialization** — first-class, as an
evolution of `allo.dataflow`. Instead of hand-writing every PE and every FIFO with `meta_if` PID
chains and absolute grid arithmetic (`fifo_A[i, j+1].put(...)`), the user writes one work unit,
declares a topology, and declares boundary roles; the compiler synthesizes interconnect, halo,
and placement.

The work is organized as six milestones (M1..M6). The **committed / funded scope is M1 → M2 →
M3** (per the success-bar decision, DEC-2): the frontend, a *rolled* structural op that survives
to codegen, the HLS synthesis-time win, and the hierarchy + collectives + structural-RTL-glue
backend. **M4 (folding/banking + FFT + mini-TPU), M5 (fast simulator + perf model), and M6
(non-mesh/sparse/robustness) are stretch** goals pursued once M1–M3 land. Two independent claims
are proven separately so a failure in one does not sink the other: **(1) frontend concision** —
SPMW twins reproduce existing `dataflow` tests bit-identically on the simulator; **(2) the
synthesis-time win** — the rolled op lets HLS synthesize O(#roles) function bodies, not
O(P0·P1). Scope is the **simulator + Vitis HLS + structural-RTL FPGA path only**; the **AIE
backend (`tests/dataflow/aie/*`) is out of scope** and left untouched.

## Acceptance Criteria

Following TDD philosophy, each criterion carries positive tests (expected to PASS) and negative
tests (expected to FAIL when the feature works correctly).

- **AC-1: The `allo.spmw` frontend parses/builds and topologies are statically checkable.**
  Surface: `@spmw.unit`, `@spmw.region`, `spmw.map`, `spmw.build(target=)`, `spmw.Topology`/`mesh`,
  static-affine peer + key links, `ctx.rank/port`, `@pe.role`, `stream_in/out`.
  - Positive Tests: the `test_systolic` twin and `test_producer_consumer` twin build to IR; a
    mesh infers each port's element type (checked equal at both ends) and default `depth=2`;
    `depths={...}` overrides one port.
  - Negative Tests: an unmatched or asymmetric peer link (an `east` with no reciprocal `west`) is
    rejected at build; a `.put/.get` on an undeclared port errors; a topology whose declared grid
    rank ≠ its dimensionality is rejected.

- **AC-2: SPMW lowers to a rolled `spmw.map` op (in the new `spmw` dialect) that survives to
  codegen.** The op carries grid + topology (affine peer/key maps) + roles (symbol refs to
  per-role `func.func`) + optional fold/unroll, with a verifier. It round-trips through
  `allo-opt` and is consumed by passes **without per-PE expansion** on the structural path.
  `spmw.shard` lowers to the existing `allo.grid_map` sharding op.
  - Positive Tests (FileCheck-style): after frontend lowering the module contains exactly one
    `spmw.map` op and one `func.func` per role — **not** `P0*P1` per-PID funcs; `allo-opt`
    parses/verifies the mesh IR; after `spmw-unroll` the simulator IR contains per-PID
    `func.call`s equivalent to today's `df` lowering.
  - Negative Tests: the verifier rejects grid-rank ≠ topology-dims; rejects a channel key with
    two `src` or zero `sink` ports (fan-out only where a collective declares it).
  - AC-2.1: The new `spmw` dialect is registered end-to-end.
    - Positive: `spmw` ops are usable from the C++ CAPI, the Python bindings, and the HLS
      translation registry; `pip install -e .` rebuilds the extension cleanly.
    - Negative: a build that omits any of the three registration sites (CAPI, Python, CMake
      link) fails to load the dialect rather than silently ignoring `spmw` ops.

- **AC-3: Simulator equivalence (L1) — SPMW twins are bit-identical to their `df` originals.**
  The M1 migration set runs to the same numbers under `target="simulator"`; the original is kept
  as oracle until the twin passes.
  - Positive Tests: `test_producer_consumer`, `test_df_unit`, `test_stream_of_blocks`,
    `test_1D_systolic`, `test_cooperative_gemv` twins match their originals exactly.
  - Negative Tests (mutation/diagnostic tests, run deliberately — not always-on regressions): a
    twin that drops a boundary drain or mis-wires a forwarding link yields different numbers, so
    the oracle diff is shown to be discriminating.

- **AC-4: Role partition + the HLS synthesis-time win (L2–L3).** `spmw-role-partition`
  classifies PIDs by (link-presence × predicate tag); HLS emits one C++ function **body** per
  role plus a rolled instantiation loop. The hard metric is the count of distinct synthesized
  function bodies, not wall-clock (DEC-4).
  - Positive Tests: emitted HLS C++ contains a number of role function bodies **bounded by the
    role count**, with **no per-PID cloned body names** (`gemm_i_j`); that count is **constant**
    as the grid scales — e.g. `test_tiled_systolic` at 8×8 / 16×16 / 32×32 emits the same ~9
    bodies — and the csim result matches numpy (L2). For a 2D mesh with both extents ≥ 3, the
    role set is exactly {interior, 4 edges, 4 corners} = 9. Synthesis wall-clock is measured and
    **reported as a trend** across those sizes, not asserted as a threshold.
  - Negative Tests: growing the grid adds no new function *bodies* (only instances); two interior
    PIDs whose required static behavior differs (distinct predicate tags — e.g. a checkerboard
    parity body) are **not** merged into one role; a degenerate grid (1D, or `2×N` with an extent
    < 3) yields the correct smaller role set rather than a hard-coded 9.

- **AC-5: Hierarchy, collectives, structural RTL glue (L1–L4) — the committed top of scope.**
  Structural nesting (region in unit), `spmw.shard`, `scatter`/`gather` (declared fan-out/fan-in
  keys), a free-running `ap_ctrl_none` role-IP export, a structural-Verilog emitter (walk
  `spmw.map` → `generate` loops + one FIFO per key), and a new `target="vitis_rtl"` (per-role HLS
  IP → `.xo`/`v++`/XRT), with the host call `mod(A,B,C)` unchanged.
  - Positive Tests: `test_multi_cache_gemm` (★) scatter/gather reach L4 hw_emu; a tiled GEMM =
    one tile synthesized once + a grid of instances; the structural `top.v` co-sims vs the M1
    oracle on `test_systolic`, and `test_systolic` (★) reaches L4 hw_emu on this path.
  - Negative Tests: a top with mismatched FIFO depths or an unmatched producer/consumer is
    rejected by the **topology verifier / channel resolution before RTL is generated** (fail
    early), not only at elaboration/co-sim.

- **AC-6: Folding, banking, memory hierarchy (L2–L4) — stretch (M4).** `fold`/`unroll` map
  attrs; channel→buffer reclassification in `spmw-resolve-channels`; XOR/F2 banking as a new
  partition-**function** extension with static injectivity verification; non-affine Python key
  functions (FFT butterfly). The logical `space=` memory model (AC-7) is consumed here.
  - Positive Tests: `test_fft` (★, new) `fft_spatial` and `fft_folded` both csim-correct; the
    folded HLS report shows conflict-free banked access at II=1; `test_mini_tpu` (★, new)
    exercises hierarchy + heterogeneous units + shared/banked memory at L4 hw_emu.
  - Negative Tests: a bank function non-injective over a cycle's access set is rejected with a
    conflict diagnostic (or serialized with a reported fallback), never silently wrong; a fold
    that would break FIFO order keeps a buffer rather than a stream.

- **AC-7: Logical memory abstraction built up front (DEC-5).** `spmw.shared`/`banked`/`view`
  with a logical `space=` level resolved to the target `Memory` resource model, introduced as
  foundational infrastructure (not deferred), with `resource=` as an explicit escape hatch.
  - Positive Tests: `spmw.shared(T, space="L2")` and `spmw.banked(T, on="col", space="L2")`
    resolve to concrete `Memory` resources on the FPGA target and round-trip through the IR; a
    design pins a buffer with `resource="URAM"` and the emitter honors it.
  - Negative Tests: an unknown `space=` level, or a `banked` layout that overlaps writers without
    a single-writer/`phase` guarantee, is rejected at build rather than producing a race.

- **AC-8: Fast simulator + analytic perf/area model — stretch (M5).** A coroutine functional
  sim over the rolled IR (task-per-PID, cooperative scheduling on bounded FIFOs) with no
  OMP-section-per-PE explosion and no `OMP_NUM_THREADS`-dependent deadlock; a Tier-1 analytic SDF
  model; Tier-2 token clock; area+latency = Σ(role_area × instances) from the O(#roles) HLS
  reports.
  - Positive Tests: coroutine sim matches `spmw-unroll` outputs on `test_systolic` with no
    thread-count env var; the analytic area+latency estimate for the 64×64 systolic array falls
    within a stated tolerance of the actual csynth/cosim numbers.
  - Negative Tests: simulation no longer deadlocks as a function of host thread count; an analytic
    estimate outside tolerance is flagged as a model failure, not accepted.

- **AC-9: Existing suite stays green; AIE untouched.**
  - Positive Tests: the full non-AIE `tests/dataflow` suite passes at every milestone boundary.
  - Negative Tests: no file under `tests/dataflow/aie/` and no AIE lowering path is modified.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
All six milestones land. The three hardware-guaranteed examples — **systolic (`test_systolic`),
FFT (`test_fft`), mini-TPU (`test_mini_tpu`)** — reach L4 (hw/hw_emu) with archived
LUT/FF/DSP/BRAM + latency/II + synthesis-wall-clock reports; the M5 analytic model is validated
against those reports; the structural-RTL path builds an `.xclbin`; the full non-AIE
`tests/dataflow` folder is migrated to SPMW twins.

### Lower Bound (Minimum Acceptable Scope)
**M1 + M2.** `allo.spmw` for a mesh with roles/auto-halo lowers to the rolled `spmw.map` op that
survives to codegen, proven by (a) bit-identical simulator equivalence on the M1 migration set
and (b) HLS emitting a constant O(#roles) set of role function bodies as the array scales, with
`test_systolic` reaching **L2 csim**. This is the safety floor if M3's new RTL backend hits
trouble.

**Committed target (per DEC-2): complete through M3** — the structural-Verilog / `vitis_rtl`
backend and `test_multi_cache_gemm` at L4 hw_emu, `test_systolic` at L4. M4–M6 are stretch.

### Allowed Choices
- **Fixed by user decision:** a **new `spmw` dialect** (DEC-1), not an extension of `allo`
  (`spmw.map`/`spmw.rank` + Topology/Link/Role attrs live in the new dialect; `spmw.shard`
  reuses the existing `allo.grid_map`). The **logical `space=` memory model is built up front**
  (DEC-5). The **role-body count is the hard synthesis gate**; wall-clock is a reported trend and
  II=1 is hard only for the guaranteed FFT/mini-TPU examples (DEC-4). Bit-identical simulator
  equivalence is a hard gate throughout.
- Can use: `!allo.stream` HLS emission (`mlir/lib/Translation/EmitVivadoHLS.cpp`),
  `func_predicate_tags`, `allo.grid_map`, `Layout`/`Shard`/`Replicate`/`Memory`,
  `partition`/`buffer_at`, `move_stream_to_interface`; new MLIR passes; structural Verilog via
  Verilog-2001 `generate`/`genvar`.
- Cannot use: CIRCT / ExportVerilog (dropped for the direct HLS-IP-glue route); hand-written
  unpipelined RTL datapaths (per-PE datapath must be HLS output); any change to the AIE backend
  or `tests/dataflow/aie/*`.

> **Deterministic where the draft is deterministic:** the FFT twiddle numerics, DSP-skip cases,
> SIMD body, and the exact F2 swizzle are carried verbatim from `feature/allo-fft`; SPMW owns the
> *structure*, not the datapath arithmetic. The five surface layers and the `spmw.map` op contract
> are specified in the draft and are not open design choices.

## Feasibility Hints and Suggestions

> Reference only — one possible path, not a mandate.

### Conceptual Approach
`SPMW frontend (topology + roles + shard) → rolled spmw.map op → spmw-role-partition
(link-presence × predicate tag; uses the topology affine maps + grid, so it runs BEFORE channel
materialization) → spmw-resolve-channels (enumerate keys → per-key FIFO arrays, or buffers under
fold) → per-role kernel set O(#roles) → {simulator | HLS role funcs + rolled instantiation loop |
structural RTL glue over per-role IPs}`. Folding (stretch, M4) is a map attribute, not a rewrite;
under fold a channel reclassifies to a banked buffer.

**De-risking note (load-bearing).** The "rolled IR" is **not free today**: `allo/ir/builder.py`
around line 2021 still expands `np.ndindex(*mapping)` to one FuncOp per PID; the `unroll=False`
dedup path (builder around 2022-2032, keyed on `func_predicate_tags`) is real but **AIE-only** and
only dedups FuncOps — it is not a structural op consumed by HLS/RTL. So M1 must *build* the rolled
`spmw.map` op; M2 makes it survive to HLS. `func_predicate_tags` encode `meta_if` control classes,
not topology link-presence — role partition adds the link-presence analysis. `ctx.rank()` is
**compile-time constant** for M1/M2/M3 (preserving `df.get_pid()` specialization, which enables
role merging and HLS quality); a **runtime** rank is only considered under folding (M4, DEC-3).

The `GridMapOp` scaffolding already added by commit #555 lives in the `allo` dialect and is
**verifier-only with no Python producer/consumer today** — it becomes the sharding lowering target
for `spmw.shard`, while the new `spmw` dialect owns `spmw.map`/`spmw.rank`/topology.

### Relevant References (paths verified against the tree)
- `allo/dataflow.py` — `region` (~:560), `kernel(mapping=)` (~:541), `gather`/`scatter` stubs
  (~:32-44), `move_stream_to_interface` (defined ~:52; symbolic-slice block ~:123-222), the AIE
  `unroll=False` build path (~:625-636).
- `allo/ir/builder.py` — mapping expansion + `unroll=False` dedup by predicate tag (~:2021-2091);
  `get_pid` handling (~:1171-1178); stream `put`/`get` lowering (~:2569-2613).
- `allo/ir/infer.py` — `func_predicate_tags` population (~:776, :836).
- `allo/backend/simulator.py` — `_inject_omp_parallel_sections` (one `omp.section` per PE,
  ~:783-806) + usleep spinloop / atomic-write machinery; `OMP_NUM_THREADS` is a test/runtime
  convention, not set in code.
- `mlir/lib/Translation/EmitVivadoHLS.cpp` — the **real** emitter (the draft's
  `mlir/lib/CAPI/Translation/EmitVivadoHLS.cpp` is only a ~20-line CAPI wrapper): `StreamType` →
  `hls::stream` (~:92-111), stream get/put dispatch (~:663-667), construct + depth pragma
  (~:1719), get (~:1785), put (~:1909); only `dataflow`/`inline` func attrs are honored today
  (~:2730) — `ap_ctrl_none` is **not** emittable and is new M3 work, modeled on that attr handling.
- `mlir/include/allo/Dialect/AlloOps.td` (~:1152) + `mlir/lib/Dialect/AlloOps.cpp` (~:307) —
  `GridMapOp` (tensors/sharding/grid), from #555, under a `// SPMW operations` header.
- `allo/memory.py` — `Layout`/`Shard`/`Replicate` (~:10, :34-40) and `Memory` (~:82); models
  concrete FPGA resources today, not logical levels (the `space=` resolver is new).
- `allo/customize.py` — `partition` = Complete/Block/Cyclic only (~:272; XOR/F2 needs a
  partition-function extension); `buffer_at` (~:704).
- New-dialect wiring (the chosen path): `.td` in `mlir/include/allo/Dialect/`, C++ in
  `mlir/lib/Dialect/` (flat today — a `SPMW/` subdir is a *new* convention needing its own nested
  CMake), registration in `mlir/lib/CAPI/Dialect/Registration.cpp` (~:33), CAPI
  `mlir/lib/CAPI/Dialect/Dialects.cpp`, Python `mlir/lib/Bindings/AlloModule.cpp` (~:278), plus
  CMake `IncGen` targets + `EMBED_CAPI_LINK_LIBS`.

## Dependencies and Sequence

### Milestones
1. **M1 — Frontend + rolled IR on the simulator (committed).** Python surface +
   `spmw.build(target=)`; the new `spmw` dialect (ops/attrs/verifier + registration + CMake);
   the logical `space=` memory model; frontend → rolled lowering (reuse `func_predicate_tags`,
   one role func per predicate tag); `spmw-unroll` → per-PID calls. Gate: bit-identical sim vs
   `df` (AC-1/AC-2/AC-3/AC-7).
2. **M2 — Regularity to HLS, pure HLS (committed).** `spmw-role-partition`;
   `spmw-resolve-channels` (streams → per-key FIFO arrays); HLS role emission + rolled
   instantiation loop under `#pragma HLS dataflow`. Gate: constant O(#roles) function bodies as
   the array scales + csim match (AC-4); synthesis wall-clock reported. Structural RTL and
   `ap_ctrl_none` are **not** in M2.
3. **M3 — Hierarchy, collectives, structural RTL glue (committed top of scope).** Nesting +
   `shard` + `scatter`/`gather` (fan-out keys); free-running `ap_ctrl_none` role-IP export;
   structural-Verilog emitter; `.xo`/`v++`/XRT + `target="vitis_rtl"`. Gate: structural `top.v`
   co-sims vs oracle; `test_multi_cache_gemm` L4 hw_emu; `test_systolic` L4 (AC-5).
4. **M4 — Folding, buffers, bank conflicts (stretch).** `fold`/`unroll`; channel→buffer
   reclassification; XOR/`Layout` banking + partition-function + injectivity; non-affine key
   functions; `test_fft`, `test_mini_tpu`. Gate: AC-6.
5. **M5 — Fast simulator + perf estimation (stretch).** Coroutine sim; Tier-1 analytic SDF;
   Tier-2 token clock + role area characterization. Gate: AC-8.
6. **M6 — Non-mesh topologies, sparse, robustness (stretch).** Butterfly/bitonic/crossbar/tree
   generators; topology checks; `spmw.phase()` + single-writer verification; migrate `sparse/*`.

### Critical path
`M1 → M2 → M3` is the committed line; `M2 → M5` and `M2 → M4 → M6` are stretch branches. M1→M2
is the highest-value slice (frontend concision + synthesis-time win, no new RTL backend, HLS
datapath preserved).

### Verification ladder (each migrated test climbs as far as its milestone supports)
L0 Build (`allo-opt` round-trips) · L1 Sim-equiv (bit-identical vs original) · L2 Csim
(`vitis_hls` csim vs numpy) · L3 Csynth (report archived: LUT/FF/DSP/BRAM + latency/II + synth
wall-clock vs flat `df` baseline) · L4 Hw (`hw_emu` cosim or `hw` on-board via XRT). **Every
level (L0–L4), including the build, simulator, and IR round-trip tests, runs on the remote
`brg-zhang-xcel` host** — see Implementation Notes → Development Workflow.

## Task Breakdown

Each task carries exactly one routing tag: `coding` (implemented by Claude) or `analyze`
(executed via Codex, `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1.1 | `allo.spmw` Python surface (unit/region/map, Topology/mesh, peer+key links, ctx.rank/port, roles, stream_in/out) | AC-1 | coding | - |
| task1.2 | New `spmw` dialect: ops/attrs/verifier `.td` + C++, registration (CAPI + Python), CMake IncGen/subdir wiring | AC-2, AC-2.1 | coding | - |
| task1.3 | Frontend → rolled `spmw.map` lowering (one role func per predicate tag; reuse func_predicate_tags) | AC-2, AC-3 | coding | task1.1, task1.2 |
| task1.4 | `spmw-unroll` lowering → per-PID `func.call` | AC-3 | coding | task1.2, task1.3 |
| task1.5 | M1 SPMW twins + oracle-diff harness (incl. mutation/diagnostic negatives) | AC-3, AC-9 | coding | task1.4 |
| task1.6 | `spmw.build(target=)` dispatch (simulator, vitis_hls) integrated with `Schedule.build` | AC-1 | coding | task1.1 |
| task1.7 | Logical `space=` memory-level model + resolver to `Memory` (shared/banked/view surface, resource= escape hatch) | AC-7 | coding | task1.1 |
| task2.1 | `spmw-role-partition` (link-presence × predicate tag; degenerate-grid aware) | AC-4 | coding | task1.3 |
| task2.2 | `spmw-resolve-channels` (streams): distinct keys → per-key FIFO arrays | AC-4 | coding | task2.1 |
| task2.3 | HLS role emission + rolled instantiation loop under `#pragma HLS dataflow` | AC-4 | coding | task2.2 |
| task2.4 | Role-partition soundness / interior-equivalence argument | AC-4 | analyze | task2.1 |
| task2.5 | Synthesis-time measurement harness + 8/16/32 scaling experiment (defined metric) | AC-4 | analyze | task2.3 |
| task3.1 | Nesting + `spmw.shard` (→ `allo.grid_map`) + `scatter`/`gather` (declared fan-out/fan-in keys) | AC-5 | coding | task2.2 |
| task3.2 | Free-running `ap_ctrl_none` role-IP export (new emitter func attr) | AC-5 | coding | task2.3 |
| task3.3 | Structural-Verilog emitter (walk `spmw.map` → generate loops + one FIFO per key) | AC-5 | coding | task3.2 |
| task3.4 | `.xo`/`v++`/XRT packaging + `target="vitis_rtl"`; hierarchical IP reuse | AC-5 | coding | task3.3 |
| task3.5 | Structural `top.v` co-sim vs the M1 oracle | AC-5 | coding | task3.4 |
| task4.1 | `fold`/`unroll` map attrs; wire shared/banked/view consumers onto the M1 memory model | AC-6 | coding | task2.2, task1.7 |
| task4.2 | Channel → buffer reclassification | AC-6 | coding | task4.1 |
| task4.3 | XOR/partition-function banking + static injectivity verify | AC-6 | coding | task4.2 |
| task4.4 | Port FFT numerics (twiddle/DSP-skip/SIMD/F2 swizzle) verbatim from `feature/allo-fft` | AC-6 | coding | task4.3 |
| task5.1 | Coroutine functional sim on rolled IR (task-per-PID; no OMP-section-per-PE) | AC-8 | coding | task1.4 |
| task5.2 | Tier-1 analytic SDF model (throughput/latency/min-depth/deadlock) | AC-8 | coding | task5.1 |
| task5.3 | Tier-2 token clock + role-area characterization; cross-check vs L3/L4 reports | AC-8 | analyze | task5.2, task2.5 |
| task6.1 | Non-mesh generators (butterfly/bitonic/crossbar/tree) + topology checks | AC-1, AC-2 | coding | task4.3 |
| task6.2 | `spmw.phase()` + single-writer verification | AC-7 | coding | task1.7 |
| task6.3 | Migrate `tests/dataflow/sparse/*` twins | AC-9 | coding | task6.1 |

## Claude-Codex Deliberation

### Agreements
- The draft's "rolled IR already exists" premise is overstated; M1 must *build* the rolled
  `spmw.map` op, and M2 is where it must survive to HLS. The `unroll=False` path is AIE-only and
  only dedups FuncOps.
- `func_predicate_tags` encode `meta_if` control classes, not topology link-presence; role
  partition is the genuinely new analysis on top.
- M2's honest, tool-noise-free claim is a constant O(#roles) synthesized function-body count;
  full-array hardware execution needs M3's structural glue.
- `ap_ctrl_none`, `target="vitis_rtl"`, XOR/F2 banking, the structural-Verilog backend, and the
  logical `space=` model are genuinely new work, not "reuse."
- `ctx.rank()` is compile-time for M1/M2/M3; runtime rank is a folding-time (M4) question.
- Topology key maps are static-affine for M1/M2/M3; arbitrary Python key functions arrive in M4.
- AIE stays out of scope; the existing non-AIE suite stays green throughout.
- The draft cites the wrong HLS emitter file (a CAPI wrapper); the real emitter is
  `mlir/lib/Translation/EmitVivadoHLS.cpp`.

### Resolved Disagreements
- **"No new backend" for the lower bound** → corrected to "no new **RTL** backend"; M2 makes
  additive changes to the existing Vitis HLS emitter (role emission, rolled loop, channel arrays).
- **M2 full-array L4** → moved to M3 (structural glue); M2's gate is L2 csim + the role-body count.
- **"~flat synthesis wall-clock" as a hard AC** → demoted to a reported trend; the hard metric is
  the constant O(#roles) function-body count (confirmed by the user, DEC-4).
- **`ap_ctrl_none` in M2** → moved to M3 (free-running IP export).
- **Dialect placement** → Claude/Codex defaulted to extending `allo`; the user chose a separate
  `spmw` dialect (DEC-1), accepting the registration/CMake cost. Recorded and reflected in AC-2.1
  and task1.2.
- **Memory-abstraction timing** → Claude/Codex defaulted to deferring to M4; the user chose to
  build the logical `space=` model up front (DEC-5). Recorded in AC-7 and task1.7.

### Convergence Status
- Final Status: `converged`. Two consecutive Codex review rounds produced no `REQUIRED_CHANGES`
  and no `DISAGREE`; the four material user decisions (DEC-1, DEC-2, DEC-4, DEC-5) are resolved.
  One decision (DEC-3) is intentionally deferred to the M4 stretch milestone with a working
  default and does not block the committed M1–M3 scope.

## Pending User Decisions

- **DEC-1: Dialect placement — RESOLVED.** Decision: **new `spmw` dialect** under
  `mlir/{include,lib}/allo/Dialect/SPMW/`. `spmw.map`/`spmw.rank` + Topology/Link/Role attrs live
  there; `spmw.shard` reuses the existing `allo.grid_map`. Accepts the cost of registration in
  three sites (CAPI, Python, CMake link) and a new per-dialect subdirectory convention.
  - Claude Position: extend the `allo` dialect for M1/M2 to minimize churn.
  - Codex Position: same (extend `allo`).
  - Tradeoff Summary: user prioritized clean long-term separation and the original design intent
    over the lower-friction path; the extra plumbing is one-time and captured in task1.2.
  - Decision Status: **new `spmw` dialect** (user).

- **DEC-2: Success bar / funded scope — RESOLVED.** Decision: **complete through M3** (frontend +
  synthesis-time win + hierarchy/collectives/structural-RTL, with `test_multi_cache_gemm` and
  `test_systolic` at L4). M4–M6 are stretch.
  - Claude Position: commit M1+M2 as the funded floor; M3+ as stretch.
  - Codex Position: flagged as needing the user's call.
  - Tradeoff Summary: through-M3 proves RTL scaling (a whole new backend) but stops short of
    FFT/mini-TPU-on-hardware and the perf model.
  - Decision Status: **through M3** (user).

- **DEC-3: Runtime `ctx.rank` under folding — PENDING (deferred, M4).** For folded maps (M4),
  does a physical PE running F logical PIDs get a *runtime* `ctx.rank`, or is fold restricted to
  cases where rank stays a compile-time constant per unrolled copy? (M1/M2/M3 are compile-time,
  already fixed.)
  - Claude Position: start compile-time-only; add runtime rank in M4 only if a target example
    needs it.
  - Codex Position: `N/A - open question` (needs a decision only when M4 begins).
  - Tradeoff Summary: runtime rank generalizes folding but costs HLS specialization quality;
    does not affect the committed M1–M3 scope.
  - Decision Status: `PENDING` (revisit at M4 start).

- **DEC-4: Metric hardness — RESOLVED.** Decision: **role-body count is the hard gate**;
  synthesis wall-clock is a reported trend; II=1 conflict-free banking is hard for the guaranteed
  FFT/mini-TPU examples and best-effort-with-fallback elsewhere. Bit-identical simulator
  equivalence is a hard gate throughout.
  - Claude Position / Codex Position: same as the decision.
  - Decision Status: **role-count hard, wall-clock/II softer** (user).

- **DEC-5: Memory abstraction timing — RESOLVED.** Decision: **build the logical `space=` model
  up front** (task1.7, AC-7), with `resource=` as the escape hatch, rather than deferring to M4.
  - Claude Position: reuse concrete `Memory`/`partition`/`buffer_at` through M1–M3, add `space=`
    in M4.
  - Codex Position: same (defer).
  - Tradeoff Summary: user prioritized a clean hierarchy-agnostic API and earlier example clarity
    over less up-front design.
  - Decision Status: **build `space=` up front** (user).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-",
  "Milestone", "Phase", "Step", or similar workflow markers — these belong in this plan document,
  not in the resulting codebase. Use descriptive, domain-appropriate naming (e.g. role names like
  `pe_interior`/`pe_load_a`, op names like `spmw.map`) instead.
- Every source file carries the Apache-2.0 license header (lint enforces it). Python is formatted
  with `black==24.8.0`, C++ with `clang-format`.
- Re-run `pip install -v -e .` after any C++ change under `mlir/` (the compiled extension must be
  rebuilt); pure-Python edits under `allo/` need no rebuild.

### Development Workflow
- **All tests run on the remote `brg-zhang-xcel` host — no exceptions.** This includes not only
  the HLS/synthesis/on-board runs (L2–L4) but also the build (L0), simulator sim-equivalence
  (L1), and IR round-trip tests. Nothing in the verification ladder is validated locally; sync
  local edits up first (`rsync` per CLAUDE.md), then build/run on `brg-zhang-xcel` with its env
  (`PATH` to the prebuilt conda env, `LLVM_BUILD_DIR`, and — for L2+ — the sourced Vitis env).
- **Commit after each feature.** Every time a task (a feature — e.g. one dialect op, one pass,
  one backend emitter, one migrated test twin) is implemented and its gating test passes on
  `brg-zhang-xcel`, commit the change before moving on. Keep commits small and scoped to a single
  feature/task so the history maps cleanly onto the task breakdown, and so a regression can be
  bisected to one feature. Work on a feature branch (the current branch is `feat/spmw`), not on
  `main`.

--- Original Design Draft Start ---

# SPMW: A Single-Program Multiple-Work-Unit Frontend for Allo

**Design + implementation plan.** SPMW is a Python interface that exploits the
*regularity* of spatial accelerators: write one work unit, declare how it is
replicated, and declare how the copies talk to each other — instead of materializing
every PE and every wire by hand as HLS/RTL forces you to. It is an **evolution of
`allo.dataflow`**: every surface construct either reuses an existing primitive
(`@df.region`, `@df.kernel(mapping=)`, `df.get_pid()`, `Stream[T,d]`, `.put()/.get()`,
`allo.meta_if`, the `allo.grid_map`/sharding ops, `allo/memory.py` `Layout`) or lowers
to a new but small MLIR dialect that preserves structure to codegen.

- **Part I — Programming model** (§1–2): the API.
- **Part II — Worked examples** (§3): GEMM, hierarchical GEMM, daisy-chain GEMM,
  mini-TPU, FFT.
- **Part III — Compilation & implementation** (§4–8): the rolled IR, the `spmw`
  dialect, the HLS + RTL-glue backend, folding/banking, the simulator, and a concrete
  phased implementation plan.

---

# Part I — Programming model

## 1. The problem, and what Allo has today

An accelerator is rarely irregular. A systolic array is *one* PE program, replicated
over a grid, wired to neighbors by a fixed rule. HLS/RTL make you pay for that
regularity three times: (1) **instantiation** (replicate the PE P0·P1 times),
(2) **interconnect** (name/connect every FIFO + boundary loaders/drains), and
(3) **boundary specialization** (edge PEs differ from interior, hand-written). The
cost is not typing — it is **synthesis time** (the tool re-elaborates every instance),
**verification** (a bug in one hand-copied variant surfaces late), and **DSE**
(resizing the array is a structural rewrite).

`allo.dataflow` already solves #1 (`@df.kernel(mapping=[P0,P1])` writes the PE once).
But the current systolic examples (`test_systolic.py`, `test_weight_stationary_gemm.py`,
`test_unified_systolic.py`) spend most lines on #2 and #3: a long
`meta_if(i==0)/meta_elif(j==0)/…` chain, every data movement spelled as absolute grid
arithmetic like `fifo_A[i, j+1].put(local_A[i-1,k])`. The regularity is there but
re-derived by hand in every kernel; the interconnect topology exists only implicitly.

**SPMW makes the three sources of regularity first-class:**

| Regularity        | HLS/RTL today          | `allo.dataflow` today          | SPMW                                     |
|-------------------|------------------------|--------------------------------|------------------------------------------|
| Replication       | manual generate/unroll | `mapping=[P0,P1]` ✅            | keep                                     |
| Interconnect      | name every wire        | absolute `fifo[i,j+1]` math     | **named links from a topology**          |
| Boundary variants | hand-write each        | `meta_if` PID chains           | **roles / auto-halo (topology-driven)**  |
| Hierarchy         | nested modules by hand | `L1_/L2_/L3_` naming convention | **structural nesting + cross-level chans**|
| Data placement    | manual address math    | manual `local_A[i-1,k]` slicing | **declarative `shard()` → `grid_map`**   |

It keeps the two axes you named — **single-program** (like Triton) and **explicit
interconnect** (`.put()/.get()`) — and adds the two things Triton lacks: a **declared
interconnect topology** and **hierarchical composition of kernels with mapping**.

## 2. The API

Five layers, each lowering to the one below; drop down whenever the sugar doesn't fit.
Throughout: `import allo.spmw as spmw`.

```
Layer 4  Composition   nesting, cross-level channels, shard(), scatter/gather
Layer 3  Roles / halo  @unit.role(...), stream_in/stream_out (auto loaders/drains)
Layer 2  Topology      mesh/ring/tree/crossbar/Topology; named links (a peer function)
Layer 1  Work unit     @spmw.unit — single program + rank() + port I/O
Layer 0  Data links    Stream[T,d] (FIFO)  |  arrays + spmw.shared/banked (memory)
```

### 2.1 `unit` vs `region` (the two program kinds)

- **`@spmw.unit`** — the single program that runs at **each** grid point; it is what
  `map` *replicates*. Analogue of today's `@df.kernel`.
- **`@spmw.region`** — a **composition scope**, *not* replicated: it holds top-level
  tensor arguments, declares channels/buffers, and calls `spmw.map`. Analogue of
  today's `@df.region`. A region can also be a **reusable sub-graph** called from
  inside a unit — that is the nesting mechanism (§2.4).

```python
@spmw.unit
def pe(ctx):
    a = ctx.west.get()          # port I/O; the topology binds .west to the right FIFO
    b = ctx.north.get()
    c = ctx.acc + a * b         # ctx.acc = unit-local state
    ctx.east.put(a); ctx.south.put(b)
```

`ctx.rank()` == today's `df.get_pid()`. `ctx.west` is sugar for `ctx.port("west")`.

### 2.2 Topology (Layer 2) — one model for all interconnect

A topology's `link` function gives, for each local **port**, its far end. There are two
forms; the first covers everything neighbor-based, the second is only for permutations
and collectives.

**Peer form (point-to-point — the common case).** `port -> (peer_rank, peer_port)`:

```python
def mesh(shape):
    return spmw.Topology(grid=shape, link=lambda i, j: {
        "east": ((i, j+1), "west"),  "west":  ((i, j-1), "east"),
        "south":((i+1, j), "north"), "north": ((i-1, j), "south")})
```

Read `"east": ((i,j+1), "west")` as "my `east` connects to the `west` of the PE at
(i,j+1)." The port name (`east`) is this PE's *local* handle — the body writes
`ctx.east.put(a)`. The peer + peer-port name the other end. **That is all** — no separate
channel object, no key, no alias table (`east` **is** the name; there is no `"E"`
shorthand). The compiler makes one FIFO per directed edge and checks the pairing is
symmetric ((i,j).east ↔ (i,j+1).west). A port whose peer is out of range is a **boundary**
port (drives roles/halo, §2.3); a put/get on it is elided (the generated `meta_if(j<Ct)`
guard). Element type is **inferred** from the body (checked equal at both ends); depth
defaults to 2, overridable via `depths=`.

Mapping needs no `channels=`:

```python
spmw.map(pe, grid=mesh)                       # types inferred, depths default
spmw.map(pe, grid=mesh, depths={"east": 4})   # optional per-port tuning
```

**Key form (permutation / collective — the general case).** When the far end is *not* a
simple neighbor coordinate, ports rendezvous by a shared key:
`port -> (channel_key, "src"|"sink")`; ports emitting the same key are the same FIFO. The
FFT butterfly needs this — its consumer is "the stage-(s+1) butterfly that reads lane
`up`", not a fixed offset — so it writes key `("lane", s+1, up)` and the reader reads the
same key (§3.5). Fan-out/fan-in keys (one src, many sinks) express broadcast and the
scatter/gather collectives (§3.3). **Peer form desugars to key form** (the key is derived
from the (rank,port)+(peer,peer_port) pair), so there is one lowering — use peer form for
meshes/rings/trees/chains, key form only for permutations and collectives.

Non-mesh neighbor topologies are the same peer-form mechanism with a different `link`
(built-ins `spmw.torus/ring/tree`; `spmw.crossbar` uses key form). **Multiple interconnect
patterns on one PE are just more links in one topology** (§3.3) — there is no "list of
topologies."

### 2.3 Roles & auto-halo (Layer 3)

Boundary specialization, explicit → automatic.

**Roles** — role-specific bodies selected by *which links are missing* (the topology
knows), replacing the `meta_if` chain:

```python
@spmw.unit
def pe(ctx): ...                          # interior (default)
@pe.role("west")                          # west edge (no west peer): load A
def pe_load_a(ctx):
    for k in range(K): ctx.east.put(ctx.a_tile[ctx.row, k])
@pe.role("east", "south")                 # east/south edges: drain
def pe_drain(ctx):
    for _ in range(K): ctx.west.get()
```

**Auto-halo** (sugar over roles) — for an operand that just streams across the array,
declare the *flow direction*; the loader edge, forwarding, and drain edge are
synthesized:

```python
spmw.stream_in (A, into=pe, flow="W->E")
spmw.stream_in (B, into=pe, flow="N->S")
spmw.stream_out(C, from_=pe, where="local", as_="c_local")
```

### 2.4 Composition & hierarchy (Layer 4)

1. **Structural nesting** — a unit body may `spmw.map` a child grid or call another
   `@spmw.region`. Real nesting (child sub-array inside each parent PE), replacing the
   `L1_/L2_/L3_` naming convention with structure.
2. **Cross-level channels** — parent and child grids communicate through channels/
   buffers at the level boundary (§3.2, §3.4).
3. **`spmw.shard(A, grid=, tiling=, along=)`** — slices a tensor across a grid so each
   unit gets its tile; lowers to `allo.grid_map`/sharding.

### 2.5 Data links: streams vs memory (Layer 0)

Two connection kinds. **Streams** (`Stream[T,d]`, `.put/.get`) are ordered, addressless,
self-synchronizing FIFOs. **Memory** is an addressed array. A plain array declared in a
unit (`buf: float32[Rt,Ct]`) is a *local* buffer exactly as in Allo today — no new type.
The only things a bare type can't express are the **memory level** and **cross-PE
sharing**, so those are placement *wrappers*, not a distinct type:

```python
w = spmw.shared(float32[K, Ct], space="L2")            # replicated RA memory (Layout.Replicate)
c = spmw.banked(float32[Rt, Ct], on="row", space="L2") # per-row banks (Layout.Shard)
b = spmw.shared(float32[K, Ct], space="L2", double=True)# ping-pong (buffer_at)
```

| Aspect     | `Stream[T,d]`            | array / `spmw.shared` / `spmw.banked`  |
|------------|--------------------------|----------------------------------------|
| Access     | `.put`/`.get`            | `buf[i,j]` random read/write           |
| Ordering   | FIFO, implicit           | none — programmer/schedule imposes it  |
| Sync       | **free** (blocking)      | explicit (banks/phases/single-writer)  |
| Lowers to  | `!allo.stream`           | `memref` + `Layout`                    |

`space=` is a **logical level** resolved to the target's `Memory` resource model in
`allo/memory.py` (FPGA: URAM/BRAM/LUTRAM/DDR/HBM; AIE: L3/L2/L1/reg), with an optional
explicit `resource=`. So it is hierarchy-agnostic — not a fixed frontend enum. Levels
connect by **views** whose realization is a knob: `spmw.view(B, into=w, tiling=(K,Ct),
how="dma")` where `how = dma` (block copy) `| stream` (FIFO feed via `.to()`) `| remote`
(addressed load in place).

**Collectives** — `spmw.scatter`/`gather` distribute/collect along a chain with a
pack/unpack rule (§3.3); they are what the existing `df.gather/scatter` stubs
(`allo/dataflow.py:32-44`) should compile to.

**The honest cost — synchronization.** FIFOs synchronize for free; shared RA memory
does not. Stance: (1) prefer `banked` (single-writer-per-bank, statically checkable);
(2) for shared read/write require explicit `spmw.phase()` barriers or per-phase
`readonly`/`writeonly`; (3) determinism is default, unsynchronized sharing is opt-in
and flagged. This is the one place the abstraction costs the user something, surfaced
rather than hidden.

---

# Part II — Worked examples

## 3.1 Systolic GEMM (the flagship)

Analogue of `test_systolic.py`, whose ~50 lines are dominated by the `meta_if` halo. In
SPMW the interior PE is the whole story; the halo is declared:

```python
M, N, K = 32, 32, 32
mesh = spmw.mesh((M, N))

@spmw.unit
def pe(ctx):                                    # ALL you hand-write
    c: float32 = 0
    for k in range(K):
        a = ctx.west.get(); b = ctx.north.get()
        c += a * b
        ctx.east.put(a); ctx.south.put(b)       # systolic forwarding; no [i,j+1]
    ctx.c_local[0] = c

@spmw.region()
def gemm(A: float32[M,K], B: float32[K,N], C: float32[M,N]):
    spmw.map(pe, grid=mesh)                      # no channels= needed; types inferred
    spmw.stream_in (A, into=pe, flow="W->E")
    spmw.stream_in (B, into=pe, flow="N->S")
    spmw.stream_out(C, from_=pe, where="local", as_="c_local")
```

Grid size, dataflow, and tiling are parameters, not structure. Weight-stationary is a
body edit plus `spmw.stationary(B, at=pe)` — interconnect untouched.

## 3.2 Hierarchical tiled GEMM: L3 host → L2 tile engines → L1 mesh

Analogue of the L1/L2/L3 streams in `test_tiled_systolic.py`/`test_unified_systolic.py`
(180 lines of index math), with levels as structure. `pe` from §3.1 is reused verbatim.

```python
M, N, K = 512, 512, 512
Rt, Ct  = 16, 16;  TM, TN = M//Rt, N//Ct

@spmw.region()                                     # L1: one output tile
def tile_gemm(a_tile: float32[Rt,K], b_tile: float32[K,Ct], c_tile: float32[Rt,Ct]):
    spmw.map(pe, grid=spmw.mesh((Rt,Ct)))
    spmw.stream_in(a_tile, into=pe, flow="W->E"); spmw.stream_in(b_tile, into=pe, flow="N->S")
    spmw.stream_out(c_tile, from_=pe, where="local", as_="c_local")

@spmw.unit                                         # L2: one tile engine drives one L1 mesh
def tile_engine(ctx):
    tile_gemm(ctx.a_shard, ctx.b_shard, ctx.c_shard)     # structural nesting L2 -> L1

@spmw.region()                                     # L3: host shards operands over the L2 grid
def tiled_gemm(A: float32[M,K], B: float32[K,N], C: float32[M,N]):
    spmw.map(tile_engine, grid=spmw.Grid((TM,TN)))
    spmw.shard(A, tiling=(Rt,K), along="row", as_="a_shard")   # -> grid_map op
    spmw.shard(B, tiling=(K,Ct), along="col", as_="b_shard")
    spmw.shard(C, tiling=(Rt,Ct),             as_="c_shard")
```

Three levels, zero duplication: L1 written once, L2 instantiates by *mapping*, L3 only
shards; a 512×512 design and a 4×4 toy differ by four constants. `shard` replaces
`local_A[m*Mt+i-1,k]` address math; cross-level wiring is the three args of
`tile_gemm(...)`, not hand-managed stream arrays.

## 3.3 Daisy-chain multi-cache GEMM (multiple links, one topology)

`test_multi_cache_gemm.py` fuses several interconnect patterns on one PE: nearest-
neighbor A/B forwarding **and** a per-column packed partial-sum chain. This is **one
topology with more links** — no "list of topologies", no `chain_per_col` primitive:

```python
Rt, Ct, K = 2, 2, 16
PackC = UInt(Rt * 8)

def mxu_links(i, j):                                                     # all peer form
    return {"east":((i,j+1),"west"),      "west": ((i,j-1),"east"),      # A forwarding
            "south":((i+1,j),"north"),     "north":((i-1,j),"south"),    # B forwarding
            "psum_out":((i+1,j),"psum_in"),"psum_in":((i-1,j),"psum_out")}# per-column C chain
mxu = spmw.Topology(grid=(Rt,Ct), link=mxu_links)

@spmw.unit
def pe(ctx):
    i, j = ctx.rank()
    c: int8 = 0
    for k in range(K):
        a = ctx.west.get(); b = ctx.north.get()
        c += a * b
        ctx.east.put(a); ctx.south.put(b)          # elided on E/S boundary
    word: PackC = ctx.psum_in.get_or(0)            # 0 at top row, else upstream word
    word[i*8:(i+1)*8] = c
    ctx.psum_out.put(word)                          # bottom row's word is drained (below)
```

The "per-column chain" is simply the `psum_out → psum_in` link to the `(i+1,j)` neighbor:
same column (`j` fixed), next row. It is just another peer-form link in the same topology
— nothing special, no separate primitive. Distribution of packed A/B words from off-chip and the
bottom-row collection use **collectives** over the same style of chain links:

```python
@spmw.region()
def MXU(A: ..., B: ..., C: ...):
    spmw.map(pe, grid=mxu)
    spmw.scatter(A, into=pe.edge("west"),  unpack=lambda w,r: w[r*8:(r+1)*8])  # packed A down col 0
    spmw.scatter(B, into=pe.edge("north"), unpack=lambda w,c: w[c*8:(c+1)*8])  # packed B along row 0
    spmw.gather (C, from_=pe.port("psum_out"), order="reverse")            # collect column words
```

The whole halo (every `meta_if` in the original's lines 56-98) collapses to two
`scatter`s and a `gather`. Packing/unpacking (the "multi-cache" essence) stays explicit
— genuine algorithmic content. `order="reverse"` surfaces the original's `Ct-j`
reversal as a parameter.

## 3.4 Mini-TPU (complex hierarchical composition)

A TPU-style inference tile: **weight loader + Unified Buffer + the §3.1 MXU + accumulators
+ a vector activation unit**, wired at the top region by streams and shared/banked
buffers. It shows region reuse, a 1D vector `map`, and buffer-connected stages.

```python
Rt, Ct, K = 16, 16, 256

# ---- MXU: reuse the systolic GEMM as a sub-region (output-stationary) --------------
@spmw.region()
def mxu(act: float32[Rt, K], wgt: float32[K, Ct], psum: float32[Rt, Ct]):
    spmw.map(pe, grid=spmw.mesh((Rt, Ct)))               # pe from §3.1
    spmw.stream_in (act, into=pe, flow="W->E")           # activations stream in from the UB
    spmw.stream_in (wgt, into=pe, flow="N->S")           # weights stream in from weight buffer
    spmw.stream_out(psum, from_=pe, where="local", as_="c_local")

# ---- Activation unit: a 1D vector engine, one lane per output column ---------------
@spmw.unit
def act_pe(ctx):
    j = ctx.rank()                                       # column this lane owns
    for r in range(Rt):
        x = ctx.col_in.get()                             # a psum element from the MXU column
        y = x + ctx.bias[j]                              # bias add
        ctx.col_out.put(y if y > 0.0 else 0.0)           # ReLU

# ---- Mini-TPU top: DRAM -> UB/weights -> MXU -> accumulate -> activation -> UB -----
@spmw.region()
def mini_tpu(ACT: float32[Rt, K], WGT: float32[K, Ct], BIAS: float32[Ct],
             OUT: float32[Rt, Ct]):
    # on-chip memory hierarchy (levels resolve to target Memory resources)
    ub    = spmw.shared(float32[Rt, K], space="L2")            # Unified Buffer (activations)
    wbuf  = spmw.shared(float32[K, Ct], space="L2")            # weight buffer
    psum  = spmw.banked(float32[Rt, Ct], on="col", space="L2") # accumulators, one bank per column

    spmw.view(ACT, into=ub,   how="dma")                 # DRAM -> UB
    spmw.view(WGT, into=wbuf, how="dma")                 # DRAM -> weight buffer

    # stage 1: matmul on the systolic array (reads UB + weights, writes accumulator banks)
    mxu(ub, wbuf, psum)                                  # structural nesting: top -> MXU mesh

    # stage 2: bias + ReLU as a 1D vector map over the Ct columns of the accumulator
    spmw.map(act_pe, grid=spmw.Grid((Ct,)), bind={"bias": BIAS})
    spmw.stream_in (psum, into=act_pe, flow="col", index=lambda r, j: (r, j))  # psum col j -> lane j
    spmw.stream_out(OUT,  from_=act_pe, flow="col")
```

What this demonstrates, concretely:

- **Hierarchy by composition, not duplication.** `mini_tpu` (top region) instantiates
  `mxu` (a region = the whole systolic array) and `act_pe` (a 1D vector unit) as
  sub-blocks. The MXU internally maps `pe` over `Rt×Ct` — three levels
  (top → array → PE) with each piece written once.
- **Two connection kinds in one design.** UB/weights/accumulators are **memory**
  (`shared`/`banked`, placed at `L2`); the MXU↔activation and DRAM↔on-chip paths are
  **views/streams**. The `banked(on="col")` accumulator gives the activation stage a
  conflict-free per-column read.
- **Heterogeneous units.** The array PE (`pe`, a MAC) and the vector PE (`act_pe`, a
  bias+ReLU lane) are *different* units mapped onto *different* grids, connected
  through the `psum` buffer. This is the case Triton (single kernel) and hand-RTL
  (hand-wired blocks) can't express cleanly.
- **DSE knobs.** MXU size (`Rt,Ct`), reduction depth (`K`), accumulator banking, and
  the memory levels are all parameters over the same program — grow the tile or move a
  buffer to URAM without rewriting structure.

To extend toward a fuller TPU (multiple layers, pooling/normalization), add more vector
units as 1D maps and chain them through UB — the composition pattern is unchanged.

## 3.5 FFT: folding and bank conflicts

**One unit + one topology; spatial and folded variants differ only in `spmw.map` knobs.**
(Validated against the real `feature/allo-fft/test_fft.py`, which independently uses
`fold={dim:F}` dicts and XOR-swizzle banking — §7.)

```python
N, S, HALF, QN = 256, 8, 128, 64
_eps = float(np.finfo(np.float32).eps)                          # epsilon-snap twiddles
TWR = np.where(np.abs(np.cos(-2*np.pi*np.arange(HALF)/N))<_eps, 0.0, np.cos(...)).astype(np.float32)
TWI = np.where(np.abs(np.sin(-2*np.pi*np.arange(HALF)/N))<_eps, 0.0, np.sin(...)).astype(np.float32)

def bfly_pair(s, b):
    stride = 1 << s
    upper  = ((b >> s) << (s+1)) | (b & (stride-1))
    return upper, upper | stride                               # lower = upper ^ (1<<s)
def twiddle_index(s, b): return (b & ((1<<s)-1)) << (S-1-s)
def bit_reverse(i):
    r = 0
    for k in range(S): r |= ((i>>k)&1) << (S-1-k)
    return r

@spmw.unit                                                     # ONE butterfly, both variants
def bfly(ctx):
    s, b = ctx.rank()
    a  = ctx.up_in.get(); bb = ctx.lo_in.get()
    tw_idx = twiddle_index(s, b)          # ConstExpr in spatial, dynamic under fold
    with allo.meta_if(tw_idx == 0):     bw = bb                          # tw=(1,0), no DSP
    with allo.meta_elif(tw_idx == QN):  bw = spmw.complex(bb.im, -bb.re) # tw=(0,-1), no DSP
    with allo.meta_else():
        tr: float32 = TWR[tw_idx]; ti: float32 = TWI[tw_idx]
        bw = spmw.complex(bb.re*tr - bb.im*ti, bb.re*ti + bb.im*tr)      # the only DSP
    ctx.up_out.put(spmw.complex(a.re+bw.re, a.im+bw.im))
    ctx.lo_out.put(spmw.complex(a.re-bw.re, a.im-bw.im))

def bfly_links(s, b):                                          # butterfly wiring via shared keys
    up, lo = bfly_pair(s, b)
    return {"up_in": (("lane",s,up),  "sink"), "lo_in": (("lane",s,lo),  "sink"),
            "up_out":(("lane",s+1,up),"src"),  "lo_out":(("lane",s+1,lo),"src")}
bfly.topo = spmw.Topology(grid=(S,HALF), link=bfly_links)      # channels inferred from keys

@spmw.region()                                                 # A: fully spatial (1024 PEs, all FIFOs)
def fft_spatial(Xr:float32[N],Xi:float32[N],Yr:float32[N],Yi:float32[N]):
    spmw.map(bfly, grid=(S,HALF), topo=bfly.topo)
    spmw.stream_in ((Xr,Xi), into="lane", at_stage=0, index=bit_reverse)
    spmw.stream_out((Yr,Yi), from_="lane", at_stage=S)

U = 32
@spmw.region()                                                 # B: folded (8 PEs, banked buffers)
def fft_folded(Xr:float32[N],Xi:float32[N],Yr:float32[N],Yi:float32[N]):
    spmw.map(bfly, grid=(S,HALF), topo=bfly.topo,
             fold   = {1: HALF},                               # 128 butterflies -> 1 PE (time-mux)
             unroll = {1: U},                                  # U butterflies/cycle, II=1
             layout = {"lane": spmw.Layout.xor_bank(banks=2*U, peer="topology")})  # F2 swizzle
    spmw.stream_in ((Xr,Xi), into="lane", at_stage=0, index=bit_reverse)
    spmw.stream_out((Yr,Yi), from_="lane", at_stage=S)
```

Here the `link` function emits keys like `("lane", s+1, up)` — a **permutation network**
where the natural key is a computed slot, not a neighbor offset. Same key rule as the
mesh; no separate channel declaration. When a channel becomes a buffer under folding is
compiler-decided (§4.5): spatial → every `lane` edge is a FIFO; folded → one PE per
stage random-accesses pairs `(i, i^(1<<s))`, so `lane[s]` becomes an XOR-banked `memref`
serving `2U` conflict-free accesses/cycle. What stays hand-written is the *datapath*
numerics (twiddle snapping, DSP-skip cases, SIMD body). SPMW owns the structure.

---

# Part III — Compilation & implementation

## 4.1 The principle: a rolled IR that survives to codegen

A concise frontend is worthless if the compiler flattens it. Allo's default path does:
`@df.kernel(mapping=[P0,P1])` at `builder.py:2021` runs `for dim in np.ndindex(*mapping)`
and emits **one `FuncOp` per PID** (`gemm_0_0`, …), so Vitis HLS gets P0·P1 distinct
functions and re-schedules per function — synthesis time **O(P0·P1)**
(`test_tiled_systolic.py:97-104` asserts the distinct funcs).

The fix already exists, wired only to AIE: at `builder.py:2022-2032`, when `unroll=False`,
PIDs with the same `func_predicate_tags` **role tag** are deduplicated, interconnect is
kept **symbolic** (`move_stream_to_interface(unroll=False)`, `dataflow.py:123-222`), and
`allo.grid_map` is a **rolled** structured op. The plan makes this rolled path canonical
and teaches the backends to consume it.

**Design principle: `spmw.map` (the op) is a structural op that survives to codegen.** It
carries the grid, topology, sharding, and symbol references to per-role unit funcs, and
is *never* expanded to per-PE funcs on the HLS/RTL path. The clean split:

- **role `func.func`s = the datapath** → HLS C++, one per role, synthesized once each;
- **`spmw.map` = the structure** → structural RTL (`generate` loop + FIFO per channel
  key) instantiating the HLS-exported role IPs.

*(Op name: `spmw.map` — matches the frontend verb and reads as "map a unit over a grid",
the topology-aware generalization of `allo.grid_map`. Alternative if the overload with
the frontend function is undesirable: `spmw.fabric`. Renamed from the earlier working
name `spmw.array`, which collided with data arrays.)*

## 4.2 The `spmw` dialect

New dialect under `mlir/{include,lib}/allo/Dialect/SPMW/`, registered alongside
`AlloDialect`. Reuses `!allo.stream<T,depth>` (the Vivado emitter already handles it,
`EmitVivadoHLS.cpp:663-667,1719,1785,1909`).

```tablegen
def SPMW_LinkAttr : SPMW_Attr<"Link","link"> {           // one channel family in a topology
  let parameters = (ins StringRefParameter:$port, "AffineMapAttr":$keyMap,   // rank -> channel key
                        StringRefParameter:$dir, "IntegerAttr":$depth); }    // "src"|"sink"
def SPMW_TopologyAttr : SPMW_Attr<"Topology","topology"> {
  let parameters = (ins ArrayRefParameter<"LinkAttr">:$links); }
def SPMW_RoleAttr : SPMW_Attr<"Role","role"> {
  let parameters = (ins "Attribute":$predicate, FlatSymbolRefParameter:$unit); }

def SPMW_MapOp : SPMW_Op<"map", [AttrSizedOperandSegments]> {   // ROLLED, survives to codegen
  let arguments = (ins Variadic<AnyStaticShapeMemRef>:$tensors, DenseI64ArrayAttr:$grid,
                       SPMW_TopologyAttr:$topology, ArrayAttr:$sharding, ArrayAttr:$roles,
                       OptionalAttr<DenseI64ArrayAttr>:$fold, OptionalAttr<DenseI64ArrayAttr>:$unroll);
  let hasVerifier = 1; }   // grid rank == topology dims; roles cover grid; each key: 1 src + 1 sink
def SPMW_RankOp : SPMW_Op<"rank", [Pure]> { let results = (outs Variadic<Index>:$ids); }
```

Role units are `func.func`s tagged `spmw.role`, `spmw.ports` (arg↔port order),
`hls.ap_ctrl="none"` (free-running). End-to-end IR for a 2×2 mesh:

```mlir
func.func @pe_interior(%w:!allo.stream<i8,2>, %e:!allo.stream<i8,2>, %n:..., %s:...)
  attributes {spmw.role="interior", spmw.ports=["west","east","north","south"], hls.ap_ctrl="none"} { ... }
// @pe_load_a, @pe_load_b, @pe_drain similarly

func.func @top(%A:memref<2x2xi8>, %B:memref<2x2xi8>, %C:memref<2x2xi8>) attributes {dataflow} {
  spmw.map (%A,%B,%C) grid=[2,2]
    // peer-form mesh links desugar to keyed channels; e.g. east@(i,j) and west@(i,j+1) share key (0,i,j)
    topology=#spmw.topology<[#spmw.link<port="east",keyMap=affine_map<(i,j)->(0,i,j)>,dir="src",depth=2>,
                             #spmw.link<port="west",keyMap=affine_map<(i,j)->(0,i,j-1)>,dir="sink",depth=2>,
                             ...vertical links...]>
    roles=[#spmw.role<pred="interior",unit=@pe_interior>, #spmw.role<pred="j==0",unit=@pe_load_a>, ...]
    : memref<2x2xi8>, memref<2x2xi8>, memref<2x2xi8>
}
```

Channels are the distinct `keyMap` images; the verifier checks each has exactly one
`src` and one `sink` port (fan-out rules for collectives are an extension).

## 4.3 Compilation flow

```
SPMW frontend (topology + roles + shard)
        │  topology -> symbolic links (affine peer/key maps); roles -> predicate tags
        ▼
spmw.map   (ROLLED: one body per role, grid, sharding, keyed links)
        │  spmw-role-partition:   PID classes = (link-presence) x (predicate tag)
        │                         mesh -> {interior, 4 edges, 4 corners} = 9, ANY size
        │  spmw-resolve-channels: enumerate keys -> FIFO arrays or buffers (§4.5)
        ▼
per-role kernel set  (O(#roles) funcs; NOT O(P0·P1))
        ├── simulator : spmw-unroll OR interpret rolled body, task per PID (§4.7)
        ├── HLS       : one C++ func per role + rolled instantiation loop (§4.4)
        └── RTL glue  : one HLS-exported IP per role + `generate` loop + FIFO/key (§4.4, §4.6)
```

The **role-partition pass** is the only genuinely new analysis, and it is cheap: static
grouping on the rolled body using the topology (link presence per PID) and
`func_predicate_tags` (control-flow class). No unrolling to compute it.

## 4.4 Backend: HLS datapath + structural RTL glue (the mix)

**HLS emits one function per role**; the array is a rolled instantiation loop with FIFOs
as arrays, so HLS schedules **once per role**:

```cpp
void pe_interior(hls::stream<int8>& a_in, hls::stream<int8>& a_out, ...) { /* MAC */ }
void pe_load_a(...); void pe_load_b(...); void pe_drain(...);   // a handful total
void top(...) {
  #pragma HLS dataflow
  hls::stream<int8> fifo_A[Rt][Ct+1]; hls::stream<int8> fifo_B[Rt+1][Ct]; ...
  for (int i=0;i<Rt;i++){ #pragma HLS unroll
    for (int j=0;j<Ct;j++){ #pragma HLS unroll
      pe_interior(fifo_A[i][j], fifo_A[i][j+1], fifo_B[i][j], fifo_B[i+1][j], ...); }}
}
```

Scheduling cost = **O(#roles)**, not O(P0·P1); elaboration still touches P0·P1 instances
but instancing a pre-synthesized module is far cheaper than re-scheduling.

Methods to bypass HLS synthesis-time blowup, by leverage: (1) **role dedup** (#funcs =
#roles, sound only because the topology makes interior-equivalence known by
construction); (2) **rolled dataflow instantiation** (one module, N instances);
(3) **hierarchical IP reuse** (synthesize an L1 tile once, instantiate at L2/L3 — a
256×256 array = one 16×16 tile + a grid of instances); (4) **space↔time folding** (don't
unroll the map loop → time-multiplex; §4.5).

## 4.5 Folding, memory layout, bank conflicts

Folding instantiates `size/F` physical units, each running `F` logical PEs in time — a
map attribute, not a rewrite: `spmw.map(pe, ..., fold={dim: F})`. Two consequences, both
in `spmw-resolve-channels`:

**(a) Channels reclassify to buffers.** For each channel key, after `fold`: if one
physical PE both produces and consumes it in **non-FIFO order** → `memref` **buffer**;
else `!allo.stream`. FFT spatial: all `lane` keys are streams. FFT folded: one PE per
stage random-accesses pairs `(i, i^(1<<s))` → `lane[s]` becomes a buffer. **Forced by
fold; not user-declared.**

**(b) Buffers need conflict-free banking, derived from the topology.** With
`unroll={dim:U}` a buffer sees `2U` accesses/cycle with index set `{i, i^(1<<s)}`. The
access set *is* the topology key function, so banking is derivable and checkable:

| Topology | Per-cycle access set | Conflict-free banking |
|----------|----------------------|-----------------------|
| mesh / affine `(i,j±1)` | contiguous sub-tile | each PE owns a bank = `partition(Block/Cyclic)` + `Layout.Shard` — free |
| butterfly `i^(1<<s)` | `{i, i^(1<<s)}`, per stage | `bank(i)=(i&(banks-1))^(((i>>s)&1)<<(log2 banks-1))` (F2 XOR swizzle) |

The framework **derives and statically verifies** the bank function (injectivity over
each cycle's access set) → conflict-free `banks`-way access at II=1 (`#pragma HLS
array_partition`). Fail → conflict diagnostic + serialize fallback (reported), or the
user supplies a `Layout`. **Concrete new work:** today's `partition()` supports only
Block/Cyclic, so the XOR/butterfly case needs a partition-*function* extension (the real
branch's `f2_layout`). Reuses `partition()`, `Layout`/`Shard`/`Replicate`, `buffer_at`.

## 4.6 RTL glue over HLS IPs (no CIRCT)

For arrays so large that even one Vitis run over P0·P1 *instances* is the bottleneck,
generate the array structure directly and let HLS own only the datapath:

1. **HLS-export one IP per role.** Emit each role as a free-running (`ap_ctrl_none`) HLS
   function, run `csynth`+`export_design` → an RTL IP with the standard `hls::stream`
   FIFO handshake ports (`dout/empty_n/read`, `din/full_n/write`). O(#roles) HLS runs.
2. **Emit a structural Verilog top** by walking `spmw.map`: a `generate` loop per role
   over its PID range instantiating the role IP, and **one FIFO primitive per channel
   key** (`xpm_fifo_axis` or a small `hls_fifo`) wiring producer→consumer. `generate`/
   `genvar` is Verilog-2001 structural syntax — Vivado synthesizes the IP module once
   and elaborates N instances. Backpressure self-synchronizes; no global FSM.
3. **Package** the role IPs + the top as a Vitis RTL kernel (`.xo`) → `v++` → `.xclbin`
   → XRT, so the host call `mod(A,B,C)` is unchanged from the HLS path.

Because `#instances = #grid nodes` and `#FIFOs = #distinct keys` both come from the
topology, the emitter is a direct walk of `spmw.map` — the op *is* the netlist.
**Performance parity:** each PE's RTL *is* HLS output, so per-PE II/throughput/Fmax match
the pure-HLS design; only replication moved out of HLS. Parity needs matched FIFO depths;
avoid hand-writing an unpipelined RTL datapath (hence HLS-per-PE, never hand `hw`/`comb`).
This is strictly simpler than a CIRCT/ExportVerilog path and reuses the HLS datapath you
already trust — the earlier CIRCT discussion is dropped in favor of this direct route.

## 4.7 Fast simulator with performance/latency estimation

Today's simulator fully unrolls (`_inject_omp_parallel_sections`, `simulator.py:783-806`
wraps each per-PE call in an `omp.section` → 1024 sections for 32×32, hence
`OMP_NUM_THREADS=64` and the deadlock-fix machinery). The rolled `spmw.map` removes the
need. Three fidelity tiers, one representation, one scheduler:

- **Tier 0 — Functional (fastest).** JIT one function per role (~9); run each physical PE
  as a **coroutine** with blocking get/put on bounded FIFOs, cooperatively scheduled —
  no OS-thread oversubscription, no thread-count-dependent deadlock. Build O(#roles), run
  O(work). A folded PE loops over its logical PIDs → fewer tasks.
- **Tier 1 — Analytic (no execution, instant).** Treat the rolled graph as SDF:
  closed-form **throughput** (steady-state II via max-cycle-ratio), **latency**
  (fill + steady×iters + drain), **min FIFO depths**, **deadlock-freedom**. Cost
  O(#roles + #edges), array-size-independent — the DSE workhorse.
- **Tier 2 — Cycle-approximate (execution + clock).** Same coroutines with a virtual
  timestamp per token (`get` resolves at `max(ready, consumer_free)`; FIFO stalls
  propagate). Event-driven → O(#tokens). Use when data-dependent control makes Tier-1
  rates inexact.
- **Tier 3 — RTL cosim** = ground truth, slow, sign-off only.

**Latency + area nearly free:** §4.4/§4.6 synthesize one IP per role, whose report gives
II/latency + LUT/FF/DSP/BRAM; the rolled form knows `#instances per role`, so
**total resource = Σ(role_area × instances)** and **total latency = model over the
instanced graph** — an area+latency estimate for a 1024-PE array from ~9 HLS runs,
sweepable across grid/fold without re-synthesizing. Boundaries: Tier 1 exact only for
static-rate SDF; Tier 2 bounded by the latency model; P&R congestion/Fmax at scale still
needs implementation.

---

## 5. What lowers to what (reuse map)

| SPMW surface | Lowers to / reuses |
|--------------|--------------------|
| `@spmw.unit`, `ctx.rank()`, `ctx.port()` | `func.func` role + `spmw.rank` (was `df.get_pid`) |
| `spmw.map(pe, grid, topo)` | `spmw.map` op (rolled) → `spmw-unroll` gives today's `df.kernel` behavior |
| `spmw.Topology` / `mesh` / `link` (peer form; key form for permutations) | `SPMW_TopologyAttr` (affine peer/key maps); one FIFO per edge |
| `@pe.role` / `stream_in(flow=)` | `SPMW_RoleAttr` → generated `meta_if` predicates over `rank` |
| structural nesting (region in unit) | nested `func.func` + parent→child rewiring (`move_stream_to_interface`) |
| `spmw.shard` / `spmw.shared` / `banked` | `allo.grid_map` sharding + `memory.py` `Layout`/`Shard`/`Replicate`/`Memory` |
| `spmw.scatter` / `gather` | collective over chain-style links (the `df.gather/scatter` stubs) |
| `fold` / `unroll` / `Layout.xor_bank` | `spmw-resolve-channels` (stream↔buffer) + `partition`/`buffer_at` |
| `spmw.build(target=)` | simulator / vitis_hls / vitis_rtl |

---

## 6. Implementation plan

Six milestones. Each is independently useful, gated by a concrete test, and leaves the
existing `allo.dataflow` suite green. Each milestone also **migrates a concrete slice of
`tests/dataflow` to the new interface** (§6.7); the goal is to convert the folder's
FPGA-targeted tests. Three examples — the **systolic array, the FFT, and the mini-TPU** —
are **hardware-guaranteed**: validated all the way through Vivado HLS synthesis and
hardware (hw/hw_emu), not just the simulator.

**Scope:** this plan targets the simulator and the Vitis HLS / RTL-glue FPGA path only.
The **AIE-related tests (`tests/dataflow/aie/*`) are out of scope** and are not handled
here — the AIE backend is a separate lowering path and is left untouched.

**Verification ladder** — each migrated test climbs as far as its milestone supports:

| Level | Check | How |
|-------|-------|-----|
| **L0 Build** | frontend → `spmw.map` round-trips | `allo-opt` parses/verifies the IR |
| **L1 Sim-equiv** | **bit-identical** to the original `df` test | `target="simulator"`; original kept as oracle until the twin passes |
| **L2 Csim** | functional match vs numpy | `target="vitis_hls", mode="csim"` |
| **L3 Csynth** | synthesis completes; report captured | `mode="csyn"`; record LUT/FF/DSP/BRAM + latency/II, and synth wall-clock vs the flat `df` baseline |
| **L4 Hw** | functional match on RTL/board | `mode="hw_emu"` (RTL cosim) or `mode="hw"` (on-board via XRT) |

### M1 — Frontend + rolled IR on the simulator (proves the model)

| # | Task | Files | Test gate |
|---|------|-------|-----------|
| 1.1 | `spmw` Python surface (`unit`/`region`/`map`, `Topology`/`mesh`, peer/key links, `ctx.rank/port`) | `allo/spmw.py` (new) | builds AST |
| 1.2 | `spmw` dialect (`spmw.map`, `spmw.rank`, `Topology/Link/Role` attrs, verifier) | `mlir/{include,lib}/allo/Dialect/SPMW/*` + registry + CMake | `allo-opt` round-trips §4.2 IR |
| 1.3 | Frontend → `spmw.map` (one op + one role func per predicate tag; reuse `func_predicate_tags`) | `allo/ir/builder.py` (branch `:2021`) | emits rolled IR for `test_systolic` |
| 1.4 | `spmw-unroll` lowering → per-PID `func.call` | `mlir/lib/Conversion/SPMWUnroll.cpp` | **bit-identical sim** vs `df` |

**Migrates (to L1):** `test_producer_consumer`, `test_df_unit`, `test_stream_of_blocks`,
`test_1D_systolic`, `test_cooperative_gemv`. **Gate:** each SPMW twin simulates to the
same numbers as its `df` original. Deps: 1.1→1.3, 1.2→1.3→1.4.

### M2 — Regularity to HLS (synthesis-time win, no new backend)

| # | Task | Files | Test gate |
|---|------|-------|-----------|
| 2.1 | `spmw-role-partition` (classes = link-presence × predicate tag) | `mlir/lib/Transforms/SPMWRolePartition.cpp` | mesh → 9 roles, any size |
| 2.2 | `spmw-resolve-channels` (streams): distinct keys → per-key FIFO arrays | `mlir/lib/Transforms/SPMWResolveChannels.cpp` | `fifo[i][j]`-shaped arrays |
| 2.3 | HLS role emission + rolled instantiation loop (`ap_ctrl_none`) | `EmitVivadoHLS.cpp` (+ Tapa/Catapult/Intel) | `test_tiled_systolic` HLS has ~9 funcs not P² |

**Migrates (to L2–L4):** `test_systolic` **★ guaranteed (L4)**, `test_weight_stationary_gemm`,
`test_packed_systolic`, `test_systolic_conv` (L4 hw_emu), `test_smith_waterman_systolic`
(L4 hw), `test_tiled_systolic` (L2 + assert role-count), `test_wrap_movement` (L2).
**Gate:** synthesis wall-clock ~flat as the array scales (16×16→64×64) and csim/hw match.
Deps: M1→2.1→2.2→2.3.

### M3 — Hierarchy, collectives, structural RTL glue over HLS IPs

| # | Task | Files | Test gate |
|---|------|-------|-----------|
| 3.1 | Nesting + `shard` + `scatter`/`gather` collectives (fan-out keys) | `allo/spmw.py`, `spmw-resolve-channels` | tiled/hierarchical L1 sim |
| 3.2 | Structural-Verilog emitter (walk `spmw.map` → `generate` loops + one FIFO per key) | `mlir/lib/Translation/EmitStructuralVerilog.cpp` (+ CAPI + binding) | `top.v` elaborates in Vivado |
| 3.3 | Per-role HLS IP export + `.xo`/`v++`/XRT packaging; hierarchical IP reuse | `allo/backend/rtl.py` (new) | `target="vitis_rtl"` builds `.xclbin`; tiled GEMM = 1 tile synth + grid |

**Migrates (to L1–L4):** `test_tiled_gemm`, `test_hierachical`, `test_mlp` (L4 hw),
`test_daisy_chain_gemm` (L4 hw_emu), `test_multi_cache_gemm` **★ collectives + L4 hw_emu**,
`test_unified_systolic` (L4 hw_emu; `flowtag` → Tier-2 sim in M5). **Gate:** structural
`top.v` co-sims vs the M1 oracle on `test_systolic` then `test_multi_cache_gemm`; on-board
via XRT, host call unchanged. Deps: M2→3.1→3.2→3.3.

### M4 — Folding, buffers, bank conflicts, memory hierarchy

| # | Task | Files | Test gate |
|---|------|-------|-----------|
| 4.1 | `fold`/`unroll` map attrs; `spmw.shared`/`banked`/`view` + `space=` → `Memory` | `allo/spmw.py`, `spmw.map` attrs | parses, verifier checks |
| 4.2 | Channel→buffer reclassification | extend `spmw-resolve-channels` | FFT folded: `lane` → buffer |
| 4.3 | XOR/`Layout` banking + partition-function; injectivity verify | `allo/memory.py`, `customize.py::partition` | no bank conflict in HLS report |

**Migrates (to L2–L4):** `test_fft` (new, from `feature/allo-fft`) **★ guaranteed (L4 hw)**,
`test_mini_tpu` (new, §3.4) **★ guaranteed (L4 hw_emu)**, `test_pingpong_gemm` (double-buffer
via `shared(double=True)`). **Gate:** `fft_spatial` and `fft_folded` (§3.5) both
csim-correct; folded HLS report shows conflict-free banked access at II=1; mini-TPU csynth
+ hw_emu match numpy. Deps: M2→4.1→4.2→4.3.

### M5 — Fast simulator with performance estimation

| # | Task | Files | Test gate |
|---|------|-------|-----------|
| 5.1 | Coroutine functional sim on rolled IR (task-per-PID, no full unroll) | `allo/backend/simulator.py` | matches `spmw-unroll`; no `OMP_NUM_THREADS` hack |
| 5.2 | Tier-1 analytic model (SDF throughput/latency/min-depth/deadlock) | `allo/backend/perf.py` (new) | matches cycle counts on `test_systolic` |
| 5.3 | Tier-2 token clock + role characterization (area+latency = Σ role×instances) | `perf.py` + M2/M3 reports | estimate within tolerance vs cosim on `flowtag` FFT |

**Validates against:** the L3/L4 reports produced by the guaranteed set — the analytic
area+latency estimate for the 64×64 systolic array must fall within tolerance of the
actual csynth/cosim numbers. Deps: M1→5.1; M2→5.3.

### M6 — Non-mesh topologies, sparse, robustness (hardening)

Butterfly/bitonic/crossbar/tree generators; topology checks (each key 1 src + 1 sink or a
declared fan-out, depth consistency, unhandled-boundary detection); `spmw.phase()` +
single-writer verification. **Migrates:** `tests/dataflow/sparse/*` (L1 sim). The
`tests/dataflow/aie/*` tests are out of scope (§6 scope note). Deps: M4.

### 6.7 Test migration matrix (all of `tests/dataflow`)

Every existing file gets an SPMW twin; the original is retained as the L1 oracle until the
twin passes, then replaced. ★ = hardware-guaranteed (must reach Vivado HLS + hw/hw_emu).

| Test file | Milestone | SPMW features exercised | Target level |
|-----------|-----------|-------------------------|--------------|
| `test_producer_consumer.py` | M1 | 1D chain, roles (first/last) | L2 |
| `test_df_unit.py` | M1 | unit basics, const arrays, uint, index calc | L2 |
| `test_stream_of_blocks.py` | M1 | block-payload streams | L2 |
| `test_1D_systolic.py` | M1/M2 | 1D/`[2,P0]` mesh, forwarding | L1 |
| `test_cooperative_gemv.py` | M1/M2 | 1D map + reduction | L1 |
| **`test_systolic.py` ★** | **M2** | **mesh, roles, auto-halo, output-stationary** | **L4 hw_emu** |
| `test_weight_stationary_gemm.py` | M2 | mesh, `stationary()`, WS dataflow | L2 |
| `test_packed_systolic.py` | M2 | packed streams over mesh | L2 |
| `test_systolic_conv.py` | M2 | conv systolic mesh | L4 hw_emu |
| `test_smith_waterman_systolic.py` | M2 | mesh, data-dependent max | L4 hw |
| `test_tiled_systolic.py` | M2 | mesh + tiling; **role-count assertion** | L2 |
| `test_wrap_movement.py` | M2 | stream wrap/movement | L2 |
| `test_tiled_gemm.py` | M3 | `shard`, tiling | L1 |
| `test_hierachical.py` | M3 | nested regions, hierarchy | L1 |
| `test_mlp.py` | M3 | multi-layer composition (TPU-lite) | L4 hw |
| `test_daisy_chain_gemm.py` | M3 | chain links, collectives | L4 hw_emu |
| **`test_multi_cache_gemm.py` ★** | **M3** | **multi-link topology, `scatter`/`gather`, packed** | **L4 hw_emu/hw** |
| `test_unified_systolic.py` | M3/M5 | L1/L2/L3, `flowtag` (Tier-2 sim) | L4 hw_emu |
| `test_pingpong_gemm.py` | M4 | `shared(double=True)` ping-pong | L1 |
| **`test_fft.py` ★ (new)** | **M4** | **butterfly topology, `fold`, XOR banking** | **L4 hw** |
| **`test_mini_tpu.py` ★ (new)** | **M4** | **hierarchy, heterogeneous units, shared/banked memory** | **L4 hw_emu** |
| `sparse/*.py` | M6 | sparse systolic | L1 |
| `aie/*.py` | — | **out of scope** (separate AIE backend; not handled here) | — |

### 6.8 Hardware-guaranteed examples (the three that must run Vivado / Vivado HLS)

For **systolic (`test_systolic`), FFT (`test_fft`), and mini-TPU (`test_mini_tpu`)**, a
passing simulator run is not sufficient. Each must, in CI-with-tools (guarded by
`hls.is_available("vitis_hls")`):

1. **L2 csim** vs numpy (`atol` per existing tests);
2. **L3 csynth** completes and the report is archived — capture **LUT/FF/DSP/BRAM**,
   **latency/II**, and the **synthesis wall-clock**; assert wall-clock stays ~flat as the
   grid scales (the O(#roles) claim) by comparing 8×8 vs 16×16 vs 32×32;
3. **L4** — systolic and mini-TPU via `mode="hw_emu"` (RTL cosim), FFT via `mode="hw"`
   (on-board), each matching numpy;
4. **cross-check M5**: the analytic latency/area estimate must match the L3/L4 numbers
   within tolerance — this is how the perf model is validated, and how a regression in the
   synthesis-time win is caught.

These three cover the full feature surface: mesh + roles + auto-halo (systolic),
non-mesh topology + fold + bank-conflict-free memory (FFT), and hierarchy + heterogeneous
units + memory levels (mini-TPU). If all three reach L4, the interface is proven
end-to-end on real tools.

### Critical path

```
M1 ──▶ M2 ──▶ M3   (HLS win, then hierarchy + structural RTL glue)
        │  └──▶ M5  (perf sim reuses M2/M3 role IPs)
        └──▶ M4 ──▶ M6
```

M1→M2 is the highest-value slice: frontend concision **and** the synthesis-time win, no
new backend, no performance risk (HLS datapath preserved). M3 and M4 are independent
branches off M2; M5 reuses the role IPs M2/M3 produce.

**Genuinely new code:** one dialect (M1), three passes (role-partition, resolve-channels,
unroll), one structural-Verilog emitter (M3), a perf model (M5). Everything else reuses
`func_predicate_tags` (`builder.py`), `!allo.stream` HLS emission (`EmitVivadoHLS.cpp`),
`grid_map` sharding (`AlloOps.td`), and `Layout`/`partition`/`buffer_at`.

---

## 7. FFT case-study verdict (validation against `feature/allo-fft`)

The real tuned FFT has three variants (scalar feed-forward, vectorized FFT-256 with F2
banking, folded FFT-256). **SPMW represents all three structurally, and the branch
independently uses the two mechanisms this design centers on** — `fold={dim:F}` dicts and
XOR-swizzle banking (`f2_layout`) — validation, not coincidence.
- *Maps cleanly:* feed-forward = §3.5 spatial; its `get_upper_idx`/`lower=upper^(1<<s)`
  **is** the topology `link`. Folded = §3.5 folded (same `fold` syntax). Its
  `bank(idx)=(idx&(W-1))^(((idx>>s)&1)<<(logW-1))` **is** the §4.5 butterfly banking.
- *SPMW improves:* declarative interconnect (one `link` vs `get_upper_idx` across 3
  bodies); one program vs three variants; banking as a **checked** Layout; role/stage
  specialization vs `ConstExpr` duplication.
- *SPMW only carries (honest):* the vectorized SIMD body, the twiddle DSP opts, and the
  exact `f2_layout` swizzle. Auto-derived banking is a research goal; the practical path
  is a **library** of conflict-free layouts keyed off the topology and checked. Parity
  requires preserving the datapath numerics verbatim.

## 8. Open design questions

- **Link naming.** `ctx.east/west` for meshes, `ctx.port("parent")` for trees/rings —
  keep `port` as the base, directional names as mesh sugar.
- **Teaching path.** Roles first (legible structure) or auto-halo first (shortest)?
  Recommend roles-first, auto-halo as the shortcut.
- **Schedule split.** FIFO depths on links (`depths=`); top-level array partitioning on
  the schedule object.
- **Collectives & fan-out keys.** `scatter`/`gather` need channel keys with one src and
  many sinks (or vice versa) — extend the "1 src + 1 sink" verifier with declared
  fan-out/fan-in rules.
- **Banking beyond F2.** How far to auto-derive conflict-free layouts before falling back
  to the checked-library approach.

--- Original Design Draft End ---
