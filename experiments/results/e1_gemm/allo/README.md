# E1 baseline: Allo's library systolic GEMM (int8 x int8 -> int32) on the U280

Experiment E1, system `Allo`: Allo's own library systolic array
(`allo/library/systolic.py`: `systolic[TyA, TyB, TyC, M, K, N, Mt, Nt]` with the
library schedule `schedule_systolic`), Mt = Nt = S for S in {4, 8, 16, 32}, two
workloads per S -- the tile (M = N = K = S) and the primary workload
M = N = K = 128 -- built through Vitis HLS 2023.2 and Vivado 2023.2 for
`xcu280-fsvh2892-2L-e` at a 3.333 ns (300 MHz) clock. Per (S, workload):
RTL co-simulation cycles with a bit-exact check against numpy (seeds 0/1/2),
and the routed resources / timing of Vitis HLS's out-of-context Vivado
implementation of the exported RTL.

This package was produced by three agent sessions in a row (two were cut off;
their build lanes kept running on the host). Section "Build provenance" says
which session's build every row comes from. Last collection: 2026-09-07 13:34:40
(`build_status.json` has the machine-readable state; rows of builds still in
progress carry status `running` / `timeout` and are refreshed automatically by
`scripts/drv3/e1_allo_finish.sh` while it runs).

## Results (`results.csv`, `results.json`)

| run_id | mode | status | valid | first_out_cyc | completion_cyc | interval_cyc | LUT | FF | DSP | BRAM36/18 | URAM | WNS_ns | csynth_s | synth_s | place_s | route_s | total_s | from |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| allo_S4_tile_cosim | cosim | pass | True | 112 | 136 | 137 |  |  |  |  |  |  | 59.635 |  |  |  | 122.461 | w1 |
| allo_S4_tile_impl | impl | pass |  |  |  |  | 3693 | 4459 | 16 | 0/3 | 0 | 0.625 | 59.635 | 146.0 | 277.0 | 66.0 | 1195.406 | w1 |
| allo_S4_128_cosim | cosim | pass | True | 316471 | 334909 | 334910 |  |  |  |  |  |  | 90.382 |  |  |  | 555.66 | w1 |
| allo_S4_128_impl | impl | pass |  |  |  |  | 4193 | 4808 | 16 | 18/5 | 0 | 0.792 | 90.382 | 185.0 | 241.0 | 52.0 | 1250.752 | w1 |
| allo_S8_tile_cosim | cosim | pass | True | 228 | 306 | 307 |  |  |  |  |  |  | 79.582 |  |  |  | 322.245 | w1 |
| allo_S8_tile_impl | impl | pass |  |  |  |  | 8419 | 8220 | 64 | 0/3 | 0 | 0.598 | 79.582 | 180.0 | 215.0 | 58.0 | 1370.114 | w1 |
| allo_S8_128_cosim | cosim | pass | True | 106807 | 125245 | 125246 |  |  |  |  |  |  | 80.113 |  |  |  | 647.242 | w1 |
| allo_S8_128_impl | impl | pass |  |  |  |  | 10177 | 9301 | 64 | 20/3 | 0 | 0.428 | 80.113 | 183.0 | 225.0 | 64.0 | 1415.816 | w1 |
| allo_S16_tile_cosim | cosim | timeout_cosim |  |  |  |  |  |  |  |  |  |  | 423.269 |  |  |  | 443.363 | w1 |
| allo_S16_tile_impl | impl | not_run_impl |  |  |  |  |  |  |  |  |  |  | 423.269 |  |  |  | 443.363 | w1 |
| allo_S16_128_cosim | cosim | timeout_cosim |  |  |  |  |  |  |  |  |  |  | 484.747 |  |  |  | 511.642 | w1 |
| allo_S16_128_impl | impl | not_run_impl |  |  |  |  |  |  |  |  |  |  | 484.747 |  |  |  | 511.642 | w1 |
| allo_S32_tile_cosim | cosim | timeout |  |  |  |  |  |  |  |  |  |  | 8419.433 |  |  |  |  | w1 |
| allo_S32_tile_impl | impl | timeout |  |  |  |  |  |  |  |  |  |  | 8419.433 |  |  |  |  | w1 |
| allo_S32_128_cosim | cosim | timeout |  |  |  |  |  |  |  |  |  |  | 8083.877 |  |  |  |  | w1 |
| allo_S32_128_impl | impl | timeout |  |  |  |  |  |  |  |  |  |  | 8083.877 |  |  |  |  | w1 |

Row conventions:

* Two rows per build: `cosim` (RTL co-simulation: cycles and the numeric
  check) and `impl` (Vivado implementation of the exported RTL, run by Vitis
  HLS's `export_design -flow impl`, out of context). `pnr_ooc` rows do not
  occur: Allo's impl flow routed for every finished build, so the separate
  out-of-context P&R script (`/scratch/hc676/autosa_pnr_ooc.sh`) was not
  needed.
* `status`: `pass`; `functional_fail` (RTL output differs from numpy or the
  cosim reports Fail); `timing_fail` (routed WNS < 0); `route_fail` (nets
  with routing errors); `tool_fail` (a tool stage errored, see
  `failure_reason`); `running` (build in progress at collection time);
  `timeout` (still in progress after the ~3 h budget for the 32x32 builds,
  2026-09-07 01:45; left running, refreshed when it ends); `pending` (not
  started: queued behind the two heavy-job slots); `interrupted`.
  A row that is not `pass` has its values empty or partial and the reason in
  `failure_reason`; nothing is estimated.
* `validation_pass` (cosim rows): true only when the cosim report says
  `Pass` and the testbench, re-run by cosim against the RTL, prints
  `E1 TB ALL PASS seeds=3` with `max_abs_err=0` for all three seeds
  (`tb_seed_results`).
* Cycles (cosim rows, see "How the cycles were measured"):
  `completion_cycles` = ap_start-to-ap_done latency of a warm launch
  (Vitis's transaction 1); `steady_interval_cycles` = launch-to-launch
  interval of back-to-back launches (transaction 1's interval);
  `first_output_cycles` = cycle of the first C write beat on `m_axi_gmem2`
  counted from that launch's ap_start (from a VCD of the same cosim
  snapshot). Cold launch 0 (which Vitis counts from its harness's AXI-lite
  register programming) and the report's min/avg/max are in the extra columns.
* Resources / timing (impl rows): post-route, from Vivado's own reports of the
  `impl_1` run (`bd_0_wrapper_utilization_placed.rpt`,
  `bd_0_wrapper_timing_summary_routed.rpt`, `bd_0_wrapper_route_status.rpt`;
  the equivalent `gemm_*_routed.rpt` report-level-2 files are in the same
  report directory). `bram_18k_equiv` = 2 x `bram_36k` + `bram_18k`;
  `unrouted` = nets with routing errors (0 = fully routed); `wns_ns` /
  `tns_ns` at the 3.333 ns constraint (positive WNS = timing met at 300 MHz;
  `cp_post_impl_ns` is the achieved period Vitis HLS reports).
* Wall times: `hls_wall_s` = `csynth_design`; `vivado_synth_s` = the
  `synth_design` elapsed time of the IP's synthesis run
  (`bd_0_hls_inst_0_synth_1`; the block-design wrapper's own synth is
  `wrapper_synth_s`); `place_s` / `route_s` = `place_design` /
  `route_design` elapsed times from `impl_1/runme.log` (`opt_s`,
  `phys_opt_s` extra); `total_wall_s` = generation + csim + csynth + cosim
  (cosim row) or generation + csim + csynth + the whole `export_design -flow
  impl` stage (impl row); `run_total_wall_s` = the whole build. The host was
  shared and heavily loaded (load average 14-67 during these runs), so every
  wall time is an upper bound.
* `axi_masters`: the external interface, from the csynth report and the RTL
  (see "External interface").
* Extra columns after `failure_reason`: provenance (`source_agent`,
  `build_dir`, `build_state`, `collected_at`), every stage time, the cosim
  report's min/avg/max, the cold-launch and VCD numbers, Vitis HLS's own
  estimates (`hls_est_*`), `cp_post_synth_ns` / `cp_post_impl_ns`, SRL/CLB,
  `pe_count`, `tiles_folded`, `output_stationary`, `tb_seed_results`,
  `axi_data_widths`.
* `reports/<run_id>/`: cosim rows carry the csynth report (`gemm_csynth.rpt`,
  `csynth.rpt`, `csynth.xml`), the cosim report and log, the per-transaction
  cycle file (`gemm.performance.result.transaction.xml`), the VCD analysis
  (`firstout.json`, `firstout.log`), the full `hls.log`, the generated
  `kernel.cpp` (and Allo's verbatim `kernel_allo.cpp`, the un-postprocessed
  `kernel_plain.cpp`), `tb.cpp`, `run_e1.tcl`, `stages.txt`, `gen.log`,
  `e1_gen_info.json`; impl rows carry Vitis HLS's export reports
  (`gemm_export.rpt`, `export_impl.rpt/.xml`, `export_syn.rpt/.xml`), the
  Vivado `impl_1` reports (utilization placed, timing summary routed, route
  status, DRC, methodology, power, clock utilization, control sets, io, bus
  skew), the report-level-2 `gemm_*_synth/routed.rpt` files, the
  `synth_1` / `bd_0_hls_inst_0_synth_1` / `impl_1` `runme.log`s and the
  export's `vivado.log`.

## Build provenance and state at collection time

| S | workload | canonical build | agent | state | tool processes now |
|---|---|---|---|---|---|
| 4 | tile | `/scratch/hc676/e1_allo_w1/S4_tile` | w1 | complete | - |
| 4 | 128 | `/scratch/hc676/e1_allo_w1/S4_128` | w1 | complete | - |
| 8 | tile | `/scratch/hc676/e1_allo_w1/S8_tile` | w1 | complete | - |
| 8 | 128 | `/scratch/hc676/e1_allo_w1/S8_128` | w1 | complete | - |
| 16 | tile | `/scratch/hc676/e1_allo_w1/S16_tile` | w1 | complete_with_failures | - |
| 16 | tile | `/scratch/hc676/e1_allo_v2_S16_tile` | v2 | interrupted (not used) | - |
| 16 | 128 | `/scratch/hc676/e1_allo_w1/S16_128` | w1 | complete_with_failures | - |
| 32 | tile | `/scratch/hc676/e1_allo_w1/S32_tile` | w1 | running | 260126 vitis_hls, 260170 vitis_hls |
| 32 | 128 | `/scratch/hc676/e1_allo_w1/S32_128` | w1 | running | 241336 vitis_hls, 241380 vitis_hls |
| 32 | 128 | `/scratch/hc676/e1_allo_v2_S32_128` | v2 | aborted (not used) | - |

Second complete build of the same design (not used for `results.csv`, kept as a cross-check):

| S | workload | agent | finished | cosim | valid | completion | interval | first out | impl | LUT | FF | DSP | BRAM36/18 | URAM | WNS | reports |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | tile | v2 | 2026-09-06T21:15:52 | pass | True | 136 | 137 | 112 | pass | 3693 | 4459 | 16 | 0/3 | 0 | 0.625 | `reports/_duplicates/v2/reports/` |
| 4 | 128 | v2 | 2026-09-06T21:44:57 | pass | True | 334909 | 334910 | 316471 | pass | 4193 | 4808 | 16 | 18/5 | 0 | 0.792 | `reports/_duplicates/v2/reports/` |

Lane queues of the two earlier drivers at collection time (they keep running after their sessions ended):

* w1 lane1: builds ['4_tile', '32_128', '8_tile', '8_128']; now building nothing; remaining []; finished
* w1 lane2: builds ['32_tile', '16_128', '16_tile', '4_128']; now building nothing; remaining []; finished
* v2 laneC: builds ['4_128', '8_tile', '8_128', '16_tile']; now building nothing; remaining []; finished

E1 tool processes on the host at collection time: 241336 vitis_hls (/scratch/hc676/e1_allo_w1/S32_128), 241380 vitis_hls (/scratch/hc676/e1_allo_w1/S32_128/out.prj/solution1), 260126 vitis_hls (/scratch/hc676/e1_allo_w1/S32_tile), 260170 vitis_hls (/scratch/hc676/e1_allo_w1/S32_tile/out.prj/solution1)

`reports/S4_tile_prev/` is agent 1's very first S4_tile build (20:34, before
the cosim `depth` fix, see "History"): its cosim failed for that reason and
its impl numbers equal the canonical S4_tile build's (3693 LUT / 4459 FF /
16 DSP / 3 RAMB18 / WNS +0.625 ns), as expected since `depth` is not
synthesised. `results_prev_agent_S4_tile_nodepth.csv` is agent 2's record of
it.

## What was built

For each S and workload the design is

```python
def gemm(A: int8[M, K], B: int8[K, N], C: int32[M, N]):
    systolic[int8, int8, int32, M, K, N, S, S, "gemm"](A, B, C)

s = allo.customize(gemm)
s.compose(systolic, instantiate=[int8, int8, int32, M, K, N, S, S], id="gemm")
s.build(target="vitis_hls", mode="csyn", project=OUTDIR)   # Allo's Vitis project + kernel.cpp
```

`compose` applies the library's own `schedule_systolic` (`allo/library/systolic.py`):
partition `local_C` on both dims, `local_A` on dim 1, `local_B` on dim 2;
pipeline the `load_A_tile` / `load_B_tile` / `store_C_tile` loops; fuse the
`mi`/`ni` tile loops; pipeline the PE's `reduction` k loop (II = 1); `unfold`
the S x S PE grid into S*S spatial PE functions; and `to` A and B through
FIFOs between neighbouring PEs (`A_fifo` along axis 1, `B_fifo` along axis 0,
depth S + 1). Nothing was tuned; the only edit to Allo's emitted `kernel.cpp`
is the simulation-only `depth=` attribute on the three `m_axi` pragmas (see
"Cosim"); `kernel_allo.cpp` in every report directory is Allo's verbatim
output. The wrapper is named `gemm` rather than making `systolic` the top
because Allo's Vitis post-processing keys on `line.startswith("void " + top)`
and a top named `systolic` also mangles `systolic_tile`.

### Output-stationary PE (confirmed on every build)

Every one of the S*S unfolded PE functions `PE_kernel_gemm_<i>_<j>` is, in
Allo's emitted C++ (S4_128 shown; only K changes):

```c++
int32_t v = 0;
l_reduction_k: for (int k = 0; k < 128; k++) {      // #pragma HLS pipeline II=1
  int8_t a = A_in.read();  int8_t b = B_in.read();
  v = (int32_t)((ap_int<33>)v + (ap_int<33>)((int16_t)a * (int16_t)b));
  A_out.write(a);  B_out.write(b);
}
C[i][j] = v;                                         // the only store to C, after the k loop
```

The generator checks this on the emitted C++ of every build (exactly one
store to the C argument, at function level after a pipelined k loop of trip
count K; S*S PE functions; the fused tile loop's trip count) and records it
in `e1_gen_info.json` (`output_stationary: true`, `pe_count`,
`tile_loops.outer_trip`). The accumulator stays in the PE and A and B flow
through: an output-stationary array with one DSP MAC per PE
(`mac_muladd_8s_8s_<w>s_<w>_4_1`, w = 16 + log2(K) bits: 18 for K = 4, 23
for K = 128).

### External interface (identical for every build)

* Top `gemm(int8_t *A, int8_t *B, int32_t *C)`, `ap_ctrl_hs`, one
  `s_axi_control` AXI4-Lite slave (32-bit data, 6-bit address: the three
  64-bit pointers).
* Three AXI4 masters, one per argument, on separate bundles `m_axi_gmem0`
  (A), `m_axi_gmem1` (B), `m_axi_gmem2` (C): 64-bit addresses, burst length
  16, 16 outstanding reads and writes, no widening (`Max Widen Bitwidth 0`).
  The csynth report's data width (SW -> HW) is 8 -> 8 bits for A and B and
  32 -> 32 for C; the physical AXI data buses are all 32 bits wide
  (`C_M_AXI_GMEM{0,1,2}_DATA_WIDTH = 32` in the RTL), so one int8 element
  travels per 32-bit beat on gmem0/gmem1 (byte strobes) and one int32 per
  beat on gmem2. Allo's `vitis_hls` target emits plain `m_axi ...
  offset=slave` pragmas and the project uses the Vivado IP flow
  (`-flow_target vivado`), so Vitis HLS does not widen the ports.
* The top is sequential, not a dataflow region: `load_buf0` (A ->
  on-chip `buf0[M][K]`), `load_buf1` (B -> `buf1[K][N]`), `systolic_gemm`,
  then `store_res2` (`buf2[M][N]` -> C). The whole M x K / K x N / M x N
  arrays are buffered on chip; for the 128 workload that is 16 KiB + 16 KiB
  + 64 KiB and the AXI transfers alone are 3 x 16384 beats. That is why the
  128-workload latency is dominated by the loads/stores and by the per-tile
  A/B tile loads (128 + 128 cycles per tile) rather than by the S x S array.

### Temporal tile folding

`systolic_gemm` keeps a single S x S array and iterates the fused tile loop
`l_outer_tile_mi_ni_fused` over (M/S) x (N/S) tiles one after another (load
the A tile only when `ni == 0`, load the B tile, run `systolic_tile_gemm` --
a dataflow region: an edge-feeding `data_load` loop, the S*S PEs, a
`data_drain` loop -- then store the C tile). The trip count is checked
against (M/S)*(N/S) by the generator:

| S | tile workload (M=N=K=S) | 128 workload: tiles folded temporally | PE k-loop trip (K) | PEs | emitted kernel.cpp |
|---|---|---|---|---|---|
| 4  | 1 tile | 1024 | 4 / 128 | 16   | 34 KB / 35 KB |
| 8  | 1 tile | 256  | 8 / 128 | 64   | 126 KB / 127 KB |
| 16 | 1 tile | 64   | 16 / 128 | 256 | 507 KB / 509 KB |
| 32 | 1 tile | 16   | 32 / 128 | 1024 | 2.1 MB / 2.1 MB |

(Generation-only check `scripts/drv2/e1_allo_gencheck.sh`,
`/scratch/hc676/e1_allo_v2_gencheck/`; codegen takes 3-15 s for S <= 16 and
~100-140 s for S = 32.)

## Flow and exact commands

Environment on `brg-zhang-xcel`:

```bash
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh
```

Per build (`scripts/drv_w1/e1_allo_run.sh S W` for agent 1's tree,
`scripts/drv2/e1_allo_run.sh S W [OUTDIR]` for agent 2's; the two are the
same flow with different output roots):

1. `python3 e1_allo_gen.py S W OUTDIR` (W = `tile` | `128`): the Allo codegen
   above; the structural checks; the inputs for seeds 0, 1, 2
   (`numpy.random.default_rng(seed).integers(-128, 128)` int8, with rows /
   columns forced to -128 and 127 so every build exercises -128*-128,
   127*-128 and 127*127 and K-long runs of each; the int32 reference is
   computed in int64 and checked not to overflow); a C++ testbench that reads
   the `.bin` vectors, poisons C, runs the kernel once per seed and exits
   non-zero on any mismatch (`E1 TB ALL PASS seeds=3` on success); and
   `run_e1.tcl`. The two drivers' generators differ only in one assertion's
   wording (`diff scripts/drv_w1/e1_allo_gen.py scripts/drv2/e1_allo_gen.py`);
   every `run_e1.tcl` is byte-identical.
2. `vitis_hls -f run_e1.tcl` (in OUTDIR): Allo's own `codegen_tcl` template
   (project `out.prj`, `-flow_target vivado`, `set_top gemm`, part,
   `create_clock -period 3.333`, `config_export -vivado_report_level 2 -rtl
   verilog`) plus the four stages Allo's mode string selects, each timed
   (`E1_STAGE <name> START/END/FAILED ... ELAPSED_S`): `csim_design -O`,
   `csynth_design`, `cosim_design -rtl verilog -trace_level none` (xsim),
   `export_design -flow impl -rtl verilog -format ip_catalog`.
   Allo's `HLSModule` cannot run these itself on this host: its `vitis_hls`
   platform accepts only `csim/csyn/sw_emu/hw_emu/hw` and its `vivado_hls`
   platform (which has `impl`) needs a `vivado_hls` binary that 2023.2 no
   longer ships, so the driver runs the same template through `vitis_hls`
   directly.
3. `export_design -flow impl` is Vitis HLS's standard Vivado run
   (`run_vivado.tcl`): package the IP, put it in a block design, synthesise
   it **out of context** (`synth_design -mode out_of_context`, directive
   `sdx_optimization_effort_high`), then `opt_design`, `place_design`,
   `phys_opt_design`, `route_design` in `impl_1` with no I/O buffers and the
   3.333 ns `ap_clk` constraint -- the "Allo impl mode if it routes" path of
   the task, comparable to the AutoSA OOC flow of this evaluation (no pins,
   no platform shell, not a `v++` link).
4. First output (agent 1's `scripts/drv_w1/e1_allo_firstout.py BUILD_DIR`,
   run by `e1_allo_postproc.sh` / `scripts/drv3/e1_allo_finish.sh` after a
   cosim finishes): see "How the cycles were measured".
5. Lanes (two heavy tool jobs at most for this package on the shared host):
   agent 1 -- `setsid nohup bash e1_allo_lane.sh lane1 4_tile 32_128 8_tile
   8_128 ...` and `... lane2 32_tile 16_128 16_tile 4_128`; agent 2 --
   `e1_allo_lane3.sh laneC 4_128 8_tile 8_128 16_tile` (a fallback lane that
   drops a design the other driver finished and defers one it is running).
6. Collection: `python3 scripts/drv3/e1_allo_final.py [--deadline EPOCH]`
   (imports agent 1's parsers `scripts/drv_w1/e1_allo_collect.py`; picks the
   canonical build per design; writes `results.csv/json/md`, `README.md`,
   `build_status.json`, `reports/`). `scripts/drv3/e1_allo_finish.sh` re-runs
   it every 10 min until every design is complete and measured.

## How the cycles were measured

* **Cosim frame.** `cosim_design` runs the C testbench once to record the
  vectors, simulates the RTL (`apatb_gemm_top`, Vitis's AXI models) for the
  three seed launches back to back, and re-runs the testbench's checks on the
  RTL outputs. Its per-transaction file
  (`gemm.performance.result.transaction.xml`) gives latency and interval per
  launch. Transaction 0 is cold: Vitis counts it from its harness's AXI-lite
  offset programming, before `ap_start`, so it is longer (S4_tile 167 vs 136
  cycles). Transactions 1 and 2 are warm and identical;
  `completion_cycles` / `steady_interval_cycles` are transaction 1's latency
  / interval (the report's min); the report's min/avg/max and the cold
  numbers are in the extra columns.
* **First output.** Vitis reports no port-level timing, so
  `e1_allo_firstout.py` re-elaborates the *same* compiled cosim snapshot
  (`out.prj/solution1/sim/verilog/xsim.dir/gemm`, same xelab line as Vitis's
  `run_xsim.sh` plus `-debug typical`) and re-runs it under xsim with a VCD
  of `ap_clk / ap_rst_n / ap_start / ap_done`, the `gmem2` write handshake
  and the `gmem0` read handshake, then counts on `ap_clk` (sampled on the
  falling edge, i.e. what the next rising edge samples): start_k = first
  cycle with `ap_start` after reset / after the previous `ap_done`; done_k;
  first_write_k = first cycle with `gmem2_WVALID & WREADY`.
  `first_output_cycles` = first_write - start of launch 1 (warm);
  `vcd_latency_warm` = done - start in the same frame. The script accepts a
  result only if the VCD's launch-to-launch interval reproduces the cosim
  report's warm interval (within 1 cycle) and the report's warm latency
  exceeds the VCD's by 0..6 handshake cycles (`frame_checks` in
  `firstout.json`); otherwise `first_output_cycles` is left empty with the
  reason. For S4_tile: VCD latency 133 vs report 136 (3 cycles), intervals
  137 = 137.
* For this design the first C beat cannot precede the end of `systolic_gemm`
  (the top is sequential and `store_res2` starts after it), so
  `first_output_cycles` is close to `completion_cycles` minus the M*N-beat
  store: the array itself is output-stationary and the C tile is only
  written back after the whole GEMM.

## Tool versions, host, sources

* Vitis HLS 2023.2 (SW Build 4023990 on Oct 11 2023), Vivado v2023.2
  (64-bit), part `xcu280-fsvh2892-2L-e`, clock 3.333 ns.
* Host `brg-zhang-xcel.ece.cornell.edu`: 2 x Intel Xeon Silver 4214
  (48 threads), 219 GB, shared with other users and with the other packages
  of this evaluation (load average 14-67 during these runs).
* Allo commit `d7ce0da49d0da574a36b51507a1253ce11dd6bec`
  (`../../sources/commit.txt`). The checkout used, `/scratch/hc676/allo`, is
  an rsync of the worktree at that commit: `allo/library/systolic.py`,
  `allo/backend/hls.py`, `allo/customize.py`, `allo/passes.py`,
  `allo/ir/transform.py` and `mlir/lib/Translation/EmitVivadoHLS.cpp` have
  the md5 of the blobs of that commit (verified by agent 2 and again by agent
  3 against a clone that has the commit; the server clone's own `.git` HEAD
  reads `f436658` because the files were synced over an older clone). Python
  3.12.13, numpy 2.5.1 (`/scratch/hc676/allo-agent/bin/python3`).

## History and caveats

* **Three sessions.** Agent 1 (`scripts/drv_prev/` then `scripts/drv_w1/`,
  builds `/scratch/hc676/e1_allo_w1/S<S>_<W>`) and agent 2
  (`scripts/drv2/`, builds `/scratch/hc676/e1_allo_v2_S<S>_<W>`) ran the
  same package concurrently, discovered each other through the shared build
  paths (`/scratch/hc676/e1_allo_S*`, whose 20:50 runs are agent 2's first
  attempt: every one failed in generation, `gen rc=1`, because of the
  `code.count(" depth=") == 3` assertion that also counted the PE stream
  pragmas -- fixed in both generators), and coordinated through notes
  (`COORDINATION_w1.txt`, `NOTE_NAMESPACE.txt`, in `scripts/`). Both were
  cut off; their lanes kept running. Agent 3 (`scripts/drv3/`) killed nothing,
  built nothing twice, chose the canonical build per design (earliest
  complete build; agent 1's build wins a tie because it carries the VCD
  first-output measurement), collected everything from the reports and left
  `e1_allo_finish.sh` running to refresh this package as the remaining builds
  end.
* **Two heavy jobs.** Both 32x32 builds (agent 1's lane1 / lane2) held the
  package's two heavy-job slots from ~21:00; the S8 and S16 builds are queued
  behind them in those lanes (lane1: 8_tile, 8_128 after 32_128; lane2:
  16_128, 16_tile, then a duplicate 4_128 after 32_tile; agent 2's laneC
  takes 8_tile / 8_128 / 16_tile only if a slot frees up first). A 32x32
  build still in progress after the ~3 h budget (2026-09-07 01:45) is recorded as
  `timeout` with the stages it finished; it is not killed (it was started by
  an earlier session) and its row is replaced by the real result when it
  ends.
* **Cosim depth.** Vitis HLS refuses to co-simulate an `m_axi` pointer port
  without a `depth` ("A depth specification is required for MAXI interface
  port 'gmem0' for cosimulation"), and Allo emits none because its own path
  is `v++` `hw_emu`/`hw`. `depth` is a simulation-only attribute, so the
  generator adds `depth=<element count>` (A: M*K, B: K*N, C: M*N) to the
  three `m_axi` pragmas of a copy; nothing else in the kernel is edited.
* **Duplicates.** S4_tile was completed by both agents (identical impl
  numbers, identical warm cosim cycles; see the cross-check table above);
  agent 2's duplicate S32_128 was stopped by agent 2 seconds after it
  started (`ABORTED` in `/scratch/hc676/e1_allo_v2_S32_128/stages.txt`).
* **Cosim latency spread.** The cosim report's min/avg/max over the three
  seed launches differ only because launch 0 is cold (see above); the two
  warm launches always agree, and the design is data-independent.
* **AXI width.** The csynth report's "8 -> 8" for A/B is the HLS data width;
  the AXI4 bus is 32 bits (one int8 per beat), so the 128-workload loads are
  16384 beats each -- the interface, not the array, bounds the 128
  workload's latency for every S.
* The `impl` rows are Vitis HLS's out-of-context Vivado run of the packaged
  IP (no pins, no shell), the same kind of number as the AutoSA OOC P&R rows
  of this evaluation, not a `v++` link into the U280 platform.
* Vitis HLS's `gemm_export.rpt` reports the same LUT/FF/DSP as the Vivado
  utilization report and counts BRAM in 18k blocks (`bram_18k_equiv`).

## Notes added by other sessions

### w1_status_and_s32_retry.txt

E1-Allo COORDINATION NOTE from driver "w1" (/scratch/hc676/e1_allo_w1, scripts in e1_allo_w1/drv), written 2026-09-06T21:20:11

Two agent instances are running the same E1 Allo package on this host: w1 (this one) and v2 (/scratch/hc676/e1_allo_drv2,
builds in /scratch/hc676/e1_allo_v2_*). Both deliver to /scratch/hc676/spmw_eval_remaining_2026-09-06/e1_gemm/allo/.

w1 status (S_workload: state):
  4_tile : DONE 21:15:48 (cosim pass, warm latency 136, first output 112, impl pass 3693/4459/16/3 BRAM18k/0 URAM, WNS +0.625)
  32_128 : RUNNING lane1, started 21:15:48 (gen ~2 min, then csynth/cosim/export)   <- v2 laneD started the same build at 21:16:42
  32_tile: RUNNING lane2, started 20:57:58 (in csynth)
  queued : lane1 -> 8_tile, 8_128 ; lane2 -> 16_128, 16_tile, 4_128
w1 run ids in the package: allo_S<S>_<W>_cosim / allo_S<S>_<W>_impl (reports/allo_*), rows in results.csv/results.json,
README.md; w1 will also leave copies named results_w1.csv / README_w1.md so nothing is lost if results.csv is overwritten.

Proposal to avoid building everything twice (each of us has 2 lanes; the box also runs other users jobs, load 20-45):
  - do not start a build the other driver has already FINISHED with rc=0 (stages.txt has E1_RUN_DONE rc=0 and both
    "E1_STAGE cosim END" and "E1_STAGE export_impl END"); its reports can be parsed by either collector (same layout).
  - a build the other driver has STARTED (stages.txt exists, no E1_RUN_DONE) is best left to it; re-queue it last and
    build it only if it has not finished by then.
  - w1 keeps 32_128 (started first) and 32_tile. If v2 drops its 32_128, both 32x32 builds finish ~2x sooner.
w1 never deletes or renames anything outside /scratch/hc676/e1_allo_w1 and reports/allo_* .
2026-09-07 09:05 both S32 builds (tile and 128) stopped after 12 h: Vitis
had passed C simulation on all three seeds and then spent 10 h inside
"Generating RTL test bench" at 98% CPU and 13.5 GB, writing nothing since
23:34. They were holding the lanes queue, which still had 8_tile, 8_128,
16_tile and 16_128 behind them. Record S=32 as a timeout with that reason.

--- w1 update 2026-09-07 09:10 (in reply to the 09:05 note and NOTE_DRV3) ---
Agreed on the S=32 diagnosis, and thank you for stopping them: I confirmed from the projects themselves that neither
wrote a single file after 23:34 / 23:45 (find -newermt 2026-09-07 = 0 files) while burning CPU inside
"Generating RTL test bench". Cause recorded as a cosim testbench-generation hang, not a design failure.

What w1 is doing now, since the two builds passed csynth (8419 s / 8084 s) and their synthesis output is intact
(out.prj/solution1/syn: 1034 reports, ~2070 .v):
  * e1_allo_retry.sh re-runs ONE stage on the existing project, so no csynth is repeated;
  * retryA: 32_tile export_design -flow impl (cap 8 h), then 32_tile cosim with config_cosim
    -disable_deadlock_detection (cap 2 h);
  * retryB: the same for 32_128.
  The deadlock-detection flag is the fix for the hang: for a dataflow region Vitis puts a per-process/per-FIFO
  deadlock monitor in the generated testbench, which is what does not finish at 1024 PEs. It changes the
  simulation harness only, never the RTL, and it is recorded in the README as a deviation.
  Both retry lanes wait for a free slot (they count this driver build dirs that have a vitis_hls/vivado process),
  so w1 still uses at most 2 heavy jobs in total, lanes and retries together. Timeouts are hard (timeout(1)),
  so nothing can hold the queue for 12 h again.
  Lanes now: 8_tile (cosim), 16_128 (csynth), then 8_128 and 16_tile.

Collector: w1 updated /scratch/hc676/e1_allo_w1/drv/e1_allo_collect.py at 09:08 so that stage markers are read from
hls.log too (a killed run never copies them into stages.txt -- without this the 32x32 rows say the csynth was
interrupted, when in fact csynth finished and cosim hung). It also reports statuses timeout_cosim / not_run_impl /
interrupted_csynth with the cause, uses the LAST attempt of a re-run stage, and picks up hls_retry_*.log. If drv3
imports the collector from /scratch/hc676/e1_allo_w1/drv/ these fixes are already live; if it imports a copy under
the package scripts/ directory, please re-copy it.

Package writing: drv3 can keep being the writer while builds are in flight. When w1 finishes its last build it will
write its own results.csv / results.json / README.md / reports/allo_* once, from its own collector, as its task
requires -- it will create /scratch/hc676/e1_allo_drv3/STOP_FINISHER first, wait for the finisher to exit, and
leave build_status.json and anything drv3 wrote that it does not itself produce untouched.
