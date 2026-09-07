# E1 baseline: AutoSA int8 output-stationary systolic GEMM on the U280 (4x4, 8x8, 16x16, 32x32)

Assembled 2026-09-07 13:31:27 EDT on brg-zhang-xcel by two agents in sequence (the first one's process died at about 22:45 with its server jobs still running; the second one inspected, reused and completed them; provenance is recorded per row). Work directory `/scratch/hc676/e1_autosa_2026-09-06` (gen/, hls/, hlsnp/, pnr/, np/, logs/, scripts/). All tool runs on a shared, loaded server, at most two heavy tool jobs at a time (see `logs/lanes.log`, reproduced at the end).

## What was measured

AutoSA's output-stationary int8 x int8 -> int32 matrix-multiply systolic array (`space_time[3]`, i and j as the space loops, k in time) at 4x4, 8x8, 16x16 and 32x32 PEs, part xcu280-fsvh2892-2L-e, clock 3.333 ns (300 MHz), Vitis HLS / Vivado 2023.2. `data_t = char`, `acc_t = int`. Two rows per array size:

* `cosim`: Vitis HLS `csynth_design` then `cosim_design -rtl verilog` (xsim) of `kernel0` with a numpy-driven testbench: three launches with the operands of numpy seeds 0, 1, 2 (uniform int8 in [-128, 127] with the extrema pinned, see below) and an exact compare of every int32 output against the numpy reference in the RTL post-check; then a re-run of the same xsim snapshot with a VCD for the cycle numbers (method below).
* `pnr_ooc`: Vivado out-of-context `synth_design`, `opt_design`, `place_design`, `phys_opt_design`, `route_design` of `kernel0` on the RTL csynth wrote inside the cosim project (the same RTL the cosim ran). OOC because kernel0 has 1351 top-level port bits, more than the package has pins; no I/O buffers, no platform shell, no AXI interconnect, no harness logic of any kind.

**Problem sizes and what a tile is.** AutoSA's kernel is a tiled loop nest: `array_part[S,S,16]` makes the SxS array process one SxSx16 tile at a time and iterate over the tiles of the problem inside one kernel launch. The 4x4, 8x8 and 16x16 arrays run the 16x16x16 problem (`kernel.h`: I = J = K = 16): 16, 4 and 1 tiles per launch. A 32x32 output-stationary array needs at least 32x32 outputs: on the 16x16x16 problem AutoSA silently clamps `array_part[32,32,16]` to the loop bounds and emits the 16x16 design again (the previous agent's evidence, `reports/autosa_mm32i8z_unsupported/evidence.txt`; those rows are kept in `results_all_variants.csv` as `unsupported`), so the 32x32 array is measured on the 32x32x16 problem (I = J = 32, K = 16, one tile per launch; design `mm32i8z32`). The `workload` column carries the problem of each row; the K tile is 16 everywhere, so one PE does 16 MACs per tile at every size. The sibling SPMW/Allo packages define their launch as one SxS tile, so their 4x4/8x8 rows are one tile where AutoSA's are 16 and 4 tiles; the per-tile interval column is the like-for-like number there.

## Two kernel forms (why there are `z` designs)

The `mm16i8` input program the package started from (`/scratch/hc676/autosa/work/mm16i8/kernel.c`) has no `C[i][j] = 0` inside the scop. AutoSA implements exactly that: the PE accumulator `local_C` is never reset, so in an array that is time-multiplexed over several tiles every tile after the first starts from the previous tile's sum, and in a single-tile launch the result depends on the accumulator register's reset value. The previous agent found this with AutoSA's own testbench (4x4: 240 = 256 - 16 wrong outputs, 8x8: 192 = 256 - 64, 16x16: passes by the reset value while the C model prints `Failed with 256 errors!`) and generated the same designs from `kernel.c` with one line added, `C[i][j] = 0;` inside the scop (design names ending in `z`, variant `cinit`). That is the form in AutoSA's documentation (`docs/tutorials/matrix_multiplication.rst`) and in `autosa_tests/large/mm*`, `mm_hbm`, `mm_getting_started`; note that this checkout's `autosa_tests/mm/kernel.c` has the line commented out. Problem size, types, `--sa-sizes`, `simd_info.json` and the tool recipes are identical; AutoSA's generated PE gains `local_C[0][0] = 0;` before the k loop. `results.csv` holds the `cinit` designs (the ones that validate); `results_all_variants.csv` adds the as-specified (`asis`) rows from the previous agent's runs and the clamped 32x32 evidence.

## Numeric validation

`np/mk_tb_data.py` (numpy 2.0.2) draws A (I x K) and B (J x K) as uniform int8 in [-128, 127] with `numpy.random.default_rng(seed)`, seeds 0, 1, 2, and pins rows 0..3 of A and B: row 0 all -128 in both (C[0][0] = K x 16384 = 262144, the largest positive sum), row 1 A = 127 / B = -128 (C[1][1] = -K x 16256 = -260096, the most negative), row 2 both 127 (K x 16129), row 3 alternating -128/127 against 127/-128. The reference is `A.astype(int32) @ B.astype(int32).T` (exact, |C| <= 262144). The generated testbench (`np/mk_tb.py`, `kernel_host.cpp` in each cosim report directory) keeps AutoSA's host-buffer handling (byte copies into the packed 128-bit A/B words), poisons the C buffer with 0xAB before each launch, calls `kernel0` once per seed and compares every output exactly, printing one `SEED n: PASS/FAIL` line per launch and AutoSA's `Passed!` / `Failed with N errors!` verdict; it returns nonzero on any mismatch, so Vitis' `C/RTL co-simulation finished: PASS` also reflects the check (AutoSA's own testbench returns 0 unconditionally, which is why the previous agent had to read the printed verdict). `validation_pass` is the post-check on the RTL outputs of all three seeds. Per-seed operand statistics are in `meta.json` next to the data and in `results.json`.

Seed 0 of the 16x16x16 data, for the record: A has 116 negative / 139 positive entries, 24 x -128 and 41 x 127; B 124 / 130, 40 x -128, 25 x 127; C spans -260096..262144 with 125 negative outputs.

## How the cycles were measured

Vitis' cosim report gives a latency and an interval per launch, but for an `s_axilite`-controlled kernel those are measured by the testbench around its own AXI-lite control transactions (the 4x4 report says 850/813/813 cycles for the three launches while the kernel's own ap_start-to-ap_done is 788 every time). All cycle columns of `results.csv` therefore come from a VCD of the design's own signals: after the cosim, the snapshot it left in `sol/sim/verilog` is re-elaborated with the identical `xelab` command plus `-debug typical` (`fo_xelab.cmd` in the report directory) and re-run with `xsim -tclbatch fo_vcd.tcl`, which logs `ap_clk`, `ap_rst_n`, the kernel's internal `ap_start/ap_done/ap_idle/ap_ready` and the `gmem_A`/`gmem_B` read and `gmem_C` write handshakes of `/apatb_kernel0_top/AESL_inst_kernel0`. `np/parse_vcd.py` samples on the falling edge of `ap_clk`; a launch starts at the cycle `ap_start` rises and ends at the cycle `ap_done` is high; the launch's outputs are its I x J `gmem_C` WVALID & WREADY beats in order. Definitions:

* `first_output_cycles`: cycles from the launch's ap_start rise to its first gmem_C write beat (steady launch; the cold launch 0 is in `first_output_cold_cycles`; they were equal everywhere).
* `completion_cycles`: ap_start rise to ap_done of the steady launch (`completion_cold_cycles` for launch 0). The cosim report's latency min/avg/max and interval min/avg/max over the three launches are in the `cosim_*` columns.
* `steady_interval_cycles`: the per-tile interval. With several tiles per launch it is the median spacing of the first write beats of consecutive S*S-beat tile groups within the steady launch (`tile_interval_cycles`, min/max in the notes and every tile's beat cycle in `firstout.json`); with one tile per launch (16x16 on 16x16x16, 32x32 on 32x32x16) it is the launch-to-launch interval of ap_start (`launch_interval_cycles`), which includes the testbench's AXI-lite re-arm between launches (about 27 cycles at 4x4).
* `last_output_cycles`: the launch's last gmem_C write beat. `tiles_per_launch` = (I/S)(J/S).

The as-specified (`asis`) rows in `results_all_variants.csv` were not re-run (their kernel form is wrong); their `completion_cycles` is the previous agent's single-launch cosim-report latency and their first-output/interval columns are empty.

## Exact commands

AutoSA generation, run from `/scratch/hc676/autosa/AutoSA` (`./autosa` = `autosa_scripts/autosa.py`; venv `/scratch/hc676/autosa/venv`, python 3.9.25; `PATH`/`LD_LIBRARY_PATH` prefixed with `/scratch/hc676/autosa/llvm9`; `scripts/gen_all.sh`, `scripts/gen_z.sh` by the previous agent, the 32x32x16 one by hand, its `cmd` file in the report directory):

```
# mm4i8z: 4x4 array, problem 16x16x16, 16 PE_wrapper instances
./autosa /scratch/hc676/e1_autosa_2026-09-06/gen/mm4i8z/kernel.c --config=./autosa_config/autosa_config.json --target=autosa_hls_c --output-dir=/scratch/hc676/e1_autosa_2026-09-06/gen/mm4i8z/out --sa-sizes={kernel[]->space_time[3];kernel[]->array_part[4,4,16];kernel[]->latency[1,1];kernel[]->simd[1]} --simd-info=/scratch/hc676/e1_autosa_2026-09-06/gen/mm4i8z/simd_info.json --hls
# mm8i8z: 8x8 array, problem 16x16x16, 64 PE_wrapper instances
./autosa /scratch/hc676/e1_autosa_2026-09-06/gen/mm8i8z/kernel.c --config=./autosa_config/autosa_config.json --target=autosa_hls_c --output-dir=/scratch/hc676/e1_autosa_2026-09-06/gen/mm8i8z/out --sa-sizes={kernel[]->space_time[3];kernel[]->array_part[8,8,16];kernel[]->latency[1,1];kernel[]->simd[1]} --simd-info=/scratch/hc676/e1_autosa_2026-09-06/gen/mm8i8z/simd_info.json --hls
# mm16i8z: 16x16 array, problem 16x16x16, 256 PE_wrapper instances
./autosa /scratch/hc676/e1_autosa_2026-09-06/gen/mm16i8z/kernel.c --config=./autosa_config/autosa_config.json --target=autosa_hls_c --output-dir=/scratch/hc676/e1_autosa_2026-09-06/gen/mm16i8z/out --sa-sizes={kernel[]->space_time[3];kernel[]->array_part[16,16,16];kernel[]->latency[1,1];kernel[]->simd[1]} --simd-info=/scratch/hc676/e1_autosa_2026-09-06/gen/mm16i8z/simd_info.json --hls
# mm32i8z32: 32x32 array, problem 32x32x16, 1024 PE_wrapper instances
./autosa /scratch/hc676/e1_autosa_2026-09-06/gen/mm32i8z32/kernel.c --config=./autosa_config/autosa_config.json --target=autosa_hls_c --output-dir=/scratch/hc676/e1_autosa_2026-09-06/gen/mm32i8z32/out --sa-sizes={kernel[]->space_time[3];kernel[]->array_part[32,32,16];kernel[]->latency[1,1];kernel[]->simd[1]} --simd-info=/scratch/hc676/e1_autosa_2026-09-06/gen/mm32i8z32/simd_info.json --hls
```
### The space-time transform

`--sa-sizes` is AutoSA's space-time transform, identical for every size except the two array_part numbers:
* `kernel[]->space_time[3]`: AutoSA enumerates 6 systolic arrays for this loop nest (`gen.log`: `6 systolic arrays generated`, after the RAR dependence candidates `[1,0,0]` for B and `[0,1,0]` for A) and this picks id 3, which is `kernel_id: 3` in `out/resource_est/design_info.json`. It is the 2-D array with the i and j loops in space and the k loop in time: the generated `PE(int idx, int idy, ...)` takes its two coordinates as constants, reads one A and one B element per cycle from its west and north FIFOs, accumulates into the register `local_C[0][0]` over the 16 k iterations at II=1, forwards both operands east and south (`fifo_A_out`, `fifo_B_out`) and writes the accumulator to the drain FIFO on the last k -- output-stationary, one int32 accumulator per PE (`kernel_kernel.cpp` in each report directory).
* `kernel[]->array_part[S,S,16]`: the array processes an SxSx16 tile at a time and loops over the problem's tiles inside a launch (the PE's two outer loops `c0`, `c1`).
* `kernel[]->latency[1,1]`: no latency-hiding tiling, so each PE holds exactly one C element.
* `kernel[]->simd[1]`: no SIMD on the reduction loop. `simd_info.json` marks k as the reduction (`gen.log`: band member position 2, `reduction property: y`, legal to vectorize with score 15) -- AutoSA would vectorize it at `simd[n]`, which this package did not use, so the PE is one int8 x int8 multiply per cycle.

The I/O network AutoSA builds around it (from `design_info.json` and the generated modules): A and B enter through an L3 module that reads memory 16 int8 per word (`n_lane: 16`, the 128-bit AXI ports), fan out through per-row/column L2 modules that unpack to 1 element per cycle into the PE FIFOs, and C leaves through L1/L2/L3 drain modules at `n_lane: 1` (one int32 per beat, the 32-bit AXI port).

The kernel's `local_C` reset per tile comes from `C[i][j] = 0` in the input (see the kernel forms section).

Vitis HLS, in `hls/<design>/` (`scripts/run_hls2.sh` for 4/8/16; `hls/mm32i8z32/csynth.tcl` + `cosim.tcl` for 32x32 so that its P&R could start on the RTL while the cosim ran). The three `m_axi` pragmas get `depth=` values sized to the problem (A, B: I*K/16 and J*K/16 128-bit words; C: I*J int32) because Vitis refuses cosim on an m_axi port without a depth; `patched_pragmas.txt` in each report directory shows them:

```
open_project cosim_prj
set_top kernel0
add_files kernel_kernel.cpp
add_files -tb kernel_host.cpp
open_solution sol
set_part xcu280-fsvh2892-2L-e
create_clock -period 3.333 -name default
csynth_design
cosim_design -rtl verilog
exit
```
The numpy cosim of the 4/8/16 designs (`np/run_cosim_np.sh`) copies the synthesized project (`hls/<d>` -> `hlsnp/<d>`), replaces `kernel_host.cpp` with the generated testbench and runs `open_project cosim_prj; open_solution sol; cosim_design -rtl verilog` (no re-synthesis: the RTL is byte-identical to the one placed and routed). The 32x32 project was created with the numpy testbench.

VCD re-run, in `cosim_prj/sol/sim/verilog` of the numpy cosim (`fo_xelab.cmd`, `fo_vcd.tcl` in the report directory):

```
/opt/xilinx/Vivado/2023.2/bin/xelab xil_defaultlib.apatb_kernel0_top glbl -Oenable_linking_all_libraries  -prj kernel0.prj -L smartconnect_v1_0 -L axi_protocol_checker_v1_1_12 -L axi_protocol_checker_v1_1_13 -L axis_protocol_checker_v1_1_11 -L axis_protocol_checker_v1_1_12 -L xil_defaultlib -L unisims_ver -L xpm  -L floating_point_v7_1_16 -L floating_point_v7_0_21 --lib "ieee_proposed=./ieee_proposed" -s kernel0_fo -debug typical
xsim --noieeewarnings kernel0_fo -tclbatch fo_vcd.tcl
# fo_vcd.tcl:
open_vcd kernel0_fo.vcd
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/ap_clk
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/ap_rst_n
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/ap_start
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/ap_done
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/ap_idle
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/ap_ready
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_C_AWVALID
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_C_AWREADY
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_C_WVALID
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_C_WREADY
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_C_BVALID
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_C_BREADY
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_A_ARVALID
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_A_ARREADY
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_A_RVALID
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_A_RREADY
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_B_ARVALID
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_B_ARREADY
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_B_RVALID
log_vcd /apatb_kernel0_top/AESL_inst_kernel0/m_axi_gmem_B_RREADY
run all
close_vcd
quit
```
P&R OOC (`scripts/run_pnr2.sh`, `np/run_pnr3.sh` for 32x32; `proc stage` prints `E1 STAGE <name> <seconds>` around each step and `E1 NPORTS` is `llength [get_ports]` after synthesis):

```
proc stage {name body} {
  set t0 [clock milliseconds]
  uplevel 1 $body
  puts "E1 STAGE $name [expr {([clock milliseconds] - $t0) / 1000.0}]"
}
create_project -in_memory -part xcu280-fsvh2892-2L-e
add_files [glob /scratch/hc676/e1_autosa_2026-09-06/hls/mm4i8z/cosim_prj/sol/syn/verilog/*.v]
add_files -fileset constrs_1 /scratch/hc676/e1_autosa_2026-09-06/pnr/mm4i8z/clock.xdc
stage synth {synth_design -top kernel0 -part xcu280-fsvh2892-2L-e -mode out_of_context}
puts "E1 NPORTS [llength [get_ports]]"
report_utilization -file /scratch/hc676/e1_autosa_2026-09-06/pnr/mm4i8z/util_synth.rpt
stage opt {opt_design}
stage place {place_design}
stage physopt {phys_opt_design}
stage route {route_design}
report_utilization -file /scratch/hc676/e1_autosa_2026-09-06/pnr/mm4i8z/util.rpt
report_timing_summary -file /scratch/hc676/e1_autosa_2026-09-06/pnr/mm4i8z/timing.rpt
report_route_status -file /scratch/hc676/e1_autosa_2026-09-06/pnr/mm4i8z/route_status.rpt
set wns [get_property SLACK [get_timing_paths -delay_type max]]
puts "AUTOSA ROUTED WNS $wns"
puts "IMPLEMENTATION OK"
```
`clock.xdc`: `create_clock -period 3.333 -name ap_clk [get_ports ap_clk]`. Collection: `scripts/np/collect2.py` (imports the previous agent's parsers from `scripts/collect.py`), `scripts/np/write_readme2.py` (this file).

## Source revision and tool versions

* AutoSA `b61a1b4132d631600696feba59eb606acb34d304 2021-11-23 18:02:23 -0800 Add min/max template in tapa code` at /scratch/hc676/autosa/AutoSA (local state: `M src/autosa_common.cpp;  ? src/barvinok;  ? src/cJSON;  m src/isl;  ? src/pet; ?? src/autosa_common.cpp.orig`).
* Vitis HLS Version:        2023.2 (Build 4023990 on Oct 11 2023); Vivado Simulator v2023.2; Vivado v.2023.2 (lin64) Build 4029153 Fri Oct 13 20:13:54 MDT 2023; numpy 2.0.2 (the AutoSA venv's python 3.9.25).
* Part xcu280-fsvh2892-2L-e, clock period 3.333 ns in every tool (the csynth report prints the target rounded to 3.33).

## Results (`results.csv`)

| run_id | status | valid | first out | completion | steady interval | tile interval | tiles | LUT | FF | DSP | BRAM36 | BRAM18 | BRAM18-eq | URAM | WNS ns | TNS ns | unrouted | HLS s | synth s | place s | route s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| autosa_mm4i8z_cosim | pass | true | 63 | 788 | 47 | 47 | 16 | null | null | null | null | null | null | null | null | null | null | 85.5 | null | null | null | 185.2 |
| autosa_mm4i8z_pnr_ooc | pass | true | 63 | 788 | 47 | 47 | 16 | 8129 | 17481 | 16 | 4 | 1 | 9 | 0 | 0.536 | 0.000 | 0 | 85.5 | 352.6 | 339.3 | 91.9 | 864.7 |
| autosa_mm8i8z_cosim | pass | true | 69 | 795 | 183 | 183 | 4 | null | null | null | null | null | null | null | null | null | null | 389.9 | null | null | null | 611.0 |
| autosa_mm8i8z_pnr_ooc | pass | true | 69 | 795 | 183 | 183 | 4 | 19755 | 38762 | 64 | 4 | 1 | 9 | 0 | 0.331 | 0.000 | 0 | 389.9 | 508.3 | 429.5 | 117.6 | 1146.7 |
| autosa_mm16i8z_cosim | pass | true | 71 | 797 | 824 | null | 1 | null | null | null | null | null | null | null | null | null | null | 2169.6 | null | null | null | 3607.9 |
| autosa_mm16i8z_pnr_ooc | pass | true | 71 | 797 | 824 | null | 1 | 47434 | 81265 | 256 | 4 | 1 | 9 | 0 | 0.455 | 0.000 | 0 | 2169.6 | 667.0 | 491.0 | 298.2 | 1606.3 |
| autosa_mm32i8z32_cosim | pass | true | 87 | 2989 | 3011 | null | 1 | null | null | null | null | null | null | null | null | null | null | 36088.4 | null | null | null | 51124.7 |
| autosa_mm32i8z32_pnr_ooc | pass | true | 87 | 2989 | 3011 | null | 1 | 169841 | 262017 | 1024 | 5 | 1 | 11 | 0 | 0.048 | 0.000 | 0 | 36088.4 | 3080.8 | 1399.6 | 824.7 | 5744.4 |

Other variants (`results_all_variants.csv`, previous agent's runs):

| run_id | variant | status | valid | completion (cosim rpt) | LUT | FF | DSP | BRAM18-eq | URAM | WNS ns | total s | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| autosa_mm4i8_cosim | asis | functional_fail | false | 850 | null | null | null | null | null | null | 93.1 | testbench post-check on the RTL outputs printed 'Failed with 240 errors!' (Vitis' PASS only reflects the testbench's return 0); C-model run printed 'Failed with |
| autosa_mm4i8_pnr_ooc | asis | pass | false | 850 | 8334 | 18149 | 16 | 9 | 0 | 0.56 | 585.6 |  |
| autosa_mm8i8_cosim | asis | functional_fail | false | 850 | null | null | null | null | null | null | 300.0 | testbench post-check on the RTL outputs printed 'Failed with 192 errors!' (Vitis' PASS only reflects the testbench's return 0); C-model run printed 'Failed with |
| autosa_mm8i8_pnr_ooc | asis | pass | false | 850 | 20301 | 41302 | 64 | 9 | 0 | 0.463 | 1054.2 |  |
| autosa_mm16i8_cosim | asis | pass | true | 859 | null | null | null | null | null | null | 3602.7 |  |
| autosa_mm16i8_pnr_ooc | asis | pass | true | 859 | 53253 | 96913 | 256 | 9 | 0 | 0.543 | 1794.4 |  |
| autosa_mm32i8_cosim | asis | unsupported | null | null | null | null | null | null | null | null | null | AutoSA cannot size a 32x32 output-stationary array on the 16x16x16 problem: array_part[32,32,16] is clamped to the loop bounds without a warning and the generat |
| autosa_mm32i8_pnr_ooc | asis | unsupported | null | null | null | null | null | null | null | null | null | AutoSA cannot size a 32x32 output-stationary array on the 16x16x16 problem: array_part[32,32,16] is clamped to the loop bounds without a warning and the generat |
| autosa_mm32i8z_cosim | cinit | unsupported | null | null | null | null | null | null | null | null | null | AutoSA cannot size a 32x32 output-stationary array on the 16x16x16 problem: array_part[32,32,16] is clamped to the loop bounds without a warning and the generat |
| autosa_mm32i8z_pnr_ooc | cinit | unsupported | null | null | null | null | null | null | null | null | null | AutoSA cannot size a 32x32 output-stationary array on the 16x16x16 problem: array_part[32,32,16] is clamped to the loop bounds without a warning and the generat |

## Interface, modules, csynth estimates, cosim report

| design | problem | PEs | tiles/launch | AXI masters (data bits) | AXI-lite | port bits (Vivado) | HLS modules | RTL files | csynth latency | csynth interval | csynth est LUT/FF/DSP/BRAM18k | cosim rpt latency min/avg/max | cosim rpt interval min/avg/max | csynth s | cosim s (3 launches) | VCD s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mm4i8z | 16x16x16 | 16 | 16 | gmem_A (128), gmem_B (128), gmem_C (32) | control | 1351 | 75 | 141 | 542..542 | 514..514 | 22537/16004/16/82 | 813/825/850 | 813/831/850 | 85.53 | 62.99 | 32.2 |
| mm8i8z | 16x16x16 | 64 | 4 | gmem_A (128), gmem_B (128), gmem_C (32) | control | 1351 | 199 | 381 | 430..430 | 386..386 | 63725/38627/64/146 | 813/825/850 | 813/831/850 | 389.89 | 146.82 | 69.8 |
| mm16i8z | 16x16x16 | 256 | 1 | gmem_A (128), gmem_B (128), gmem_C (32) | control | 1351 | 639 | 1245 | 401..401 | 322..322 | 170984/107823/256/148 | 822/834/859 | 822/840/859 | 2169.64 | 1188.75 | 245.1 |
| mm32i8z32 | 32x32x16 | 1024 | 1 | gmem_A (128), gmem_B (128), gmem_C (32) | control | 1351 | 2287 | 4509 | 1297..1297 | 1154..1154 | 631297/381047/1024/276 | 3009/3021/3046 | 3009/3027/3046 | 36088.4 | 13700.3 | 1332.0 |
| mm4i8 | 16x16x16 | 16 | 16 | gmem_A (128), gmem_B (128), gmem_C (32) | control | 1351 | 75 | 141 | 542..542 | 514..514 | 22361/16180/16/82 | 850/850/850 | null/null/null | 46.39 | 46.74 | null |
| mm8i8 | 16x16x16 | 64 | 4 | gmem_A (128), gmem_B (128), gmem_C (32) | control | 1351 | 199 | 381 | 430..430 | 386..386 | 63021/39331/64/146 | 850/850/850 | null/null/null | 175.18 | 124.8 | null |
| mm16i8 | 16x16x16 | 256 | 1 | gmem_A (128), gmem_B (128), gmem_C (32) | control | 1351 | 639 | 1245 | 401..401 | 322..322 | 168680/110895/256/148 | 859/859/859 | null/null/null | 1926.97 | 1675.74 | null |

`PEs` = PE_wrapper calls in the generated kernel0; `HLS modules` = functions Vitis HLS synthesised (one `*_csynth.rpt` each: AutoSA passes the PE coordinates as constants, so every PE_wrapper_N is its own module, which is also why csynth time grows with the array); `port bits` = Vivado's `get_ports` count on kernel0, the number that forces the OOC flow; the csynth latency/interval are Vitis' static estimates for one launch.

## Per-tile detail (VCD, steady launch)

* `mm4i8z` (16x16x16, 16 tile(s) of 16 output beats): launch starts [83, 898, 1713], latency [788, 788, 788], first output [63, 63, 63], A read beats 16 (first at 10), B read beats 64, tile first-beat cycles [63, 105, 152, 199, 241, 288, 335, 377, 424, 471, 513, 560, 607, 649, 696, 743], tile intervals [42, 47, 47, 42, 47, 47, 42, 47, 47, 42, 47, 47, 42, 47, 47], drain span per tile [40, 46, 42, 40, 46, 42, 40, 46, 42, 40, 46, 42, 40, 46, 42, 40].
* `mm8i8z` (16x16x16, 4 tile(s) of 64 output beats): launch starts [83, 898, 1713], latency [795, 795, 795], first output [69, 69, 69], A read beats 16 (first at 10), B read beats 32, tile first-beat cycles [69, 247, 430, 613], tile intervals [178, 183, 183], drain span per tile [177, 182, 178, 177].
* `mm16i8z` (16x16x16, 1 tile(s) of 256 output beats): launch starts [83, 907, 1731], latency [797, 797, 797], first output [71, 71, 71], A read beats 16 (first at 10), B read beats 16, tile first-beat cycles [71], tile intervals None, drain span per tile [721].
* `mm32i8z32` (32x32x16, 1 tile(s) of 1024 output beats): launch starts [83, 3094, 6105], latency [2989, 2989, 2989], first output [87, 87, 87], A read beats 32 (first at 10), B read beats 32, tile first-beat cycles [87], tile intervals None, drain span per tile [2897].

## Provenance

* Previous agent (lanes A/B, 20:28-22:57): AutoSA generation of all 16x16x16 designs, csynth + AutoSA-testbench cosim of mm{4,8,16}i8 and mm{4,8,16}i8z, OOC P&R of all six (the mm16i8z P&R was its last job and finished at 22:57, after its process had died), the clamped-32x32 evidence, `scripts/collect.py` (whose report parsers this collection imports) and the first `results.csv` (kept as `results_prev_agent_2056.csv`, `summary_prev_agent_2056.json`).
* This agent (22:55 onwards): the numpy data and testbench, the three-seed cosim and VCD measurement of every cinit design, the 32x32 design on 32x32x16 (generation, csynth, cosim, P&R), `collect2.py`, `write_readme2.py`, this README. Every row's `provenance` column says which runs its numbers come from.

## Caveats

* Out-of-context P&R: kernel0 alone, no I/O buffers, no XDMA shell, no HBM/AXI interconnect, no clock crossing -- set up in AutoSA's favour relative to a full bitstream.
* Memory model: the cosim's AXI masters talk to Vitis' behavioural AXI memory models with no DRAM latency model beyond the testbench's handshakes; A and B are read through 128-bit `gmem_A`/`gmem_B` (one 16-int8 row per word) and C is written through the 32-bit `gmem_C` port one int32 per beat, so a launch's drain needs I*J beats and the drain, not the array, paces the 16x16 and 32x32 single-tile launches. AutoSA chose these widths from the data types and `simd[1]`; wider C ports need SIMD on the output or a different `--sa-sizes`, which this package did not explore.
* `hls_wall_s` is csynth's elapsed time; `cosim_wall_s` the numpy cosim's vitis_hls wall time (3 launches); `vcd_wall_s` xelab + xsim of the re-run; `total_wall_s` on a cosim row is their sum and on a pnr_ooc row the whole vivado run including opt/phys_opt/reports; `synth_s`/`place_s`/`route_s` the stage times. Wall times are on a shared server with other users' jobs.
* `bram_18k_equiv` = 2 x RAMB36 + RAMB18 of the routed design; `lut`/`ff`/`dsp`/`uram` = CLB LUTs, CLB Registers, DSPs, URAM from `report_utilization` after routing; `wns_ns`/`tns_ns` from `report_timing_summary`; `unrouted` from `report_route_status`.
* The 32x32 row's problem (32x32x16) differs from the others (16x16x16) for the reason given above; its P&R and cosim are 1024-PE runs and their wall times are what they are.

## Files

* `results.csv`: the eight (size, mode) rows of the cinit designs; the 33 columns asked for, then the extra columns `variant`, `tiles_per_launch`, `tile_interval_cycles`, `launch_interval_cycles`, `last_output_cycles`, `first_output_cold_cycles`, `completion_cold_cycles`, `cosim_latency_min/avg/max`, `cosim_interval_min/avg/max`, `cosim_wall_s`, `vcd_wall_s`, `top_port_bits`, `numeric_validation`, `provenance`, `notes`. Empty cell = not measured (the `failure_reason` or `notes` says why).
* `results_all_variants.csv`: the same plus the `asis` rows and the clamped 32x32 evidence rows.
* `results.json`: rows, the other-variant rows and everything parsed per design (csynth report, cosim reports of both testbenches, the VCD analysis with every launch and tile, utilisation/timing/route status, stage times, interface, data statistics, provenance).
* `reports/<run_id>/`: cosim rows: `cosim_np.log` (numpy cosim), `kernel0_cosim.rpt`, `kernel0_csynth.rpt`, `cosim.log` (the previous agent's AutoSA-testbench cosim, where it exists), `kernel_host.cpp` (numpy testbench), the generated sources, `patched_pragmas.txt`, `firstout.json`/`firstout.log` (VCD analysis), `fo_xelab.cmd`, `fo_vcd.tcl`, `fo_xsim.log`, `kernel0_fo.vcd.gz`, `meta.json` (data statistics), `cmd`/`gen.log` (AutoSA); pnr_ooc rows: `util.rpt`, `util_synth.rpt`, `timing.rpt`, `route_status.rpt`, `viv.log`, `impl.tcl`, `clock.xdc`, `pnr_result.txt`.
* `scripts/`: the previous agent's lane/HLS/P&R/collect scripts; `scripts/np/`: this agent's data generator, testbench generator, cosim+VCD runner, VCD parser, lanes, P&R runner, collector, README writer.
* `results_prev_agent_2056.csv`, `summary_prev_agent_2056.json`: the previous agent's collection as it left it, kept for comparison (superseded by results.csv / results.json).

`/scratch/hc676/harvest.sh` (another session's, every 20 minutes) re-runs `<workspace>/scripts/collect.py` for this package, and that path was the previous agent's collector, which rewrites `results.csv` in its older 16-row layout (it did so at 12:49 and its `summary.json` was removed afterwards). That file is now a wrapper that runs `np/collect2.py` + `np/write_readme2.py` instead, so a harvest re-run refreshes this package rather than overwriting it; the original is kept verbatim as `scripts/collect_prev_agent_original.py` and as `scripts/collect_parsers.py`, the module whose report parsers the current collector imports.

## Job log (`logs/lanes.log`)

```
2026-09-06 20:28:14 lane B start hls 4
2026-09-06 20:28:14 lane A start hls 8
2026-09-06 20:29:51 lane B end hls 4 rc=0
2026-09-06 20:29:51 lane B start hls 16
2026-09-06 20:33:18 lane A end hls 8 rc=0
2026-09-06 20:33:18 lane A start pnr 4
2026-09-06 20:41:59 lane A end pnr 4 rc=0
2026-09-06 20:41:59 lane A start pnr 4
2026-09-06 20:51:45 lane A end pnr 4 rc=0
2026-09-06 20:51:45 lane A start pnr 8
2026-09-06 21:09:19 lane A end pnr 8 rc=0
2026-09-06 21:09:19 lane A start hls2 mm4i8z
2026-09-06 21:12:07 lane A end hls2 mm4i8z rc=0
2026-09-06 21:12:07 lane A start hls2 mm8i8z
2026-09-06 21:23:11 lane A end hls2 mm8i8z rc=0
2026-09-06 21:23:11 lane A start pnr2 mm4i8z
2026-09-06 21:30:00 lane B end hls 16 rc=0
2026-09-06 21:30:00 lane B start pnr2 mm8i8z
2026-09-06 21:37:36 lane A end pnr2 mm4i8z rc=0
2026-09-06 21:37:36 lane A start hls2 mm16i8z
2026-09-06 21:49:07 lane B end pnr2 mm8i8z rc=0
2026-09-06 21:49:07 lane B start pnr 16
2026-09-06 22:19:01 lane B end pnr 16 rc=0
2026-09-06 22:19:01 lane B start pnr2 mm16i8z
2026-09-06 22:30:43 lane A end hls2 mm16i8z rc=0
2026-09-06 22:30:43 lane A queue empty, exiting
2026-09-06 22:55:13 lane32 start csynth mm32i8z32
2026-09-06 22:57:48 lane B end pnr2 mm16i8z rc=0
2026-09-06 22:57:48 lane B queue empty, exiting
2026-09-06 22:58:07 lane_np waiting for pnr/mm16i8z/pnr.done
2026-09-06 22:58:07 lane_np start cosim_np mm4i8z
2026-09-06 22:59:47 lane_np end cosim_np mm4i8z rc=0
2026-09-06 22:59:47 lane_np start cosim_np mm8i8z
2026-09-06 23:03:29 lane_np end cosim_np mm8i8z rc=0
2026-09-06 23:03:29 lane_np start cosim_np mm16i8z
2026-09-06 23:27:29 lane_np end cosim_np mm16i8z rc=0
2026-09-06 23:27:29 lane_np waiting for hls/mm32i8z32/csynth.done
2026-09-07 08:56:56 lane32 end csynth mm32i8z32 rc=0
2026-09-07 08:56:56 lane32 start pnr3 mm32i8z32
2026-09-07 08:57:30 lane_np start cosim_np mm32i8z32
2026-09-07 10:32:40 lane32 end pnr3 mm32i8z32 rc=0
2026-09-07 13:08:06 lane_np end cosim_np mm32i8z32 rc=0
2026-09-07 13:08:06 lane_np queue empty, exiting
```
