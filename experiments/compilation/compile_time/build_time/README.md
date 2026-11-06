# Fig. 13(a), arXiv version: E1's memory-connected SPMW builds at 4/8/16/32

The output-stationary int8 GEMM in AutoSA's shape
(`tests/dataflow/spmw/test_spmw_autosa_match.py`, design `autosa` of
`scripts/spmw_build_array.py`): the SxS mesh with the loaders that stream the A
and B tiles from memory and a drain that writes C out (`MemIn`/`MemOut` at the
edges). One launch is one SxS tile with K = S. Built by the split backend (one
Vitis HLS project per role, an RTL fabric of register-slice FIFOs between the
sites) and placed and routed out of context on xcu280-fsvh2892-2L-e at a
3.333 ns target (Vitis HLS and Vivado 2023.2), 2026-09-06.

`results.csv` and `results.json` hold the same rows, one per (size, mode).
Fig. 13(a) reads the four `S<n>_pnr` rows of `results.json`. The `report_paths`
column names the build directories, which are not here: the P&R builds'
`cost.json`, `util.rpt` and `timing.rpt` are in
[`../../../performance/gemm_arrays/spmw_kernel/S<n>/report/`](../../../performance/gemm_arrays/spmw_kernel/),
and the memsim and memcosim runs kept only these rows.

## Modes and status

- `pnr`: synth, opt, place, phys_opt, route out of context (`spmw_harness`
  wraps the array so its boundary FIFOs are real endpoints); LUT/FF/DSP/BRAM/
  URAM from `report_utilization` on the routed design, WNS/TNS from
  `report_timing_summary`, `achieved_ns` = target - WNS. Stage times are
  the script's own stopwatch around each Vivado stage; `hls_wall_s` the
  concurrent HLS (`csynth_design` and `export_design`) of the 15 roles, 8
  workers; `hls_sum_job_s` the sum of the jobs' own elapsed times.
- `memcosim` (`--memory --cosim`): **unsupported** in this flow. The
  out-of-context array testbench drives streams and has no AXI memory model,
  so the build stops at elaboration (`Module <feed_up_load_io> not found`).
  The memory-mapped path is validated instead by:
- `memsim`: the design packaged as the kernel (`scripts/spmw_package_kernel.py
  --sim`: roles, AXI movers, control register file) and simulated in xsim
  against behavioural AXI RAM with one launch's operands
  (`scripts/spmw_kernel_sim.py`). `completion_cycles` there is (done - start) /
  4 ns from the testbench's timestamps (its clock is 4 ns; the kernel is
  written through AXI-lite and `done` is seen by polling the control register,
  one poll about eight cycles), so it is the launch's latency through the
  movers, from the start bit to the last drain beat, not the array's tile
  cycles. The kernel has three AXI masters (two feeders, one drain) of 512
  bits. Results: 66 / 100 / 152 / 298 cycles at 4 / 8 / 16 / 32, every drain
  byte matching the reference.

## Caveats

- Out-of-context P&R: no shell, no HBM/PCIe.
- The tile workload only (M = N = K = S per launch); larger workloads are
  host-side loops over tiles and are not timed here.
- Status `pass` for P&R means routed with WNS >= 0 and no unrouted nets.
- One build a size, no repetition; the load average was not recorded.
