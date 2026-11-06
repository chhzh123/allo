# E1, SPMW side: output-stationary int8 GEMM arrays at 4/8/16/32

Two SPMW designs, both from `scripts/spmw_build_array.py`'s registry, built
by the split backend (one Vitis HLS project per role, an RTL fabric of
register-slice FIFOs between the sites), simulated bit-exactly against the
design's own reference (`SPMW COSIM PASS n/n tokens`) and placed and routed
out of context on xcu280-fsvh2892-2L-e at a 3.333 ns target (Vitis HLS and
Vivado 2023.2). `results.csv` has one row per (size, mode); `reports/S<n>_<mode>/`
the build log, the xsim log for cosims, and `cost.json`, `util.rpt`,
`timing.rpt`, `route.rpt`, `util_synth.rpt` for P&R runs.

## Designs

- **`gemm8`** (`spmw/` here; `tests/dataflow/spmw/test_spmw_gemm.py`):
  the SxS output-stationary mesh. Each PE holds one int32 accumulator,
  streams A east and B south, and multiplies int8 x int8 in a DSP
  (`config_op mul -impl dsp`); the C tile is gathered from every site
  (`MemOut`, `gather`). One launch is one SxS tile with K = S (the tile
  workload of the handoff). Stream-connected: the harness feeds the edge
  FIFOs directly, which is the array's own interface; in a kernel the
  feeders are the memory-mapped movers of E3's engine.
- **`autosa`** (`spmw_mem/` here; `tests/dataflow/spmw/test_spmw_autosa.py`):
  the same mesh in AutoSA's shape, with the loaders that stream the A and B
  tiles from memory and a drain that writes C out (`MemIn`/`MemOut` at the
  edges, memory-mapped movers in the kernel). The P&R rows implement the
  array with its loaders and drain; the memory-connected validation is
  the kernel-level simulation below.

## Modes and status

- `cosim`: RTL cosimulation of the assembled array in xsim, every output
  token compared with the reference; `completion_cycles` = cycles from the
  first input token to the last output token of the launch,
  `first_output_cycles` = cycles to the first output token.
- `pnr`: synth, opt, place, phys_opt, route out of context (`spmw_harness`
  wraps the array so its boundary FIFOs are real endpoints); LUT/FF/DSP/BRAM/
  URAM from `report_utilization` on the routed design, WNS/TNS from
  `report_timing_summary`, `achieved_ns` = target - WNS. Stage times are
  the script's own stopwatch around each Vivado stage; `hls_wall_s` the
  concurrent HLS of the roles (8 workers), `hls_sum_job_s` the sum of the
  jobs' own elapsed times.
- `memcosim` (`--memory --cosim`): **unsupported** in this flow, for two
  different reasons that the rows carry: `gemm8` gathers C from every site
  (a `MemOut` per site has no mover), and the out-of-context array testbench
  drives streams and has no AXI memory model, so the `autosa` build stops
  at elaboration (`Module <feed_up_load_io> not found`). The memory-mapped
  path is validated instead by:
- `memsim` (`spmw_mem/reports/S<n>_memsim/`): the `autosa` design packaged
  as the kernel (`spmw_package_kernel.py --sim`: roles, AXI movers, control
  register file) and simulated in xsim against behavioural AXI RAM with
  one launch's operands (`spmw_kernel_sim.py`), the same flow that
  validated the E3 engine before its bitstreams. `completion_cycles` there
  is (done - start) / 4 ns from the testbench's timestamps (its clock is
  4 ns; the kernel is written through AXI-lite and `done` is seen by
  polling the control register, one poll about eight cycles), so it is the
  launch's latency through the movers, from the start bit to the last drain
  beat, not the array's tile cycles. The kernel has three AXI masters (two
  feeders, one drain) of 512 bits; the earlier paper number (377 cycles at
  16x16) was a kernel with 18 masters. Results: 66 / 100 / 152 / 298 cycles
  at 4 / 8 / 16 / 32, every drain byte matching the reference.

## Caveats

- Out-of-context P&R: no shell, no HBM/PCIe; the 302 kernel of E3 is the
  in-context data point for this array family (16x16 engine, timing met at
  300 MHz in the Vitis shell).
- The tile workload only (M = N = K = S per launch); larger workloads are
  host-side loops over tiles, as in E3, and are not timed here.
- Status `pass` for P&R means routed with WNS >= 0 and no unrouted nets.
