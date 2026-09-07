# E2 baseline: the existing Allo FFT (MachSuite `fft/strided`) through Allo's Vitis HLS flow at 300 MHz

Package E2 of the SPMW evaluation, Allo side. Host, tools and part as for HP-FFT: brg-zhang-xcel, Vitis HLS 2023.2,
Vivado 2023.2 (SW build 4029153), xsim 2023.2, xcu280-fsvh2892-2L-e, 3.333 ns. Produced by two agents (the first
built the projects and ran csim/csynth/cosim with an LCG stimulus and 4 calls; the second re-ran them with the numpy
seeds-0/1/2 stimulus and 6 calls, ran the P&R, and re-assembled the package); `run_by` in `results.csv` says which.
The first agent's README and table are preserved as `README_previous_agent.md` / `results_previous_agent.csv`.

## Which Allo FFT this is, and what it computes

Allo tree: `/scratch/hc676/allo` (rsync of worktree `hc/spmw-allo-implementation-99c949`, HEAD `f436658a`), python
`/scratch/hc676/allo-agent/bin/python3` with `LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build`. The tree
contains two FFTs under `examples/machsuite/fft/`:

* `strided/strided_fft.py` -- MachSuite `fft_strided`: iterative **radix-2 decimation-in-frequency**, in-place on two
  `float32[FFT_SIZE]` arrays (real, imag), twiddles passed in as `float32[FFT_SIZE/2]` arrays (real_twid, img_twid).
  **Natural-order input, bit-reversed-order output** (DIF with natural input). `FFT_SIZE` is a module-level constant
  (1024); it can be patched to any power of two before `allo.customize`, exactly as the example's `run_test.py` does,
  so 128/256/512/1024 all come from the same source. Two nested `while` loops (`span`, `odd`), no Allo schedule
  primitives in the example, no pipelining/unrolling in the source, hence no unroll factor (`unroll_factor` is empty).
* `transpose/transpose_fft.py` -- MachSuite `fft_transpose`: fixed 512-point radix-8, not radix-2 and not
  size-parameterised; not used.

The radix-2 one is what is measured. It is **not** a streaming design and it is not natural-order at the output;
validation compares against `numpy.fft.fft(x)[bitrev(k)]`. Twiddles are supplied as the forward-transform values
`W_N^k = exp(-2*pi*j*k/N)`, k < N/2 (the MachSuite harness feeds `cos(2*pi*k/N), sin(2*pi*k/N)`, the conjugate
direction; the twiddles are inputs, so the direction is a property of the data, not of the design). Other Allo FFTs
exist on branches of the Allo repository but not in the evaluated tree (`origin/hpfft`: `examples/fft/fft.py`, a
streaming radix-2 dataflow re-implementation of HP-FFT with an unroll factor; `origin/feature/allo-fft`:
`tests/dataflow/test_fft.py`, a vectorised FFT-256); they were not run (the task was the existing FFT in the tree).

## Flow (exact commands)

```
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh
python3 /scratch/hc676/e2_allo/gen_projects.py          # previous agent: generates the 8 projects
```

`gen_projects.py` does, for N in {128, 256, 512, 1024} and wrap_io in {True, False}:

```python
strided_fft.FFT_SIZE = N; strided_fft.FFT_SIZE_HALF = N // 2
s = allo.customize(strided_fft.fft)
mod = s.build(target="vitis_hls", mode="csyn", project=f"strided_n{N}_{wrap|raw}",
              configs={"frequency": 300, "device": "u280", "num_output_args": 2}, wrap_io=wrap)
```

`num_output_args=2` is needed because the kernel is in-place (real/img are both read and written and are not the
last arguments; without it Allo's vitis_hls backend raises "Output arguments must appear at the end"); it only
affects Allo's host-code generation, which is not used here. `wrap_io=True` is Allo's default for `vitis_hls`: the
top `fft` becomes load_buf0..3 -> kernel on local buffers -> store_res0/1, all four arguments on separate `m_axi`
bundles (gmem0..3, 32-bit, 1 float per beat) with an `s_axi_control` block; `wrap_io=False` ("raw") keeps the same
`m_axi` ports but the kernel loops access them directly. Allo writes `kernel.cpp`, `kernel.h`, `run.tcl`
(`open_solution -flow_target vivado`, `set_part xcu280-fsvh2892-2L-e`, `create_clock -period 3.33`, `csynth_design`)
and an XRT `host.cpp`.

Each project was then run with Allo's `run.tcl` modified in two places (`run_e2.tcl` / `run_e2s.tcl`): the XRT host
replaced by the E2 testbench (`add_files -tb tb.cpp -cflags "-std=gnu++0x -DE2_N=<N> -DE2_NT=<NT>"`) and
`csim_design` / `cosim_design -trace_level port -rtl verilog` added around Allo's `csynth_design`. One thing was
changed in the generated `kernel.cpp` copies: `depth=<N|N/2>` was appended to Allo's four `#pragma HLS interface
m_axi ...` lines (Vitis refuses to cosimulate an m_axi port without a depth; it does not change the RTL). Allo's
`1000/300` is formatted as `3.33` ns (300.3 MHz); the OOC P&R constrains 3.333 ns.

* Previous agent (`run_allohls2.sh <N> <wrap|raw>`, projects `strided_n<N>_<var>/`): `tb.cpp`, 4 calls, transform 0 =
  HP-FFT's shipped test signal, transforms 1..3 from the LCG(0x12345678) in [-1,1). Rows `*_cosim_lcg`.
* This agent (`run_allocosim_s.sh <N> wrap`, fresh copies `strided_n<N>_wrap_s/` of the same kernel.cpp/run.tcl):
  `tb_s.cpp`, 6 calls, inputs from `gen_stimulus.py <N> 6` = numpy `RandomState(seed).uniform(-1,1,(2,N,2))` as
  float32 for seeds 0, 1, 2 (calls 0-1 seed 0, 2-3 seed 1, 4-5 seed 2). Rows `*_cosim` (primary).

Cycle numbers come from Vitis' cosim transaction report (`sim/report/verilog/result.transaction.rpt`, per-call
latency ap_start -> ap_done and call-to-call interval), since the block is controlled through `s_axi_control`
register bits, not ports: `completion_cycles` = latency of the first call, `steady_interval_cycles` = mean call
interval over the last three quarters of the calls (all per-call values in `notes`), `transforms_streamed` = calls.
`first_output_cycles` is not observable: the results are written back to `m_axi` by the store stage at the end of the
call. Validation: the RTL outputs as received by Vitis' post-check testbench run (`sim/wrapc_pc/e2_outputs.txt`,
i.e. what the RTL wrote to the real/img buffers) against `numpy.fft.fft` of the exact float32 inputs, permuted to
bit-reversed index order, `allclose(atol=1e-4, rtol=1e-4)`, per seed (`validation/<run_id>/`).

P&R: `run_pnrallo.sh <N> <wrap|raw>` = `pnr_ooc.sh out.prj/solution1/syn/verilog fft <prj>/pnr`, identical to the
HP-FFT recipe (Vivado 2023.2, `synth_design -mode out_of_context`, opt, place, phys_opt, route at 3.333 ns; top =
`fft`, which includes the four `m_axi` adapters and the `s_axi_control` block Allo's flow emits). Same column
definitions as the HP-FFT README (`bram_18k_equiv` = 2*RAMB36 + RAMB18, WNS/TNS post-route, `unrouted` from
`report_route_status`, stage wall times).

## Findings and caveats

1. **What HLS makes of the MachSuite kernel.** The two `while` loops have no trip-count bound, so csynth reports no
   latency/interval (empty in `hls_estimates.csv`). Vitis auto-pipelines the inner `while (odd < N)` loop
   (`VITIS_LOOP_123_2`) but only at **II = 25** because of the in-place read-modify-write of `real[]`/`img[]` through
   BRAM (`HLS 200-880 II violation` x7); the outer stage loop is not pipelined. So the design does one radix-2
   butterfly every 25 cycles: per-call latency 11 586 / 26 261 / 58 808 / 130 299 cycles for N = 128 / 256 / 512 /
   1024 (wrap_io, LCG runs; the seeded runs reproduce these -- see the tables), i.e. 25 x (N/2) x log2(N) plus the
   load/store loops (2N + N beats). Calls do not overlap: the call interval is latency + 1. Without buffers
   (`wrap_io=False`, "raw") every array access is an m_axi transaction and the same kernel takes 63 849 cycles at
   N = 256.
2. **Correctness and ordering.** The RTL output passes against `numpy.fft.fft` only when the reference is permuted
   to bit-reversed order (max |err| ~5e-6 .. 2e-5, max normalised error ~1e-7); the natural-order comparison fails
   (max |err| ~1e2), which confirms the DIF bit-reversed output.
3. **Resources.** 6.4-6.6k LUT, 7.7-7.8k FF, 16 DSP, 8-14 BRAM18 (HLS estimates) for the wrapped kernel, essentially
   independent of N (only the buffers grow); the routed numbers are in the `pnr_ooc` rows and include the AXI
   adapters.
4. **Relation to HP-FFT.** At N = 256 the Allo kernel needs ~26.3k cycles per transform against 140.5 cycles for
   HP-FFT UF1 (steady state), ~185x slower, for ~3.5x fewer LUTs and 4.5x fewer DSPs (HLS estimates); it is a
   sequential MachSuite port, not a parallel design, and offers no unroll knob, so a same-budget comparison against a
   UF sweep is not meaningful. It is natural-order in, bit-reversed out; HP-FFT is natural-order both ways. No new
   design was written for this package.
5. **Deviations from a stock Allo run, all listed:** `configs["num_output_args"]=2`; `depth=` appended to the
   generated `m_axi` pragmas for cosim; the E2 testbench instead of Allo's XRT `host.cpp`; `csim_design`/`cosim_design`
   added to Allo's `run.tcl`; Allo's clock string 3.33 ns vs 3.333 ns in the OOC P&R; 4 (LCG) / 6 (seeded) calls per
   cosim (the kernel is not pipelined across calls, so more calls add nothing but simulation time: a 1024-point call
   is 130k cycles).
6. The "raw" (`wrap_io=False`) variants are extras: their seeded cosim was not rerun (the previous agent's LCG rows
   stand, 4 transforms validated) and their P&R is queued after everything else; any `not_run` P&R row was still in
   `queue_pnr/pending/` at assembly time (`finalize2.sh` picks it up later).

## Files

`results.csv` / `results.json` (cosim and pnr_ooc rows, typed JSON with a `meta` block), `hls_estimates.csv` (csynth
rows), `reports/<run_id>/` (hls.log, csynth.rpt, solution1_data.json, kernel.cpp/kernel.h/tb.cpp/run_e2*.tcl,
fft_cosim.rpt, lat.rpt, result.transaction.rpt, stimulus map, validator verdicts; util/timing/route-status reports,
vivado.log and impl.tcl for P&R), `validation/<run_id>/` (inputs/outputs of the RTL and of the C model, hex float32;
validator; JSON verdicts per seed), `scripts/`, `summary.md` (tables below).

## Results tables (rendered from results.csv / hls_estimates.csv on 2026-09-06 23:09 EDT; re-run /scratch/hc676/e2_hpfft/finalize2.sh to refresh)

### RTL cosimulation (xsim), cycles at 3.333 ns -- primary rows

| N | config | status | stimulus | RTL vs numpy | max abs err | first out (cyc) | 1st transform complete (cyc) | steady interval (cyc) | fed | completed | run by | failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | strided_fft FFT_SIZE=128 wrap_io=True | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 4.872e-06 |  | 11586 | 11559.0 | 4 | 4 | previous_agent |  |
| 128 | strided_fft FFT_SIZE=128 wrap_io=False | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 4.872e-06 |  | 27430 | 27403.0 | 4 | 4 | previous_agent |  |
| 256 | strided_fft FFT_SIZE=256 wrap_io=True | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 9.707e-06 |  | 26261 | 26234.0 | 4 | 4 | previous_agent |  |
| 256 | strided_fft FFT_SIZE=256 wrap_io=False | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 9.707e-06 |  | 63849 | 63822.0 | 4 | 4 | previous_agent |  |
| 512 | strided_fft FFT_SIZE=512 wrap_io=True | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 1.453e-05 |  | 58808 | 58781.0 | 4 | 4 | previous_agent |  |
| 512 | strided_fft FFT_SIZE=512 wrap_io=False | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 1.453e-05 |  | 145772 | 145745.0 | 4 | 4 | previous_agent |  |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=True | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 2.187e-05 |  | 130299 | 130272.0 | 4 | 4 | previous_agent |  |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=False | ok | transform 0 = HP-FFT shipped test signal, transforms 1..3 = LCG(0x12345678) U[-1,1) | True | 2.187e-05 |  | 327791 | 327764.0 | 4 | 4 | previous_agent |  |

### Vivado 2023.2 out-of-context synth + place + route at 3.333 ns (post-route numbers)

| N | config | status | LUT | FF | DSP | BRAM18-eq | RAMB36 | RAMB18 | URAM | WNS ns | TNS ns | unrouted | synth s | place s | route s | total s | run by | failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | strided_fft FFT_SIZE=128 wrap_io=True | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 102_pnrallo_128_wrap; optional extra) |
| 128 | strided_fft FFT_SIZE=128 wrap_io=False | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 115_pnrallo_128_raw; optional extra) |
| 256 | strided_fft FFT_SIZE=256 wrap_io=True | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 100_pnrallo_256_wrap; optional extra) |
| 256 | strided_fft FFT_SIZE=256 wrap_io=False | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 113_pnrallo_256_raw; optional extra) |
| 512 | strided_fft FFT_SIZE=512 wrap_io=True | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 103_pnrallo_512_wrap; optional extra) |
| 512 | strided_fft FFT_SIZE=512 wrap_io=False | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 116_pnrallo_512_raw; optional extra) |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=True | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 101_pnrallo_1024_wrap; optional extra) |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=False | not_run |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | P&R queued but not reached before the package was assembled (queue job 114_pnrallo_1024_raw; optional extra) |

### Vitis HLS 2023.2 csynth estimates (hls_estimates.csv; not routed numbers)

| N | config | latency (cyc) | interval (cyc) | LUT | FF | DSP | BRAM18 | URAM | HLS est. ns | budget ns | hls s | csim | C model vs numpy | run by | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | strided_fft FFT_SIZE=128 wrap_io=True |  |  | 6608 | 7802 | 16 | 8 | 0 | 2.431 | 2.43 | 245 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 128 | strided_fft FFT_SIZE=128 wrap_io=False |  |  | 5883 | 6259 | 20 | 8 | 0 | 2.431 | 2.43 | 307 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 256 | strided_fft FFT_SIZE=256 wrap_io=True |  |  | 6562 | 7756 | 16 | 10 | 0 | 2.431 | 2.43 | 291 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 256 | strided_fft FFT_SIZE=256 wrap_io=False |  |  | 5891 | 6265 | 20 | 8 | 0 | 2.431 | 2.43 | 370 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 512 | strided_fft FFT_SIZE=512 wrap_io=True |  |  | 6448 | 7710 | 16 | 12 | 0 | 2.516 | 2.43 | 463 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 512 | strided_fft FFT_SIZE=512 wrap_io=False |  |  | 5895 | 6269 | 20 | 8 | 0 | 2.431 | 2.43 | 554 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=True |  |  | 6464 | 7727 | 16 | 14 | 0 | 2.515 | 2.43 | 723 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=False |  |  | 5899 | 6271 | 20 | 8 | 0 | 2.431 | 2.43 | 958 | True | True | previous_agent | no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays) |

### Directives Vitis HLS 2023.2 ignored, removed or could not honour (hls.log / csynth.rpt)

| N | config | warnings (code, meaning, count) |
|---|---|---|
| 128 | strided_fft FFT_SIZE=128 wrap_io=True | HLS 200-880 II-violation(memory-dependence) x7; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 128 | strided_fft FFT_SIZE=128 wrap_io=False | HLS 200-880 II-violation(memory-dependence) x16; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 256 | strided_fft FFT_SIZE=256 wrap_io=True | HLS 200-880 II-violation(memory-dependence) x7; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 256 | strided_fft FFT_SIZE=256 wrap_io=False | HLS 200-880 II-violation(memory-dependence) x16; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 512 | strided_fft FFT_SIZE=512 wrap_io=True | HLS 200-871 estimated-clock-exceeds-target x2; HLS 200-880 II-violation(memory-dependence) x7; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 512 | strided_fft FFT_SIZE=512 wrap_io=False | HLS 200-880 II-violation(memory-dependence) x16; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=True | HLS 200-871 estimated-clock-exceeds-target x2; HLS 200-880 II-violation(memory-dependence) x7; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
| 1024 | strided_fft FFT_SIZE=1024 wrap_io=False | HLS 200-880 II-violation(memory-dependence) x16; HLS 200-960 cannot-flatten-loop x1; RTGEN 206-101 rtgen-warning x1 |
