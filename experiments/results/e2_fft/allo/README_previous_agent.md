# E2 baseline: the existing Allo FFT (MachSuite `fft/strided`) through Allo's Vitis HLS flow at 300 MHz

Package E2 of the SPMW evaluation, Allo side. Host, tools and part as for HP-FFT: brg-zhang-xcel, Vitis HLS
2023.2, Vivado 2023.2, xcu280-fsvh2892-2L-e.

## Which Allo FFT this is, and what it computes

Allo tree: `/scratch/hc676/allo` (rsync of worktree `hc/spmw-allo-implementation-99c949`, HEAD `f436658a`,
`sources/commit.txt` of the evaluation records the tree snapshot). The tree contains two FFTs under
`examples/machsuite/fft/`:

* `strided/strided_fft.py` — MachSuite `fft_strided`: iterative **radix-2 decimation-in-frequency**, in-place on two
  `float32[FFT_SIZE]` arrays (real, imag), twiddles passed in as `float32[FFT_SIZE/2]` arrays (real_twid, img_twid).
  Natural-order input, **bit-reversed-order output** (DIF with natural input). `FFT_SIZE` is a module-level constant
  (`psize.json`: full = 1024, small = 64); it can be patched to any power of two before `allo.customize`, exactly as
  `run_test.py` does — so 128/256/512/1024 are all available from the same source. Two nested `while` loops
  (`span`, `odd`), no Allo schedule primitives are applied in the example, no pipelining/unrolling in the source.
* `transpose/transpose_fft.py` — MachSuite `fft_transpose`: fixed **512-point radix-8** (three 8-point passes with
  transposes through a padded scratchpad), natural-order output; not radix-2 and not size-parameterised.

The radix-2 one is the closest to HP-FFT and is what is measured here. It is **not** a streaming design and it is not
natural-order at the output; `validate_fft.py ... bitrev` compares against `numpy.fft.fft(x)[bitrev(k)]`.
Twiddles were supplied as the forward-transform values `W_N^k = exp(-2*pi*j*k/N)`, k < N/2 (the MachSuite test
harness in `run_test.py` feeds `cos(2*pi*k/N), sin(2*pi*k/N)`, i.e. the conjugate direction; the twiddles are inputs,
so the direction is a property of the data, not of the design).

Other Allo FFTs exist on branches of the Allo repository but not in the evaluated tree: `origin/hpfft`
(`examples/fft/fft.py`, a streaming radix-2 dataflow re-implementation of HP-FFT in Allo with an unroll factor) and
`origin/feature/allo-fft` (`tests/dataflow/test_fft.py`, vectorized FFT-256 with F2 bank swizzling). They were not run
in this package (out of scope: the task was the existing MachSuite FFT; they are noted so the comparison can be
extended).

## Flow (exact commands)

```
export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh
python3 /scratch/hc676/e2_allo/gen_projects.py
```

`gen_projects.py` does, for N in {128, 256, 512, 1024} and wrap_io in {True, False}:

```python
strided_fft.FFT_SIZE = N; strided_fft.FFT_SIZE_HALF = N // 2
s = allo.customize(strided_fft.fft)
mod = s.build(target="vitis_hls", mode="csyn", project=f"strided_n{N}_{wrap|raw}",
              configs={"frequency": 300, "device": "u280", "num_output_args": 2}, wrap_io=wrap)
```

`num_output_args=2` is needed because the kernel is in-place (real/img are both read and written and are not the
last arguments; without it Allo's vitis_hls backend raises "Output arguments must appear at the end"). This only
affects Allo's host-code generation, which is not used here. `wrap_io=True` is Allo's default for `vitis_hls`: the
top `fft` becomes load_buf0..3 -> kernel on local BRAM buffers -> store_res0/1, all four arguments on separate
`m_axi` bundles (gmem0..3, 32-bit) with an `s_axi_control` block; `wrap_io=False` ("raw") keeps the same `m_axi`
ports but the kernel loops access them directly.

Allo writes `kernel.cpp`, `kernel.h`, `run.tcl` (`open_solution -flow_target vivado`, `set_part xcu280-fsvh2892-2L-e`,
`create_clock -period 3.33`, `csynth_design`) and an XRT `host.cpp`. Each project was then run with
`run_e2.tcl` = Allo's `run.tcl` with the XRT host replaced by `tb.cpp` (`add_files -tb tb.cpp -cflags
"-std=gnu++0x -DE2_N=<N> -DE2_NT=4"`) and `csim_design` / `cosim_design -trace_level port -rtl verilog` added
around Allo's `csynth_design`. Two things were changed in the generated `kernel.cpp` copies and nowhere else:
`depth=<N|N/2>` was appended to Allo's four `#pragma HLS interface m_axi ...` lines (Vitis refuses to cosimulate an
m_axi port without a depth; it does not change the RTL), and nothing else. Note Allo's `1000/300` is formatted as
`3.33` ns, i.e. 300.3 MHz; the OOC P&R uses 3.333 ns.

`tb.cpp` calls `fft(real, img, real_twid, img_twid)` 4 times (the kernel is not pipelined across calls, so 4 calls
give the first-call latency and three call-to-call intervals); transform 0 = HP-FFT's shipped test signal,
transforms 1..3 unit-scale uniform random, same LCG as the HP-FFT testbench. Cycle numbers come from Vitis'
cosim report (`fft_cosim.rpt`, min/avg/max latency and interval over the 4 calls), since the control interface is
`s_axi_control` (ap_start/ap_done are register bits, not ports).

P&R: `pnr_ooc.sh out.prj/solution1/syn/verilog fft <prj>/pnr` — identical to the HP-FFT recipe.
## Findings and caveats

1. **What HLS makes of the MachSuite kernel.** The two `while` loops have no trip-count bound, so csynth reports
   no latency/interval (`completion_cycles`/`steady_interval_cycles` are null in the csynth rows, with the reason
   in `failure_reason`). Vitis auto-pipelines the inner `while (odd < N)` loop (`VITIS_LOOP_123_2`) but only at
   **II = 25** because of the in-place read-modify-write of `real[]`/`img[]` through BRAM (`HLS 200-880 II
   violation` x7); the outer stage loop is not pipelined. So the design does one radix-2 butterfly every 25 cycles:
   measured per-call latency = 11 586 / 26 261 / 58 808 / 130 299 cycles for N = 128 / 256 / 512 / 1024 (wrap_io),
   which is 25 x (N/2) x log2(N) plus the load/store loops (256+256+128+128 beats for N = 256). Calls do not overlap:
   the call interval is latency + 1. Without buffers (`wrap_io=False`, "raw") every array access is an m_axi
   transaction and the same kernel takes 63 849 cycles at N = 256.

2. **Correctness and ordering.** RTL output checked against `numpy.fft.fft` (double) of the dumped inputs with
   atol = rtol = 1e-4 for 4 transforms per size: passes only when the reference is permuted to bit-reversed order
   (max |err| 4.9e-6 .. 2.2e-5, max normalised error ~1e-7); the natural-order comparison fails (max |err| ~130),
   which confirms the DIF bit-reversed output. Twiddles fed as W_N^k = exp(-2*pi*j*k/N).

3. **Resources.** 6.4-6.6k LUT, 7.7-7.8k FF, 16 DSP, 8-14 BRAM18 (HLS estimate) for the wrapped kernel, essentially
   independent of N (only the buffers grow). The `pnr_ooc` rows include the four `m_axi` adapters and the
   `s_axi_control` block, which are part of what Allo's Vitis flow emits.

4. **Relation to HP-FFT.** At N = 256 the Allo kernel needs ~26.3k cycles per transform against 143 cycles for
   HP-FFT UF1 (steady state), i.e. ~180x slower, for ~3.5x fewer LUTs and 4.5x fewer DSPs (HLS estimates); it is a
   sequential MachSuite port, not a parallel design, and offers no unroll knob, so a same-budget comparison against a
   UF sweep is not meaningful. It is natural-order in, bit-reversed out; HP-FFT is natural-order both ways. No new
   design was written for this package; the branch designs listed above would be the Allo-native streaming
   counterparts if the comparison is extended.

5. **Deviations from a stock Allo run, all listed:** `configs["num_output_args"]=2` (needed for in-place arguments);
   `depth=` appended to the generated `m_axi` pragmas for cosim; `tb.cpp` instead of Allo's XRT `host.cpp`;
   `csim_design`/`cosim_design` added to Allo's `run.tcl`; Allo's clock string is 3.33 ns (300.3 MHz) while the OOC
   P&R constrains 3.333 ns; 4 transforms per cosim (kernel not pipelined across calls, so more calls add nothing).
   Cycle counts are Vitis' cosim transaction report (`lat.rpt`, `result.transaction.rpt`), not a VCD, because the
   block is controlled through `s_axi_control` registers.

## Results (rendered from results.csv on 2026-09-06 21:38 — re-run /scratch/hc676/e2_hpfft/finalize.sh to refresh)

### csynth (Vitis HLS 2023.2, xcu280, 3.333 ns; HLS estimates)

| N | config | status | latency (cyc) | interval (cyc) | LUT | FF | DSP | BRAM18 | URAM | hls s | C-model vs numpy | max abs err | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | fft_strided FFT_SIZE=128 wrap_io=False | ok |  |  | 5883 | 6259 | 20 | 8 | 0 | 307 | True | 4.872e-06 |  |
| 128 | fft_strided FFT_SIZE=128 wrap_io=True | ok |  |  | 6608 | 7802 | 16 | 8 | 0 | 245 | True | 4.872e-06 |  |
| 256 | fft_strided FFT_SIZE=256 wrap_io=False | ok |  |  | 5891 | 6265 | 20 | 8 | 0 | 370 | True | 9.707e-06 |  |
| 256 | fft_strided FFT_SIZE=256 wrap_io=True | ok |  |  | 6562 | 7756 | 16 | 10 | 0 | 291 | True | 9.707e-06 |  |
| 512 | fft_strided FFT_SIZE=512 wrap_io=False | ok |  |  | 5895 | 6269 | 20 | 8 | 0 | 554 | True | 1.453e-05 |  |
| 512 | fft_strided FFT_SIZE=512 wrap_io=True | ok |  |  | 6448 | 7710 | 16 | 12 | 0 | 463 | True | 1.453e-05 |  |
| 1024 | fft_strided FFT_SIZE=1024 wrap_io=False | ok |  |  | 5899 | 6271 | 20 | 8 | 0 | 958 | True | 2.187e-05 |  |
| 1024 | fft_strided FFT_SIZE=1024 wrap_io=True | ok |  |  | 6464 | 7727 | 16 | 14 | 0 | 723 | True | 2.187e-05 |  |

### RTL cosimulation (xsim), cycles at 3.333 ns

| N | config | status | RTL vs numpy | max abs err | max norm err | first out (cyc) | 1st transform done (cyc) | steady interval (cyc) | setup s | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | fft_strided FFT_SIZE=128 wrap_io=False | ok | True | 4.872e-06 | 1.497e-07 |  | 27430 | 27403 |  | cosim report over 4 calls: latency min/avg/max 27388/27398/27430, call interval min/avg/max 27389/27403/27431, total 109597; kernel not pipelined across calls; output in bit-reversed order (validated as such) |
| 128 | fft_strided FFT_SIZE=128 wrap_io=True | ok | True | 4.872e-06 | 1.497e-07 |  | 11586 | 11559 |  | cosim report over 4 calls: latency min/avg/max 11544/11554/11586, call interval min/avg/max 11545/11559/11587, total 46221; kernel not pipelined across calls; output in bit-reversed order (validated as such) |
| 256 | fft_strided FFT_SIZE=256 wrap_io=False | ok | True | 9.707e-06 | 1.273e-07 |  | 63849 | 63822 |  | cosim report over 4 calls: latency min/avg/max 63807/63817/63849, call interval min/avg/max 63808/63822/63850, total 255273; kernel not pipelined across calls; output in bit-reversed order (validated as such) |
| 256 | fft_strided FFT_SIZE=256 wrap_io=True | ok | True | 9.707e-06 | 1.273e-07 |  | 26261 | 26234 |  | cosim report over 4 calls: latency min/avg/max 26219/26229/26261, call interval min/avg/max 26220/26234/26262, total 104921; kernel not pipelined across calls; output in bit-reversed order (validated as such) |
| 512 | fft_strided FFT_SIZE=512 wrap_io=False | ok | True | 1.453e-05 | 1.330e-07 |  | 145772 | 145745 |  | cosim report over 4 calls: latency min/avg/max 145730/145740/145772, call interval min/avg/max 145731/145745/145773, total 582965; kernel not pipelined across calls; output in bit-reversed order (validated as such) |
| 512 | fft_strided FFT_SIZE=512 wrap_io=True | ok | True | 1.453e-05 | 1.330e-07 |  | 58808 | 58781 |  | cosim report over 4 calls: latency min/avg/max 58766/58776/58808, call interval min/avg/max 58767/58781/58809, total 235109; kernel not pipelined across calls; output in bit-reversed order (validated as such) |
| 1024 | fft_strided FFT_SIZE=1024 wrap_io=False | ok | True | 2.187e-05 | 1.450e-07 |  | 327791 | 327764 |  | cosim report over 4 calls: latency min/avg/max 327749/327759/327791, call interval min/avg/max 327750/327764/327792, total 1311041; kernel not pipelined across calls; output in bit-reversed order (validated as such) |
| 1024 | fft_strided FFT_SIZE=1024 wrap_io=True | ok | True | 2.187e-05 | 1.450e-07 |  | 130299 | 130272 |  | cosim report over 4 calls: latency min/avg/max 130257/130267/130299, call interval min/avg/max 130258/130272/130300, total 521073; kernel not pipelined across calls; output in bit-reversed order (validated as such) |

### Directives Vitis HLS 2023.2 ignored, removed or could not honour (from hls.log / csynth.rpt)

| N | config | warnings (code, meaning, count) |
|---|---|---|
| 128 | fft_strided FFT_SIZE=128 wrap_io=False | HLS 200-880 x16; HLS 200-960 x1 |
| 128 | fft_strided FFT_SIZE=128 wrap_io=True | HLS 200-1992 x6; HLS 200-880 x7; HLS 200-960 x1 |
| 256 | fft_strided FFT_SIZE=256 wrap_io=False | HLS 200-880 x16; HLS 200-960 x1 |
| 256 | fft_strided FFT_SIZE=256 wrap_io=True | HLS 200-1992 x6; HLS 200-880 x7; HLS 200-960 x1 |
| 512 | fft_strided FFT_SIZE=512 wrap_io=False | HLS 200-880 x16; HLS 200-960 x1 |
| 512 | fft_strided FFT_SIZE=512 wrap_io=True | HLS 200-1016 x2; HLS 200-1992 x6; HLS 200-871 x2; HLS 200-880 x7; HLS 200-960 x1 |
| 1024 | fft_strided FFT_SIZE=1024 wrap_io=False | HLS 200-880 x16; HLS 200-960 x1 |
| 1024 | fft_strided FFT_SIZE=1024 wrap_io=True | HLS 200-1016 x2; HLS 200-1992 x6; HLS 200-871 x2; HLS 200-880 x7; HLS 200-960 x1 |
