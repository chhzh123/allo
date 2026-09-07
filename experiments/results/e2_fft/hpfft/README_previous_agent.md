# E2 baseline: HP-FFT (UCLA-VAST/HP-FFT-HLS) rerun on the Alveo U280 at 300 MHz

Package E2 of the SPMW evaluation. Everything here was produced on brg-zhang-xcel (2x Xeon Silver 4214, 48 threads,
219 GB) with Vitis HLS 2023.2 (SW build 4023990) and Vivado 2023.2 (SW build 4029153), part xcu280-fsvh2892-2L-e,
target clock 3.333 ns (300 MHz). Nothing was run at the authors' 250 MHz / Versal xcvp1802 / Vitis 2024.2 setting;
their own reports for that setting are in `/scratch/hc676/HP-FFT-HLS/results/` and are quoted below only for context.

## Sources and revisions

* HP-FFT: `/scratch/hc676/HP-FFT-HLS`, git `c4611b8` ("final"). Read-only; every run works on a copy under
  `/scratch/hc676/e2_hpfft/<size>/<config>/`.
* Shipped configurations: `n256/` and `n1024/`, each with `UF1 UF2 UF4 UF8 UF16 UF32 no_StagePipeline original_C_style`.
* `n128/` and `n512/` are **derived**, see "128- and 512-point variants".

## What one UF configuration is, physically (from the sources)

`FFT.cpp` is a radix-2 DIT FFT on `std::complex<float>`, natural-order input, natural-order output, forward
transform, no scaling. `FFT_TOP(hls::stream<hls::vector<complex<float>,2*UF>>& in, ...& out)` is a dataflow region
(`#pragma HLS dataflow disable_start_propagation`) of log2(N)+2 processes connected by ping-pong buffers
(`data_k[N]`, `array_partition cyclic factor=UF`):

1. `reverse_input_stream_UF<k>`: reads N/(2UF) input beats, writes the samples in bit-reversed order into a
   2UF-bank buffer, re-streams them (hand-written per UF; the bank permutation is the bit-reversal of the
   log2(2UF)-bit lane index).
2. `FFT_Stage1_vectorstream_parameterize`: stage 1 (twiddle = 1) on the streamed vector, UF butterflies per beat.
3. `FFT_stage_spatial_unroll<s>` for s = 2..log2(N): each stage is a loop nest over N/2 butterflies with
   `#pragma HLS unroll factor=UF` on the butterfly loop and `#pragma HLS pipeline`, plus
   `#pragma HLS performance target_ti=N/(2*UF)`; i.e. **UF radix-2 butterflies per clock per stage**. The twiddle
   multiply is 4 fmul (DSP) + 2 fadd/fsub (fabric, `bind_op impl=fabric`), the butterfly add/sub is 4 fadd/fsub
   (fabric). k = 0 (tw = 1) and k = bflyStep/2 (tw = -j) are special-cased without multiplies.
4. `output_result_array_to_stream`: N/(2UF) output beats.

So a UF configuration has UF*log2(N) butterflies in flight (one per stage per cycle), consumes and produces
**2*UF complex samples (128*UF bits) per clock** at its boundary, and its ideal transform interval is
N/(2UF) cycles. Only the process with the largest interval sets the real interval (dataflow), which is what the
csynth Interval column reports.

Interface (from `csynth.rpt`, HW Interfaces): `in_r`/`out_r` are `ap_fifo` ports (hls::stream default, no
interface pragma in the sources), data width 128*UF bits (`hls::vector<complex<float>,2UF>`, element u in bits
[64u+63:64u], real in the low word), block control `ap_ctrl_hs` (ap_start/ap_done/ap_ready/ap_idle). There is no
AXI anywhere in HP-FFT's top; a system integration would need an AXI-stream/DMA shell around it.

`no_StagePipeline` (their "Baseline2"): array interface `FFT_TOP(complex<float> dataIn[N], dataOut[N])`
(ap_memory ports), one pipelined butterfly loop per stage, stages sequential, ping-pong `data[2][N]`.
`original_C_style`: the plain C loop nest with no directives except trip counts.

## Flow (exact commands)

HLS, per configuration (`/scratch/hc676/e2_hpfft/run_hls.sh <size> <cfg>`, queued 4 at a time):

```
source /work/shared/common/allo/vitis_2023.2_u280.sh; ulimit -n 8192
cd /scratch/hc676/e2_hpfft/<size>/<cfg>; vitis_hls -f project.tcl      # project.tcl sources ../../common.tcl
```

`common.tcl` is the shipped one with three changes: solution name `FFT_300MHz`, `set_part xcu280-fsvh2892-2L-e`
and `create_clock -period 3.333` hard-coded (the shipped file reads them from env.sh), and the trailing
`export_design -flow syn -rtl verilog` removed (Vivado synthesis is done by the OOC P&R below instead, so every
Vivado job goes through one serial queue). `config_compile -unsafe_math_optimizations` is kept as shipped. csim runs
the shipped testbench (2 calls of `FFT_TOP`, batch_size 1; it prints differences but asserts nothing).

RTL cosimulation (`run_cosim3.sh`, see caveat 1): the shipped testbench is replaced by `testbench_e2.cpp`
(design untouched): 32 transforms back-to-back, transform 0 = the shipped test signal, transforms 1..31 =
uniform random in [-1,1) from a fixed-seed LCG, inputs/outputs dumped as IEEE-754 hex.
`cosim_design -setup -trace_level port -rtl verilog`, then the `xelab` line of Vitis' own `run_xsim.sh`, then

```
xsim --noieeewarnings FFT_TOP -tclbatch e2_vcd.tcl   # open_vcd/log_vcd of the DUT ports; run in 2 us slices until
                                                     # the autotb output FIFO has NT*N/(2UF) writes, or done_cnt == NT
```

`cosim_events.py` counts rising `ap_clk` edges in the VCD; an input beat is `in_r_read && in_r_empty_n`, an output
beat `out_r_write && out_r_full_n`. Transform j completes at its last output beat. The RTL output data are the
`out_r_din` values at those handshakes; `validate_fft.py` compares them (as complex128) with `numpy.fft.fft` of the
dumped inputs, `allclose(atol=1e-4, rtol=1e-4)`; `max_norm_error` = max over transforms of max|y-ref| / max|ref|.

Place-and-route (`pnr_ooc.sh`, one Vivado at a time):

```
create_project -force pnr <cfg>/pnr/proj -part xcu280-fsvh2892-2L-e
add_files [glob build/FFT_300MHz/syn/verilog/*.v]; source each *_ip.tcl (floating-point IP as HLS configured it)
create_clock -period 3.333 -name ap_clk [get_ports ap_clk]
synth_design -top FFT_TOP -mode out_of_context; opt_design; place_design; phys_opt_design; route_design
report_utilization; report_utilization -hierarchical; report_timing_summary; report_route_status
```

Out-of-context: no I/O buffers, no pin constraints, ports unconstrained (no input/output delays), default
Vivado directives. `wns_ns`/`tns_ns` are from `report_timing_summary` after routing; a negative WNS means the
routed design does not run at 300 MHz and the row is marked `timing_fail` (fmax ~ 1/(3.333 - WNS)).

## 128- and 512-point variants

The repository has no generator. Size enters only through `#define FFT_NUM` / `#define EXP2_FFT` in `FFT.h`, but
`FFT.cpp` instantiates the stage pipeline by hand (one `static complex<float> data_k[FFT_NUM]`, one
`array_partition` pragma and one `FFT_stage_spatial_unroll<k>` call per stage). The n256 -> n1024 diff shipped by
the authors is exactly: header + two more buffers/pragmas/stage calls + retargeted output call (plus, for UF4 and
UF8 only, unrelated hand-edits of the reorder stage). `gen_size.py` applies that transformation; regenerating
n1024 from n256 (and vice versa) reproduces the shipped files byte-for-byte for UF1/UF16/UF32 and both array
baselines, up to one pragma line for UF2, and up to the authors' reorder-stage hand-edits for UF4/UF8.
n128 is derived from n256 and n512 from n1024 (nearest shipped size). These rows are labelled
`HP-FFT (derived size; see README)` / run ids `hpfft-derived_*`; they are not the authors' artifact, and they were
validated the same way (csim, RTL cosim vs numpy).
## Findings and caveats (read before quoting any number)

1. **HP-FFT's final transform is never flushed.** Under Vitis HLS 2023.2 the top-level `ap_done` of `FFT_TOP` is
   asserted once per transform *except for the last one*: with 32 transforms streamed in, all 4096 input beats are
   consumed, 31x128 output beats appear, and the RTL then idles forever (verified on n256/UF1, n256/UF2, n256/UF4,
   n1024/UF1; a control run feeding 33 transforms completed exactly 32, run id `hpfft_n256_UF1 (control run: 33
   transforms fed)`). The reorder and stage processes are free-running auto-rewind pipelines inside a dataflow region
   with `disable_start_propagation`; HLS says so at synthesis (`HLS 200-656 Deadlocks can occur since process ... is
   instantiated in a dataflow region with ap_ctrl_none or without start propagation and contains an auto-rewind
   pipeline`). Consequences: (a) a plain `cosim_design` never terminates (the authors ship `common.tcl` with
   `cosim_design` commented out); every cosim row here therefore comes from `cosim_design -setup` + a bounded xsim run
   with a VCD (`run_cosim4.sh`), and the "31/32" note in the cosim rows is this effect, not a data error; (b) in a
   system the block only delivers transform k after transform k+1 has been pushed in, so single-transform latency
   through this IP is unbounded without a dummy trailing transform; steady-state throughput is unaffected.
   `steady_interval_cycles` is the median completion-to-completion distance over the last three quarters of the
   completed transforms; the raw per-transform intervals are in `cosim_events.json` (they alternate, e.g. 138/143 for
   n256/UF1, 522/529 for n1024/UF1, because the reorder stage's three phases and the stage PIPO handshakes do not
   divide the transform evenly).

2. **UF4 does not pipeline under 2023.2 (II = latency).** n256/UF4: latency 423, interval 424 (csynth); measured
   400 cycles per transform in RTL against an ideal N/(2UF) = 32. Same for n1024/UF4 (1575/1576) and derived
   n128/UF4 (234/232). Cause (csynth.rpt module table): the UF4 reorder function `reverse_input_stream_UF4` is not a
   dataflow region and contains a 279-cycle, non-pipelined "Loop 1" (8 x 35 cycles) plus a 33-cycle
   `Pipeline_2` with an II violation: these are the constructor-initialisation loops of the *non-static* local
   `complex<float> data_rev_stream[8][32]` / `data_in_cyclic[8][32]` arrays that the UF4 source (unlike UF1/UF2/UF16/
   UF32, which declare them `static`) introduces; 2023.2 keeps them, so that process takes 423 cycles and, being the
   slowest process of the dataflow region, sets the transform interval. The identical numbers appear in my earlier
   250 MHz rerun on the U280 (`/scratch/hc676/hpfft_u280`, II 424), so this is a tool-version effect, not a
   300 MHz effect: the authors' own report (Vitis 2024.2, xcvp1802, 250 MHz) shows n256/UF4 at latency 397 /
   interval 32. Only 2023.2 is installed here, so no 2024.2 control run was possible.

3. **UF8/UF16/UF32 miss their `performance` targets.** csynth reports `perf-target-missed` for a stage loop
   (e.g. n256/UF8: target 16, achieved 48; n1024/UF8: 64 -> 192) and `HLS 214-189 Pipeline directive for loop
   'R_Group_loop' ... removed because the loop is unrolled completely` for the stage whose butterfly-group loop has
   exactly UF butterflies. Resulting intervals: n256/UF8 67 (ideal 16), n1024/UF8 211 (64), n256/UF16 60 (8),
   n1024/UF16 180 (32) — all already at 250 MHz too. The authors' 2024.2/Versal reports show the ideal intervals
   (16, 64, 8, 32). Whatever the paper's "UFx" delivered on 2024.2, on this toolchain the achieved parallelism of
   UF>=4 is far below UF butterflies per stage per cycle; the parallelism column states the *designed* parallelism.

4. **300 MHz timing.** HLS's own estimate exceeds its 2.43 ns budget (3.333 - 0.9 uncertainty) for every UF1
   config and for n1024/UF2, n1024/UF4, n256/UF16, n1024/UF16 and n128/UF8 (`HLS 200-871`, listed per row); HLS
   continues anyway. The number that matters is the post-route WNS in the `pnr_ooc` rows: positive = 300 MHz met
   out-of-context; negative = the row is marked `timing_fail` and the achievable clock is ~1/(3.333 - WNS).
   Directives HLS ignored outright: `HLS 207-5573 performance pragma is ignored, because it is not in a loop` (the
   function-level `#pragma HLS performance` lines), plus the 214-189 removals above; `HLS 200-885 II violation` on the
   UF4 init loops; the full per-config list is in the "Directives ... ignored" table and in
   `reports/<run_id>/harvest.json` (`warning_codes`, `warning_samples`, `perf_pragma_misses`).

5. **Other caveats.** csim in the shipped flow uses the authors' testbench, which prints differences and asserts
   nothing; the `validation_pass` of csynth rows is the C model checked by `validate_fft.py` through the E2
   testbench (same C code). All numbers are HLS-IP-level (no AXI shell, no DMA, no host): the OOC P&R has no I/O
   constraints and the resource numbers exclude any interconnect. `bram_18k_equiv` in `pnr_ooc` rows is
   2*RAMB36 + RAMB18 from `report_utilization`; in csynth/cosim rows it is HLS's BRAM_18K estimate. The HP-FFT
   authors count BRAM as 36K blocks in the paper (their README). The 250 MHz/Versal numbers quoted for context are
   theirs (results/), not reruns. Rows with `status != ok` carry the reason in `failure_reason`; empty cells are
   nulls (not measured), never zeros.

## Results (rendered from results.csv on 2026-09-06 21:38 — re-run /scratch/hc676/e2_hpfft/finalize.sh to refresh)

### csynth (Vitis HLS 2023.2, xcu280, 3.333 ns; HLS estimates)

| N | config | status | latency (cyc) | interval (cyc) | LUT | FF | DSP | BRAM18 | URAM | hls s | C-model vs numpy | max abs err | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | UF1 | ok | 632 | 83 | 21311 | 23503 | 60 | 0 | 0 | 47 | True | 4.446e-06 | HLS timing estimate 2.541 ns exceeds 2.430 ns budget (period 3.333 - uncertainty 0.90); synthesis continued |
| 128 | UF2 | ok | 343 | 52 | 36985 | 41414 | 120 | 0 | 0 | 52 |  |  |  |
| 128 | UF4 | ok | 234 | 232 | 78982 | 73610 | 210 | 0 | 0 | 65 |  |  |  |
| 128 | UF8 | ok | 197 | 43 | 117746 | 129724 | 327 | 0 | 0 | 105 |  |  | HLS timing estimate 2.433 ns exceeds 2.430 ns budget (period 3.333 - uncertainty 0.90); synthesis continued |
| 256 | UF1 | ok | 1291 | 147 | 23217 | 26249 | 72 | 40 | 0 | 47 | True | 1.329e-05 | HLS timing estimate 2.642 ns exceeds 2.430 ns budget (period 3.333 - uncertainty 0.90); synthesis continued |
| 256 | UF1 (control run: 33 transforms fed) | ok | 1291 | 147 | 23217 | 26249 | 72 | 40 | 0 | 96 | True | 1.329e-05 | HLS timing estimate 2.642 ns exceeds 2.430 ns budget (period 3.333 - uncertainty 0.90); synthesis continued |
| 256 | UF2 | ok | 650 | 84 | 45632 | 48101 | 144 | 0 | 0 | 55 | True | 1.329e-05 |  |
| 256 | UF4 | ok | 423 | 424 | 92237 | 85651 | 258 | 0 | 0 | 75 | True | 1.329e-05 |  |
| 256 | UF8 | ok | 321 | 67 | 143535 | 165953 | 423 | 0 | 0 | 118 | True | 1.329e-05 | HLS timing estimate 2.433 ns exceeds 2.430 ns budget (period 3.333 - uncertainty 0.90); synthesis continued |
| 256 | UF16 | ok | 240 | 60 | 269798 | 293511 | 789 | 0 | 0 | 227 |  |  | HLS timing estimate 2.613 ns exceeds 2.430 ns budget (period 3.333 - uncertainty 0.90); synthesis continued |
| 256 | no_StagePipeline | ok | 3172 | 645 ~ 3173 | 5359 | 7551 | 48 | 4 | 0 | 35 |  |  |  |
| 256 | original_C_style | ok | 19753 | 2294 ~ 19754 | 19632 | 18220 | 206 | 4 | 0 | 40 |  |  | HLS timing estimate 2.604 ns exceeds 2.430 ns budget (period 3.333 - uncertainty 0.90); synthesis continued |
| 512 | UF1 | ok | 2718 | 275 | 28854 | 30159 | 84 | 44 | 0 | 51 |  |  | HLS timing estimate 2.642 ns exceeds 2.430 ns budget (period 3.333 - uncertainty 0.90); synthesis continued |
| 512 | UF2 | ok | 1309 | 148 | 51885 | 52429 | 168 | 64 | 0 | 59 |  |  | HLS timing estimate 2.611 ns exceeds 2.430 ns budget (period 3.333 - uncertainty 0.90); synthesis continued |
| 512 | UF4 | ok | 734 | 272 | 111342 | 97760 | 306 | 0 | 0 | 91 |  |  |  |
| 512 | UF8 | ok | 510 | 115 | 181928 | 179656 | 519 | 0 | 0 | 174 |  |  |  |
| 1024 | UF1 | ok | 5809 | 531 | 37166 | 34081 | 96 | 96 | 0 | 59 | True | 2.608e-05 | HLS timing estimate 2.646 ns exceeds 2.430 ns budget (period 3.333 - uncertainty 0.90); synthesis continued |
| 1024 | UF2 | ok | 2736 | 276 | 62600 | 58364 | 192 | 88 | 0 | 78 | True | 2.608e-05 | HLS timing estimate 2.619 ns exceeds 2.430 ns budget (period 3.333 - uncertainty 0.90); synthesis continued |
| 1024 | UF4 | ok | 1575 | 1576 | 115671 | 106246 | 354 | 112 | 0 | 96 | True | 2.608e-05 | HLS timing estimate 2.619 ns exceeds 2.430 ns budget (period 3.333 - uncertainty 0.90); synthesis continued |
| 1024 | UF8 | ok | 946 | 211 | 214522 | 203261 | 615 | 0 | 0 | 167 | True | 2.608e-05 |  |
| 1024 | UF16 | ok | 680 | 180 | 361108 | 460962 | 1173 | 0 | 0 | 546 |  |  | HLS timing estimate 2.613 ns exceeds 2.430 ns budget (period 3.333 - uncertainty 0.90); synthesis continued |
| 1024 | no_StagePipeline | ok | 11018 | 1859 ~ 11019 | 5122 | 7481 | 48 | 13 | 0 | 45 |  |  |  |

### RTL cosimulation (xsim), cycles at 3.333 ns

| N | config | status | RTL vs numpy | max abs err | max norm err | first out (cyc) | 1st transform done (cyc) | steady interval (cyc) | setup s | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | UF1 | ok | True | 4.446e-06 | 1.444e-07 | 679 | 742 | 78 | 66 | note: 31/32 transforms completed; the last transform's outputs are never flushed without a following transform (top-level ap_done never fires, HLS 200-656); intervals measured over the completed ones |
| 256 | UF1 | ok | True | 1.329e-05 | 1.949e-07 | 1400 | 1527 | 143 | 65 | note: 31/32 transforms completed; the last transform's outputs are never flushed without a following transform (top-level ap_done never fires, HLS 200-656); intervals measured over the completed ones |
| 256 | UF1 (control run: 33 transforms fed) | ok | True | 1.329e-05 | 1.949e-07 | 1400 | 1527 | 140.5 | 106 | note: 32/33 transforms completed; the last transform's outputs are never flushed without a following transform (top-level ap_done never fires, HLS 200-656); intervals measured over the completed ones |
| 256 | UF2 | ok | True | 1.329e-05 | 1.949e-07 | 703 | 766 | 75 | 66 | note: 31/32 transforms completed; the last transform's outputs are never flushed without a following transform (top-level ap_done never fires, HLS 200-656); intervals measured over the completed ones |
| 256 | UF4 | ok | True | 1.329e-05 | 1.949e-07 | 782 | 813 | 400 | 68 | note: 31/32 transforms completed; the last transform's outputs are never flushed without a following transform (top-level ap_done never fires, HLS 200-656); intervals measured over the completed ones |
| 256 | UF8 | ok | True | 1.329e-05 | 1.949e-07 | 284 | 299 | 57 | 116 | note: 31/32 transforms completed; the last transform's outputs are never flushed without a following transform (top-level ap_done never fires, HLS 200-656); intervals measured over the completed ones |
| 1024 | UF1 | ok | True | 2.608e-05 | 1.668e-07 | 6298 | 6809 | 529 | 66 | note: 31/32 transforms completed; the last transform's outputs are never flushed without a following transform (top-level ap_done never fires, HLS 200-656); intervals measured over the completed ones |
| 1024 | UF2 | running |  |  |  |  |  |  | 100 | cosim still in progress at assembly time (re-run finalize.sh) |
| 1024 | UF4 | running |  |  |  |  |  |  | 102 | cosim still in progress at assembly time (re-run finalize.sh) |
| 1024 | UF8 | running |  |  |  |  |  |  | 188 | cosim still in progress at assembly time (re-run finalize.sh) |

### Vivado 2023.2 out-of-context place-and-route at 3.333 ns (post-route)

| N | config | status | LUT | FF | DSP | BRAM18 | URAM | WNS ns | TNS ns | synth s | place s | route s | total s | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | UF1 | ok | 22403 | 22244 | 72 | 82 | 0 | 0.283 | 0.0 | 328 | 302 | 99 | 819 |  |
| 256 | UF2 | ok | 45482 | 37809 | 132 | 104 | 0 | 0.317 | 0.0 | 427 | 702 | 516 | 1825 |  |

### Directives Vitis HLS 2023.2 ignored, removed or could not honour (from hls.log / csynth.rpt)

| N | config | warnings (code, meaning, count) |
|---|---|---|
| 128 | UF1 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x8; HLS 200-805 internal-stream-default-depth x1; HLS 200-871 estimated-clock-exceeds-target x3; HLS 207-5573 performance-pragma-ignored(not-in-loop) x4; HLS 214-111 static-in-dataflow-treated-local x9; HLS 214-114 non-canonical-dataflow-region x3 |
| 128 | UF2 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x9; HLS 200-805 internal-stream-default-depth x2; HLS 207-5573 performance-pragma-ignored(not-in-loop) x5; HLS 214-111 static-in-dataflow-treated-local x8; HLS 214-114 non-canonical-dataflow-region x3; HLS 214-358 index-bit-extension x3; SYN 201-303 memory-assignment-not-applied x12 |
| 128 | UF4 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x6; HLS 200-805 internal-stream-default-depth x2; HLS 200-885 II-violation x1; HLS 200-960 cannot-flatten-loop x1; HLS 207-5573 performance-pragma-ignored(not-in-loop) x6; HLS 214-111 static-in-dataflow-treated-local x6; HLS 214-114 non-canonical-dataflow-region x2; HLS 214-189 pipeline-directive-removed(loop-fully-unrolled) x1; HLS 214-358 index-bit-extension x3; SYN 201-303 memory-assignment-not-applied x42 |
| 128 | UF8 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x7; HLS 200-805 internal-stream-default-depth x2; HLS 200-885 II-violation x2; HLS 207-5573 performance-pragma-ignored(not-in-loop) x4; HLS 214-111 static-in-dataflow-treated-local x6; HLS 214-114 non-canonical-dataflow-region x2; HLS 214-189 pipeline-directive-removed(loop-fully-unrolled) x4; HLS 214-358 index-bit-extension x5; SYN 201-303 memory-assignment-not-applied x122; perf-target-missed R_Pair_loop_R_Group_loop: target 8 achieved 24 |
| 256 | UF1 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x8; HLS 200-805 internal-stream-default-depth x1; HLS 200-871 estimated-clock-exceeds-target x7; HLS 207-5573 performance-pragma-ignored(not-in-loop) x4; HLS 214-111 static-in-dataflow-treated-local x10; HLS 214-114 non-canonical-dataflow-region x3 |
| 256 | UF1 (control run: 33 transforms fed) | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x8; HLS 200-805 internal-stream-default-depth x1; HLS 200-871 estimated-clock-exceeds-target x7; HLS 207-5573 performance-pragma-ignored(not-in-loop) x4; HLS 214-111 static-in-dataflow-treated-local x10; HLS 214-114 non-canonical-dataflow-region x3 |
| 256 | UF2 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x9; HLS 200-805 internal-stream-default-depth x2; HLS 207-5573 performance-pragma-ignored(not-in-loop) x5; HLS 214-111 static-in-dataflow-treated-local x9; HLS 214-114 non-canonical-dataflow-region x3; HLS 214-358 index-bit-extension x3; SYN 201-303 memory-assignment-not-applied x13 |
| 256 | UF4 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x6; HLS 200-805 internal-stream-default-depth x2; HLS 200-885 II-violation x3; HLS 200-960 cannot-flatten-loop x1; HLS 207-5573 performance-pragma-ignored(not-in-loop) x6; HLS 214-111 static-in-dataflow-treated-local x7; HLS 214-114 non-canonical-dataflow-region x2; HLS 214-189 pipeline-directive-removed(loop-fully-unrolled) x1; HLS 214-358 index-bit-extension x3; SYN 201-303 memory-assignment-not-applied x44 |
| 256 | UF8 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x7; HLS 200-805 internal-stream-default-depth x2; HLS 200-885 II-violation x2; HLS 207-5573 performance-pragma-ignored(not-in-loop) x4; HLS 214-111 static-in-dataflow-treated-local x7; HLS 214-114 non-canonical-dataflow-region x2; HLS 214-189 pipeline-directive-removed(loop-fully-unrolled) x4; HLS 214-358 index-bit-extension x5; SYN 201-303 memory-assignment-not-applied x127; perf-target-missed R_Pair_loop_R_Group_loop: target 16 achieved 48 |
| 256 | UF16 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x7; HLS 200-805 internal-stream-default-depth x2; HLS 200-885 II-violation x4; HLS 207-5573 performance-pragma-ignored(not-in-loop) x4; HLS 214-111 static-in-dataflow-treated-local x7; HLS 214-114 non-canonical-dataflow-region x2; HLS 214-189 pipeline-directive-removed(loop-fully-unrolled) x4; HLS 214-358 index-bit-extension x5; RTGEN 206-101 rtgen-warning x1; SYN 201-303 memory-assignment-not-applied x326; perf-target-missed R_Pair_loop_R_Group_loop: target 8 achieved 40 |
| 256 | no_StagePipeline | HLS 200-960 cannot-flatten-loop x1 |
| 256 | original_C_style | HLS 200-871 estimated-clock-exceeds-target x1; HLS 200-885 II-violation x4; HLS 200-960 cannot-flatten-loop x1; HLS 214-358 index-bit-extension x1 |
| 512 | UF1 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x8; HLS 200-805 internal-stream-default-depth x1; HLS 200-871 estimated-clock-exceeds-target x8; HLS 207-5573 performance-pragma-ignored(not-in-loop) x4; HLS 214-111 static-in-dataflow-treated-local x11; HLS 214-114 non-canonical-dataflow-region x3 |
| 512 | UF2 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x9; HLS 200-805 internal-stream-default-depth x2; HLS 200-871 estimated-clock-exceeds-target x5; HLS 207-5573 performance-pragma-ignored(not-in-loop) x5; HLS 214-111 static-in-dataflow-treated-local x10; HLS 214-114 non-canonical-dataflow-region x3; HLS 214-358 index-bit-extension x3; SYN 201-303 memory-assignment-not-applied x14 |
| 512 | UF4 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x6; HLS 200-805 internal-stream-default-depth x2; HLS 200-885 II-violation x5; HLS 207-5573 performance-pragma-ignored(not-in-loop) x6; HLS 214-111 static-in-dataflow-treated-local x8; HLS 214-114 non-canonical-dataflow-region x2; HLS 214-189 pipeline-directive-removed(loop-fully-unrolled) x1; HLS 214-358 index-bit-extension x3; SYN 201-303 memory-assignment-not-applied x46 |
| 512 | UF8 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x9; HLS 200-805 internal-stream-default-depth x2; HLS 200-885 II-violation x2; HLS 207-5573 performance-pragma-ignored(not-in-loop) x5; HLS 214-111 static-in-dataflow-treated-local x10; HLS 214-114 non-canonical-dataflow-region x3; HLS 214-189 pipeline-directive-removed(loop-fully-unrolled) x1; HLS 214-358 index-bit-extension x5; RTGEN 206-101 rtgen-warning x19; SYN 201-303 memory-assignment-not-applied x132; perf-target-missed R_Pair_loop_R_Group_loop: target 32 achieved 96 |
| 1024 | UF1 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x8; HLS 200-805 internal-stream-default-depth x1; HLS 200-871 estimated-clock-exceeds-target x9; HLS 207-5573 performance-pragma-ignored(not-in-loop) x4; HLS 214-111 static-in-dataflow-treated-local x12; HLS 214-114 non-canonical-dataflow-region x3 |
| 1024 | UF2 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x9; HLS 200-805 internal-stream-default-depth x2; HLS 200-871 estimated-clock-exceeds-target x6; HLS 207-5573 performance-pragma-ignored(not-in-loop) x5; HLS 214-111 static-in-dataflow-treated-local x11; HLS 214-114 non-canonical-dataflow-region x3; HLS 214-358 index-bit-extension x3; SYN 201-303 memory-assignment-not-applied x15 |
| 1024 | UF4 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x6; HLS 200-805 internal-stream-default-depth x2; HLS 200-871 estimated-clock-exceeds-target x5; HLS 200-960 cannot-flatten-loop x1; HLS 207-5573 performance-pragma-ignored(not-in-loop) x6; HLS 214-111 static-in-dataflow-treated-local x9; HLS 214-114 non-canonical-dataflow-region x2; HLS 214-189 pipeline-directive-removed(loop-fully-unrolled) x1; HLS 214-358 index-bit-extension x3; SYN 201-303 memory-assignment-not-applied x48 |
| 1024 | UF8 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x9; HLS 200-805 internal-stream-default-depth x2; HLS 200-885 II-violation x2; HLS 207-5573 performance-pragma-ignored(not-in-loop) x5; HLS 214-111 static-in-dataflow-treated-local x11; HLS 214-114 non-canonical-dataflow-region x3; HLS 214-189 pipeline-directive-removed(loop-fully-unrolled) x1; HLS 214-358 index-bit-extension x5; RTGEN 206-101 rtgen-warning x35; SYN 201-303 memory-assignment-not-applied x137; perf-target-missed R_Pair_loop_R_Group_loop: target 64 achieved 192 |
| 1024 | UF16 | HLS 200-471 dataflow-form-issues x1; HLS 200-656 possible-deadlock(auto-rewind-in-dataflow) x7; HLS 200-805 internal-stream-default-depth x2; HLS 200-885 II-violation x4; HLS 207-5573 performance-pragma-ignored(not-in-loop) x4; HLS 214-111 static-in-dataflow-treated-local x9; HLS 214-114 non-canonical-dataflow-region x2; HLS 214-189 pipeline-directive-removed(loop-fully-unrolled) x4; HLS 214-358 index-bit-extension x5; RTGEN 206-101 rtgen-warning x7; SYN 201-303 memory-assignment-not-applied x348; perf-target-missed R_Pair_loop_R_Group_loop: target 32 achieved 160 |
| 1024 | no_StagePipeline | HLS 200-960 cannot-flatten-loop x1 |
