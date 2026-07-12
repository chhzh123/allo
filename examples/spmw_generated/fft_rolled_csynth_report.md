# SPMW key-form FFT — rolled HLS csynth report (archived L3 evidence)

Vitis HLS 2023.2, part `xcu280-fsvh2892-2L-e`, 300 MHz target (`create_clock -period 3.33`).
Generated from `spmw.build(_fft_region(N, fold=...), target="rolled")` → `csynth_design`
(`tests/dataflow/spmw/test_fft.py::_fft_region`). Reproduced by
`test_fft_folded_rolled_csynth_ii1` (II=1) and
`test_fft_folded_rolled_body_count_constant_across_scale` (constant body count).

## The synthesis-time win (AC-4 / DEC-4): O(#roles), and where it holds

| Design | N | fold | synthesized `bfly` bodies | DSP | FF | LUT | BRAM/URAM | butterfly-loop II | csynth wall-clock |
|--------|---|------|---------------------------|-----|----|-----|-----------|-------------------|-------------------|
| **folded** | 8  | `{1:4}` (full) | **2** | **44** | 10372 | 5729  | 0 / 0 | **1** | 21.4 s |
| **folded** | 16 | `{1:8}` (full) | **2** | **44** | 16086 | 11483 | 0 / 0 | **1** | 142.8 s |
| spatial | 8  | none | 12 | 174 | 28915 | 18121 | 0 / 0 | — (fully unrolled) | 17.0 s |
| spatial | 16 | none | 32 | 490 | 77331 | 47959 | 0 / 0 | — (fully unrolled) | 24.8 s |

## Folded FFT (N=8) — top module & per-role area/latency (machine-parseable, systolic schema)

Actual `csynth_design` of the folded N=8 FFT (`fold={1:4}`, full fold) via the O(#roles) rolled path
(`target="rolled"`), reproduced by `test_fft_folded_rolled_csynth_ii1`. These two tables use the **same**
`| Metric | Value |` and `| role | instances | LUT | FF | DSP | BRAM | URAM | latency |` schema as
`systolic_rolled_perf_report.md`, so `allo.backend.perf.load_csynth_report` loads the FFT into the
identical `CsynthEvidence` (top resources + latency/II + per-role areas) the systolic / Mini-TPU use —
the FFT is a **first-class** evidence source, not DSP-only.

### Top module `top` — performance & resources (actual csynth)
| Metric | Value |
|--------|-------|
| Pipeline type | dataflow |
| Latency | 91 cycles |
| Interval (II) | 92 |
| Estimated Fmax | 349 MHz |
| LUT | 5729 |
| FF | 10372 |
| DSP | 44 |
| BRAM | 0 |
| URAM | 0 |
| csynth wall-clock | 21.4 s |

### Per-role modules (`Σ(role_area × instances)` inputs — machine-parseable)
The folded FFT time-shares one physical butterfly PE — synthesized as two bodies, `bfly`/`bfly_1`, the
emitter's even/odd twiddle specializations — across all `S·HALF` butterflies, and runs `S = log2(N) = 3`
sequential stage loops (`top_Pipeline_VITIS_LOOP_54/58/62`, latency 26 each). All instances are 1: the
physical PE and the 3 stage loops are **not** replicated (folding is time-multiplexing, not spatial
replication), which is exactly why the compute area is O(#roles) and constant as N scales (win table).

| role | instances | LUT | FF | DSP | BRAM | URAM | latency |
|------|-----------|-----|----|-----|------|------|---------|
| bfly | 1 | 1620 | 3185 | 24 | 0 | 0 | 17 |
| bfly_1 | 1 | 1196 | 2485 | 20 | 0 | 0 | 17 |
| stage | 3 | 668 | 1239 | 0 | 0 | 0 | 26 |

### Area/latency law check (`Σ(role_area × instances)` vs actual top)
- **DSP** = 1·24 + 1·20 + 3·0 = **44** — matches the top DSP **exactly** (all compute is the two
  butterfly bodies; the stage loops carry no DSP). This is the FFT compute area law: the butterfly
  bodies are O(#roles), constant as N scales 8 → 16 (win table), so `ΣDSP` is scale-invariant.
- **FF** = 3185 + 2485 + 3·1239 = 9387 — within **~9.5%** of the top 10372 (residual: twiddle/lane ROMs
  72 + top registers 913).
- **LUT** = 1620 + 1196 + 3·668 = 4820 — within **~15.9%** of the top 5729 (top-level muxing/control).
- **Latency** = Σ stage latencies = 3·26 = **78** — within **~14.3%** of the top 91 (the folded FFT runs
  its S=3 stages sequentially; the residual is dataflow fill/drain). The butterfly body itself is
  pipelined at **II=1** (`bfly`/`bfly_1` II=1, the AC-6 hard gate), so the compute is throughput-optimal.

So `Σ(role_area × instances)` reconstructs the folded FFT within a ~20% tolerance (DSP exact) and the
S-stage latency model tracks the actual top latency — the **same** area+latency law validated on the
systolic mesh, now report-backed for the folded key-form FFT (`allo.backend.perf.analyze_fft_sdf`
supplies the matching analytic structure: S sequential stages, HALF butterflies/stage, II=1, a constant
physical-PE count).

## Reading the numbers

- **Folded FFT = the O(#roles) synthesis win.** The synthesized butterfly-body count and the compute
  resource (DSP = 44) are **constant** as N scales 8 → 16 — one physical butterfly PE is time-shared
  across all `HALF` butterflies, and the twiddle is read from a runtime table (`TW0/TW1[idx]`) so HLS
  does **not** clone the body per butterfly. Only the lane-buffer registers (FF/LUT) grow with N.
  Every folded butterfly loop schedules at **II=1** (conflict-free per-stage register arrays).
- **Spatial FFT = full-parallel correctness variant, O(P) bodies.** It passes the per-`(s,b)` twiddle
  as a **compile-time constant argument**, so HLS specializes `bfly` per butterfly instance and the
  synthesized body count grows with the grid (12 → 32 = `S·HALF`). The spatial top proves numerical
  correctness (csim vs `numpy.fft.fft`) and one source-level `bfly` body, but it is not the
  synthesis-time win — that is the folded path above.
- **Contrast with the systolic mesh.** The systolic `pe_interior` has *no* per-instance constant
  args (all inputs are FIFOs), so HLS shares one module even fully unrolled — the systolic rolled top
  is O(#roles) at synthesis. The FFT butterfly differs only because of its per-`(s,b)` twiddle
  constant, which is why folding (runtime twiddle table) is what recovers O(#roles) for the FFT.

## L4 — hardware execution (strict, plan.md §6.8)
§6.8 singles out the FFT for **on-board** `mode="hw"` execution (systolic + Mini-TPU are `hw_emu`). The
L4 run uses the **`fft_spatial` desugar path** (`target="vitis_hls"` → `allo.dataflow` → Vitis Makefile
flow), the same path the systolic twin / Mini-TPU use — distinct from the *rolled* O(#roles) synthesis
evidence above (this is the numerical-correctness-on-silicon rung, not a body-count claim). The four
FFT operands (`Xr/Xi/Yr/Yi`) are complete-partitioned so HLS dataflow sees a single reader/writer per
bank (`build_dataflow` HLS operand-partition spec). N=8, inputs `np.random.seed(42)` (imag=0), compared
to `np.fft.fft` at `rtol=atol=1e-3`. Run on `brg-zhang-xcel` (`XDEVICE =
xilinx_u280_gen3x16_xdma_1_202211_1`, board Device-Ready), Vitis 2023.2.

- **L4-emulation checkpoint (`mode="hw_emu"`, RTL co-sim):** emulation exited cleanly ("All the
  simulator processes exited successfully / Finished execution!"); `Yr/Yi` matched `np.fft.fft`
  (**max abs error 2.98e-07**), wall-clock 751.4 s. In-tree: `test_fft_hw_emu_runs_and_matches_numpy`
  (guarded on `hls.is_available` + `XDEVICE`).
- **L4 on-board (`mode="hw"`, bitstream + XRT on the U280):** `make run TARGET=hw PLATFORM=$XDEVICE`
  (Vivado place&route bitstream + XRT host on the board).

```
XDEVICE=... SPMW_RUN_HW=1 python3 -c "build _fft_region(8) mode=hw; module(inp_re, inp_im, out_re, out_im)"
...
INFO: [v++ 60-586] Created ./build_dir.hw.../fft_df.link.xclbin        # bitstream built
Loading: './build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/fft_df.xclbin'   # programmed onto the U280
FFT_HW match=True maxerr=2.98023e-07 elapsed_s=7255.2
```

- **Result:** the FFT bitstream built, was loaded onto the physical U280 via XRT, ran, and `Yr/Yi`
  matched numpy `np.fft.fft` (**max abs error 2.98e-07**) — the strict §6.8 on-board L4.
- **Wall-clock:** 7255.2 s (~2 h) end-to-end (`mode="hw"` Vivado place&route bitstream build + XRT
  on-board execution) on `brg-zhang-xcel` (Alveo U280, Vitis 2023.2).

In-tree: `test_fft_hw_on_board_matches_numpy` (guarded on `hls.is_available` + `XDEVICE` **and**
`SPMW_RUN_HW=1`, because the bitstream build takes hours).

## Scope / caveats
- Small-N proof case (N=8, 16). The folded lane buffers are `(S+1)·N` fully-partitioned registers per
  lane family; a resource-optimal conflict-free bank scheme (the `feature/allo-fft` stage-dependent
  XOR swizzle) would reduce the register footprint at larger N — tracked as a queued refinement.
- Latency/II above is the butterfly-loop initiation interval from the per-loop pipeline reports;
  BRAM/URAM are 0 because the lane state is register-partitioned at this scale.
- The L4 hardware run exercises the `fft_spatial` desugar realization; the rolled O(#roles) evidence
  (the table above) is a separate csynth artifact and is not re-run at L4.
