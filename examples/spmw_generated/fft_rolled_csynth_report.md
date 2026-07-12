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

## Scope / caveats
- Small-N proof case (N=8, 16). The folded lane buffers are `(S+1)·N` fully-partitioned registers per
  lane family; a resource-optimal conflict-free bank scheme (the `feature/allo-fft` stage-dependent
  XOR swizzle) would reduce the register footprint at larger N — tracked as a queued refinement.
- Latency/II above is the butterfly-loop initiation interval from the per-loop pipeline reports;
  BRAM/URAM are 0 because the lane state is register-partitioned at this scale.
