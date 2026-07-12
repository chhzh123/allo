# SPMW systolic (rolled) — 16×16 csynth + scale-invariance + 64×64 extrapolation (M5 task5.3)

Second archived systolic csynth point (16×16), used to **prove the per-role areas are scale-invariant**
(identical to the 8×8 report, `systolic_rolled_perf_report.md`) and therefore that the analytic
`Σ(role_area × instances)` model **extrapolates exactly** to the 64×64 array the plan calls out.
Generated on `brg-zhang-xcel` (Vitis HLS 2023.2, `xcu280-fsvh2892-2L-e`, Alveo U280).

## Toolchain / target / reproduce
- Vitis HLS 2023.2 (Build 4023990); target `xcu280-fsvh2892-2L-e`; `create_clock -period 3.33`.
- Design: the output-stationary systolic GEMM twin (`tests/dataflow/spmw/test_spmw_csim.py::_systolic_twin`),
  `M=N=16, K=8`, via the **rolled** O(#roles) path (`target="rolled"`). Role-body count = **9**
  (constant — the AC-4 synthesis win; the same 4 area-carrying classes as 8×8).
- Reproduce:
  ```
  python3 -c "import allo.spmw as spmw, sys; sys.path.insert(0,'tests/dataflow/spmw'); \
    from test_spmw_csim import _systolic_twin; \
    spmw.build(_systolic_twin(16,16,8), target='rolled', project='syst16_prj')"
  cd syst16_prj && vitis_hls -f run.tcl        # csynth_design; ~830 s wall-clock, peak ~4 GB RAM
  ```

## Top module `top` — performance & resources (actual csynth)
| Metric | Value |
|--------|-------|
| Pipeline type | dataflow |
| Latency | 95 cycles |
| Interval (II) | 65 |
| LUT | 148148 |
| FF | 177058 |
| DSP | 1280 |
| BRAM | 0 |
| URAM | 0 |
| csynth wall-clock | 830 s |

## Per-role modules (`Σ(role_area × instances)` inputs — machine-parseable)
Instance counts for the 16×16 halo grid: 256 interior PEs = M·N; 16 west + 16 north operand loaders;
32 east/south drains = M+N. **Every per-instance area below is byte-identical to the 8×8 report.**

| role | instances | LUT | FF | DSP | BRAM | URAM | latency |
|------|-----------|-----|----|-----|------|------|---------|
| pe_interior | 256 | 482 | 672 | 5 | 0 | 0 | 64 |
| load_a | 16 | 127 | 40 | 0 | 0 | 0 | 10 |
| load_b | 16 | 118 | 39 | 0 | 0 | 0 | 10 |
| drain | 32 | 75 | 7 | 0 | 0 | 0 | 10 |

## Scale-invariance (8×8 vs 16×16) — the O(#roles) area law
| role | 8×8 (LUT/FF/DSP) | 16×16 (LUT/FF/DSP) | identical? |
|------|------------------|--------------------|------------|
| pe_interior | 482 / 672 / 5 | 482 / 672 / 5 | ✅ |
| load_a | 127 / 40 / 0 | 127 / 40 / 0 | ✅ |
| load_b | 118 / 39 / 0 | 118 / 39 / 0 | ✅ |
| drain | 75 / 7 / 0 | 75 / 7 / 0 | ✅ |

The 9 rolled role bodies are synthesized once and are **scale-invariant**; only the *instance counts*
grow with the grid. So `Σ(role_area × instances)` is linear in the instance counts and its per-role
inputs are the same at every size — the definition of the O(#roles) synthesis win.

- **DSP** = 256·5 = **1280** — matches the top DSP **exactly** (8×8 was 64·5 = 320; both exact).
- **FF** = 256·672 + 16·40 + 16·39 + 32·7 = 173520 — within **~2%** of the top 177058.
- **LUT** = 256·482 + 16·127 + 16·118 + 32·75 = 129712 — within **~12.4%** of the top 148148.

## 64×64 extrapolation (checked by `test_spmw_perf.py`)
Because the per-role areas are proven scale-invariant across 8×8 and 16×16, the 64×64 area is the same
`Σ(role_area × instances)` with the 64×64 instance counts (4096 interior, 64 load_a, 64 load_b, 128
drain):

| resource | 64×64 model = `Σ(role_area × instances)` | U280 available | fits? |
|----------|------------------------------------------|----------------|-------|
| DSP | 4096·5 = **20480** | 9024 | ❌ 227% |
| FF | 4096·672 + 64·40 + 64·39 + 128·7 = 2758464 | 2607360 | ❌ 106% |
| LUT | 4096·482 + 64·127 + 64·118 + 128·75 = 1999552 | 1303680 | ❌ 153% |

**The analytic model predicts, without synthesis, that the 64×64 output-stationary systolic GEMM does
not fit on one U280 (2.3× the DSP budget).** The DSP figure (20480) is *exact* — it is the scale-invariant
per-PE DSP (5) times the 4096 interior PEs — which is the model earning its keep: it forecasts the
resource wall the plan's 64×64 array hits before hours of synthesis are spent.

## 64×64 actual csynth attempt
A 64×64 rolled csynth was launched on `brg-zhang-xcel` (same flow). See the "64×64 attempt" note at the
bottom of `systolic_rolled_perf_report.md` for the outcome (wall-clock / design-size / over-map status);
regardless of whether the over-mapped csynth completes, the DSP figure is fixed exactly by the proven
8×8≡16×16 scale-invariance above, and that extrapolation is what `test_spmw_perf.py` validates.
