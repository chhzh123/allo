# SPMW systolic (rolled) — L3 csynth perf/area report (archived, for M5 task5.3)

Archived actual csynth evidence for the analytic area+latency model (`allo/backend/perf.py`, M5
task5.3 / AC-8). Generated on `brg-zhang-xcel`.

## Toolchain / target / reproduce
- Vitis HLS 2023.2 (Build 4023990), Vivado IP flow; target `xcu280-fsvh2892-2L-e` (Alveo U280).
- Design: the output-stationary systolic GEMM twin (`tests/dataflow/spmw/test_spmw_csim.py::_systolic_twin`),
  `M=N=8, K=8`, via the **rolled** O(#roles) path (`target="rolled"`). Role-body count = **9**
  (interior + 4 edges + 4 corners) — constant as the grid scales (AC-4).
- Reproduce:
  ```
  python3 -c "import allo.spmw as spmw, sys; sys.path.insert(0,'tests/dataflow/spmw'); \
    from test_spmw_csim import _systolic_twin; \
    spmw.build(_systolic_twin(8,8,8), target='rolled', project='syst_prj')"
  cd syst_prj && vitis_hls -f run.tcl        # csynth_design; ~85 s wall-clock, Fmax 366.43 MHz
  ```

## Top module `top` — performance & resources (actual csynth)
| Metric | Value |
|--------|-------|
| Pipeline type | dataflow |
| Latency | **79 cycles** |
| Interval (II) | 65 |
| Estimated Fmax | 366.43 MHz |
| LUT | **38948** |
| FF | **44690** |
| DSP | **320** |
| BRAM | 0 |
| URAM | 0 |
| csynth wall-clock | 85.25 s |

## Per-role modules (`Σ(role_area × instances)` inputs — machine-parseable)
Each row is one synthesized role/module class, its per-instance area, how many instances the 8×8 halo
grid places, and its module latency. (`instances`: 64 interior PEs = M·N; 8 west + 8 north operand
loaders; 16 east/south drains = M+N.)

| role | instances | LUT | FF | DSP | BRAM | URAM | latency |
|------|-----------|-----|----|-----|------|------|---------|
| pe_interior | 64 | 482 | 672 | 5 | 0 | 0 | 64 |
| load_a | 8 | 127 | 40 | 0 | 0 | 0 | 10 |
| load_b | 8 | 118 | 39 | 0 | 0 | 0 | 10 |
| drain | 16 | 75 | 7 | 0 | 0 | 0 | 10 |

## Area law check (`Σ(role_area × instances)` vs actual top)
- **DSP** = 64·5 = **320** — matches the top DSP **exactly** (the compute is entirely the 64 interior
  MAC PEs; loaders/drains carry no DSP).
- **FF** = 64·672 + 8·40 + 8·39 + 16·7 = 43752 — within **~2%** of the top 44690 (the residual is
  top-level dataflow glue).
- **LUT** = 64·482 + 8·127 + 8·118 + 16·75 = 34008 — within **~13%** of the top 38948 (LUT carries more
  top-level interconnect/control overhead).
So `Σ(role_area × instances)` reconstructs the actual csynth area within a ~15% relative tolerance
(DSP exact). The per-role areas are scale-invariant, so the total is linear in the instance counts
(O(#roles) at synthesis: the 9 role bodies stay constant while the instance counts grow with the grid).

## Latency
The dataflow critical path is dominated by the operand **load** (`load_a`/`load_b`, latency 10) feeding
the **PE** compute (`pe_interior`, latency 64); `load_latency + pe_latency = 74` is within **~6%** of
the actual top latency 79 (the residual is dataflow fill/drain overhead). The Tier-2 token clock's
*abstract* II=1 wavefront depth (`K+M+N-2`) is a different, structural quantity (round 16) — it is not
the HLS-scheduled cycle count, because the rolled PE body here is a non-II=1 sequential K loop.
