# Mini-TPU (§3.4) — L3 csynth report (archived)

Hardware-guaranteed example per `plan.md` §6.8. Generated out of band on `brg-zhang-xcel` and archived
here (the in-tree test `tests/dataflow/spmw/test_mini_tpu.py::test_mini_tpu_csyn_emits_synthesizable_project`
asserts the synthesizable project is emitted; the actual csynth run is captured below).

## Design
The composed §3.4 Mini-TPU: a nested output-stationary systolic **MXU** mesh (`spmw.mesh((Rt, Ct))`,
the §3.1 PE) feeding a distinct 1-D bias+ReLU **activation** stage (`spmw.Grid((Ct,))`) through the
per-column `psum` connection. Dims `Rt=Ct=4, K=8`. Reached via the desugar-to-`allo.dataflow` path
(`spmw.build(..., target="vitis_hls", mode="csyn")`), the same path the systolic twin uses; the
operands are complete-partitioned so HLS dataflow sees a single reader/writer per bank.

## Toolchain / target
- Vitis HLS 2023.2 (Build 4023990), Vivado IP flow
- Target: `xcu280-fsvh2892-2L-e` (Alveo U280, `virtexuplusHBM`)
- csynth `csynth_design` completed cleanly (`EXIT=0`, "All loop constraints were satisfied")
- **Synthesis wall-clock:** `csynth_design` 62.18 s elapsed; total `vitis_hls -f run.tcl` 64.62 s
  (peak allocated memory 1013 MB) on `brg-zhang-xcel`.

## Top module `mini_tpu_df` — performance & resources
| Metric | Value |
|--------|-------|
| Pipeline type | **dataflow** (MXU stage ∥ activation stage) |
| Latency | **73 cycles / 243.09 ns** |
| Interval (II) | **67** |
| Estimated Fmax | **411.37 MHz** |
| BRAM | 8 (~0%) |
| DSP | 112 (1%) |
| FF | 34475 (1%) |
| LUT | 27121 (2%) |
| URAM | 0 |

## Module structure (heterogeneous composition, synthesized as distinct modules)
- **16 `mxu_i_j` PE modules** — the Rt×Ct=16 interior systolic MAC PEs (each an 8-deep `k` MAC loop,
  II=1 on the streaming PEs; the interior forwarding PEs pipeline the accumulation).
- **4 `act_i` activation lanes** — the 1-D bias+ReLU vector engine, one lane per output column (each
  1539 FF / 1222 LUT / 8 BRAM), reading only its own column's psum (the per-column banking intent).
- `load_buf{0,1,2}` operand loaders and `store_res3` output drainer around the dataflow region.

This is a genuine hardware realization of the hierarchy + heterogeneous units + memory/stream
composition §3.4 describes: the MXU and the activation synthesize as their own concurrent dataflow
processes, connected only by the per-column psum streams.

## Synthesis-time scaling — scope of the O(#roles) claim
This Mini-TPU path is the **desugar-to-`allo.dataflow`** path (`target="vitis_hls"`), the same path the
systolic twin's csim/csynth/hw_emu use. That path **clones one HLS function body per grid point** — the
16 distinct `mxu_i_j` PE modules + 4 `act_i` lanes above are O(P), not O(#roles). **The O(#roles)
synthesis-time win is therefore NOT claimed for the Mini-TPU here**; it is scoped to the *rolled*
emitter paths (`target="rolled"`: the systolic rolled top stays 9 role bodies as the grid grows, and
the folded FFT stays a constant `bfly` count — see `fft_rolled_csynth_report.md`). The Mini-TPU's
contribution to the plan is the *hierarchy + heterogeneous units + memory hierarchy* surface reaching
real hardware, not a body-count-scaling claim.

## L4 — hw_emu RTL co-simulation (strict, plan.md §6.8)
Run on `brg-zhang-xcel` (`XDEVICE = xilinx_u280_gen3x16_xdma_1_202211_1`), Vitis 2023.2. Build for
`mode="hw_emu"` and call the module with deterministic inputs (`np.random.seed(0)`), which runs
`make run TARGET=hw_emu PLATFORM=$XDEVICE` (builds the emulation `.xclbin`, runs the OpenCL host under
`XCL_EMULATION_MODE=hw_emu`) and reads `OUT` back:

```
cd <project>; make run TARGET=hw_emu PLATFORM=$XDEVICE
...
INFO: [HW-EMU 06-1] All the simulator processes exited successfully
Finished execution!
HWEMU_RESULT match=True maxerr=2.38419e-07 elapsed_s=752.9
OUT_ROW0 = [2.66427064, 2.18002462, 3.20723176, 1.55044007]
REF_ROW0 = [2.66427064, 2.18002486, 3.20723152, 1.55044007]   # np.maximum(ACT@WGT + bias, 0)
```

- **Result:** the emulation exits cleanly (`EXIT=0`) and `OUT` matches numpy `np.maximum(ACT @ WGT +
  bias, 0)` (**max abs error 2.38e-07**, float round-off).
- **Wall-clock:** 752.9 s end-to-end (`mode="hw_emu"` xclbin build + RTL co-sim + host run) on
  `brg-zhang-xcel`.

Covered in-tree by `test_mini_tpu_hw_emu_runs_and_matches_numpy` (guarded on `hls.is_available` **and**
`XDEVICE`, so it runs where the emulation platform is configured and skips otherwise).

## Ladder status
- **L2 csim**: MATCH vs `np.maximum(ACT @ WGT + bias, 0)` (`test_mini_tpu_vitis_hls_csim`).
- **L3 csynth**: this report (project emitted in-test; synthesis run + resources/latency/II/wall-clock
  archived here).
- **L4 hw_emu**: RTL co-simulation runs and matches numpy — see the L4 section above
  (`test_mini_tpu_hw_emu_runs_and_matches_numpy`).
