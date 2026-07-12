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

## Ladder status
- **L2 csim**: MATCH vs `np.maximum(ACT @ WGT + bias, 0)` (`test_mini_tpu_vitis_hls_csim`).
- **L3 csynth**: this report (project emitted in-test; synthesis run + numbers archived here).
- **L4 hw_emu**: emulation project emitted (`test_mini_tpu_hw_emu_emits_emulation_project`); the RTL
  emulation run is out of band.
