# Fig. 13(c), arXiv version: simulating the 16x16 mini-TPU

How long the stream-fed programmable mini-TPU of Section 6.1.2 takes to
simulate one long program, three ways: SPMW's OpenMP simulator (`spmw.build(..., target="simulator")`,
one thread a unit), Vitis HLS's C simulation of the HLS code SPMW emits
(`csim_design` with its defaults), and RTL simulation of the assembled array in
xsim (`scripts/spmw_build_array.py --design ptpu-<program> --cosim`). Measured on
brg-zhang-xcel (48 threads, shared) on 2026-10-23, after the simulator fixes of
`32d1db81` and with the FIFO depth floor of `6268c1d4`.

## What is plotted

Execution time after compilation, seconds; medians, with the runs in brackets.

| Workload | RTL cycles | SPMW | Vitis HLS csim | xsim | csim / SPMW | xsim / SPMW |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA | 65,586 | **1.934** (1.819, 2.049) | 35.57 (35.55-35.64) | 92.3 (92.1, 92.5) | **18.4x** | **47.7x** |
| DeepSeek | 229,426 | **6.94** (5.765, 8.115) | 124.58 (123.84-125.11) | 329.85 (326.6, 333.1) | **18.0x** | **47.5x** |

| Column | Records | Field |
|---|---|---|
| SPMW | `depth/simulator.ndjson`, `size` 16, `depth_floor` 64: two rows a program | `rerun_s` |
| Vitis HLS csim | `csim.ndjson`, `size` 16: two builds a program, two runs each | all four `hot_s` |
| xsim | `cosim.ndjson`, `size` 16: two rows a program; raw logs `logs/cosim_S16_<program>_r<n>.log` | `cosim_steps_s.xsim` |
| RTL cycles | `cosim.ndjson` and the logs' `SPMW CYCLES total=` | `cycles` |

## The workload

`ptpu_workload(16, "llama")` and `ptpu_workload(16, "dsv4")` of
`tests/dataflow/spmw/test_spmw_ptpu.py`, the programs `ptpu-llama` and
`ptpu-dsv4`: one launch of the engine's instruction program computing the
gate/up projection slice of an FFN, 64 tokens times 64 columns of each of
`W_gate` and `W_up`, requantised to int8, with K = 2,048 (LLaMA-3.2-1B) or
K = 7,168 (DeepSeek-V4-Pro). Operands are seeded random int8, so the outputs
are not model inference. Every run checks all 8,192 outputs against the
workload's golden.

The engine is stream-fed: 14 roles, 289 instances, 817 FIFOs at 16x16, the same
hardware for every program (`ptpu/S16/report/same_hardware.txt`).

## What each time is

- **SPMW** (`rerun_s`): one process builds the simulator (`build_s`, 6.9 to
  13.3 s), runs it once (`run_s`) and then again; the second, in-process call is
  the plotted time. `ALLO_SIM_DEPTH=64` raises every channel to at least 64
  entries in the simulator only. Results do not depend on it, it cuts blocking,
  and it does not change the RTL. The plotted runs kept 36 to 40 cores busy
  (`rerun_cpu_s / rerun_s`).
- **Vitis HLS csim** (`hot_s`): `spmw.build(..., target="vitis_hls",
  mode="csim")` writes the whole design as one `kernel.cpp`; `csim_design`
  compiles its default debug `csim.exe` and runs it once; the plotted times are
  two further runs of that `csim.exe`. They include process start-up and
  reading the operands from a file. No call was reordered (`order.reordered`
  is false).
- **xsim** (`cosim_steps_s.xsim`): the xsim command alone. The frontend, HLS of
  the 14 roles (6 at once), Vivado's elaboration of the array, testbench
  generation, IP preparation, `xvlog` and `xelab` are timed apart and excluded.
  It is cycle-accurate; the other two are functional only. Whole commands took
  339.8 to 341.1 s (LLaMA) and 605.4 to 610.3 s (DeepSeek).

## Files

| Path | What |
|---|---|
| `depth/simulator.ndjson` | SPMW at 4x4, 8x8 and 16x16, depth floors 0 and 64, two runs each; the 16x16 floor-64 rows are plotted |
| `csim.ndjson`, `cosim.ndjson` | Vitis csim and xsim at 4x4 and 16x16, two runs each; the 16x16 rows are plotted |
| `depth/csim.ndjson`, `depth/cosim.ndjson` | the 8x8 supplement; `depth/logs/` has its driver log and the cosim logs (added in `207e07b1`) |
| `logs/` | the 4x4 and 16x16 cosim logs and the driver log |
| `simulator.ndjson` | the first SPMW runs, 4x4 and 16x16, at the default depth (no floor) |
| `simulator_before_fix.ndjson` | SPMW before `32d1db81`: the 16x16 pair and one 4x4 LLaMA run |
| `fibers.txt` | hand-kept notes on units as fibers over a worker pool; not plotted |
| `ptpu/S16/` | E9's build of the engine, 2026-10-01 and 02: `source/` (= `tests/dataflow/spmw/test_spmw_ptpu.py`), the per-role HLS C++, wrappers and fabric RTL in `generated/`, its build logs, csynth, cosim-cycle and P&R reports in `report/` |
| `scripts/` | `e9_simtime.py` (one simulator or csim record) and `e9_simtime.sh` (the driver) |

`ptpu/S16/` is the array that the cosim command builds and xsim runs. The
2026-10-23 runs rebuilt it in fresh directories, which were not kept; their logs
report the same 14 roles, 289 instances and cycle counts. `ptpu/S16/` predates
`32d1db81`, so it is not byte-for-byte what those runs generated. The single
`kernel.cpp` that csim compiled was not kept.

## Records that are not plotted

- **The depth floor.** At the default depth the same 16x16 runs take 9.38 to
  9.45 s (LLaMA) and 31.91 to 32.13 s (DeepSeek) hot
  (`depth/simulator.ndjson`, `depth_floor` 0), or 9.12 to 9.30 s and 30.97 to
  31.22 s in `simulator.ndjson`.
- **The fix of `32d1db81`.** Before it, 16x16 took 11.04 to 11.30 s (LLaMA) and
  35.93 to 36.59 s (DeepSeek) hot, and 4x4 LLaMA 61.7 s against 1.75 to 1.83 s
  after.
- **Fibers.** The records carry no `fibers` field: they ran one thread a unit.
  The fiber runs are only the notes in `fibers.txt`.
- **4x4 and 8x8**, all three ways. The 4x4 cycles are 1,048,590 (LLaMA) and
  3,670,030 (DeepSeek); 8x8, 262,170 and 917,530.

## Caveats

- **The machine is shared.** The load average is recorded at the end of each
  run: 1-minute values of 29 to 60 for the plotted SPMW runs (much of it their
  own threads), 6.0 to 13.4 for csim and 5.3 to 6.9 for xsim at 16x16. The two
  DeepSeek SPMW runs differ by 1.4x (5.77 and 8.12 s), so their median is
  their mean. At 4x4 the load reached 98, and xsim's two runs differ by up
  to 2x (72.1 and 107.8 s; 179.8 and 356.4 s).
- **The timers are not the same kind.** SPMW is timed in-process; Vitis csim
  includes process start-up and operand-file input; xsim is one step of a
  longer command.
- **SPMW and csim check function, not cycles.** Only xsim models RTL timing.
- **The kept records are older than the scripts.** `e9_simtime.py` now also
  writes `fibers`, `runs`, `vitis_run` and `hot_runs`; these records predate
  them.

## Reproducing

On brg-zhang-xcel. The driver's `T` is the checkout it imports
(`/scratch/hc676/simtime/allo`), `SIMTIME_OUT` is where records go, and it sets
`PYTHONPATH` to `T` and `T/tests/dataflow/spmw` and raises the stack limit,
without which Vitis's C simulation dies at once.

    SIZE=16 PHASES=simulator scripts/e9_simtime.sh 2 llama dsv4      # SPMW at DEPTHS="0 64" -> simulator.ndjson
    SIZE=16 PHASES="csim cosim" scripts/e9_simtime.sh 2 llama dsv4   # Vitis csim and xsim -> csim/cosim.ndjson, logs/
    ALLO_SIM_DEPTH=64 python3 scripts/e9_simtime.py llama 16          # one SPMW record
    python3 scripts/e9_simtime.py dsv4 16 csim <dir>                  # one csim record

`REORDER=1` moves the generated calls into producer order before csim; no
record here used it. The tree before this layout, with the other sizes' builds
and a controlled rerun, is commit `c20b2ee8` (`experiments/e9_vta/`).
