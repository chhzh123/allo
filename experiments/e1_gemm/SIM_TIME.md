# E1 simulation time: the SPMW array, functionally and in RTL

How long it takes to simulate the design of E1's `spmw_mesh` rows -- the int8
output-stationary array, one S x S x S tile a launch (`gemm8`,
`tests/dataflow/spmw/test_spmw_gemm_int8.py`) -- three ways:

- **SPMW's parallel OpenMP simulator**, `spmw.build(..., target="simulator")`:
  the design lowered to Allo's OpenMP simulator, one thread a site and the
  channels as queues, compiled for the host by LLVM.
- **Vitis HLS's own C simulation**: `spmw.build(..., target="vitis_hls")`
  writes the HLS project and `vitis_hls` runs `csim_design`, with its defaults,
  on that code and a testbench that checks every result against a C reference.
  SPMW's own `mode="csim"` compiles the same code with the host compiler
  instead; this is the tool's flow.
- **RTL cosimulation with Vivado**: E1's own command, `spmw_build_array.py
  --design gemm8 --size S --frequency 300 --jobs 8 --cosim`. Each role through
  Vitis HLS (eight at once), the array assembled and elaborated in Vivado, then
  xsim. The script now times each step of the cosimulation (`cost.json`,
  `cosim_seconds`); before 2026-10-22 it timed the HLS and the elaboration but
  not the simulation, which is why E1 had no SPMW cosimulation time.

Measured on brg-zhang-xcel (48 cores, shared, load average 5 to 15) on
2026-10-22, from a private copy of this branch: `scripts/e1_simtime.sh` drives
`scripts/e1_simtime.py` and the build script, a fresh process and a fresh
directory a run. Three runs a size for the simulator, two for the others;
every run passes, and every cosimulation counts E1's cycles (16, 28, 52, 100).
The cells give the range over the runs. The raw records are in `sim_time/`
(the first line of `cosim.ndjson` is a 4x4 run finished before the run count
was lowered to two, and is not in the tables).

## Functional simulation

Two states, the same for both. **Cold**: nothing compiled -- a fresh process
goes from the design to checked results, compiling on the way. **Hot**: the
simulation is already compiled and is run again on the same inputs.

| Array | Cold, simulator: build + first run | Cold, Vitis: SPMW writes the project + `csim_design` | the same with `vitis_hls` starting | Hot, simulator: second run | Hot, Vitis: `csim.exe` again |
|---|---:|---:|---:|---:|---:|
| 4x4 | **1.1-1.5 s** | **7.9-8.2 s** | 11.5-11.8 s | **0.002 s** | **0.005 s** |
| 8x8 | **2.0-2.6 s** | **8.3-8.5 s** | 12.0-12.2 s | **0.008-0.048 s** | **0.006 s** |
| 16x16 | **9.0-9.4 s** | **9.6-9.7 s** | 13.3-13.5 s | **0.09-0.18 s** | **0.011-0.012 s** |
| 32x32 | **86-92 s** | **14.9-15.2 s** | 18.5-18.8 s | **1.39-1.52 s** | **0.052-0.054 s** |

- The simulator's build is elaboration, lowering and compiling; its first run
  adds the runtime's first-use costs, and its second run, in the same
  process, is the simulation alone. Build and first run apart: 1.0-1.4 s and
  0.14 s at 4x4, 1.6-2.2 and 0.35-0.44 at 8x8, 6.6-6.9 and 2.4-2.7 at 16x16,
  63-68 and 21.8-23.9 at 32x32.
- Vitis's `csim_design` compiles the testbench and the design (in debug mode,
  its default) and runs them; SPMW writing the project takes 0.8 to 7.1 s of
  the cold figure, and `vitis_hls` starting and setting up the project 3.6 to
  3.7 s more, which the third column adds. The hot figure runs the `csim.exe` each
  cold run left, in a new process, with the libraries `run_sim.tcl` sets
  (`sim_time/csim_hot.txt`).

**Cold, the simulator is the faster up to 16x16 and the slower at 32x32**:
about 6 times faster at 4x4, 3 to 4 times at 8x8, level at 16x16, 6 times
slower at 32x32. **Hot, Vitis's binary is the faster from 8x8 up**: 1.3 to 9
times at 8x8, 7 to 16 times at 16x16 and 26 to 29 times at 32x32; only at 4x4
does the simulator's in-process call (2 ms) beat a process start (5 ms). The simulator
runs one thread for each of the 1,024 sites and their edge units on 48 cores
and passes every token through a blocking queue; Vitis's C simulation runs
the same processes one after another in one thread. Both grow with the array,
the simulator's build most of all: from 1 second to 63-68.

## Why the simulator slows down, and when it is the faster

Probed on 2026-10-22 with `scripts/e1_simprobe.py`, one run each
(`sim_time/probe.ndjson`).

**Most of the 32x32 cold time is not the simulation.** Of a 69.6 s build,
52.5 s is one Python pass in Allo's simulator, `_process_function_streams`
(`allo/backend/simulator.py`), which for every call in each of the 1,089
functions scans the whole module for its callee -- quadratic in the cells.
SPMW's own lowering is 1.7 s. The first run's 20 s beyond the second are user
time (28 s against 11 s), which looks like LLVM generating the module's
machine code on the first call; that is inferred, not traced.

**The simulation itself is synchronisation.** Every cell is an OS thread, and
a put on a full queue or a get on an empty one loops on `omp flush`,
`taskyield` and `usleep(1)`. The second 32x32 run takes 1.33 s and 52.5 s of
CPU, 41.7 of it in the kernel, for 32,768 multiply-adds; at 16x16, 0.24 s and
10.5 s of CPU with 5.5 million context switches, for 4,096.

**A longer workload turns it around.** The same 16x16 array on E12's tiled
GEMMs (`gemmtiles`), 1,024 and 8,192 steps a cell instead of E1's 16:

| Steps a cell | Simulator, hot | Vitis csim, hot | Simulator, cold | Vitis csim, cold |
|---|---:|---:|---:|---:|
| 16 (E1, 16^3) | 0.09-0.24 s | 0.011-0.012 s | 9.0-9.4 s | 9.6-9.7 s |
| 1,024 (E12, 64^3) | 0.147 s | 0.48-0.49 s | 16.4 s | 11.2 s |
| 8,192 (E12, 128^3) | 1.09 s | 3.63-3.71 s | 17.2 s | 14.8 s |

A short launch is mostly filling and draining the array, when the threads poll
empty queues and every hand-over waits on the scheduler. In a long one the
queues stay full, a thread rarely waits, and the 48 cores share the work;
Vitis's C simulation is one thread and grows with the work. Hot, the simulator
goes from 8 to 20 times slower than Vitis's binary to 3.3 times faster. Cold,
its fixed cost -- a 12 s build for this design -- still leaves it behind at
these lengths. More cells are the other way: the build's quadratic pass, the
first-run compile and the threads beyond the cores all grow with the array.

**Vitis's C simulation needs a sequential order; the simulator does not.**
`csim_design` runs a dataflow region's processes one after another in the
order they are called, and SPMW emits the tiled design's 16 `frame` units
after the 256 cells they feed, so it stops at "an hls::stream is read while
empty". The Vitis figures above are from the same generated code with the
frame calls moved ahead of the cells (`REORDER=1`), which changes nothing in
hardware. The simulator runs the processes concurrently and needs no order.

**Three fixes, recorded and not made.**
- `_process_function_streams` finds each callee by scanning every operation
  of the module; a name-to-function table built once makes it linear, which
  should remove most of the 52 s at 32x32.
- A waiting thread polls with `usleep(1)`; blocking on the queue (a futex or
  a condition variable) would stop idle threads from taking cores, and fewer
  threads than cells, each running several cells in turn, would keep the
  count near the cores.
- SPMW could call a region's processes producer first, so that Vitis's C
  simulation runs the generated code as it is.

## RTL cosimulation

"End to end" is the wall time of E1's one command,

    python3 scripts/spmw_build_array.py --design gemm8 --size S \
        --frequency 300 --jobs 8 --cosim --out <fresh dir>

from launch to exit, taken by the driver with `date` on either side. In order,
it runs:

1. Python and `import allo` (1 to 2 s; the part the steps below leave over).
2. **Frontend**: elaborate the fabric and write one HLS project per role
   (0.7 to 1.8 s).
3. **HLS**: per role -- 9 at every size -- one Vitis HLS run of
   `csynth_design` and `export_design -format ip_catalog`, which packages the
   role as an IP; 8 at once, so two waves of about 40 s each.
4. **Vivado elaborates the array**: one Vivado batch reads the fabric's
   SystemVerilog and each role's RTL and IP, generates the IPs' synthesis
   targets and runs `synth_design -rtl`: elaboration only, a check that the
   assembled array is well formed. `--frequency` is not used for it; it is
   for `--synth` and `--pnr`. The simulation does not need this step, but the
   flow always runs it.
5. **Cosimulation**: render the testbench with the stimulus and the expected
   results; a second Vivado batch imports each role's IP and generates its
   simulation models; `xvlog` compiles the SystemVerilog and the Verilog;
   `xelab` elaborates the testbench and the array into a simulation snapshot;
   `xsim` runs the launch to the end and compares every result.

| Array | End to end | 2. Frontend | 3. HLS, 9 roles (8 at once) | 4. Vivado elaborates | 5. IP models | xvlog | xelab | **xsim** | **step 5** | steps 4 + 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 188 s | 0.7 s | 83 s | 64.6 s | 23.6-23.8 s | 3.7 s | 6.1 s | **5.1-5.2 s** | **38.5-38.8 s** | 103 s |
| 8x8 | 195 s | 0.8-0.9 s | 85-86 s | 67.3-68.1 s | 23.2-23.9 s | 3.5-3.7 s | 6.3-6.8 s | **5.2-5.6 s** | **38.2-40.0 s** | 106-107 s |
| 16x16 | 198-199 s | 0.9 s | 83 s | 72.0-73.0 s | 23.2-23.3 s | 3.5 s | 8.5 s | **5.1-5.2 s** | **40.3-40.5 s** | 113 s |
| 32x32 | 237-243 s | 1.6-1.8 s | 83-85 s | 101-104 s | 23.7-24.0 s | 3.9 s | 15.6-15.9 s | **6.0-6.3 s** | **49.2-50.1 s** | 151-154 s |

The testbench takes under 0.1 s and is not shown. The simulation itself is 5
to 6 seconds at every size: a launch is 16 to 100 cycles. What grows is
elaborating 1,024 instances, in Vivado and in xelab. The IP models are a
Vivado batch the flow starts for every design.

## Beside the baselines

E1's AutoSA and Allo cosimulation times (`compile_time.csv`, one run each in
2026-09, Vitis HLS `cosim_design` alone, after `csynth_design`): AutoSA 42,
108, 851 and 13,886 s; Allo 51, 232, 131 and 1,124 s, at 4x4 to 32x32. With
their `csynth_design` the same span as SPMW's end to end is AutoSA 83, 279,
2,528 and 48,854 s and Allo 110, 311, 555 and 7,102 s. The closest SPMW
figures are its step 5, 38 to 50 s, and its end to end, 188 to 243 s. They are not the same work -- `cosim_design` generates a
testbench, compiles, elaborates and simulates, and SPMW's flow divides these
differently -- and the baselines' were taken at another time and load, so the
caution in `COMPILE_TIME.md` applies: compare the growth, not a ratio.
