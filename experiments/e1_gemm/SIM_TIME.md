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

| Array | Simulator: build | first run | second run | **build + first run** | Vitis csim: SPMW writes the project | `vitis_hls` | of which `csim_design` | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 1.0-1.4 s | 0.14 s | 0.002 s | **1.1-1.5 s** | 0.8-1.0 s | 10.8-10.9 s | 7.1-7.2 s | **11.5-11.8 s** |
| 8x8 | 1.6-2.2 s | 0.35-0.44 s | 0.008-0.048 s | **2.0-2.6 s** | 1.1-1.2 s | 10.9-11.0 s | 7.2-7.3 s | **12.0-12.2 s** |
| 16x16 | 6.6-6.9 s | 2.4-2.7 s | 0.09-0.18 s | **9.0-9.4 s** | 2.2-2.3 s | 11.1-11.2 s | 7.4-7.5 s | **13.3-13.5 s** |
| 32x32 | 63-68 s | 21.8-23.9 s | 1.4-1.5 s | **86-92 s** | 6.9-7.1 s | 11.7 s | 8.0-8.1 s | **18.5-18.8 s** |

The simulator's build is elaboration, lowering and compiling; its first run
includes what the runtime does on first use, and the second run is the
simulation alone. Vitis's `csim_design` is its compile and run; the rest of
`vitis_hls` is the tool starting and setting up the project.

**The simulator is the faster up to 16x16 and the slower at 32x32.** At 4x4
a functional run takes about a tenth of what Vitis's takes, at 16x16 two
thirds; at 32x32 it takes about five times as long. Both of its parts grow
with the array: the build from about 1 second to 63-68, and the first run --
one thread for each of the 1,024 sites and its edge units, on 48 cores, with
the runtime's first-use costs -- to 22-24. Vitis's C simulation runs the same
processes one after another in one thread and grows only from 7 to 8 seconds.
With a build in hand, rerunning the simulator is 1.5 seconds at 32x32.

## RTL cosimulation

| Array | End to end | HLS, 9 roles (8 at once) | Vivado elaborates the array | IP models | xvlog | xelab | **xsim** | **cosimulation** | after HLS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 188 s | 83 s | 64.6 s | 23.6-23.8 s | 3.7 s | 6.1 s | **5.1-5.2 s** | **38.5-38.8 s** | 103 s |
| 8x8 | 195 s | 85-86 s | 67.3-68.1 s | 23.2-23.9 s | 3.5-3.7 s | 6.3-6.8 s | **5.2-5.6 s** | **38.2-40.0 s** | 106-107 s |
| 16x16 | 198-199 s | 83 s | 72.0-73.0 s | 23.2-23.3 s | 3.5 s | 8.5 s | **5.1-5.2 s** | **40.3-40.5 s** | 113 s |
| 32x32 | 237-243 s | 83-85 s | 101-104 s | 23.7-24.0 s | 3.9 s | 15.6-15.9 s | **6.0-6.3 s** | **49.2-50.1 s** | 151-154 s |

"Cosimulation" is the testbench, the IP models (a Vivado batch run that
generates simulation models for the Xilinx IP the roles use, started for
every design), xvlog, xelab and xsim. "After HLS" adds the Vivado elaboration of the assembled
array, which the cosimulation flow runs before simulating.

The simulation itself is 5 to 6 seconds at every size: a launch is 16 to 100
cycles. What grows is elaborating 1,024 instances, in Vivado and in xelab.

## Beside the baselines

E1's AutoSA and Allo cosimulation times (`compile_time.csv`, one run each in
2026-09, Vitis HLS `cosim_design` alone, after `csynth_design`): AutoSA 42,
108, 851 and 13,886 s; Allo 51, 232, 131 and 1,124 s, at 4x4 to 32x32. The
closest SPMW figure is its cosimulation, 38 to 50 s, or 103 to 154 s with the
array's elaboration. They are not the same work -- `cosim_design` generates a
testbench, compiles, elaborates and simulates, and SPMW's flow divides these
differently -- and the baselines' were taken at another time and load, so the
caution in `COMPILE_TIME.md` applies: compare the growth, not a ratio.
