# E9 simulation time: the programmable mini-TPU's long programs

E1 measures how long it takes to simulate a design whose launch is 16 to 100
cycles (`../e1_gemm/SIM_TIME.md`). This is the other end: the programmable
mini-TPU of this experiment (`ptpu`, `tests/dataflow/spmw/test_spmw_ptpu.py`)
running the two programs E9 cosimulated, whose launches are 65 thousand to
3.7 million cycles -- the longest simulations in any of these experiments.

| Program | 4x4 engine | 16x16 engine |
|---|---:|---:|
| LLaMA-3.2-1B's FFN gate and up projections | 1,048,590 cycles | 65,586 |
| One DeepSeek-V4-Pro routed expert | 3,670,030 | 229,426 |

Three ways, as in E1: SPMW's parallel OpenMP simulator
(`spmw.build(..., target="simulator")`), Vitis HLS's own C simulation
(`csim_design`, with its defaults, on the HLS code SPMW emits), and RTL
cosimulation in xsim through `spmw_build_array.py --design ptpu-<program>
--cosim`, E9's own command.

Measured on brg-zhang-xcel (48 cores, shared) on 2026-10-23, after the
simulator fixes of 32d1db81, two runs each by `scripts/e9_simtime.sh`; every
run passes against the workload's golden, and every cosimulation counts the
cycles E9 recorded. Raw records in `sim_time/`.

## Functional simulation

**Cold** is a fresh process from the design to checked results; **hot** runs
an already-compiled simulation again.

| Program | Cycles | Simulator, cold | Vitis csim, cold | **Simulator, hot** | **Vitis csim, hot** | Hot speedup |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA, 4x4 | 1,048,590 | 12.8-13.1 s | 59.3-60.3 s | **1.75-1.83 s** | 40.7-42.8 s | **23x** |
| DeepSeek, 4x4 | 3,670,030 | 40.4-41.7 s | 189.5-192.2 s | **6.34-6.35 s** | 143.3-150.3 s | **23x** |
| LLaMA, 16x16 | 65,586 | 21.2 s | 50.1 s | **9.12-9.30 s** | 35.6 s | **3.9x** |
| DeepSeek, 16x16 | 229,426 | 49.2-49.7 s | 145.3-147.6 s | **30.97-31.22 s** | 123.8-125.1 s | **4.0x** |

**On these programs the simulator wins outright**, cold and hot, at both
sizes -- the opposite of E1, where a 16-cycle launch left it 2.3 times behind
Vitis's C simulation at 32x32. A long launch is what it is for: the FIFOs
stay busy, so a thread rarely waits.

**The futex fix is what made the 4x4 engine fast.** Before 32d1db81 the same
LLaMA program took 61.7 s hot; it now takes 1.79, **34 times shorter**
(`sim_time/simulator_before_fix.ndjson`). At 16x16 the same fix was worth
1.2 times (11.0-11.3 -> 9.1-9.3 s for LLaMA, 35.9-36.6 -> 31.0-31.2 for
DeepSeek), for the reason the next section measures.

## Why 354 threads buy only 4x

The simulator gives every unit a thread: 42 sections at 4x4 and 354 at 16x16
(counted in the lowered IR -- 256 cells and the sequencer, 16 taps, 16 lanes
and the loader and drain units). Normalising the work as **unit-cycles**,
units times simulated cycles, and taking the cores actually busy as the run's
CPU time over its wall time:

| Case | Threads | Cores busy | Simulator | per core | Vitis csim (1 core) | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaMA, 4x4 | 42 | 32.5 | 24.6 M u-c/s | 0.76 M | 1.06 M u-c/s | 23.3x | **72%** |
| DeepSeek, 4x4 | 42 | 32.4 | 24.3 M | 0.75 M | 1.05 M | 23.1x | **72%** |
| LLaMA, 16x16 | 354 | 36.0 | 2.52 M | 0.070 M | 0.65 M | 3.9x | **11%** |
| DeepSeek, 16x16 | 354 | 35.7 | 2.61 M | 0.073 M | 0.65 M | 4.0x | **11%** |

The two programs agree within 2% at each size, so this is a property of the
engine and not of the workload.

- **At 4x4 the simulator is close to ideal.** 42 threads fit in 48 cores, a
  thread sustains 0.75-0.76 M unit-cycles a second against Vitis's 1.06 M on
  one core, and the speedup is about the number of cores busy.
- **At 16x16 a thread does 10 times less useful work.** 36 cores are busy,
  but they spend their time waiting rather than multiplying: 354 threads on 48
  cores is 7.4 times oversubscribed, so a thread holding a core is often
  spinning on a FIFO whose producer is not scheduled -- it runs out its spin,
  sleeps on the futex, and the producer advances only when the OS next
  schedules it. The accounting shows it: 331 s of CPU for 9.2 s of wall clock,
  on 23 M unit-cycles that Vitis does on one core in 36 s.

So the limit is the oversubscription, not the thread count. Running several
units per thread, so the thread count stays near the core count, is the fix
that would address it -- the half of 32d1db81's second change that was not
made. If 16x16 reached the 4x4 efficiency it would be about 25 times faster
than Vitis's C simulation rather than 4.

## RTL cosimulation

| Program | Cycles | **xsim** | Testbench | HLS, 14 roles (6 at once) | Vivado elaborates | End to end |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA, 4x4 | 1,048,590 | **72-108 s** | 22-45 s | 130-208 s | 68-104 s | 356-583 s |
| DeepSeek, 4x4 | 3,670,030 | **180-356 s** | 72-128 s | 123-129 s | 64-71 s | 627-913 s |
| LLaMA, 16x16 | 65,586 | **92.1-92.5 s** | 5.1 s | 123-125 s | 73.8-74.1 s | 340-341 s |
| DeepSeek, 16x16 | 229,426 | **327-333 s** | 17.7-17.9 s | 126 s | 73.3-74.1 s | 605-610 s |

"End to end" is the whole command, as in E1: the frontend, one Vitis HLS run a
role, Vivado elaborating the assembled array, then the testbench, the IP
models, xvlog, xelab and xsim.

**The 4x4 rows are two runs apart, not a range of one quantity.** Another
user's job took the machine to a load average of 112 while they ran, which is
why xsim reports 72 and 108 s for the same LLaMA launch and 180 and 356 s for
the same DeepSeek one. The 16x16 rows, and everything in the functional
tables, ran at a load of 6 to 68, most of that the measurement's own threads.

Against the fixed simulator, xsim is **40 times slower at 4x4** (72-108 s
against 1.79) and **10 times at 16x16** (92 s against 9.2). The testbench
step grows with the launch because it renders every input token and every
expected result as text for `$readmemh`.

## Reproducing

    scripts/e9_simtime.py <program> <size>              # the OpenMP simulator
    scripts/e9_simtime.py <program> <size> csim <dir>   # Vitis HLS's csim
    SIZE=<size> scripts/e9_simtime.sh <reps> llama dsv4 # both, and the cosim

`<program>` is `llama`, `dsv4`, `micro` or `mixed`. `PHASES` picks which of
`simulator`, `csim` and `cosim` run; `SIMTIME_OUT` is where the records go.
The driver raises the stack limit: the generated top copies each operand into
a local array, which is megabytes for these programs, and Vitis's C simulation
dies at once without it. `REORDER=1` moves the generated calls into producer
order before that simulation, which is how the code before 32d1db81 could be
simulated at all; `order.reordered` in the records says whether it was used.
