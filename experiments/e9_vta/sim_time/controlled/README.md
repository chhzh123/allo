# E9 simulation time, remeasured on a quiet machine

The question: is the 8x8 engine's being the fastest to simulate (hot, depth
floor 64) an artefact of other users' load? The numbers in `../depth/` were
taken one size after another on the shared machine, and their only record of
its state is the load average, which counts the simulator's own threads.

`scripts/e9_simtime_controlled.py` takes every case again with all three sizes
interleaved in rotating order. Each run waits until the whole machine is using
at most a set number of cores (the gate). A run is taken again, up to three
times, if the rest of the machine used more than a set number of cores during
it (the acceptance limit), judged on the mean of its timed runs and its build.
Each record carries that reading for every timed run: the machine's busy time
less the simulator's own. `ext_load.py --monitor` also logs other users' CPU
use every two seconds. Driver logs are in local time (EDT until 1 Nov 02:00,
EST after); the records use epoch seconds.

**Answer: no.** With the three sizes interleaved on a quiet machine, 8x8 is
still the fastest for both programs and 16x16 is about as slow as 4x4. The
cause is what a simulated step costs at 16x16, below.

## gate4: from 1 Nov, gate 4 cores (running)

The simulator phases at a 4-core gate and acceptance limit, then Vitis csim
and xsim cosimulation at 12. Their timed steps run on one thread for minutes
at a time: a few busy cores barely touch them, but a 4-core window that long
almost never comes.

    E9_GATE=4 E9_ACCEPT=4 scripts/e9_simtime_controlled.sh <tree> <out> sim64 cores sim0
    E9_GATE=12 E9_ACCEPT=12 scripts/e9_simtime_controlled.sh <tree> <out> cosim csim

### sim64: depth floor 64, 01:40-01:58

Three processes a case, each a build, a first run and five hot runs. Median and
range over the 15 hot runs, and others' cores over all 18 timed runs. One run
was rejected (LLaMA 4x4, others at 4.1 cores) and taken again.

| Program | Engine | Threads | Hot s | Cold s | Others' cores |
|---|---|---:|---:|---:|---:|
| LLaMA | 4x4 | 42 | 1.76 (1.55-2.07) | 12.5 (12.3-12.9) | 2.33 (2.22-3.96) |
| LLaMA | 8x8 | 114 | **0.87** (0.82-0.92) | 8.3 (8.0-8.5) | 3.13 (1.85-4.78) |
| LLaMA | 16x16 | 354 | 1.63 (1.15-2.21) | 12.8 (12.8-13.6) | 2.40 (1.98-3.19) |
| DeepSeek | 4x4 | 42 | 5.89 (5.43-6.65) | 39.8 (39.7-40.6) | 2.40 (2.28-4.62) |
| DeepSeek | 8x8 | 114 | **2.76** (2.65-2.91) | 21.3 (21.1-21.5) | 3.05 (1.91-4.09) |
| DeepSeek | 16x16 | 354 | 4.92 (4.29-7.66) | 22.2 (20.4-22.7) | 2.81 (1.85-3.29) |

Hot time is the work in unit-cycles (units times cycles), times the CPU a
unit-step costs, over the cores busy. LLaMA, medians of the hot runs:

| | 4x4 | 8x8 | 16x16 |
|---|---:|---:|---:|
| Cycles | 1,048,590 | 262,170 | 65,586 |
| Unit-cycles | 44.0 M | 29.9 M | 23.2 M |
| CPU per unit-step | 1.33 us | 1.16 us | 2.93 us |
| Context switches per unit-step | 0.000 | 0.065 | 0.59 |
| Cores busy | 33.5 | 39.2 | 41.5 |
| Hot | 1.76 s | 0.87 s | 1.63 s |

From 4x4 to 8x8 the work falls by a third and a step costs about the same, so
the time halves. From 8x8 to 16x16 the work falls by a fifth more, but a step
costs 2.5 times as much: the 16x16 engine's threads stop and switch about once
every two unit-steps, against once every fifteen at 8x8.

### cores: pinned to a CPU set, 01:58-02:03

LLaMA hot, floor 64, one process a case under `taskset`: CPUs 0-11 are one
socket's cores, 0-23 both sockets' cores, 0-47 every hyperthread. Hot time,
then CPU and context switches per unit-step:

| Engine | Threads | 12 CPUs | 24 CPUs | 48 CPUs |
|---|---:|---:|---:|---:|
| 4x4 | 42 | 2.00 s (1.96-2.04) | 2.00 s (1.87-2.13) | 1.65 s (1.48-1.66) |
| 8x8 | 114 | 2.69 s (2.32-3.29) | 1.29 s (1.25-1.44) | 0.82 s (0.82-0.83) |
| 16x16 | 354 | 4.83 s (4.64-5.09) | 3.93 s (3.79-4.23) | 1.55 s (1.54-1.82) |
| 4x4 | | 0.49 us, 0.03 | 0.86 us, 0.02 | 1.25 us, 0.00 |
| 8x8 | | 1.05 us, 0.22 | 0.88 us, 0.14 | 1.11 us, 0.05 |
| 16x16 | | 2.47 us, 0.94 | 3.72 us, 1.04 | 2.77 us, 0.63 |

This tested whether 16x16 is expensive only because it has more threads than
CPUs. It is not: on 12 CPUs 8x8 has 9.5 threads a CPU, more than 16x16's 7.4
on 48, and still costs 1.05 us a step against 2.77, switching 0.22 times a
step against 0.63.
More threads a CPU make every engine switch more, but the bigger engine
switches more at the same ratio. Units as fibers, which switch without the
kernel, halved the 16x16 time and doubled the 8x8 one in `../fibers.txt`
(2.07 to 1.03 s, 0.86 to 1.70 s). And 4x4 is barely faster on 48 CPUs than on
12, so most of its CPU time on 48 is spent spinning.

## gate3: 26-27 Oct, gate and acceptance limit 3 cores (stopped)

It reached its 30-hour deadline with 3 runs accepted; the first waited 22.9
hours for the gate to open. Other users were almost never under 3 cores:
`gate3/monitor_10min.ndjson` summarises the monitor in 10-minute buckets
(users labelled, not named), and none of the 181 has a median under 3 cores.
13 are under 4 and 71 under 8.

Depth floor 64; median and range over the five hot runs of one process, and
others' cores over all six runs:

| Program | Engine | Threads | Hot s | Cold s | Others' cores |
|---|---|---:|---:|---:|---:|
| LLaMA | 4x4 | 42 | 1.70 (1.50-2.05) | 12.7 | 2.56 (2.38-2.74) |
| LLaMA | 8x8 | 114 | 0.90 (0.84-0.91) | 8.2 | 2.39 (1.97-2.86) |
| DeepSeek | 4x4 | 42 | 6.33 (5.66-6.55) | 39.9 | 2.48 (2.45-2.61) |
| DeepSeek | 8x8, rejected | 114 | 2.82 (2.74-2.98) | 21.7 | 4.46 (3.46-5.35) |

These agree with gate4 and with `../depth/` (LLaMA 1.56-1.76 s and 0.90-1.00).

## busy_machine_example.ndjson

The driver's smoke test, just before gate3, with the limits set so nothing
waited, while other users held 43 to 47 of the 48 cores. A LLaMA 4x4 simulator
run took 76 to 147 s instead of about 1.7, and Vitis's csim 66 to 69 s instead
of 41 to 43.
