# E9 simulation time, remeasured on a quiet machine

The question: is the 8x8 engine's being the fastest to simulate (hot, depth
floor 64) an artefact of other users' load? The numbers in `../depth/` were
taken one size after another on the shared machine, and their only record of
its state is the load average, which counts the simulator's own threads.

`scripts/e9_simtime_controlled.py` takes every case again with all three sizes
interleaved in rotating order. Each run waits until the whole machine is using
at most a set number of cores (the gate). A run is taken again, up to three
times, if the rest of the machine used more than a set number of cores during
it (the acceptance limit). Each record carries that reading for every timed
run: the machine's busy time less the simulator's own. `ext_load.py --monitor`
also logs other users' CPU use every two seconds. Driver logs are in local
time (EDT until 1 Nov 02:00, EST after); the records use epoch seconds.

## gate3: 26-27 Oct, gate and acceptance limit 3 cores (stopped)

It reached its 30-hour deadline with 3 runs accepted; the first waited 22.9
hours for the gate to open. Other users were almost never under 3 cores:
`monitor_10min.ndjson` summarises the monitor in 10-minute buckets (users
labelled, not named), and none of the 181 has a median under 3 cores. 13 are
under 4 and 71 under 8.

Depth floor 64; median and range over the five hot runs of one process, and
others' cores over all six runs:

| Program | Engine | Threads | Hot s | Cold s | Others' cores |
|---|---|---:|---:|---:|---:|
| LLaMA | 4x4 | 42 | 1.70 (1.50-2.05) | 12.7 | 2.56 (2.38-2.74) |
| LLaMA | 8x8 | 114 | 0.90 (0.84-0.91) | 8.2 | 2.39 (1.97-2.86) |
| DeepSeek | 4x4 | 42 | 6.33 (5.66-6.55) | 39.9 | 2.48 (2.45-2.61) |
| DeepSeek | 8x8, rejected | 114 | 2.82 (2.74-2.98) | 21.7 | 4.46 (3.46-5.35) |

On a quiet machine 8x8 is still about twice as fast as 4x4, as in
`../depth/` (LLaMA 1.56-1.76 s against 0.90-1.00). No 16x16 case ran.

## gate4: from 1 Nov, gate 4 cores (running)

The same driver, with the simulator phases at a 4-core gate and acceptance
limit, then Vitis csim and xsim cosimulation at 12. Their timed steps run on
one thread for minutes at a time: a few busy cores barely touch them, but a
4-core window that long almost never comes.

    E9_GATE=4 E9_ACCEPT=4 scripts/e9_simtime_controlled.sh <tree> <out> sim64 cores sim0
    E9_GATE=12 E9_ACCEPT=12 scripts/e9_simtime_controlled.sh <tree> <out> cosim csim

Its first pass, 1 Nov 01:40-01:46, one process a case so far
(`gate4/records.ndjson`, columns as above). A run is accepted on its mean
others' cores, the build's reading included, which is why DeepSeek 4x4 passes
with a median above 4:

| Program | Engine | Threads | Hot s | Cold s | Others' cores |
|---|---|---:|---:|---:|---:|
| LLaMA | 4x4 | 42 | 1.76 (1.65-2.03) | 12.9 | 3.46 (3.24-3.96) |
| LLaMA | 8x8 | 114 | 0.90 (0.83-0.91) | 8.0 | 3.17 (2.84-3.48) |
| LLaMA | 16x16 | 354 | 1.90 (1.15-2.21) | 12.8 | 3.06 (3.02-3.19) |
| DeepSeek | 4x4 | 42 | 5.70 (5.67-6.45) | 39.8 | 4.14 (3.37-4.62) |
| DeepSeek | 8x8 | 114 | 2.77 (2.72-2.91) | 21.3 | 3.09 (2.99-3.46) |

With the three sizes measured minutes apart on a quiet machine, 8x8 is still
the fastest and 16x16 is as slow as 4x4: the pattern comes from the
simulator, not from other users' load.

## busy_machine_example.ndjson

The driver's smoke test, just before gate3, with the limits set so nothing
waited, while other users held 43 to 47 of the 48 cores. A LLaMA 4x4 simulator
run took 76 to 147 s instead of about 1.7, and Vitis's csim 66 to 69 s instead
of 41 to 43.
