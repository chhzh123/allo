# Gemmini WS 32x32 cycles: not measured yet, and why

`results.csv` leaves the 32x32 Gemmini WS cycle cell blank. It is not a
failure of the design or the driver, and it is not an estimate waiting to be
filled in from the 4x4/8x8/16x16 trend.

## What has been measured

| Array | Cycles (preload + compute) | Wall time | Netlist |
|---|---:|---:|---:|
| 4x4 | 17 | minutes | -- |
| 8x8 | 33 | minutes | -- |
| 16x16 | 65 | 9,506 s | 2.8 MB lowered FIRRTL |
| 32x32 | -- | capped | 10.6 MB lowered FIRRTL |

All three measured sizes pass the driver's golden-matmul check
(`correct=true`).

## Why 32x32 is hard

`MeshDriver.scala` calls `chiseltest.RawTester.test` with no annotations,
which selects **Treadle**, a FIRRTL interpreter. Interpretation cost grows
with netlist size, and 32x32's lowered FIRRTL is 3.8x that of 16x16, which
itself took two and a half hours. This is also why the original sweep, which
wrapped every size in `timeout 5400`, returned 4x4 and 8x8 and nothing for the
larger two: they were killed mid-simulation and emitted no output at all, so
there was no error line to find either.

## Attempt 1: hit a six-hour cap

`rc=124` after 21,600 s, no cycles. No `OutOfMemory` in the log, so the
six-hour wall was the binding constraint rather than heap -- but the run was
sitting at 2.8 GB against `SBT_OPTS="-Xmx4G"`, close enough to the ceiling
that garbage collection was likely taking a real share of the time.

## Attempt 2: also hit its cap

`scripts/gemmini_s32_long.sh`, same driver and same check, 24-hour cap and
`-Xmx24G`. **`rc=124` after 86,400 s, no cycles, and `oom_lines=0`.** The
larger heap was used -- it reached 6.9 GB against the first attempt's 2.8 GB
ceiling -- so the garbage-collection pressure was real and removing it was not
enough. Treadle simply does not finish a 1,024-tile mesh.

Two attempts, 30 hours of simulation, no number. The cell stays blank.

## Where this leaves it

The backend is the whole problem: `MeshDriver` calls `chiseltest.RawTester.test`
with no annotations, so it runs on Treadle, a FIRRTL interpreter, and 32x32's
lowered FIRRTL is 10.6 MB against 16x16's 2.8 MB -- and 16x16 alone took
9,506 s. Nothing about the design or the driver is at fault; all three measured
sizes pass the golden-matmul check.

The options, in preference order, none of them free:

1. **Install Verilator and switch the backend.** The real fix -- it compiles to
   C++ instead of interpreting, and the RTL is identical so the cycle count
   cannot change. It is not installed on the machine and installing it is the
   machine owner's call, not this experiment's.
2. **Emit the Verilog and drive it under xsim**, which is available. This means
   re-implementing the req/resp protocol in a SystemVerilog testbench, and this
   experiment has already recorded what that costs when it goes wrong: a wrong
   protocol makes the mesh emit well-formed rows of **zeros** with correct
   handshaking, correct row count and plausible timing. It caught the driver's
   author three times.
3. **Leave the cell blank**, which is where it is.

It is not being filled from the 17/33/65 trend. Those fit `4*dim+1` exactly and
would "give" 129, and a curve through three points is not a measurement.