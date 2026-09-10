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
六-hour wall was the binding constraint rather than heap -- but the run was
sitting at 2.8 GB against `SBT_OPTS="-Xmx4G"`, close enough to the ceiling
that garbage collection was likely taking a real share of the time.

## Attempt 2: running

`scripts/gemmini_s32_long.sh`, same driver and same check, with a 24-hour cap
and `-Xmx24G`. Ten minutes in it was already resident at 6.5 GB, well past
what the first attempt could have allocated, which supports the
garbage-collection reading above.

## If it fails again

In preference order: install Verilator and switch the backend, which is the
real fix and needs the machine's owner to agree; or emit the Verilog and drive
it under xsim, which is available but means re-implementing the req/resp
protocol, and this experiment has already recorded that getting that protocol
wrong yields well-formed rows of **zeros** with correct handshaking and
plausible timing. The cell stays blank until a run produces it.
