# E1's AutoSA designs through TAPA, 2026-11-02

Every AutoSA number elsewhere in E1 (and in E5 and E12) comes from
`--target=autosa_hls_c`: AutoSA writes one Vitis HLS C++ kernel and Vitis HLS
compiles it as one project. AutoSA also has a TAPA backend,
`--target=autosa_tapa`, which writes the same modules as TAPA tasks
([TAPA](https://github.com/UCLA-VAST/tapa) synthesizes each distinct task once and
assembles the top-level RTL itself). This folder rebuilds E1's AutoSA ladder
through that backend and measures cycles, routed area and clock, and compile time
the way E1 does.

**Result:** through TAPA, AutoSA compiles the 32x32 array in 141 s instead of
35,023 s, and its compile time is nearly flat in the array size, because the
design has the same 18 distinct tasks from 8x8 up (one `PE` task for all 1,024
cells). It routes to the same LUTs, fewer registers and a similar clock, but takes
1.04x to 1.46x as many cycles, because its result writes do not burst.

## What was built

The configurations are E1's AutoSA row: `space_time[3]`, `array_part[S,S,S]`,
`latency[1,1]`, `simd[1]`, with `--host-serialize` at 8x8, 16x16 and 32x32 and the
narrow build at 4x4 (`S*/source/cmd`). Tools: AutoSA at `b61a1b4`, TAPA
0.1.20250127 (`/usr/local/bin/tapa` on brg-zhang-xcel), Vitis HLS and Vivado
2023.2, `xcu280-fsvh2892-2L-e`, 3.333 ns, the same as E1.

The 2021 backend does not compile with this TAPA as generated. `scripts/gen.sh`
makes these changes and nothing else (the hardware is AutoSA's):

- **Kernel signatures:** `tapa::read_only_mmap` / `write_only_mmap` become
  `tapa::mmap`. Current TAPA kernels take `tapa::mmap`; the directional types are
  host-side only.
- **The `kernel0` prototype moves from `kernel_kernel.h` into the host file.** In
  the header, TAPA's rewrite of the top task's prototype swallows the first token
  of the next declaration, and Vitis HLS then fails to parse the top task.
- **Host:** `tapa::task().invoke(kernel0, ...)` becomes
  `tapa::invoke(kernel0, <bitstream>, ...)`, which also runs TAPA's RTL
  simulation when given the `.xo`, and the host parses gflags so the `--xosim_*`
  options work. At 4x4 the scalar C port is passed as the mmap itself
  (`std::move(buffer_C[0])`) instead of `.vectorized<1>()`.

Each design passes TAPA's software simulation and its RTL simulation (`tapa
fast cosim`, xsim), checked against AutoSA's own host golden model.

## Cycles, area and clock

Cycles are E1's: first memory beat to last, counted at the rising edge on the
values before it (`../scripts/beats.py`'s rule). TAPA's testbench gets a counter
for that (`scripts/fastcosim_beats.sh`, passed to the host as
`--xosim_executable`). Area and clock come from E1's out-of-context flow on TAPA's
RTL (`scripts/pnr.sh`, the same `pnr.tcl`); `scripts/util.py` reads both flows'
reports the same way and reproduces E1's 8x8 and 16x16 rows exactly. The Vitis
HLS rows are E1's (`../results.csv`).

| Array | Flow | Cycles | LUT | FF | DSP | BRAM18 | WNS | Clock |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | Vitis HLS | 78 | 4,974 | 8,171 | 16 | 6 | +0.512 ns | 354 MHz |
| | TAPA | 81 | 5,134 | 6,393 | 16 | 3 | +1.108 ns | 449 MHz |
| 8x8 | Vitis HLS | 130 | 15,343 | 25,536 | 64 | 46 | +0.738 ns | 385 MHz |
| | TAPA | 156 | 21,111 | 22,448 | 64 | 46 | +0.394 ns | 340 MHz |
| 16x16 | Vitis HLS | 374 | 48,957 | 84,062 | 256 | 46 | +0.380 ns | 339 MHz |
| | TAPA | 507 | 54,752 | 57,541 | 256 | 46 | +0.441 ns | 346 MHz |
| 32x32 | Vitis HLS | 1,222 | 186,369 | 320,772 | 1,024 | 48 | +0.101 ns | 309 MHz |
| | TAPA | 1,786 | 186,473 | 196,873 | 1,024 | 46 | +0.195 ns | 319 MHz |

Both flows meet 300 MHz at every size, so the cycles decide the latency: at 32x32
TAPA's design takes 1.46x as long.

**Where the cycles go.** Both flows move the same beats (1 + 1 + 4 at 8x8, 16 +
16 + 64 at 32x32). Rebuilding E1's 8x8 Vitis HLS design and replaying its cosim
(`scripts/hlsc_beats.sh`, `logs/hlsc_S8_beats.txt`) puts its four C beats in four
consecutive cycles, 126 to 129 cycles after the operands; Vitis HLS infers a
length-4 burst on the `ap_uint<512>` C port. TAPA's C beats start earlier (76
cycles after the operands) and finish later (155). Its task writes C through
`tapa::vec_t<int,16>*` with `offset=direct`, no burst is inferred, and every beat
is its own write transaction (16 address beats for 16 data beats at 16x16,
`logs/S16_write_probe.txt`). The beats leave about 25 cycles apart at every size,
where both flows' HLS schedules put the writer loop at II = 16; whether the write
path or the drain chain feeding it sets the 25 was not traced. The two
testbenches also use different memory models (Vitis cosim's, TAPA's `axi_ram`),
which this does not separate.

## Compile time

`tapa compile` with 8 and with 1 HLS job (`-j`), one run each, against E1's
single Vitis HLS `csynth_design` (`../compile_time.csv`). `scripts/phases.py`
splits TAPA's log into its phases (`logs/phases.txt`).

| Array | Vitis HLS csynth | TAPA, 8 jobs | TAPA, 1 job | TAPA HLS phase (8 jobs) | TAPA RTL assembly | TAPA `.xo` packing | Tasks |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 40.8 s | 76.5 s | 324.3 s | 40.8 s | 2.2 s | 31.1 s | 15 |
| 8x8 | 182.4 s | 95.8 s | 374.0 s | 58.9 s | 3.1 s | 31.3 s | 18 |
| 16x16 | 1,783.4 s | 101.5 s | 376.9 s | 59.1 s | 6.3 s | 33.4 s | 18 |
| 32x32 | 35,022.9 s | 141.3 s | 421.3 s | 58.6 s | 38.3 s | 40.5 s | 18 |

TAPA's HLS phase is flat because the task count is; what grows is its own RTL
assembly (stitching 2,214 task instances at 32x32) and the packing. Each task job
takes 17 to 21 s; their sum is 265 to 333 s.

These are single runs on a shared machine, and the stages are not SPMW's: SPMW's
E1 figure is the HLS phase alone (synthesis and `export_design` of each role),
with its frontend, which writes the fabric RTL, timed apart. Compare stage by
stage, not total against total.

## Files

- `S*/source/`: AutoSA's input (`kernel.c`, `kernel.h`) and command (`cmd`).
- `S*/generated/`: the TAPA kernel and host after `gen.sh`'s changes.
- `S*/report/`: per-task HLS reports (`*_csynth.rpt`), `compile_j8.log`,
  `compile_j1.log`, `cosim.log` (the beat count and the check), and the routed
  `util.rpt`, `timing.rpt` with their `pnr.tcl`.
- `logs/`: the driver log and the result lines each script appended.
- `results.csv`: `scripts/make_results.py` from `logs/`, beside E1's rows.

## Rerun

On brg-zhang-xcel, from `/scratch/hc676/e22_autosa_tapa` with `scripts/` copied
there: `bash all.sh` generates, compiles (8 then 1 job, one at a time), simulates
and routes all four sizes; `bash collect.sh`-style staging and
`python3 scripts/make_results.py experiments/e1_gemm/autosa_tapa` rebuild this
folder. Never set `ulimit -s unlimited` before a TAPA host: its simulation
library then fails to allocate stacks.
