# Mini-TPU: SPMW, Gemmini and VTA as whole engines

Backs **Fig. 10** of the arXiv version (Section 6.1.2, *Mini-TPU*): cycle
latency against effective throughput on two FFN slices (a, b), and the 16x16
engines' routed resources (c). Each engine is whole and runs its own
instructions. Every file the figure reads is unchanged since `98c8486`
(3 October), the commit the paper pins.

## The engines

All three are n x n int8 arrays, n = 4, 8 and 16, with int32 accumulation and a
shift-and-clip epilogue. Each is measured from a memory port in to a memory
port out: instruction fetch, operand loads, compute and result stores.

- **SPMW** (`ptpu_mem/`) is the systolic array of
  [`test_spmw_ptpu_mem.py`](../../../tests/dataflow/spmw/test_spmw_ptpu_mem.py)
  composed with instruction dispatch, vector lanes and memory movers. It
  fetches its program and operands through one 64-bit read port and writes
  its results through one 64-bit write port. One GEMM instruction, four 64-bit
  words, gives tensor addresses, tile dimensions, reduction length and the
  epilogue. The GEMM runs a K block at a time. Each weight block is used once,
  as it arrives, and is never stored outside the array. One activation chunk
  of up to 64 rows is kept in distributed RAM. The partial sums stay in the
  lanes' accumulators, 8,192 int32 sums in block RAM. The units are the cells
  (`pe`), requester (`req`), dealer (`deal`), sequencer (`head`), edge taps
  (`etap`), micro-op queue and taps (`uq`, `tap`), lanes (`lane`), result taps
  (`ctap`), packer (`pack`) and write requester (`wreq`).
- **Gemmini** (`gemmini_full/`) is the whole `Gemmini` RoCC accelerator,
  built from its own Chisel at the commits in `pins.txt`. It is elaborated in
  rocket-chip's example system, which has a 64-bit system bus
  (`source/harness/src/main/scala/gen/ElaborateGemmini.scala`), and includes
  the reservation station, loop unrollers, load, store and execute
  controllers, scratchpad and accumulator, DMA, TLB and mesh. The
  configuration is `matmul`: `leanConfig` (weight-stationary only) with the
  accumulator's float32 scale replaced by a right shift, no scale on load,
  and convolution loop unrolling, pooling, training and depthwise
  convolutions and first-layer optimisations switched off.
  `scripts/ramfix.py` rewrites the accumulator banks wider than 128 bits as
  64-bit columns with byte masks, so that Vivado infers RAM rather than
  registers. Nothing else in the RTL is touched.
- **VTA** (`vta_core/`, `core/`) is VTA's Chisel `Core`: fetch, load,
  compute, store and every scratchpad. It has batch 1,
  `blockIn = blockOut = n`, and the shipped storage depths: 2,048 input rows,
  1,024 weight blocks and 2,048 accumulator rows. The elaboration source is
  `vta_core/source/ElaborateVTA.scala` (F1 shell, 64-bit memory interface).
  The VTA checkout's commit is not recorded.

Each engine runs its native program:

| Engine | Program | LLaMA, n = 4 / 8 / 16 | DeepSeek, n = 4 / 8 / 16 |
|---|---|---:|---:|
| SPMW | GEMM instructions fetched from memory | 1 / 1 / 1 | 1 / 1 / 1 |
| Gemmini | its library's: `tiled_matmul_auto` tiling, one `loop_ws` a tile, expanded by its loop unroller | 41 / 41 / 41 commands | 125 / 125 / 113 |
| VTA | a 128-bit stream as its runtime launches one: loads, GEMMs, ALU passes, store, finish, synchronised by dependency tokens and paged through the scratchpads' two halves | 103 / 55 / 31 | 343 / 175 / 91 |

## Workloads

64-token gate/up projection slices with 64 columns per projection:

| Slice | M | K | N | MKN |
|---|---:|---:|---:|---:|
| LLaMA 3.2 1B | 64 | 2,048 | 128 | 16,777,216 |
| DeepSeek-V4-Pro expert | 64 | 7,168 | 128 | 58,720,256 |

[`test_spmw_llama_ffn.py`](../../../tests/dataflow/spmw/test_spmw_llama_ffn.py)
writes the stimulus: random int8 inputs, a shift and the golden output. All
three benches read the same file. Each run compares its whole output with the
golden, and all 18 runs in the figure pass (`errs=0`, `timeout=0`). These are
random-int8 projection slices, not inference at model precision, and they say
nothing about accuracy.

## How it is measured

- **Cycles, C**, come from an xsim run (Vivado 2023.2) of the whole engine's
  RTL under each engine's bench:
  - `scripts/spmw_mem_bench.py`: from the launch token to `done`, which
    follows the acknowledgement of the last write.
  - `scripts/gemmini_rocc_bench.py`: from the first RoCC command until the
    accelerator is idle.
  - `scripts/vta_core_bench.py`: from launch to finish.

  Command issue is ideal: Gemmini is offered a command in the cycle the
  previous one is taken. Memory is ideal and 64 bits wide: a request is
  taken whenever there is room, and one beat a cycle is returned from the
  next cycle on. VTA's five read channels (instructions, micro-ops, inputs,
  weights, accumulator) are served independently, above its arbiter.
- **Floor**: MKN / n^2, one array pass a cycle.
- **Clock, f**: every design is placed and routed out of context on
  `xcu280-fsvh2892-2L-e` at more than one target. Gemmini and VTA use
  retiming in synthesis and in both `phys_opt_design` passes. f is the
  reciprocal of the shortest achieved period (target - WNS) over the
  targets.
- **Effective throughput**: 2 MKN min(300 MHz, f) / C, counting two
  operations per MAC.
- **Resources** come from the selected route. SPMW's are those of the `dut`
  instance (`p30_area_by_unit.txt`), which excludes the bench harness that
  `--pnr` routes around it; at 16x16 the flat `p30_util.rpt` has 2 more LUTs
  and 96 more FFs. Gemmini's and VTA's are the whole routed module. BRAM18
  counts a RAMB36 as two.
- **One SPMW build runs every program.** The routed build is
  `ptpumem-micro`. The five-GEMM program, `ptpumem-mixed`, built separately,
  generates 78 files that are byte-identical to it at every size
  (`ptpu_mem/S<n>/report/same_hardware.txt`).

## Routes

Achieved period in ns (target - WNS). The selected route is in bold.

| Engine | Target | 4x4 | 8x8 | 16x16 | Reports |
|---|---:|---:|---:|---:|---|
| SPMW | 3.333 | 2.872 | 2.975 | 3.246 | `ptpu_mem/S<n>/report/{timing.rpt,area_by_unit.txt}` |
| | 3.0 | **2.821** | **2.727** | **2.996** | `ptpu_mem/S<n>/report/p30_*` |
| Gemmini | 3.333 | **6.595** | **6.795** | 6.771 | `gemmini_full/route/matmul_<n>/` |
| | 6.0 | 6.797 | 6.871 | **6.553** | `gemmini_full/route/matmul_<n>_p6/` |
| VTA | 3.333 | 4.887 | 4.760 | 4.848 | `core/w<n>/` |
| | 4.3 | 4.839 | 4.858 | **4.456** | `vta_core/route/w<n>_p43/` |
| | 4.6 | **4.684** | **4.595** | 4.787 | `vta_core/route/w<n>_p46/` |

`ptpu_mem/routes.txt` lists SPMW's routes. `gemmini_full/routes.json` lists
Gemmini's and VTA's, each with its best route under `best`. Every route has
zero unrouted nets.

Worst paths of the selected routes:

- **SPMW**: from a cell into a partial-sum link, the one into a lane at 4x4
  and 8x8 and the one into the next cell at 16x16.
- **Gemmini**: at 4x4 and 8x8, from the store controller's counters into
  the scratchpad's write queue (25 logic levels); at 16x16, in the DMA writer
  (31 levels).
- **VTA**: at 4x4 and 8x8, from an instruction queue into a DMA command
  generator; at 16x16, the accumulator URAM's read-modify-write.

## Results

**Hardware**, on the selected routes. The 16x16 rows are Fig. 10(c).

| n | Engine | f (MHz) | f used | LUT | FF | DSP | BRAM18 | URAM |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | SPMW | 354.5 | 300 | 4,860 | 5,582 | 0 | 22 | 0 |
| | Gemmini | 151.6 | 151.6 | 31,545 | 21,320 | 27 | 128 | 4 |
| | VTA | 213.5 | 213.5 | 10,099 | 3,441 | 0 | 28 | 2 |
| 8 | SPMW | 366.7 | 300 | 10,872 | 11,799 | 0 | 24 | 0 |
| | Gemmini | 147.2 | 147.2 | 38,941 | 25,648 | 25 | 160 | 0 |
| | VTA | 217.6 | 217.6 | 15,403 | 4,747 | 0 | 44 | 6 |
| 16 | SPMW | 333.8 | 300 | 35,544 | 33,603 | 0 | 28 | 0 |
| | Gemmini | 152.6 | 152.6 | 65,529 | 39,989 | 29 | 32 | 8 |
| | VTA | 224.4 | 224.4 | 33,923 | 7,387 | 0 | 140 | 12 |

**Cycles and effective throughput**, Fig. 10(a, b):

| Slice | n | Floor | SPMW | Gemmini | VTA | SPMW GOP/s | Gemmini | VTA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaMA | 4 | 1,048,576 | 1,048,625 | 1,073,610 | 1,058,331 | 9.60 | 4.74 | 6.77 |
| | 8 | 262,144 | 262,211 | 274,359 | 269,578 | 38.39 | 18.00 | 27.09 |
| | 16 | 65,536 | 66,246 | 77,605 | 74,911 | 151.95 | 65.98 | 100.52 |
| DeepSeek | 4 | 3,670,016 | 3,670,065 | 3,723,974 | 3,680,381 | 9.60 | 4.78 | 6.81 |
| | 8 | 917,504 | 917,571 | 945,033 | 925,364 | 38.40 | 18.29 | 27.62 |
| | 16 | 229,376 | 230,086 | 271,099 | 239,011 | 153.13 | 66.11 | 110.27 |

The cycle counts come from `ptpu_mem/results.txt` (runs `<slice>_<n>`),
`gemmini_full/sim/results.txt` (`<slice>_matmul_<n>`) and
`vta_core/results.txt` (`<slice>_w<n>`), where `<slice>` is `llama` or
`dsv4`. SPMW's and VTA's result lines match their per-run `xsim.log`. For
Gemmini only the consolidated result lines and the testbenches are kept, not
the simulator logs.

The paper's claims, as these tables give them:

- **Throughput.** SPMW has 2.01-2.32 times Gemmini's: 2.03, 2.13 and 2.30
  on LLaMA, and 2.01, 2.10 and 2.32 on DeepSeek. It has 1.39-1.51 times
  VTA's: 1.42, 1.42 and 1.51, and 1.41, 1.39 and 1.39.
- **Cycles at 16x16 on LLaMA.** SPMW takes 14.6% fewer cycles than Gemmini
  and 11.6% fewer than VTA.
- **Distance from the floor.** SPMW is 49, 67 and 710 cycles over it on
  either slice, at most 1.08% (LLaMA, 16x16). Gemmini is 1.5-18.4% over, and
  VTA 0.3-14.3%.
- **LUTs at 16x16.** SPMW has 46% fewer than Gemmini (35,544 against
  65,529), and 1.05 times VTA's.

## Also here

- **Scope splits.** Gemmini's by block (`gemmini_full/route/*/split.txt`)
  with its execute scope's clock (`scope.txt`); VTA's into control and
  datapath + scratchpads (`core/split.txt`), with that scope's timing with and
  without the input and output buffers (`core/w<n>/scope*_timing.rpt`,
  `vta_core/route/*/scope.txt`).

## Layout

| Path | What |
|---|---|
| `ptpu_mem/S<n>/source/` | the design, `test_spmw_ptpu_mem.py` (identical to the one in `tests/`) |
| `ptpu_mem/S<n>/generated/` | per-role HLS C++, wrappers and Vitis scripts, and the fabric (`spmw_top.sv`, `spmw_fifo.sv`, `spmw_const.sv`, `spmw_harness.sv`) |
| `ptpu_mem/S<n>/report/` | `build.log`, csynth reports, cosim cycles, the 3.333 ns route (no prefix) and the 3.0 ns route (`p30_`); `mixed_*` is the five-GEMM build; `same_hardware.txt` |
| `ptpu_mem/sim/<run>/` | each bench run's testbench `tbm.sv`, instruction words `ins.hex` and `xsim.log` |
| `ptpu_mem/results.txt`, `routes.txt` | every run's result lines; every route |
| `ptpu_mem/scripts/` | `spmw_build.sh` (build, cosim, route), `reroute.sh` (route again at another target), `hier_mem.sh` (`dut` area by unit), `routes.sh`, `same_hw.sh`, `bench.sh`, `export_mem.sh` |
| `gemmini_full/source/` | the sbt build and `ElaborateGemmini.scala` (it defines `lean`, `shift` and `matmul`; the routes and runs here are `matmul`) |
| `gemmini_full/pins.txt` | the commit of every repository Gemmini was built from, and the Chisel, Scala and firtool versions |
| `gemmini_full/route/matmul_<n>[_p6]/` | each route's `pnr.tcl`, utilisation, timing, `split.txt`, scope timing, `gemmini_params.h`, and `ramfix.txt` where it applied |
| `gemmini_full/sim/<run>/` | each run's testbench and RoCC command lists; `results.txt` holds the result lines |
| `gemmini_full/routes.json` | Gemmini's and VTA's routes, made by `collect_routes.py` |
| `gemmini_full/scripts/` | `elab.sh`, `closure.py`, `ramfix.py`, `pnr_gemmini.sh`, `split.py`, `collect_routes.py`, `run_rocc_xsim.sh` and its batch drivers, scope timing, `export.sh`; `dtc` is a stand-in for the device-tree compiler |
| `vta_core/<slice>_w<n>/` | each run's testbench, instruction and micro-op images and `xsim.log`; `results.txt` |
| `vta_core/route/w<n>_p43`, `_p46` | `Core` routed at 4.3 and 4.6 ns |
| `vta_core/run_core.sh`, `run_all.sh`, `scope_relaxed.sh` | one run; every run; scope timing on those routes |
| `core/w<n>/` | `Core` routed at 3.333 ns, with hierarchical utilisation and scope timings; `split.txt` |
| `scripts/` | the three benches; VTA's elaboration (`elab_vta.sh`, `elab_core_widths.sh`), its route (`pnr_vta_rtl.sh`), split (`vta_core_split.py`) and scope timing (`vta_core_scope*_timing.tcl`) |

## Reproducing

Everything ran on brg-zhang-xcel with Vitis HLS and Vivado 2023.2. The
drivers name their working directories under `/scratch/hc676`, so they are a
record of what ran as much as a recipe. `run_rocc_xsim.sh` and `run_core.sh`
expect their bench copied into that directory, and the stimulus files are not
committed. Paths are relative to this folder, except for the stimulus
generator, which runs from the repository root.

    # stimulus, from the repository root
    python3 tests/dataflow/spmw/test_spmw_llama_ffn.py --out llama_slice.txt
    python3 tests/dataflow/spmw/test_spmw_llama_ffn.py --k 7168 --out dsv4_slice.txt

    # SPMW: build, cosim and route at 3.333 ns; route again at 3.0 ns; split; run
    ptpu_mem/scripts/spmw_build.sh ptpumem-micro <n>
    PNR= ptpu_mem/scripts/spmw_build.sh ptpumem-mixed <n>; ptpu_mem/scripts/same_hw.sh <n>
    ptpu_mem/scripts/reroute.sh <n> 3.0
    ptpu_mem/scripts/hier_mem.sh <route dir>
    ptpu_mem/scripts/bench.sh llama|dsv4 <n>
    ptpu_mem/scripts/routes.sh

    # Gemmini: RTL for simulation and for synthesis; route; run
    SIM=1 gemmini_full/scripts/elab.sh <n> matmul
    gemmini_full/scripts/elab.sh <n> matmul
    gemmini_full/scripts/pnr_gemmini.sh matmul <n> [6.0 p6]
    gemmini_full/scripts/run_rocc_xsim.sh matmul <n> <stim> llama|dsv4
    python3 gemmini_full/scripts/collect_routes.py > gemmini_full/routes.json

    # VTA: elaborate Core (n = 16, then 4 and 8); route; run
    scripts/elab_vta.sh Core
    scripts/elab_core_widths.sh
    scripts/pnr_vta_rtl.sh Core <rtl dir> Core_w<n>[_p43|_p46] [4.3|4.6]
    vta_core/run_core.sh <n> <stim> llama|dsv4

`scripts/spmw_build_array.py` (repository root) builds the designs
`ptpumem-micro`, `-llama`, `-dsv4` and `-mixed`. `export_mem.sh` and
`gemmini_full/scripts/export.sh` copied the runs into this layout.

## Not here

The rest of E9 (fixed-function projection and SwiGLU runs, the fixed
workload, the stream-fed `ptpu` engine, `results.csv`, their scripts) and the
runs Fig. 10 does not plot (the microbenchmark, the five-GEMM program and the
stalling-memory runs on every engine, Gemmini's `lean` and `shift`
configurations and its 128-bit-bus runs) are at `c20b2ee8:experiments/e9_vta/`. The stream-fed 16x16 build
used by Fig. 13(c) and the simulation-time write-up are in
[`../../compilation/simulation_time/`](../../compilation/simulation_time/README.md).
