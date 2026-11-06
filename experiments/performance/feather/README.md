# FEATHER: Fig. 12 of the arXiv version

FEATHER (Tong et al., ISCA 2024) is NEST, an n x n array of PEs with a private
weight file each, feeding BIRRD, a butterfly reduction and reordering network.
Three designs run on n x n arrays, n = 4, 8, 16:

- FEATHER's released RTL (`github.com/maeri-project/FEATHER`, `FEATHER_RTL/RTL`
  at 225951b) with its weight-buffer controller corrected. It loads one PE
  weight a cycle.
- The same RTL with a row-wise weight loader on top. It loads a row of PEs a
  cycle.
- The SPMW port, [`spmw/S<n>/source/spmw_feather.py`](spmw/S4/source/spmw_feather.py).

Workloads: a 128^3 int8 GEMM, and a 3x3 int8 convolution (16x16 spatial, 64
input and 64 output channels, stride 1, pad 1, NHWC), both tiled as the Allo
drivers tile them ([`examples/feather/`](../../../examples/feather/)) with the
host summing the partial tiles. Latency is xsim of the RTL; resources are
Vivado 2023.2 out-of-context place and route on `xcu280-fsvh2892-2L-e` at
3.333 ns.

## Names

The CSV's `rtl_fixed` is the paper's **RTL**, not its **RTL (fixed)**. Both
RTL variants carry the controller correction; in the figure, "fixed" means the
row loader.

| Fig. 12 | `results.csv` rows | Reports |
|---|---|---|
| RTL | `rtl_fixed_*`, `pnr_rtl_fixed_N*` | `feather_rtl/S<n>/report/sim_corrected_*`, `pnr_corrected_*` |
| RTL (fixed) | `rtl_rowload_*`, `pnr_rtl_rowload_N*` | `feather_rtl/S<n>/report/sim_rowload_*`, `pnr_rowload_*` |
| SPMW, panels (a, b) | `spmw_*_stream_general` | `spmw/S<n>/report/sim_*_stream_general_result.json` |
| SPMW, panel (c) | `pnr_spmw_feather_fabmul_N*` | `spmw/S<n>/report/pnr_fabmul_*` |

## Panels (a, b): a new weight feed every tile

Completion cycles of the whole workload, with every tile loading its own
weights (the RTL's MODE 0; SPMW's `feather_stream`, which streams every
operand).

| Workload | n | Tiles | RTL | RTL (fixed) | SPMW | RTL (fixed) / SPMW |
|---|---:|---:|---:|---:|---:|---:|
| GEMM | 4 | 32,768 | 2,097,169 | 524,305 | 262,182 | 2.00x |
| | 8 | 4,096 | 2,097,179 | 262,171 | 65,644 | 3.99x |
| | 16 | 512 | 2,097,197 | 131,117 | 16,720 | 7.84x |
| Conv | 4 | 196,608 | 12,582,929 | 3,145,745 | 1,572,902 | 2.00x |
| | 8 | 32,768 | 16,777,243 | 2,097,179 | 524,396 | 4.00x |
| | 16 | 4,096 | 16,777,261 | 1,048,621 | 131,408 | 7.98x |
| Cycles a tile | | | n^3 | n^2 | 2n | n/2 |

- **RTL** loads one PE weight a cycle: n^2 PEs with n slots each, n^3 cycles,
  though its weight port is n bytes wide.
- **RTL (fixed)** drives all n bytes, a row of PEs a cycle, but still loads
  the n rows one after another, n slots each: n^2 cycles.
- **SPMW** gives each PE pair's weight file its own stream, so every pair fills
  its 2n slots at once: 2n cycles.

Convolution pads the nine taps to 12, 16 and 16. Operands are `--pattern full`,
seed 0: every stored byte in [0, 256), read as int8 by SPMW and as uint8 with
zero point 0 by the RTL. The RTL is timed from the first cycle of its first
weight feed to the last tile's completion
([`e4_feather_gen.py`](scripts/loader/e4_feather_gen.py) `check`), SPMW from
reset release to the last tile's last output
([`e4_spmw_run.py`](scripts/reporting/e4_spmw_run.py)). The host's reduction
is not timed. Every row is checked: the RTL tile by tile, bit-exact against a
model of its own arithmetic, and SPMW token by token in cosimulation. The
reduced result is then checked against numpy (0 bad elements at every point).

## Resident weights: the compute rate

One weight feed, then every tile's activations through tile 0's weights (the
RTL's MODE 1; SPMW's `feather_stream_x`). This measures datapath throughput,
not the workload's result, so there is no host check. All three designs run at
n cycles a tile.

| Workload | n | RTL | RTL (fixed) | SPMW | Cycles a tile |
|---|---:|---:|---:|---:|---:|
| GEMM | 4 | 131,149 | 131,101 | 131,118 | 4 |
| | 8 | 33,299 | 32,851 | 32,856 | 8 |
| | 16 | 12,317 | 8,477 | 8,340 | 16 |
| Conv | 4 | 786,509 | 786,461 | 786,478 | 4 |
| | 8 | 262,675 | 262,227 | 262,232 | 8 |
| | 16 | 69,661 | 65,821 | 65,684 | 16 |

The startups do not compare like for like. `feather_stream_x` receives each
PE's whole weight file in one beat on its own port, and the RTL's figure
contains its weight feed (`weight_feed_cycles`: 64/512/4,096, and 16/64/256
with the row loader). With the feed netted out (`fill_cycles`), the RTL fills
in 17/27/45 cycles and SPMW in 50/96/164.

## Panel (c): resources

Resident-weight compute ports, multiplies in LUTs. Both sides use 0 DSP, BRAM
and URAM, meet timing, and route every net. Fmax = 1000 / (3.333 - WNS).

| n | Design | LUT | FF | WNS (ns) | Fmax (MHz) |
|---:|---|---:|---:|---:|---:|
| 4 | SPMW | 3,505 | 4,291 | +0.869 | 406 |
| | RTL | 2,309 | 3,378 | +1.174 | 463 |
| 8 | SPMW | 14,863 | 18,453 | +0.707 | 381 |
| | RTL | 9,694 | 15,332 | +0.355 | 336 |
| 16 | SPMW | 54,439 | 72,585 | +0.291 | 329 |
| | RTL | 57,499 | 91,251 | +0.120 | 311 |

Against the RTL, SPMW uses +51.8%, +53.3% and -5.3% LUTs and +27.0%, +20.4% and
-20.5% FFs at n = 4, 8 and 16.

- **SPMW** is `feather(n, n)`, the single-tile engine. Activations, weight
  files and switch commands are parallel `MemIn` ports, and the design is
  routed inside `spmw_harness` (LFSR drivers and sinks around `spmw_top`). Its
  integer multiplies are bound to fabric (`#pragma HLS bind_op ... op=mul
  impl=fabric`, `SPMW_BIND_MUL=1`, Vivado `-max_dsp 0`). That binding came
  later than the latency source: these builds ran from
  [`tests/dataflow/spmw/test_spmw_feather.py`](../../../tests/dataflow/spmw/test_spmw_feather.py)
  at ed056f9a (unchanged at HEAD). That file differs from `source/spmw_feather.py`
  only by the `spmw_bind_mul_fabric` flag.
- **RTL** is `feather_top` out of context: NEST, BIRRD, the controller and the
  SRAMs at shipped defaults (depth-4 SRAM register arrays), with the corrected
  controller and the PE-wise loader.

This is not an equal-bandwidth, full-system comparison. The RTL's count
includes its controller and the narrow-port logic that delivers weights. SPMW
exposes parallel ports and does not count whatever would feed them.

The row loader was routed too (it is not in the figure), at 2,291 / 9,690 /
58,465 LUTs, 3,378 / 15,335 / 91,249 FFs and WNS +1.074 / +0.386 / +0.115 ns.

## Checks on the RTL variants

- **Validation matrix**: single tiles over the GEMM layout program, random
  programs and, at 4x4, the drivers' four conv programs, crossed with three
  operand patterns, three zero-point pairs and three seeds. Both variants pass
  all runs: 162 at 4x4, 54 at 8x8 and 54 at 16x16
  (`sim_*_single_tiles_validate_summary.json`).
- **Weight files**: in every row-loader resident run the bench dumps all
  n^2 PEs' files and compares them with the one-PE-a-cycle image. 0 of
  64 / 512 / 4,096 slots are wrong.
- **Negative controls**, at 4x4: the old select fed the row-packed image, and
  the new select fed the old image (`rtl_rowload_negctl_*`). Both fail, with 0 of
  32,768 tiles correct and 60 of 64 PE slots wrong. They show that the check
  can fire.

## Files

| Path | What it is |
|---|---|
| `results.csv` | 53 rows: the workload rows above, the validation and negative-control rows, the nine P&R rows |
| `spmw/S<n>/source/spmw_feather.py` | the SPMW port as the latency runs read it: `test_spmw_feather.py` at 23eedb2. One file; the size is an argument |
| `spmw/S<n>/generated/` | the HLS C++ and RTL that `spmw_build_array.py --design feather` emits from that source: the single-tile engine, 9 roles at 4x4 and 11 at 8x8 and 16x16. It predates the fabric binding, so its multiplies carry no `bind_op` |
| `spmw/S<n>/report/` | the cosimulation results of the four workload runs (GEMM and conv, streamed and resident), and panel (c)'s routed utilisation and timing |
| `feather_rtl/S<n>/source/` | the repository's first FEATHER bench and its driver ([`tests/dataflow/spmw/tb_feather_rtl.sv`](../../../tests/dataflow/spmw/tb_feather_rtl.sv), [`scripts/spmw_feather_rtl.py`](../../../scripts/spmw_feather_rtl.py)). It drives the shipped controller with odd-index weights zeroed. No row here comes from it; the rows come from `scripts/*/tb_feather_rtl_general.sv` |
| `feather_rtl/S<n>/report/` | per variant: each run's `check.json` and `meta.json`, the validation summaries, and the routed utilisation, hierarchy, timing, route status and `impl.tcl` |
| [`scripts/feather_controller.diff`](scripts/feather_controller.diff), `scripts/harness/apply_patch.py` | the controller correction: the PEs' ping/pong select toggles once per complete feed rather than once per sweep |
| [`scripts/loader/row_loader.diff`](scripts/loader/row_loader.diff), `scripts/loader/apply_row_loader.py` | the row loader, applied on top: the PE matches the row field of its id, and the controller wraps `pe_sel` at n |
| `scripts/harness/` | the first pass. It holds the generator, RTL model and checker (`e4_feather_gen.py`), the general bench, `e4_rtl_run.py`, the validation matrix (`e4_matrix.py`), RTL P&R (`pnr_ooc.sh`, `pnr_chain.sh`) and the SPMW workload driver. `README_static.md` is that pass's notes and describes its own directory layout |
| `scripts/reporting/` | the second pass. `patch_gen.py` and `patch_spmw.py` add conv programs at n = 8, 16 and the host check, and `chains.sh` ran the `rtl_fixed_*` and `spmw_*` workload rows |
| `scripts/loader/` | the row loader's runs. It holds the merged generator (`--loader row`), `run_sims.sh`, `run_validate.sh`, `run_conv.sh`, `run_pnr.sh` and `run_fabmul_pnr.sh` (panel (c)'s SPMW builds), plus `compare_images.py` and the `append_*.py` scripts that wrote the `rtl_rowload_*` and `pnr_spmw_feather_fabmul_*` rows |

## What is not here

- **The two patched RTL trees.** They were built on brg-zhang-xcel in
  `/scratch/hc676/e4_feather_fixed/RTL` (corrected) and
  `/scratch/hc676/feather_loader/RTL_new` (row loader). Only the patches are
  here.
- **The generated code of the latency builds** (`feather_stream`,
  `feather_stream_x`) and of panel (c)'s fabric-bound builds. `generated/` is
  the closest: the same source's single-tile engine.
- **The RTL runs' bus traces and tile dumps.**
- **What the paper does not use**: 32x32, the uncorrected controller, and the
  DSP-inferred and streamed SPMW place and route. These are in commit
  `c20b2ee8` under `experiments/e4_feather/`.

## Reproducing

On brg-zhang-xcel with Vitis HLS and Vivado 2023.2 sourced, from the
repository root, with `F=experiments/performance/feather`. The drivers name
their `/scratch/hc676` working directories and the checkout they import
(`/scratch/hc676/allo`).

    # the two RTL variants, from FEATHER_RTL/RTL at 225951b
    python3 $F/scripts/harness/apply_patch.py <rtl>/feather_controller.v
    python3 $F/scripts/loader/apply_row_loader.py <rtl>        # RTL (fixed) only

    # RTL: --mode 0 is a feed a tile (panels a, b), --mode 1 resident;
    # --loader pe for RTL, row for RTL (fixed); conv: --workload conv --conv 64,16,16,64
    python3 $F/scripts/loader/e4_rtl_run.py --rtl <rtl> --N 16 --out <dir> \
        --workload gemm --gemm 128,128,128 --pattern full --seed 0 \
        --mode 0 --zpa 0 --zpw 0 --loader row
    python3 $F/scripts/loader/e4_matrix.py --rtl <rtl> --N 16 --tag row \
        --set validate --loader row --out <root>

    # SPMW: feather_stream; --resident for feather_stream_x
    python3 $F/scripts/reporting/e4_spmw_run.py --N 16 --workload gemm \
        --gemm 128,128,128 --pattern full --seed 0 --out <dir>

    # panel (c)
    bash $F/scripts/loader/pnr_ooc.sh <rtl> 16 <dir>
    SPMW_BIND_MUL=1 python3 scripts/spmw_build_array.py --design feather \
        --size 16 --frequency 300 --jobs 6 --pnr --out <dir>
