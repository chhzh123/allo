# Fig. 13(a,b), arXiv version: compilation time of the GEMM arrays

Two measurements of the int8 output-stationary GEMM arrays of Fig. 9(b), at
4x4 to 32x32: (a) how long one SPMW build takes from HLS through routing, and
(b) how long each flow takes from its source kernel to complete RTL.

| Path | What it holds | Used for |
|---|---|---|
| [`build_time/`](build_time/) | E1's memory-connected SPMW builds, one a size: stage times, routed area and timing | Fig. 13(a) |
| [`kernel_to_rtl/`](kernel_to_rtl/) | source-to-RTL runs of SPMW and AutoSA (TAPA), 2026-11-03; the AutoSA and Allo frontend timings | Fig. 13(b): SPMW (parallel, serial), AutoSA (TAPA), the frontend term of AutoSA (Vitis HLS) and Allo |
| `e1_compile_time.csv` | every tool time E1's builds recorded, one run each | Fig. 13(b): the csynth term of AutoSA (Vitis HLS) and Allo |
| [`e1_compile_time.md`](e1_compile_time.md) | what that CSV is, and why it is not a comparison by itself | |
| [`sycl/`](sycl/) | the same array in SYCL, compiled by oneAPI DPC++ 2025.0 for Agilex 7 | Fig. 13(b): SYCL/oneAPI |

Everything ran on brg-zhang-xcel (two Xeon Silver 4214, 48 threads, shared)
with Vitis HLS and Vivado 2023.2 for `xcu280-fsvh2892-2L-e` at 3.333 ns, except
SYCL.

## Fig. 13(a): build time, HLS through routing

The design is the AutoSA-shaped SPMW array with its loaders and drain
(`tests/dataflow/spmw/test_spmw_autosa_match.py`): 15 roles at every size, 24
to 1,088 instances. One build a size, out of context, 2026-09-06. The values are
the `S<n>_pnr` rows of `build_time/results.json` (`results.csv` is the same
table). Seconds; the total is also in minutes, as plotted.

| Array | HLS | Synthesis | Placement | Routing | Other Vivado | Total | HLS share |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | 91.8 | 92.4 | 171.9 | 35.7 | 35.2 | 427.1 (7.1 min) | 21.5% |
| 8x8 | 99.0 | 163.4 | 237.6 | 75.8 | 54.6 | 630.4 (10.5 min) | 15.7% |
| 16x16 | 105.4 | 607.1 | 759.4 | 534.2 | 154.7 | 2,160.8 (36.0 min) | 4.9% |
| 32x32 | 110.6 | 2,096.3 | 1,093.2 | 1,650.3 | 407.2 | 5,357.7 (89.3 min) | **2.1%** |

- **HLS** is `hls_wall_s`: the wall time of the 15 role projects'
  `csynth_design` and `export_design`, eight at a time. `hls_sum_job_s` (687 to
  821 s) is the same work summed over jobs.
- **Synthesis, Placement, Routing** are `synth_s`, `place_s`, `route_s`, the
  build script's stopwatch around each Vivado stage.
- **Other Vivado** is `vivado_total_s` less those three: the rest of the Vivado
  run, such as `opt_design` and `phys_opt_design`.
- **Total** is `hls_wall_s + vivado_total_s`. The frontend is not in it: these
  builds had no frontend timer.
- The same builds' `cost.json`, `util.rpt` and `timing.rpt` are in
  [`../../performance/gemm_arrays/spmw_kernel/S<n>/report/`](../../performance/gemm_arrays/spmw_kernel/):
  their HLS wall, Vivado total, LUTs and WNS match these rows.

## Fig. 13(b): source kernel to complete RTL

Seconds. SPMW and AutoSA (TAPA) at eight workers are medians of three complete
runs; SPMW at one worker is one run.

| Array | SPMW, 8 workers | SPMW, 1 worker | AutoSA (TAPA) | AutoSA (Vitis HLS) | Allo | SYCL/oneAPI |
|---|---:|---:|---:|---:|---:|---:|
| 4x4 | 54.4 | 315.9 | 46.8 | 42.67 | 62.36 | 29.3 |
| 8x8 | 54.5 | 316.0 | 65.9 | 184.28 | 82.83 | 76.1 |
| 16x16 | 54.5 | 316.0 | 69.8 | 1,785.34 | 432.31 | 589.3 |
| 32x32 | **56.6** | **319.2** | 104.0 | 35,025.02 | 6,055.32 | 19,222.7 |

At 32x32 SPMW with eight workers is 619x, 107x and 340x faster than AutoSA
(Vitis HLS), Allo and SYCL; with one worker, 110x, 19x and 60x. Against AutoSA
(TAPA) it is 1.84x (104.0 / 56.6), and TAPA's RTL assembly alone is 39.2 s.
At 4x4, AutoSA (both backends) and SYCL are faster.

| Series | File | Value |
|---|---|---|
| SPMW, 8 / 1 workers | `kernel_to_rtl/results.csv`, flow `spmw_kernel_rtl`, `jobs` 8 / 1 | `total_s`: the median of `runs.ndjson`'s `total_s`, not a sum of stage medians |
| AutoSA (TAPA) | the same, flow `tapa`, `jobs` 8 | `total_s` |
| AutoSA (Vitis HLS) | `kernel_to_rtl/frontends.ndjson` flow `autosa_hls_c`, plus `e1_compile_time.csv` framework `autosa`, stage `hls_csynth` | `frontend_s` + `seconds`; interface `narrow` at 4x4, `wide` at 8x8 to 32x32 |
| Allo | `frontends.ndjson` flow `allo`, plus `e1_compile_time.csv` framework `allo`, stage `csynth` | `wall_s` + `seconds` |
| SYCL/oneAPI | `sycl/report/results.csv` and `sycl/report/split/split_raw.txt` | median of `hls_done_s` - `host`; the error bars are the min and max |

AutoSA's interface at each size is the configuration Fig. 9(b) reports for it,
the lowest-latency one built (`../../performance/gemm_arrays/`): narrow at 4x4,
`--host-serialize` (512-bit) above.

### What each flow covers

Included: the frontend, HLS and RTL assembly. Excluded: IP packaging, Vivado
elaboration, and place and route.

- **SPMW** (`kernel_to_rtl/scripts/spmw_kernel_rtl.py`):
  `scripts/spmw_package_kernel.py --sim` with each unit's `export_design`
  removed, since the array is assembled from the synthesized RTL. 15 roles and 3
  feeders, 18 HLS jobs at every size; the memory loaders and drain are in. The
  fabric RTL is written during elaboration, in the frontend. At 32x32: 4.7 s
  frontend, 52.2 s HLS.
- **AutoSA (TAPA)**: AutoSA `--target=autosa_tapa`, then `tapa analyze ...
  synth -j 8 link`. 15 tasks at 4x4, 18 above, each synthesized once. Its `.xo`
  packing is excluded. At 32x32: 5.5 s frontend, 59.1 s HLS, 39.2 s `tapa link`.
- **AutoSA (Vitis HLS) and Allo**: one HLS project for the whole array, one
  process. The frontend (AutoSA `--target=autosa_hls_c`; Allo's schedule and
  `build(target="vitis_hls")`) was timed once a size on 2026-11-03, and its
  output is byte-identical to the file E1 synthesized. The csynth figure is E1's
  `csynth_design` command alone, so `vitis_hls` start-up and project setup are
  outside it.
- **SYCL/oneAPI** (`sycl/source/e1_sycl.cpp`): the array written as AutoSA
  structures it, every cell a kernel of its own (S^2 cells plus two loaders and
  a store) joined by pipes. `icpx -fsycl -fintelfpga -Xshardware
  -fsycl-link=early -Xstarget=Agilex7 -Xsclock=300MHz`: everything up to RTL and
  reports, nothing of Quartus. The plotted time runs from the start of the
  compile to `aoc`'s "First stage compilation completed", less the host
  compile (4.4 to 112.3 s, profiled once a size by `sycl_split.sh`). Three runs
  at 4x4 to 16x16, one at 32x32 (5 h 23 min end to end). Intel's emulator
  passes at 4x4 to 16x16 and times out at 32x32 (1,200 s). There is no Agilex 7
  device here: this is compile time only, with no routed resources or latency.

### Not plotted

- `kernel_to_rtl/`'s `spmw_mesh_rtl` and `spmw_mesh` rows and logs: the array
  without its memory interface. The paper uses the kernel flow, the scope
  AutoSA's design has. Its one-worker TAPA rows (425.1 s at 32x32) and
  `superseded_runs.ndjson` are records too.
- `e1_compile_time.csv`'s cosim, csim, export and P&R rows, its `spmw_*` rows,
  and AutoSA's other interface.
- `sycl/scripts/sycl_table.py`'s whole-compile and `aoc`-only columns. Its E1
  columns are the narrow AutoSA csynth and SPMW mesh HLS-only figures
  (81 to 110 s), not what Fig. 13(b) plots.

## Caveats

- **The machine is shared.** Load averages: 6.3 to 10.5 during the eight-worker
  `kernel_to_rtl` runs, up to 17.0 during the one-worker batch, 7.9 to 8.3 for
  the frontends, 2.3 to 5.5 at the start of the SYCL runs at 4x4 to 16x16 and
  14.3 at 32x32. E1's csynth runs saw about 5 to 24 (`e1_compile_time.md`).
  The 2026-09-06 builds of panel (a) recorded none.
- **The one-worker SPMW rows are single runs, all sixteen started at once**
  (each one core). The eight-worker TAPA rows ran after the SPMW ones, not
  interleaved with them.
- **The AutoSA (Vitis HLS) and Allo totals join two measurements**: a csynth
  run from September, one run each, and a frontend run from 2026-11-03. Their
  timers also differ from SPMW's and TAPA's, which time the whole command.
- **SYCL differs from the rest in device family (Agilex 7, not the U280)** and
  in design, which is hand-written in AutoSA's structure rather than generated.
- **Panel (a)'s HLS is not panel (b)'s.** (a) runs `csynth_design` and
  `export_design` for 15 roles (110.6 s at 32x32); (b) runs `csynth_design`
  only, for 18 units (52.2 s).
- **Panel (a) is one build a size**, with no repetition.

## Reproducing

The drivers name their working directories under `/scratch/hc676` on
brg-zhang-xcel and the checkout they import, so they are a record of what ran
as much as a recipe.

    # Fig. 13(b), SPMW and AutoSA (TAPA): one flow, one size, one worker count
    kernel_to_rtl/scripts/run_one.sh spmw_kernel_rtl 32 8 0
    kernel_to_rtl/scripts/cmp2.sh        # the SPMW eight-worker rows
    kernel_to_rtl/scripts/cmp3.sh        # the TAPA eight-worker rows, then every one-worker row
    python3 kernel_to_rtl/scripts/make_cmp.py kernel_to_rtl/runs.ndjson kernel_to_rtl/results.csv
    kernel_to_rtl/scripts/fe_run.sh      # the AutoSA and Allo frontends (its fe.ndjson is frontends.ndjson)

    # Fig. 13(b), SYCL
    sycl/scripts/intel_unpack.sh /scratch/hc676/intel   # the compiler, no root
    sycl/scripts/sycl_emu.sh 4 8 16 32                  # check it in the emulator
    sycl/scripts/sycl_compile.sh 3 4 8 16 32            # time it (after_rep1.sh: 32x32 once)
    sycl/scripts/sycl_split.sh 4 8 16 32                # the host compile and the steps before aoc
    python3 sycl/scripts/sycl_table.py                  # its tables, from report/ and ../e1_compile_time.csv

`make_cmp.py` regenerates `kernel_to_rtl/results.csv` byte for byte from
`runs.ndjson`. Panel (a)'s builds were made by `scripts/spmw_build_array.py
--design autosa ... --pnr` at eight jobs; the driver that ran the four sizes was
not kept. The tree before this layout, with E5's other compile-time studies,
is commit `c20b2ee8` (`experiments/e5_compile/`).
