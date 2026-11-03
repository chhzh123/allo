# Compile time against AutoSA through TAPA, one protocol, 2026-11-03

E5's AutoSA column is AutoSA's Vitis HLS backend, one HLS project for the whole
array. AutoSA's TAPA backend (`../../e1_gemm/autosa_tapa/`) synthesizes each
distinct task once, the same reuse SPMW's roles have, so it is the comparison
that tests SPMW's compile-time claim. This measures SPMW and AutoSA-on-TAPA on the
same machine, at the same time, with the same worker cap and the same stage
coverage: **source to complete RTL, frontend and HLS included, nothing after
RTL** (no packaging, no Vivado elaboration, no P&R).

**Result:** SPMW's mesh reaches RTL in 35 s at every size from 4x4 to 32x32 with
8 workers; AutoSA through TAPA takes 47 s at 4x4 and 104 s at 32x32, so SPMW is
1.3x to 2.9x faster. With one worker SPMW's mesh is 1.9x to 2.6x faster. SPMW's
kernel (the same array with its memory loaders and drain, the scope AutoSA's
design has) is slower at 4x4 with 8 workers (54.4 s against 46.8 s), then 1.2x
faster at 8x8 and 1.8x faster at 32x32; with one worker it is 1.02x to 1.33x
faster.

## What each flow covers

| Flow | Command | Frontend | HLS | After HLS |
|---|---|---|---|---|
| AutoSA -> TAPA | AutoSA `--target=autosa_tapa` + `gen.sh`'s kernel fixes, then `tapa analyze ... synth -j N link` | AutoSA's codegen, the fixes, `tapa analyze` | one `csynth_design` per distinct task (15 at 4x4, 18 above) | `tapa link`: the top-level RTL with every task instance |
| SPMW mesh, RTL only | `scripts/spmw_mesh_rtl.py`: E5's `spmw_ablate_compile.py`, `export_design` removed | Python start-up, elaboration, per-role codegen, the fabric RTL | one `csynth_design` per role (9) | none: the fabric was written in the frontend |
| SPMW kernel, RTL only | `scripts/spmw_kernel_rtl.py`: `spmw_package_kernel.py --sim`, `export_design` removed | the same, for 15 roles and 3 feeders | one `csynth_design` per unit (18) | the kernel wrapper (milliseconds) |
| SPMW mesh as built | `spmw_ablate_compile.py --mode shared-parallel`, E5's own measurement | as above | `csynth_design` + `export_design` per role | none |

**Why `export_design` is removed.** It packages each role as an IP (it starts a
Vivado), about 25 s of a 38 s role job. SPMW assembles its array and its kernel
from `prj/sol/syn/verilog/*.v`, which `csynth_design` writes; the exported IP is
read only for the `.xci` of Xilinx IP cores, and E1's exported mesh roles have
none at 4x4 or 32x32. TAPA likewise takes its tasks' RTL from synthesis and
packs once at the end (its `.xo` packing, 31 to 40 s, is excluded here). The mesh
as built is kept as a row so the flow E5 measured can be read beside it.

## Results

8 workers (`-j 8`, `--jobs 8`), median of three runs, one compile at a time, the
flows in a seeded random order per size. Seconds.

| Array | Flow | Frontend | HLS | After HLS | Total | Range |
|---|---|---:|---:|---:|---:|---|
| 4x4 | AutoSA -> TAPA | 3.9 | 40.4 | 2.2 | **46.8** | 46.7-46.8 |
| | SPMW mesh, RTL only | 1.9 | 32.8 | - | **34.8** | 34.5-34.8 |
| | SPMW kernel, RTL only | 3.0 | 51.4 | - | 54.4 | 53.4-54.6 |
| | SPMW mesh as built | 2.0 | 82.4 | - | 84.4 | one run |
| 8x8 | AutoSA -> TAPA | 4.2 | 58.8 | 2.9 | **65.9** | 65.8-66.4 |
| | SPMW mesh, RTL only | 2.2 | 33.0 | - | **35.2** | 34.8-35.7 |
| | SPMW kernel, RTL only | 3.2 | 51.2 | - | 54.5 | 53.9-55.4 |
| | SPMW mesh as built | 2.1 | 82.3 | - | 84.4 | one run |
| 16x16 | AutoSA -> TAPA | 4.4 | 58.8 | 6.2 | **69.8** | 69.7-70.2 |
| | SPMW mesh, RTL only | 2.2 | 32.8 | - | **35.0** | 35.0-35.3 |
| | SPMW kernel, RTL only | 3.1 | 51.4 | - | 54.5 | 54.4-55.5 |
| | SPMW mesh as built | 2.3 | 83.3 | - | 85.6 | one run |
| 32x32 | AutoSA -> TAPA | 5.5 | 59.1 | 39.2 | **104.0** | 103.8-104.1 |
| | SPMW mesh, RTL only | 2.9 | 32.8 | - | **35.7** | 35.5-36.1 |
| | SPMW kernel, RTL only | 4.7 | 52.2 | - | 56.6 | 56.5-57.2 |
| | SPMW mesh as built | 3.0 | 83.2 | - | 86.2 | one run |

1 worker, one run each, all sixteen at once (each is one core). Seconds.

| Array | AutoSA -> TAPA | SPMW mesh, RTL only | SPMW kernel, RTL only | SPMW mesh as built |
|---|---:|---:|---:|---:|
| 4x4 | 321.0 | 165.7 | 315.9 | 398.4 |
| 8x8 | 385.0 | 164.4 | 316.0 | 398.6 |
| 16x16 | 389.1 | 165.7 | 316.0 | 399.4 |
| 32x32 | 425.1 | 166.1 | 319.2 | 400.1 |

## Where the time goes

- **Per HLS job**, a TAPA task takes about 18.5 s (16.3 s of it in
  `csynth_design`) and an SPMW role about 15 s (13 s). The rest of a job is
  `vitis_hls` start-up and header parsing, which every job pays.
- **With 8 workers the HLS wall is set by rounds of jobs:** the mesh's 9 roles
  are two rounds (33 s), the kernel's 15 roles and then 3 feeders three (51 s),
  TAPA's 18 tasks three (59 s). The job counts do not grow with the array in any
  flow, so neither does the HLS phase.
- **What grows is TAPA's top-level RTL:** `tapa link` takes 2.2 s at 4x4 and
  39.2 s at 32x32, where it wires 2,214 task instances. SPMW writes its fabric
  RTL in the frontend, 1.9 to 2.9 s for the mesh.
- **SPMW as built** spends another 25 s a role in `export_design`, which is what
  E5's earlier figures (79.7 s on a quiet machine) contain; here, at load 7 to 9,
  it is 84 to 86 s.

## How it was run

On brg-zhang-xcel, Vitis HLS 2023.2, `xcu280-fsvh2892-2L-e`, 3.333 ns, from a
private copy of the branch at `9de31f71` (`/scratch/hc676/e22_autosa_tapa/allo`)
linked to the compiled MLIR library every recent SPMW result used. AutoSA at
`b61a1b4`, TAPA 0.1.20250127. Every run is a fresh directory and records its load
average at the start and end (`runs.ndjson`; 6 to 10 during the 8-worker runs, up
to 17 during the one-worker batch).

- `scripts/run_one.sh` runs one flow at one size and worker count;
  `scripts/parse_run.py` turns its log into one line of `runs.ndjson`.
- `scripts/cmp2.sh` ran the SPMW 8-worker rows (02:04 to 02:33);
  `scripts/cmp3.sh` the TAPA 8-worker rows (02:45 to 02:59) and then every
  one-worker row (02:59 to 03:06). `driver.log` has every run in order.
- `scripts/make_cmp.py runs.ndjson results.csv` makes the tables.

**Two corrections, both on the TAPA side, before these rows.** The first TAPA rows
ran `analyze ... synth` without `link`, so they stopped before TAPA's top-level
RTL; and they ran with SPMW's conda environment first on `PATH`, where `tapa
analyze` takes 34 s instead of 4 s on the same source. Both are fixed in
`run_one.sh` (TAPA runs in its own environment, as `tapa compile` did in E1's
`autosa_tapa/`); the superseded rows, and the first one-worker batch that ran
beside them, are kept in `superseded_runs.ndjson`.

## Caveats

- **The designs are not the same scope.** AutoSA's includes its memory interface
  (9 of its 18 tasks); SPMW's mesh is the array with boundary buffers. SPMW's
  kernel is the like-for-like scope and is in the tables.
- **The TAPA 8-worker rows ran after the SPMW ones**, not interleaved with them,
  at the same load (6 to 10).
- **The one-worker rows are single runs, all sixteen at once**, so each shares the
  machine with the other fifteen.
- **The as-built mesh is one run per point.**
