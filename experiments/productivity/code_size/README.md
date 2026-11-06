# Code size: design-only source lines

Table 3 of the paper (arXiv version).

Nothing is built here: the folder counts source lines. A line counts when it
holds code and belongs to the design; configuration, tests, references,
comment-only lines and docstrings are excluded on both sides. The rule and the
categories are in [`counts/README.md`](counts/README.md). The counted files
are the repository's own sources, mostly outside `experiments/`; the records
pin each one by sha256 and each selected range by line numbers and sha256.

## Layout

| Path | Contents |
|---|---|
| `results.csv` | the three rows Table 3 takes from here: `fft_folded`, `systolic_gemm`, `feather`. `report_paths` is relative to `counts/` |
| `counts/loc_recount_2026-09-06/` | the recount: `manifest.json` (what each row selects, by symbol), `counts.json` (every range and its sha256), `counts.md` (tables and per-design notes). 19 rows; Table 3 uses three |
| `counts/fft_folded/` | the folded FFT, counted separately on brg-zhang-xcel; the same three files |
| `scripts/spmw_loc2.py` | the counter, sha256 `68a47ad0...`, the hash the paper's table script asserts. The repository's `scripts/spmw_loc2.py` (`fea0ba57...`) is the same program after black (identical AST); `fft_folded` was counted with that one |
| `sources/fft_folded_spmw.py` | the folded SPMW FFT, `tests/dataflow/spmw/test_spmw_fft_sdf.py` as committed in d98c3538 |

## Table 3, cell by cell

| Row | HLS | RTL | SPMW | Ratio | Counted here |
|---|---:|---:|---:|---:|---|
| Systolic array | 94 | -- | 24 | 3.92x | HLS and SPMW: `systolic_gemm` |
| FFT | 299 | 5,284 | 106 | 2.82x | HLS and SPMW: `fft_folded`; not RTL |
| Mini-TPU | 673 | 3,226 | 154 | 4.37x | SPMW only, derived from `gpt_stage_v1` |
| FEATHER | -- | 1,802 | 97 | 18.58x | SPMW only: `feather` |

Ratios are HLS / SPMW, except FEATHER (RTL / SPMW).

- **Systolic array.** HLS: [`gemm_output_stationary.cpp`](../../../examples/spmw/baselines/hls/gemm_output_stationary.cpp),
  99 lines = 94 design + 5 config. SPMW: `gemm_int8_of` in
  [`test_spmw_gemm_int8.py`](../../../tests/dataflow/spmw/test_spmw_gemm_int8.py).
- **FFT.** HLS: HP-FFT's `n256/UF1/FFT.cpp` at c4611b8, every function and
  both tables; its `FFT.cpp` and `FFT.h` are the files in
  [`../../performance/fft/n256/hpfft/N256/source/UF1/`](../../performance/fft/n256/hpfft/N256/source/UF1/)
  (same sha256). The record's title calls UF1 one sample per cycle; its ideal
  interval is N/2, two samples per cycle, as the paper says. SPMW: `twiddles`,
  `bitrev` and `fft_sdf_of` in [`sources/fft_folded_spmw.py`](sources/fft_folded_spmw.py).
- **Mini-TPU.** SPMW 154 is the stage engine as measured,
  [`gpt_stage_v1.py`](../../../tests/dataflow/spmw/gpt_stage_v1.py): its
  `stage_engine` (148, without the `engine.spmw_parts` hook) plus `VpuIO`
  (6) from [`test_spmw_tpu_isa.py`](../../../tests/dataflow/spmw/test_spmw_tpu_isa.py).
  The `gpt_stage_v1` row of `counts/loc_recount_2026-09-06/counts.md` reads 163
  because it also lists the unused module-level `MacIO` (9); that row is not
  in `results.csv`.
- **FEATHER.** SPMW: `feather` and its helpers `reverse_bits`, `birrd_shape`,
  `stage_bits` in [`test_spmw_feather.py`](../../../tests/dataflow/spmw/test_spmw_feather.py),
  without the cosim hooks at the end of `feather()`.

**Not counted here:** Spiral's RTL (5,284), VTA's HLS (673: 523 C++ + 150 Tcl,
VTA d4a15f6), its Chisel (3,226) and FEATHER's RTL (1,802). They come from the
paper's own recounts (`spiral_loc_20261011`, `loc_vta_chisel_20261011`,
`loc_rtl_20260913`) in its evidence archive, not in this repository; Spiral's
generated RTL is not redistributed.

## Pinned revisions

- The recount ran at d7ce0da. The five files its three rows read are identical
  at 23eedb2, where the paper pins its SPMW counts. Since then
  `test_spmw_feather.py` gained a build hook in `feather()` (ed056f9a), which
  shifts the manifest's line exclusion: the same manifest at HEAD gives
  FEATHER 102. Count at 23eedb2.
- `fft_folded` counted `test_spmw_fft_sdf.py` before its last five lines of
  `fft_sdf_of` were added (lines 177-181 of the kept file: four comment lines
  and the HLS hook `engine.spmw_pipeline_style = "flp"`). The kept file counts
  107; without those lines it hashes to the record's `eae8d087...` and counts
  106.

## Reproducing

From this folder:

    mkdir -p /tmp/allo_23eedb2 && git archive 23eedb2 | tar -x -C /tmp/allo_23eedb2
    python3 scripts/spmw_loc2.py counts/loc_recount_2026-09-06/manifest.json \
        --root allo=/tmp/allo_23eedb2 --root paper=<paper checkout> \
        --json counts.json --markdown counts.md --included included_files.txt

`--included` writes the per-range list `included_files.txt`; it was never
committed, and `counts.json` holds the same ranges and hashes. The manifest's
four `fragment_*` rows count the paper's listings, which have changed since
(the Fig. attn one no longer parses); without them, every other row reproduces
at 23eedb2. For the FFT, point `--root hpfft=` at HP-FFT-HLS at c4611b8
(the manifest also reads
`n256/UF1/testbench.cpp`, test code, not kept here) and `--root allo=` at a
tree whose `tests/dataflow/spmw/test_spmw_fft_sdf.py` is the counted file.
The counter imports only the standard library; its tests are
`tests/dataflow/spmw/test_spmw_loc2.py`.

The pre-cleanup E7 tree (`experiments/e7_loc/`) is at commit c20b2ee8.
