# E7: lines of code, HLS against SPMW

**This experiment does not use the `<framework>/<size>/{source,generated,report}`
layout, and forcing it would be misleading.** E7 builds nothing: it counts
source lines. There is no per-size design, nothing is generated, and there is
no synthesis or place-and-route to report. Its artefacts are the count records
and the manifest that says exactly which files were counted at which commit.

The counted files are not copied here either. They *are* the repository's own
sources; `counts/*/included_files.txt` names every one and
`counts/*/manifest.json` pins the revision, which is what makes the count
reproducible without duplicating the tree into an experiment folder.

## Read the denominator first

Twenty rows, and **only 8 have a ratio at all**:

| Status | Rows | Meaning |
|---|---:|---|
| `pass` | 8 | both an HLS and an SPMW implementation exist and were counted |
| `unsupported` | 11 | **no HLS counterpart exists to compare against** |
| `environment_blocked` | 1 | AutoSA's generated GEMM; the counter could not run |

The 11 `unsupported` rows are the designs where only an SPMW version was
written -- the FFT butterfly network, the complete mini-TPU stage engine,
FEATHER and its variants, and three of the four paper listings. Their
`spmw_design_lines` are recorded and their HLS column is empty. That is not a
ratio of infinity; it is an absent measurement, and it should not be read as
one direction of the result.

## The eight rows that do compare

| Workload | Kind | HLS | SPMW | Ratio |
|---|---|---:|---:|---:|
| Tiled GEMM, two-level | design | 153 | 32 | 4.78 |
| Systolic GEMM, output-stationary int8 | design | 94 | 24 | 3.92 |
| Multi-cache GEMM, daisy-chained drain | design | 113 | 40 | 2.83 |
| Mini-TPU matrix unit, weight-stationary | design | 113 | 40 | 2.83 |
| Folded radix-2 FFT, HP-FFT n256/UF1 | design | 299 | 106 | 2.82 |
| Grouped attention-PV | design | 112 | 53 | 2.11 |
| Matrix unit, staged double-buffered | variant | 113 | 54 | 2.09 |
| Paper listing, Fig. comparison | fragment | 30 | 26 | **1.15** |

Median 2.83, range 1.15 to 4.78.

**The 1.15 is worth as much attention as the 4.78.** The paper's own figure
listing -- the fragment a reader actually sees -- is the closest of the eight,
because a figure is already trimmed to the interesting lines on both sides. The
larger ratios come from complete designs, where the HLS version carries
boilerplate the figure omits. Quoting a single number from this experiment
without saying which kind of artefact it counts would misrepresent it, so
`results.csv` carries a `kind` column (`design`, `variant`, `fragment`,
`generated`) and an `in_paper_table` flag.

## Files

- `counts/loc_recount_2026-09-06/` -- the recount: `counts.json`, `counts.md`,
  `included_files.txt` (every counted file), `manifest.json` (the revision) and
  `counter_sha256.txt` (the hash of the counter that produced them).
- `counts/fft_folded/` -- the same four records for the folded FFT, counted
  separately.
- `scripts/spmw_loc2.py` -- the counter itself. `counter_sha256.txt` is its
  hash, so the records can be checked against the script that made them.
