# E4: FEATHER — see `experiments/e4_feather/README.md`

This directory is a **results bundle**: `results.csv` and `results.json` here
are byte-identical copies of the experiment's own, and are the point of the
bundle.

The README that used to sit here was an assembled snapshot that had gone stale
against its own data -- it showed `not_run` for GEMM 16x16 and conv 8x8/16x16
rows that `results.csv` records as passing. Rather than keep a second,
diverging account of the same experiment, it is replaced by this pointer.

- **The experiment's README** is `experiments/e4_feather/README.md`.
- **The long-form write-up** it was assembled from is
  `experiments/e4_feather/scripts/reporting/README_body.md`, with
  `assemble_readme.sh` beside it.
