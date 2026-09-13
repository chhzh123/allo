# E4: FEATHER — see `experiments/e4_feather/README.md`

This directory is a **results bundle**: `results.csv` here is a byte-identical
copy of the experiment's own, and is the point of the bundle. `results.json`
and `README_tables.md` are rendered from it.

The README that used to sit here was an assembled snapshot that had gone stale
against its own data -- it showed `not_run` for GEMM 16x16 and conv 8x8/16x16
rows that `results.csv` records as passing. Rather than keep a second,
diverging account of the same experiment, it is replaced by this pointer.

- **The experiment's README** is `experiments/e4_feather/README.md`.
- **The long-form write-up** it was assembled from is
  `experiments/e4_feather/scripts/reporting/README_body.md`, with
  `assemble_readme.sh` beside it.

## Why the tables are rendered now, and not collected

`README_tables.md` and `results.json` went stale twice in the same way: they
were built by `e4_collect.py` / `e4b_collect.py`, which read run directories on
the machine **and looked rows up by a hard-coded list of run ids**. Anything
whose id was not on that list -- the row-wise loader's rows, then the
re-measured conv ones -- was dropped without a word, while the file went on
saying `results.csv` was its source. By the time it was caught the JSON was 19
rows behind the CSV.

Both are now written by
`experiments/e4_feather/scripts/reporting/render_bundle_tables.py`, which reads
the CSV beside them and nothing else, groups rows by what is in them rather
than by name, and ends with a coverage count and a list of every row that
landed in neither table. Regenerate with:

    python3 experiments/e4_feather/scripts/reporting/render_bundle_tables.py \
        experiments/e4_feather/results.csv experiments/results/e4_feather
