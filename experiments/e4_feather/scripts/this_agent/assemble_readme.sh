#!/bin/bash
# README.md = README_body.md + "## 9. Results" (README_results.md, written last) + the collector's tables
B=/scratch/hc676/e4b_work
R=/scratch/hc676/spmw_eval_remaining_2026-09-06/e4_feather
{ cat $B/README_body.md; echo; cat $B/README_results.md 2>/dev/null; echo; echo "## 10. Tables (as written by e4b_collect.py; results.csv is the source)"; echo; cat $R/README_tables.md; } > $R/README.md
echo "README.md: $(wc -l < $R/README.md) lines"
