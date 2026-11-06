# E7: design-only source size, HLS against SPMW, counted language-aware

`scripts/spmw_loc2.py` (in the Allo tree, with fixtures and 22 tests in
`tests/dataflow/spmw/test_spmw_loc2.py`) counts a physical line when it holds
code: comment-only lines, docstrings and standalone string statements do not
count, and code with a trailing comment does. Python is parsed with `tokenize`
and `ast`, C/C++ and SystemVerilog with a comment- and string-aware stripper,
TeX listings by extracting the `minted`/`lstlisting` body first. Selection is
by symbol, line range or regular expression; a missing or ambiguous symbol is
an error rather than a zero, and each touched file's unselected remainder is
reported so the categories sum to the whole file.

The same inclusion rule is applied on both sides. **design** is the unit and PE
bodies, interface and port declarations, topology and link rules, loaders,
drains and boundary bindings, design-specific interface code (AXI and HLS
pragmas, fabric signatures) and design-specific helpers. **config** is imports
and includes, constants, type aliases and sizes, and the instantiation at a
chosen size. **workload** is programs and their encoders. **test** is tests,
references, operand generators and host benchmark code, reported and excluded.
Reusable framework code (`allo/spmw`, `hls_stream.h`, `ap_int.h`) is excluded
on both sides and not reported.

## What is here

- `loc_recount_2026-09-06/` -- the main recount: `manifest.json` (what each row
  selects, by symbol), `counts.json`, `counts.md` (the tables), 
  `included_files.txt` (a sha256 per file and per selected range),
  `counter_sha256.txt` and an archived copy of the counter.
- `fft_folded/` -- the FFT addendum, added once both sides had the same folded
  architecture: HP-FFT's one-sample-per-cycle kernel (`n256/UF1`, checkout
  c4611b8) against the SPMW delay-feedback pipeline.
- `results.csv` / `results.json` -- one row per design, both sides' design and
  config counts and their ratio; rows whose HLS side has no counterpart carry
  status `unsupported` with the reason.

## Why these numbers differ from the archived ones

1. The archived SPMW counts were produced with the C/C++ rule applied to
   Python (the old counter wrote the selected region to `<file>.py.region` and
   dispatched on the suffix), so docstrings and comment-only lines counted, and
   regions ran to the first `def test_`, sweeping in instantiations and operand
   generators. Every difference reconciles exactly; `counts.md` shows the
   arithmetic per design.
2. Three HLS baselines were rewritten on 3 September after the archive was
   taken (they fixed synthesis and deadlock problems), so their counts moved.
3. The archived HLS numbers were whole files (design plus 4-10 configuration
   lines); the headline now counts design only on both sides.

## Caveats

Source size measures description effort, not development time. The tiled GEMM
pair is type-mismatched (int8 HLS against float32 SPMW) and structurally
different; the mini-TPU HLS kernel carries a tile loop and 128-bit packing that
its SPMW counterpart does not. Designs with no hand-written HLS counterpart
(the complete stage engine, FEATHER) are reported with their SPMW side only.
