# E6: grouped attention-PV, three systems

Same layout as E1 to E5: `<system>/M<n>/{source,generated,report}`, M being the
sequence length -- 6, 64 and 4096. `hls_baseline/all_M/` is the exception and
is named for it: the HLS baseline's place-and-route does not depend on M, so it
is one folder rather than a copy in each.

## Scope, before any comparison

The HLS baseline was routed at **three scopes**, and its `run_id` says which:
`_array` (the PE array alone), `_kernel` (the array with its memory interface)
and `_export` (the exported IP). They differ by nearly 2x in lookup tables. The
SPMW rows carry **one unlabelled scope**. Nothing in the data says which HLS
scope it corresponds to, so the table below shows all three and the
`array` column is the conservative comparison -- the smallest HLS figure
against the SPMW one. Treat the ratio as bounded, not exact.

## Results at M = 4096

| System | Variant | PEs | DSP | Cycles | LUT array | LUT kernel |
|---|---|---:|---:|---:|---:|---:|
| HLS baseline | conventional | 16 | 16 | 37,706 | 4,422 | 9,411 |
| HLS baseline | conventional, packed lanes | 16 | 16 | 4,957 | 4,375 | 9,452 |
| HLS baseline | grouped | 16 | 16 | 4,470 | 4,046 | 7,070 |
| SPMW conventional | 2-pass | 16 | 16 | 8,256 | 1,207 | -- |
| SPMW grouped | const seed | 16 | 16 | 5,499 | 1,037 | -- |
| SPMW grouped | zero seed | 16 | 16 | **4,136** | 1,072 | -- |
| SPMW conventional | paper ungrouped 4x2 | **8** | **8** | 5,483 | 529 | -- |

Two things to read carefully:

- **The last row uses half the hardware.** `paper_ungrouped_4x2` has 8 PEs and
  8 DSPs, not 16. Its 5,483 cycles are not comparable with the rows above it on
  cycles alone; on cycles times PEs it is close to the 16-PE grouped variants.
- **Grouping helps the HLS baseline far more than it helps SPMW**, because the
  HLS conventional baseline is so much worse to begin with: 37,706 to 4,470 is
  8.4x for HLS, while SPMW goes 8,256 to 4,136, a factor of 2.0. SPMW's
  *conventional* design is already 4.6x faster than the HLS conventional one.
  So the grouped-vs-conventional speedup is not a property of grouping alone;
  it depends on what the ungrouped baseline was.

Every SPMW cosimulation was run at three seeds (`_s0/_s1/_s2`) and all three
agree exactly at every M and variant, which is why `results.csv` carries three
identical rows rather than a mean.

## A gap, recorded

`generated/` is empty in this experiment. E6's variants are not one registry
design at several sizes -- they are separate drivers -- so the generated code
cannot be re-staged from a single `--design`/`--size` pair the way E2's and
E4's were. Each `generated/README.md` says so and points at the drivers in
`scripts/`. This is the one place where these folders do not fully reproduce
the table on their own.

## Files

`report/` holds each run prefixed by variant and mode: `run.json` per
cosimulation, and utilisation, hierarchy, timing and route status per
place-and-route. `results_detail.csv` carries the per-pass breakdown that
`results.csv` summarises. Left behind: `job_logs/`, `diagnostics/`,
`source_change/` and a 1.1 MB `hls_failed_before_depth_fix/` tree from an
earlier attempt. The full tree is 32 MB on the machine.
