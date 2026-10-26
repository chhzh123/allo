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

## A 16x16 array, 2026-10-24

The rows above are the paper's 4x4 array. `array16/` is the same pair -- the
paper's `mac`, `act` and `drain` bodies unchanged -- on a **16x16 array of 256
PEs**, cut into G column slabs for G = 2, 4 and 8, at M = 1,024 to 8,192. All
20 runs are RTL cosimulations through E1's flow (`spmw_build_array.py
--cosim`), every output token checked against numpy.

**What one launch computes.** `Y[M, d] = P[M, L] @ V[L, d]` with reduction
`L = 16G` and width `d = 16/G`, so `L x d = 256` -- the PE count -- whatever G
is:

| G | P | V | Y | Conventional passes |
|---:|---|---|---|---:|
| 2 | `[M, 32]` | `[32, 8]` | `[M, 8]` | 2 |
| 4 | `[M, 64]` | `[64, 4]` | `[M, 4]` | 4 |
| 8 | `[M, 128]` | `[128, 2]` | `[M, 2]` | 8 |

This is a **tile** of attention's P.V, not the whole of it. Attention is
`(B,H,S,S) @ (B,H,S,D)`, reducing over the S keys; here M is the query rows
and the reduction is the L keys the array holds at once, with d of the D head
dimensions. A whole head takes `S/L` key tiles times `D/d` width tiles, which
is `S x D / 256` launches for any G, and the accumulation across key tiles is
not measured here. The same is true of the 4x4 rows above, where L = 8 and
d = 2.

**Cycles a launch**, first input to last output (`array16/results.csv`):

| M | G=2 | G=4 | G=8 |
|---:|---:|---:|---:|
| 1,024 | 1,982 | 1,907 | **1,883** |
| 2,048 | 3,803 | 3,546 | **3,248** |
| 4,096 | 7,443 | 6,823 | **5,979** |
| 8,192 | 14,725 | 13,377 | **11,440** |
| fill (first output) | 199 | 371 | 642 |
| closed form | `floor((16M+1)/9) + 162` | `floor(8M/5) + 270` | `floor(4M/3) + 518` |

The G=2 and G=8 forms are exact at all four lengths; G=4's is exact from
M = 2,048 and one cycle high at 1,024. The per-row rates, 16/9, 8/5 and 4/3,
are `2w/(w+1)` for a slab `w = 16/G` columns wide -- an observation across
three points, not a derived law. A likely reason, not traced: an activation
forwards east only inside its slab, so a wider slab is a longer horizontal
dependency a row.

**G = 8 is the fastest at every length.** The shorter psum chain of G = 2
(32 PEs in series against 128) gives it a 199-cycle fill against 642, but the
fill never pays for its slower rate; even at M = 1,024, where the fill is a
third of G = 8's launch, G = 8 is 5% ahead.

**Against the conventional mesh,** one pass is exactly `M + 128` cycles at
all four lengths. Its fabric does not depend on G -- only V's zero padding
does -- and the G = 2 and G = 8 runs give identical cycles; G = 4's pass was
not run and the row below uses the same one. The product needs G passes:

| M = 4,096 | Conventional (G passes) | Grouped | Grouped is faster by |
|---|---:|---:|---:|
| G = 2 | 8,448 | 7,443 | 1.14x |
| G = 4 | 16,896 | 6,823 | 2.48x |
| G = 8 | 33,792 | 5,979 | 5.65x |

So what grouping buys depends on how many passes it displaces: 5.65x at the
width-2 shape, 1.14x at width 8, the shape nearest a usable head dimension.
The conventional pass is the faster loop -- one row a cycle, against the
grouped design's 4/3 to 16/9 -- and loses only by needing G of them.

Four things these numbers are not:

- **One seed a point.** The 4x4 rows ran three and agreed exactly; these have
  not been shown to.
- **The conventional totals are G times one measured pass.** The partial is
  not carried through G chained launches, so reloading V and restarting the
  launch are not counted, and the real totals are somewhat higher.
- **No place-and-route.** There is no clock or area for the 256-PE array;
  at 300 MHz, G = 8 at M = 8,192 would be 38.1 us, but 300 MHz is not shown.
- **The G = 8 rows first ran under the registry's earlier name**,
  `attnpv16-<kind>-<seq>`, for the same design. `attnpv16g8-grouped-1024`
  reproduces its row to the cycle (1,883, first output 642;
  `array16/logs/check_g8_grouped_1024.log`).

To rerun: `array16/scripts/attn16g.sh <G> grouped conv` on the build machine
(design `test_spmw_attention_passes.pv_of`, registry `attnpv16g<G>-<kind>-<M>`).
`array16/report/` keeps each run's `cost.json`, `array16/logs/` its build log.
