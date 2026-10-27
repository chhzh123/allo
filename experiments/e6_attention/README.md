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
three points, not a derived law. The likely cause is the seed, not the slab:
these launches seed the partial-sum chain with `spmw.stream_in(0, ...)`, which
the backend folds into the top-row cells, and the 4x4 study traced that fold to
a stall loop that costs every constant-seeded design rows
(`../results/e6_attention/README.md`, "The 3-of-4 rhythm"; grouped there:
5,499 cycles folded, 4,136 streamed). Not re-traced at 16x16, but the
whole-head design below seeds from a stream and its grouped array runs a row a
cycle.

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

## A whole head on the 16x16 array, 2026-10-26

`head16/` is one head's **whole** P.V rather than a tile of it:
`Y[S, 8] = P[S, S] @ V[S, 8]`, int32, in one launch on the same 256 PEs, at
S = 64 to 1,024. Both schemes are RTL cosimulations through E1's flow, every
output token checked against numpy.

- **Ungrouped**: the plain 16x16 mesh. A P row passes across all 16 columns,
  so every column sees the same 16 keys: a key tile is 16 keys, D = 8 fills
  8 columns and the other 8 hold zeros. S/16 tiles.
- **Grouped, G = 2**: two slabs of 8 columns, the partial sums of slab 0
  running on into the top of slab 1. A key tile is 32 keys and every cell
  holds part of V. S/32 tiles.

How the tiles add up, the same in both: each cell keeps its V entry for every
tile (a file of S/keys entries, loaded before the first row); the S query rows
stream through once per tile; each row's partial sum leaves the bottom of the
chain into a FIFO S deep (`fold` to `seed_sums`) and re-enters at the top on
the next tile, so the array's own adders sum across the key tiles. The last
tile's sums are Y. The seed is a unit's stream, not a folded constant.

**Cycles for the whole head**, reset to last output (`head16/results.csv`):

| S | Ungrouped | Grouped, G = 2 | Grouped is faster by |
|---:|---:|---:|---:|
| 64 | 481 | 429 | 1.12x |
| 128 | 1,222 | 825 | 1.48x |
| 256 | 4,302 | 2,294 | 1.88x |
| 512 | 16,606 | 8,446 | 1.97x |
| 1,024 | 65,790 | 33,038 | 1.99x |

**Closed form.** Once a tile is longer than a partial sum's trip round the
loop, both run a row a cycle and take `T(S + 1) + c` cycles for `T = S/keys`
tiles: ungrouped `c = 190`, exact from S = 128; grouped `c = 238`, exact from
S = 256. Grouped streams `S^2/32` rows against `S^2/16`, so the ratio tends to
2. Below that, a tile waits for the tile before it: a sum takes about 86 cycles
round the 16-cell chain and 166 round the 32-cell one (from the first outputs,
about five cycles a cell), so at S = 64 both are bound by the loop and grouped
at S = 128 is 71 cycles over its form.

**Against Table 2's tile numbers** (`array16/`, G = 2: 1.14x at M = 4,096):
those launches computed one `[M, 32] @ [32, 8]` tile seeded with the folded
constant, which runs 16/9 cycles a row, and the ungrouped figure was twice one
measured pass. Here both schemes compute the whole product, both are measured
whole, and both run a row a cycle.

**Both are built with HLS's flushable pipelines** (`spmw_pipeline_style =
"flp"`). With the default, a cell's pipeline freezes while an input is
missing, so it finishes a row only when a later row arrives and the chain holds
about three rows a cell. A grouped tile of 64 rows then never reaches `fold` to
feed the next tile: 0 of 512 outputs (`head16/logs/stp_grouped_64.log`). The
ungrouped S = 64 head, 16 cells deep, passed that way in 468 cycles
(`stp_conv_64.log`).

Four things these numbers are not:

- **One seed a point.**
- **No place-and-route**, so no clock or area. Each cell's weight file arrives
  as one token of S/keys bytes (64 bytes ungrouped at S = 1,024), the
  transport that failed place-and-route at 64 entries on the 16x16 stage
  engine; a board build would load V another way.
- **The weights are loaded once, before the first row**, all cells at once
  (S/keys cycles, in the totals). A TPU shifts the next tile's weights in
  while the current one computes; here no tile waits for weights either way.
- **The loop FIFO is the accumulator**: S int32 a lane, 8 lanes grouped and
  16 ungrouped, block RAM in the RTL.

To rerun: `head16/scripts/attnhead.sh 64 128 256 512 1024` on the build machine
(design `test_spmw_attention_passes.pv_head`, registry
`attnhead16d8-<grouped|conv>-<S>`). `head16/report/` keeps each run's
`cost.json`, `head16/logs/` its build log.
