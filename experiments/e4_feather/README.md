# E4: FEATHER, as published RTL and as an SPMW port

Same layout as E1 to E3: `<framework>/S<n>/{source,generated,report}`.
`feather_rtl/` is the published Verilog in its three variants (shipped,
corrected, and corrected with the row-wise weight loader); `spmw/` is the port.
Report files are prefixed by variant and mode, because one array size carries
several of both.

## The compute is identical; the difference was a one-off weight load, and it is now fixed

**`cycles_per_tile` is the same on both sides at every size: 4.0, 8.0, 16.0 --
on the GEMM workload and on the conv one alike.** Once loaded, the two arrays
compute at exactly the same rate, which is what one would expect of the same
architecture expressed twice. That was true before this section's change and it
is still true after it -- the change touches the weight select and nothing
else. Both workloads are measured below; conv is the one FEATHER's own paper
evaluates on, and it does not change the conclusion.

Everything that separated them was the initial weight load. The published
controller admits **one processing element per cycle** (`r_pe_sel <= r_pe_sel +
1`, free-running) and the array holds `N^2` elements with an `N`-deep weight
file each, so a load cost `N^3` cycles. That is the whole of the end-to-end
gap, and at 16x16 it was 4,096 cycles against 8,192 of compute.

**It is a controller limitation, not an architectural one.** The weight port is
already `N` bytes wide, and the loader used one byte of it per cycle. Driving
the full width loads a whole **row** of the array at a time, `N^2` cycles
rather than `N^3`. The rest of this section is that change and its measurement,
not a prediction.

### The change: two lines, both in the select

`scripts/loader/apply_row_loader.py` (and `row_loader.diff`) carries it, on top
of the corrected controller. Nothing in the datapath, the reduction network or
the arithmetic moves.

The wires were already there:

| already in the published RTL | |
|---|---|
| `feather_top.v:51` | `WEIGHTS_DATA_WIDTH = 8*DPE_COL_NUM` -- the port is `N` bytes |
| `feather_top.v:523` | `w_dpe_weights[0][COL]` is that port's byte `COL` |
| `feather_top.v:524` | `w_dpe_weights_valid[0][COL]` -- a valid per column |
| `feather_controller.v:794` | in a feed state every one of those valids is `~0`, i.e. asserted |

So all `N` columns are already offered their own byte, with their own valid,
every feed cycle. What threw `N-1` of them away was the select, in two places:

- `feather_pe.v` stored only when `i_pe_sel == THIS_PE_ID`, and `THIS_PE_ID` is
  the *global* `DPE_ROW_NUM*col + row` spanning `0..N^2-1`. It now matches the
  **row field** of that id, `THIS_PE_ID % WEIGHTS_DEPTH`.
- `feather_controller.v` free-ran `r_pe_sel` through all `N^2` ids. It now
  wraps at `WEIGHTS_DEPTH-1`, so a sweep is `N` cycles, not `N^2`.

The `o_pe_sel` daisy chain does not fight this, which was the thing to check
before relying on it: the weight bytes, `pe_sel` and the ping/pong select all
descend the *same* per-column chain a row a cycle, so PE row `r` still meets
its own `pe_sel` value in the same cycle as its own weight byte. The chain is
what makes the row-wise select work rather than what blocks it.

The SRAM image is re-packed to match -- `N` distinct weights per word instead
of one (`e4_feather_gen.py --loader row`). The image is `N` times shorter and
every one of its byte lanes is live; before, at most one lane of each row was.
**Each PE ends up holding exactly the same bytes in exactly the same slots.**

### What it measures

Weights resident (MODE 1), the `gemm128 ..._resident_general` rows:

| Array | | published loader | row-wise loader | SPMW port |
|---|---|---:|---:|---:|
| 4x4 | first output | 81 | **33** | 50 |
| | completion | 131,149 | **131,101** | 131,118 |
| | `cycles_per_tile` | 4.0 | 4.0 | 4.0 |
| 8x8 | first output | 539 | **91** | 96 |
| | completion | 33,299 | **32,851** | 32,856 |
| | `cycles_per_tile` | 8.0 | 8.0 | 8.0 |
| 16x16 | first output | 4,141 | **301** | 164 |
| | completion | 12,317 | **8,477** | 8,340 |
| | `cycles_per_tile` | 16.0 | 16.0 | 16.0 |

The load itself lands exactly on the prediction. The feed is `N^2` cycles --
16, 64, 256 -- and the first output moves earlier by exactly `N^3 - N^2`:

| Array | `N^3`, published | `N^2`, predicted | measured feed | first output earlier by | `N^3 - N^2` |
|---|---:|---:|---:|---:|---:|
| 4x4 | 64 | 16 | **16** | 48 | 48 |
| 8x8 | 512 | 64 | **64** | 448 | 448 |
| 16x16 | 4,096 | 256 | **256** | 3,840 | 3,840 |

**The compute rate is untouched**, which is the thing that had to hold: 4.0,
8.0, 16.0 as before, and in every run the steady interval's min, median and max
are all the same number, so it is not an average hiding a stall.

The end-to-end comparison with SPMW changes shape completely:

| Array | SPMW total | FEATHER before | after | before | after |
|---|---:|---:|---:|---:|---:|
| 4x4 | 131,118 | 131,149 | 131,101 | RTL +0.02% | **RTL 0.01% faster** |
| 8x8 | 32,856 | 33,299 | 32,851 | RTL +1.35% | **RTL 0.02% faster** |
| 16x16 | 8,340 | 12,317 | 8,477 | RTL +47.7% | RTL +1.64% |

So the 47.7% figure was the loader, as this section said before it was fixed.
On first output SPMW is now ahead only at 16x16, 164 against 301, and behind at
4x4 and 8x8. The earlier "1.6x, 5.6x, 25.2x latency advantage" reading was
wrong twice over: it compared a startup phase against a compute rate, and the
startup phase was a controller artefact worth a factor of `N`.

Where the load is paid *per tile* rather than once -- MODE 0, the
`..._feed_general` rows -- the same change is worth the same factor of `N` on
the whole workload, and those runs also check the assembled 128x128x128 GEMM
against numpy on the host, which the resident runs cannot:

| Array | `cycles_per_tile` before | after | completion before | after | host reduction |
|---|---:|---:|---:|---:|---|
| 4x4 | 64.0 | **16.0** | 2,097,169 | **524,305** | pass |
| 8x8 | 512.0 | **64.0** | 2,097,179 | **262,171** | pass |
| 16x16 | 4,096.0 | **256.0** | 2,097,197 | **131,117** | pass |

### The conv workload, re-measured

The section above is the 128x128x128 GEMM. `results.csv` also carries a conv
workload -- **16x16x64 -> 64 channels, 3x3, stride 1, pad 1, NHWC int8**, the
drivers' own tiling with `RS = 9` padded up to a multiple of the array side,
the host accumulating over channel and tap blocks -- and its FEATHER column was
measured *before* the loader fix, so it was still carrying an `N^3` feed. It is
re-measured here on the same three variants.

Getting there needed one piece of housekeeping. Two copies of the generator had
diverged: the conv programs for `AW = 8` and `16` live in the one this
experiment's reporting used, and `--loader row` in the one the loader fix used.
They are merged (`scripts/loader/e4_feather_gen.py`, a clean three-way merge --
the two changesets touch disjoint code, and `selftest` covers both the
conv-at-any-`AW` reduction programs and the two loaders agreeing on every PE's
weight file). **The merged generator reproduces every recorded conv baseline to
the cycle** -- 81/786,509, 539/262,675, 4,141/69,661 -- which is what makes the
new rows comparable to the old ones rather than merely adjacent to them.

Weights resident (MODE 1), the comparable pair:

| Array | | published loader | row-wise loader | SPMW port |
|---|---|---:|---:|---:|
| 4x4 | first output | 81 | **33** | 50 |
| | completion | 786,509 | **786,461** | 786,478 |
| | `cycles_per_tile` | 4.0 | 4.0 | 4.0 |
| 8x8 | first output | 539 | **91** | 96 |
| | completion | 262,675 | **262,227** | 262,232 |
| | `cycles_per_tile` | 8.0 | 8.0 | 8.0 |
| 16x16 | first output | 4,141 | **301** | 164 |
| | completion | 69,661 | **65,821** | 65,684 |
| | `cycles_per_tile` | 16.0 | 16.0 | 16.0 |

**The conv workload says exactly what the GEMM one says**, which is the point
of running it. `cycles_per_tile` is identical on both sides at every size and
was identical before the change too, so the compute rate is the same
architecture expressed twice. Everything that separated them was the load, and
the saving is `N^3 - N^2` to the cycle -- 48, 448 and 3,840 -- the same law the
GEMM runs obey.

End to end the shape matches as well:

| Array | SPMW total | FEATHER before | after | before | after |
|---|---:|---:|---:|---:|---:|
| 4x4 | 786,478 | 786,509 | 786,461 | RTL +0.004% | **RTL 0.002% faster** |
| 8x8 | 262,232 | 262,675 | 262,227 | RTL +0.17% | **RTL 0.002% faster** |
| 16x16 | 65,684 | 69,661 | 65,821 | RTL +6.05% | RTL +0.21% |

On first output SPMW is ahead only at 16x16, 164 against 301, and behind at 4x4
and 8x8 -- the same crossover the GEMM rows show, for the same reason.

Where the load is paid **per tile** (MODE 0), the change is worth the same
factor of `N`, and these runs also check the host reduction against numpy,
which the resident runs cannot. There is no SPMW column here: SPMW's per-tile
variant is `feather_stream`, a different design that streams every operand, not
this one re-fed.

| Array | `cycles_per_tile` before | after | completion before | after | host reduction |
|---|---:|---:|---:|---:|---|
| 4x4 | 64.0 | **16.0** | 12,582,929 | **3,145,745** | pass |
| 8x8 | 512.0 | **64.0** | 16,777,243 | **2,097,179** | pass |
| 16x16 | 4,096.0 | not yet run | 16,777,261 | | |

The 16x16 MODE 0 pair is a 16.7-million-cycle xsim and is still running; its
`before` column is the recorded `rtl_fixed_conv_N16_feed_general` row. The
other five conv rows are in `results.csv` as `rtl_rowload_conv_*`.

### How it was checked, because a wrong weight protocol looks fine

A mesh fed the wrong weights still runs: it emits well-formed rows with correct
handshaking, the right row count and plausible timing. Every number above
therefore rests on a bit-exact check, not on the output looking sensible.

- **Every run is bit-exact against the model of the RTL's own arithmetic.**
  32,768 / 4,096 / 512 tiles in the resident runs, the same again in the feed
  runs, zero bad elements anywhere.
- **The PEs' weight files are dumped and compared, slot by slot.** The bench
  reads `r_local_weights_buffer_ping` out of all `N^2` PEs at the end of the
  feed and checks it against the image the published loader builds: 0 of 64,
  512 and 4,096 slots wrong. This is now a pass/fail condition of the run, not
  a diagnostic printout.
- **The two images are compared directly** (`scripts/loader/compare_images.py`),
  by replaying each one through the feed rule its own hardware implements: the
  same `N^2 x N` array of PE files comes out of both, the new image is exactly
  `N` times shorter, and its rows carry `N` live byte lanes where the old ones
  carried at most 1.
- **The full validation matrix passes at every size**: every BIRRD program the
  drivers ship, x operand pattern (small / full / sparse), x zero point
  ((0,0), (7,5), (128,128)), x seed -- 162 runs at 4x4 and 54 at 8x8 and 16x16,
  the same counts the published loader passes, all bit-exact.
- **Two negative controls cross the image and the hardware** and both fail
  (`rtl_rowload_negctl_*`): the old select fed the new image, and the new
  select fed the old image. 0 of 32,768 tiles correct and 60 of 64 PE slots
  wrong in each. They are on the record because they are what shows the check
  can fire.

The baselines were re-run through the same harness rather than quoted, and
reproduce the recorded rows exactly: 81 / 131,149, 539 / 33,299, 4,141 / 12,317
resident, and 81 / 2,097,169 at 4x4 in feed mode.

### The 8x replication is a different thing, and it is not removable

The old text here said the `AW//2` copies of every weight in
`examples/feather/gemm.py:85` are "one copy per switch stage". That reading is
wrong, and it is wrong in a way only 16x16 hides: the copy count is `AW//2`,
which is 8 at 16x16 where `2*log2(N)` is also 8, but 4 at 8x8 against 6 stages
and 2 at 4x4 against 4. The coincidence is the whole of the resemblance.

What the copies actually are: the array's **column** dimension is the GEMM's
`M` dimension. Column `j` carries output row `j mod Mt` of the `A` tile, and
`j // Mt` picks which half of the `K` split it is reducing. The weight a PE
needs depends on `(i, k)` and on that half -- **not** on `j`. So the `Mt =
AW//2` columns of a half all need the same weight byte at the same time, and
the measurement confirms it: `weights[i, j, k]` is constant across the left
`Mt` columns and across the right `Mt`, and the two halves differ.

That is ordinary weight-stationary reuse across `M`, not a reduction-network
artefact. Each PE has a **private** weight file and there is no path from one
PE's file to another, so the copies must physically exist. **The replication is
not removable, and it should not be changed.**

Its *load cost*, separately, is not fully removed by this change either. A
16x16 tile holds `Kt*Nt = 512` distinct weights; the row-wise loader moves
4,096 bytes in 256 cycles, so it is still sending each distinct weight eight
times. Delivering only the 512 distinct bytes over the `N`-byte port would be
32 cycles, but it needs the loader to fan **one** lane out to `Mt` columns --
a change to the weight write path, not to the select, and outside what this
one does. So of the two compounding choices the earlier text identified, one is
now fixed and one is real:

| what | cycles at 16x16 |
|---|---:|
| the published loader, one slot a cycle | 4,096 |
| **the row-wise loader, the full `N`-byte port** | **256** |
| a loader that also broadcast within a `K` half (not built) | 32 |
| SPMW's entire first output at 16x16 | 164 |

## The shipped RTL is broken, and every number here uses the corrected one

FEATHER's published controller **fails its own RTL simulation** at 4x4, 8x8 and
16x16 -- the three `rtl_orig_*` `fail` rows in `results.csv`. The fault is in
the PEs' ping/pong select: it toggles once per weight per visit, and the
shipped toggle puts a PE's even-index weights in the wrong half, so the array
computes against stale weights. `scripts/harness/apply_patch.py` carries the
one-line correction and the reasoning.

Every `rtl_fixed_*` row -- which is every FEATHER number quoted anywhere in
this experiment -- uses the corrected controller. Both variants are placed and
routed, and their area and timing are within noise of each other, so the
correction costs nothing and is not a thumb on the scale. The broken rows are
kept rather than deleted, because a baseline that does not pass its own
testbench is a finding about the baseline.

## The comparable pair, and the thing that has to be read with it

Two SPMW variants were built. Only **`feather` (single tile, resident
weights)** is a port of what the RTL does; `feather_stream` streams every
operand instead and is a different design, three to ten times larger, failing
timing at 16x16. The table below is the comparable pair.

| Array | System | LUT | FF | DSP | Slack |
|---|---|---:|---:|---:|---:|
| 4x4 | SPMW port | 2,305 | 3,639 | **16** | +1.200 ns |
| | FEATHER RTL, corrected | 2,309 | 3,378 | **0** | +1.174 ns |
| 8x8 | SPMW port | 9,974 | 14,989 | **64** | +0.866 ns |
| | FEATHER RTL, corrected | 9,694 | 15,332 | **0** | +0.355 ns |
| 16x16 | SPMW port | 31,019 | 49,673 | **256** | +0.338 ns |
| | FEATHER RTL, corrected | 57,499 | 91,251 | **0** | +0.120 ns |
| 32x32 | SPMW port | not run | | | |
| | FEATHER RTL, corrected | 338,694 | 624,270 | **0** | **-0.227 ns** |

**The DSP column is the caveat.** FEATHER's RTL routes with **zero DSPs** at
every size; the SPMW port uses one per element -- 16, 64, 256. So the LUT
comparison is not like for like: at 16x16 the port's 26,000 fewer lookup tables
and 41,000 fewer registers are bought with 256 DSP blocks the RTL does not use.
Vivado is inferring DSPs from the port's multiply and implementing FEATHER's in
fabric. Anyone reading the LUT column as an area win should read the DSP column
in the same row; the honest statement is that the two designs land in different
parts of the device, not that one is half the size of the other.

At 4x4 and 8x8 the lookup-table counts are within 3 per cent of each other,
which is the more informative result: at those sizes the port is neither
cheaper nor dearer, it just moves the multipliers.

The row-wise loader lands on top of the corrected RTL's row of this table
without moving it -- within 1.7% on LUTs, within 0.03 ns on slack, and it is
the corrected RTL's cycle counts rather than its area that the change alters.
The numbers are under "Three controller variants" below.

**Both 32x32 builds fail timing** (-0.227 ns shipped, -0.206 ns corrected), and
the SPMW port was never run at that size, so 32x32 supports no comparison at
all.

## Three controller variants

`results.csv` carries the published controller, a corrected one, and the
corrected one with the row-wise weight loader on top.

The shipped controller **fails its own RTL simulation** at 4x4, 8x8 and 16x16
(three `fail` rows); the corrected one passes at every size. Both are kept and
both are placed and routed, because the area and timing numbers are within
noise of each other and the correction is a control-path fix, not an
architectural change.

The `rtl_rowload_*` rows are the corrected controller plus the row-wise select
(`scripts/loader/`), and it is placed and routed too, at the same 3.333 ns out
of context on the same part:

| Array | LUT corrected | LUT row-wise | FF corrected | FF row-wise | slack corrected | slack row-wise |
|---|---:|---:|---:|---:|---:|---:|
| 4x4 | 2,309 | 2,291 | 3,378 | 3,378 | +1.174 ns | +1.074 ns |
| 8x8 | 9,694 | 9,690 | 15,332 | 15,335 | +0.355 ns | +0.386 ns |
| 16x16 | 57,499 | 58,465 | 91,251 | 91,249 | +0.120 ns | +0.115 ns |

All three route with zero unrouted nets, zero TNS and still zero DSPs. The
differences are inside the same band as the shipped-vs-corrected pair, which is
a control-path fix of the same kind and differs by 96 LUTs and 0.058 ns at
16x16 while being logically the same size. **The load is `N` times cheaper and
the area and timing are unchanged.**

Worth recording that this contradicts the obvious guess. Narrowing `r_pe_sel`'s
range from `N^2` to `N` looks like it should shrink the design -- the daisy
chain carries one `PE_SEL_WIDTH`-bit register per PE, `2*log2(N)` bits where
`log2(N)` now suffice, 2,048 flip-flops at 16x16. Not one of them goes away.
The upper bits are only constant because the counter never reaches them, and
that is a statement about the counter's reachable states rather than a constant
Vivado can propagate, so the full width survives. Getting the flip-flops back
would mean narrowing the parameter itself, and it is not worth a wider change
for 1% of the registers.

Two `rtl_rowload_negctl_*` rows are `fail` **on purpose**: they cross the
weight image with the wrong select, and they are kept as the evidence that the
bit-exact check catches a wrong weight protocol.

## Files

- `source/` -- the SPMW port (`spmw_feather.py`) on one side; the RTL driver
  and its testbench (`spmw_feather_rtl.py`, `tb_feather_rtl.sv`) on the other.
- `generated/` -- what the split backend emitted for the port, one `.cpp` and
  one `.sv` per role, re-staged from the same `scripts/spmw_build_array.py`
  entry point the measured builds used (9 roles at 4x4, 11 at 8x8 and 16x16).
  `feather_rtl/*/generated/` holds a note instead: hand-written RTL has no
  generation step.
- `report/` -- routed utilisation, hierarchy, timing and route status per
  variant, and the simulation results per workload and mode.
- `scripts/` -- what drove the runs, split by which agent produced them.
  `scripts/loader/` is the row-wise weight loader: `apply_row_loader.py` and
  `row_loader.diff` are the RTL change, `e4_feather_gen.py --loader row` packs
  the matching image, `compare_images.py` checks the old and new images deliver
  the same PE files, and `append_rows.py` wrote the `rtl_rowload_*` rows.

Left behind: each RTL simulation's `bus.log` is 2.8 MB of bus trace and
`tiles.log.gz` its tile dump; nothing in the table reads them. The full tree is
33 MB on the machine.
