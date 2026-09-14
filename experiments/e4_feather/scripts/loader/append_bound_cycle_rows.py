#!/usr/bin/env python3
"""E4: the SPMW port's cycle rows re-measured with the multiply bound to fabric,
and the fill/feed split that makes the RTL rows comparable to them.

    append_bound_cycle_rows.py <results.csv>

Two corrections in one pass.

**The port's cycles moved.** `impl=fabric` is not only a resource directive:
HLS swaps a 4-cycle fused DSP multiply-add (`mac_muladd_8s_8s_32s_32_4_1`) for
a 1-cycle fabric multiply (`mul_8s_8s_16_1_1`) plus an adder, the PE gets a
stage shallower, and a partial sum crossing N cells saves about 2N. The old
rows stay; these sit beside them.

**The RTL's figure contains a weight feed and the port's does not.** The RTL's
`first_output_cycles` is `done[0] - F0 + 1`, measured from the first cycle of
the feed; the port measured is `feather_stream_x`, whose weights are a
`spmw.MemIn` and resident before the clock starts. The testbench's own
definitions give the split -- `A_BASE = G + WLEN`, tile 0's first row at
`A0 + A_BASE + LAT` with `LAT = N + 5 + 2 log2 N`, the reported figure being
tile 0 complete -- so the figure is `WLEN + LAT + N`, which reconstructs 33, 91
and 301 exactly. `fill_cycles` records `LAT + N` and `weight_feed_cycles`
records `WLEN`, on every row of both systems, so the comparable column is in
the data rather than only in the README.
"""

import csv
import math
import sys

#: SPMW port, weights resident, multiply bound to fabric: tile 0 complete.
#: Measured by `rerun_spmw_bound.sh`; the interval is unchanged at N.
BOUND = {
    ("gemm128", 4): (42, 131110), ("gemm128", 8): (80, 32840), ("gemm128", 16): (132, 8308),
    ("conv", 4): (42, 786470), ("conv", 8): (80, 262216), ("conv", 16): (132, 65652),
}


def fill(n):
    """The array's own latency to a complete tile, weights already in place."""
    return (n + 5 + 2 * int(math.log2(n))) + n


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = list(reader)
    for extra in ("fill_cycles", "weight_feed_cycles"):
        if extra not in fields:
            fields.append(extra)

    by = {r["run_id"]: r for r in rows}
    added, split = [], 0

    # 1. the split, on every resident-weight row of either system
    for r in rows:
        rid = r["run_id"]
        if "_resident_" not in rid or not r.get("first_output_cycles"):
            continue
        n = int(r["array_size"].split("x")[0])
        if rid.startswith("spmw_"):
            # The port's weights are a `spmw.MemIn`: resident, no feed, so the
            # whole figure is fill. `fill(n)` is the *RTL's* LAT + N and says
            # nothing about the port -- applying it here was a bug that made
            # every SPMW row claim the RTL's latency.
            r["fill_cycles"] = r["first_output_cycles"]
            r["weight_feed_cycles"] = "0"
        else:
            r["fill_cycles"] = str(fill(n))
            r["weight_feed_cycles"] = str(int(r["first_output_cycles"]) - fill(n))
        split += 1

    # 2. the port's re-measured rows
    for (key, n), (first, completion) in BOUND.items():
        rid = f"spmw_{key}_N{n}_resident_general_fabmul"
        if rid in by:
            print(f"  {rid} already present, skipped")
            continue
        base = dict(by[f"spmw_{key}_N{n}_resident_general"])
        base.update(
            run_id=rid,
            variant=base["variant"] + ", integer multiply bound to fabric",
            first_output_cycles=str(first),
            completion_cycles=str(completion),
            fill_cycles=str(first),
            weight_feed_cycles="0",
            source="this agent (fabric-multiply binding)",
        )
        added.append(base)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows + added:
            writer.writerow(r)
    print(f"  split written on {split} resident rows; {len(added)} rows added")
    for r in added:
        print(f"  + {r['run_id']:44s} first {r['first_output_cycles']:>4} "
              f"completion {r['completion_cycles']:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
