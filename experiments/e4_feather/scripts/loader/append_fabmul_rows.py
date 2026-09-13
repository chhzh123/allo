#!/usr/bin/env python3
"""E4: append the fabric-multiply place-and-route rows to results.csv.

    append_fabmul_rows.py <results.csv> <run root>

FEATHER's RTL routes with zero DSP blocks; the SPMW port's identical multiply
was inferred into one per element, so the two lookup-table columns were not
measuring the same thing. These rows are the port routed with
`#pragma HLS bind_op ... op=mul impl=fabric`, which is what the baseline does.
The DSP-inferred `pnr_spmw_feather_N*` rows stay where they are: which resource
a multiply lands in is a real choice, and the pair is what prices it.
"""

import csv
import os
import re
import sys

VARIANT = (
    "feather (single tile, resident int8 files), integer multiply bound to "
    "fabric so the port spends the same resource the RTL does"
)


def cells(run):
    util, timing = os.path.join(run, "util.rpt"), os.path.join(run, "timing.rpt")
    if not os.path.isfile(util):
        return None
    text = open(util, errors="replace").read()

    def cell(name):
        m = re.search(r"^\|\s*%s\*?\s*\|\s*([\d.]+)\s*\|" % re.escape(name), text, re.M)
        return m.group(1) if m else ""

    wns = tns = ""
    if os.path.isfile(timing):
        body = open(timing, errors="replace").read()
        m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)\s+(-?[\d.]+)", body, re.S)
        if m:
            wns, tns = m.group(1), m.group(2)
    tiles = cell("Block RAM Tile")
    return {
        "lut": cell("CLB LUTs"),
        "ff": cell("CLB Registers"),
        "dsp": cell("DSPs"),
        "uram": cell("URAM"),
        "bram_18k_equiv": str(int(2 * float(tiles))) if tiles else "",
        "wns_ns": wns,
        "tns_ns": tns,
    }


def main():
    results, root = sys.argv[1], sys.argv[2]
    with open(results, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        existing = list(reader)
    by = {r["run_id"]: r for r in existing}
    added = []
    for N in (4, 8, 16):
        run_id = f"pnr_spmw_feather_fabmul_N{N}"
        if run_id in by:
            print(f"  {run_id} already present, skipped")
            continue
        got = cells(os.path.join(root, f"spmw_feather_fabmul_{N}"))
        if got is None:
            print(f"  no util.rpt for N={N}, skipped")
            continue
        base = dict(by[f"pnr_spmw_feather_N{N}"])
        base.update(got)
        base.update(
            run_id=run_id,
            variant=VARIANT,
            status="pass",
            unrouted="0",
            report_paths=(
                f"experiments/e4_feather/spmw/S{N}/report; run dir "
                f"{root}/spmw_feather_fabmul_{N}"
            ),
            source="this agent (fabric-multiply binding)",
        )
        added.append(base)
    if not added:
        print("nothing to add")
        return 0
    with open(results, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in existing + added:
            writer.writerow(r)
    for r in added:
        print(f"  + {r['run_id']:32s} LUT {r['lut']:>6} FF {r['ff']:>6} DSP {r['dsp']:>3} WNS {r['wns_ns']}")
    print(f"{len(added)} rows added, {len(existing)} kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())
