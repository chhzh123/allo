#!/usr/bin/env python3
"""Every figure the README states, checked against `results.csv`.

A writeup drifts from its data the moment a number is corrected in one and
not the other, and this experiment corrected several -- the mesh rates twice,
Gemmini's clock once.  So the figures are asserted here rather than trusted.

It has already earned its place twice.  The collector was pairing a
fabric-bound run at one row count with a DSP-allowed run at the other and
reporting 56 cycles a row where both bindings measure 31; and the two SPMW
place-and-route rows were written to the CSV tagged `cycles` only, so
anything filtering on `kind == "pnr"` could not see them.
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: (engine, bind_mul, scale_latency, period, lut, ff, dsp, ns, MHz)
PNR = [
    ("gemmini", "", "4", "3.333", 71389, 21474, 500, 28.708, 34.8),
    ("spmw", "0", "", "3.333", 113341, 153869, 755, 3.261, 306.7),
    ("spmw", "1", "", "3.333", 233725, 170759, 83, 3.852, 259.6),
]

#: The device-utilisation table: every cell is `used`, `available` and the
#: percentage Vivado printed, so the capacities are checked too rather than
#: being a hardcoded idea of what a U280 holds.
#: resource -> {key: (used, available, pct)}
UTILISATION = {
    "lut":       {"gemmini": (71389, 1303680, "5.48"),
                  "spmw0": (113341, 1303680, "8.69"),
                  "spmw1": (233725, 1303680, "17.93")},
    "lut_logic": {"gemmini": (70744, 1303680, "5.43"),
                  "spmw0": (103993, 1303680, "7.98"),
                  "spmw1": (228638, 1303680, "17.54")},
    "lutmem":    {"gemmini": (645, 600960, "0.11"),
                  "spmw0": (9348, 600960, "1.56"),
                  "spmw1": (5087, 600960, "0.85")},
    "ff":        {"gemmini": (21474, 2607360, "0.82"),
                  "spmw0": (153869, 2607360, "5.90"),
                  "spmw1": (170759, 2607360, "6.55")},
    "carry8":    {"gemmini": (5116, 162960, "3.14"),
                  "spmw0": (6440, 162960, "3.95"),
                  "spmw1": (16350, 162960, "10.03")},
    "dsp":       {"gemmini": (500, 9024, "5.54"),
                  "spmw0": (755, 9024, "8.37"),
                  "spmw1": (83, 9024, "0.92")},
    "bram":      {"gemmini": (0, 2016, "0.00"),
                  "spmw0": (0, 2016, "0.00"),
                  "spmw1": (0, 2016, "0.00")},
    "uram":      {"gemmini": (0, 960, "0.00"),
                  "spmw0": (0, 960, "0.00"),
                  "spmw1": (0, 960, "0.00")},
    "iob":       {"gemmini": (0, 624, "0.00"),
                  "spmw0": (35, 624, "5.61"),
                  "spmw1": (35, 624, "5.61")},
}

#: (engine, activation, row length, marginal cycles per row)
MARGINAL = [
    ("spmw", "layernorm", "256", 31.0), ("spmw", "softmax", "64", 4.0),
    ("spmw", "igelu", "256", 16.0), ("spmw", "igelu", "1024", 64.0),
    ("spmw", "none", "256", 16.0), ("spmw", "none", "768", 48.0),
    ("gemmini", "layernorm", "256", 130.0), ("gemmini", "softmax", "64", 41.0),
    ("gemmini", "igelu", "256", 16.0), ("gemmini", "igelu", "1024", 64.0),
    ("gemmini", "none", "768", 48.0),
]


def main():
    rows = list(csv.DictReader(open(f"{ROOT}/results.csv")))
    readme = open(f"{ROOT}/README.md").read()
    bad = []
    n = 0

    pnr = {(r["engine"], r["bind_mul"], r["scale_latency"], r["period_ns"]): r
           for r in rows if r["kind"] == "pnr"}
    for eng, bind, lat, per, lut, ff, dsp, ns, mhz in PNR:
        r = pnr.get((eng, bind, lat, per))
        if r is None:
            bad.append(f"{eng}/{bind or '-'}: no such row in results.csv")
            continue
        for field, exp in (("lut", lut), ("ff", ff), ("dsp", dsp),
                           ("achieved_ns", ns), ("achieved_mhz", mhz)):
            n += 1
            if float(r[field]) != float(exp):
                bad.append(f"{eng}/{bind or '-'} {field}: csv {r[field]} "
                           f"!= expected {exp}")
            elif str(exp) not in readme and f"{exp:,}" not in readme:
                bad.append(f"{eng}/{bind or '-'} {field}: {exp} is not in "
                           f"the README")

    key = {"gemmini": ("gemmini", "", "4", "3.333"),
           "spmw0": ("spmw", "0", "", "3.333"),
           "spmw1": ("spmw", "1", "", "3.333")}
    for res, per in UTILISATION.items():
        for who, (used, avail, pct) in per.items():
            r = pnr.get(key[who])
            if r is None:
                continue
            for field, exp in ((res, used), (res + "_avail", avail),
                               (res + "_pct", pct)):
                n += 1
                got = r.get(field, "")
                if str(got) != str(exp):
                    bad.append(f"{who} {field}: csv {got!r} != expected {exp!r}")
            # the percentage has to be in the README, the raw count only when
            # it is not zero (the zero rows are written as "0%" there)
            if pct not in readme and used:
                bad.append(f"{who} {res}: {pct}% is not in the README")

    # The marginal rates the README tabulates are the shipped configuration:
    # scalar links 16 deep, multiplies in fabric.
    marg = {(r["engine"], r["activation"], r["len"]): r for r in rows
            if r["kind"] == "marginal"
            and (r["engine"] == "gemmini"
                 or (r["depth"] == "16" and r["bind_mul"] == "1"))}
    for eng, act, ln, exp in MARGINAL:
        n += 1
        r = marg.get((eng, act, ln))
        if r is None:
            bad.append(f"{eng} {act} len={ln}: no marginal rate in results.csv")
        elif float(r["per_row"]) != exp:
            bad.append(f"{eng} {act} len={ln}: csv {r['per_row']} != {exp}")

    for b in bad:
        print(f"  BAD  {b}")
    print(f"{n - len(bad)}/{n} README figures agree with results.csv")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
