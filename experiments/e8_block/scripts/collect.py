#!/usr/bin/env python3
"""Read every log and report E8 produced into one table.

Nothing is computed here that a run did not measure, except the marginal rate
-- the difference between two row counts -- and that is arithmetic on two
measurements, with both shown.
"""

import argparse
import csv
import glob
import os
import re
import sys

ROOT = "/scratch/hc676/e8_block"


def read_util(path):
    """LUT / FF / DSP / BRAM out of a Vivado utilization report."""
    out = {}
    if not os.path.exists(path):
        return out
    # Each row is `| Site Type | Used | Fixed | Prohibited | Available | Util% |`,
    # so the device capacity and the percentage come out of the same report as
    # the count -- no capacity table to keep in step with the part.
    pat = {
        "lut": r"CLB LUTs",
        "lut_logic": r"LUT as Logic",
        "lutmem": r"LUT as Memory",
        "ff": r"CLB Registers",
        "carry8": r"CARRY8",
        "dsp": r"DSPs",
        "bram": r"Block RAM Tile",
        "uram": r"URAM",
        "iob": r"Bonded IOB",
    }
    text = open(path, errors="replace").read()
    for k, name in pat.items():
        m = re.search(r"\|\s*" + name + r"\s*\|\s*(\d+)\s*\|\s*\d+\s*\|"
                      r"\s*\d+\s*\|\s*(\d+)\s*\|\s*([\d.<]+)", text)
        if not m:   # the DSP and BRAM tables have no Prohibited column
            m = re.search(r"\|\s*" + name + r"\s*\|\s*(\d+)\s*\|\s*\d+\s*\|"
                          r"\s*\d+\s*\|\s*(\d+)\s*\|\s*([\d.<]+)", text)
        if m:
            out[k] = int(m.group(1))
            out[k + "_avail"] = int(m.group(2))
            out[k + "_pct"] = m.group(3)
    return out


def read_wns(path):
    if not os.path.exists(path):
        return {}
    text = open(path, errors="replace").read()
    m = re.search(r"^\s*WNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)\s", text,
                  re.M | re.S)
    return {"wns_ns": float(m.group(1))} if m else {}


def gemmini_pnr():
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/gem_pnr_*")):
        tag = os.path.basename(d)[len("gem_pnr_"):]
        if not os.path.exists(f"{d}/DONE"):
            continue
        period = float(tag.split("_")[0].replace("p", "."))
        lat = 8 if tag.endswith("_l8") else 4
        r = dict(engine="gemmini", design="MxuVpuNorm", dim=16,
                 scale_latency=lat, period_ns=period)
        r.update(read_util(f"{d}/util.rpt"))
        r.update(read_wns(f"{d}/timing.rpt"))
        if "wns_ns" in r:
            r["achieved_ns"] = round(period - r["wns_ns"], 3)
            r["achieved_mhz"] = round(1000 / r["achieved_ns"], 1)
        rows.append(r)
    return rows


def gemmini_cycles():
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/gb_*")) + sorted(glob.glob(f"{ROOT}/gembench_*")):
        log = None
        for cand in (f"{d}/xsim.log", f"{d}/xsim.jou"):
            if os.path.exists(cand):
                log = cand
        text = ""
        for f in glob.glob(f"{d}/*.log"):
            text += open(f, errors="replace").read()
        m = re.search(r"GEM RESULT mode=(\w+) outs=(\d+)/(\d+) errs=(\d+) "
                      r"cycles=(\d+)", text)
        if not m:
            continue
        act, got, want, errs, cyc = m.groups()
        parts = os.path.basename(d).split("_")
        nrow = int(parts[-2]) if parts[-2].isdigit() else None
        ln = int(parts[-1]) if parts[-1].isdigit() else None
        rows.append(dict(engine="gemmini", activation=act, nrow=nrow, len=ln,
                         outputs=int(got), expected=int(want),
                         errors=int(errs), cycles=int(cyc),
                         passed=(got == want and errs == "0")))
    return rows


def spmw_runs():
    rows = []
    for log in sorted(glob.glob(f"{ROOT}/logs/sw_block*.log")):
        if log.endswith(".done"):
            continue
        text = open(log, errors="replace").read()
        d = re.search(r"E8 BLOCK design: mode=(\d+) nrow=(\d+) len=(\d+) "
                      r"nbeat=(\d+) nacc=(\d+)(?: tiles=(\d+) steps=(\d+))?",
                      text)
        c = re.search(r"SPMW CYCLES total=(\d+) first_out=(\d+) first_in=(\d+)",
                      text)
        p = re.search(r"SPMW COSIM (PASS|FAIL).*?\((\d+)/(\d+) tokens, "
                      r"(\d+) errors\)", text)
        # The width and the multiply binding come from the tag: the build's
        # own output does not echo its command line.
        tag = os.path.basename(log)[len("sw_"):-len(".log")]
        size = re.search(r"_(\d+)_r\d+_l\d+", tag)
        bind = 0 if "_m0" in tag else 1
        if not (d and c):
            continue
        name = {0: "none", 1: "relu", 2: "layernorm", 3: "igelu",
                4: "softmax"}[int(d.group(1))]
        r = dict(engine="spmw", design="block", bind_mul=bind,
                 activation=name, dim=int(size.group(1)) if size else None,
                 nrow=int(d.group(2)), len=int(d.group(3)),
                 nbeat=int(d.group(4)), nacc=int(d.group(5)),
                 tiles=int(d.group(6)) if d.group(6) else 4,
                 depth=int(re.search(r"depth=(\d+)", text).group(1))
                 if re.search(r"depth=(\d+)", text) else 2,
                 cycles=int(c.group(1)), first_out=int(c.group(2)),
                 passed=bool(p and p.group(1) == "PASS"),
                 errors=int(p.group(4)) if p else None,
                 log=os.path.basename(log))
        out = log.replace("/logs/", "/").replace(".log", "")
        for rpt in ("pnr/util.rpt", "util_pnr.rpt", "util.rpt"):
            u = read_util(f"{out}/{rpt}")
            if u:
                r.update(u)
                r.update(read_wns(f"{out}/{rpt.replace('util', 'timing')}"))
                # The array build constrains 300 MHz unless told otherwise.
                r["period_ns"] = 3.333
                if "wns_ns" in r:
                    r["achieved_ns"] = round(3.333 - r["wns_ns"], 3)
                    r["achieved_mhz"] = round(1000 / r["achieved_ns"], 1)
                r["util_from"] = rpt
                break
        rows.append(r)
    return rows


def marginal(rows,
             key=("engine", "activation", "dim", "len", "depth", "bind_mul")):
    """`(fixed, per_row)` from two row counts of the same configuration."""
    by = {}
    for r in rows:
        if r.get("nrow") is None or r.get("cycles") is None:
            continue
        if r.get("tiles", 4) != 4:
            continue   # the mesh-rate runs vary tiles, not rows
        by.setdefault(tuple(r.get(k) for k in key), []).append(r)
    out = []
    for k, rs in sorted(by.items(), key=lambda kv: [str(x) for x in kv[0]]):
        rs = sorted(rs, key=lambda r: r["nrow"])
        if len(rs) < 2:
            continue
        lo, hi = rs[0], rs[-1]
        if hi["nrow"] == lo["nrow"]:
            continue
        per_row = (hi["cycles"] - lo["cycles"]) / (hi["nrow"] - lo["nrow"])
        out.append(dict(zip(key, k), per_row=round(per_row, 2),
                        fixed=round(lo["cycles"] - per_row * lo["nrow"], 1),
                        lo_nrow=lo["nrow"], lo_cycles=lo["cycles"],
                        hi_nrow=hi["nrow"], hi_cycles=hi["cycles"],
                        beats=lo.get("nbeat") or (lo["len"] // 16),
                        per_beat=round(per_row / ((lo.get("nbeat") or
                                                   lo["len"] // 16)), 3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    gp, gc, sr = gemmini_pnr(), gemmini_cycles(), spmw_runs()

    print("== place and route ==")
    hdr = ("engine", "design", "dim", "bind_mul", "scale_latency",
           "period_ns", "lut", "ff", "dsp", "bram", "wns_ns", "achieved_ns",
           "achieved_mhz")
    print(" ".join(f"{h:>13s}" for h in hdr))
    for r in gp:
        print(" ".join(f"{str(r.get(h, '')):>13s}" for h in hdr))
    for r in sr:
        if "lut" in r:
            print(" ".join(f"{str(r.get(h, '')):>13s}" for h in hdr))

    print()
    print("== cycles, as measured ==")
    hdr = ("engine", "activation", "dim", "nrow", "len", "nbeat", "tiles",
           "depth", "cycles", "passed")
    print(" ".join(f"{h:>11s}" for h in hdr))
    for r in sorted(gc + sr, key=lambda r: (r["engine"], r["activation"],
                                            r.get("dim") or 0,
                                            r.get("len") or 0,
                                            r.get("nrow") or 0)):
        print(" ".join(f"{str(r.get(h, '')):>11s}" for h in hdr))

    print()
    print("== marginal cost of a normalisation row ==")
    m = marginal(gc + sr)
    hdr = ("engine", "activation", "dim", "len", "depth", "bind_mul",
           "beats", "per_row", "per_beat", "lo_cycles", "hi_cycles")
    print(" ".join(f"{h:>11s}" for h in hdr))
    for r in m:
        print(" ".join(f"{str(r.get(h, '')):>11s}" for h in hdr))

    if a.csv:
        # An SPMW run that was routed is both a cycle measurement and a
        # place-and-route one, and it is written as both: tagging it only
        # `cycles` hid the two SPMW area rows from anything filtering on
        # `kind == "pnr"`.
        rows = ([dict(r, kind="pnr") for r in gp] +
                [dict(r, kind="cycles") for r in gc + sr] +
                [dict(r, kind="pnr") for r in sr if "lut" in r] +
                [dict(r, kind="marginal") for r in m])
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
