# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulate tiles through the original FEATHER RTL, for the SPMW comparison.

FEATHER (Tong et al., ISCA 2024) ships its RTL for synthesis, without a
testbench: https://github.com/maeri-project/FEATHER, FEATHER_RTL/RTL.
`tests/dataflow/spmw/tb_feather_rtl.sv` drives `feather_top` as the shipped
controller behaves (its header says how) through one N x N tile, then
through NP tiles back to back with the weights resident, and prints the
cycle at every phase boundary; this script compiles and runs it with xsim
for each array size and tabulates the numbers.

    source <Vitis>/settings64.sh
    python3 scripts/spmw_feather_rtl.py --feather /path/to/FEATHER --out /tmp/feather_rtl --sizes 4,8,16,32
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TB = os.path.join(HERE, "..", "tests", "dataflow", "spmw", "tb_feather_rtl.sv")

FIELDS = {
    "feed": r"weight feed (\d+) cycles driven \(N\^3=(\d+) by design\)",
    "store": r"last PE store (\d+) cycles after feed start",
    "pass": r"activation pass (\d+) cycles",
    "first": r"first tile row at \+(\d+)",
    "total": r"feed start to last row of tile 1 (\d+)",
    "per_tile": r"(\d+) cycles a tile",
    "wrong_pe": r"weight files (\d+) PE\(s\) wrong",
    "wrong_col": r"tile columns (\d+) wrong",
    "verdict": r"FEATHER RTL N=\d+ NP=\d+: (PASS|FAIL|TIMEOUT)",
}


def run(cmd, cwd, log):
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        sys.exit(f"{cmd[0]} failed (rc={rc}), see {log}")


def simulate(size, passes, rtl, out, tb):
    work = os.path.join(out, f"run{size}x{passes}")
    os.makedirs(work, exist_ok=True)
    sources = sorted(os.path.join(rtl, f) for f in os.listdir(rtl) if f.endswith(".v"))
    run(
        ["xvlog", "-sv", "--relax", *sources, tb], work, os.path.join(work, "xvlog.log")
    )
    run(
        [
            "xelab",
            "tb",
            "-s",
            "sim",
            "--relax",
            "--timescale",
            "1ns/1ps",
            "-generic_top",
            f"N={size}",
            "-generic_top",
            f"NP={passes}",
        ],
        work,
        os.path.join(work, "xelab.log"),
    )
    run(["xsim", "sim", "-R"], work, os.path.join(work, "xsim.log"))
    with open(os.path.join(work, "xsim.log"), encoding="utf-8") as f:
        text = "".join(line for line in f if "FEATHER RTL" in line)
    got = {}
    for key, pat in FIELDS.items():
        m = re.search(pat, text)
        got[key] = m.group(1) if m else "?"
    return got, text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--feather", required=True, help="a checkout of maeri-project/FEATHER"
    )
    parser.add_argument("--out", required=True, help="work directory")
    parser.add_argument(
        "--sizes", default="4,8,16,32", help="array sizes N (AW = AH = N)"
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="tiles back to back for the throughput run (1 skips it)",
    )
    parser.add_argument("--tb", default=TB)
    args = parser.parse_args()
    rtl = os.path.join(args.feather, "FEATHER_RTL", "RTL")
    if not os.path.isdir(rtl):
        sys.exit(f"no RTL at {rtl}")
    os.makedirs(args.out, exist_ok=True)
    tb = os.path.abspath(args.tb)
    rows = []
    for size in (int(s) for s in args.sizes.split(",")):
        got, text = simulate(size, 1, rtl, args.out, tb)
        print(text, end="")
        if args.passes > 1:
            more, text = simulate(size, args.passes, rtl, args.out, tb)
            print(text, end="")
            got["per_tile"] = more["per_tile"]
            got["verdict"] += "/" + more["verdict"]
        rows.append((size, got))
    print()
    print("FEATHER RTL, one N x N tile, cycles (xsim of the shipped RTL):")
    print(
        "  N   weight feed (driven / design)   last store   activation pass   first tile row"
        "   feed start -> last output   cycles a tile, weights resident   check"
    )
    for size, g in rows:
        print(
            f"  {size:<3} {g['feed']:>10} / {size ** 3:<12} {g['store']:>10} {g['pass']:>15} {g['first']:>16}"
            f" {g['total']:>22} {g['per_tile']:>22}   "
            f"{g['verdict']} ({g['wrong_pe']} PE files, {g['wrong_col']} columns wrong)"
        )


if __name__ == "__main__":
    main()
