# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""VTA's routed engine split into its control and its datapath + scratchpads.

Usage: ``vta_core_split.py 4 w4/util_hier.rpt 8 w8/util_hier.rpt ...``

Control is what SPMW's and Gemmini's measured scopes have no counterpart of:
instruction fetch, the three instruction queues, the semaphores and the event
counters. Everything else -- `TensorGemm`, `TensorAlu` and every scratchpad
with its load and store logic -- is the rest. The datapath's own split is not
trusted: synthesis retimes across the hierarchy, so inside the engine
`TensorGemm` reports 7,563 lookup tables where routed alone it is 21,598.
"""
import re
import sys

CONTROL = [
    "Core/(Core)",
    "Core/fetch",
    "Core/ecounters",
    "Core/compute/inst_q",
    "Core/compute/s_0",
    "Core/compute/s_1",
    "Core/load/inst_q",
    "Core/load/s",
    "Core/store/inst_q",
    "Core/store/s",
]
ROW = re.compile(r"^\|(\s*)(\S+)\s+\|\s*(\S+)\s*\|" + r"\s*(\d+)\s*\|" * 9)


def split(path):
    rows, stack = {}, []
    for line in open(path):
        m = ROW.match(line)
        if not m:
            continue
        depth = (len(m.group(1)) - 1) // 2
        stack = stack[:depth] + [m.group(2)]
        v = [int(m.group(i)) for i in range(4, 13)]
        rows["/".join(stack)] = dict(lut=v[0], ff=v[4], b36=v[5], b18=v[6], uram=v[7])
    return rows


for width, path in zip(sys.argv[1::2], sys.argv[2::2]):
    rows = split(path)
    top = rows["Core"]
    ctrl = {k: sum(rows[c][k] for c in CONTROL if c in rows) for k in top}
    bram = lambda r: r["b36"] + r["b18"] / 2
    rest = {k: top[k] - ctrl[k] for k in top}
    print(
        f"w{width}: engine {top['lut']} LUT {top['ff']} FF {bram(top):g} BRAM {top['uram']} URAM"
        f" | control {ctrl['lut']} LUT {ctrl['ff']} FF {bram(ctrl):g} BRAM {ctrl['uram']} URAM"
        f" | datapath+scratchpads {rest['lut']} LUT {rest['ff']} FF {bram(rest):g} BRAM {rest['uram']} URAM"
    )
