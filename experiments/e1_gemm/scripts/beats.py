#!/usr/bin/env python3
"""First and last memory beat from a cosim VCD, the definition SPMW's testbench
reports: the first beat any master moves to the last, control traffic excluded.

A beat transfers *at* the rising clock edge, on the VALID/READY values that held
before it -- which is exactly what `always @(posedge ap_clk) if (VALID && READY)`
samples in the SPMW testbench, so both sides count the same event. Sampling
mid-cycle instead reads the values the edge itself produced and silently loses
the last beat of every burst: invisible on an eight-beat burst, total on a
one-beat one. So this walks timestamp by timestamp and, when the clock rises,
counts on the state accumulated *before* that timestamp's changes.

The per-port census is the check on that: each design's beat count must equal
its own arithmetic (at 8x8, 8 + 8 + 64 beats narrow and 1 + 1 + 4 wide).

    beats.py <vcd> [name:valid:ready ...]

with no port list meaning AutoSA's three masters. Allo names its ports
differently and is read as:

    beats.py <vcd> A:m_axi_gmem0_RVALID:m_axi_gmem0_RREADY \
                   B:m_axi_gmem1_RVALID:m_axi_gmem1_RREADY \
                   C:m_axi_gmem2_WVALID:m_axi_gmem2_WREADY
"""
import collections
import re
import sys

PORTS = [tuple(a.split(":", 2)) for a in sys.argv[2:]] or [
    ("A", "m_axi_gmem_A_RVALID", "m_axi_gmem_A_RREADY"),
    ("B", "m_axi_gmem_B_RVALID", "m_axi_gmem_B_RREADY"),
    ("C", "m_axi_gmem_C_WVALID", "m_axi_gmem_C_WREADY")]
NEED = ["ap_clk"] + [s for _, a, b in PORTS for s in (a, b)]

ids = {}
with open(sys.argv[1], errors="replace") as handle:
    for line in handle:
        match = re.match(r"\$var\s+\w+\s+\d+\s+(\S+)\s+([A-Za-z0-9_]+)", line)
        if match and match.group(2) in NEED:
            ids[match.group(1)] = match.group(2)
        if line.startswith("$enddefinitions"):
            break
    missing = set(NEED) - set(ids.values())
    if missing:
        sys.exit("VCD lacks: " + ", ".join(sorted(missing)))

    val = collections.defaultdict(lambda: "x")
    hits = collections.defaultdict(list)
    pending, cycle, clk_before = [], 0, "x"

    def settle():
        """Apply one timestamp's changes, counting a beat if the clock rose."""
        global cycle, clk_before
        rose = clk_before != "1" and any(n == "ap_clk" and v == "1" for v, n in pending)
        if rose:
            cycle += 1
            for name, valid, ready in PORTS:
                if val[valid] == "1" and val[ready] == "1":
                    hits[name].append(cycle)
        for value, name in pending:
            val[name] = value
            if name == "ap_clk":
                clk_before = value
        del pending[:]

    for line in handle:
        line = line.strip()
        if not line:
            continue
        if line[0] == "#":
            settle()
        elif line[0] in "01xzXZ" and len(line) > 1 and line[1:] in ids:
            pending.append((line[0], ids[line[1:]]))
    settle()

for name, _, _ in PORTS:
    seen = hits[name]
    print("  %s: %4d beats  first %-6s last %-6s"
          % (name, len(seen), seen[0] if seen else "-", seen[-1] if seen else "-"))
every = sorted(c for seen in hits.values() for c in seen)
if not every:
    sys.exit("no memory beat seen")
print("BEATS first %d last %d cycles %d" % (every[0], every[-1], every[-1] - every[0] + 1))
