#!/usr/bin/env python3
"""First and last memory beat from a cosim VCD, the definition SPMW's testbench
reports: the first beat any master moves to the last, control traffic excluded.

    beats.py <vcd>
"""
import sys

NEED = ["ap_clk", "m_axi_gmem_A_RVALID", "m_axi_gmem_A_RREADY",
        "m_axi_gmem_B_RVALID", "m_axi_gmem_B_RREADY",
        "m_axi_gmem_C_WVALID", "m_axi_gmem_C_WREADY"]
sym, val = {}, {}
with open(sys.argv[1], errors="replace") as h:
    for line in h:
        line = line.strip()
        if line.startswith("$var"):
            parts = line.split()
            code, name = parts[3], parts[4]
            base = name.split("[")[0]
            if base in NEED:
                sym.setdefault(base, code)
        elif line.startswith("$enddefinitions"):
            break
    missing = [n for n in NEED if n not in sym]
    if missing:
        sys.exit("VCD lacks: " + ", ".join(missing))
    code_of = {v: k for k, v in sym.items()}
    cycle, prev_clk = 0, "0"
    first = last = None
    for line in h:
        line = line.strip()
        if not line or line[0] == "#":
            continue
        if line[0] in "01xzXZ" and len(line) > 1:
            bit, code = line[0], line[1:]
            name = code_of.get(code)
            if name is None:
                continue
            val[name] = bit
            if name == "ap_clk":
                # count falling edges, as the earlier analysis did
                if prev_clk == "1" and bit == "0":
                    cycle += 1
                    moved = ((val.get("m_axi_gmem_A_RVALID") == "1" and val.get("m_axi_gmem_A_RREADY") == "1")
                             or (val.get("m_axi_gmem_B_RVALID") == "1" and val.get("m_axi_gmem_B_RREADY") == "1")
                             or (val.get("m_axi_gmem_C_WVALID") == "1" and val.get("m_axi_gmem_C_WREADY") == "1"))
                    if moved:
                        if first is None:
                            first = cycle
                        last = cycle
                prev_clk = bit
if first is None:
    sys.exit("no memory beat seen")
print("BEATS first %d last %d cycles %d" % (first, last, last - first + 1))
