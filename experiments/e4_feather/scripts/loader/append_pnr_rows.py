#!/usr/bin/env python3
"""E4: append the row-wise-loader OOC place-and-route rows to results.csv."""
import csv, re, sys

def util(path):
    v = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        if not line.startswith("|"): continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 2: continue
        try: used = float(c[1])
        except ValueError: continue
        if c[0].startswith("CLB LUTs") and "lut" not in v: v["lut"] = used
        elif c[0] == "CLB Registers" and "ff" not in v: v["ff"] = used
        elif c[0] == "DSPs" and "dsp" not in v: v["dsp"] = used
        elif c[0].startswith("RAMB36") and "b36" not in v: v["b36"] = used
        elif c[0].startswith("RAMB18") and "b18" not in v: v["b18"] = used
        elif c[0] == "URAM" and "uram" not in v: v["uram"] = used
    return v

def timing(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    i = txt.index("WNS(ns)")
    nums = txt[i:].split("\n")[2].split()
    return float(nums[0]), float(nums[1])

def unrouted(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"Nets with Routing Errors\s*:\s*(\d+)", txt)
    return int(m.group(1)) if m else 0

WALL = {4: 323, 8: 458, 16: 1387}
STAGES = {
    4: "{'synth': '92.936', 'opt': '5.484', 'place': '166.524', 'physopt': '0.531', 'route': '29.469'}",
    8: "{'synth': '161.392', 'opt': '7.946', 'place': '194.591', 'physopt': '2.19', 'route': '59.266'}",
    16: "{'synth': '508.021', 'opt': '30.01', 'place': '488.657', 'physopt': '13.776', 'route': '284.859'}",
}

def main():
    results, root = sys.argv[1], sys.argv[2]
    with open(results, encoding="utf-8") as f:
        rd = csv.DictReader(f); fields = rd.fieldnames; existing = list(rd)
    have = {r["run_id"] for r in existing}
    rows = []
    for N in (4, 8, 16):
        rep = f"{root}/S{N}/report"
        u = util(f"{rep}/pnr_rowload_util.rpt")
        wns, tns = timing(f"{rep}/pnr_rowload_timing.rpt")
        unr = unrouted(f"{rep}/pnr_rowload_route.rpt")
        rid = f"pnr_rtl_rowload_N{N}"
        if rid in have: continue
        rows.append({
            "run_id": rid, "experiment_id": "E4", "system": "FEATHER RTL",
            "variant": "corrected controller + row-wise weight loader (a row of PEs a cycle: the load is N^2, not N^3)",
            "workload": "feather_top out of context: N x N NEST + BIRRD + controller + SRAMs (shipped defaults: depth-4 SRAM register arrays)",
            "array_size": f"{N}x{N}", "weight_mode": "n/a", "reorder_program": "n/a",
            "implementation_mode": "pnr_ooc", "target_mhz": "300",
            "status": "pass" if wns >= 0 and unr == 0 else "fail",
            "validation_pass": "timing met" if wns >= 0 else "timing not met",
            "lut": int(u["lut"]), "ff": int(u["ff"]), "dsp": int(u.get("dsp", 0)),
            "bram_18k_equiv": int(2 * u.get("b36", 0) + u.get("b18", 0)), "uram": int(u.get("uram", 0)),
            "wns_ns": wns, "tns_ns": tns, "unrouted": unr, "total_wall_s": WALL[N],
            "report_paths": f"experiments/e4_feather/feather_rtl/S{N}/report/pnr_rowload_*; run dir /scratch/hc676/feather_loader/pnr/rowload_N{N}; stages (s): {STAGES[N]}",
            "failure_reason": "", "source": "this agent (feather_loader dirs)",
        })
    assert rows, "nothing to add"
    with open(results, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        for r in rows: w.writerow({k: r.get(k, "") for k in fields})
    for r in rows:
        print(f"+ {r['run_id']:22s} {r['status']} lut={r['lut']} ff={r['ff']} dsp={r['dsp']} wns={r['wns_ns']} tns={r['tns_ns']} unrouted={r['unrouted']}")

if __name__ == "__main__":
    main()
