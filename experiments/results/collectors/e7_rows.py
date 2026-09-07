#!/usr/bin/env python3
"""e7_loc/results.csv from the recount's counts.json `summary` (nothing recomputed)."""
import csv, glob, json, os, sys

root = sys.argv[1]
rows, seen = [], set()
for path in sorted(glob.glob(os.path.join(root, "**", "counts.json"), recursive=True)):
    with open(path, encoding="utf-8") as h:
        doc = json.load(h)
    where = os.path.relpath(os.path.dirname(path), root)
    for d in doc.get("summary", []):
        key = (d.get("id"), where)
        if key in seen:
            continue
        seen.add(key)
        hls, spmw = d.get("hls_design"), d.get("spmw_design")
        rows.append({
            "run_id": f"{d.get('id','')}" + ("" if where == "." else f"@{where}"),
            "experiment_id": "E7", "system": "HLS vs SPMW", "workload": d.get("title", ""),
            "kind": d.get("kind", "design"), "in_paper_table": d.get("table", False),
            "implementation_mode": "source_count",
            "status": "pass" if (hls and spmw) else ("unsupported" if spmw else "environment_blocked"),
            "hls_design_lines": hls if hls is not None else "",
            "spmw_design_lines": spmw if spmw is not None else "",
            "ratio_design": d.get("ratio_design", ""),
            "hls_config_lines": d.get("hls_config", ""), "spmw_config_lines": d.get("spmw_config", ""),
            "hls_test_lines": d.get("hls_test", ""), "spmw_test_lines": d.get("spmw_test", ""),
            "report_paths": os.path.join(where, "counts.md"),
            "failure_reason": "" if (hls and spmw) else "no matched counterpart on the HLS side",
        })
with open(os.path.join(root, "results.csv"), "w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(os.path.join(root, "results.json"), "w", encoding="utf-8") as h:
    json.dump(rows, h, indent=1)
print(f"{len(rows)} rows -> {root}/results.csv")
for r in rows:
    if r["in_paper_table"]:
        print(f"  {r['run_id']:22s} {str(r['hls_design_lines']):>5s} / {str(r['spmw_design_lines']):<5s} ratio={r['ratio_design']}")
