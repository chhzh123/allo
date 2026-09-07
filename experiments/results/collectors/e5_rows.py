#!/usr/bin/env python3
"""e5_compile/results.csv from the compilation runs' own `E5 ...` lines.

    python3 e5_rows.py /scratch/hc676/e5/e5_runs.log <bundle>/e5_compile

Each line is one (design, size, mode, repetition) written by
scripts/spmw_ablate_compile.py at the end of that run; a run that timed out
or failed writes a line saying so. Medians and ranges over the repetitions
are computed here, nothing else.
"""
import csv, json, os, re, statistics, sys

log, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rows = []
for line in open(log, encoding="utf-8", errors="replace"):
    if not line.startswith("E5 "):
        continue
    row = {}
    for token in line[3:].split():
        if "=" in token:
            k, v = token.split("=", 1)
            row[k] = v
    # A run that timed out or failed writes `run=<design>_<size>_<mode>`
    # instead of the three fields, because it never reached the report line.
    if "design" not in row and "run" in row:
        parts = re.sub(r"_r\d+$", "", row["run"]).rsplit("_", 2)
        if len(parts) == 3:
            row["design"], row["size"], row["mode"] = parts
    if row.get("design"):
        rows.append(row)

def num(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return None

table = []
groups = {}
for r in rows:
    key = (r.get("design"), r.get("size"), r.get("mode"))
    groups.setdefault(key, []).append(r)
for (design, size, mode), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], int(kv[0][1] or 0), kv[0][2])):
    walls = [w for w in (num(r, "hls_wall_s") for r in rs) if w is not None]
    jobs = [w for w in (num(r, "hls_sum_job_elapsed_s") for r in rs) if w is not None]
    cpus = [(num(r, "cpu_user_s") or 0) + (num(r, "cpu_sys_s") or 0) for r in rs if num(r, "cpu_user_s") is not None]
    front = [w for w in (num(r, "frontend_s") for r in rs) if w is not None]
    bad = [r for r in rs if r.get("status")]
    table.append({
        "run_id": f"{design}_{size}_{mode}", "experiment_id": "E5",
        "system": "SPMW split backend", "design": design, "size": size, "mode": mode,
        "workers": rs[0].get("jobs", ""), "roles": rs[0].get("roles", ""),
        "instances": rs[0].get("instances", ""), "projects": rs[0].get("projects", ""),
        "repetitions": len(rs), "completed_jobs": rs[0].get("completed_jobs", ""),
        "status": "pass" if (walls and not bad) else ("timeout" if bad else "environment_blocked"),
        "timed_out_repetitions": len(bad),
        "completed_jobs_when_timed_out": ", ".join(
            r.get("completed_jobs", "") for r in bad
        ),
        "timeout_s": bad[0].get("timeout", "") if bad else "",
        "frontend_s_median": round(statistics.median(front), 2) if front else "",
        "hls_wall_s_median": round(statistics.median(walls), 1) if walls else "",
        "hls_wall_s_min": round(min(walls), 1) if walls else "",
        "hls_wall_s_max": round(max(walls), 1) if walls else "",
        "hls_sum_job_elapsed_s_median": round(statistics.median(jobs), 1) if jobs else "",
        "cpu_s_median": round(statistics.median(cpus), 1) if cpus else "",
        "load_start": rs[0].get("load_start", ""),
        "report_paths": f"reports/{design}_{size}_{mode}_r0",
        "failure_reason": "; ".join(sorted({r.get("status", "") for r in bad})) if bad else "",
    })
if table:
    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as h:
        json.dump({"runs": rows, "summary": table}, h, indent=1)
print(f"{len(rows)} runs -> {len(table)} points")
for t in table:
    print(f"  {t['run_id']:28s} projects={t['projects']:>5s} wall={t['hls_wall_s_median']!s:>7s}s "
          f"(min {t['hls_wall_s_min']!s} max {t['hls_wall_s_max']!s}) jobsum={t['hls_sum_job_elapsed_s_median']!s:>8s} "
          f"cpu={t['cpu_s_median']!s:>8s} reps={t['repetitions']} {t['status']}")
