"""Fold the twelve-product measurement into the trial rows."""
import csv, json, os, re

S = "/scratch/hc676/agentstudy"
steady = {}
name = None
for line in open(S + "/steady_all.log"):
    line = line.strip()
    if line.startswith("====="):
        name = line.split()[1]; steady[name] = {"iv": [], "lat": None, "res": None}
    elif name:
        m = re.search(r"STUDY INTERVAL \d+ (\d+)", line)
        if m: steady[name]["iv"].append(int(m.group(1)))
        m = re.search(r"STUDY PRODUCT 0 .*latency (\d+)", line)
        if m: steady[name]["lat"] = int(m.group(1))
        m = re.search(r"STUDY RESULT (\S+)", line)
        if m: steady[name]["res"] = m.group(1)
        if "STUDY BUILD" in line: steady[name]["res"] = "BUILD_FAIL"

rows = list(csv.DictReader(open(S + "/results/results.csv")))
for r in rows:
    s = steady.get(r["trial"])
    iv = s["iv"] if s else []
    sustained = max(iv[-4:]) if len(iv) >= 4 else (max(iv) if iv else None)
    r["steady_result"] = (s or {}).get("res") or ""
    r["steady_latency"] = (s or {}).get("lat") or ""
    r["steady_interval_first"] = iv[0] if iv else ""
    r["steady_interval"] = sustained if sustained is not None else ""
    meets = (r["steady_result"] == "CORRECT" and r["steady_latency"]
             and sustained is not None
             and 22 <= int(r["steady_latency"]) <= 64 and sustained <= 16)
    r["meets_bars_steady"] = "yes" if meets else "no"
    r["meets_bars_visible"] = "yes" if r["first_build_meeting_bars"] else "no"

with open(S + "/results/results.csv", "w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(S + "/results/results.json", "w", encoding="utf-8") as h:
    json.dump({"trials": rows}, h, indent=1)

by_arm = {}
for r in rows:
    a = by_arm.setdefault(r["arm"], {"n": 0, "vis": 0, "steady": 0, "tok": []})
    a["n"] += 1
    a["vis"] += r["meets_bars_visible"] == "yes"
    a["steady"] += r["meets_bars_steady"] == "yes"
    a["tok"].append(int(r["tokens"]))
print("%-6s %-6s %-24s %-24s %s" % ("arm", "n", "meets bars (2 products)", "meets bars (12 products)", "median tokens"))
for arm in ("spmw", "rtl", "hls"):
    a = by_arm[arm]; t = sorted(a["tok"])
    print("%-6s %-6d %-24s %-24s %d" % (arm, a["n"], "%d of %d" % (a["vis"], a["n"]),
          "%d of %d" % (a["steady"], a["n"]), t[len(t) // 2]))
