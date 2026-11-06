#!/usr/bin/env python3
"""The agent study's trials, as a table and as rows.

    collect_trials.py <trials-dir> <out-dir>

Reads each trial's transcript and summary and writes results.csv, results.json
and a readable per-build breakdown. A trial still running is included with its
state so far, marked as such; nothing is inferred.
"""
import csv, glob, json, os, re, sys

BARS = {"lat_lo": 22, "lat_hi": 64, "interval": 16}


def cycles(text):
    ok = "STUDY RESULT CORRECT" in text
    lat = re.search(r"STUDY PRODUCT 0 .*latency (\d+)", text)
    iv = re.findall(r"STUDY INTERVAL \d+ (\d+)", text)
    return ok, (int(lat.group(1)) if lat else None), (min(int(x) for x in iv) if iv else None)


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rows, detail = [], []
    for path in sorted(glob.glob(os.path.join(src, "*.transcript.jsonl"))):
        name = os.path.basename(path).replace(".transcript.jsonl", "")
        if "__" not in name:
            continue
        model, arm = name.rsplit("__", 1)
        model = model.replace("_", "/", 1).replace("_", ".")
        builds, tokens, routes = 0, 0, 0
        first_pass, submitted, stopped = None, False, "running"
        model_s = tool_s = wall_s = 0.0
        best = None
        for line in open(path, errors="replace"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = r.get("kind")
            tokens = max(tokens, r.get("tokens", 0))
            if k == "segment":
                builds = r["build"]
                ok, lat, iv = cycles(r.get("verdict") or "")
                detail.append({"trial": name, "model": model, "arm": arm,
                               "build": r["build"], "tokens": r["tokens"],
                               "model_seconds": r["model_seconds"],
                               "tool_seconds": r["tool_seconds"],
                               "verdict": r.get("verdict"),
                               "stages": json.dumps(r.get("stages") or {})})
            if k == "build":
                ok, lat, iv = cycles(r.get("output") or "")
                if ok and lat and iv:
                    meets = BARS["lat_lo"] <= lat <= BARS["lat_hi"] and iv <= BARS["interval"]
                    if meets and first_pass is None:
                        first_pass = builds or 1
                    if best is None or (iv, lat) < best[:2]:
                        best = (iv, lat, ok)
            if k == "route":
                routes += 1
            if k == "submit":
                submitted = True
            if k == "end":
                stopped = r.get("stopped_by", "ended")
                model_s = r.get("model_seconds", 0.0)
                tool_s = r.get("tool_seconds", 0.0)
                wall_s = r.get("wall_seconds", 0.0)
        rows.append({
            "trial": name, "model": model, "arm": arm,
            "state": "complete" if stopped != "running" else "running",
            "stopped_by": stopped, "submitted": submitted,
            "builds_used": builds, "routes_used": routes, "tokens": tokens,
            "first_build_meeting_bars": first_pass if first_pass else "",
            "best_latency": best[1] if best else "", "best_interval": best[0] if best else "",
            "model_seconds": round(model_s, 1), "tool_seconds": round(tool_s, 1),
            "wall_seconds": round(wall_s, 1),
        })
    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(out, "results_per_build.csv"), "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(detail[0].keys())); w.writeheader(); w.writerows(detail)
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as h:
        json.dump({"trials": rows, "builds": detail}, h, indent=1)
    for r in rows:
        print("  %-34s %-9s %-14s builds=%d routes=%d tok=%-7d first_ok=%-3s lat=%-5s iv=%-5s"
              % (r["trial"], r["arm"], r["stopped_by"], r["builds_used"], r["routes_used"],
                 r["tokens"], r["first_build_meeting_bars"] or "-",
                 r["best_latency"] or "-", r["best_interval"] or "-"))


if __name__ == "__main__":
    main()
