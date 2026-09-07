import json, os, re, glob
T = "/scratch/hc676/agentstudy/trials"
BARS = {"lat_lo": 22, "lat_hi": 64, "interval": 16}


def parse(verdict_lines):
    """correctness, first-product latency and steady interval from a build."""
    text = "\n".join(verdict_lines)
    ok = "STUDY RESULT CORRECT" in text
    lat = re.search(r"STUDY PRODUCT 0 .*latency (\d+)", text)
    iv = re.findall(r"STUDY INTERVAL \d+ (\d+)", text)
    return ok, (int(lat.group(1)) if lat else None), (min(int(x) for x in iv) if iv else None)


rows = []
for d in sorted(glob.glob(T + "/*__*")):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    path = d.rstrip("/") + ".transcript.jsonl"
    if not os.path.isfile(path):
        rows.append((name, 0, 0, None, None, None, "no transcript", False))
        continue
    tokens = builds = 0
    best = None
    last = None
    submitted = False
    stopped = "running"
    hist = []
    for line in open(path, errors="replace"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        tokens = max(tokens, r.get("tokens", 0))
        builds = max(builds, r.get("builds", 0))
        if r["kind"] == "build":
            ok, lat, iv = parse([r.get("output", "")])
            hist.append((ok, lat, iv))
            last = (ok, lat, iv)
            if ok and lat and iv:
                passes = BARS["lat_lo"] <= lat <= BARS["lat_hi"] and iv <= BARS["interval"]
                if passes and best is None:
                    best = len(hist)          # the build at which every cycle bar was met
        if r["kind"] == "submit":
            submitted = True
        if r["kind"] == "end":
            stopped = r.get("stopped_by", "ended")
    rows.append((name, tokens, builds, best, last, len(hist), stopped, submitted))

print("%-34s %8s %6s %6s %-22s %-9s %s" %
      ("trial", "tokens", "builds", "1st ok", "last build", "stopped", "submitted"))
for name, tok, b, best, last, nb, stopped, sub in rows:
    if last:
        ok, lat, iv = last
        desc = "%s lat=%s iv=%s" % ("correct" if ok else "WRONG", lat, iv)
    else:
        desc = "-"
    print("%-34s %8d %6d %6s %-22s %-9s %s" %
          (name, tok, b, best if best else "-", desc, stopped, sub))
