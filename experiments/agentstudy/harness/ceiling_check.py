import glob, json, os
print("  did any completed turn hit a provider ceiling (which the bound now prevents)?")
for f in sorted(glob.glob("/scratch/hc676/agentstudy/trials/*.transcript.jsonl")):
    peak = 0
    hits = 0
    for line in open(f, errors="replace"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "assistant":
            c = int((r.get("usage") or {}).get("completion_tokens") or 0)
            peak = max(peak, c)
            if c in (131072, 128000, 65536, 32768):
                hits += 1
    name = os.path.basename(f).replace(".transcript.jsonl", "")
    flag = "  <-- hit a ceiling %d time(s)" % hits if hits else ""
    print("    %-34s peak completion %6d%s" % (name, peak, flag))
