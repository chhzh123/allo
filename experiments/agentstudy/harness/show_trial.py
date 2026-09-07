import json, os, sys
T = sys.argv[1]
path = T.rstrip("/") + ".transcript.jsonl"
if not os.path.isfile(path):
    print("no transcript at", path); raise SystemExit
for line in open(path):
    r = json.loads(line)
    k = r["kind"]
    if k == "assistant":
        print("  assistant t=%ss tokens=%s calls=%s fallback=%s"
              % (r["t"], r["tokens"], r.get("calls"), r.get("fallback")))
    elif k == "segment":
        print("  segment build=%s tokens=%s model_s=%s tool_s=%s stages=%s"
              % (r["build"], r["tokens"], r["model_seconds"], r["tool_seconds"], r.get("stages")))
        print("          verdict: %s" % (r.get("verdict"),))
    elif k == "build":
        print("  build output: %s" % (r.get("output", "")[:300].replace("\n", " | "),))
    elif k == "api_error":
        print("  API ERROR attempt=%s %s" % (r.get("attempt"), r.get("detail", "")[:200]))
    elif k in ("start", "submit", "end"):
        fields = {kk: vv for kk, vv in r.items() if kk not in ("system", "kind", "last_build")}
        print("  %s: %s" % (k, fields))
