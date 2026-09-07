import json, sys
p = sys.argv[1].rstrip("/") + ".transcript.jsonl"
for line in open(p, errors="replace"):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    k = r["kind"]
    if k == "assistant":
        calls = r.get("calls") or []
        names = [c.get("name") if isinstance(c, dict) else c for c in calls]
        cmds = [(c.get("args") or {}).get("command") or (c.get("args") or {}).get("path")
                for c in calls if isinstance(c, dict)]
        print("assistant t=%-8s tok=%-7s calls=%s %s" % (r["t"], r["tokens"], names, cmds))
        if not calls:
            print("    text: %r" % ((r.get("text") or "")[:400],))
    elif k == "tool_result":
        print("    -> %s %s : %s" % (r.get("name"), r.get("args"), (r.get("result") or "")[:200].replace("\n", " | ")))
    elif k == "api_error":
        print("    API ERROR: %s" % (r.get("detail", "")[:200],))
