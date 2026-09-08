p = "/scratch/hc676/agentstudy/harness/agent.py"
s = open(p).read()
changed = []

if '"--routes"' not in s:
    old = '    ap.add_argument("--builds", type=int, default=40)'
    assert s.count(old) == 1
    s = s.replace(old, old + '\n    ap.add_argument("--routes", type=int, default=3)')
    changed.append("--routes argument")

# The counters must be initialised in __init__ only; the same line also appears
# where a segment resets, so anchor on the constructor's neighbour.
if "self.routes = 0" not in s:
    old = """        self.segments = []
        self.seg_tokens = 0
        self.seg_model_seconds = 0.0"""
    assert s.count(old) == 1, "constructor anchor appears %d times" % s.count(old)
    s = s.replace(old, old + "\n        self.routes = 0\n        self.route_history = []")
    changed.append("route counters")

if "{ROUTES}" not in s:
    old = '.replace("{STEPS}", str(args.builds))'
    assert s.count(old) == 1
    s = s.replace(old, old + '\n              .replace("{ROUTES}", str(args.routes))')
    changed.append("prompt slot")

open(p, "w").write(s)
print("added:", ", ".join(changed) if changed else "nothing missing")
