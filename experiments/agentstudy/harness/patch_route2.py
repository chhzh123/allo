import json

p = "/scratch/hc676/agentstudy/harness/agent.py"
s = open(p).read()
done = []


def sub(old, new, label):
    global s
    if new.strip() and new.split("\n")[0].strip() in s and label != "route block":
        done.append(label + " (already there)")
        return
    if s.count(old) != 1:
        done.append(label + " SKIPPED, anchor count %d" % s.count(old))
        return
    s = s.replace(old, new)
    done.append(label)


sub('"description": "Run one allowed command: build, cat <path>, or ls.",',
    '"description": "Run one allowed command: build, route, cat <path>, or ls.",',
    "tool description")

if "self.routes = 0" not in s:
    sub("""        self.seg_tokens = 0
        self.seg_model_seconds = 0.0""",
        """        self.seg_tokens = 0
        self.seg_model_seconds = 0.0
        self.routes = 0
        self.route_history = []""", "route counters")
else:
    done.append("route counters (already there)")

if 'if command == "route":' not in s:
    sub('        if command == "ls":',
        '''        if command == "route":
            if not self.build_history:
                return "refused: build something that simulates before routing it"
            if self.routes >= self.args.routes:
                return "refused: no routing runs left in the budget"
            self.routes += 1
            rtl = os.path.join(self.dir, self.arm.get("rtl_dir", "sim"))
            out = os.path.join(self.dir, "route_%d" % self.routes)
            start = time.time()
            done_run = subprocess.run(
                ["bash", os.path.join(STUDY, "harness", "pnr_ooc.sh"), rtl, "dut_norm", out],
                capture_output=True, text=True, timeout=self.args.tool_timeout,
                stdin=subprocess.DEVNULL, check=False)
            seconds = time.time() - start
            self.tool_seconds += seconds
            text = (done_run.stdout + done_run.stderr).strip()
            lines = [l for l in text.splitlines() if l.startswith("PNR")]
            body = "\\n".join(lines) if lines else text[-2000:]
            self.route_history.append(body)
            self.record("route", seconds=round(seconds, 1), output=body[:2000])
            return body + "\\n\\nRouting runs left: %d." % (self.args.routes - self.routes)
        if command == "ls":''', "route block")
else:
    done.append("route block (already there)")

sub('        return "refused: the commands that run are `build`, `cat <path>` and `ls`"',
    '        return ("refused: the commands that run are `build`, `route`, "\n'
    '                "`cat <path>` and `ls`")', "refusal text")

sub('    ap.add_argument("--builds", type=int, default=40)',
    '    ap.add_argument("--builds", type=int, default=40)\n'
    '    ap.add_argument("--routes", type=int, default=3)', "routes argument")

sub('.replace("{STEPS}", str(args.builds)))',
    '.replace("{STEPS}", str(args.builds))\n'
    '              .replace("{ROUTES}", str(args.routes)))', "prompt slot")

sub('"builds": trial.builds, "submitted": trial.submitted,',
    '"builds": trial.builds, "routes": trial.routes, "submitted": trial.submitted,',
    "summary field")

sub('                 builds_budget=args.builds,',
    '                 builds_budget=args.builds, routes_budget=args.routes,',
    "start record")
open(p, "w").write(s)
for d in done:
    print("  " + d)
