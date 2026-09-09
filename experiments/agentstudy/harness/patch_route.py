import json

# --- 1. the routing script must set up its own tools -----------------------
p = "/scratch/hc676/agentstudy/harness/pnr_ooc.sh"
s = open(p).read()
old = "RTL=$1; TOP=$2; OUT=$3"
new = """source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1
RTL=$1; TOP=$2; OUT=$3"""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("pnr_ooc.sh: environment sourced")

# --- 2. each arm says where its RTL lands ----------------------------------
p = "/scratch/hc676/agentstudy/harness/arms.json"
arms = json.load(open(p))
arms["spmw"]["rtl_dir"] = "build/sim"
arms["hls"]["rtl_dir"] = "sim"
arms["rtl"]["rtl_dir"] = "sim"
json.dump(arms, open(p, "w"), indent=1)
print("arms.json: rtl_dir per arm")

# --- 3. the agent gains a route command with its own budget ----------------
p = "/scratch/hc676/agentstudy/harness/agent.py"
s = open(p).read()

old = '''        "description": "Run one allowed command: build, cat <path>, or ls.",'''
new = '''        "description": "Run one allowed command: build, route, cat <path>, or ls.",'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''        self.segments = []
        self.seg_tokens = 0
        self.seg_model_seconds = 0.0'''
new = '''        self.segments = []
        self.seg_tokens = 0
        self.seg_model_seconds = 0.0
        self.routes = 0
        self.route_history = []'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''        if command == "ls":'''
new = '''        if command == "route":
            if not self.build_history:
                return "refused: build something that simulates before routing it"
            if self.routes >= self.args.routes:
                return "refused: no routing runs left in the budget"
            self.routes += 1
            rtl = os.path.join(self.dir, self.arm.get("rtl_dir", "sim"))
            out = os.path.join(self.dir, "route_%d" % self.routes)
            start = time.time()
            done = subprocess.run(
                ["bash", os.path.join(STUDY, "harness", "pnr_ooc.sh"), rtl, "dut_norm", out],
                capture_output=True, text=True, timeout=self.args.tool_timeout,
                stdin=subprocess.DEVNULL, check=False)
            seconds = time.time() - start
            self.tool_seconds += seconds
            text = (done.stdout + done.stderr).strip()
            lines = [l for l in text.splitlines() if l.startswith(("PNR ", "PNR_UNROUTED"))]
            body = "\\n".join(lines) if lines else text[-2000:]
            self.route_history.append(body)
            self.record("route", seconds=round(seconds, 1), output=body[:2000])
            return (body + "\\n\\nRouting runs left: %d." % (self.args.routes - self.routes))
        if command == "ls":'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''            return "refused: the commands that run are `build`, `cat <path>` and `ls`"'''
new = '''            return ("refused: the commands that run are `build`, `route`, "
                    "`cat <path>` and `ls`")'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''    ap.add_argument("--builds", type=int, default=40)'''
new = '''    ap.add_argument("--builds", type=int, default=40)
    ap.add_argument("--routes", type=int, default=3)'''
assert s.count(old) == 1
s = s.replace(old, new)

s = s.replace('.replace("{STEPS}", str(args.builds)))',
              '.replace("{STEPS}", str(args.builds)).replace("{ROUTES}", str(args.routes)))')
s = s.replace('"builds": trial.builds, "submitted": trial.submitted,',
              '"builds": trial.builds, "routes": trial.routes, "submitted": trial.submitted,')
s = s.replace('trial.record("start", arm=args.arm, model=args.model, tokens_budget=args.tokens,\n                 builds_budget=args.builds,',
              'trial.record("start", arm=args.arm, model=args.model, tokens_budget=args.tokens,\n                 builds_budget=args.builds, routes_budget=args.routes,')
open(p, "w").write(s)
print("agent.py: route command added")

# --- 4. the prompt tells them it exists ------------------------------------
p = "/scratch/hc676/agentstudy/harness/prompt_common.md"
s = open(p).read()
old = """    build         compile, elaborate and simulate your design against the
                  visible vectors, then report how many values were wrong and
                  how many cycles it took"""
new = """    build         compile, elaborate and simulate your design against the
                  visible vectors, then report how many values were wrong and
                  how many cycles it took
    route         place and route your last built design on the target device
                  and report its lookup tables, registers, multipliers and
                  worst timing slack. This takes about ten minutes and you may
                  do it {ROUTES} times, so build first and route when you think
                  you are close"""
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace("Your budget is {BUDGET} tokens and {STEPS} builds.",
              "Your budget is {BUDGET} tokens, {STEPS} builds and {ROUTES} routing runs.")
open(p, "w").write(s)
print("prompt: route documented")

p = "/scratch/hc676/agentstudy/harness/run_trial.sh"
s = open(p).read()
s = s.replace('--wall-seconds "${TRIAL_WALL:-14400}"',
              '--wall-seconds "${TRIAL_WALL:-14400}" --routes "${TRIAL_ROUTES:-3}"')
open(p, "w").write(s)
print("run_trial: routes wired")
