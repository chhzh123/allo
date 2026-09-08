S = "/scratch/hc676/agentstudy"

p = S + "/arms/spmw/build.py"
s = open(p).read()
old = """    fabric = load(path)
    graph = spmw.elaborate(fabric)
    names = sba.stage(graph, out, PART, 300.0,
                      pipeline_style=getattr(fabric, "spmw_pipeline_style", None))
    sba.synthesise(out, names, jobs=8)
    sim = os.path.join(out, "sim")
    sources = collect(out, names, sim)"""
new = """    def stage_time(label, fn):
        start = time.time()
        value = fn()
        print("STUDY STAGE %s %.2f" % (label, time.time() - start))
        return value

    fabric = load(path)
    graph = stage_time("elaborate", lambda: spmw.elaborate(fabric))
    names = stage_time("generate", lambda: sba.stage(
        graph, out, PART, 300.0,
        pipeline_style=getattr(fabric, "spmw_pipeline_style", None)))
    stage_time("synthesise", lambda: sba.synthesise(out, names, jobs=8))
    sim = os.path.join(out, "sim")
    sources = stage_time("assemble", lambda: collect(out, names, sim))"""
assert s.count(old) == 1
s = s.replace(old, new)

old = """    sba._run(["xvlog", "-sv"] + sources, sim)
    loose = [f for f in os.listdir(sim) if f.endswith(".v")]
    if loose:
        sba._run(["xvlog"] + loose, sim)
    sba._run(["xelab", "tb", "-s", "tbsim", "--timescale", "1ns/1ps",
              "--generic_top", f"NPROD={nprod}", "-L", "unisims_ver",
              "-L", "unimacro_ver", "-L", "secureip"], sim)
    done = subprocess.run(["xsim", "tbsim", "-runall", "-testplusarg",
                           f"vecdir={STUDY}/task/vectors/{vecset}"],
                          cwd=sim, capture_output=True, text=True, check=False)"""
new = """    def compile_all():
        sba._run(["xvlog", "-sv"] + sources, sim)
        loose = [f for f in os.listdir(sim) if f.endswith(".v")]
        if loose:
            sba._run(["xvlog"] + loose, sim)

    stage_time("compile", compile_all)
    stage_time("elaborate_rtl", lambda: sba._run(
        ["xelab", "tb", "-s", "tbsim", "--timescale", "1ns/1ps",
         "--generic_top", "NPROD=%d" % nprod, "-L", "unisims_ver",
         "-L", "unimacro_ver", "-L", "secureip"], sim))
    done = stage_time("simulate", lambda: subprocess.run(
        ["xsim", "tbsim", "-runall", "-testplusarg",
         "vecdir=%s/task/vectors/%s" % (STUDY, vecset)],
        cwd=sim, capture_output=True, text=True, check=False))"""
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace("import subprocess\nimport sys", "import subprocess\nimport sys\nimport time")
open(p, "w").write(s)
print("spmw arm: stage timing added")

p = S + "/harness/agent.py"
s = open(p).read()
old = """        self.submitted = False
        self.build_history = []"""
new = """        self.submitted = False
        self.build_history = []
        # Where the effort went, segment by segment. A segment is the work
        # between two builds: the model thinking and writing, then the tools
        # running. Reported per build, so a trial reads as a sequence of
        # attempts rather than as one total.
        self.segments = []
        self.seg_tokens = 0
        self.seg_model_seconds = 0.0"""
assert s.count(old) == 1
s = s.replace(old, new)

old = """            out = (done.stdout + done.stderr).strip()
            verdict = [l for l in out.splitlines() if "STUDY " in l or "MISMATCH" in l]"""
new = """            tool_seconds = time.time() - start
            out = (done.stdout + done.stderr).strip()
            stages = {}
            for line in out.splitlines():
                if line.startswith("STUDY STAGE "):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            stages[parts[2]] = round(float(parts[3]), 2)
                        except ValueError:
                            pass
            self.segments.append({
                "build": self.builds,
                "tokens": self.seg_tokens,
                "model_seconds": round(self.seg_model_seconds, 1),
                "tool_seconds": round(tool_seconds, 1),
                "stages": stages,
                "verdict": next((l.strip() for l in out.splitlines()
                                 if "STUDY RESULT" in l or "STUDY BUILD" in l), None),
            })
            self.record("segment", **self.segments[-1])
            self.seg_tokens = 0
            self.seg_model_seconds = 0.0
            verdict = [l for l in out.splitlines()
                       if ("STUDY " in l and "STUDY STAGE" not in l) or "MISMATCH" in l]"""
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace("            self.tool_seconds += time.time() - start\n",
              "            self.tool_seconds += tool_seconds\n")

old = """        self.model_seconds += time.time() - start
        usage = body.get("usage") or {}
        self.tokens += int(usage.get("total_tokens") or 0)"""
new = """        elapsed = time.time() - start
        self.model_seconds += elapsed
        self.seg_model_seconds += elapsed
        usage = body.get("usage") or {}
        spent = int(usage.get("total_tokens") or 0)
        self.tokens += spent
        self.seg_tokens += spent"""
assert s.count(old) == 1
s = s.replace(old, new)

old = """    summary = {"arm": args.arm, "model": args.model, "trial": trial.dir,"""
new = """    summary = {"arm": args.arm, "model": args.model, "trial": trial.dir,
               "segments": trial.segments,"""
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w").write(s)
print("agent loop: per-segment accounting added")
