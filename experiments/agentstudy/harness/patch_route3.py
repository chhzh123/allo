p = "/scratch/hc676/agentstudy/harness/prompt_common.md"
s = open(p).read()
if "{ROUTES}" not in s:
    old = """    build         compile, elaborate and simulate your design against the
                  visible vectors, then report how many values were wrong and
                  how many cycles it took"""
    new = """    build         compile, elaborate and simulate your design against the
                  visible vectors, then report how many values were wrong and
                  how many cycles it took
    route         place and route what you last built, on the real device, and
                  report its lookup tables, registers, multipliers and worst
                  timing slack. It takes about ten minutes and you get
                  {ROUTES} of them, so build until you are close, then route"""
    assert s.count(old) == 1
    s = s.replace(old, new)
    s = s.replace("Your budget is {BUDGET} tokens and {STEPS} builds. Both are counted for you and\nreported to you after every build.",
                  "Your budget is {BUDGET} tokens, {STEPS} builds and {ROUTES} routing runs. All\nthree are counted for you and reported after every build.")
    open(p, "w").write(s)
    print("prompt: route documented")
else:
    print("prompt: already documents route")

p = "/scratch/hc676/agentstudy/harness/run_trial.sh"
s = open(p).read()
if "--routes" not in s:
    old = '--wall-seconds "${TRIAL_WALL:-14400}"'
    assert s.count(old) == 1
    s = s.replace(old, old + ' --routes "${TRIAL_ROUTES:-3}"')
    open(p, "w").write(s)
    print("run_trial: routes wired")
else:
    print("run_trial: already wired")
