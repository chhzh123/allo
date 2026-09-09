p = "/scratch/hc676/agentstudy/harness/agent.py"
s = open(p).read()

old = '''    ap.add_argument("--tool-timeout", type=int, default=3600)'''
new = '''    ap.add_argument("--tool-timeout", type=int, default=3600)
    ap.add_argument("--wall-seconds", type=int, default=14400)'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''    while trial.tokens < args.tokens and trial.builds < args.builds:'''
new = '''    while (trial.tokens < args.tokens and trial.builds < args.builds
           and time.time() - trial.started < args.wall_seconds):'''
assert s.count(old) == 1
s = s.replace(old, new)

# The budget line the model is told must match what actually binds.
old = '''    template = open(os.path.join(STUDY, "harness", "prompt_common.md"), encoding="utf-8").read()'''
new = '''    template = open(os.path.join(STUDY, "harness", "prompt_common.md"), encoding="utf-8").read()
    # A model that spends its whole budget reasoning in one turn never reaches
    # a build; the budget is reported so it can pace itself.'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''    trial.record("end", submitted=trial.submitted,'''
new = '''    trial.record("end", submitted=trial.submitted,
                 stopped_by=("submit" if trial.submitted else
                             "tokens" if trial.tokens >= args.tokens else
                             "builds" if trial.builds >= args.builds else "wall"),'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''    summary = {"arm": args.arm, "model": args.model, "trial": trial.dir,
               "segments": trial.segments,'''
new = '''    summary = {"arm": args.arm, "model": args.model, "trial": trial.dir,
               "segments": trial.segments,
               "stopped_by": ("submit" if trial.submitted else
                              "tokens" if trial.tokens >= args.tokens else
                              "builds" if trial.builds >= args.builds else "wall"),'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w").write(s)
print("wall cap added, stop reason recorded")

# The runner: budgets that let a reasoning model actually reach a build.
p = "/scratch/hc676/agentstudy/harness/run_trial.sh"
s = open(p).read()
s = s.replace('--tokens "${TRIAL_TOKENS:-200000}" --builds "${TRIAL_BUILDS:-40}"',
              '--tokens "${TRIAL_TOKENS:-500000}" --builds "${TRIAL_BUILDS:-20}" '
              '--wall-seconds "${TRIAL_WALL:-14400}"')
open(p, "w").write(s)
print("run_trial: 500k tokens, 20 builds, 4-hour wall cap")

p = "/scratch/hc676/agentstudy/harness/run_all.sh"
s = open(p).read()
s = s.replace('export TRIAL_TOKENS=${TRIAL_TOKENS:-200000}', 'export TRIAL_TOKENS=${TRIAL_TOKENS:-500000}')
s = s.replace('export TRIAL_BUILDS=${TRIAL_BUILDS:-25}', 'export TRIAL_BUILDS=${TRIAL_BUILDS:-20}')
# All five models at once: a trial is mostly waiting on the model, and the
# tool jobs are short and intermittent.
s = s.replace("""for MODEL in $MODELS; do
  TAG=$(echo "$MODEL" | tr '/.' '__')
  echo "===== $MODEL  $(date +%FT%T)  load $(cut -d' ' -f1 /proc/loadavg)"
  for ARM in spmw hls rtl; do
    bash "$S/harness/run_trial.sh" "$ARM" "$MODEL" "${TAG}__${ARM}" \\
      > "$S/trials/${TAG}__${ARM}.log" 2>&1 &
  done
  wait""",
"""mkdir -p "$S/trials"
for MODEL in $MODELS; do
  TAG=$(echo "$MODEL" | tr '/.' '__')
  echo "===== launching $MODEL  $(date +%FT%T)  load $(cut -d' ' -f1 /proc/loadavg)"
  for ARM in spmw hls rtl; do
    bash "$S/harness/run_trial.sh" "$ARM" "$MODEL" "${TAG}__${ARM}" \\
      > "$S/trials/${TAG}__${ARM}.log" 2>&1 &
    sleep 20
  done
done
wait
for MODEL in $MODELS; do
  TAG=$(echo "$MODEL" | tr '/.' '__')
  echo "===== $MODEL\"""")
open(p, "w").write(s)
print("run_all: all five models in flight, staggered")
