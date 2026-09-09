p = "/scratch/hc676/agentstudy/PREREG.md"
s = open(p).read()
s = s.replace("| Budget | 500,000 billed tokens, 20 builds, or 4 hours, whichever comes first (see amendment 1) |",
              "| Budget | 500,000 billed tokens, 8 builds, 3 routing runs, or 4 hours, whichever comes first (amendments 1 and 3) |")
s += """
## Amendment 3, before the first graded trial

**Routing is now a tool, not only a final check.** A trial may place and route
what it last built, three times, and see the lookup tables, registers,
multipliers, worst slack and unrouted nets. Measured cost on this device for a
design of this size: 399 seconds. Without it a model could not iterate on the
timing bar at all, and would be failed on a criterion it was given no way to
observe. The final grading still routes the submitted artifact once,
authoritatively, and that run is what the table reports.

**The build budget falls from 20 to 8.** Twenty attempts is more than this task
needs and makes an all-pass table likely, which would say nothing. Eight is
chosen rather than five to hedge one asymmetry: the models have seen a great
deal of SystemVerilog and Vitis HLS and none of SPMW, so a very tight budget
risks flooring the SPMW arm for unfamiliarity and producing a result about
training data rather than about the language.

**How this is reported.** Every trial records the build at which it first met
every cycle bar, so the result is reported as a profile: the fraction of trials
passing within one build, two, and so on to eight. A reading at five builds is
therefore available from the same runs, and reading at any budget does not
depend on where the cap happened to sit. Trials are told their real budget of
eight, so a five-build reading is drawn from models that were pacing for eight,
which makes it a conservative figure rather than a flattering one.
"""
open(p, "w").write(s)
print("amendment 3 recorded")

p = "/scratch/hc676/agentstudy/harness/run_all.sh"
s = open(p).read()
s = s.replace("export TRIAL_BUILDS=${TRIAL_BUILDS:-20}", "export TRIAL_BUILDS=${TRIAL_BUILDS:-8}")
s = s.replace("export TRIAL_TOKENS=${TRIAL_TOKENS:-500000}",
              "export TRIAL_TOKENS=${TRIAL_TOKENS:-500000}\nexport TRIAL_ROUTES=${TRIAL_ROUTES:-3}")
open(p, "w").write(s)
print("run_all: 8 builds, 3 routes")
