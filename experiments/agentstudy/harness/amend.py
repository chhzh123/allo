p = "/scratch/hc676/agentstudy/PREREG.md"
s = open(p).read()
old = "| Budget | 200,000 billed tokens, or 25 builds, whichever comes first |"
new = "| Budget | 500,000 billed tokens, 20 builds, or 4 hours, whichever comes first (see amendment 1) |"
assert s.count(old) == 1
s = s.replace(old, new)
s += """
---

## Amendment 1, before the first graded trial

**What changed.** The budget was 200,000 billed tokens and 25 builds. It is now
500,000 tokens, 20 builds and a four-hour wall cap.

**Why.** A trial run of the harness, thrown away and not graded, showed one
model spending 861 seconds and 50,169 tokens on a single turn of reasoning
before touching a tool. At the original cap a reasoning-heavy model would
exhaust its whole budget in one turn and never reach a build, so the token cap
would have measured reasoning verbosity rather than design ability, and it
would have done so unequally: billed tokens include reasoning tokens, and the
five models differ by an order of magnitude in how many they emit.

**What this means for the analysis.** Builds, not tokens, is now the binding
constraint in most trials, and every model gets the same twenty attempts.
Tokens are reported as a cost rather than used as a gate, and the
budget-sensitivity question is answered by the profile curve, the fraction of
trials passing against tokens spent, which does not depend on where a cap was
placed. The stop reason of every trial is recorded, so a trial stopped by
tokens, by builds or by the clock is distinguishable in the results.

**What did not change.** The task, the architecture requirement, the pass bars
and their derivation, the documentation packs, the models, and the temperature.
"""
open(p, "w").write(s)
print("amendment recorded")
