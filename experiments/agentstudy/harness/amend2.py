p = "/scratch/hc676/agentstudy/PREREG.md"
s = open(p).read()
s += """
## Amendment 2, after a discarded launch

**What happened.** The first launch of the fifteen trials was stopped after 47
minutes and discarded. The harness kept each trial's transcript inside the
model's own working directory, so `ls` showed it and a model read it back:
one trial spent 112,000 tokens in a single turn after `cat transcript.jsonl`,
and had consumed half its budget without ever running a build.

**The fix.** The transcript and the summary now live beside the trial directory
rather than in it, and `ls` and `cat` serve only the three files a trial is
supposed to have: the task, the reference, and the design under construction.
Anything else is refused by name.

**What this does not change.** No graded result came from the discarded launch;
all fifteen trials start again from scratch. The task, the bars, the packs, the
models, the temperature and the budget are unchanged. The discarded transcripts
are kept as `discarded_launch_1/` so the reason for the restart is checkable.
"""
open(p, "w").write(s)
print("amendment 2 recorded")
