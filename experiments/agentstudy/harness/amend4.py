p = "/scratch/hc676/agentstudy/PREREG.md"
s = open(p).read()
s += """
## Amendment 4, during the graded run

**Two trials are being repeated, and this records why, because repeating a
trial is exactly the move that can turn a study into a search for a result.**

`moonshotai/kimi-k3` in the HLS arm died at 3,252 tokens on
`http.client.IncompleteRead`, a transient network fault. The retry in the
agent loop caught three exception classes and that was not one of them, so a
dropped connection ended the trial. That is a harness fault with no bearing on
the model, and the trial is repeated.

`z-ai/glm-5.3` in the SPMW arm spent 615,613 tokens across turns that returned
neither text nor a tool call, then wrote its design after the budget had
already gone. Each such turn cost about 115,000 tokens. The harness noticed
nothing and kept asking. This is closer to model behaviour than to a fault,
but the loop gave it no signal that it was spending a budget on nothing, so it
is treated as a harness deficiency and repeated. The loop now stops after
three consecutive turns that produce nothing, recording `empty_replies` as the
stop reason, and records each turn's finish reason and reasoning length so the
same thing is diagnosable rather than mysterious next time.

**What is not repeated.** `deepseek/deepseek-v4-pro` in the SystemVerilog arm
used its whole token budget across four builds, reaching a correct design at
build 1 and again at build 3 and then replacing it with a wrong one both times.
That is a result, not a fault, and it stands.

**The rule applied here**, stated so it can be checked: a trial is repeated
only when the harness, not the model, ended it. Every repeat is named in this
file with its reason, and the original transcript is kept.
"""
open(p, "w").write(s)
print("amendment 4 recorded")
