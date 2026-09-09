p = "/scratch/hc676/agentstudy/harness/agent.py"
s = open(p).read()
old = '''        payload = {"model": self.args.model, "messages": messages, "tools": TOOLS,
                   "temperature": self.args.temperature, "stream": False}'''
new = '''        # Without a limit the provider applies its own, and one model spent
        # every one of its 131,072 output tokens on reasoning and returned
        # nothing usable, three turns running. Bounding reasoning leaves room
        # for an answer. Applied identically to every model, and inside the
        # smallest completion ceiling among the five (128,000).
        payload = {"model": self.args.model, "messages": messages, "tools": TOOLS,
                   "temperature": self.args.temperature, "stream": False,
                   "max_tokens": self.args.max_tokens,
                   "reasoning": {"max_tokens": self.args.reasoning_tokens}}'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    ap.add_argument("--routes", type=int, default=3)'''
new = '''    ap.add_argument("--routes", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--reasoning-tokens", type=int, default=32768)'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w").write(s)
print("agent: one turn is bounded at 65,536 output tokens, at most 32,768 of them reasoning")

p = "/scratch/hc676/agentstudy/PREREG.md"
s = open(p).read()
s += """
## Amendment 5, during the graded run

**Every request now bounds its own output.** The loop sent no `max_tokens`, so
each provider applied its own default. One model's turns ended at exactly
131,072 completion tokens with the reasoning field consuming all of them, three
turns in a row, so it returned nothing usable and spent 615,613 tokens without
producing a design. Its actual ceiling is 943,718, so the limit was a default
rather than the model's capacity.

Raising the limit alone would let a single turn consume a whole trial budget,
so the fix bounds reasoning instead: `max_tokens` 65,536 with
`reasoning.max_tokens` 32,768, applied identically to all five models and
inside the smallest completion ceiling among them, which is 128,000. A turn can
therefore always emit an answer after thinking, and no turn can cost more than
about an eighth of the trial budget.

**Who this affects.** It is a uniform change, but not a neutral one: models that
reason at length are constrained more than terse ones. That is preferable to
the alternative, in which a model that reasons at length produces nothing at
all and is scored as having failed the task.
"""
open(p, "w").write(s)
print("amendment 5 recorded")
