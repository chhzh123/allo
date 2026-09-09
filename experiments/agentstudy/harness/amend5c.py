p = "/scratch/hc676/agentstudy/PREREG.md"
s = open(p).read()
add = """
### Which already-finished trials the bound would have changed

Amendment 5 arrived mid-run, so the trials that finished before it ran without
a limit. Checking every turn of every finished trial against the new one:

| Trial | Peak completion tokens in a turn | Inside the new 65,536 limit |
|---|---:|---|
| deepseek, SPMW | 20,954 | yes |
| deepseek, HLS | 38,639 | yes |
| deepseek, SystemVerilog | 49,779 | yes |
| kimi, SPMW | 32,289 | yes |
| kimi, SystemVerilog | 33,634 | yes |
| glm, SystemVerilog | **88,159** | **no** |

So the bound is inert for five of the six and would have constrained exactly one
turn in one trial. That trial, `z-ai/glm-5.3` in the SystemVerilog arm, is
repeated under the bound so that every reported trial ran under one set of
request parameters. Its unbounded transcript is kept, and if the repeat gives a
different outcome both are reported.

"""
anchor = "## Amendment 5, during the graded run"
assert s.count(anchor) == 1
s = s.replace("**Who this affects.**", add.strip() + "\n\n**Who this affects.**")
open(p, "w").write(s)
print("recorded which finished trials the bound would have changed")
