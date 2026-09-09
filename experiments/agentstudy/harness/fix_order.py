p = "/scratch/hc676/agentstudy/harness/agent.py"
s = open(p).read()
old = """            self.tool_seconds += tool_seconds
            tool_seconds = time.time() - start
"""
new = """            tool_seconds = time.time() - start
            self.tool_seconds += tool_seconds
"""
assert s.count(old) == 1, "the two lines are not in the expected order"
open(p, "w").write(s.replace(old, new))
print("build path: tool time is computed before it is accumulated")
