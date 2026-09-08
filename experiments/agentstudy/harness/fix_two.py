# 1. My stage-timing patch dropped --timescale from the SystemVerilog arm's
#    elaboration, so every RTL design failed with the timescale mismatch that
#    was fixed on day one. The HLS arm kept it.
p = "/scratch/hc676/agentstudy/arms/rtl/build.sh"
s = open(p).read()
old = 'xelab tb -s tbsim --generic_top "NPROD=$NPROD"'
assert s.count(old) == 1, "unexpected xelab line"
s = s.replace(old, 'xelab tb -s tbsim --timescale 1ns/1ps --generic_top "NPROD=$NPROD"')
open(p, "w").write(s)
print("rtl arm: --timescale restored")

# 2. When a build failed, the agent showed only the one-line verdict and threw
#    away the tool's error text, so a model could not tell why. It failed the
#    same way twice for want of a message.
p = "/scratch/hc676/agentstudy/harness/agent.py"
s = open(p).read()
old = '''            body = "\\n".join(verdict) if verdict else out[-4000:]'''
new = '''            failed = any("STUDY BUILD FAIL" in l for l in verdict)
            if verdict and not failed:
                body = "\\n".join(verdict)
            elif verdict:
                # A failure needs the tool's own words, not just the verdict.
                detail = [l for l in out.splitlines()
                          if l not in verdict and l.strip()
                          and not l.startswith("STUDY STAGE")]
                body = "\\n".join(verdict + detail[-40:])
            else:
                body = out[-4000:]'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w").write(s)
print("agent: a failed build now carries the tool's error text")
