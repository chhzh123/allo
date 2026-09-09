p = "/scratch/hc676/agentstudy/PREREG.md"
s = open(p).read()
anchor = "**Who this affects.**"
add = """**Both GLM trials are repeated under this amendment**, the SPMW one and the
HLS one, because both were running unbounded when they stalled: the HLS trial
reached 412,035 tokens without a single build. Its SystemVerilog trial had
already submitted a correct design before the limit bit, and stands.

"""
if "Both GLM trials are repeated" not in s:
    assert s.count(anchor) == 1
    s = s.replace(anchor, add + anchor)
    open(p, "w").write(s)
    print("amendment 5 extended to both GLM trials")
else:
    print("already extended")
