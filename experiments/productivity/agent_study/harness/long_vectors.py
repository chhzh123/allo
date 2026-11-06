"""A longer vector set, because an interval measured over two products is not
a steady state: a design with deep input buffering can accept the second
product quickly and only later back up."""
import json, os, random

sp = "/scratch/hc676/agentstudy/task/vectors"


def gen(seed, extrema):
    r = random.Random(seed)
    def mk():
        m = [[r.randint(-128, 127) for _ in range(8)] for _ in range(8)]
        if extrema:
            m[0][0], m[1][1] = -128, 127
            m[7][7], m[6][6] = 127, -128
        return m
    A, B = mk(), mk()
    C = [[sum(A[i][k] * B[k][j] for k in range(8)) for j in range(8)] for i in range(8)]
    return {"A": A, "B": B, "C": C}


cases = [gen(7000 + i, i % 2 == 0) for i in range(12)]
with open(f"{sp}/steady.json", "w") as h:
    json.dump(cases, h, indent=1)
for kind in ("a", "b", "c"):
    rows = []
    for case in cases:
        m = case[kind.upper()]
        if kind == "a":
            chans = [[m[i][k] for k in range(8)] for i in range(8)]
        elif kind == "b":
            chans = [[m[k][j] for k in range(8)] for j in range(8)]
        else:
            chans = [[m[i][j] for j in range(8)] for i in range(8)]
        width = 8 if kind != "c" else 32
        for ch in chans:
            rows += ["%0*x" % (width // 4, v & ((1 << width) - 1)) for v in ch]
    with open(f"{sp}/steady_{kind}.hex", "w") as h:
        h.write("\n".join(rows) + "\n")
print("wrote a 12-product vector set")
