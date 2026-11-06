# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""closure.py <dir> <top>: the SystemVerilog files `top` needs, from a split emission."""
import os, re, sys

d, top = sys.argv[1], sys.argv[2]
mods = {}
for f in os.listdir(d):
    if f.endswith((".sv", ".v")):
        text = open(os.path.join(d, f), errors="replace").read()
        for m in re.finditer(r"^\s*module\s+(\w+)", text, re.M):
            mods[m.group(1)] = (f, text)
seen, todo = [], [top]
while todo:
    m = todo.pop()
    if m in seen or m not in mods:
        continue
    seen.append(m)
    for inst in re.finditer(
        r"^\s*(\w+)\s+(?:#\s*\(.*?\)\s*)?\w+\s*\(", mods[m][1], re.M | re.S
    ):
        if inst.group(1) in mods and inst.group(1) not in seen:
            todo.append(inst.group(1))
files = sorted({mods[m][0] for m in seen})
print("\n".join(files))
print(f"# {len(seen)} modules in {len(files)} files under {top}", file=sys.stderr)
