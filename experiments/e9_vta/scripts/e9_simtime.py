# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""How long E9's programmable mini-TPU takes to simulate one of its programs.

    python3 e9_simtime.py <program> <size>               # SPMW's OpenMP simulator
    python3 e9_simtime.py <program> <size> csim <dir>    # Vitis HLS's own C simulation

`<program>` is `llama` or `dsv4` (or `micro`, `mixed`): `ptpu_workload` of
`tests/dataflow/spmw/test_spmw_ptpu.py`, the launches E9 cosimulated for 1.05
and 3.67 million cycles at 4x4. One JSON line on stdout; every result is
checked against the workload's golden.

The simulator: `spmw.build(..., target="simulator")`, the build and two runs
timed apart; `ALLO_SIM_DEPTH` raises every channel's depth, which a functional
run's results do not depend on and which saves most of its synchronisation,
and the record carries the floor it ran at. Vitis's C simulation: `spmw.build(..., target="vitis_hls")`
writes the HLS project; `csim_design` runs it with its defaults and a
testbench that reads the operands from a file. Vitis runs a dataflow region's
processes one after another in the order they are called, so the generated
code is checked against the order its streams imply, and `order` reports
whether it already holds (SPMW emits producer first since 32d1db81) -- with
REORDER=1 the calls are moved into that order first, which is how the code
before that commit could be simulated at all. The compiled `csim.exe` is then
run twice more for the hot figure.
"""

import json
import os
import re
import resource
import subprocess
import sys
import time

# pylint: disable=wrong-import-position
import numpy as np

import allo.spmw as spmw
from test_spmw_ptpu import ptpu_workload

ARRAYS = ("A", "W", "Prog", "Bias")
CTYPES = {"int8": "int8_t", "int16": "int16_t", "int32": "int32_t", "int64": "int64_t"}
LIBS = ":".join(
    "/opt/xilinx/Vitis_HLS/2023.2/lnx64/" + p
    for p in (
        "tools/fpo_v7_1",
        "tools/fft_v9_1",
        "tools/fir_v7_0",
        "tools/dds_v6_0",
        "csim",
    )
)


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def simulator(program, size):
    engine = ptpu_workload(size, program)
    want = engine.spmw_ptpu["expected"]
    ops = [engine.spmw_operands[k] for k in ARRAYS]
    floor = int(os.environ.get("ALLO_SIM_DEPTH", "0"))
    t0 = time.time()
    module = spmw.build(engine, target="simulator", fifo_depth_floor=floor or None)
    out = {
        "program": program,
        "size": size,
        "depth_floor": floor,
        "fibers": os.environ.get("ALLO_SIM_FIBERS", "") not in ("", "0"),
        "build_s": round(time.time() - t0, 3),
    }
    for key in ("run_s", "rerun_s"):
        Y = np.zeros(want.shape, dtype=np.int32)
        c0, t0 = cpu(), time.time()
        module(*ops, Y)
        out[key] = round(time.time() - t0, 3)
        out[key.replace("_s", "_cpu_s")] = round(cpu() - c0, 2)
        out[key.replace("_s", "_pass")] = bool(np.array_equal(Y, want))
    out["load"] = [round(x, 2) for x in os.getloadavg()]
    print(json.dumps(out), flush=True)


def _functions(code):
    """Each function's parameters, and which of its streams it reads and writes."""
    found = {}
    for m in re.finditer(r"^void (\w+)\((.*?)\)\s*\{", code, re.S | re.M):
        name, params = m.group(1), m.group(2)
        end = code.find("\n}\n", m.end())
        body = code[m.end() : end]
        names = [
            re.findall(r"(\w+)\s*(?:\[[^\]]*\])*\s*$", p.strip())[0]
            for p in params.split(",")
        ]
        found[name] = (names, body)
    return found


def order_top(code):
    """Call the top's processes producer first, or say why no order exists."""
    functions = _functions(code)
    head = code.index("void top(")
    start = code.index("{", head) + 1
    end = code.index("\n}\n", start)
    lines = code[start:end].split("\n")
    calls = [
        (i, m.group(1), [a.strip() for a in m.group(2).split(",")])
        for i, line in enumerate(lines)
        if (m := re.match(r"\s*(\w+)\((.*)\);", line)) and m.group(1) in functions
    ]
    streams = set(re.findall(r"hls::stream<[^;]*>\s+(\w+)\s*;", code[start:end]))
    writers, readers, users = {}, {}, {}
    after = {k: set() for k in range(len(calls))}
    for k, (_i, callee, args) in enumerate(calls):
        params, body = functions[callee]
        for arg, param in zip(args, params):
            if arg in streams:
                if re.search(rf"\b{param}\.write\(", body):
                    writers.setdefault(arg, set()).add(k)
                if re.search(rf"\b{param}\.read\(", body):
                    readers.setdefault(arg, set()).add(k)
            elif re.fullmatch(r"[A-Za-z_]\w*", arg):
                # An array two calls share keeps their order: loaders before
                # the processes that read it, stores after those that write.
                for j in users.get(arg, ()):
                    after[j].add(k)
                users.setdefault(arg, []).append(k)
    for stream, ks in writers.items():
        for w in ks:
            after[w] |= readers.get(stream, set()) - {w}
    need = {k: 0 for k in after}
    for k, nxt in after.items():
        for n in nxt:
            need[n] += 1
    ready = [k for k in sorted(need) if need[k] == 0]
    order = []
    while ready:
        k = ready.pop(0)
        order.append(k)
        for n in sorted(after[k]):
            need[n] -= 1
            if need[n] == 0:
                ready.append(n)
    if len(order) < len(calls):
        stuck = sorted({calls[k][1] for k in need if need[k] > 0})
        return None, {"cycle_among": stuck, "calls": len(calls)}
    moved = sum(1 for a, b in zip(order, range(len(calls))) if a != b)
    slots = [i for i, _c, _a in calls]
    text = [lines[i] for i, _c, _a in calls]
    for slot, k in zip(slots, order):
        lines[slot] = text[k]
    return code[:start] + "\n".join(lines) + code[end:], {
        "calls": len(calls),
        "moved": moved,
    }


TESTBENCH = """#include <cstdint>
#include <cstdio>
#include <vector>
#include "kernel.h"
#define OPERANDS "%(path)s"
int main() {
  FILE *f = fopen(OPERANDS, "rb");
  std::vector<%(A)s> a(%(nA)d);
  std::vector<%(W)s> w(%(nW)d);
  std::vector<%(Prog)s> prog(%(nProg)d);
  std::vector<%(Bias)s> bias(%(nBias)d);
  std::vector<int32_t> want(%(nY)d), y(%(nY)d, 0);
  if (!f || fread(a.data(), sizeof(a[0]), a.size(), f) != a.size() ||
      fread(w.data(), sizeof(w[0]), w.size(), f) != w.size() ||
      fread(prog.data(), sizeof(prog[0]), prog.size(), f) != prog.size() ||
      fread(bias.data(), sizeof(bias[0]), bias.size(), f) != bias.size() ||
      fread(want.data(), 4, want.size(), f) != want.size()) {
    printf("NO OPERANDS\\n");
    return 2;
  }
  fclose(f);
  top(a.data(), w.data(), prog.data(), bias.data(), y.data());
  int bad = 0;
  for (size_t i = 0; i < y.size(); i++) bad += y[i] != want[i];
  printf(bad ? "PTPU CSIM FAIL %%d\\n" : "PTPU CSIM PASS %%d\\n", bad);
  return bad != 0;
}
"""

CSIM_TCL = """open_project csim_prj -reset
set_top top
add_files kernel.cpp
add_files -tb tb.cpp
open_solution -reset sol -flow_target vivado
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.333
csim_design
exit
"""


def csim(program, size, out):
    engine = ptpu_workload(size, program)
    want = engine.spmw_ptpu["expected"]
    ops = {k: np.ascontiguousarray(engine.spmw_operands[k]) for k in ARRAYS}
    res = {"program": program, "size": size}
    t0 = time.time()
    spmw.build(engine, target="vitis_hls", mode="csim", project=out)
    res["codegen_s"] = round(time.time() - t0, 3)
    path = os.path.join(out, "kernel.cpp")
    code, info = order_top(open(path, encoding="utf-8").read())
    info["reordered"] = bool(os.environ.get("REORDER"))
    res["order"] = info
    if code is None:
        res["pass"] = False
        res["error"] = "the processes form a cycle: no sequential order runs them"
        print(json.dumps(res), flush=True)
        return
    if info["reordered"]:
        open(path, "w", encoding="utf-8").write(code)
    operands = os.path.abspath(os.path.join(out, "operands.bin"))
    with open(operands, "wb") as handle:
        for k in ARRAYS:
            handle.write(ops[k].tobytes())
        handle.write(np.ascontiguousarray(want.astype(np.int32)).tobytes())
    fields = {"path": operands, "nY": want.size}
    for k in ARRAYS:
        fields[k] = CTYPES[str(ops[k].dtype)]
        fields["n" + k] = ops[k].size
    open(os.path.join(out, "tb.cpp"), "w", encoding="utf-8").write(TESTBENCH % fields)
    open(os.path.join(out, "csim.tcl"), "w", encoding="utf-8").write(CSIM_TCL)
    t0 = time.time()
    done = subprocess.run(
        ["vitis_hls", "-f", "csim.tcl"],
        cwd=out,
        capture_output=True,
        text=True,
        check=False,
    )
    res["vitis_wall_s"] = round(time.time() - t0, 3)
    found = re.search(
        r"Finished Command csim_design.*?Elapsed time: ([\d.]+) seconds", done.stdout
    )
    res["csim_design_s"] = float(found.group(1)) if found else None
    res["pass"] = "PTPU CSIM PASS" in done.stdout
    if not res["pass"]:
        errors = re.findall(r"^(?:ERROR|@E).*$", done.stdout, re.M)
        res["error"] = errors[0][:160] if errors else done.stdout[-300:]
    build = os.path.join(out, "csim_prj", "sol", "csim", "build")
    if res["pass"] and os.path.isfile(os.path.join(build, "csim.exe")):
        env = dict(
            os.environ,
            LD_LIBRARY_PATH=os.environ.get("LD_LIBRARY_PATH", "") + ":" + LIBS,
        )
        hot = []
        for _ in range(2):
            t0 = time.time()
            again = subprocess.run(
                ["./csim.exe"],
                cwd=build,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            hot.append(round(time.time() - t0, 3))
            res["hot_pass"] = "PTPU CSIM PASS" in again.stdout
        res["hot_s"] = hot
    res["load"] = [round(x, 2) for x in os.getloadavg()]
    print(json.dumps(res), flush=True)


def main():
    program, size = sys.argv[1], int(sys.argv[2])
    if sys.argv[3:4] == ["csim"]:
        csim(program, size, sys.argv[4])
    else:
        simulator(program, size)


if __name__ == "__main__":
    main()
