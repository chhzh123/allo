# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""How long a functional run of E1's array takes, two ways.

    python3 e1_simtime.py <size>                # SPMW's parallel OpenMP simulator
    python3 e1_simtime.py <size> csim <dir>     # Vitis HLS's own C simulation

Both run `gemm_int8_of(size)`, the design of every `spmw_mesh` row in E1, and
print one JSON line.

The simulator: `spmw.build(..., target="simulator")` lowers the design to
Allo's OpenMP simulator -- one thread a site, the channels as queues --
compiled for the host by LLVM. The build (elaboration, lowering, compiling)
and the run are timed apart, the run twice, and the result is checked against
numpy. `ALLO_SIM_DEPTH` raises every channel's depth, which a functional run's
results do not depend on and which saves most of its synchronisation; the
record carries the floor it ran at.

Vitis's C simulation: `spmw.build(..., target="vitis_hls")` writes the HLS
project, and `vitis_hls` runs `csim_design`, with its defaults, on that code
and a testbench that checks every result against a C reference. (SPMW's own
`mode="csim"` compiles the same code with the host compiler instead; this is
the tool's flow.) The tool's run is timed whole and by its own report.

One run a process, so that no build reuses what an earlier one loaded.
"""

import json
import os
import re
import subprocess
import sys
import time

START = time.time()
# pylint: disable=wrong-import-position
import numpy as np

import allo.spmw as spmw
from test_spmw_gemm_int8 import gemm_int8_of

IMPORTED = time.time() - START

#: A testbench for `csim_design`: the design's top takes A, B and C flat.
TESTBENCH = """#include <cstdint>
#include <cstdio>
#include "kernel.h"
#define S %d
int main() {
  static int8_t a[S * S], b[S * S];
  static int32_t c[S * S];
  for (int i = 0; i < S * S; i++) {
    a[i] = (int8_t)((i * 37 + 11) %% 256 - 128);
    b[i] = (int8_t)((i * 53 + 7) %% 256 - 128);
    c[i] = 0;
  }
  top(a, b, c);
  int bad = 0;
  for (int i = 0; i < S; i++)
    for (int j = 0; j < S; j++) {
      int32_t want = 0;
      for (int k = 0; k < S; k++)
        want += (int32_t)a[i * S + k] * (int32_t)b[k * S + j];
      bad += want != c[i * S + j];
    }
  printf(bad ? "E1 CSIM FAIL %%d of %%d\\n" : "E1 CSIM PASS %%d of %%d\\n", bad, S * S);
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


def csim(size, out):
    """Vitis HLS's own C simulation of the HLS code SPMW emits."""
    load = os.getloadavg()
    t0 = time.time()
    spmw.build(gemm_int8_of(size), target="vitis_hls", mode="csim", project=out)
    t1 = time.time()
    with open(os.path.join(out, "tb.cpp"), "w", encoding="utf-8") as handle:
        handle.write(TESTBENCH % size)
    with open(os.path.join(out, "csim.tcl"), "w", encoding="utf-8") as handle:
        handle.write(CSIM_TCL)
    done = subprocess.run(
        ["vitis_hls", "-f", "csim.tcl"],
        cwd=out,
        capture_output=True,
        text=True,
        check=False,
    )
    t2 = time.time()
    found = re.search(
        r"Finished Command csim_design.*?Elapsed time: ([\d.]+) seconds", done.stdout
    )
    print(
        json.dumps(
            {
                "size": size,
                "codegen_s": round(t1 - t0, 3),
                "vitis_wall_s": round(t2 - t1, 3),
                "csim_design_s": float(found.group(1)) if found else None,
                "rc": done.returncode,
                "pass": "E1 CSIM PASS" in done.stdout,
                "allo": os.path.dirname(os.path.dirname(spmw.__file__)),
                "load_before": [round(x, 2) for x in load],
                "load_after": [round(x, 2) for x in os.getloadavg()],
            }
        ),
        flush=True,
    )


def main():
    size = int(sys.argv[1])
    if sys.argv[2:3] == ["csim"]:
        csim(size, sys.argv[3])
        return
    rng = np.random.default_rng(3)
    a = rng.integers(-128, 128, size=(size, size)).astype(np.int8)
    b = rng.integers(-128, 128, size=(size, size)).astype(np.int8)
    want = a.astype(np.int32) @ b.astype(np.int32)
    load = os.getloadavg()
    floor = int(os.environ.get("ALLO_SIM_DEPTH", "0"))
    t0 = time.time()
    fabric = gemm_int8_of(size)
    t1 = time.time()
    module = spmw.build(fabric, target="simulator", fifo_depth_floor=floor or None)
    t2 = time.time()
    c = np.zeros((size, size), dtype=np.int32)
    module(a, b, c)
    t3 = time.time()
    first = bool(np.array_equal(c, want))
    c = np.zeros((size, size), dtype=np.int32)
    module(a, b, c)
    t4 = time.time()
    print(
        json.dumps(
            {
                "size": size,
                "sites": size * size,
                "depth_floor": floor,
                "fibers": os.environ.get("ALLO_SIM_FIBERS", "") not in ("", "0"),
                "import_s": round(IMPORTED, 3),
                "define_s": round(t1 - t0, 3),
                "build_s": round(t2 - t1, 3),
                "run_s": round(t3 - t2, 3),
                "rerun_s": round(t4 - t3, 3),
                "pass": first and bool(np.array_equal(c, want)),
                "cpus": os.cpu_count(),
                "allo": os.path.dirname(os.path.dirname(spmw.__file__)),
                "load_before": [round(x, 2) for x in load],
                "load_after": [round(x, 2) for x in os.getloadavg()],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
