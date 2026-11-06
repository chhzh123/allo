# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time Allo's frontend for E1's GEMM: the steps of e1_allo_gen.py that the
Vitis HLS project needs (import, customize + schedule, build(vitis_hls)).

usage: fe_allo.py S PROJECT_DIR GEN_DIR   (FE_T0 = the shell's start time)
E1's generator also emits an un-postprocessed copy, test inputs and a
testbench; those are inspection and cosimulation, so they are not timed here.
"""
import json
import os
import sys
import time

T0 = float(os.environ["FE_T0"])


def stamp():
    return round(time.time() - T0, 3)


S = int(sys.argv[1])
out = os.path.abspath(sys.argv[2])
sys.path.insert(0, sys.argv[3])
t_start = stamp()
import e1_allo_gen as gen  # numpy, allo, allo.library.systolic

t_import = stamp()
os.makedirs(out, exist_ok=True)
s = gen.build_schedule(S, S, S, S)
t_schedule = stamp()
s.build(target="vitis_hls", mode="csyn", project=out)
t_build = stamp()
print(
    "FE_ALLO",
    json.dumps(
        {
            "interpreter_s": t_start,
            "import_s": round(t_import - t_start, 3),
            "schedule_s": round(t_schedule - t_import, 3),
            "build_s": round(t_build - t_schedule, 3),
            "frontend_s": t_build,
        }
    ),
)
