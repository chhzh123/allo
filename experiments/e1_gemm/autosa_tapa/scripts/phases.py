# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""phases.py <build dir>...: split a tapa compile log into its phases and sum the
per-task HLS job times (each task's own vitis_hls.log), the serial-cost estimate
E5 uses for SPMW."""
import glob, os, re, sys
from datetime import datetime


def t(line):
    m = re.match(r"[IWE](\d{4}) (\d\d:\d\d:\d\d\.\d+)", line)
    return (
        datetime.strptime(m.group(1) + " " + m.group(2), "%m%d %H:%M:%S.%f")
        if m
        else None
    )


for d in sys.argv[1:]:
    lines = open(os.path.join(d, "compile.log")).read().splitlines()
    stamp = {}
    for key, pat in [
        ("start", "tapa version"),
        ("hls", "running hls"),
        ("rtl", "extracting RTL files"),
        ("pack", "packaging RTL code"),
        ("end", "generated the v++ xo file"),
    ]:
        for line in lines:
            if pat in line and t(line):
                stamp[key] = t(line)
                break
    sec = lambda a, b: (stamp[b] - stamp[a]).total_seconds()
    jobs = []
    for log in glob.glob(os.path.join(d, "work/hls/*/vitis_hls.log")):
        m = re.findall(
            r"Total elapsed time: ([0-9.]+) seconds", open(log, errors="replace").read()
        )
        if m:
            jobs.append(float(m[-1]))
    print(
        "%-10s analyze %5.1f  hls %6.1f  rtl %5.1f  pack %5.1f  total %6.1f | tasks %3d  sum of HLS jobs %7.1f  longest %6.1f"
        % (
            os.path.basename(d),
            sec("start", "hls"),
            sec("hls", "rtl"),
            sec("rtl", "pack"),
            sec("pack", "end"),
            sec("start", "end"),
            len(jobs),
            sum(jobs),
            max(jobs),
        )
    )
