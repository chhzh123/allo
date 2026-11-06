# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""parse_run.py <flow> <S> <jobs> <rep> <rc> <t0> <t1> <t2> <load0> <load1> <dir> <log>:
one JSON line for run_one.sh, with each flow's stages and its HLS jobs' times."""
import glob
import json
import os
import re
import sys
from datetime import datetime

flow, size, jobs, rep, rc, t0, t1, t2, load0, load1, out, log = sys.argv[1:]
t0, t1, t2 = float(t0), float(t1), float(t2)
rec = {
    "flow": flow,
    "size": int(size),
    "jobs": int(jobs),
    "rep": int(rep),
    "rc": int(rc),
    "start": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
    "load_start": load0,
    "load_end": load1,
    "total_s": round(t2 - t0, 1),
}


def text(path):
    try:
        with open(path, errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def num(pattern, body):
    m = re.search(pattern, body)
    return float(m.group(1)) if m else None


def job_logs(paths):
    """Each HLS job's elapsed time and its csynth / export split."""
    total, csynth, export, n = 0.0, 0.0, 0.0, 0
    for path in paths:
        body = text(path)
        el = re.findall(r"Total elapsed time: ([0-9.]+) seconds", body)
        if not el:
            continue
        n += 1
        total += float(el[-1])
        for cmd, acc in (("csynth_design", "c"), ("export_design", "e")):
            m = re.findall(
                r"Finished Command " + cmd + r" .*?Elapsed time: ([0-9.]+) seconds",
                body,
            )
            if m and acc == "c":
                csynth += float(m[-1])
            if m and acc == "e":
                export += float(m[-1])
    return {
        "hls_jobs": n,
        "hls_jobs_sum_s": round(total, 1),
        "hls_csynth_sum_s": round(csynth, 1),
        "hls_export_sum_s": round(export, 1),
    }


body = text(log + ".log")
if flow == "tapa":
    rec["frontend_s"] = round(t1 - t0, 1)  # AutoSA codegen + the kernel fixes

    def stamp(pattern):
        for line in body.splitlines():
            if pattern in line:
                m = re.match(r"[IWE](\d{4}) (\d\d:\d\d:\d\d\.\d+)", line)
                if m:
                    return datetime.strptime(
                        m.group(1) + " " + m.group(2), "%m%d %H:%M:%S.%f"
                    )
        return None

    def last_stamp():
        for line in reversed(body.splitlines()):
            m = re.match(r"[IWE](\d{4}) (\d\d:\d\d:\d\d\.\d+)", line)
            if m:
                return datetime.strptime(
                    m.group(1) + " " + m.group(2), "%m%d %H:%M:%S.%f"
                )
        return None

    s = {
        k: stamp(p)
        for k, p in [
            ("start", "tapa version"),
            ("hls", "running hls"),
            ("rtl", "extracting RTL files"),
        ]
    }
    s["end"] = last_stamp()
    if all(s.values()):
        rec["analyze_s"] = round((s["hls"] - s["start"]).total_seconds(), 1)
        rec["hls_s"] = round((s["rtl"] - s["hls"]).total_seconds(), 1)
        rec["assembly_s"] = round((s["end"] - s["rtl"]).total_seconds(), 1)
    rec.update(job_logs(glob.glob(os.path.join(out, "work/hls/*/vitis_hls.log"))))
elif flow in ("spmw_kernel", "spmw_kernel_rtl"):
    # its own staging timer leaves out elaboration; total - HLS is the whole frontend
    rec["staging_s"] = num(
        r"staged \d+ role\(s\) and \d+ feeder\(s\) in ([0-9.]+)s", body
    )
    rec["hls_s"] = num(r"HLS: ([0-9.]+)s", body)
    if rec["hls_s"] is not None:
        rec["frontend_s"] = round(rec["total_s"] - rec["hls_s"], 1)
    rec.update(
        job_logs(
            glob.glob(os.path.join(out, "roles/*/vitis_hls.log"))
            + glob.glob(os.path.join(out, "feeders/*/vitis_hls.log"))
        )
    )
elif flow in ("spmw_mesh", "spmw_mesh_rtl"):
    rec["staging_s"] = num(r" frontend_s=([0-9.]+)", body)  # elaboration + staging
    rec["hls_s"] = num(r" hls_wall_s=([0-9.]+)", body)
    if rec["hls_s"] is not None:
        rec["frontend_s"] = round(rec["total_s"] - rec["hls_s"], 1)
    rec.update(job_logs(glob.glob(os.path.join(out, "*/vitis_hls.log"))))
print(json.dumps(rec))
