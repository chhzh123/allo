#!/usr/bin/env python3
"""E1/E2 SPMW collector: the sweep's report directories -> results.csv/json.

    python3 e12_collect.py <bundle_dir> <experiment_id> <system> <axis> <workload_fmt>

<bundle_dir>/reports/<AXIS><n>_<mode>/ holds build.log (the build's stdout),
xsim.log for cosims, and cost.json/util.rpt/timing.rpt for P&R runs. The
axis letter is S (array side) or N (FFT points). Nothing is estimated: a
value the reports do not hold is left empty with a failure_reason.
"""
import csv
import glob
import json
import os
import re
import sys


def grab(pattern, text, cast=float, default=None):
    m = re.search(pattern, text, re.M)
    return cast(m.group(1)) if m else default


def util(path):
    if not os.path.isfile(path):
        return {}
    text = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for key, name in (("lut", "CLB LUTs"), ("ff", "CLB Registers"), ("bram_tiles", "Block RAM Tile"), ("uram", "URAM"), ("dsp", "DSPs")):
        m = re.search(r"^\|\s*%s\*?\s*\|\s*([\d.]+)\s*\|" % re.escape(name), text, re.M)
        if m:
            out[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    m36 = re.search(r"^\|\s*RAMB36E2\s*\|\s*(\d+)", text, re.M)
    m18 = re.search(r"^\|\s*RAMB18E2\s*\|\s*(\d+)", text, re.M)
    b36 = int(m36.group(1)) if m36 else 0
    b18 = int(m18.group(1)) if m18 else 0
    out["bram_36k"] = b36
    out["bram_18k"] = b18
    out["bram_18k_equiv"] = 2 * b36 + b18
    return out


def timing(path):
    if not os.path.isfile(path):
        return {}
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)", text, re.S)
    if not m:
        return {}
    return {"wns_ns": float(m.group(1)), "tns_ns": float(m.group(2)), "failing_endpoints": int(m.group(3)), "whs_ns": float(m.group(5))}


def main():
    bundle, exp, system, axis, workload_fmt = sys.argv[1:6]
    rows = []
    for d in sorted(glob.glob(os.path.join(bundle, "reports", axis + "*_*"))):
        tag = os.path.basename(d)
        m = re.match(axis + r"(\d+)_(\w+)$", tag)
        if not m:
            continue
        n, mode = int(m.group(1)), m.group(2)
        log_path = os.path.join(d, "build.log")
        log = open(log_path, encoding="utf-8", errors="replace").read() if os.path.isfile(log_path) else ""
        row = {
            "run_id": tag, "experiment_id": exp, "system": system,
            "workload": workload_fmt.format(n=n), "size": n, "mode": mode,
            "implementation_mode": {"cosim": "rtl_cosim", "memcosim": "rtl_cosim_memory", "pnr": "pnr_ooc"}.get(mode, mode),
            "target_mhz": 300, "status": "", "validation_pass": "", "first_output_cycles": "",
            "completion_cycles": "", "tokens_checked": "", "lut": "", "ff": "", "dsp": "",
            "bram_18k_equiv": "", "bram_36k": "", "bram_18k": "", "uram": "", "wns_ns": "", "tns_ns": "",
            "achieved_ns": "", "hls_wall_s": "", "hls_sum_job_s": "", "vivado_total_s": "",
            "synth_s": "", "place_s": "", "route_s": "", "unrouted": "", "roles": "", "instances": "",
            "report_paths": f"reports/{tag}", "failure_reason": "",
        }
        cost = grab(r'^\S+: (\{.*\})$', log, cast=str)
        if cost:
            try:
                c = json.loads(cost)
                row["roles"], row["instances"] = c.get("roles", ""), c.get("instances", "")
            except json.JSONDecodeError:
                pass
        row["hls_wall_s"] = grab(r"^HLS: ([\d.]+)s wall", log)
        row["hls_sum_job_s"] = grab(r"\(([\d.]+)s of CPU work", log)
        if mode in ("cosim", "memcosim"):
            m2 = re.search(r"SPMW COSIM (PASS|FAIL) \((\d+)/(\d+) tokens, (\d+) errors\)", log)
            if m2:
                row["validation_pass"] = m2.group(1) == "PASS"
                row["tokens_checked"] = int(m2.group(3))
                row["status"] = "pass" if m2.group(1) == "PASS" else "functional_fail"
                if m2.group(1) == "FAIL":
                    row["failure_reason"] = f"{m2.group(4)} of {m2.group(3)} tokens differ"
            else:
                row["status"] = "unsupported"
                if "Module <" in log or "xelab failed" in log:
                    row["failure_reason"] = ("the out-of-context array testbench drives streams and has no AXI "
                                             "memory model; memory-mapped movers are simulated at kernel level "
                                             "(spmw_kernel_sim.py) and validated on the board")
                else:
                    reason = grab(r"^(SystemExit: .*|.*no mover for.*|ERROR: .*)$", log, cast=str)
                    row["failure_reason"] = (reason or "no cosim verdict in build.log")[:200]
            row["completion_cycles"] = grab(r"SPMW CYCLES total=(\d+)", log, cast=int)
            row["first_output_cycles"] = grab(r"first_out=(\d+)", log, cast=int)
        elif mode == "memsim":
            # The kernel-level simulation: the packaged kernel (roles, AXI
            # movers, control) against behavioural AXI RAM; cycles from the
            # testbench's start/done timestamps (ps) at its 4 ns clock, the
            # done seen through AXI-lite polls (one poll ~8 cycles).
            sim_path = os.path.join(d, os.path.basename(d).replace("_memsim", "") and "e1_memsim_%d.sim.log" % n)
            sim = open(sim_path, encoding="utf-8", errors="replace").read() if os.path.isfile(sim_path) else ""
            t0 = grab(r"SPMW TB: starting at (\d+)", sim, cast=int)
            t1 = grab(r"SPMW TB: done after (?:\d+) poll\(s\) at (\d+)", sim, cast=int)
            bad = grab(r"SPMW TB RESULT (\d+) of (\d+) byte", sim, cast=int)
            total_bytes = grab(r"SPMW TB RESULT \d+ of (\d+) byte", sim, cast=int)
            if t0 is not None and t1 is not None:
                row["completion_cycles"] = (t1 - t0) // 4000
            row["tokens_checked"] = total_bytes
            row["validation_pass"] = bad == 0 if bad is not None else ""
            row["status"] = "pass" if bad == 0 else ("functional_fail" if bad else "unsupported")
            if bad:
                row["failure_reason"] = f"{bad} of {total_bytes} drain bytes differ"
            elif bad is None:
                row["failure_reason"] = "no testbench verdict in the sim log"
            pkg = os.path.join(d, "e1_memsim_%d.package.log" % n)
            plog = open(pkg, encoding="utf-8", errors="replace").read() if os.path.isfile(pkg) else ""
            row["hls_wall_s"] = grab(r"^\s*HLS: ([\d.]+)s", plog)
            row["implementation_mode"] = "kernel_sim_axi_ram"
            masters = grab(r"(\d+) AXI master", plog, cast=int)
            row["failure_reason"] = row["failure_reason"] or ""
            row["roles"] = row["roles"] or ""
            if masters is not None:
                row["workload"] += f"; kernel with {masters} AXI masters"
        elif mode == "pnr":
            u = util(os.path.join(d, "util.rpt"))
            t = timing(os.path.join(d, "timing.rpt"))
            row.update({k: v for k, v in u.items() if k in row})
            row.update({k: v for k, v in t.items() if k in row})
            row["synth_s"] = grab(r"stage synth ([\d.]+)", log)
            row["place_s"] = grab(r"stage place ([\d.]+)", log)
            row["route_s"] = grab(r"stage route ([\d.]+)", log)
            row["unrouted"] = grab(r"unrouted (\d+)", log, cast=int)
            row["vivado_total_s"] = grab(r"Vivado implemented \d+ instances in ([\d.]+)s", log)
            row["achieved_ns"] = grab(r"array clock: ([\d.]+) ns achieved", log)
            if t:
                row["status"] = "pass" if t["wns_ns"] >= 0 and (row["unrouted"] in (0, None)) else "timing_fail"
                if row["unrouted"]:
                    row["status"] = "resource_fail"
                    row["failure_reason"] = f"{row['unrouted']} unrouted nets"
            else:
                row["status"] = "timeout" if not log.strip().endswith(")") and "Vivado implemented" not in log else "unsupported"
                row["failure_reason"] = "no timing report (the run did not finish P&R)"
        rows.append(row)
    with open(os.path.join(bundle, "results.csv"), "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(bundle, "results.json"), "w", encoding="utf-8") as h:
        json.dump(rows, h, indent=1)
    for r in rows:
        print(f"{r['run_id']:>16} {r['status']:>18} cyc={r['completion_cycles']!s:>6} first={r['first_output_cycles']!s:>5} "
              f"LUT={r['lut']!s:>7} FF={r['ff']!s:>7} DSP={r['dsp']!s:>5} BRAM18={r['bram_18k_equiv']!s:>4} WNS={r['wns_ns']!s:>7} "
              f"hls={r['hls_wall_s']!s:>6} viv={r['vivado_total_s']!s:>7} {r['failure_reason'][:60]}")


if __name__ == "__main__":
    main()
