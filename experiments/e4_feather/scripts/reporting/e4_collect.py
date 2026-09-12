#!/usr/bin/env python3
"""E4: gather every run into results.csv, plus reports/ and validation/ copies.

Nothing here is computed from a model: every number is read out of a run's
own check.json / result.json / Vivado reports, and a run that did not finish
gets nulls and a failure_reason.
"""

import csv
import glob
import gzip
import json
import os
import re
import shutil

ROOT = "/scratch/hc676"
RES = f"{ROOT}/spmw_eval_remaining_2026-09-06/e4_feather"
RTL_RUNS = f"{ROOT}/e4_rtl_runs"
SPMW = f"{ROOT}/e4_spmw"
PNR = f"{ROOT}/e4_pnr"

COLS = [
    "run_id", "experiment_id", "system", "workload", "numeric_semantics", "array_shape",
    "implementation_mode", "target_mhz", "status", "validation_pass", "first_output_cycles",
    "completion_cycles", "steady_interval_cycles", "weight_load_cycles", "lut", "ff", "dsp",
    "bram_18k_equiv", "uram", "wns_ns", "tns_ns", "hls_wall_s", "vivado_synth_s", "place_s",
    "route_s", "total_wall_s", "bandwidth_assumption", "report_paths", "failure_reason",
]

RTL_SEM = "uint8 operands with zero points (zpa={zpa}, zpw={zpw}); PE: 9-bit unsigned differences, unsigned 18-bit products, 32-bit accumulation; BIRRD 32-bit; legal operands have d = u - zp >= 0 (then equal to the signed int8 dot product)"
SPMW_SEM = "int8 x int8 -> int32 (the same d values as the RTL run, zero point 0; streams carry int32 words)"
RTL_BW = "weight feed: one SRAM row of N bytes read per cycle, one byte stored (pe_sel selects one PE a cycle) = N^3 cycles a tile; activations one row of N bytes per cycle; instruction one word per row; every SRAM pre-loaded, the fill is outside the measured cycles"
RTL_BW_RES = "weights resident in the PE files (one feed, then only activations: one row of N bytes per cycle, an instruction word per row)"
SPMW_BW = "every edge port fed at one token a cycle by the bench: per tile each pair-PE takes 2 x_in tokens and 2N w_in tokens (int32 words carrying int8), each switch one cmd token; the array's weight intake is N^2/2 tokens a cycle (N^2/2 useful bytes a cycle) against the RTL's one byte a cycle"
SPMW_BW_RES = "weight files and commands resident (loaded once by the fabric's feeds at launch, inside the measured cycles); per tile each pair-PE takes 2 x_in tokens at one a cycle"


def rd(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def jload(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def util_counts(path):
    vals = {}
    for line in rd(path).splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        label = cells[0]
        try:
            used = float(cells[1])
        except ValueError:
            continue
        if label.startswith("CLB LUTs") and "lut" not in vals:
            vals["lut"] = used
        elif label == "CLB Registers" and "ff" not in vals:
            vals["ff"] = used
        elif label == "DSPs" and "dsp" not in vals:
            vals["dsp"] = used
        elif label.startswith("RAMB36") and "bram36" not in vals:
            vals["bram36"] = used
        elif label.startswith("RAMB18") and "bram18" not in vals:
            vals["bram18"] = used
        elif label == "URAM" and "uram" not in vals:
            vals["uram"] = used
    if "bram36" in vals or "bram18" in vals:
        vals["bram18k"] = 2 * vals.get("bram36", 0) + vals.get("bram18", 0)
    return vals


def timing_summary(path):
    lines = rd(path).splitlines()
    for i, line in enumerate(lines):
        if "WNS(ns)" in line and "TNS(ns)" in line:
            for j in range(i + 1, min(i + 5, len(lines))):
                nums = re.findall(r"-?\d+\.\d+", lines[j])
                if len(nums) >= 2:
                    return float(nums[0]), float(nums[1])
    return None, None


def row(**kw):
    r = {c: None for c in COLS}
    r["experiment_id"] = "E4"
    r.update(kw)
    return r


def copy_into(dst_dir, paths):
    os.makedirs(dst_dir, exist_ok=True)
    out = []
    for p in paths:
        if os.path.isfile(p):
            shutil.copy(p, dst_dir)
            out.append(os.path.join(dst_dir, os.path.basename(p)))
    return out


def e4_lines(xsim_log):
    return "\n".join(l for l in rd(xsim_log).splitlines() if l.startswith("E4 ") or "FEATHER RTL" in l)


# -- RTL xsim runs ---------------------------------------------------------------


def rtl_run_row(run_id, system, run_dir, workload, feed):
    c = jload(os.path.join(run_dir, "check.json"))
    m = jload(os.path.join(run_dir, "meta.json"))
    if c is None or m is None:
        return row(run_id=run_id, system=system, workload=workload, implementation_mode="xsim", status="not_run",
                   failure_reason=f"no check.json in {run_dir}")
    N = m["N"]
    vdir = os.path.join(RES, "validation", run_id)
    copied = copy_into(vdir, [os.path.join(run_dir, f) for f in ("meta.json", "check.json", "operands.npz", "expect.npy")])
    with open(os.path.join(vdir, "xsim_e4_lines.log"), "w", encoding="utf-8") as f:
        f.write(e4_lines(os.path.join(run_dir, "xsim.log")))
    bus = os.path.join(run_dir, "bus.log")
    if os.path.isfile(bus) and os.path.getsize(bus) < 20_000_000:
        shutil.copy(bus, vdir)
    elif os.path.isfile(bus):
        with open(bus, "rb") as fi, gzip.open(os.path.join(vdir, "bus.log.gz"), "wb") as fo:
            shutil.copyfileobj(fi, fo)
    rdir = os.path.join(RES, "reports", run_id)
    copy_into(rdir, [os.path.join(run_dir, f) for f in ("xsim.log", "gen.log", "check.log")])
    ok = c.get("status") == "pass" and c.get("host_check", "pass") in ("pass", None)
    interval = c.get("steady_interval_cycles")
    if isinstance(interval, dict):
        interval_s = f"{interval['median']:.1f} (min {interval['min']}, max {interval['max']}, mean {interval['mean']:.2f})"
    else:
        interval_s = None
    # the feed length is the spacing of the select's toggles (one per feed);
    # the bench prints the first six and every 512th, so use consecutive ones
    tg = c.get("toggles") or []
    spacings = sorted({tg[i + 1][1] - tg[i][1] for i in range(len(tg) - 1) if tg[i + 1][0] == tg[i][0] + 1})
    wl = None
    if feed and spacings:
        wl = "/".join(str(s) for s in spacings) + f" per tile (measured select-toggle spacing, one toggle per feed; first toggle {c.get('first_toggle_rel_F0')} cycles after the feed start)"
    elif feed:
        wl = str(N ** 3) + " (design: one PE a cycle; single feed, no second toggle to measure)"
    elif not feed:
        wl = "0 (resident)"
    reason = None
    if not ok:
        fb = c.get("first_bad") or {}
        reason = f"{c.get('tiles_ok')}/{c.get('tiles')} tiles match; {c.get('bad_elements')} bad elements; host_check={c.get('host_check')}; first bad tile {fb.get('tile')}"
        if c.get("buffers"):
            reason += "; PE buffer dump per index k [ping, pong] of N^2 files: " + json.dumps(c["buffers"])
    return row(
        run_id=run_id, system=system, workload=workload, array_shape=f"{N}x{N}", implementation_mode="xsim",
        target_mhz=None, status="pass" if ok else "fail",
        numeric_semantics=RTL_SEM.format(zpa=m["zpa"], zpw=m["zpw"]) + f"; seed {m['seed']}; pattern {m['pattern']}; signed_equals_rtl={m.get('signed_equals_rtl')}",
        validation_pass=f"{c.get('tiles_ok')}/{c.get('tiles')} tiles bit-exact vs the RTL-arithmetic model" + (f"; host reduction vs numpy: {c.get('host_check')} ({c.get('host_bad')} bad)" if c.get("host_check") else ""),
        first_output_cycles=c.get("first_output_cycles"), completion_cycles=c.get("completion_cycles"),
        steady_interval_cycles=interval_s, weight_load_cycles=wl,
        total_wall_s=(c.get("gen_s") or 0) + (c.get("build_s") or 0) + (c.get("sim_s") or 0),
        bandwidth_assumption=RTL_BW if feed else RTL_BW_RES,
        report_paths=f"{rdir}; {vdir}; run dir {run_dir}", failure_reason=reason,
    )


def matrix_rows(rows):
    for f in sorted(glob.glob(f"{RTL_RUNS}/matrix/matrix_*_validate.json")) + sorted(glob.glob(f"{RTL_RUNS}/matrix/matrix_*_orig.json")):
        d = jload(f)
        if not d:
            continue
        tag = "feather_rtl_fixed" if "fixed" in os.path.basename(f) else "feather_rtl_original"
        N = d["N"]
        runs = d["runs"]
        npass = sum(1 for r in runs.values() if r.get("status") == "pass")
        run_id = f"rtl_{'fixed' if tag.endswith('fixed') else 'orig'}_N{N}_tile_general_{d['set']}"
        vdir = os.path.join(RES, "validation", run_id)
        os.makedirs(vdir, exist_ok=True)
        shutil.copy(f, vdir)
        # one representative run's files, and every run's check.json
        for name in runs:
            src = os.path.join(RTL_RUNS, "matrix", f"{'fixed' if tag.endswith('fixed') else 'orig'}_N{N}_{d['set']}", name)
            sub = os.path.join(vdir, name)
            copy_into(sub, [os.path.join(src, x) for x in ("meta.json", "check.json", "operands.npz", "expect.npy")])
            with open(os.path.join(sub, "xsim_e4_lines.log"), "w", encoding="utf-8") as fo:
                fo.write(e4_lines(os.path.join(src, "xsim.log")))
        firsts = {r.get("first_output_cycles") for r in runs.values() if r.get("first_output_cycles")}
        bufs = next((r.get("buffers") for r in runs.values() if r.get("buffers")), None)
        zps = sorted({n.split("_zp")[1].split("_")[0] for n in runs})
        programs = sorted({n.split("_")[1] for n in runs})
        patterns = sorted({n.split("_")[2] for n in runs})
        reason = None
        if npass != len(runs):
            fails = [n for n, r in runs.items() if r.get("status") != "pass"]
            reason = f"{len(fails)} of {len(runs)} single-tile runs fail; e.g. {fails[0]}: {runs[fails[0]].get('bad_elements')} bad elements"
            if tag.endswith("original"):
                reason += "; every PE file is split across the two buffers (even indices in ping, odd in pong, dump per k [ping, pong]): " + json.dumps(bufs)
        rows.append(row(
            run_id=run_id, system=tag, workload=f"single tile, general weights; programs {programs}; patterns {patterns}; zero points {zps}; seeds 0/1/2",
            numeric_semantics=RTL_SEM.format(zpa="{0,7,128}", zpw="{0,5,128}"), array_shape=f"{N}x{N}", implementation_mode="xsim",
            status="pass" if npass == len(runs) else "fail", validation_pass=f"{npass}/{len(runs)} runs bit-exact vs the RTL-arithmetic model",
            first_output_cycles="/".join(str(v) for v in sorted(firsts)) if firsts else None,
            completion_cycles="/".join(str(v) for v in sorted(firsts)) if firsts else None,
            weight_load_cycles=str(N ** 3), bandwidth_assumption=RTL_BW,
            total_wall_s=d.get("elapsed_s"), report_paths=f"{vdir}; matrix dir {RTL_RUNS}/matrix", failure_reason=reason,
        ))


FIELDS = {
    "feed": r"weight feed (\d+) cycles driven \(N\^3=(\d+) by design\)",
    "pass": r"activation pass (\d+) cycles",
    "first": r"first tile row at \+(\d+)",
    "total": r"feed start to last row of tile 1 (\d+)",
    "per_tile": r"(\d+) cycles a tile",
    "wrong_pe": r"weight files (\d+) PE\(s\) wrong",
    "wrong_col": r"tile columns (\d+) wrong",
    "verdict": r"FEATHER RTL N=\d+ NP=\d+: (PASS|FAIL|TIMEOUT)",
}


def restricted_rows(rows):
    for N in (4, 8, 16, 32):
        d1 = f"{RTL_RUNS}/restricted/run{N}x1"
        d2 = f"{RTL_RUNS}/restricted/run{N}x2"
        t1, t2 = rd(f"{d1}/xsim.log"), rd(f"{d2}/xsim.log")
        run_id = f"rtl_orig_N{N}_tile_restricted"
        if not t1:
            rows.append(row(run_id=run_id, system="feather_rtl_original", workload="single tile, restricted (odd-index weights zero, pass-through BIRRD)", array_shape=f"{N}x{N}", implementation_mode="xsim", status="not_run", failure_reason="restricted bench run missing"))
            continue
        g = {}
        for k, pat in FIELDS.items():
            m = re.search(pat, t1)
            g[k] = m.group(1) if m else None
        m2 = re.search(FIELDS["per_tile"], t2)
        v2 = re.search(FIELDS["verdict"], t2)
        rdir = os.path.join(RES, "reports", run_id)
        copy_into(rdir, [f"{d1}/xsim.log", f"{d2}/xsim.log"])
        os.rename(os.path.join(rdir, "xsim.log"), os.path.join(rdir, "xsim_np2.log")) if False else None
        ok = g["verdict"] == "PASS" and (v2 is None or v2.group(1) == "PASS")
        rows.append(row(
            run_id=run_id, system="feather_rtl_original",
            workload="single tile, restricted: odd-index weights zero, pass-through BIRRD (tests/dataflow/spmw/tb_feather_rtl.sv, the record's baseline); NP=2 for the resident interval",
            numeric_semantics="uint8, zero points 0, odd-index weights zero", array_shape=f"{N}x{N}", implementation_mode="xsim",
            status="pass" if ok else "fail", validation_pass=f"{g['verdict']} ({g['wrong_pe']} PE files wrong, {g['wrong_col']} columns wrong); NP=2 {v2.group(1) if v2 else None}",
            first_output_cycles=g["total"], completion_cycles=g["total"],
            steady_interval_cycles=(m2.group(1) + " (weights resident, NP=2)") if m2 else None,
            weight_load_cycles=f"{g['feed']} driven ({N ** 3} by design, +N^2 for the bench's parity sweep)",
            bandwidth_assumption=RTL_BW, report_paths=rdir,
            failure_reason=None if ok else "restricted bench did not pass",
        ))


# -- SPMW cosim runs ---------------------------------------------------------------


def spmw_row(run_id, run_dir, workload):
    r = jload(os.path.join(run_dir, "result.json"))
    if r is None:
        return row(run_id=run_id, system="spmw", workload=workload, implementation_mode="cosim", status="not_run", failure_reason=f"no result.json in {run_dir}")
    N = r["N"]
    vdir = os.path.join(RES, "validation", run_id)
    copy_into(vdir, [os.path.join(run_dir, "result.json"), os.path.join(run_dir, "operands.npz")])
    tl = os.path.join(run_dir, "sim", "tiles.log")
    if os.path.isfile(tl):
        with open(tl, "rb") as fi, gzip.open(os.path.join(vdir, "tiles.log.gz"), "wb") as fo:
            shutil.copyfileobj(fi, fo)
    rdir = os.path.join(RES, "reports", run_id)
    copy_into(rdir, [os.path.join(run_dir, "sim", "xsim.log"), os.path.join(run_dir, "sim", "xelab.log"), os.path.join(run_dir, "cost.json")])
    iv = r.get("interval")
    interval_s = f"{iv['median']:.1f} (min {iv['min']}, max {iv['max']}, mean {iv['mean']:.2f}, mean of last 32 {iv['mean_last32']:.2f})" if iv else None
    t = r.get("timing", {})
    ok = r.get("status") == "pass"
    return row(
        run_id=run_id, system="spmw", workload=workload, numeric_semantics=SPMW_SEM, array_shape=f"{N}x{N}",
        implementation_mode="cosim", target_mhz=300, status="pass" if ok else "fail",
        validation_pass=r.get("cosim"), first_output_cycles=r.get("first_tile_done"), completion_cycles=r.get("last_tile_done"),
        steady_interval_cycles=interval_s,
        weight_load_cycles=("0 (resident)" if r.get("resident") else f"{2 * N} w_in tokens per pair-PE per tile at one a cycle (inside the interval)"),
        hls_wall_s=t.get("hls_wall_s"), total_wall_s=sum(v for v in t.values() if isinstance(v, (int, float))),
        bandwidth_assumption=SPMW_BW_RES if r.get("resident") else SPMW_BW,
        report_paths=f"{rdir}; {vdir}; run dir {run_dir}",
        failure_reason=None if ok else r.get("cosim", "cosim did not pass"),
    )


# -- P&R ---------------------------------------------------------------------------


def pnr_rtl_row(variant, N):
    d = f"{PNR}/{variant}_{N}"
    system = "feather_rtl_original" if variant == "orig" else "feather_rtl_fixed"
    run_id = f"pnr_rtl_{variant}_N{N}"
    res = rd(f"{d}/result.txt")
    if not res:
        return row(run_id=run_id, system=system, workload="feather_top OOC (shipped defaults: depth-4 SRAM reg arrays)", array_shape=f"{N}x{N}", implementation_mode="pnr_ooc", target_mhz=300, status="not_run", failure_reason="not finished" if os.path.isdir(d) else "not started")
    stages = dict(re.findall(r"E4 STAGE (\w+) ([\d.]+)", res))
    wns = re.search(r"E4 WNS ([-\d.]+)", res)
    unrouted = re.search(r"E4 UNROUTED (\d+)", res)
    secs = re.search(r"seconds=(\d+)", res)
    ok = "IMPLEMENTATION OK" in res
    u = util_counts(f"{d}/util.rpt")
    w2, tns = timing_summary(f"{d}/timing.rpt")
    rdir = os.path.join(RES, "reports", run_id)
    copy_into(rdir, [f"{d}/{x}" for x in ("result.txt", "util.rpt", "util_synth.rpt", "util_hier.rpt", "timing.rpt", "route.rpt", "impl.tcl")])
    with open(os.path.join(rdir, "viv_tail.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(rd(f"{d}/viv.log").splitlines()[-60:]))
    errs = [l for l in rd(f"{d}/viv.log").splitlines() if l.startswith("ERROR")][:3]
    reason = None
    if not ok:
        reason = "resource_fail: " + " | ".join(errs) if any("utilization" in e.lower() or "resource" in e.lower() or "place" in e.lower() for e in errs) else ("Vivado failed: " + " | ".join(errs))
    return row(
        run_id=run_id, system=system, workload="feather_top OOC P&R, shipped defaults (depth-4 SRAM reg arrays, N x N NEST + BIRRD + controller)",
        numeric_semantics="uint8 x uint8 -> 32-bit, as shipped", array_shape=f"{N}x{N}", implementation_mode="pnr_ooc", target_mhz=300,
        status="pass" if ok else ("resource_fail" if reason and reason.startswith("resource") else "fail"),
        validation_pass="timing met" if (ok and wns and float(wns.group(1)) >= 0) else ("timing NOT met" if ok else None),
        lut=u.get("lut"), ff=u.get("ff"), dsp=u.get("dsp"), bram_18k_equiv=u.get("bram18k"), uram=u.get("uram"),
        wns_ns=float(wns.group(1)) if wns else w2, tns_ns=tns,
        vivado_synth_s=stages.get("synth"), place_s=stages.get("place"), route_s=stages.get("route"),
        total_wall_s=int(secs.group(1)) if secs else None,
        bandwidth_assumption=f"unrouted nets: {unrouted.group(1) if unrouted else None}; stages (s): {stages}",
        report_paths=rdir, failure_reason=reason,
    )


def pnr_spmw_row(design, N):
    d = f"{PNR}/spmw_{design}_{N}"
    run_id = f"pnr_spmw_{design}_N{N}"
    done = rd(f"{d}/E4_DONE")
    log = rd(f"{d}/build.log")
    label = {"feather-stream": "feather_stream (streamed operands, int32 word files) --pnr", "feather": "feather (single tile, resident int8 files) --pnr"}[design]
    if not done:
        return row(run_id=run_id, system="spmw", workload=label, array_shape=f"{N}x{N}", implementation_mode="pnr_ooc", target_mhz=300, status="not_run", failure_reason="not finished" if os.path.isdir(d) else "not started")
    ok = "IMPLEMENTATION OK" in log or "Vivado implemented" in log
    stages = dict(re.findall(r"^\s+stage (\w+) ([\d.]+)", log, re.M))
    unrouted = re.search(r"^\s+unrouted (\d+)", log, re.M)
    hls = re.search(r"HLS: ([\d.]+)s wall", log)
    wns = re.search(r"WNS ([-+][\d.]+) ns", log)
    secs = re.search(r"seconds=(\d+)", done)
    u = util_counts(f"{d}/util.rpt")
    w2, tns = timing_summary(f"{d}/timing.rpt")
    rdir = os.path.join(RES, "reports", run_id)
    copy_into(rdir, [f"{d}/{x}" for x in ("build.log", "util.rpt", "util_synth.rpt", "timing.rpt", "route.rpt", "cost.json", "E4_DONE", "assemble.tcl")])
    errs = [l for l in log.splitlines() if "ERROR" in l][:3]
    reason = None if ok else ("resource_fail: " + " | ".join(errs) if any("utilization" in e.lower() for e in errs) else "build failed: " + " | ".join(errs or log.splitlines()[-3:]))
    return row(
        run_id=run_id, system="spmw", workload=label + " (spmw_harness top: LFSR drivers around spmw_top, as the SPMW flow places every array)",
        numeric_semantics="int8 x int8 -> int32", array_shape=f"{N}x{N}", implementation_mode="pnr_ooc", target_mhz=300,
        status="pass" if ok else ("resource_fail" if reason and reason.startswith("resource") else "fail"),
        validation_pass=("timing met" if wns and float(wns.group(1)) >= 0 else ("timing NOT met" if wns else None)) if ok else None,
        lut=u.get("lut"), ff=u.get("ff"), dsp=u.get("dsp"), bram_18k_equiv=u.get("bram18k"), uram=u.get("uram"),
        wns_ns=float(wns.group(1)) if wns else w2, tns_ns=tns,
        hls_wall_s=float(hls.group(1)) if hls else None, vivado_synth_s=stages.get("synth"), place_s=stages.get("place"), route_s=stages.get("route"),
        total_wall_s=int(secs.group(1)) if secs else None,
        bandwidth_assumption=f"unrouted nets: {unrouted.group(1) if unrouted else None}; stages (s): {stages}",
        report_paths=rdir, failure_reason=reason,
    )


def main():
    os.makedirs(f"{RES}/reports", exist_ok=True)
    os.makedirs(f"{RES}/validation", exist_ok=True)
    rows = []
    restricted_rows(rows)
    matrix_rows(rows)
    for N, wl, zps in ((8, "gemm128", "7,5"), (16, "gemm128", "0,0")):
        rows.append(rtl_run_row(f"rtl_fixed_N{N}_{wl}_feed", "feather_rtl_fixed", f"{RTL_RUNS}/wl_gemm_N{N}_m0", f"GEMM M=N=K=128, drivers' tiling (Mt={N // 2}, Kt={2 * N}, Nt={N}), tiles = {(128 // (N // 2)) * (128 // (2 * N)) * (128 // N)}, a weight feed per tile", True))
        rows.append(rtl_run_row(f"rtl_fixed_N{N}_{wl}_resident", "feather_rtl_fixed", f"{RTL_RUNS}/wl_gemm_N{N}_m1", f"GEMM M=N=K=128 tiles' activations and programs through one resident weight set (tile 0's), {(128 // (N // 2)) * (128 // (2 * N)) * (128 // N)} passes back to back", False))
    rows.append(rtl_run_row("rtl_fixed_N4_conv_feed", "feather_rtl_fixed", f"{RTL_RUNS}/wl_conv_N4_m0", "conv 16x16x64 -> 64, 3x3, stride 1, pad 1, NHWC; drivers' tiling with RS padded 9 -> 12: tiles = 256 * 16 * 16 * 3 = 196608, a weight feed per tile", True))
    rows.append(rtl_run_row("rtl_fixed_N4_conv_resident", "feather_rtl_fixed", f"{RTL_RUNS}/wl_conv_N4_m1", "conv tiles' activations and programs through one resident weight set (tile 0's), 196608 passes back to back", False))
    for N, wl in ((8, "gemm128"), (16, "gemm128")):
        rows.append(spmw_row(f"spmw_N{N}_{wl}_stream", f"{SPMW}/wl_gemm_N{N}_stream", f"GEMM M=N=K=128, drivers' tiling, feather_stream(N, N, NT) with NT = {(128 // (N // 2)) * (128 // (2 * N)) * (128 // N)} tiles in one launch, every operand streamed"))
        rows.append(spmw_row(f"spmw_N{N}_{wl}_resident", f"{SPMW}/wl_gemm_N{N}_x", f"GEMM tiles' activations through resident weights and commands (tile 0's), feather_stream_x(N, N, NT), NT = {(128 // (N // 2)) * (128 // (2 * N)) * (128 // N)}"))
    rows.append(spmw_row("spmw_N4_conv_stream", f"{SPMW}/wl_conv_N4_stream", "conv 16x16x64 -> 64 3x3 pad 1, drivers' tiling (RS padded to 12), feather_stream(4, 4, 196608), every operand streamed"))
    rows.append(spmw_row("spmw_N4_conv_resident", f"{SPMW}/wl_conv_N4_x", "conv tiles' activations through resident weights and commands (tile 0's), feather_stream_x(4, 4, 196608)"))
    for N in (4, 8, 16, 32):
        rows.append(pnr_rtl_row("orig", N))
        rows.append(pnr_rtl_row("fixed", N))
        rows.append(pnr_spmw_row("feather-stream", N))
        rows.append(pnr_spmw_row("feather", N))
    with open(f"{RES}/results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    print(f"{len(rows)} rows -> {RES}/results.csv")
    for r in rows:
        print(f"  {r['run_id']:<40} {r['status']:<14} {str(r['first_output_cycles'])[:12]:<12} {str(r['completion_cycles'])[:12]:<12} {str(r['steady_interval_cycles'])[:40]:<40} lut={r['lut']} wns={r['wns_ns']}")
    tables(rows)
    # the scripts and the diff, for reproduction
    sdir = os.path.join(RES, "scripts")
    os.makedirs(sdir, exist_ok=True)
    for f in glob.glob("/scratch/hc676/e4_work/*"):
        if os.path.isfile(f):
            shutil.copy(f, sdir)
    shutil.copy("/scratch/hc676/e4_feather_fixed/feather_controller.diff", sdir)


def fmt(v, digits=0):
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}" if digits else f"{v:.0f}"
    return str(v)


def tables(rows):
    by = {r["run_id"]: r for r in rows}
    out = []
    out.append("### Table A: single tiles, general weights, xsim (validation matrices)\n")
    out.append("| system | N | runs (programs x patterns x zero points x seeds) | bit-exact | tile latency, cycles from feed start | note |")
    out.append("|---|---|---|---|---|---|")
    for N in (4, 8, 16):
        for tag, system in (("orig", "original RTL"), ("fixed", "corrected RTL")):
            r = by.get(f"rtl_{tag}_N{N}_tile_general_{'orig' if tag == 'orig' else 'validate'}")
            if not r:
                continue
            note = "every PE file split across ping/pong (see the buffer dump)" if tag == "orig" else ("all pass" if r["status"] == "pass" else r["failure_reason"])
            out.append(f"| {system} | {N} | {r['validation_pass'].split(' runs')[0].split('/')[1]} | {r['validation_pass'].split(' ')[0]} | {fmt(r['first_output_cycles'])} | {note} |")
        r = by.get(f"rtl_orig_N{N}_tile_restricted")
        if r:
            out.append(f"| original RTL, restricted bench (record) | {N} | 1 (odd-index weights zero) | {r['validation_pass']} | {fmt(r['first_output_cycles'])} | feed driven {r['weight_load_cycles']} |")
    r = by.get("rtl_orig_N32_tile_restricted")
    if r:
        out.append(f"| original RTL, restricted bench (record) | 32 | 1 (odd-index weights zero) | {r['validation_pass']} | {fmt(r['first_output_cycles'])} | feed driven {r['weight_load_cycles']} |")
    out.append("\n### Table B: whole workloads, measured completion events (cycles)\n")
    out.append("| workload | N | system | mode | tiles | first tile done | last tile done (whole workload) | steady-state interval per tile | weight load per tile | validation |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for wl, N, T in (("GEMM 128^3", 8, 4096), ("GEMM 128^3", 16, 512), ("conv 16x16x64->64 3x3", 4, 196608)):
        key = "gemm128" if wl.startswith("GEMM") else "conv"
        for rid, system, mode in (
            (f"rtl_fixed_N{N}_{key}_feed", "corrected RTL", "a weight feed per tile"),
            (f"spmw_N{N}_{key}_stream", "SPMW feather_stream", "every operand streamed per tile"),
            (f"rtl_fixed_N{N}_{key}_resident", "corrected RTL", "weights resident"),
            (f"spmw_N{N}_{key}_resident", "SPMW feather_stream_x", "weights resident"),
        ):
            r = by.get(rid)
            if not r:
                continue
            out.append(f"| {wl} | {N} | {system} | {mode} | {T} | {fmt(r['first_output_cycles'])} | {fmt(r['completion_cycles'])} | {fmt(r['steady_interval_cycles'])} | {fmt(r['weight_load_cycles'])} | {r['status']}: {fmt(r['validation_pass'])[:80]} |")
    out.append("\n### Table C: place-and-route on xcu280 at 3.333 ns (300 MHz)\n")
    out.append("| design | N | status | LUT | FF | DSP | BRAM (18k eq.) | URAM | WNS ns | TNS ns | unrouted | synth s | place s | route s | total s | HLS s |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for N in (4, 8, 16, 32):
        for rid, label in ((f"pnr_rtl_orig_N{N}", "original RTL, feather_top OOC"), (f"pnr_rtl_fixed_N{N}", "corrected RTL, feather_top OOC"), (f"pnr_spmw_feather-stream_N{N}", "SPMW feather_stream (spmw_harness)"), (f"pnr_spmw_feather_N{N}", "SPMW feather, single tile (spmw_harness)")):
            r = by.get(rid)
            if not r:
                continue
            unr = re.search(r"unrouted nets: (\S+);", r["bandwidth_assumption"] or "")
            out.append(f"| {label} | {N} | {r['status']} | {fmt(r['lut'])} | {fmt(r['ff'])} | {fmt(r['dsp'])} | {fmt(r['bram_18k_equiv'])} | {fmt(r['uram'])} | {fmt(r['wns_ns'], 3)} | {fmt(r['tns_ns'], 3)} | {unr.group(1) if unr else '-'} | {fmt(r['vivado_synth_s'], 1)} | {fmt(r['place_s'], 1)} | {fmt(r['route_s'], 1)} | {fmt(r['total_wall_s'])} | {fmt(r['hls_wall_s'], 1)} |")
    text = "\n".join(out) + "\n"
    with open(os.path.join(RES, "README_tables.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
