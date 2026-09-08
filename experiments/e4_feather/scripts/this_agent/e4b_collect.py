#!/usr/bin/env python3
"""E4: gather every run of the package -- the previous agent's (e4_*) and this
one's (e4b_*) -- into results.csv / results.json, plus reports/<run_id>/ and
validation/<run_id>/ copies.

Nothing here is computed from a model: every number is read out of a run's own
check.json / result.json / Vivado reports; a run that did not finish gets empty
cells and a failure_reason. The `source` column says whose run it is.
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
PREV_RTL = f"{ROOT}/e4_rtl_runs"
PREV_SPMW = f"{ROOT}/e4_spmw"
NEW_RTL = f"{ROOT}/e4b_rtl_runs"
NEW_SPMW = f"{ROOT}/e4b_spmw"
PNR = f"{ROOT}/e4_pnr"
PREV = "previous agent (e4_* dirs)"
MINE = "this agent (e4b_* dirs)"

COLS = [
    "run_id", "experiment_id", "system", "variant", "workload", "array_size", "weight_mode",
    "reorder_program", "implementation_mode", "target_mhz", "status", "validation_pass", "tiles",
    "first_output_cycles", "completion_cycles", "cycles_per_tile", "lut", "ff", "dsp",
    "bram_18k_equiv", "uram", "wns_ns", "tns_ns", "unrouted", "hls_wall_s", "total_wall_s",
    "report_paths", "failure_reason", "source",
]

GEMM_TILES = {4: 32768, 8: 4096, 16: 512}
CONV_TILES = {4: 196608, 8: 32768, 16: 4096}


def gemm_wl(N):
    return (f"gemm M=N=K=128 int8, drivers' tiling Mt={N // 2} Kt={2 * N} Nt={N} "
            f"(loop n,m,k), {GEMM_TILES[N]} tiles, host accumulates the K partials")


def conv_wl(N):
    rs_pad = ((9 + N - 1) // N) * N
    return (f"conv 16x16x64 -> 64ch 3x3 s1 p1 NHWC int8, drivers' tiling (RS 9 padded to {rs_pad}, "
            f"tile = [{N} taps x {N} channels] x {N} out-channels), {CONV_TILES[N]} tiles, host accumulates over channel and tap blocks")


def gemm_prog(N):
    return f"drivers' GEMM layout program for AW={N} (examples/feather/gemm.py), one program for every tile"


def conv_prog(N):
    if N == 4:
        return "drivers' four conv programs (examples/feather/convolution.py), program (p*Q+q) mod 4 per tile"
    return f"conv reduce programs for AW={N} found by e4_feather_gen.find_reduce_program (all {N} column sums into column (p*Q+q) mod {N}), verified vs feather_ref and the RTL model"


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
    for k in ("lut", "ff", "dsp", "uram", "bram18k"):
        if k in vals:
            vals[k] = int(vals[k])
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
    r["target_mhz"] = 300
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
    return "\n".join(l for l in rd(xsim_log).splitlines() if l.startswith("E4 ") or "FEATHER RTL" in l or l.startswith("SPMW "))


def interval_text(iv):
    if not isinstance(iv, dict):
        return None
    s = f"median {iv['median']:.0f} (min {iv['min']}, max {iv['max']}, mean {iv['mean']:.3f}"
    if "mean_last32" in iv:
        s += f", mean of last 32 {iv['mean_last32']:.2f}"
    return s + ")"


# -- RTL xsim runs ---------------------------------------------------------------


def rtl_run_row(run_id, run_dir, variant, workload, reorder, weight_mode, source, feed):
    c = jload(os.path.join(run_dir, "check.json"))
    m = jload(os.path.join(run_dir, "meta.json"))
    base = dict(run_id=run_id, system="FEATHER RTL", variant=variant, workload=workload, reorder_program=reorder,
                weight_mode=weight_mode, implementation_mode="rtl_sim", source=source)
    if c is None or m is None:
        running = os.path.isdir(run_dir)
        return row(**base, status="not_run", failure_reason=("run not finished (no check.json yet)" if running else "not started"))
    N = m["N"]
    vdir = os.path.join(RES, "validation", run_id)
    copy_into(vdir, [os.path.join(run_dir, f) for f in ("meta.json", "check.json")])
    rdir = os.path.join(RES, "reports", run_id)
    copy_into(rdir, [os.path.join(run_dir, f) for f in ("meta.json", "check.json", "check.log", "gen.log")])
    with open(os.path.join(rdir, "xsim_e4_lines.log"), "w", encoding="utf-8") as f:
        f.write(e4_lines(os.path.join(run_dir, "xsim.log")))
    bus = os.path.join(run_dir, "bus.log")
    if os.path.isfile(bus) and os.path.getsize(bus) < 5_000_000:
        shutil.copy(bus, rdir)
    ok = c.get("status") == "pass" and c.get("host_check") in ("pass", None)
    host = c.get("host_check")
    signed = m.get("signed_equals_rtl")
    val = f"{c.get('tiles_ok')}/{c.get('tiles')} tiles bit-exact vs the RTL-arithmetic model"
    if host:
        val += f"; host reduction vs numpy: {host} ({c.get('host_bad')} bad elements)"
    val += f"; signed_equals_rtl={signed}"
    reason = None
    if not ok:
        fb = c.get("first_bad") or {}
        reason = f"{c.get('tiles_ok')}/{c.get('tiles')} tiles match the model; {c.get('bad_elements')} bad elements; host_check={host}; first bad tile {fb.get('tile')}"
        if c.get("buffers"):
            reason += "; PE buffer dump per weight index k [ping, pong] of N^2 files: " + json.dumps(c["buffers"])
    iv = c.get("steady_interval_cycles")
    return row(
        **base, array_size=f"{N}x{N}", status="pass" if ok else "fail", validation_pass=val, tiles=c.get("tiles"),
        first_output_cycles=c.get("first_output_cycles"), completion_cycles=c.get("completion_cycles"),
        cycles_per_tile=(iv["median"] if isinstance(iv, dict) else None),
        total_wall_s=round((c.get("gen_s") or 0) + (c.get("build_s") or 0) + (c.get("sim_s") or 0), 1),
        report_paths=f"{rdir}; {vdir}; run dir {run_dir}", failure_reason=reason,
    )


def matrix_rows(rows):
    for f in sorted(glob.glob(f"{PREV_RTL}/matrix/matrix_*_validate.json")) + sorted(glob.glob(f"{PREV_RTL}/matrix/matrix_*_orig.json")):
        d = jload(f)
        if not d:
            continue
        fixed = "fixed" in os.path.basename(f)
        N = d["N"]
        set_dir = f"{PREV_RTL}/matrix/{'fixed' if fixed else 'orig'}_N{N}_{d['set']}"
        # the per-run check.json files are the truth (the summaries were written
        # before some mode-1 reruns)
        runs = {}
        for name in sorted(d["runs"]):
            c = jload(os.path.join(set_dir, name, "check.json"))
            runs[name] = c if c else d["runs"][name]
        npass = sum(1 for r in runs.values() if r.get("status") == "pass")
        run_id = f"rtl_{'fixed' if fixed else 'orig'}_N{N}_single_tiles_{d['set']}"
        vdir = os.path.join(RES, "validation", run_id)
        os.makedirs(vdir, exist_ok=True)
        shutil.copy(f, vdir)
        for name in runs:
            sub = os.path.join(vdir, name)
            copy_into(sub, [os.path.join(set_dir, name, x) for x in ("meta.json", "check.json")])
        rdir = os.path.join(RES, "reports", run_id)
        os.makedirs(rdir, exist_ok=True)
        with open(os.path.join(rdir, "summary.json"), "w", encoding="utf-8") as fo:
            json.dump({"set_dir": set_dir, "runs": runs}, fo, indent=1, default=int)
        firsts = sorted({r.get("first_output_cycles") for r in runs.values() if r.get("first_output_cycles")})
        bufs = next((r.get("buffers") for r in runs.values() if r.get("buffers")), None)
        zps = sorted({n.split("_zp")[1].split("_")[0] for n in runs})
        programs = sorted({n.split("_")[1] for n in runs})
        patterns = sorted({n.split("_")[2] for n in runs})
        reason = None
        if npass != len(runs):
            fails = [n for n, r in runs.items() if r.get("status") != "pass"]
            reason = f"{len(fails)} of {len(runs)} single-tile runs fail; e.g. {fails[0]}: {runs[fails[0]].get('bad_elements')} bad elements"
            if not fixed:
                reason += "; every PE file is split across ping/pong (even indices in one buffer, odd in the other; dump per k [ping, pong]): " + json.dumps(bufs)
        rows.append(row(
            run_id=run_id, system="FEATHER RTL", variant="corrected controller" if fixed else "shipped controller",
            workload=f"single tiles: {len(runs)} runs = programs {programs} x patterns {patterns} x zero points {zps} x seeds 0/1/2, one tile each, weights resident",
            array_size=f"{N}x{N}", weight_mode="general (single tiles: small / full-range / sparse patterns, zero points 0, 7/5, 128)",
            reorder_program="GEMM layout program, random programs" + (", the four conv programs" if N == 4 else ""),
            implementation_mode="rtl_sim", status="pass" if npass == len(runs) else "fail",
            validation_pass=f"{npass}/{len(runs)} runs bit-exact vs the RTL-arithmetic model", tiles=1,
            first_output_cycles="/".join(str(v) for v in firsts) if firsts else None,
            completion_cycles="/".join(str(v) for v in firsts) if firsts else None,
            total_wall_s=d.get("elapsed_s"), report_paths=f"{rdir}; {vdir}; matrix dir {set_dir}", failure_reason=reason, source=PREV,
        ))


FIELDS = {
    "feed": r"weight feed (\d+) cycles driven \(N\^3=(\d+) by design\)",
    "total": r"feed start to last row of tile 1 (\d+)",
    "per_tile": r"(\d+) cycles a tile",
    "wrong_pe": r"weight files (\d+) PE\(s\) wrong",
    "wrong_col": r"tile columns (\d+) wrong",
    "verdict": r"FEATHER RTL N=\d+ NP=\d+: (PASS|FAIL|TIMEOUT)",
}


def restricted_rows(rows):
    for N in (4, 8, 16, 32):
        d1, d2 = f"{PREV_RTL}/restricted/run{N}x1", f"{PREV_RTL}/restricted/run{N}x2"
        t1, t2 = rd(f"{d1}/xsim.log"), rd(f"{d2}/xsim.log")
        run_id = f"rtl_orig_N{N}_single_tile_restricted_bench"
        base = dict(run_id=run_id, system="FEATHER RTL", variant="shipped controller", array_size=f"{N}x{N}",
                    workload="single tile (NP=1) and two tiles back to back (NP=2), the record's restricted bench tests/dataflow/spmw/tb_feather_rtl.sv",
                    weight_mode="constrained (odd-index weights zeroed, zero points 0)", reorder_program="pass-through BIRRD",
                    implementation_mode="rtl_sim", source=PREV)
        if not t1:
            rows.append(row(**base, status="not_run", failure_reason="restricted bench run missing"))
            continue
        g = {k: (re.search(p, t1).group(1) if re.search(p, t1) else None) for k, p in FIELDS.items()}
        m2 = re.search(FIELDS["per_tile"], t2)
        v2 = re.search(FIELDS["verdict"], t2)
        rdir = os.path.join(RES, "reports", run_id)
        os.makedirs(rdir, exist_ok=True)
        if os.path.isfile(f"{d1}/xsim.log"):
            shutil.copy(f"{d1}/xsim.log", os.path.join(rdir, "xsim_np1.log"))
        if os.path.isfile(f"{d2}/xsim.log"):
            shutil.copy(f"{d2}/xsim.log", os.path.join(rdir, "xsim_np2.log"))
        ok = g["verdict"] == "PASS" and (v2 is None or v2.group(1) == "PASS")
        rows.append(row(
            **base, status="pass" if ok else "fail",
            validation_pass=f"NP=1 {g['verdict']} ({g['wrong_pe']} PE files wrong, {g['wrong_col']} columns wrong); NP=2 {v2.group(1) if v2 else None}",
            tiles=1, first_output_cycles=g["total"], completion_cycles=g["total"],
            cycles_per_tile=(int(m2.group(1)) if m2 else None),
            report_paths=rdir, failure_reason=None if ok else "restricted bench did not pass",
        ))


# -- SPMW cosim runs ---------------------------------------------------------------


def spmw_row(run_id, run_dir, workload, reorder, weight_mode, source):
    r = jload(os.path.join(run_dir, "result.json"))
    resident = "_x" in os.path.basename(run_dir)
    base = dict(run_id=run_id, system="SPMW port", variant="feather_stream_x (weights and commands resident)" if resident else "feather_stream (every operand streamed)",
                workload=workload, reorder_program=reorder, weight_mode=weight_mode, implementation_mode="cosim", source=source)
    if r is None:
        running = os.path.isdir(run_dir)
        return row(**base, status="not_run", failure_reason=("run not finished (no result.json yet)" if running else "not started"))
    N = r["N"]
    vdir = os.path.join(RES, "validation", run_id)
    copy_into(vdir, [os.path.join(run_dir, "result.json")])
    rdir = os.path.join(RES, "reports", run_id)
    copy_into(rdir, [os.path.join(run_dir, "result.json"), os.path.join(run_dir, "sim", "xsim.log"), os.path.join(run_dir, "cost.json")])
    tl = os.path.join(run_dir, "sim", "tiles.log")
    if os.path.isfile(tl):
        with open(tl, "rb") as fi, gzip.open(os.path.join(rdir, "tiles.log.gz"), "wb") as fo:
            shutil.copyfileobj(fi, fo)
    t = r.get("timing", {})
    ok = r.get("status") == "pass" and r.get("host_check") in ("pass", None)
    val = r.get("cosim")
    if r.get("host_check"):
        val += f"; host reduction vs numpy: {r['host_check']} ({r.get('host_bad')} bad elements)"
    elif not resident:
        val += "; (no host reduction recorded for this run)"
    iv = r.get("interval")
    return row(
        **base, array_size=f"{N}x{N}", status="pass" if ok else "fail", validation_pass=val, tiles=r.get("NT"),
        first_output_cycles=r.get("first_tile_done"), completion_cycles=r.get("last_tile_done"),
        cycles_per_tile=(iv["median"] if isinstance(iv, dict) else None),
        hls_wall_s=t.get("hls_wall_s"), total_wall_s=round(sum(v for v in t.values() if isinstance(v, (int, float))), 1),
        report_paths=f"{rdir}; {vdir}; run dir {run_dir}",
        failure_reason=None if ok else (r.get("cosim") or "cosim did not pass") + (f"; host_check={r.get('host_check')}" if r.get("host_check") else ""),
    )


# -- P&R ---------------------------------------------------------------------------


def pnr_rtl_row(variant, N):
    d = f"{PNR}/{variant}_{N}"
    run_id = f"pnr_rtl_{variant}_N{N}"
    base = dict(run_id=run_id, system="FEATHER RTL", variant="shipped controller" if variant == "orig" else "corrected controller",
                workload="feather_top out of context: N x N NEST + BIRRD + controller + SRAMs (shipped defaults: depth-4 SRAM register arrays)",
                array_size=f"{N}x{N}", weight_mode="n/a", reorder_program="n/a", implementation_mode="pnr_ooc", source=PREV)
    res = rd(f"{d}/result.txt")
    if not res:
        return row(**base, status="not_run", failure_reason=("P&R still running (no result.txt)" if os.path.isdir(d) else "not started"))
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
    reason = None if ok else "Vivado failed: " + " | ".join(errs)
    wns_v = float(wns.group(1)) if wns else w2
    return row(
        **base, status="pass" if ok else "fail",
        validation_pass=("timing met" if wns_v is not None and wns_v >= 0 else "timing NOT met") if ok else None,
        lut=u.get("lut"), ff=u.get("ff"), dsp=u.get("dsp"), bram_18k_equiv=u.get("bram18k"), uram=u.get("uram"),
        wns_ns=wns_v, tns_ns=tns, unrouted=(int(unrouted.group(1)) if unrouted else None),
        total_wall_s=int(secs.group(1)) if secs else None,
        report_paths=rdir + f"; stages (s): {stages}", failure_reason=reason,
    )


def pnr_spmw_row(design, N):
    d = f"{PNR}/spmw_{design}_{N}"
    run_id = f"pnr_spmw_{design}_N{N}"
    label = {"feather-stream": "feather_stream (streamed operands, int32 word files), NT=16", "feather": "feather (single tile, resident int8 files)"}[design]
    base = dict(run_id=run_id, system="SPMW port", variant=label,
                workload=f"{label}; top spmw_harness (LFSR drivers and sinks around spmw_top, the SPMW flow's standard P&R harness)",
                array_size=f"{N}x{N}", weight_mode="n/a", reorder_program="n/a", implementation_mode="pnr_ooc", source=MINE + " (e4b_pnr_worker.sh)")
    done = rd(f"{d}/E4_DONE")
    log = rd(f"{d}/build.log")
    if not done:
        return row(**base, status="not_run", failure_reason=("P&R still running (no E4_DONE)" if os.path.isdir(d) else "not started: queued behind the FEATHER RTL 32x32 P&R runs"))
    if done.startswith("skipped"):
        return row(**base, status="not_run", failure_reason=done.strip())
    if done.startswith("claimed"):
        running = os.path.isdir(f"{d}/lock")
        return row(**base, status="not_run", failure_reason=("P&R running (claimed by e4b_pnr_worker.sh, not finished)" if running else "queued: e4b_pnr_worker.sh starts it when a Vivado slot of this package frees (behind the FEATHER RTL 32x32 P&R runs)"))
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
    reason = None if ok else ("build failed: " + " | ".join(errs or log.splitlines()[-3:]))
    wns_v = float(wns.group(1)) if wns else w2
    return row(
        **base, status="pass" if ok else "fail",
        validation_pass=("timing met" if wns_v is not None and wns_v >= 0 else "timing NOT met") if ok else None,
        lut=u.get("lut"), ff=u.get("ff"), dsp=u.get("dsp"), bram_18k_equiv=u.get("bram18k"), uram=u.get("uram"),
        wns_ns=wns_v, tns_ns=tns, unrouted=(int(unrouted.group(1)) if unrouted else None),
        hls_wall_s=float(hls.group(1)) if hls else None, total_wall_s=int(secs.group(1)) if secs else None,
        report_paths=rdir + f"; stages (s): {stages}", failure_reason=reason,
    )


# -- the rows ----------------------------------------------------------------------


def main():
    os.makedirs(f"{RES}/reports", exist_ok=True)
    os.makedirs(f"{RES}/validation", exist_ok=True)
    rows = []
    GEN = "general (int8 -128..127 on the SPMW port / uint8 0..255 with zero point 0 on the RTL: the same stored bytes, seed 0)"
    # this agent's general-weight whole workloads, both systems
    for N in (4, 8, 16):
        rows.append(spmw_row(f"spmw_gemm128_N{N}_stream_general", f"{NEW_SPMW}/wl_gemm_N{N}_stream", gemm_wl(N), gemm_prog(N), GEN, MINE))
        rows.append(rtl_run_row(f"rtl_fixed_gemm128_N{N}_feed_general", f"{NEW_RTL}/wl_gemm_N{N}_m0", "corrected controller", gemm_wl(N) + "; a weight feed per tile (MODE 0)", gemm_prog(N), GEN, MINE, True))
        rows.append(spmw_row(f"spmw_gemm128_N{N}_resident_general", f"{NEW_SPMW}/wl_gemm_N{N}_x", gemm_wl(N) + "; every tile's activations through tile 0's resident weights (datapath throughput, not the workload's result)", gemm_prog(N), GEN, MINE))
        rows.append(rtl_run_row(f"rtl_fixed_gemm128_N{N}_resident_general", f"{NEW_RTL}/wl_gemm_N{N}_m1", "corrected controller", gemm_wl(N) + "; one feed, then every tile's activations through tile 0's resident weights (MODE 1; datapath throughput, not the workload's result)", gemm_prog(N), GEN, MINE, False))
    for N in (4, 8, 16):
        rows.append(spmw_row(f"spmw_conv_N{N}_stream_general", f"{NEW_SPMW}/wl_conv_N{N}_stream", conv_wl(N), conv_prog(N), GEN, MINE))
        rows.append(rtl_run_row(f"rtl_fixed_conv_N{N}_feed_general", f"{NEW_RTL}/wl_conv_N{N}_m0", "corrected controller", conv_wl(N) + "; a weight feed per tile (MODE 0)", conv_prog(N), GEN, MINE, True))
        rows.append(spmw_row(f"spmw_conv_N{N}_resident_general", f"{NEW_SPMW}/wl_conv_N{N}_x", conv_wl(N) + "; every tile's activations through tile 0's resident weights and program", conv_prog(N), GEN, MINE))
        rows.append(rtl_run_row(f"rtl_fixed_conv_N{N}_resident_general", f"{NEW_RTL}/wl_conv_N{N}_m1", "corrected controller", conv_wl(N) + "; one feed, then every tile's activations and programs through tile 0's resident weights (MODE 1)", conv_prog(N), GEN, MINE, False))
    # what the RTL makes of operands on both sides of the zero point
    for N in (4, 8, 16):
        rows.append(rtl_run_row(f"rtl_fixed_N{N}_tiles_mixed_sign_demo", f"{NEW_RTL}/demo_mixed_N{N}", "corrected controller",
                                "4 single tiles back to back, weights resident (MODE 1): stored bytes 0..255 with zero points 128/128, i.e. differences in [-128, 127]",
                                gemm_prog(N), "mixed-sign differences (u - zp < 0 for u < zp): outside the RTL's unsigned datapath", MINE, False))
    # the shipped controller on a general-weight workload
    rows.append(rtl_run_row("rtl_orig_gemm128_N4_feed_general", f"{NEW_RTL}/wl_gemm_N4_m0_orig", "shipped controller", gemm_wl(4) + "; a weight feed per tile (MODE 0)", gemm_prog(4), GEN, MINE, True))
    # the previous agent's constrained whole workloads (same tilings, small operand ranges)
    CON = {8: "constrained (d in [0, 8), zero points 7/5 on the RTL; the same d as int8 on the SPMW port)",
           16: "constrained (d in [0, 32), zero points 0)", 4: "constrained (d in [0, 8), zero points 128/128 on the RTL)"}
    for N in (8, 16):
        rows.append(spmw_row(f"spmw_gemm128_N{N}_stream_constrained", f"{PREV_SPMW}/wl_gemm_N{N}_stream", gemm_wl(N), gemm_prog(N), CON[N], PREV))
        rows.append(rtl_run_row(f"rtl_fixed_gemm128_N{N}_feed_constrained", f"{PREV_RTL}/wl_gemm_N{N}_m0", "corrected controller", gemm_wl(N) + "; a weight feed per tile (MODE 0)", gemm_prog(N), CON[N], PREV, True))
        rows.append(spmw_row(f"spmw_gemm128_N{N}_resident_constrained", f"{PREV_SPMW}/wl_gemm_N{N}_x", gemm_wl(N) + "; activations through tile 0's resident weights", gemm_prog(N), CON[N], PREV))
        rows.append(rtl_run_row(f"rtl_fixed_gemm128_N{N}_resident_constrained", f"{PREV_RTL}/wl_gemm_N{N}_m1", "corrected controller", gemm_wl(N) + "; one feed, activations through tile 0's resident weights (MODE 1)", gemm_prog(N), CON[N], PREV, False))
    rows.append(spmw_row("spmw_conv_N4_stream_constrained", f"{PREV_SPMW}/wl_conv_N4_stream", conv_wl(4), conv_prog(4), CON[4], PREV))
    rows.append(rtl_run_row("rtl_fixed_conv_N4_feed_constrained", f"{PREV_RTL}/wl_conv_N4_m0", "corrected controller", conv_wl(4) + "; a weight feed per tile (MODE 0)", conv_prog(4), CON[4], PREV, True))
    rows.append(spmw_row("spmw_conv_N4_resident_constrained", f"{PREV_SPMW}/wl_conv_N4_x", conv_wl(4) + "; activations through tile 0's resident weights and program", conv_prog(4), CON[4], PREV))
    rows.append(rtl_run_row("rtl_fixed_conv_N4_resident_constrained", f"{PREV_RTL}/wl_conv_N4_m1", "corrected controller", conv_wl(4) + "; one feed, activations and programs through tile 0's resident weights (MODE 1)", conv_prog(4), CON[4], PREV, False))
    # single tiles: the validation matrices and the record's restricted bench
    matrix_rows(rows)
    restricted_rows(rows)
    # P&R
    for N in (4, 8, 16, 32):
        rows.append(pnr_spmw_row("feather-stream", N))
        rows.append(pnr_spmw_row("feather", N))
        rows.append(pnr_rtl_row("fixed", N))
        rows.append(pnr_rtl_row("orig", N))
    with open(f"{RES}/results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    with open(f"{RES}/results.json", "w", encoding="utf-8") as f:
        json.dump({"experiment_id": "E4", "columns": COLS, "rows": rows}, f, indent=1, default=int)
    print(f"{len(rows)} rows -> {RES}/results.csv")
    for r in rows:
        print(f"  {r['run_id']:<46} {str(r['status']):<8} tiles={str(r['tiles']):<7} first={str(r['first_output_cycles']):<8} done={str(r['completion_cycles']):<9} per_tile={str(r['cycles_per_tile']):<7} lut={r['lut']} wns={r['wns_ns']}")
    tables(rows)
    sdir = os.path.join(RES, "scripts")
    os.makedirs(sdir, exist_ok=True)
    for src, sub in ((f"{ROOT}/e4_work", "previous_agent"), (f"{ROOT}/e4b_work", "this_agent")):
        dst = os.path.join(sdir, sub)
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(f"{src}/*"):
            if os.path.isfile(f) and not f.endswith((".log", ".done")):
                shutil.copy(f, dst)
    shutil.copy(f"{ROOT}/e4_feather_fixed/feather_controller.diff", sdir)


def fmt(v, digits=0):
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}" if digits else f"{v:.0f}"
    return str(v)


def tables(rows):
    by = {r["run_id"]: r for r in rows}
    out = ["### Whole workloads, general weights (this agent's runs): cycles measured in xsim on both sides\n",
           "| workload | N | tiles | system | mode | first tile done | last tile done (whole workload) | cycles per tile (steady state, median) | validation |",
           "|---|---|---|---|---|---|---|---|---|"]
    for wl, key in (("GEMM 128^3", "gemm128"), ("conv 16x16x64->64 3x3", "conv")):
        for N in (4, 8, 16):
            for rid, system, mode in (
                (f"spmw_{key}_N{N}_stream_general", "SPMW port feather_stream", "every operand streamed per tile"),
                (f"rtl_fixed_{key}_N{N}_feed_general", "FEATHER RTL (corrected)", "a weight feed per tile"),
                (f"spmw_{key}_N{N}_resident_general", "SPMW port feather_stream_x", "weights resident"),
                (f"rtl_fixed_{key}_N{N}_resident_general", "FEATHER RTL (corrected)", "weights resident"),
            ):
                r = by.get(rid)
                if not r:
                    continue
                out.append(f"| {wl} | {N} | {fmt(r['tiles'])} | {system} | {mode} | {fmt(r['first_output_cycles'])} | {fmt(r['completion_cycles'])} | {fmt(r['cycles_per_tile'])} | {r['status']}: {fmt(r['validation_pass'])[:90]} |")
    out.append("\n### The same workloads with the previous agent's constrained operands (its runs)\n")
    out.append("| workload | N | system | mode | first tile done | last tile done | cycles per tile | validation |")
    out.append("|---|---|---|---|---|---|---|---|")
    for wl, key, Ns in (("GEMM 128^3", "gemm128", (8, 16)), ("conv 16x16x64->64 3x3", "conv", (4,))):
        for N in Ns:
            for rid, system, mode in (
                (f"spmw_{key}_N{N}_stream_constrained", "SPMW port feather_stream", "streamed"),
                (f"rtl_fixed_{key}_N{N}_feed_constrained", "FEATHER RTL (corrected)", "a feed per tile"),
                (f"spmw_{key}_N{N}_resident_constrained", "SPMW port feather_stream_x", "resident"),
                (f"rtl_fixed_{key}_N{N}_resident_constrained", "FEATHER RTL (corrected)", "resident"),
            ):
                r = by.get(rid)
                if not r:
                    continue
                out.append(f"| {wl} | {N} | {system} | {mode} | {fmt(r['first_output_cycles'])} | {fmt(r['completion_cycles'])} | {fmt(r['cycles_per_tile'])} | {r['status']}: {fmt(r['validation_pass'])[:70]} |")
    out.append("\n### Place-and-route, xcu280-fsvh2892-2L-e, Vivado 2023.2, 3.333 ns\n")
    out.append("| design | N | status | LUT | FF | DSP | BRAM (18k eq.) | URAM | WNS ns | TNS ns | unrouted | total s | HLS s |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for N in (4, 8, 16, 32):
        for rid, label in ((f"pnr_spmw_feather-stream_N{N}", "SPMW port feather_stream (spmw_harness)"), (f"pnr_spmw_feather_N{N}", "SPMW port feather, single tile (spmw_harness)"),
                           (f"pnr_rtl_fixed_N{N}", "FEATHER RTL corrected, feather_top OOC"), (f"pnr_rtl_orig_N{N}", "FEATHER RTL shipped, feather_top OOC")):
            r = by.get(rid)
            if not r:
                continue
            out.append(f"| {label} | {N} | {r['status']} | {fmt(r['lut'])} | {fmt(r['ff'])} | {fmt(r['dsp'])} | {fmt(r['bram_18k_equiv'])} | {fmt(r['uram'])} | {fmt(r['wns_ns'], 3)} | {fmt(r['tns_ns'], 3)} | {fmt(r['unrouted'])} | {fmt(r['total_wall_s'])} | {fmt(r['hls_wall_s'], 1)} |")
    text = "\n".join(out) + "\n"
    with open(os.path.join(RES, "README_tables.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
