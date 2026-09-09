#!/scratch/hc676/allo-agent/bin/python3
"""Assemble the E2 FFT package for both systems into
  <OUT>/{hpfft,allo}/results.csv, results.json   (requested schema: one row per (config, N, mode))
  <OUT>/{hpfft,allo}/hls_estimates.csv           (csynth rows: HLS estimates, kept out of results.csv)
  <OUT>/{hpfft,allo}/reports/<run_id>/, validation/<run_id>/
Merges the previous agent's runs (base config dirs, LCG stimulus) with this agent's seeded reruns (<cfg>_s dirs,
numpy seeds 0/1/2). Safe to re-run at any time."""
import csv, glob, json, os, re, shutil, subprocess, sys, datetime, statistics
E = "/scratch/hc676/e2_hpfft"; A = "/scratch/hc676/e2_allo"
OUT = "/scratch/hc676/spmw_eval_remaining_2026-09-06/e2_fft"
PY = sys.executable
MY_START = 1788749400   # 2026-09-06 22:50 EDT: everything with a done-file older than this was produced by the previous agent
COLS = ["run_id","experiment_id","system","workload","numeric_semantics","implementation_mode","target_mhz","status","validation_pass",
        "first_output_cycles","completion_cycles","steady_interval_cycles","transforms_streamed","lut","ff","dsp","bram_18k_equiv","bram_36k","bram_18k","uram",
        "wns_ns","tns_ns","unrouted","hls_wall_s","synth_s","place_s","route_s","total_wall_s","interface","unroll_factor","report_paths","failure_reason",
        "transform_size","config","ordering","stimulus","seeds","transforms_completed","cosim_wall_s","max_abs_error","max_norm_error","run_by","notes"]
HCOLS = ["run_id","system","transform_size","config","unroll_factor","latency_cycles","interval_cycles","pipeline_type","lut","ff","dsp","bram_18k","uram",
         "timing_estimate_ns","timing_budget_ns","hls_wall_s","csim_pass","cmodel_validation_pass","cmodel_max_abs_error","ignored_directives","run_by","report_paths","notes"]
NUMSEM = "complex float32 (IEEE-754 binary32 real and imaginary parts; forward transform, unscaled)"
WARN_NAMES = {"HLS 207-5573": "performance-pragma-ignored(not-in-loop)", "HLS 214-189": "pipeline-directive-removed(loop-fully-unrolled)",
              "HLS 200-885": "II-violation", "HLS 200-871": "estimated-clock-exceeds-target", "HLS 200-960": "cannot-flatten-loop",
              "HLS 200-656": "possible-deadlock(auto-rewind-in-dataflow)", "HLS 200-805": "internal-stream-default-depth",
              "HLS 214-114": "non-canonical-dataflow-region", "HLS 214-111": "static-in-dataflow-treated-local", "HLS 200-471": "dataflow-form-issues",
              "HLS 200-880": "II-violation(memory-dependence)", "SYN 201-303": "memory-assignment-not-applied", "RTGEN 206-101": "rtgen-warning", "HLS 214-358": "index-bit-extension"}
def kv(fn):
    r = {}
    if os.path.exists(fn):
        for tok in open(fn).read().split():
            if "=" in tok: k, v = tok.split("=", 1); r.setdefault(k, v)
    return r
def jload(fn):
    try: return json.load(open(fn))
    except Exception: return None
def mtime(fn): return os.path.getmtime(fn) if os.path.exists(fn) else None
def cp(src, dst_dir, name=None):
    if src and os.path.exists(src):
        os.makedirs(dst_dir, exist_ok=True); shutil.copy(src, os.path.join(dst_dir, name or os.path.basename(src))); return True
    return False
def util_rpt(fn):
    if not os.path.exists(fn): return {}
    R = open(fn).read(); g = {}
    for key, pat in [("lut", r"CLB LUTs\*?"), ("ff", r"CLB Registers"), ("ramb36", r"RAMB36/FIFO\*?"), ("ramb18", r"RAMB18"), ("uram", r"URAM"), ("dsp", r"DSPs")]:
        m = re.search(r"^\|\s*" + pat + r"\s*\|\s*([\d.]+)\s*\|", R, re.M)
        if m: g[key] = int(float(m.group(1)))
    if "ramb36" in g or "ramb18" in g: g["bram_18k_equiv"] = 2 * g.get("ramb36", 0) + g.get("ramb18", 0)
    return g
def timing_rpt(fn):
    if not os.path.exists(fn): return {}
    R = open(fn).read(); r = {}
    m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n[-\s]+\n\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)", R, re.S)
    if m: r.update(wns=float(m.group(1)), tns=float(m.group(2)), failing_endpoints=int(m.group(3)), total_endpoints=int(m.group(4)), whs=float(m.group(5)), ths=float(m.group(6)))
    r["timing_met"] = ("All user specified timing constraints are met" in R)
    return r
def route_rpt(fn):
    if not os.path.exists(fn): return {}
    R = open(fn).read(); r = {}
    def num(label):
        m = re.search(r"# of " + label + r"\.*\s*:\s*(\d+)", R); return int(m.group(1)) if m else None
    r["routing_errors"] = num("nets with routing errors"); r["routable_nets"] = num("routable nets"); r["fully_routed_nets"] = num("fully routed nets")
    r["unrouted_nets"] = num("unrouted nets")
    if r["unrouted_nets"] is None and r["routable_nets"] is not None and r["fully_routed_nets"] is not None: r["unrouted_nets"] = r["routable_nets"] - r["fully_routed_nets"]
    r["fully_routed"] = (r["routing_errors"] == 0) and (r["unrouted_nets"] == 0)
    return r
def hls_metrics(js, top):
    J = jload(js)
    if not J: return None
    t = J["ModuleInfo"]["Metrics"].get(top)
    if not t: return None
    def mx(s):
        v = [int(x) for x in re.findall(r"\d+", str(s or ""))]; return max(v) if v else None
    return dict(latency_worst=mx(t["Latency"]["LatencyWorst"]), interval=mx(t["Latency"]["PipelineII"]), pipeline_type=t["Latency"].get("PipelineType"),
                lut=int(t["Area"]["LUT"]), ff=int(t["Area"]["FF"]), dsp=int(t["Area"]["DSP"]), bram_18k=int(t["Area"]["BRAM_18K"]), uram=int(t["Area"]["URAM"]),
                timing_estimate=float(t["Timing"]["Estimate"]), timing_target=float(t["Timing"]["Target"]), timing_uncertainty=float(t["Timing"]["Uncertainty"]))
def warn_codes(log):
    codes, samples = {}, {}
    for l in log.splitlines():
        mm = re.match(r"(WARNING|ERROR|CRITICAL WARNING): \[(\S+ \S+)\] (.*)", l)
        if mm: c = mm.group(2); codes[c] = codes.get(c, 0) + 1; samples.setdefault(c, mm.group(3)[:200])
    return codes, samples
def warn_summary(codes, rpt):
    parts = [f"{c} {WARN_NAMES[c]} x{n}" for c, n in sorted(codes.items()) if c in WARN_NAMES]
    if rpt and os.path.exists(rpt):
        R = open(rpt).read()
        sec = R.split("Performance Pragma Report")[1].split("\n+ ")[0] if "Performance Pragma Report" in R else ""
        for l in sec.splitlines():
            if re.search(r"\|\s*no\s*\|", l):
                f = [x.strip() for x in l.strip().strip("|").split("|")]
                parts.append(f"perf-target-missed {f[0].strip('o ')}: target {f[1]} achieved {f[2]}")
    return "; ".join(parts)
def queue_state():
    st = {}
    for state in ("pending", "running", "done", "deferred"):
        for f in glob.glob(f"{E}/queue_pnr/{state}/*"):
            try: kind, size, cfg = open(f).read().split()
            except ValueError: continue
            st[(kind, size, cfg)] = (state, os.path.basename(f))
    return st
QS = queue_state()
LOG_PREV = open(f"{E}/logs/queue_pnr.log").read() if os.path.exists(f"{E}/logs/queue_pnr.log") else ""
LOG_MINE = open(f"{E}/logs/pnr_worker.log").read() if os.path.exists(f"{E}/logs/pnr_worker.log") else ""
def pnr_run_by(kind, size, cfg, done_file):
    job = QS.get((kind, size, cfg), (None, None))[1]
    if job and re.search(r"done " + re.escape(job) + r"\s*$", LOG_MINE, re.M): return "this_agent"
    if job and re.search(r"done " + re.escape(job) + r"\s*$", LOG_PREV, re.M): return "previous_agent"
    if job and re.search(r"start " + re.escape(job) + r":", LOG_PREV): return "previous_agent"
    m = mtime(done_file)
    return "previous_agent" if (m and m < MY_START) else "this_agent"
def pnr_fields(pdir, done):
    d = kv(done); txt = open(done).read() if os.path.exists(done) else ""
    mph = re.search(r"^E2_PHASES (.*)$", txt, re.M); ph = dict(re.findall(r"(\w+)=(\d+)", mph.group(1))) if mph else {}
    u = util_rpt(f"{pdir}/util.rpt"); t = timing_rpt(f"{pdir}/timing.rpt"); rt = route_rpt(f"{pdir}/route_status.rpt")
    ok = d.get("rc") == "0" and re.search(r"^E2 IMPLEMENTATION OK", txt, re.M) is not None
    return d, ph, u, t, rt, ok, txt
def fill_pnr(r, pdir, done, rp):
    pd, ph, u, t, rt, ok, txt = pnr_fields(pdir, done)
    for f in ["util.rpt", "util_hier.rpt", "util_synth.rpt", "timing.rpt", "route_status.rpt", "vivado.log", "impl.tcl", "clock.xdc"]: cp(f"{pdir}/{f}", rp)
    cp(done, rp)
    r.update(total_wall_s=int(pd["total_wall_s"]) if pd.get("total_wall_s") else None, synth_s=int(ph["synth"]) if "synth" in ph else None,
             place_s=int(ph["place"]) if "place" in ph else None, route_s=int(ph["route"]) if "route" in ph else None)
    if ok:
        r.update(lut=u.get("lut"), ff=u.get("ff"), dsp=u.get("dsp"), bram_18k_equiv=u.get("bram_18k_equiv"), bram_36k=u.get("ramb36"), bram_18k=u.get("ramb18"), uram=u.get("uram"),
                 wns_ns=t.get("wns"), tns_ns=t.get("tns"), unrouted=rt.get("unrouted_nets"))
        notes = [f"opt {ph.get('opt')} s, phys_opt {ph.get('physopt')} s; {t.get('failing_endpoints')} failing of {t.get('total_endpoints')} endpoints; WHS {t.get('whs')} ns; routable nets {rt.get('routable_nets')}, routing errors {rt.get('routing_errors')}"]
        if t.get("wns") is not None and t["wns"] < 0:
            r.update(status="timing_fail", failure_reason=f"routed, but WNS {t['wns']} ns < 0 at 3.333 ns ({t.get('failing_endpoints')} failing endpoints, TNS {t.get('tns')} ns): not a 300 MHz result; fmax ~ {1000/(3.333 - t['wns']):.1f} MHz")
        elif not rt.get("fully_routed", True):
            r.update(status="route_fail", failure_reason=f"unrouted nets: {rt.get('unrouted_nets')}, routing errors: {rt.get('routing_errors')}")
        else:
            r.update(status="ok", failure_reason="")
        r["notes"] = "; ".join(notes)
    else:
        err = [l for l in txt.splitlines() if l.startswith("ERROR")][:2]
        r.update(status="fail", failure_reason=("vivado rc=" + str(pd.get("rc")) + ": " + " | ".join(err))[:400])
    return r
def not_run_pnr(r, kind, size, cfg, est_lut):
    state, job = QS.get((kind, size, cfg), (None, None))
    if state == "running": r.update(status="running", failure_reason=f"P&R in progress at assembly time (queue job {job})")
    elif state == "pending": r.update(status="not_run", failure_reason=f"P&R queued but not reached before the package was assembled (queue job {job}; optional extra)")
    elif state == "deferred": r.update(status="not_run", failure_reason=f"P&R deferred: optional extra, HLS estimate {est_lut} LUTs; an OOC P&R of that size takes hours on the shared machine (queue job {job} in queue_pnr/deferred)")
    else: r.update(status="not_run", failure_reason="P&R not run (optional extra, never queued)")
    return r
def typed(v):
    if v is None or v == "": return None
    if isinstance(v, (bool, int, float)): return v
    return v
def emit(rows, r): rows.append({c: ("" if r.get(c) is None else r.get(c)) for c in COLS})
def emit_h(rows, r): rows.append({c: ("" if r.get(c) is None else r.get(c)) for c in HCOLS})
def seed_notes(v):
    if not v: return ""
    if "per_seed" in v:
        return "per seed: " + ", ".join(f"seed {s}: {'pass' if x['validation_pass'] else 'FAIL'} ({x['transforms_checked']} transforms, max abs err {x['max_abs_error']:.2e})" for s, x in v["per_seed"].items()) + (f"; unchecked transforms {v['unchecked_transforms']}" if v.get("unchecked_transforms") else "")
    return f"{v.get('transforms_checked')} transforms checked"
rows_h, rows_a, hls_h, hls_a = [], [], [], []
# ================================================================== HP-FFT
def hp_cosim_row(base, d, cfg_dir, tag, size, N, UF, seeded, rid, m):
    """cosim row from config dir d (seeded or LCG)."""
    cd = kv(f"{d}/cosim.done"); rp = f"{OUT}/hpfft/reports/{rid}"; vp = f"{OUT}/hpfft/validation/{rid}"
    S = f"{d}/build/FFT_300MHz/sim"
    for f in ["cosim.log", "cosim.done", "cosim_events.json", "validate_wrapc.json", "validate_rtl_vcd.json", "cosim.tcl", "testbench.cpp", "e2_stimulus.json", "hls.done", f"{S}/report/FFT_TOP_cosim.rpt", f"{S}/report/verilog/lat.rpt", f"{S}/verilog/e2_vcd.tcl", f"{S}/verilog/e2_xsim.log", f"{S}/verilog/e2_xelab.log"]:
        cp(f if f.startswith("/") else f"{d}/{f}", rp)
    for f in ["e2_rtl_inputs.txt", "e2_rtl_outputs.txt", "validate_rtl_vcd.json", "validate_wrapc.json", "e2_stimulus.json"]: cp(f"{d}/{f}", vp)
    cp(f"{S}/wrapc/e2_outputs.txt", vp, "e2_outputs_cmodel.txt"); cp(f"{S}/wrapc/e2_inputs.txt", vp, "e2_inputs_cmodel.txt")
    for f in ["validate_fft2.py", "cosim_events.py", "gen_stimulus.py"]: cp(f"{E}/{f}", vp)
    nt = int(cd.get("NT", 32)); ev = jload(f"{d}/cosim_events.json") or {}
    if os.path.exists(f"{d}/e2_rtl_outputs.txt") and os.path.getsize(f"{d}/e2_rtl_outputs.txt") > 0:
        args = [PY, f"{E}/validate_fft2.py", f"{d}/e2_rtl_inputs.txt", f"{d}/e2_rtl_outputs.txt", str(N), str(nt), "natural"] + (["--seeds", f"{d}/e2_stimulus.json"] if seeded else [])
        with open(f"{d}/validate_rtl_vcd.json", "w") as vf: subprocess.run(args, stdout=vf, stderr=subprocess.DEVNULL)
        cp(f"{d}/validate_rtl_vcd.json", vp); cp(f"{d}/validate_rtl_vcd.json", rp)
    vr = jload(f"{d}/validate_rtl_vcd.json")
    if vr is None and os.path.exists(f"{S}/wrapc_pc/e2_outputs.txt"):
        # array-interface configs: no port-data dump from the VCD; the RTL outputs are what Vitis' post-check
        # testbench run (wrapc_pc) received from the RTL
        with open(f"{d}/validate_wrapc_pc.json", "w") as vf: subprocess.run([PY, f"{E}/validate_fft2.py", f"{S}/wrapc_pc/e2_inputs.txt", f"{S}/wrapc_pc/e2_outputs.txt", str(N), str(nt), "natural"], stdout=vf, stderr=subprocess.DEVNULL)
        vr = jload(f"{d}/validate_wrapc_pc.json"); cp(f"{d}/validate_wrapc_pc.json", vp); cp(f"{d}/validate_wrapc_pc.json", rp)
        cp(f"{S}/wrapc_pc/e2_inputs.txt", vp, "e2_inputs_rtl.txt"); cp(f"{S}/wrapc_pc/e2_outputs.txt", vp, "e2_outputs_rtl.txt")
        if vr: vr["source"] = "RTL outputs as received by Vitis' post-check testbench run (sim/wrapc_pc)"
    done_txt = open(f"{d}/cosim.done").read() if os.path.exists(f"{d}/cosim.done") else ""
    in_progress = ("xsim_rc" not in done_txt) and cd.get("rc") == "0" and os.path.exists(f"{d}/cosim.done")
    full = (ev.get("transforms_completed") or 0) >= nt - 1
    cosim_log = open(f"{d}/cosim.log").read() if os.path.exists(f"{d}/cosim.log") else ""
    stream_flow = cd.get("method", "").startswith("setup")
    ok = (cd.get("rc") == "0") and cd.get("xsim_rc") == "0" and full and not in_progress and ((cd.get("xelab_rc") == "0") if stream_flow else ("C/RTL co-simulation finished: PASS" in cosim_log))
    wall = sum(int(cd.get(k, 0) or 0) for k in ("cosim_wall_s", "xelab_wall_s", "xsim_wall_s"))
    stim = (f"numpy.random.RandomState seeds 0/1/2, re,im ~ U[-1,1) float32; {nt} transforms = 11 per seed" if seeded else f"transform 0 = HP-FFT shipped test signal, transforms 1..{nt-1} = LCG(0x12345678) U[-1,1)")
    r = dict(base, run_id=rid, implementation_mode="cosim", workload=f"{N}-point radix-2 complex FFT; {nt} transforms " + ("streamed back-to-back through one FFT_TOP" if stream_flow else "as back-to-back FFT_TOP calls (array interface, ap_ctrl_hs)"),
             transforms_streamed=nt, hls_wall_s=kv(f"{d}/hls.done").get("hls_wall_s"), cosim_wall_s=wall, stimulus=stim, seeds=("0,1,2" if seeded else ""),
             run_by=("this_agent" if seeded else "previous_agent"), report_paths=os.path.relpath(rp, OUT) + "," + os.path.relpath(vp, OUT))
    if ok:
        ivs = ev.get("completion_intervals_all") or []; tail = ivs[len(ivs) // 4:]
        r.update(status="ok", first_output_cycles=ev.get("first_output_cycles"), completion_cycles=ev.get("first_transform_completion_cycles"),
                 steady_interval_cycles=(round(sum(tail) / len(tail), 1) if tail else ev.get("steady_interval_cycles_median")), transforms_completed=ev.get("transforms_completed"), failure_reason="")
        notes = ([f"steady interval = mean of the last {len(tail)} completion-to-completion distances (median {statistics.median(tail)}, min {min(tail)}, max {max(tail)})"] if tail else [])
        if ev.get("transforms_completed") == nt - 1: notes.append(f"{nt-1}/{nt} transforms completed: the last transform fed is never flushed without a following transform (top-level ap_done never fires, HLS 200-656); intervals over the completed ones")
        if ev.get("completion_intervals_all"): notes.append(f"batch of {ev.get('transforms_completed')} transforms complete at cycle {ev.get('batch_completion_cycles')} after the first input beat; ap_done pulses {ev.get('ap_done_pulses')}; input stall cycles in transform 0: {ev.get('input_stall_cycles_transform0')}")
        if vr:
            r.update(validation_pass=vr["validation_pass"], max_abs_error=f"{vr['max_abs_error']:.3e}", max_norm_error=f"{vr['max_norm_error']:.3e}")
            notes.append(seed_notes(vr) + ("; " + vr["source"] if vr.get("source") else "; RTL outputs read from the out_r port handshakes in the VCD"))
            if vr["validation_pass"] is False: r.update(status="validation_fail", failure_reason="RTL output differs from numpy.fft beyond atol=rtol=1e-4")
        else:
            r.update(status="fail", failure_reason="no RTL output data extracted from the VCD")
        r["notes"] = "; ".join(notes)
    elif in_progress or (not os.path.exists(f"{d}/cosim.done") and os.path.isdir(d)):
        r.update(status="running", failure_reason="cosim still in progress at assembly time")
    else:
        log = open(f"{d}/cosim.log").read() if os.path.exists(f"{d}/cosim.log") else ""
        err = [l for l in log.splitlines() if l.startswith("ERROR")][:2]
        r.update(status="fail", failure_reason=("cosim rc=" + str(cd.get("rc")) + f" xelab_rc={cd.get('xelab_rc')} xsim_rc={cd.get('xsim_rc')} transforms_completed={ev.get('transforms_completed')}/{nt}: " + " | ".join(err))[:400])
    return r
for d in sorted(glob.glob(f"{E}/n*/*")):
    size, cfg = d.split("/")[-2], d.split("/")[-1]
    if not os.path.isdir(d) or not os.path.exists(f"{d}/FFT.h") or cfg.endswith("_s"): continue
    N = int(size[1:]); L = N.bit_length() - 1
    derived = size in ("n128", "n512")
    system = "HP-FFT (derived size, see README)" if derived else "HP-FFT"
    tag = "hpfft-derived" if derived else "hpfft"
    m_uf = re.match(r"UF(\d+)", cfg); uf = int(m_uf.group(1)) if m_uf else None
    control = "_nt" in cfg; cfg_dir = cfg
    if uf:
        iface = f"ap_fifo in/out: hls::stream<hls::vector<complex<float>,{2*uf}>>, {128*uf}-bit data each way = {2*uf} complex float32 samples per beat, {N//(2*uf)} beats per transform; block control ap_ctrl_hs"
        par = f"{uf} radix-2 butterflies per cycle in each of {L} dataflow stages"
    elif cfg == "no_StagePipeline":
        iface = "ap_memory dataIn/dataOut (complex<float>[N] arrays, 64-bit words, 2 ports each); ap_ctrl_hs"; par = "1 pipelined butterfly/cycle, stages sequential (authors' Baseline2)"
    else:
        iface = "ap_memory dataIn/dataOut (complex<float>[N] arrays, 64-bit words); ap_ctrl_hs"; par = "sequential C loop nest, no directives (authors' original_C_style)"
    cfg_label = cfg.split("_nt")[0] + f" (control run: {cfg.split('_nt')[1]} transforms fed)" if control else cfg
    base = dict(experiment_id="E2", system=system, numeric_semantics=NUMSEM, target_mhz=300, interface=iface, unroll_factor=uf,
                transform_size=N, config=cfg_label, ordering="natural-order input, natural-order output (DIT with bit-reversal reorder stage)")
    m = hls_metrics(f"{d}/build/FFT_300MHz/FFT_300MHz_data.json", "FFT_TOP"); hd = kv(f"{d}/hls.done")
    # ---- csynth (HLS estimates) rows: base dir and seeded copy
    for dd, sfx in ((d, ""), (f"{d}_s", "_s")):
        hdd = kv(f"{dd}/hls.done")
        if not hdd: continue
        mm = hls_metrics(f"{dd}/build/FFT_300MHz/FFT_300MHz_data.json", "FFT_TOP")
        rid = f"{tag}_{size}_{cfg_dir}{sfx}_csynth"; rp = f"{OUT}/hpfft/reports/{rid}"
        for f in ["hls.log", "hls.done", "FFT.h", "FFT.cpp", "project.tcl", "build/FFT_300MHz/syn/report/csynth.rpt", "build/FFT_300MHz/FFT_300MHz_data.json"]: cp(f"{dd}/{f}", rp)
        cp(f"{E}/common.tcl", rp); (cp(f"{E}/gen_size.py", rp) if derived else None)
        log = open(f"{dd}/hls.log").read() if os.path.exists(f"{dd}/hls.log") else ""
        codes, samples = warn_codes(log)
        json.dump({"warning_codes": codes, "warning_samples": samples}, open(f"{rp}/warnings.json", "w"), indent=1)
        vc = jload(f"{dd}/validate_wrapc.json")
        r = dict(run_id=rid, system=system, transform_size=N, config=cfg_label + (" (seeded rerun, this agent)" if sfx else ""), unroll_factor=uf, hls_wall_s=hdd.get("hls_wall_s"),
                 csim_pass=("CSim done with 0 errors" in log), run_by=("this_agent" if sfx else "previous_agent"), report_paths=os.path.relpath(rp, OUT))
        if hdd.get("rc") == "0" and mm:
            r.update(latency_cycles=mm["latency_worst"], interval_cycles=mm["interval"], pipeline_type=mm["pipeline_type"], lut=mm["lut"], ff=mm["ff"], dsp=mm["dsp"], bram_18k=mm["bram_18k"], uram=mm["uram"],
                     timing_estimate_ns=mm["timing_estimate"], timing_budget_ns=round(mm["timing_target"] - mm["timing_uncertainty"], 3),
                     ignored_directives=warn_summary(codes, f"{dd}/build/FFT_300MHz/syn/report/csynth.rpt"),
                     notes=("HLS timing estimate exceeds the budget (HLS 200-871); synthesis continued" if mm["timing_estimate"] > mm["timing_target"] - mm["timing_uncertainty"] else ""))
            if vc: r.update(cmodel_validation_pass=vc["validation_pass"], cmodel_max_abs_error=f"{vc['max_abs_error']:.3e}")
        else:
            err = [l for l in log.splitlines() if l.startswith("ERROR")][:2]
            r.update(notes=("csynth rc=" + str(hdd.get("rc")) + ": " + " | ".join(err))[:300])
        emit_h(hls_h, r)
    if control:
        # control run (33 transforms fed, previous agent): one cosim row only
        if os.path.exists(f"{d}/cosim.done"):
            emit(rows_h, hp_cosim_row(base, d, cfg_dir, tag, size, N, uf, False, f"{tag}_{size}_{cfg_dir}_cosim", m))
        continue
    # ---- cosim rows: seeded rerun (primary) and previous agent's LCG run
    ds = f"{d}_s"; have_s = os.path.isdir(ds)
    if have_s:
        emit(rows_h, hp_cosim_row(base, ds, cfg_dir, tag, size, N, uf, True, f"{tag}_{size}_{cfg_dir}_cosim", m))
        if os.path.exists(f"{d}/cosim.done"): emit(rows_h, hp_cosim_row(base, d, cfg_dir, tag, size, N, uf, False, f"{tag}_{size}_{cfg_dir}_cosim_lcg", m))
    elif os.path.exists(f"{d}/cosim.done"):
        emit(rows_h, hp_cosim_row(base, d, cfg_dir, tag, size, N, uf, False, f"{tag}_{size}_{cfg_dir}_cosim", m))
    elif hd:
        emit(rows_h, dict(base, run_id=f"{tag}_{size}_{cfg_dir}_cosim", implementation_mode="cosim", workload=f"{N}-point radix-2 complex FFT", status="not_run",
                          failure_reason="RTL cosimulation not run (optional extra: the previous agent's queue did not reach it and it was not re-queued; csynth row in hls_estimates.csv)", run_by="", report_paths=""))
    # ---- pnr row
    rid = f"{tag}_{size}_{cfg_dir}_pnr"; rp = f"{OUT}/hpfft/reports/{rid}"
    r = dict(base, run_id=rid, implementation_mode="pnr_ooc", workload=f"{N}-point radix-2 complex FFT (one FFT_TOP instance, out-of-context)", hls_wall_s=hd.get("hls_wall_s"), report_paths=os.path.relpath(rp, OUT))
    if os.path.exists(f"{d}/pnr.done"):
        r["run_by"] = pnr_run_by("pnr", size, cfg, f"{d}/pnr.done"); fill_pnr(r, f"{d}/pnr", f"{d}/pnr.done", rp)
    elif hd:
        r["report_paths"] = ""; not_run_pnr(r, "pnr", size, cfg, m["lut"] if m else "?")
    else:
        continue
    emit(rows_h, r)
# ================================================================== Allo
def allo_cosim_row(base, d, N, var, seeded, rid, m):
    S = f"{d}/out.prj/solution1"; rp = f"{OUT}/allo/reports/{rid}"; vp = f"{OUT}/allo/validation/{rid}"
    hd = kv(f"{d}/hls.done"); log = open(f"{d}/hls.log").read() if os.path.exists(f"{d}/hls.log") else ""
    nt = int(hd.get("NT", 4))
    for f in ["hls.log", "hls.done", "tb.cpp", "run_e2.tcl", "run_e2s.tcl", "e2_stimulus.json", "validate_wrapc.json", "validate_wrapc_pc.json", f"{S}/sim/report/fft_cosim.rpt", f"{S}/sim/report/verilog/lat.rpt", f"{S}/sim/report/verilog/result.transaction.rpt"]:
        cp(f if f.startswith("/") else f"{d}/{f}", rp)
    for w in ("wrapc", "wrapc_pc"):
        if os.path.exists(f"{S}/sim/{w}/e2_outputs.txt"):
            args = [PY, f"{E}/validate_fft2.py", f"{S}/sim/{w}/e2_inputs.txt", f"{S}/sim/{w}/e2_outputs.txt", str(N), str(nt), "bitrev"] + (["--seeds", f"{d}/e2_stimulus.json"] if seeded else [])
            with open(f"{d}/validate_{w}.json", "w") as vf: subprocess.run(args, stdout=vf, stderr=subprocess.DEVNULL)
            cp(f"{d}/validate_{w}.json", vp); cp(f"{d}/validate_{w}.json", rp)
            cp(f"{S}/sim/{w}/e2_inputs.txt", vp, f"e2_inputs_{'rtl' if w == 'wrapc_pc' else 'cmodel'}.txt"); cp(f"{S}/sim/{w}/e2_outputs.txt", vp, f"e2_outputs_{'rtl' if w == 'wrapc_pc' else 'cmodel'}.txt")
    cp(f"{d}/e2_stimulus.json", vp); cp(f"{E}/validate_fft2.py", vp); cp(f"{E}/gen_stimulus.py", vp)
    vr = jload(f"{d}/validate_wrapc_pc.json")
    passed = "C/RTL co-simulation finished: PASS" in log
    tr = open(f"{S}/sim/report/verilog/result.transaction.rpt").read() if os.path.exists(f"{S}/sim/report/verilog/result.transaction.rpt") else ""
    lats = [int(x) for x in re.findall(r"transaction\s+\d+:\s+(\d+)\s+(?:\d+|x)", tr)]
    ints = [int(x) for x in re.findall(r"transaction\s+\d+:\s+\d+\s+(\d+)", tr)]
    stim = (f"numpy.random.RandomState seeds 0/1/2, re,im ~ U[-1,1) float32; {nt} transforms = {nt//3} per seed" if seeded else f"transform 0 = HP-FFT shipped test signal, transforms 1..{nt-1} = LCG(0x12345678) U[-1,1)")
    r = dict(base, run_id=rid, implementation_mode="cosim", workload=f"{N}-point radix-2 complex FFT; {nt} kernel calls back-to-back (one transform per call)", transforms_streamed=nt,
             hls_wall_s=hd.get("hls_wall_s"), stimulus=stim, seeds=("0,1,2" if seeded else ""), run_by=("this_agent" if seeded else "previous_agent"),
             report_paths=os.path.relpath(rp, OUT) + "," + os.path.relpath(vp, OUT))
    if not os.path.exists(f"{d}/hls.done"):
        r.update(status="running", failure_reason="csim/csynth/cosim still in progress at assembly time"); return r
    if passed and lats:
        tail = ints[len(ints) // 4:] if len(ints) > 2 else ints
        r.update(status="ok", first_output_cycles=None, completion_cycles=lats[0], steady_interval_cycles=(round(sum(tail) / len(tail), 1) if tail else None), transforms_completed=len(lats), failure_reason="")
        notes = [f"Vitis cosim transaction report: per-call latency {lats} (ap_start to ap_done via s_axi_control), call intervals {ints}; calls do not overlap; first_output_cycles not observable (results are written back to m_axi by the store stage at the end of the call)"]
        if vr:
            r.update(validation_pass=vr["validation_pass"], max_abs_error=f"{vr['max_abs_error']:.3e}", max_norm_error=f"{vr['max_norm_error']:.3e}")
            notes.append(seed_notes(vr) + "; RTL outputs compared in bit-reversed index order")
            if vr["validation_pass"] is False: r.update(status="validation_fail", failure_reason="RTL output differs from numpy.fft (bit-reversed order) beyond atol=rtol=1e-4")
        r["notes"] = "; ".join(notes)
    else:
        err = [l for l in log.splitlines() if l.startswith("ERROR")][:2]
        r.update(status="fail", failure_reason=("cosim did not PASS (rc=" + str(hd.get("rc")) + "): " + " | ".join(err))[:400])
    return r
for d in sorted(glob.glob(f"{A}/strided_n*_*")):
    mm = re.match(r"strided_n(\d+)_(wrap|raw)$", os.path.basename(d))
    if not mm: continue
    N, var = int(mm.group(1)), mm.group(2); S = f"{d}/out.prj/solution1"
    iface = ("m_axi x4, 32-bit data (float real[N], img[N], real_twid[N/2], img_twid[N/2]; real/img in-place) + s_axi_control; " +
             ("Allo wrap_io=True: load loops into on-chip buffers -> kernel -> store loops (1 float per beat per port)" if var == "wrap" else "wrap_io=False: kernel loops access m_axi directly"))
    base = dict(experiment_id="E2", system="Allo (examples/machsuite/fft/strided: radix-2 DIF, in-place, sequential)", numeric_semantics=NUMSEM, target_mhz=300, interface=iface, unroll_factor=None,
                transform_size=N, config=f"strided_fft FFT_SIZE={N} wrap_io={var == 'wrap'}", ordering="natural-order input, bit-reversed-order output (DIF); validated against numpy.fft permuted to bit-reversed index order")
    m = hls_metrics(f"{S}/solution1_data.json", "fft"); hd = kv(f"{d}/hls.done")
    for dd, sfx in ((d, ""), (f"{d}_s", "_s")):
        hdd = kv(f"{dd}/hls.done")
        if not hdd: continue
        SS = f"{dd}/out.prj/solution1"; mm2 = hls_metrics(f"{SS}/solution1_data.json", "fft")
        rid = f"allo-strided_n{N}_{var}{sfx}_csynth"; rp = f"{OUT}/allo/reports/{rid}"
        for f in ["hls.log", "hls.done", "kernel.cpp", "kernel.h", "tb.cpp", "run.tcl", "run_e2.tcl", "run_e2s.tcl", f"{SS}/syn/report/csynth.rpt", f"{SS}/solution1_data.json", f"{SS}/syn/report/fft_csynth.rpt"]: cp(f if f.startswith("/") else f"{dd}/{f}", rp)
        log = open(f"{dd}/hls.log").read() if os.path.exists(f"{dd}/hls.log") else ""
        codes, samples = warn_codes(log); json.dump({"warning_codes": codes, "warning_samples": samples}, open(f"{rp}/warnings.json", "w"), indent=1)
        vc = jload(f"{dd}/validate_wrapc.json")
        r = dict(run_id=rid, system=base["system"], transform_size=N, config=base["config"] + (" (seeded rerun, this agent)" if sfx else ""), hls_wall_s=hdd.get("hls_wall_s"),
                 csim_pass=("CSim done with 0 errors" in log), run_by=("this_agent" if sfx else "previous_agent"), report_paths=os.path.relpath(rp, OUT))
        if mm2 and "Finished Command csynth_design" in log:
            r.update(latency_cycles=mm2["latency_worst"], interval_cycles=mm2["interval"], pipeline_type=mm2["pipeline_type"], lut=mm2["lut"], ff=mm2["ff"], dsp=mm2["dsp"], bram_18k=mm2["bram_18k"], uram=mm2["uram"],
                     timing_estimate_ns=mm2["timing_estimate"], timing_budget_ns=round(mm2["timing_target"] - mm2["timing_uncertainty"], 3), ignored_directives=warn_summary(codes, None),
                     notes="no latency/interval from csynth: the two while loops have no trip-count bound; inner loop auto-pipelined at II=25 (HLS 200-880 memory dependence on the in-place real[]/img[] arrays)")
            if vc: r.update(cmodel_validation_pass=vc["validation_pass"], cmodel_max_abs_error=f"{vc['max_abs_error']:.3e}")
        else:
            err = [l for l in log.splitlines() if l.startswith("ERROR")][:2]
            r.update(notes=("rc=" + str(hdd.get("rc")) + ": " + " | ".join(err))[:300])
        emit_h(hls_a, r)
    ds = f"{d}_s"
    if os.path.isdir(ds):
        emit(rows_a, allo_cosim_row(base, ds, N, var, True, f"allo-strided_n{N}_{var}_cosim", m))
        if hd: emit(rows_a, allo_cosim_row(base, d, N, var, False, f"allo-strided_n{N}_{var}_cosim_lcg", m))
    elif hd:
        emit(rows_a, allo_cosim_row(base, d, N, var, False, f"allo-strided_n{N}_{var}_cosim", m))
    rid = f"allo-strided_n{N}_{var}_pnr"; rp = f"{OUT}/allo/reports/{rid}"
    r = dict(base, run_id=rid, implementation_mode="pnr_ooc", workload=f"{N}-point radix-2 complex FFT (one fft kernel instance with its m_axi/s_axi adapters, out-of-context)", hls_wall_s=hd.get("hls_wall_s"), report_paths=os.path.relpath(rp, OUT))
    if os.path.exists(f"{d}/pnr.done"):
        r["run_by"] = pnr_run_by("pnrallo", str(N), var, f"{d}/pnr.done"); fill_pnr(r, f"{d}/pnr", f"{d}/pnr.done", rp)
    elif hd:
        r["report_paths"] = ""; not_run_pnr(r, "pnrallo", str(N), var, m["lut"] if m else "?")
    else:
        continue
    emit(rows_a, r)
# ================================================================== write
def tool_versions():
    v = {}
    for fn, key, pat in [(f"{E}/n256/UF1/hls.log", "vitis_hls", r"Vitis HLS.*?\bv(\d\S*)"), (f"{E}/n256/UF1/hls.log", "vitis_hls_build", r"SW Build (\d+)"), (f"{E}/n256/UF1/build/FFT_300MHz/sim/verilog/e2_xsim.log", "xsim", r"xsim v(\S+)"), (f"{E}/n256/UF1/pnr/vivado.log", "vivado", r"Vivado v(\S+)"), (f"{E}/n256/UF1/pnr/vivado.log", "vivado_build", r"SW Build (\d+)")]:
        if os.path.exists(fn):
            mm = re.search(pat, open(fn).read()); v[key] = mm.group(1) if mm else None
    return v
def jsonify(r):
    o = {}
    for k, v in r.items():
        if v == "" or v is None: o[k] = None
        elif isinstance(v, bool): o[k] = v
        elif isinstance(v, (int, float)): o[k] = v
        elif isinstance(v, str) and re.fullmatch(r"-?\d+", v): o[k] = int(v)
        elif isinstance(v, str) and re.fullmatch(r"-?\d+\.\d*(e-?\d+)?", v): o[k] = float(v)
        elif v in ("True", "False"): o[k] = (v == "True")
        else: o[k] = v
    return o
meta = {"experiment_id": "E2", "generated": datetime.datetime.now().isoformat(timespec="seconds"), "host": "brg-zhang-xcel", "part": "xcu280-fsvh2892-2L-e", "target_period_ns": 3.333, "target_mhz": 300,
        "tools": tool_versions(), "checkouts": {"HP-FFT": "/scratch/hc676/HP-FFT-HLS @ c4611b8402d82b3d2ea2603eb4070b7ec5d269fe (final)", "Allo": "/scratch/hc676/allo @ f436658ab62847467ba5196c414b969d46fe2da9 (worktree hc/spmw-allo-implementation-99c949)"},
        "definitions": {"cycle": "rising ap_clk edges of the RTL cosimulation at 3.333 ns", "first_output_cycles": "first output beat minus first input beat of the first transform (HP-FFT stream interface; not observable for Allo)",
                         "completion_cycles": "HP-FFT: last output beat of the first transform minus its first input beat; Allo: ap_start-to-ap_done latency of the first kernel call (Vitis cosim transaction report)",
                         "steady_interval_cycles": "mean distance between consecutive transform completions over the last three quarters of the completed transforms (Allo: mean call-to-call interval over the last three quarters of the calls); median/min/max in notes",
                         "transforms_streamed": "transforms fed to the design in one cosimulation; transforms_completed = transforms whose outputs appeared",
                         "resources": "pnr_ooc rows: Vivado report_utilization after route_design (bram_18k_equiv = 2*RAMB36 + RAMB18); cosim rows carry no resource numbers (HLS estimates are in hls_estimates.csv)",
                         "wns_ns/tns_ns": "report_timing_summary after route_design, setup, single clock ap_clk at 3.333 ns, out-of-context (no I/O delays)", "unrouted": "report_route_status: routable nets minus fully routed nets",
                         "validation": "numpy.allclose(atol=1e-4, rtol=1e-4) of the RTL outputs against numpy.fft.fft (complex128) of the exact float32 inputs fed to the RTL; per-seed detail in notes and validation/<run_id>/",
                         "run_by": "previous_agent = produced by the earlier agent of this package (its queues/logs), this_agent = produced by the continuation agent (seeded reruns, pnr_worker.sh jobs)"},
        "status_values": ["ok", "timing_fail", "route_fail", "validation_fail", "fail", "running", "not_run"]}
for sub, rows, hrows in (("hpfft", rows_h, hls_h), ("allo", rows_a, hls_a)):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    with open(f"{OUT}/{sub}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rows]
    with open(f"{OUT}/{sub}/hls_estimates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HCOLS); w.writeheader(); [w.writerow(r) for r in hrows]
    json.dump({"meta": dict(meta, system=sub), "rows": [jsonify(r) for r in rows], "hls_estimates": [jsonify(r) for r in hrows]}, open(f"{OUT}/{sub}/results.json", "w"), indent=1)
    print(sub, len(rows), "rows,", len(hrows), "hls rows ->", f"{OUT}/{sub}/")
