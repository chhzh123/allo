#!/scratch/hc676/allo-agent/bin/python3
"""Assemble E2 FFT results: results.csv + reports/<run_id>/ + validation/<run_id>/ for HP-FFT and Allo."""
import csv, glob, json, math, os, re, shutil, subprocess, sys
E = "/scratch/hc676/e2_hpfft"; A = "/scratch/hc676/e2_allo"
OUT = "/scratch/hc676/spmw_eval_remaining_2026-09-06/e2_fft"
COLS = ["run_id","experiment_id","system","transform_size","config","parallelism","interface","implementation_mode","target_mhz","status","validation_pass","max_abs_error","max_norm_error","first_output_cycles","completion_cycles","steady_interval_cycles","lut","ff","dsp","bram_18k_equiv","uram","wns_ns","tns_ns","hls_wall_s","vivado_synth_s","place_s","route_s","total_wall_s","ignored_directives","report_paths","failure_reason"]
WARN_NAMES = {"HLS 207-5573": "performance-pragma-ignored(not-in-loop)", "HLS 214-189": "pipeline-directive-removed(loop-fully-unrolled)",
              "HLS 200-885": "II-violation", "HLS 200-871": "estimated-clock-exceeds-target", "HLS 200-960": "cannot-flatten-loop",
              "HLS 200-656": "possible-deadlock(auto-rewind-in-dataflow)", "HLS 200-805": "internal-stream-default-depth",
              "HLS 214-114": "non-canonical-dataflow-region", "HLS 214-111": "static-in-dataflow-treated-local", "HLS 200-471": "dataflow-form-issues",
              "SYN 201-303": "memory-assignment-not-applied", "RTGEN 206-101": "rtgen-warning", "HLS 214-358": "index-bit-extension"}
def kv(fn):
    r = {}
    if os.path.exists(fn):
        for tok in open(fn).read().split():
            if "=" in tok: k, v = tok.split("=", 1); r[k] = v
    return r
def jload(fn):
    try: return json.load(open(fn))
    except Exception: return None
def cp(src, dst_dir, name=None):
    if src and os.path.exists(src):
        os.makedirs(dst_dir, exist_ok=True); shutil.copy(src, os.path.join(dst_dir, name or os.path.basename(src))); return True
    return False
def util_rpt(fn):
    if not os.path.exists(fn): return {}
    R = open(fn).read(); g = {}
    for key, pat in [("lut", r"CLB LUTs\*?"), ("ff", r"CLB Registers"), ("ramb36", r"RAMB36/FIFO\*?"), ("ramb18", r"RAMB18"), ("uram", r"URAM"), ("dsp", r"DSPs")]:
        m = re.search(r"^\|\s*" + pat + r"\s*\|\s*([\d.]+)\s*\|", R, re.M)
        if m: g[key] = float(m.group(1))
    if "ramb36" in g or "ramb18" in g: g["bram_18k_equiv"] = int(2 * g.get("ramb36", 0) + g.get("ramb18", 0))
    return g
def timing_rpt(fn):
    if not os.path.exists(fn): return {}
    R = open(fn).read(); r = {}
    m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n[-\s]+\n\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)", R, re.S)
    if m: r.update(wns=float(m.group(1)), tns=float(m.group(2)), failing_endpoints=int(m.group(3)), whs=float(m.group(5)), ths=float(m.group(6)))
    r["timing_met"] = ("All user specified timing constraints are met" in R)
    return r
def route_rpt(fn):
    if not os.path.exists(fn): return {}
    R = open(fn).read(); r = {}
    def num(label):
        m = re.search(r"# of " + label + r"\.*\s*:\s*(\d+)", R); return int(m.group(1)) if m else None
    r["routing_errors"] = num("nets with routing errors"); r["routable_nets"] = num("routable nets"); r["fully_routed_nets"] = num("fully routed nets")
    r["unrouted_nets"] = (r["routable_nets"] - r["fully_routed_nets"]) if (r["routable_nets"] is not None and r["fully_routed_nets"] is not None) else None
    r["fully_routed"] = (r["routing_errors"] == 0) and (r["unrouted_nets"] == 0)
    return r
def pnr_row(pdir, done):
    d = kv(done); txt = open(done).read() if os.path.exists(done) else ""
    mph = re.search(r"^E2_PHASES (.*)$", txt, re.M)
    ph = dict(re.findall(r"(\w+)=(\d+)", mph.group(1))) if mph else {}
    u = util_rpt(f"{pdir}/util.rpt"); t = timing_rpt(f"{pdir}/timing.rpt"); rt = route_rpt(f"{pdir}/route_status.rpt")
    ok = d.get("rc") == "0" and re.search(r"^E2 IMPLEMENTATION OK", txt, re.M) is not None
    return d, ph, u, t, rt, ok, txt
def hls_warn_summary(h):
    parts = []
    for code, n in sorted(h.get("warning_codes", {}).items()):
        c = code.replace("WARNING ", "")
        if c in WARN_NAMES: parts.append(f"{c} {WARN_NAMES[c]} x{n}")
    for l in h.get("perf_pragma_misses", []):
        f = [x.strip() for x in l.strip("|").split("|")]
        parts.append(f"perf-target-missed {f[0].strip('o ')}: target {f[1]} achieved {f[2]}")
    return "; ".join(parts)
rows_h, rows_a = [], []
def emit(rows, r):
    rows.append({c: ("" if r.get(c) is None else r.get(c)) for c in COLS})
# ------------------------------------------------------------------ HP-FFT
for d in sorted(glob.glob(f"{E}/n*/*")):
    size, cfg = d.split("/")[-2], d.split("/")[-1]
    if not os.path.isdir(d) or not os.path.exists(f"{d}/FFT.h"): continue
    N = int(size[1:]); L = N.bit_length() - 1
    derived = size in ("n128", "n512")
    system = "HP-FFT (derived size; see README)" if derived else "HP-FFT"
    tag = "hpfft-derived" if derived else "hpfft"
    m_uf = re.match(r"UF(\d+)", cfg); uf = int(m_uf.group(1)) if m_uf else None
    control = "_nt" in cfg; cfg_dir = cfg
    if uf:
        par = f"{uf} radix-2 butterflies/cycle in each of {L} dataflow stages ({uf*L} butterflies in flight); {2*uf} complex samples/cycle at each boundary"
        iface = f"ap_fifo in/out: hls::stream<hls::vector<complex<float>,{2*uf}>>, {128*uf}-bit data each way ({N//(2*uf)} beats/transform); ap_ctrl_hs"
    elif cfg == "no_StagePipeline":
        par = "1 radix-2 butterfly/cycle (pipelined inner loop), log2N stages run sequentially; no stage overlap"; iface = "ap_memory dataIn/dataOut (complex<float>[N], 64-bit words, 2 ports each); ap_ctrl_hs"
    else:
        par = "sequential C-style radix-2 (original_C_style): no pipeline directives"; iface = "ap_memory dataIn/dataOut (complex<float>[N], 64-bit words); ap_ctrl_hs"
    if control: cfg = cfg.split("_")[0] + f" (control run: {cfg.split('_nt')[1]} transforms fed)"
    base = dict(experiment_id="E2", system=system, transform_size=N, config=cfg, parallelism=par, interface=iface, target_mhz=300)
    if os.path.exists(f"{d}/hls.done"):
        with open(f"{d}/harvest.json", "w") as hf: subprocess.run([sys.executable, f"{E}/harvest_hls.py", d], stdout=hf, stderr=subprocess.DEVNULL)
    h = jload(f"{d}/harvest.json") or {}
    hd = kv(f"{d}/hls.done"); m = h.get("metrics")
    # csynth row
    if hd:
        rid = f"{tag}_{size}_{cfg_dir}_csynth"; rp = f"{OUT}/hpfft/reports/{rid}"
        for f in ["hls.log", "harvest.json", "FFT.h", "project.tcl", "build/FFT_300MHz/syn/report/csynth.rpt", "build/FFT_300MHz/FFT_300MHz_data.json"]: cp(f"{d}/{f}", rp)
        cp(f"{E}/common.tcl", rp); cp(f"{E}/gen_size.py", rp) if derived else None
        vc = jload(f"{d}/validate_wrapc.json")
        r = dict(base, run_id=rid, implementation_mode="csynth", hls_wall_s=hd.get("hls_wall_s"), report_paths=os.path.relpath(rp, OUT))
        if hd.get("rc") == "0" and m:
            r.update(status="ok", completion_cycles=m["latency_worst"], steady_interval_cycles=m["interval"], lut=m["lut"], ff=m["ff"], dsp=m["dsp"], bram_18k_equiv=m["bram_18k"], uram=m["uram"],
                     ignored_directives=hls_warn_summary(h), failure_reason="")
            if float(m["timing_estimate"]) > float(m["timing_target"]) - float(m["timing_uncertainty"]):
                r["failure_reason"] = f"HLS timing estimate {m['timing_estimate']} ns exceeds {float(m['timing_target'])-float(m['timing_uncertainty']):.3f} ns budget (period 3.333 - uncertainty {m['timing_uncertainty']}); synthesis continued"
            if vc: r.update(validation_pass=vc["validation_pass"], max_abs_error=f"{vc['max_abs_error']:.3e}", max_norm_error=f"{vc['max_norm_error']:.3e}")
        else:
            err = [l for l in open(f"{d}/hls.log").read().splitlines() if l.startswith("ERROR")][:2] if os.path.exists(f"{d}/hls.log") else []
            r.update(status="fail", failure_reason=("csynth rc=" + str(hd.get("rc")) + ": " + " | ".join(err))[:400])
        emit(rows_h, r)
    # cosim row
    cd = kv(f"{d}/cosim.done")
    if cd:
        rid = f"{tag}_{size}_{cfg_dir}_cosim"; rp = f"{OUT}/hpfft/reports/{rid}"; vp = f"{OUT}/hpfft/validation/{rid}"
        S = f"{d}/build/FFT_300MHz/sim"
        for f in ["cosim.log", "cosim.done", "cosim_events.json", "validate_wrapc.json", "validate_wrapc_pc.json", "cosim.tcl", "testbench.cpp", f"{S}/report/FFT_TOP_cosim.rpt", f"{S}/report/verilog/lat.rpt", f"{S}/verilog/e2_vcd.log", f"{S}/verilog/e2_vcd.tcl"]:
            cp(f if f.startswith("/") else f"{d}/{f}", rp)
        for f in ["e2_inputs.txt", "e2_outputs.txt"]: cp(f"{S}/wrapc_pc/{f}", vp, f.replace(".txt", "_rtl.txt")); cp(f"{S}/wrapc/{f}", vp, f.replace(".txt", "_cmodel.txt"))
        cp(f"{d}/validate_wrapc_pc.json", vp); cp(f"{d}/validate_wrapc.json", vp); cp(f"{E}/validate_fft.py", vp); cp(f"{E}/cosim_events.py", vp)
        log = open(f"{d}/cosim.log").read() if os.path.exists(f"{d}/cosim.log") else ""
        passed = "C/RTL co-simulation finished: PASS" in log
        if os.path.exists(f"{d}/e2_rtl_outputs.txt") and os.path.getsize(f"{d}/e2_rtl_outputs.txt") > 0:
            with open(f"{d}/validate_rtl_vcd.json", "w") as vf: subprocess.run([sys.executable, f"{E}/validate_fft.py", f"{d}/e2_rtl_inputs.txt", f"{d}/e2_rtl_outputs.txt", str(N), cd.get("NT", "32")], stdout=vf, stderr=subprocess.DEVNULL)
        ev = jload(f"{d}/cosim_events.json") or {}; vr = jload(f"{d}/validate_rtl_vcd.json") or jload(f"{d}/validate_wrapc_pc.json")
        for f in ["e2_rtl_inputs.txt", "e2_rtl_outputs.txt", "validate_rtl_vcd.json"]: cp(f"{d}/{f}", vp)
        cp(f"{S}/verilog/e2_xsim.log", rp); cp(f"{S}/verilog/e2_xelab.log", rp)
        nt = int(cd.get("NT", 32)); full = (ev.get("transforms_completed") or 0) >= nt - 1
        in_progress = ("xsim_rc" not in open(f"{d}/cosim.done").read()) and cd.get("rc") == "0"
        ok = (cd.get("rc") == "0") and (full if cd.get("method", "").startswith("setup") else passed) and not in_progress
        r = dict(base, run_id=rid, implementation_mode="cosim", hls_wall_s=cd.get("cosim_wall_s"), report_paths=os.path.relpath(rp, OUT) + "," + os.path.relpath(vp, OUT))
        if ok:
            r.update(status="ok", first_output_cycles=ev.get("first_output_cycles"), completion_cycles=ev.get("first_transform_completion_cycles"), steady_interval_cycles=ev.get("steady_interval_cycles_median"), failure_reason="")
            if vr: r.update(validation_pass=vr["validation_pass"], max_abs_error=f"{vr['max_abs_error']:.3e}", max_norm_error=f"{vr['max_norm_error']:.3e}")
            if not ev: r["failure_reason"] = "VCD event extraction missing (see cosim report for HLS-measured latency/interval)"
            elif ev.get("transforms_completed") == nt - 1: r["failure_reason"] = f"note: {nt-1}/{nt} transforms completed; the last transform's outputs are never flushed without a following transform (top-level ap_done never fires, HLS 200-656); intervals measured over the completed ones"
            if m: r["lut"], r["ff"], r["dsp"], r["bram_18k_equiv"], r["uram"] = m["lut"], m["ff"], m["dsp"], m["bram_18k"], m["uram"]
        elif in_progress:
            r.update(status="running", failure_reason="cosim still in progress at assembly time (re-run finalize.sh)")
        else:
            err = [l for l in log.splitlines() if l.startswith("ERROR")][:2]
            r.update(status="fail", failure_reason=("cosim rc=" + str(cd.get("rc")) + (f" transforms_completed={ev.get('transforms_completed')}/{nt}" if ev else " (no events)") + ("" if passed or cd.get("method","").startswith("setup") else " (no PASS)") + ": " + " | ".join(err))[:400])
        emit(rows_h, r)
    # pnr row
    if os.path.exists(f"{d}/pnr.done"):
        rid = f"{tag}_{size}_{cfg_dir}_pnr"; rp = f"{OUT}/hpfft/reports/{rid}"
        pd, ph, u, t, rt, ok, txt = pnr_row(f"{d}/pnr", f"{d}/pnr.done")
        for f in ["util.rpt", "util_hier.rpt", "util_synth.rpt", "timing.rpt", "route_status.rpt", "vivado.log", "impl.tcl", "clock.xdc"]: cp(f"{d}/pnr/{f}", rp)
        cp(f"{d}/pnr.done", rp)
        r = dict(base, run_id=rid, implementation_mode="pnr_ooc", total_wall_s=pd.get("total_wall_s"), vivado_synth_s=ph.get("synth"), place_s=ph.get("place"), route_s=ph.get("route"), report_paths=os.path.relpath(rp, OUT))
        if ok:
            r.update(lut=int(u.get("lut", 0)), ff=int(u.get("ff", 0)), dsp=int(u.get("dsp", 0)), bram_18k_equiv=u.get("bram_18k_equiv"), uram=int(u.get("uram", 0)), wns_ns=t.get("wns"), tns_ns=t.get("tns"))
            if t.get("wns") is not None and t["wns"] < 0:
                r.update(status="timing_fail", failure_reason=f"routed but WNS {t['wns']} ns < 0 at 3.333 ns ({t.get('failing_endpoints')} failing endpoints, TNS {t.get('tns')}): NOT a 300 MHz result; fmax ~ {1000/(3.333 - t['wns']):.1f} MHz")
            elif not rt.get("fully_routed", True):
                r.update(status="route_fail", failure_reason=f"unrouted nets: {rt.get('unrouted_nets')}")
            else:
                r.update(status="ok", failure_reason="")
        else:
            err = [l for l in txt.splitlines() if l.startswith("ERROR")][:2]
            r.update(status="fail", failure_reason=("vivado rc=" + str(pd.get("rc")) + ": " + " | ".join(err))[:400])
        emit(rows_h, r)
# ------------------------------------------------------------------ Allo
for d in sorted(glob.glob(f"{A}/strided_n*_*")):
    mm = re.match(r"strided_n(\d+)_(wrap|raw)", os.path.basename(d)); N, var = int(mm.group(1)), mm.group(2)
    S = f"{d}/out.prj/solution1"
    base = dict(experiment_id="E2", system="Allo (examples/machsuite/fft/strided, radix-2 DIF, in-place, bit-reversed output)", transform_size=N, config=f"fft_strided FFT_SIZE={N} wrap_io={var=='wrap'}",
                parallelism="sequential: one radix-2 butterfly per inner while-loop iteration, no pipeline/unroll directives; stages sequential", 
                interface=("m_axi x4 (float real[N], img[N], real_twid[N/2], img_twid[N/2]; in-place real/img) + s_axilite control; " + ("load/store buffers around the kernel (Allo wrap_io=True default)" if var == "wrap" else "kernel loops access m_axi directly (wrap_io=False)")), target_mhz=300)
    hd = kv(f"{d}/hls.done"); js = jload(f"{S}/solution1_data.json"); m = None
    if js:
        t = js["ModuleInfo"]["Metrics"].get("fft"); 
        if t: m = dict(latency_worst=t["Latency"]["LatencyWorst"], interval=t["Latency"]["PipelineII"], lut=t["Area"]["LUT"], ff=t["Area"]["FF"], dsp=t["Area"]["DSP"], bram_18k=t["Area"]["BRAM_18K"], uram=t["Area"]["URAM"], timing_estimate=t["Timing"]["Estimate"], timing_target=t["Timing"]["Target"], timing_uncertainty=t["Timing"]["Uncertainty"])
    log = open(f"{d}/hls.log").read() if os.path.exists(f"{d}/hls.log") else ""
    if hd:
        rid = f"allo-strided_n{N}_{var}_csynth"; rp = f"{OUT}/allo/reports/{rid}"; vp = f"{OUT}/allo/validation/{rid}"
        for f in ["hls.log", "hls.done", "kernel.cpp", "kernel.h", "tb.cpp", "run.tcl", "run_e2.tcl", f"{S}/syn/report/csynth.rpt", f"{S}/solution1_data.json", f"{S}/syn/report/fft_csynth.rpt"]: cp(f if f.startswith("/") else f"{d}/{f}", rp)
        codes = {}
        for l in log.splitlines():
            x = re.match(r"WARNING: \[(\S+ \S+)\]", l)
            if x: codes[x.group(1)] = codes.get(x.group(1), 0) + 1
        vc = jload(f"{d}/validate_wrapc.json")
        r = dict(base, run_id=rid, implementation_mode="csynth", hls_wall_s=hd.get("hls_wall_s"), report_paths=os.path.relpath(rp, OUT), ignored_directives="; ".join(f"{k} x{v}" for k, v in sorted(codes.items()) if k.startswith("HLS 2")))
        if m and "csynth_design" in log and "Finished Command csynth_design" in log:
            r.update(status="ok", completion_cycles=m["latency_worst"], steady_interval_cycles=m["interval"], lut=m["lut"], ff=m["ff"], dsp=m["dsp"], bram_18k_equiv=m["bram_18k"], uram=m["uram"], failure_reason="")
            if vc: r.update(validation_pass=vc["validation_pass"], max_abs_error=f"{vc['max_abs_error']:.3e}", max_norm_error=f"{vc['max_norm_error']:.3e}")
        else:
            err = [l for l in log.splitlines() if l.startswith("ERROR")][:2]
            r.update(status="fail", failure_reason=("rc=" + str(hd.get("rc")) + ": " + " | ".join(err))[:400])
        emit(rows_a, r)
        # cosim row (same vitis_hls session)
        if "cosim_design" in open(f"{d}/run_e2.tcl").read() if os.path.exists(f"{d}/run_e2.tcl") else False:
            rid = f"allo-strided_n{N}_{var}_cosim"; rp = f"{OUT}/allo/reports/{rid}"; vp = f"{OUT}/allo/validation/{rid}"
            for f in [f"{S}/sim/report/fft_cosim.rpt", f"{S}/sim/report/verilog/lat.rpt", f"{d}/cosim_events.json", f"{d}/validate_wrapc_pc.json", f"{d}/validate_wrapc.json"]: cp(f, rp)
            for f in ["e2_inputs.txt", "e2_outputs.txt"]: cp(f"{S}/sim/wrapc_pc/{f}", vp, f.replace(".txt", "_rtl.txt")); cp(f"{S}/sim/wrapc/{f}", vp, f.replace(".txt", "_cmodel.txt"))
            cp(f"{d}/validate_wrapc_pc.json", vp); cp(f"{d}/validate_wrapc.json", vp)
            passed = "C/RTL co-simulation finished: PASS" in log
            for w in ("wrapc", "wrapc_pc"):
                if os.path.exists(f"{S}/sim/{w}/e2_outputs.txt"):
                    with open(f"{d}/validate_{w}.json", "w") as vf: subprocess.run([sys.executable, f"{E}/validate_fft.py", f"{S}/sim/{w}/e2_inputs.txt", f"{S}/sim/{w}/e2_outputs.txt", str(N), "4", "bitrev"], stdout=vf, stderr=subprocess.DEVNULL)
                    cp(f"{d}/validate_{w}.json", vp)
            vr = jload(f"{d}/validate_wrapc_pc.json"); vc = jload(f"{d}/validate_wrapc.json")
            crpt = open(f"{S}/sim/report/fft_cosim.rpt").read() if os.path.exists(f"{S}/sim/report/fft_cosim.rpt") else ""
            mv = re.search(r"\|\s*Verilog\s*\|\s*Pass\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)", crpt)
            lat = open(f"{S}/sim/report/verilog/lat.rpt").read() if os.path.exists(f"{S}/sim/report/verilog/lat.rpt") else ""
            r = dict(base, run_id=rid, implementation_mode="cosim", report_paths=os.path.relpath(rp, OUT) + "," + os.path.relpath(vp, OUT))
            if passed and mv:
                lmin, lavg, lmax, imin, iavg, imax, tot = map(int, mv.groups())
                r.update(status="ok", first_output_cycles=None, completion_cycles=lmax, steady_interval_cycles=iavg,
                         failure_reason=f"cosim report over 4 calls: latency min/avg/max {lmin}/{lavg}/{lmax}, call interval min/avg/max {imin}/{iavg}/{imax}, total {tot}; kernel not pipelined across calls; output in bit-reversed order (validated as such)")
                if vr: r.update(validation_pass=vr["validation_pass"], max_abs_error=f"{vr['max_abs_error']:.3e}", max_norm_error=f"{vr['max_norm_error']:.3e}")
                if m: r["lut"], r["ff"], r["dsp"], r["bram_18k_equiv"], r["uram"] = m["lut"], m["ff"], m["dsp"], m["bram_18k"], m["uram"]
            else:
                err = [l for l in log.splitlines() if l.startswith("ERROR")][:2]
                r.update(status="fail", failure_reason=("cosim did not PASS: " + " | ".join(err))[:400])
            emit(rows_a, r)
            # the csynth row of this project gets the C-model validation
            for rr in rows_a:
                if rr["run_id"] == f"allo-strided_n{N}_{var}_csynth" and vc: rr.update(validation_pass=vc["validation_pass"], max_abs_error=f"{vc['max_abs_error']:.3e}", max_norm_error=f"{vc['max_norm_error']:.3e}")
    if os.path.exists(f"{d}/pnr.done"):
        rid = f"allo-strided_n{N}_{var}_pnr"; rp = f"{OUT}/allo/reports/{rid}"
        pd, ph, u, t, rt, ok, txt = pnr_row(f"{d}/pnr", f"{d}/pnr.done")
        for f in ["util.rpt", "util_hier.rpt", "util_synth.rpt", "timing.rpt", "route_status.rpt", "vivado.log", "impl.tcl", "clock.xdc"]: cp(f"{d}/pnr/{f}", rp)
        cp(f"{d}/pnr.done", rp)
        r = dict(base, run_id=rid, implementation_mode="pnr_ooc", total_wall_s=pd.get("total_wall_s"), vivado_synth_s=ph.get("synth"), place_s=ph.get("place"), route_s=ph.get("route"), report_paths=os.path.relpath(rp, OUT))
        if ok:
            r.update(lut=int(u.get("lut", 0)), ff=int(u.get("ff", 0)), dsp=int(u.get("dsp", 0)), bram_18k_equiv=u.get("bram_18k_equiv"), uram=int(u.get("uram", 0)), wns_ns=t.get("wns"), tns_ns=t.get("tns"))
            if t.get("wns") is not None and t["wns"] < 0: r.update(status="timing_fail", failure_reason=f"routed but WNS {t['wns']} ns < 0 at 3.333 ns ({t.get('failing_endpoints')} failing endpoints): NOT a 300 MHz result; fmax ~ {1000/(3.333 - t['wns']):.1f} MHz")
            elif not rt.get("fully_routed", True): r.update(status="route_fail", failure_reason=f"unrouted nets: {rt.get('unrouted_nets')}")
            else: r.update(status="ok", failure_reason="")
        else:
            err = [l for l in txt.splitlines() if l.startswith("ERROR")][:2]
            r.update(status="fail", failure_reason=("vivado rc=" + str(pd.get("rc")) + ": " + " | ".join(err))[:400])
        emit(rows_a, r)
for sub, rows in (("hpfft", rows_h), ("allo", rows_a)):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    with open(f"{OUT}/{sub}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); [w.writerow(r) for r in rows]
    print(sub, len(rows), "rows ->", f"{OUT}/{sub}/results.csv")
