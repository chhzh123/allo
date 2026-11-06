#!/scratch/hc676/allo-agent/bin/python3
"""Harvest one HP-FFT config dir into JSON: csynth metrics, warnings by code, performance-pragma
misses, interface table, csim result, wall times."""
import json, re, sys, os, glob, collections
d = sys.argv[1]
out = {"dir": d}
def kv(fn):
    r = {}
    if os.path.exists(fn):
        for tok in open(fn).read().split():
            if "=" in tok: k, v = tok.split("=", 1); r[k] = v
    return r
out["hls_done"] = kv(f"{d}/hls.done")
js = glob.glob(f"{d}/build/FFT_300MHz/FFT_300MHz_data.json")
if js:
    J = json.load(open(js[0]))
    m = J["ModuleInfo"]["Metrics"]["FFT_TOP"]
    out["metrics"] = {"latency_best": m["Latency"]["LatencyBest"], "latency_worst": m["Latency"]["LatencyWorst"],
                      "interval": m["Latency"]["PipelineII"], "pipeline_type": m["Latency"]["PipelineType"],
                      "timing_target": m["Timing"]["Target"], "timing_uncertainty": m["Timing"]["Uncertainty"], "timing_estimate": m["Timing"]["Estimate"],
                      "bram_18k": m["Area"]["BRAM_18K"], "dsp": m["Area"]["DSP"], "ff": m["Area"]["FF"], "lut": m["Area"]["LUT"], "uram": m["Area"]["URAM"]}
    ports = []
    for p in J.get("RtlPorts", []):
        ports.append({k: p.get(k) for k in ("name", "dir", "bits", "protocol") if k in p} if isinstance(p, dict) else str(p))
    out["rtl_ports"] = ports[:40]
    out["interfaces"] = J.get("Interfaces")
log = open(f"{d}/hls.log").read() if os.path.exists(f"{d}/hls.log") else ""
codes = collections.Counter(); samples = {}
for l in log.splitlines():
    mm = re.match(r"(WARNING|ERROR|CRITICAL WARNING): \[(\S+ \S+)\] (.*)", l)
    if mm:
        c = mm.group(1) + " " + mm.group(2); codes[c] += 1; samples.setdefault(c, mm.group(3)[:220])
out["warning_codes"] = dict(codes); out["warning_samples"] = samples
out["csim_pass"] = ("CSim done with 0 errors" in log)
rpt = f"{d}/build/FFT_300MHz/syn/report/csynth.rpt"
if os.path.exists(rpt):
    R = open(rpt).read()
    sec = R.split("Performance Pragma Report")[1].split("\n+ ")[0] if "Performance Pragma Report" in R else ""
    misses = [l.strip() for l in sec.splitlines() if re.search(r"\|\s*no\s*\|", l)]
    out["perf_pragma_misses"] = misses[:20]
    rows = [l.strip() for l in sec.splitlines() if l.strip().startswith("|") and "Modules" not in l and "cycles" not in l]
    out["perf_pragma_rows"] = rows[:12]
    hw = R.split("HW Interfaces")[1].split("\n+ ")[0] if "HW Interfaces" in R else ""
    out["hw_interfaces"] = [l.strip() for l in hw.splitlines() if l.strip().startswith("|")][:30]
    top = [l for l in R.splitlines() if re.match(r"\s*\|\+ FFT_TOP", l)]
    out["csynth_top_row"] = top[0].strip() if top else None
print(json.dumps(out, indent=1))
