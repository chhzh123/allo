#!/scratch/hc676/allo-agent/bin/python3
import json, glob, os, re
E="/scratch/hc676/e2_hpfft"; A="/scratch/hc676/e2_allo"
def kv(fn):
    r={}
    if os.path.exists(fn):
        for tok in open(fn).read().split():
            if "=" in tok: k,v=tok.split("=",1); r[k]=v
    return r
def metrics(js):
    if not os.path.exists(js): return None
    m=json.load(open(js))["ModuleInfo"]["Metrics"]; top=[k for k in m if k in ("FFT_TOP","fft")][0]; t=m[top]
    return t
rows=[]
for d in sorted(glob.glob(f"{E}/n*/*"))+sorted(glob.glob(f"{A}/strided_*")):
    if not os.path.isdir(d) or d.endswith("gen_check"): continue
    if "/e2_allo/" in d:
        name="allo/"+os.path.basename(d); js=f"{d}/out.prj/solution1/solution1_data.json"
    else:
        name=os.path.relpath(d,E); js=f"{d}/build/FFT_300MHz/FFT_300MHz_data.json"
    h=kv(f"{d}/hls.done"); c=kv(f"{d}/cosim.done"); p=kv(f"{d}/pnr.done")
    t=metrics(js)
    ms = f"lat={t['Latency']['LatencyWorst']:>6} II={t['Latency']['PipelineII']:>6} est={t['Timing']['Estimate']} DSP={t['Area']['DSP']:>4} LUT={t['Area']['LUT']:>6} FF={t['Area']['FF']:>6} BR={t['Area']['BRAM_18K']:>3}" if t else "(no csynth)"
    ev={}
    if os.path.exists(f"{d}/cosim_events.json"):
        try: ev=json.load(open(f"{d}/cosim_events.json"))
        except Exception: ev={"err":1}
    val={}
    if os.path.exists(f"{d}/validate_wrapc_pc.json"):
        try: val=json.load(open(f"{d}/validate_wrapc_pc.json"))
        except Exception: val={"err":1}
    cs = f"cosim rc={c.get('rc')} {c.get('cosim_wall_s','')}s" if c else "cosim -"
    if ev: cs += f" lat1={ev.get('first_transform_completion_cycles')} II={ev.get('steady_interval_cycles_median')} ({ev.get('steady_interval_min')}-{ev.get('steady_interval_max')}) tx={ev.get('transforms_completed')}"
    if val: cs += f" rtl_ok={val.get('validation_pass')} err={val.get('max_abs_error','?'):.2e}" if 'max_abs_error' in val else " val?"
    ps = f"pnr rc={p.get('rc')} {p.get('total_wall_s','')}s" if p else "pnr -"
    if os.path.exists(f"{d}/pnr.done"):
        w=re.search(r"^E2 ROUTED WNS (\S+)", open(f"{d}/pnr.done").read(), re.M); ps += f" WNS={w.group(1)}" if w else " (no WNS)"
    print(f"{name:28s} hls rc={h.get('rc','-'):>2} {h.get('hls_wall_s','-'):>4}s | {ms} | {cs} | {ps}")
