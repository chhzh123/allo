#!/scratch/hc676/allo-agent/bin/python3
"""Render results.csv of both systems into markdown tables (summary.md next to each results.csv)."""
import csv, os, re, sys
OUT = "/scratch/hc676/spmw_eval_remaining_2026-09-06/e2_fft"
def key(r):
    m = re.match(r"UF(\d+)", r["config"]); c = int(m.group(1)) if m else (100 if "no_Stage" in r["config"] else 200)
    return (int(r["transform_size"]), c, r["config"])
for sub in ("hpfft", "allo"):
    fn = f"{OUT}/{sub}/results.csv"
    if not os.path.exists(fn): continue
    rows = sorted(csv.DictReader(open(fn)), key=key)
    out = []
    def table(mode, cols, title):
        rs = [r for r in rows if r["implementation_mode"] == mode]
        if not rs: return
        out.append(f"\n### {title}\n")
        out.append("| " + " | ".join(c[1] for c in cols) + " |"); out.append("|" + "---|" * len(cols))
        for r in rs:
            out.append("| " + " | ".join((r[c[0]] if c[0] in r else "") for c in cols) + " |")
    table("csynth", [("transform_size","N"),("config","config"),("status","status"),("completion_cycles","latency (cyc)"),("steady_interval_cycles","interval (cyc)"),("lut","LUT"),("ff","FF"),("dsp","DSP"),("bram_18k_equiv","BRAM18"),("uram","URAM"),("hls_wall_s","hls s"),("validation_pass","C-model vs numpy"),("max_abs_error","max abs err"),("failure_reason","note")], "csynth (Vitis HLS 2023.2, xcu280, 3.333 ns; HLS estimates)")
    table("cosim", [("transform_size","N"),("config","config"),("status","status"),("validation_pass","RTL vs numpy"),("max_abs_error","max abs err"),("max_norm_error","max norm err"),("first_output_cycles","first out (cyc)"),("completion_cycles","1st transform done (cyc)"),("steady_interval_cycles","steady interval (cyc)"),("hls_wall_s","setup s"),("failure_reason","note")], "RTL cosimulation (xsim), cycles at 3.333 ns")
    table("pnr_ooc", [("transform_size","N"),("config","config"),("status","status"),("lut","LUT"),("ff","FF"),("dsp","DSP"),("bram_18k_equiv","BRAM18"),("uram","URAM"),("wns_ns","WNS ns"),("tns_ns","TNS ns"),("vivado_synth_s","synth s"),("place_s","place s"),("route_s","route s"),("total_wall_s","total s"),("failure_reason","note")], "Vivado 2023.2 out-of-context place-and-route at 3.333 ns (post-route)")
    rs = [r for r in rows if r["implementation_mode"] == "csynth" and r["ignored_directives"]]
    if rs:
        out.append("\n### Directives Vitis HLS 2023.2 ignored, removed or could not honour (from hls.log / csynth.rpt)\n")
        out.append("| N | config | warnings (code, meaning, count) |"); out.append("|---|---|---|")
        for r in rs: out.append(f"| {r['transform_size']} | {r['config']} | {r['ignored_directives']} |")
    open(f"{OUT}/{sub}/summary.md", "w").write("\n".join(out) + "\n"); print(sub, "->", f"{OUT}/{sub}/summary.md", len(rows), "rows")
