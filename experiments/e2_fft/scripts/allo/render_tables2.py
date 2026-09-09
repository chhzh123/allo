#!/scratch/hc676/allo-agent/bin/python3
"""Render results.csv / hls_estimates.csv of both systems into markdown tables (summary.md next to each results.csv)."""
import csv, os, re
OUT = "/scratch/hc676/spmw_eval_remaining_2026-09-06/e2_fft"
def key(r):
    c = r.get("config", ""); m = re.match(r"UF(\d+)", c)
    k = int(m.group(1)) if m else (100 if "no_Stage" in c else 200 if "original" in c else (0 if "wrap_io=True" in c else 1))
    return (int(r["transform_size"]), k, c, r["run_id"])
def esc(s): return str(s).replace("|", "\\|")
def table(rows, cols, title, out):
    if not rows: return
    out.append(f"\n### {title}\n")
    out.append("| " + " | ".join(c[1] for c in cols) + " |"); out.append("|" + "---|" * len(cols))
    for r in rows: out.append("| " + " | ".join(esc(r.get(c[0], "")) for c in cols) + " |")
for sub in ("hpfft", "allo"):
    fn = f"{OUT}/{sub}/results.csv"
    if not os.path.exists(fn): continue
    rows = sorted(csv.DictReader(open(fn)), key=key); out = []
    cos = [r for r in rows if r["implementation_mode"] == "cosim"]
    table([r for r in cos if not r["run_id"].endswith("_lcg")],
          [("transform_size", "N"), ("config", "config"), ("status", "status"), ("stimulus", "stimulus"), ("validation_pass", "RTL vs numpy"), ("max_abs_error", "max abs err"), ("first_output_cycles", "first out (cyc)"),
           ("completion_cycles", "1st transform complete (cyc)"), ("steady_interval_cycles", "steady interval (cyc)"), ("transforms_streamed", "fed"), ("transforms_completed", "completed"), ("run_by", "run by"), ("failure_reason", "failure")],
          "RTL cosimulation (xsim), cycles at 3.333 ns -- primary rows", out)
    table([r for r in cos if r["run_id"].endswith("_lcg")],
          [("transform_size", "N"), ("config", "config"), ("status", "status"), ("validation_pass", "RTL vs numpy"), ("max_abs_error", "max abs err"), ("first_output_cycles", "first out (cyc)"),
           ("completion_cycles", "1st transform complete (cyc)"), ("steady_interval_cycles", "steady interval (cyc)"), ("transforms_streamed", "fed"), ("transforms_completed", "completed"), ("failure_reason", "failure")],
          "RTL cosimulation, previous agent's runs with the LCG stimulus (kept as cross-checks; run_id suffix _lcg)", out)
    table([r for r in rows if r["implementation_mode"] == "pnr_ooc"],
          [("transform_size", "N"), ("config", "config"), ("status", "status"), ("lut", "LUT"), ("ff", "FF"), ("dsp", "DSP"), ("bram_18k_equiv", "BRAM18-eq"), ("bram_36k", "RAMB36"), ("bram_18k", "RAMB18"), ("uram", "URAM"),
           ("wns_ns", "WNS ns"), ("tns_ns", "TNS ns"), ("unrouted", "unrouted"), ("synth_s", "synth s"), ("place_s", "place s"), ("route_s", "route s"), ("total_wall_s", "total s"), ("run_by", "run by"), ("failure_reason", "failure")],
          "Vivado 2023.2 out-of-context synth + place + route at 3.333 ns (post-route numbers)", out)
    hfn = f"{OUT}/{sub}/hls_estimates.csv"
    if os.path.exists(hfn):
        hrows = sorted(csv.DictReader(open(hfn)), key=key)
        table(hrows, [("transform_size", "N"), ("config", "config"), ("latency_cycles", "latency (cyc)"), ("interval_cycles", "interval (cyc)"), ("lut", "LUT"), ("ff", "FF"), ("dsp", "DSP"), ("bram_18k", "BRAM18"), ("uram", "URAM"),
                      ("timing_estimate_ns", "HLS est. ns"), ("timing_budget_ns", "budget ns"), ("hls_wall_s", "hls s"), ("csim_pass", "csim"), ("cmodel_validation_pass", "C model vs numpy"), ("run_by", "run by"), ("notes", "note")],
              "Vitis HLS 2023.2 csynth estimates (hls_estimates.csv; not routed numbers)", out)
        ig = [r for r in hrows if r.get("ignored_directives") and not r["run_id"].endswith("_s_csynth")]
        if ig:
            out.append("\n### Directives Vitis HLS 2023.2 ignored, removed or could not honour (hls.log / csynth.rpt)\n")
            out.append("| N | config | warnings (code, meaning, count) |"); out.append("|---|---|---|")
            for r in ig: out.append(f"| {r['transform_size']} | {esc(r['config'])} | {esc(r['ignored_directives'])} |")
    open(f"{OUT}/{sub}/summary.md", "w").write("\n".join(out) + "\n"); print(sub, "->", f"{OUT}/{sub}/summary.md", len(rows), "rows")
