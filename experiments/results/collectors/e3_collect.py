#!/usr/bin/env python3
"""E3 collector: the chained GPT-2 medium / LLaMA-7B block runs on the 302
bitstream -> results.csv, results.json and README.md in the bundle.

    python3 e3_collect.py <bundle>/e3_tpu <run_dir> [<run_dir> ...]

Every number comes from a results.json written by scripts/spmw_gpt_block.py
(one per run directory) or from the kernel's own Vivado reports; nothing is
estimated here.
"""
import csv
import json
import os
import re
import shutil
import sys

STAGE_ORDER = [
    "ln1", "Q", "K", "V", "score", "mask", "smax", "ssum", "snorm", "ctx", "O",
    "res1", "ln2", "FFN1", "FFN3", "act", "FFN2", "res2",
]
DEVICE_STAGES = {"Q", "K", "V", "score", "smax", "ssum", "snorm", "ctx", "O", "FFN1", "FFN3", "FFN2"}

# The 302 kernel's routed resources (kernel hierarchy only, from
# /scratch/hc676/e3_util_kernel.rpt and e3_timing.rpt on the linked design).
KERNEL = {
    "bitstream": "gptkern_302 (gptstage_v1 --size 16, 300 MHz, ExtraTimingOpt)",
    "lut": 132922, "ff": 116913, "dsp": 304, "bram_36k": 75, "bram_18k": 37,
    "bram_18k_equiv": 2 * 75 + 37, "bram_tiles": 93.5, "uram": 0,
    "wns_ns": 0.016, "tns_ns": 0.0, "whs_ns": 0.006, "clock_mhz": 300,
}


def load(run_dir):
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as h:
        return json.load(h)


def summarise(res):
    reps = res["reps"]
    n = len(reps)
    keys = ["wall_s", "kernel_s", "transfer_s", "pack_s", "host_math_s"]
    mean = {k: sum(r[k] for r in reps) / n for k in keys}
    best = {k: min(r[k] for r in reps) for k in keys}
    launches = reps[0]["launches"]
    mism = sum(r["mismatched_values"] for r in reps)
    stages = {}
    for name in STAGE_ORDER:
        rows = [r["stages"][name] for r in reps if name in r["stages"]]
        if not rows:
            continue
        stages[name] = {
            "launches": rows[0]["launches"],
            "kernel_ms": 1e3 * sum(x["kernel_s"] for x in rows) / len(rows),
            "load_ms": 1e3 * sum(x["load_s"] for x in rows) / len(rows),
            "read_ms": 1e3 * sum(x["read_s"] for x in rows) / len(rows),
            "pack_ms": 1e3 * sum(x["pack_s"] for x in rows) / len(rows),
            "host_ms": 1e3 * sum(x["host_s"] for x in rows) / len(rows),
            "device": name in DEVICE_STAGES,
        }
    return {
        "model": res["model"], "shape": res["shape"], "seed": res["seed"],
        "reps": n, "launches": launches, "mismatched_values": mism,
        "mean": mean, "best": best, "stages": stages,
        "device_stage_kernel_ms": 1e3 * sum(s["kernel_s"] for r in reps for k, s in r["stages"].items() if k in DEVICE_STAGES) / n,
        "host_stage_ms": 1e3 * sum(s["host_s"] for r in reps for k, s in r["stages"].items() if k not in DEVICE_STAGES) / n,
    }


def main():
    out = sys.argv[1]
    runs = sys.argv[2:]
    os.makedirs(os.path.join(out, "reports"), exist_ok=True)
    rows = []
    summary = {"kernel": KERNEL, "runs": []}
    for run in runs:
        res = load(run)
        s = summarise(res)
        tag = os.path.basename(run.rstrip("/"))
        s["run_dir"] = run
        summary["runs"].append(s)
        shutil.copy(os.path.join(run, "results.json"), os.path.join(out, "reports", f"{tag}_results.json"))
        rows.append({
            "run_id": tag, "experiment_id": "E3", "system": "SPMW mini-TPU (gptkern_302)",
            "workload": f"{s['model']} one block, seq {s['shape']['seq']}",
            "numeric_semantics": "int8 x int8 -> int32, host requantisation (see README)",
            "implementation_mode": "board", "target_mhz": 300,
            "status": "pass" if s["mismatched_values"] == 0 else "functional_fail",
            "validation_pass": s["mismatched_values"] == 0,
            "reps": s["reps"], "launches_per_block": s["launches"],
            "device_kernel_s_mean": round(s["mean"]["kernel_s"], 4),
            "device_kernel_s_best": round(s["best"]["kernel_s"], 4),
            "transfer_s_mean": round(s["mean"]["transfer_s"], 4),
            "pack_s_mean": round(s["mean"]["pack_s"], 3),
            "host_math_s_mean": round(s["mean"]["host_math_s"], 4),
            "wall_s_mean": round(s["mean"]["wall_s"], 2),
            "lut": KERNEL["lut"], "ff": KERNEL["ff"], "dsp": KERNEL["dsp"],
            "bram_18k_equiv": KERNEL["bram_18k_equiv"], "uram": KERNEL["uram"],
            "wns_ns": KERNEL["wns_ns"], "tns_ns": KERNEL["tns_ns"],
            "hybrid": "yes: LN/RMSNorm, RoPE, mask, GELU/SiLU-gate, residual, requantisation on the host",
            "report_paths": f"reports/{tag}_results.json",
            "failure_reason": "",
        })
    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as h:
        json.dump(summary, h, indent=1)
    # A per-stage table for the README.
    lines = []
    for s in summary["runs"]:
        lines.append(f"\n### {s['model']} ({s['run_dir']}, {s['reps']} reps, seed {s['seed']})\n")
        lines.append(f"{s['launches']} launches a block; mismatched values across all reps: {s['mismatched_values']}.\n")
        lines.append("| stage | where | launches | kernel ms | load ms | read ms | pack ms | host ms |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for name, st in s["stages"].items():
            where = "device" if st["device"] else "host"
            lines.append(f"| {name} | {where} | {st['launches']} | {st['kernel_ms']:.2f} | {st['load_ms']:.2f} | {st['read_ms']:.2f} | {st['pack_ms']:.1f} | {st['host_ms']:.2f} |")
        m = s["mean"]
        lines.append(f"\nMeans over reps: device kernel {m['kernel_s']*1e3:.1f} ms, transfers {m['transfer_s']*1e3:.1f} ms, "
                     f"packing {m['pack_s']:.2f} s, host math {m['host_math_s']*1e3:.1f} ms, wall {m['wall_s']:.2f} s "
                     f"(best kernel {s['best']['kernel_s']*1e3:.1f} ms).")
    with open(os.path.join(out, "stage_tables.md"), "w", encoding="utf-8") as h:
        h.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
