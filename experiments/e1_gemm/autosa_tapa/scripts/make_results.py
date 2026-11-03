# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build `autosa_tapa/results.csv` from the run's logs, beside E1's Vitis HLS rows.

    python3 make_results.py experiments/e1_gemm/autosa_tapa
"""
import csv
import os
import re
import sys

ROOT = sys.argv[1]
LOGS = os.path.join(ROOT, "logs")
E1 = os.path.join(ROOT, "..")


def lines(name):
    with open(os.path.join(LOGS, name), encoding="utf-8") as handle:
        return handle.read().splitlines()


def main():
    rows = {s: {"array": f"{s}x{s}"} for s in (4, 8, 16, 32)}
    for line in lines("compile_times.txt"):
        m = re.match(r"S=(\d+) jobs=(\d+) rc=0 wall_s=([0-9.]+)", line)
        rows[int(m.group(1))][f"tapa_compile_j{m.group(2)}_s"] = m.group(3)
    for line in lines("phases.txt"):
        m = re.match(
            r"S(\d+)_j8\s+analyze\s+([0-9.]+)\s+hls\s+([0-9.]+)\s+rtl\s+([0-9.]+)"
            r"\s+pack\s+([0-9.]+).*tasks\s+(\d+)\s+sum of HLS jobs\s+([0-9.]+)",
            line,
        )
        if m:
            row = rows[int(m.group(1))]
            row.update(
                tapa_analyze_s=m.group(2),
                tapa_hls_j8_s=m.group(3),
                tapa_rtl_assembly_s=m.group(4),
                tapa_pack_s=m.group(5),
                tapa_tasks=m.group(6),
                tapa_sum_of_hls_jobs_s=m.group(7),
            )
    for line in lines("cosim_results.txt"):
        m = re.match(r"S=(\d+) .*cycles (\d+) .*check=(\S+)", line)
        rows[int(m.group(1))].update(tapa_cycles=m.group(2), tapa_check=m.group(3))
    for line in lines("util.txt"):
        m = re.match(
            r"S(\d+)_j8\s+LUT\s+(\d+)\s+FF\s+(\d+)\s+DSP\s+(\d+)\s+BRAM18\s+(\d+)"
            r"\s+WNS\s+([-+0-9.]+)\s+(\d+) MHz",
            line,
        )
        rows[int(m.group(1))].update(
            tapa_lut=m.group(2),
            tapa_ff=m.group(3),
            tapa_dsp=m.group(4),
            tapa_bram18=m.group(5),
            tapa_wns_ns=m.group(6),
            tapa_fmax_mhz=m.group(7),
        )
    # E1's AutoSA rows through Vitis HLS: wide (--host-serialize) from 8x8, narrow at 4x4.
    with open(os.path.join(E1, "results.csv"), encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            if rec["system"] != "AutoSA":
                continue
            s = int(rec["array"].split("x")[0])
            if rec["interface"] != ("narrow" if s == 4 else "wide"):
                continue
            rows[s].update(
                hlsc_interface=rec["interface"],
                hlsc_cycles=rec["cycles_data_beats"],
                hlsc_lut=rec["lut"],
                hlsc_ff=rec["ff"],
                hlsc_dsp=rec["dsp"],
                hlsc_bram18=rec["bram_18k_equiv"],
                hlsc_wns_ns=rec["wns_ns"],
                hlsc_fmax_mhz=rec["fmax_mhz"],
            )
    with open(os.path.join(E1, "compile_time.csv"), encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            s = int(rec["array"].split("x")[0])
            if (
                rec["framework"] == "autosa"
                and rec["stage"] == "hls_csynth"
                and rec["interface"] == rows[s].get("hlsc_interface")
            ):
                rows[s]["hlsc_csynth_s"] = rec["seconds"]
    fields = ["array"] + sorted(
        {k for row in rows.values() for k in row} - {"array"},
        key=lambda k: (not k.startswith("tapa"), k),
    )
    with open(
        os.path.join(ROOT, "results.csv"), "w", newline="", encoding="utf-8"
    ) as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for s in sorted(rows):
            writer.writerow(rows[s])


if __name__ == "__main__":
    main()
