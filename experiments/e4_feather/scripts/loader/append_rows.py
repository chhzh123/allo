#!/usr/bin/env python3
"""E4: append the row-wise-loader rows to results.csv, beside the old ones.

    append_rows.py <results.csv> <report root>

Reads the check.json / meta.json this experiment's runs left in
feather_rtl/S<N>/report/ and writes one row per run. The existing
`rtl_fixed_*` rows are left exactly as they are: the new rows sit next to them
so the before and after are both on the record.
"""

import csv
import json
import os
import sys

RUN = "/scratch/hc676/feather_loader/runs"
VARIANT = "corrected controller + row-wise weight loader (a row of PEs a cycle: the load is N^2, not N^3)"
WEIGHT_MODE = (
    "general (int8 -128..127 on the SPMW port / uint8 0..255 with zero point 0 on the RTL: "
    "the same stored bytes, seed 0)"
)
TILING = {4: "Mt=2 Kt=8 Nt=4", 8: "Mt=4 Kt=16 Nt=8", 16: "Mt=8 Kt=32 Nt=16"}
TILES = {4: 32768, 8: 4096, 16: 512}


def base(N, run_id, workload, source_dir):
    return {
        "run_id": run_id,
        "experiment_id": "E4",
        "system": "FEATHER RTL",
        "variant": VARIANT,
        "workload": workload,
        "array_size": f"{N}x{N}",
        "weight_mode": WEIGHT_MODE,
        "reorder_program": f"drivers' GEMM layout program for AW={N} (examples/feather/gemm.py), one program for every tile",
        "implementation_mode": "rtl_sim",
        "target_mhz": "300",
        "report_paths": f"experiments/e4_feather/feather_rtl/S{N}/report; run dir {RUN}/{source_dir}",
        "failure_reason": "",
        "source": "this agent (feather_loader dirs)",
    }


def main():
    results, root = sys.argv[1], sys.argv[2]
    with open(results, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        existing = list(reader)
    have = {r["run_id"] for r in existing}
    rows = []

    for N in (4, 8, 16):
        rep = os.path.join(root, f"S{N}", "report")
        L2, L3 = N * N, N * N * N

        # -- MODE 1: one feed, then every tile's activations through it
        chk = json.load(open(os.path.join(rep, "sim_rowload_gemm128_resident_general_check.json"), encoding="utf-8"))
        r = base(
            N,
            f"rtl_rowload_gemm128_N{N}_resident_general",
            f"gemm M=N=K=128 int8, drivers' tiling {TILING[N]} (loop n,m,k), {TILES[N]} tiles, host accumulates the K "
            f"partials; one feed of {L2} cycles (was {L3}), then every tile's activations through tile 0's resident "
            f"weights (MODE 1; datapath throughput, not the workload's result)",
            f"row_N{N}",
        )
        r.update(
            status=chk["status"],
            validation_pass=(
                f"{chk['tiles_ok']}/{chk['tiles']} tiles bit-exact vs the RTL-arithmetic model; "
                f"all {N * N} PE weight files bit-exact against the one-PE-a-cycle image "
                f"({chk['pe_files_wrong']} of {L3} slots wrong); signed_equals_rtl={chk['signed_equals_rtl']}"
            ),
            tiles=chk["tiles"],
            first_output_cycles=chk["first_output_cycles"],
            completion_cycles=chk["completion_cycles"],
            cycles_per_tile=chk["steady_interval_cycles"]["mean"],
            total_wall_s=round(chk["gen_s"] + chk["build_s"] + chk["sim_s"], 1),
        )
        rows.append(r)

        # -- MODE 0: a weight feed per tile, so the load is paid every tile
        chk = json.load(open(os.path.join(rep, "sim_rowload_gemm128_feed_general_check.json"), encoding="utf-8"))
        r = base(
            N,
            f"rtl_rowload_gemm128_N{N}_feed_general",
            f"gemm M=N=K=128 int8, drivers' tiling {TILING[N]} (loop n,m,k), {TILES[N]} tiles, host accumulates the K "
            f"partials; a weight feed per tile, {L2} cycles each (was {L3}) (MODE 0; the whole workload, host "
            f"reduction checked against numpy)",
            f"rowfeed_N{N}",
        )
        r.update(
            status=chk["status"],
            validation_pass=(
                f"{chk['tiles_ok']}/{chk['tiles']} tiles bit-exact vs the RTL-arithmetic model; "
                f"host reduction vs numpy: {chk['host_check']} ({chk['host_bad']} bad elements); "
                f"signed_equals_rtl={chk['signed_equals_rtl']}"
            ),
            tiles=chk["tiles"],
            first_output_cycles=chk["first_output_cycles"],
            completion_cycles=chk["completion_cycles"],
            cycles_per_tile=chk["steady_interval_cycles"]["mean"],
            total_wall_s=round(chk["gen_s"] + chk["build_s"] + chk["sim_s"], 1),
        )
        rows.append(r)

        # -- the single-tile validation matrix, every program / pattern / zero point / seed
        summ = json.load(open(os.path.join(rep, "sim_rowload_single_tiles_validate_summary.json"), encoding="utf-8"))
        runs = summ["runs"]
        npass = sum(1 for x in runs.values() if x.get("status") == "pass")
        badel = sum(x.get("bad_elements", 0) or 0 for x in runs.values())
        slots = sum(x.get("pe_files_wrong") or 0 for x in runs.values())
        firsts = sorted({x.get("first_output_cycles") for x in runs.values()})
        r = base(
            N,
            f"rtl_rowload_N{N}_single_tiles_validate",
            "single tiles, general weights: every BIRRD program the drivers ship x operand pattern "
            "(small/full/sparse) x zero point ((0,0),(7,5),(128,128)) x seed (0,1,2)",
            f"matrix row_N{N}_validate",
        )
        r.update(
            status="pass" if npass == len(runs) else "fail",
            validation_pass=(
                f"{npass}/{len(runs)} runs bit-exact vs the RTL-arithmetic model ({badel} bad elements); "
                f"{slots} PE weight-file slots wrong across the set"
            ),
            tiles=1,
            first_output_cycles=firsts[0] if len(firsts) == 1 else "",
            completion_cycles=firsts[0] if len(firsts) == 1 else "",
            cycles_per_tile="",
            total_wall_s=summ["elapsed_s"],
        )
        r["report_paths"] = f"experiments/e4_feather/feather_rtl/S{N}/report/sim_rowload_single_tiles_validate_summary.json; run dir /scratch/hc676/feather_loader/matrix/row_N{N}_validate"
        rows.append(r)

    # -- the two negative controls: image and hardware crossed. Both MUST fail;
    #    they are what shows the bit-exact check would catch a wrong weight
    #    protocol rather than passing on plausible-looking output.
    rep4 = os.path.join(root, "S4", "report")
    for name, fn, what in (
        ("old_rtl_new_image", "sim_rowload_negctl_old_rtl_new_image_check.json",
         "the published one-PE-a-cycle select fed the row-packed image"),
        ("new_rtl_old_image", "sim_rowload_negctl_new_rtl_old_image_check.json",
         "the row-wise select fed the one-byte-a-row image"),
    ):
        chk = json.load(open(os.path.join(rep4, fn), encoding="utf-8"))
        assert chk["status"] == "fail", f"{name} was supposed to fail"
        r = base(4, f"rtl_rowload_negctl_N4_{name}",
                 f"NEGATIVE CONTROL, expected to fail: {what}. gemm M=N=K=128, MODE 1, otherwise identical to "
                 f"rtl_rowload_gemm128_N4_resident_general",
                 f"cross_{'oldrtl_newimg' if name.startswith('old') else 'newrtl_oldimg'}_N4")
        r.update(
            status="fail",
            validation_pass=(
                f"{chk['tiles_ok']}/{chk['tiles']} tiles bit-exact; "
                f"{chk['pe_files_wrong']} of 64 PE weight-file slots wrong"
            ),
            tiles=chk["tiles"],
            first_output_cycles="",
            completion_cycles="",
            cycles_per_tile="",
            total_wall_s=round(chk["gen_s"] + chk["build_s"] + chk["sim_s"], 1),
            failure_reason=(
                "deliberate: the weight image and the select disagree. Kept because it is the evidence that the "
                "check fires -- the mesh still ran, still handshook and still emitted N well-formed rows a tile, "
                "and the check caught it anyway"
            ),
        )
        rows.append(r)

    new = [r for r in rows if r["run_id"] not in have]
    assert new, "nothing to add"
    with open(results, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        for r in new:
            w.writerow({k: r.get(k, "") for k in fields})
    for r in new:
        print(f"+ {r['run_id']:46s} {r['status']:4s} first={r['first_output_cycles']} compl={r['completion_cycles']}")
    print(f"{len(new)} rows appended to {results} ({len(existing)} were already there)")


if __name__ == "__main__":
    main()
