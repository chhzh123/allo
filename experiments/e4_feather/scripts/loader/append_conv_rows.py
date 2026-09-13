#!/usr/bin/env python3
"""E4: append the row-wise-loader **conv** rows to results.csv.

    append_conv_rows.py <results.csv> <report root>

`append_rows.py` re-measured the GEMM workload after the row-wise loader
landed and this does the same for the conv one, which was left carrying an
`N^3` weight feed. The `rtl_fixed_conv_*` rows stay exactly as they are; these
sit beside them.

The two conv baselines were re-run as well, on a generator merged from the two
that had diverged -- the conv-at-any-AW programs live in one and `--loader row`
in the other -- and they reproduce the recorded numbers to the cycle at every
size. That is what makes the new rows comparable to the old ones rather than
just adjacent to them.
"""

import csv
import json
import os
import sys

RUN = "/scratch/hc676/feather_conv/runs"
VARIANT = (
    "corrected controller + row-wise weight loader "
    "(a row of PEs a cycle: the load is N^2, not N^3)"
)
WEIGHT_MODE = (
    "general (int8 -128..127 on the SPMW port / uint8 0..255 with zero point 0 on the RTL: "
    "the same stored bytes, seed 0)"
)
#: The drivers' conv tiling: RS = 9 padded up to a multiple of the array side.
TILING = {
    4: ("RS 9 padded to 12, tile = [4 taps x 4 channels] x 4 out-channels", 196608),
    8: ("RS 9 padded to 16, tile = [8 taps x 8 channels] x 8 out-channels", 32768),
    16: ("RS 9 padded to 16, tile = [16 taps x 16 channels] x 16 out-channels", 4096),
}


def base(N, run_id, workload, source_dir):
    return {
        "run_id": run_id,
        "experiment_id": "E4",
        "system": "FEATHER RTL",
        "variant": VARIANT,
        "workload": workload,
        "array_size": f"{N}x{N}",
        "weight_mode": WEIGHT_MODE,
        "reorder_program": (
            f"conv reduction program for AW={N}: every column sum of the tile into one "
            "output column, one program per output position in a line"
        ),
        "implementation_mode": "rtl_sim",
        "target_mhz": "300",
        "report_paths": (
            f"experiments/e4_feather/feather_rtl/S{N}/report; run dir {RUN}/{source_dir}"
        ),
        "failure_reason": "",
        "source": "this agent (feather_conv dirs)",
    }


def main():
    results, root = sys.argv[1], sys.argv[2]
    with open(results, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        existing = list(reader)
    have = {r["run_id"] for r in existing}
    rows = []

    for N in (4, 8, 16):
        rep = os.path.join(root, f"S{N}", "report")
        tiling, tiles = TILING[N]
        L2, L3 = N * N, N * N * N
        shape = (
            "conv 16x16x64 -> 64ch 3x3 s1 p1 NHWC int8, drivers' tiling "
            f"({tiling}), {tiles} tiles, host accumulates over channel and tap blocks"
        )

        for mode, tag, tail in (
            (
                "resident",
                "resident",
                f"; one feed of {L2} cycles (was {L3}), then every tile's activations through "
                "tile 0's resident weights and program (MODE 1; datapath throughput, not the "
                "workload's result)",
            ),
            (
                "feed",
                "feed",
                f"; a weight feed per tile, {L2} cycles each (was {L3}) (MODE 0; the whole "
                "workload, host reduction checked against numpy)",
            ),
        ):
            path = os.path.join(rep, f"sim_rowload_conv_{tag}_general_check.json")
            if not os.path.isfile(path):
                print(f"  skipped, no {os.path.basename(path)}")
                continue
            chk = json.load(open(path, encoding="utf-8"))
            run_id = f"rtl_rowload_conv_N{N}_{tag}_general"
            if run_id in have:
                print(f"  {run_id} already present, skipped")
                continue
            row = base(N, run_id, shape + tail, f"conv_row_N{N}_m{'1' if tag=='resident' else '0'}")
            checks = [
                f"{chk['tiles_ok']}/{chk['tiles']} tiles bit-exact vs the RTL-arithmetic model",
                f"signed_equals_rtl={chk['signed_equals_rtl']}",
            ]
            if chk.get("pe_files_ok") is not None:
                checks.insert(
                    1,
                    f"all {N * N} PE weight files bit-exact against the one-PE-a-cycle image "
                    f"({chk['pe_files_wrong']} of {L3} slots wrong)",
                )
            if chk.get("host_check") is not None:
                checks.insert(1, f"host reduction vs numpy: {chk['host_check']}")
            row.update(
                status=chk["status"],
                validation_pass="; ".join(checks),
                tiles=chk["tiles"],
                first_output_cycles=chk["first_output_cycles"],
                completion_cycles=chk["completion_cycles"],
                cycles_per_tile=chk["steady_interval_cycles"]["mean"],
                total_wall_s=round(chk["gen_s"] + chk["build_s"] + chk["sim_s"], 1),
            )
            rows.append(row)

    if not rows:
        print("nothing to add")
        return 0
    with open(results, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in existing + rows:
            writer.writerow(r)
    for r in rows:
        print(
            f"  + {r['run_id']:38s} first {r['first_output_cycles']:>6} "
            f"completion {r['completion_cycles']:>10} cyc/tile {r['cycles_per_tile']}"
        )
    print(f"{len(rows)} rows added, {len(existing)} kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())
