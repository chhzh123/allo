import csv
for label, path, keys in (
    ("SPMW mesh 8x8", "/scratch/hc676/spmw_eval_remaining_2026-09-06/e1_gemm/spmw/results.csv",
     {"size": "8", "mode": "pnr"}),
    ("SPMW kernel 8x8", "/scratch/hc676/spmw_eval_remaining_2026-09-06/e1_gemm/spmw_mem/results.csv",
     {"size": "8", "mode": "pnr"}),
    ("AutoSA 8x8", "/scratch/hc676/spmw_eval_remaining_2026-09-06/e1_gemm/autosa/results.csv",
     {"array_shape": "8x8", "implementation_mode": "pnr_ooc"}),
):
    try:
        rows = list(csv.DictReader(open(path)))
    except OSError:
        print("  %-16s no results" % label)
        continue
    for r in rows:
        if all(r.get(k) == v for k, v in keys.items()):
            print("  %-16s synth=%s place=%s route=%s total=%s"
                  % (label, r.get("synth_s"), r.get("place_s"), r.get("route_s"),
                     r.get("vivado_total_s")))
            break
