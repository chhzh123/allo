"""E4b: extend e4_spmw_run.py (the previous agent's SPMW workload driver).

1. Conv programs at any N (the generator's conv_insts(N)).
2. A host-level check: the tile rows the cosim verified token by token
   (Y, the drivers' reference per tile) reduced the way the workload's host
   would -- gemm_reduce / conv_reduce -- against numpy's A @ B or the direct
   convolution of the same int8 tensors. The cosim's PASS proves the array
   produced Y; this proves Y is the workload's answer. Recorded in
   result.json as host_check / host_bad.
3. The operands' semantics are recorded: pattern, seed, value range.
"""
import sys

path = sys.argv[1]
src = open(path).read()


def sub(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:80])
    src = src.replace(old, new)


sub('''            if N == 4:
                inst = G.conv_insts()[t % 4]
            elif t % 2 == 0:''',
    '''            if N == 4:
                inst = G.conv_insts(N)[t % 4]
            elif t % 2 == 0:''')

# the host-level reduction of the verified tile rows
sub('''    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=int)
    brief = {k: result.get(k) for k in ("status", "cosim", "NT", "total_cycles", "first_out_cycles", "first_tile_done", "last_tile_done", "interval", "timing")}''',
    '''    # The host side: the rows the cosim verified (Y, one [N, N] block a tile,
    # the drivers' column order) reduced as the workload's host reduces them,
    # against numpy on the same int8 tensors (the stored bytes read as int8:
    # with --pattern full that is the whole mixed-sign range).
    result["operands"] = {"pattern": args.pattern, "seed": args.seed, "dtype": "int8 (stored bytes read as two's complement)",
                          "range": f"[0, {G.pattern_range(args.pattern, 0)}) as bytes" + (" = [-128, 127] as int8" if G.pattern_range(args.pattern, 0) == 256 else "")}
    if result["status"] == "pass" and args.workload in ("gemm", "conv") and not args.resident and not args.limit:
        rows = [Y[t * N : (t + 1) * N].astype(np.int64) for t in range(NT)]
        coords = [t[3] for t in tiles]
        if args.workload == "gemm":
            M, K, Nn = host["dims"]
            C = G.gemm_reduce(rows, coords, M, K, Nn, N, N)
            ref = host["A"].astype(np.int8).astype(np.int64) @ host["B"].astype(np.int8).astype(np.int64)
        else:
            P, Q, _ = host["geom"]
            Cin, H, Wd, M = host["dims"]
            C = G.conv_reduce(rows, coords, P, Q, M, N, N)
            ref = G.conv_reference(host["x"].astype(np.int8), host["w"].astype(np.int8), 0, 0, 1)
        result["host_check"] = "pass" if np.array_equal(C, ref) else "fail"
        result["host_bad"] = int((C != ref).sum())
        result["host_out_range"] = [int(ref.min()), int(ref.max())]
    elif args.resident:
        result["host_check"] = None  # every tile through tile 0's weights: not the workload's result
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=int)
    brief = {k: result.get(k) for k in ("status", "cosim", "host_check", "host_bad", "NT", "total_cycles", "first_out_cycles", "first_tile_done", "last_tile_done", "interval", "timing")}''')

open(path, "w").write(src)
print("patched", path)
