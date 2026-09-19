#!/usr/bin/env python3
"""The block's cycle count on each engine, summed from measured rates.

Every rate below is the *marginal* cost of one normalisation row -- two row
counts run, the difference taken -- at the exact beat count the block uses, so
the totals are a sum of measurements and not a fit.  The pair that is
extrapolated is named as such.
"""

from block_workload import GEMMS, NORMS, gemm_tiles, macs, norm_rows

#: Marginal cycles per normalisation row, `(activation, beats) -> cycles`.
#: SPMW from the RTL cosimulation of the assembled 16x16 array, Gemmini from
#: xsim on its own generated Verilog.  Both bit-exact against
#: `spmw_block_ref` on the same stimulus.
#:
#:   activation  beats   run pair                      SPMW      Gemmini
#:   layernorm     16    nrow 4 -> 8            497 ->  621  526 -> 1046
#:   softmax        4    nrow 4 -> 8            343 ->  359  170 ->  334
#:   igelu         16    nrow 4 -> 8            415 ->  479   70 ->  134
#:   igelu         64    nrow 4 -> 8            703 ->  959  262 ->  518
#:   none          16    nrow 4 -> 8            415 ->  479    -- ->  134
#:   none          48    nrow 4 -> 8            607 ->  799  198 ->  390
PER_ROW = {
    "spmw":    {("layernorm", 16): 31, ("softmax", 4): 4,
                ("igelu", 16): 16, ("igelu", 64): 64,
                ("none", 16): 16, ("none", 48): 48},
    "gemmini": {("layernorm", 16): 130, ("softmax", 4): 41,
                ("igelu", 16): 16, ("igelu", 64): 64,
                ("none", 16): 16, ("none", 48): 48},
}

#: Cycles for one 16x16x16 mesh operation in the steady state, from E3, where
#: both meshes were measured on the same tiled int8 GEMM at the same width.
#: The mesh is unchanged on both sides here -- SPMW's cell is E3's verbatim and
#: Gemmini's is the same `MeshWithDelays` -- so these carry over, and they are
#: the one pair of numbers in this table that E8 did not re-measure.
PER_TILE = {"spmw": 16, "gemmini": 18}

#: Achieved period, from place and route on `xcu280-fsvh2892-2L-e`.
PERIOD_NS = {"spmw": None, "gemmini": 29.029}


def totals(engine):
    rows = []
    norm = 0
    for name, (act, nrow, beats) in norm_rows().items():
        key = (act, beats)
        if key not in PER_ROW[engine]:
            raise SystemExit(f"{engine}: no measured rate for {key}")
        c = nrow * PER_ROW[engine][key]
        rows.append((name, act, nrow, beats, PER_ROW[engine][key], c))
        norm += c
    gemm = sum(gemm_tiles().values()) * PER_TILE[engine]
    return rows, gemm, norm


def main():
    out = {}
    for engine in ("spmw", "gemmini"):
        rows, gemm, norm = totals(engine)
        out[engine] = (rows, gemm, norm)

    print(f"the block: {macs() / 1e6:.1f}M MACs, "
          f"{sum(gemm_tiles().values()):,} mesh tiles, "
          f"{sum(r * b * 16 for _, _, r, b in [(n, a, r, b) for n, (a, r, b) in norm_rows().items()]):,}"
          " elements through the scale path")
    print()
    hdr = f"{'pass':10s} {'activation':11s} {'rows':>5s} {'beats':>6s}"
    print(hdr + f" {'SPMW/row':>9s} {'SPMW':>9s} {'Gem/row':>8s} {'Gemmini':>9s}")
    for i, (name, act, nrow, beats, _, _) in enumerate(out["spmw"][0]):
        s = out["spmw"][0][i][5]
        g = out["gemmini"][0][i][5]
        print(f"{name:10s} {act:11s} {nrow:5d} {beats:6d} "
              f"{PER_ROW['spmw'][(act, beats)]:9d} {s:9,d} "
              f"{PER_ROW['gemmini'][(act, beats)]:8d} {g:9,d}")
    sn, gn = out["spmw"][2], out["gemmini"][2]
    sg, gg = out["spmw"][1], out["gemmini"][1]
    print(f"{'scale path':10s} {'':11s} {'':5s} {'':6s} {'':9s} {sn:9,d} "
          f"{'':8s} {gn:9,d}   {gn / sn:.2f}x")
    print(f"{'mesh':10s} {'':11s} {'':5s} {'':6s} "
          f"{PER_TILE['spmw']:9d} {sg:9,d} {PER_TILE['gemmini']:8d} "
          f"{gg:9,d}   {gg / sg:.2f}x")
    print(f"{'total':10s} {'':11s} {'':5s} {'':6s} {'':9s} "
          f"{sn + sg:9,d} {'':8s} {gn + gg:9,d}   {(gn + gg) / (sn + sg):.2f}x")

    print()
    for engine in ("spmw", "gemmini"):
        p = PERIOD_NS[engine]
        c = sum(out[engine][1:])
        if p:
            print(f"{engine:8s} {c:9,d} cycles x {p:6.3f} ns = "
                  f"{c * p / 1e6:8.3f} ms   ({1000 / p:5.1f} MHz)")
        else:
            print(f"{engine:8s} {c:9,d} cycles x (route pending)")


if __name__ == "__main__":
    main()
