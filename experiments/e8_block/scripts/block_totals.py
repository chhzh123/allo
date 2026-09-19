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

#: Cycles for one 16x16x16 mesh operation in the steady state, **measured
#: here**, on these two top levels, with the weights changing every tile.
#:
#: Gemmini: `gem_mesh_bench.py` drives `MeshWithDelays` through `MxuVpuNorm`
#: and reads `io_mesh_out`, the raw int32 result before the scale path.
#: Sixteen tiles, sixteen different weight matrices, zero wrong rows,
#: `interval_min == interval_max == 18`.  That is `S` rows of activations and
#: a two-cycle request handshake, with the *next* tile's weights shifting in
#: through `d` behind the current tile's activations -- the reload is free
#: because it is overlapped.
#:
#: SPMW: a one-beat scale path so the mesh is what is measured, at 8, 16, 24
#: and 32 resident tiles, in both multiply bindings:
#:
#:     tiles      8     16     24     32
#:     fabric   322    454    621    784
#:     DSP      423    484    653    815
#:
#: The small-tile points are **not** mesh-bound -- with one beat of scale path
#: the pipeline's own fill is 300 to 400 cycles and that is what they measure,
#: which is why the DSP binding, whose fill is larger, looks *slower* at 8
#: tiles and faster per tile across all four.  Fitting all four gives 19.4 and
#: 16.8, and neither is the mesh.  Taking the 16-to-32 span, where the mesh
#: does bind, gives 20.63 and 20.69 -- the two bindings agree, and so does the
#: structure: 16 steps plus a weight file of `dim * tiles/4` words loaded
#: serially down each row, which is 4 a tile.
#:
#: **SPMW's 16-cycle interval and Gemmini's 18 are not the same measurement**,
#: and E3 says so: SPMW's excludes the weight reload because the file is
#: resident, Gemmini's includes one it overlaps.  Comparing them directly is
#: the mistake E4 made with DSP binding, in the other direction.  The pair
#: below is like for like -- both with a new weight matrix every tile.
PER_TILE = {"spmw": 20.6, "gemmini": 18.0}

#: Achieved period, from place and route on `xcu280-fsvh2892-2L-e`, all
#: three with zero unrouted nets.  SPMW twice because the multiply binding
#: that matches Gemmini changed between E3 and E8: E3's `MxuVpu` spent no
#: DSPs, so SPMW's multiplies were bound to fabric to match; this design's
#: `MxuVpuNorm` spends 500, so the DSP-allowed build is the like-for-like one
#: and the fabric build is the other end of the same trade.
#: Gemmini's best of four routes: `latency = 4`, retimed, constrained at
#: 3.333 ns, WNS -25.375.  The other three land within 1.6 ns of it.
PERIOD_NS = {"spmw": 3.261, "spmw_fabric": 3.852, "gemmini": 28.708}


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
    gemm = round(sum(gemm_tiles().values()) * PER_TILE[engine])
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
          f"{PER_TILE['spmw']:9.1f} {sg:9,d} {PER_TILE['gemmini']:8.1f} "
          f"{gg:9,d}   {gg / sg:.2f}x")
    print(f"{'total':10s} {'':11s} {'':5s} {'':6s} {'':9s} "
          f"{sn + sg:9,d} {'':8s} {gn + gg:9,d}   {(gn + gg) / (sn + sg):.2f}x")

    print()
    base = None
    for engine in ("gemmini", "spmw", "spmw_fabric"):
        p = PERIOD_NS[engine]
        c = sum(out["spmw" if engine.startswith("spmw") else engine][1:])
        ms = c * p / 1e6
        if base is None:
            base = ms
        print(f"{engine:12s} {c:9,d} cycles x {p:6.3f} ns = {ms:7.3f} ms  "
              f"({1000 / p:5.1f} MHz)"
              + (f"   {base / ms:.1f}x" if ms != base else ""))
    print()
    print("The cycle count is the same for both SPMW bindings: the DSP "
          "binding\nchanges the pipeline's fill, not its rates -- LayerNorm "
          "is 31 cycles a row\neither way (597 -> 721 against 497 -> 621) "
          "and the mesh is 20.6 either way.")


if __name__ == "__main__":
    main()
