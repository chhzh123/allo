#!/usr/bin/env python3
"""The transformer block on three engines, with VTA's scope stated honestly.

The workload is E8's -- imported, not restated -- so all three engines are
counted against the same 12,800 mesh tiles and the same eleven scale-path
passes.

The difference VTA makes to the table is not a number, it is a column that
cannot be filled. `TensorAlu` computes `min`, `max`, `add` and `shr` and has
no multiplier at all, so LayerNorm, softmax and IGELU are not slow on VTA --
they are absent. What it can do is the plain requantisation, which is an
arithmetic shift, and a ReLU, which is a max against zero.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "e8_block", "scripts"))

from block_workload import gemm_tiles, norm_rows  # noqa: E402

#: Measured marginal cycles per normalisation row, `(activation, beats)`.
#: VTA's entries exist only for the activations its ALU can express; the rest
#: are `None`, meaning the pass leaves the chip.
#:
#:   VTA, xsim on its own Verilog: GEMM 258/514/1026 cycles at 16/32/64 tiles
#:   (16.0 a tile exactly), ALU 258/514 cycles at 256/512 rows (1.0 a row).
PER_ROW = {
    "spmw":    {("layernorm", 16): 31, ("softmax", 4): 4,
                ("igelu", 16): 16, ("igelu", 64): 64,
                ("none", 16): 16, ("none", 48): 48},
    "gemmini": {("layernorm", 16): 130, ("softmax", 4): 41,
                ("igelu", 16): 16, ("igelu", 64): 64,
                ("none", 16): 16, ("none", 48): 48},
    "vta":     {("layernorm", 16): None, ("softmax", 4): None,
                ("igelu", 16): None, ("igelu", 64): None,
                ("none", 16): 16, ("none", 48): 48},
}

#: Measured cycles for one 16x16x16 mesh operation, weights changing every
#: tile. VTA is fastest because it is not weight-stationary: it re-reads the
#: whole 16x16 matrix from a scratchpad each cycle, so it pays neither
#: Gemmini's request handshake nor SPMW's serial weight load -- and spends
#: 2048 bits a cycle of weight bandwidth instead.
PER_TILE = {"spmw": 16.0, "gemmini": 18.0, "vta": 16.0}

#: Achieved period from place and route on xcu280-fsvh2892-2L-e.
#: VTA's is `TensorGemm`'s, which is the slower of its two datapath modules
#: (`TensorAlu` closes at +0.346 ns, 334.8 MHz).
PERIOD_NS = {"spmw": 3.261, "gemmini": 28.708, "vta": 3.328}


def main():
    tiles = sum(gemm_tiles().values())
    rows = norm_rows()
    engines = ("vta", "gemmini", "spmw")

    print(f"the block: {tiles:,} mesh tiles of 16^3, "
          f"{sum(r * b * 16 for _, r, b in rows.values()):,} elements "
          f"through the scale path\n")
    hdr = f"{'pass':10s} {'activation':11s} {'rows':>5s} {'beats':>6s}"
    print(hdr + "".join(f"{e:>12s}" for e in engines))

    total = {e: 0 for e in engines}
    offchip = {e: 0 for e in engines}
    for name, (act, nrow, beats) in rows.items():
        line = f"{name:10s} {act:11s} {nrow:5d} {beats:6d}"
        for e in engines:
            per = PER_ROW[e][(act, beats)]
            if per is None:
                line += f"{'-- host --':>12s}"
                offchip[e] += nrow * beats * 16
            else:
                total[e] += nrow * per
                line += f"{nrow * per:>12,d}"
        print(line)

    print(f"\n{'scale path':10s}{'':25s}" +
          "".join(f"{total[e]:>12,d}" for e in engines))
    for e in engines:
        total[e] += round(tiles * PER_TILE[e])
    print(f"{'mesh':10s}{'':25s}" +
          "".join(f"{round(tiles * PER_TILE[e]):>12,d}" for e in engines))
    print(f"{'total':10s}{'':25s}" + "".join(f"{total[e]:>12,d}" for e in engines))
    print(f"{'off chip':10s}{'':25s}" +
          "".join(f"{offchip[e]:>12,d}" for e in engines) + "   elements")

    print()
    for e in engines:
        p = PERIOD_NS[e]
        note = "" if not offchip[e] else \
            f"  + {offchip[e]:,} elements on the host"
        if p:
            print(f"  {e:8s} {total[e]:9,d} cycles x {p:6.3f} ns = "
                  f"{total[e] * p / 1e6:7.3f} ms{note}")
        else:
            print(f"  {e:8s} {total[e]:9,d} cycles x (route pending){note}")


if __name__ == "__main__":
    main()
