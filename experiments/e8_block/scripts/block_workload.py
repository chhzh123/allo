#!/usr/bin/env python3
"""The transformer block as a count of tiles and of normalisation rows.

Both engines run the same thing, so the workload is described once, here, and
each engine contributes only its measured rates.  Nothing in this file is
measured; everything in it is arithmetic on the block's shape.

The shape is a small BERT layer: 64 tokens, 256 model width, 4 heads, a
feed-forward of 1024.  Small enough to simulate cycle-accurately on both
engines and large enough that every part of a block is present -- multi-head
attention with softmax, two LayerNorms, a GELU feed-forward, both residuals.
"""

SEQ, D_MODEL, HEADS, D_FFN = 64, 256, 4, 1024
D_HEAD = D_MODEL // HEADS
DIM = 16

#: (name, M, K, N) for every GEMM in the block, in execution order.
GEMMS = [
    ("qkv", SEQ, D_MODEL, 3 * D_MODEL),
    *[(f"qk{h}", SEQ, D_HEAD, SEQ) for h in range(HEADS)],
    *[(f"av{h}", SEQ, SEQ, D_HEAD) for h in range(HEADS)],
    ("proj", SEQ, D_MODEL, D_MODEL),
    ("ffn1", SEQ, D_MODEL, D_FFN),
    ("ffn2", SEQ, D_FFN, D_MODEL),
]

#: (name, activation, rows, row length) for every pass of the scale path.
#: `layernorm` and `softmax` reduce over their row; the rest are pointwise and
#: the row length only says how often the scalar chain turns over.
NORMS = [
    ("ln1", "layernorm", SEQ, D_MODEL),
    ("qkv_rq", "none", SEQ, 3 * D_MODEL),
    *[(f"sm{h}", "softmax", SEQ, SEQ) for h in range(HEADS)],
    ("ctx_rq", "none", SEQ, D_MODEL),
    ("proj_rq", "none", SEQ, D_MODEL),
    ("ln2", "layernorm", SEQ, D_MODEL),
    ("gelu", "igelu", SEQ, D_FFN),
    ("ffn2_rq", "none", SEQ, D_MODEL),
]


def tiles(m, k, n, dim=DIM):
    """16x16x16 mesh operations, rounding each extent up to the mesh."""
    return -(-m // dim) * -(-k // dim) * -(-n // dim)


def gemm_tiles(dim=DIM):
    return {name: tiles(m, k, n, dim) for name, m, k, n in GEMMS}


def norm_rows(dim=DIM):
    """`(activation, rows, beats per row)` per pass of the scale path."""
    return {name: (act, rows, -(-ln // dim)) for name, act, rows, ln in NORMS}


def macs():
    return sum(m * k * n for _, m, k, n in GEMMS)


def elements():
    return sum(rows * ln for _, _, rows, ln in NORMS)


def cycles(per_tile, per_row, dim=DIM):
    """Total cycles from a mesh rate and a per-activation row rate.

    `per_tile` is cycles for one `dim^3` mesh operation in the steady state.
    `per_row` maps an activation to `(fixed, per_beat)` -- the marginal cost
    of one normalisation row is `fixed + per_beat * beats`, which is how both
    engines were measured: two row counts, and the difference.
    """
    g = sum(gemm_tiles(dim).values()) * per_tile
    n = 0
    detail = {}
    for name, (act, rows, beats) in norm_rows(dim).items():
        fixed, per_beat = per_row[act]
        c = rows * (fixed + per_beat * beats)
        detail[name] = c
        n += c
    return dict(gemm=g, norm=n, total=g + n, detail=detail)


if __name__ == "__main__":
    gt = gemm_tiles()
    nr = norm_rows()
    print(f"the block: seq={SEQ} d_model={D_MODEL} heads={HEADS} d_ffn={D_FFN}")
    print(f"  {macs() / 1e6:.1f}M MACs in {sum(gt.values()):,} mesh tiles "
          f"of {DIM}^3")
    print(f"  {elements():,} elements through the scale path")
    print()
    print(f"  {'gemm':10s} {'M':>5s} {'K':>5s} {'N':>5s} {'tiles':>8s}")
    for name, m, k, n in GEMMS:
        print(f"  {name:10s} {m:5d} {k:5d} {n:5d} {gt[name]:8,d}")
    print(f"  {'total':10s} {'':5s} {'':5s} {'':5s} {sum(gt.values()):8,d}")
    print()
    print(f"  {'pass':10s} {'activation':11s} {'rows':>5s} {'beats':>6s} "
          f"{'elements':>9s}")
    for name, (act, rows, beats) in nr.items():
        print(f"  {name:10s} {act:11s} {rows:5d} {beats:6d} "
              f"{rows * beats * DIM:9,d}")
