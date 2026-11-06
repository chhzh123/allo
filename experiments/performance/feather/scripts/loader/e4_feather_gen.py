#!/usr/bin/env python3
"""E4: operands, SRAM images and the reference for the general FEATHER RTL bench.

The reference models the RTL as it is, not the drivers' numpy:

- NEST arithmetic (feather_pe.v): every operand is an unsigned reg. A PE forms
  the 9-bit differences (a - zpa) mod 512 and (w - zpw) mod 512, multiplies
  them as unsigned 9 x 9 -> 18 bits, and accumulates in 32 bits. So the RTL's
  result equals the signed dot product the drivers compute exactly when every
  difference is non-negative -- a "legal" operand here is zp + d, d in [0, 256 - zp).
  The checker verifies both: that the RTL matches this model bit for bit, and
  whether the result equals the signed mathematics.
- BIRRD (birrd_simple_cmd_flow_seq.v, birrd_2x2_simple_cmd_flow_seq.v): 2 log2 N
  stages; switch i of a stage takes ports 2i (low) and 2i+1 (high); its own
  command is the MS two bits of its per-row command word, the rest is forwarded
  a stage a cycle; 00 pass, 01 sum on the low port, 10 sum on the high port,
  11 swap; after stage s port q goes to reverse_bits(q, min(2+s, log2 N,
  2 log2 N - s)). The drivers' PS/AR/AL/SW (feather_ref) map to 00/10/01/11
  (their "add right" puts the sum on the odd port, the RTL's high one), and the
  drivers' three-stage programs at N = 4 fill the RTL's first three stages, the
  fourth passes, so the RTL's output column p carries feather_ref's column
  reverse_bits(p, 2).
- The weight feed (feather_controller.v): one SRAM row a cycle, pe_sel one PE a
  cycle, the SRAM's read register a cycle behind pe_sel, so row a of a feed is
  stored by PE (a+1) mod N^2 at index floor((a+1) / N^2); the last row of a
  feed is what PE 0 stores first in the next feed, and the row the read
  register holds when the first feed starts (address 0, loaded before the
  feed's start pulse rewrites it) is PE 0's first weight of tile 0.

`--loader row` is the same feed over the row-wise select (scripts/loader/
apply_row_loader.py): pe_sel names a row of the array, so every column's PE in
that row takes its own byte of the N-byte word in the same cycle. The rule is
the same with N in place of N^2 -- row a is stored by PE row (a+1) mod N at
index floor((a+1) / N), in every column -- and a feed is N^2 rows, not N^3.
The bytes each PE ends up holding are identical; only the packing and the
number of cycles change.
"""

import argparse
import json
import os
from math import log2

import numpy as np

PS, AR, AL, SW = 0, 1, 2, 3
NUMPY_TO_RTL = {PS: 0b00, AR: 0b10, AL: 0b01, SW: 0b11}
MASK32 = (1 << 32) - 1


# -- BIRRD, the RTL's and the drivers' ----------------------------------------


def reverse_bits(value, width):
    mask = (1 << width) - 1
    out = 0
    for i in range(width):
        if value & (1 << i):
            out |= 1 << (width - 1 - i)
    return (value & ~mask) | out


def rtl_stages(N):
    return 2 * int(log2(N))


def rtl_group_bits(s, N):
    lg = int(log2(N))
    return min(2 + s, lg, 2 * lg - s)


def rtl_birrd(line, codes, N):
    """One row of N column sums through the RTL's network; `codes[s][i]` are the
    RTL's own two-bit codes. 32-bit wrap, as the eggs add in DATA_WIDTH bits."""
    S = rtl_stages(N)
    cur = [int(v) & MASK32 for v in line]
    for s in range(S):
        nxt = [0] * N
        for i in range(N // 2):
            lo, hi = cur[2 * i], cur[2 * i + 1]
            c = int(codes[s][i])
            if c == 0:
                olo, ohi = lo, hi
            elif c == 1:
                olo, ohi = (lo + hi) & MASK32, hi
            elif c == 2:
                olo, ohi = lo, (lo + hi) & MASK32
            else:
                olo, ohi = hi, lo
            if s == S - 1:
                nxt[2 * i], nxt[2 * i + 1] = olo, ohi
            else:
                b = rtl_group_bits(s, N)
                nxt[reverse_bits(2 * i, b)] = olo
                nxt[reverse_bits(2 * i + 1, b)] = ohi
        cur = nxt
    return np.array(cur, dtype=np.uint32)


def birrd_shape(AW):
    lg = int(log2(AW))
    return (2 * lg if AW > 4 else 2 * lg - 1), AW // 2


def stage_bits(stage, AW):
    lg = int(log2(AW))
    return 2 if stage == 0 else min(lg, 2 + stage, 2 * lg - stage)


def feather_ref(iActs, weights, inst, AW, AH):
    """The drivers' semantics (tests/dataflow/spmw/test_spmw_feather.py), signed."""
    P0, P1 = birrd_shape(AW)
    cols = np.zeros((AH, AW), dtype=np.int64)
    for i in range(AH):
        for j in range(AW):
            cols[i, j] = int(
                np.dot(iActs[:, j].astype(np.int64), weights[i, j, :].astype(np.int64))
            )
    out = np.zeros((AH, AW), dtype=np.int64)
    for i in range(AH):
        line = cols[i].copy()
        for s in range(P0):
            nxt = np.zeros(AW, dtype=np.int64)
            for j in range(P1):
                l, r = line[2 * j], line[2 * j + 1]
                c = int(inst[s, j])
                ol, orr = l, r
                if c == AR:
                    orr = l + r
                elif c == AL:
                    ol = l + r
                elif c == SW:
                    ol, orr = r, l
                if s == P0 - 1:
                    nxt[2 * j], nxt[2 * j + 1] = ol, orr
                else:
                    bits = stage_bits(s, AW)
                    nxt[reverse_bits(2 * j, bits)] = ol
                    nxt[reverse_bits(2 * j + 1, bits)] = orr
            line = nxt
        out[i] = line
    return out.astype(np.int32)


def rtl_codes(inst, N, swap_arl=True):
    """The drivers' program [P0, N/2] as the RTL's per-stage codes [2 log2 N, N/2]."""
    S = rtl_stages(N)
    P0, P1 = inst.shape
    assert P1 == N // 2 and P0 <= S, (inst.shape, N)
    m = NUMPY_TO_RTL if swap_arl else {PS: 0, AR: 1, AL: 2, SW: 3}
    codes = np.zeros((S, N // 2), dtype=np.int64)
    for s in range(P0):
        for i in range(P1):
            codes[s, i] = m[int(inst[s, i])]
    return codes


def rtl_word(codes, N):
    """The controller's instruction word: switch row i at bits [i*2S +: 2S],
    stage 0 in the row's most significant two bits."""
    S = rtl_stages(N)
    word = 0
    for i in range(N // 2):
        row = 0
        for s in range(S):
            row |= int(codes[s, i]) << (2 * (S - 1 - s))
        word |= row << (i * 2 * S)
    return word


def rtl_column_of(q, N):
    """Which RTL output column carries the drivers' output column q."""
    if N == 4:
        return reverse_bits(q, 2)
    return q


def gemm_insts(AW):
    if AW == 16:
        return np.array(
            [
                [PS, SW, PS, SW, PS, SW, PS, SW],
                [PS, PS, SW, PS, PS, PS, SW, PS],
                [PS, PS, PS, PS, PS, PS, PS, PS],
                [AL, AL, AL, AL, AR, AR, AR, AR],
                [SW, SW, SW, SW, SW, SW, SW, SW],
                [PS, PS, PS, PS, PS, PS, PS, PS],
                [PS, PS, PS, PS, PS, PS, PS, PS],
                [PS, PS, PS, PS, PS, PS, PS, PS],
            ],
            dtype=np.int8,
        )
    if AW == 8:
        return np.array(
            [
                [PS, PS, PS, PS],
                [PS, PS, PS, PS],
                [AR, AR, AL, AL],
                [SW, SW, SW, SW],
                [SW, PS, PS, SW],
                [PS, PS, PS, PS],
            ],
            dtype=np.int8,
        )
    if AW == 4:
        return np.array([[PS, PS], [AR, AL], [SW, PS]], dtype=np.int8)
    P0, P1 = birrd_shape(AW)
    return np.zeros((P0, P1), dtype=np.int8)


def conv_insts(AW=4):
    """One program per output position in a line: every column sum of the
    tile into output column `pos`. AW = 4: the drivers' four programs
    (examples/feather/convolution.py). AW = 8 and 16: found by
    `find_reduce_program` under the same semantics, checked by the selftest."""
    if AW == 4:
        return [
            np.array([[AL, AL], [AL, PS], [PS, PS]], dtype=np.int8),
            np.array([[AL, AL], [AL, PS], [SW, PS]], dtype=np.int8),
            np.array([[AL, AL], [AR, PS], [PS, PS]], dtype=np.int8),
            np.array([[AL, AL], [AR, PS], [PS, SW]], dtype=np.int8),
        ]
    if AW not in _CONV_CACHE:
        _CONV_CACHE[AW] = [find_reduce_program(AW, pos) for pos in range(AW)]
    return _CONV_CACHE[AW]


_CONV_CACHE = {}


def find_reduce_program(AW, pos):
    """A BIRRD program (the drivers' semantics: PS/AR/AL/SW, the bit-reversal
    links of `stage_bits`) whose output column `pos` carries the sum of all AW
    column sums; the other output columns are unconstrained.

    Two phases over the ports' column sets (a bit set a port). Reduction:
    whenever two disjoint sets meet at a switch they are merged, on the left
    output except at the last merge, where both AL and AR are tried (at AW = 4
    that choice is the only routing there is). Routing: in every later stage
    the switch holding the total passes or swaps, the others pass. The
    remaining stages form a butterfly, so the total reaches any column; if a
    layout ever did not, the exhaustive search over all four commands a switch
    (`_reduce_program_exhaustive`) is the fallback."""
    import itertools

    P0, P1 = birrd_shape(AW)
    full = (1 << AW) - 1

    def dest(s, q):
        return q if s == P0 - 1 else reverse_bits(q, stage_bits(s, AW))

    def step(s, masks, cmds):
        # A merge's other output keeps a copy of its input (AR: (l, l + r)); the
        # workload never reads it, and a later switch must not take it for a
        # live partial sum, so the model drops it (only the live sets are kept).
        nxt = [0] * AW
        for j, c in enumerate(cmds):
            l, r = masks[2 * j], masks[2 * j + 1]
            ol, orr = (l | r, 0) if c == AL else (0, l | r) if c == AR else (r, l) if c == SW else (l, r)
            nxt[dest(s, 2 * j)] = ol
            nxt[dest(s, 2 * j + 1)] = orr
        return tuple(nxt)

    def route(s, masks, prog):
        if s == P0:
            return prog if masks[pos] == full else None
        holders = [j for j in range(P1) if masks[2 * j] or masks[2 * j + 1]]
        assert len(holders) == 1
        for c in (PS, SW):
            cmds = [PS] * P1
            cmds[holders[0]] = c
            found = route(s + 1, step(s, masks, cmds), prog + [cmds])
            if found is not None:
                return found
        return None

    def reduce_(s, masks, prog):
        pairs = [j for j in range(P1) if masks[2 * j] and masks[2 * j + 1]]
        if not pairs:
            return route(s, masks, prog)
        assert all(not (masks[2 * j] & masks[2 * j + 1]) for j in pairs)
        choices = [AL, AR] if len(pairs) == 1 else [AL]
        for c in choices:
            cmds = [PS] * P1
            for j in pairs:
                cmds[j] = c
            found = reduce_(s + 1, step(s, masks, cmds), prog + [cmds])
            if found is not None:
                return found
        return None

    prog = reduce_(0, tuple(1 << q for q in range(AW)), [])
    if prog is None:
        prog = _reduce_program_exhaustive(AW, pos)
    inst = np.array(prog, dtype=np.int8)
    assert inst.shape == (P0, P1), inst.shape
    return inst


def _reduce_program_exhaustive(AW, pos):
    """Depth-first over every command of every switch (a (stage, state) pair
    visited once; merges only of disjoint sets). Exact, and slow at AW = 16."""
    import itertools

    P0, P1 = birrd_shape(AW)
    full = (1 << AW) - 1
    seen = set()

    def dest(s, q):
        return q if s == P0 - 1 else reverse_bits(q, stage_bits(s, AW))

    def dfs(s, masks, prog):
        if s == P0:
            return prog if masks[pos] == full else None
        key = (s, masks)
        if key in seen:
            return None
        seen.add(key)
        opts = []
        for j in range(P1):
            l, r = masks[2 * j], masks[2 * j + 1]
            if l and r and not (l & r):
                opts.append([(AL, l | r, r), (AR, l, l | r), (PS, l, r), (SW, r, l)])
            elif l and r:
                opts.append([(PS, l, r), (SW, r, l)])
            elif l or r:
                opts.append([(PS, l, r), (SW, r, l)])
            else:
                opts.append([(PS, 0, 0)])
        for combo in itertools.product(*opts):
            nxt = [0] * AW
            for j, (_c, ol, orr) in enumerate(combo):
                nxt[dest(s, 2 * j)] = ol
                nxt[dest(s, 2 * j + 1)] = orr
            found = dfs(s + 1, tuple(nxt), prog + [[c for (c, _, _) in combo]])
            if found is not None:
                return found
        return None

    prog = dfs(0, tuple(1 << q for q in range(AW)), [])
    assert prog is not None, (AW, pos)
    return prog


GEMM_EXTRACT = {16: [8, 10, 11, 9, 5, 6, 7, 4], 8: [6, 5, 2, 1], 4: [2, 0]}


# -- NEST, the RTL's arithmetic ------------------------------------------------


def rtl_cols(iacts_u, weights_u, zpa, zpw):
    """cols[i, j] as the PE computes it: unsigned 9-bit differences, unsigned
    products, 32-bit accumulation. `iacts_u` [N, N] (k, j), `weights_u` [N, N, N]
    (i, j, k), both uint8 values as stored."""
    da = (iacts_u.astype(np.int64) - int(zpa)) % 512
    dw = (weights_u.astype(np.int64) - int(zpw)) % 512
    prod = np.einsum("kj,ijk->ijk", da, dw)
    return (prod.sum(axis=2) % (1 << 32)).astype(np.uint32)


def signed_cols(iacts_u, weights_u, zpa, zpw):
    da = iacts_u.astype(np.int64) - int(zpa)
    dw = weights_u.astype(np.int64) - int(zpw)
    return np.einsum("kj,ijk->ij", da, dw)


def rtl_tile(iacts_u, weights_u, codes, zpa, zpw, N):
    """The N bus rows of one tile, as the RTL produces them: [N (i), N (column)] uint32."""
    cols = rtl_cols(iacts_u, weights_u, zpa, zpw)
    return np.stack([rtl_birrd(cols[i], codes, N) for i in range(N)])


# -- tilings, the drivers' ------------------------------------------------------


def gemm_tile(A_tile, B_tile, AW, AH):
    Mt, Kt, Nt = AW // 2, 2 * AH, AH
    assert A_tile.shape == (Mt, Kt) and B_tile.shape == (Kt, Nt)
    left, right = np.hsplit(A_tile, 2)
    iActs = np.ascontiguousarray(np.hstack([left.T, right.T]))
    bl, br = np.vsplit(B_tile, 2)
    cl = np.array([bl.T] * (AW // 2))
    cr = np.array([br.T] * (AW // 2))
    weights = np.ascontiguousarray(np.vstack([cl, cr]).transpose(1, 0, 2))
    return iActs, weights


def gemm_workload(M, K, Nn, AW, AH, rng, zpa, zpw, rng_hi):
    """A [M, K], B [K, Nn] as stored uint8 (zp + d, d in [0, rng_hi)); the tiles in
    the drivers' order (n, m, k), each (iActs_u, weights_u, inst, (n, m, k))."""
    Mt, Kt, Nt = AW // 2, 2 * AH, AH
    A = ((rng.integers(0, rng_hi, size=(M, K)) + zpa) % 256).astype(np.uint8)
    B = ((rng.integers(0, rng_hi, size=(K, Nn)) + zpw) % 256).astype(np.uint8)
    inst = gemm_insts(AW)
    tiles = []
    for n in range(Nn // Nt):
        for m in range(M // Mt):
            for k in range(K // Kt):
                iActs, weights = gemm_tile(
                    A[m * Mt : (m + 1) * Mt, k * Kt : (k + 1) * Kt],
                    B[k * Kt : (k + 1) * Kt, n * Nt : (n + 1) * Nt],
                    AW,
                    AH,
                )
                tiles.append((iActs, weights, inst, (n, m, k)))
    return A, B, tiles


def gemm_reduce(tile_rows_signed, coords, M, K, Nn, AW, AH):
    """Host side: the tiles' output rows (drivers' column order, signed) into C."""
    Mt, Kt, Nt = AW // 2, 2 * AH, AH
    order = GEMM_EXTRACT[AW]
    C = np.zeros((M, Nn), dtype=np.int64)
    for rows, (n, m, k) in zip(tile_rows_signed, coords):
        part = rows[:, order]  # [Nt, Mt]
        C[m * Mt : (m + 1) * Mt, n * Nt : (n + 1) * Nt] += part.T
    return C


def conv_workload(Cin, H, Wd, M, R, S, pad, AW, AH, rng, zpa, zpw, rng_hi):
    """Batch one, NHWC input (stored uint8), OIHW weights; the drivers' tiling from
    test_conv_4x4_matches_numpy with 'same' padding and the RS reduction padded
    up to a multiple of AH with zero-difference operands. Returns the stored
    tensors and the tiles (iActs_u, weights_u, inst, (p, q, mt, ct, vn))."""
    P, Q = H + 2 * pad - R + 1, Wd + 2 * pad - S + 1
    x = ((rng.integers(0, rng_hi, size=(H, Wd, Cin)) + zpa) % 256).astype(np.uint8)  # HWC
    w = ((rng.integers(0, rng_hi, size=(M, Cin, R, S)) + zpw) % 256).astype(np.uint8)
    RS = R * S
    RS_pad = ((RS + AH - 1) // AH) * AH
    x_pad = np.full((H + 2 * pad, Wd + 2 * pad, Cin), zpa, dtype=np.uint8)
    x_pad[pad : pad + H, pad : pad + Wd, :] = x
    w_flat = np.full((M, Cin, RS_pad), zpw, dtype=np.uint8)
    w_flat[:, :, :RS] = w.reshape(M, Cin, RS)
    insts = conv_insts(AW)
    tiles = []
    for p in range(P):
        for q in range(Q):
            win = np.full((RS_pad, Cin), zpa, dtype=np.uint8)
            win[:RS] = x_pad[p : p + R, q : q + S, :].reshape(RS, Cin)
            pos = (p * Q + q) % AW
            for mt in range(0, M, AH):
                for ct in range(0, Cin, AW):
                    for vn in range(0, RS_pad, AH):
                        iActs = np.ascontiguousarray(win[vn : vn + AH, ct : ct + AW])
                        weights = np.ascontiguousarray(
                            w_flat[mt : mt + AH, ct : ct + AW, vn : vn + AH]
                        )
                        tiles.append((iActs, weights, insts[pos], (p, q, mt, ct, vn)))
    return x, w, tiles, (P, Q, RS_pad)


def conv_reduce(tile_rows_signed, coords, P, Q, M, AW, AH):
    """Host side: each tile's column `pos` holds the whole AW-channel sum for AH
    output channels; accumulate over ct and vn. Returns out[M, P, Q]."""
    out = np.zeros((M, P, Q), dtype=np.int64)
    for rows, (p, q, mt, ct, vn) in zip(tile_rows_signed, coords):
        pos = (p * Q + q) % AW
        out[mt : mt + AH, p, q] += rows[:, pos]
    return out


def conv_reference(x_u, w_u, zpa, zpw, pad):
    H, Wd, Cin = x_u.shape
    M, _, R, S = w_u.shape
    P, Q = H + 2 * pad - R + 1, Wd + 2 * pad - S + 1
    xd = np.zeros((H + 2 * pad, Wd + 2 * pad, Cin), dtype=np.int64)
    xd[pad : pad + H, pad : pad + Wd, :] = x_u.astype(np.int64) - zpa
    wd = w_u.astype(np.int64) - zpw  # [M, C, R, S]
    out = np.zeros((M, P, Q), dtype=np.int64)
    for p in range(P):
        for q in range(Q):
            win = xd[p : p + R, q : q + S, :]  # [R, S, C]
            out[:, p, q] = np.einsum("rsc,mcrs->m", win, wd)
    return out


# -- SRAM images ---------------------------------------------------------------


def pe_files(weights_u, N):
    """files[q, k]: PE q = N * col + row holds weights[row, col, k]."""
    files = np.zeros((N * N, N), dtype=np.uint8)
    for col in range(N):
        for row in range(N):
            files[N * col + row] = weights_u[row, col, :]
    return files


def feed_len(N, loader):
    """Cycles in one weight feed: a PE a cycle, or a row of PEs a cycle."""
    return N * N * N if loader == "pe" else N * N


def weight_image(files, N, loader="pe"):
    """The weight SRAM image for T feeds back to back: rows [T * L, N] uint8
    with L = feed_len(N, loader), plus the row the read register must hold when
    the first feed starts -- see the module docstring.

    `pe`  (the published select): row a is stored by PE (a+1) mod N^2, whose
          column decides which of the N byte lanes carries it; the other N-1
          lanes of that row go nowhere. L = N^3.
    `row` (the row-wise select): row a is stored by PE row (a+1) mod N in every
          column at once, lane c going to column c, so all N lanes are live.
          L = N^2. Each PE's file is byte for byte the one `pe` builds."""
    T = files.shape[0]
    L = feed_len(N, loader)
    P = N * N if loader == "pe" else N  # pe_sel's period: every PE, or every row
    rows = np.zeros((T, L, N), dtype=np.uint8)
    a = np.arange(L)
    q = (a + 1) % P
    s = (a + 1) // P
    keep = s < N
    if loader == "pe":
        rows[:, a[keep], q[keep] // N] = files[:, q[keep], s[keep]]
    else:
        cols = np.arange(N)
        rows[:, a[keep], :] = files[:, N * cols[None, :] + q[keep][:, None], s[keep][:, None]]
    if T > 1:
        # the last row of a feed is what the next feed's first pe_sel stores
        if loader == "pe":
            rows[:-1, L - 1, 0] = files[1:, 0, 0]
        else:
            rows[:-1, L - 1, :] = files[1:, N * np.arange(N), 0]
    # the row the read register holds at the feed's first cycle, when pe_sel = 0
    stale = np.zeros(N, dtype=np.uint8)
    if loader == "pe":
        stale[0] = files[0, 0, 0]
    else:
        stale[:] = files[0, N * np.arange(N), 0]
    return rows.reshape(T * L, N), stale


def write_rows_hex(path, rows):
    """Rows of N bytes as one hex word per line, byte j at bits [8j +: 8]."""
    hexchars = np.frombuffer(b"0123456789abcdef", dtype=np.uint8)
    rev = rows[:, ::-1]
    out = np.empty((rows.shape[0], 2 * rows.shape[1] + 1), dtype=np.uint8)
    out[:, 0:-1:2] = hexchars[rev >> 4]
    out[:, 1:-1:2] = hexchars[rev & 15]
    out[:, -1] = ord("\n")
    out.tofile(path)


def write_sparse_hex(path, blocks, digits):
    """`blocks` = [(address, [values...]), ...] as @addr lines and values."""
    with open(path, "w", encoding="utf-8") as f:
        for addr, values in blocks:
            f.write(f"@{addr:x}\n")
            for v in values:
                f.write(f"{v:0{digits}x}\n")


# -- runs ----------------------------------------------------------------------


def pattern_range(pattern, zp):
    """The range of the difference d for a 'legal' operand zp + d."""
    hi = 256 - zp
    # `mixed`: the stored byte covers all of [0, 256) whatever the zero point, so
    # d = u - zp is negative for u < zp -- the operands the RTL's unsigned
    # datapath cannot handle (the checker reports `signed_equals_rtl`).
    return {"small": min(8, hi), "mid": min(32, hi), "full": hi, "sparse": hi, "mixed": 256}[pattern]


def make_tiles(args, rng):
    """(tiles, host) for the run: tiles = [(iActs_u, weights_u, inst, coord)]."""
    N = args.N
    zpa, zpw = args.zpa, args.zpw
    ra, rw = pattern_range(args.pattern, zpa), pattern_range(args.pattern, zpw)
    if args.workload == "gemm":
        M, K, Nn = args.gemm
        A, B, tiles = gemm_workload(M, K, Nn, N, N, rng, zpa, zpw, min(ra, rw))
        return tiles, {"A": A, "B": B, "dims": (M, K, Nn)}
    if args.workload == "conv":
        Cin, H, Wd, M = args.conv
        x, w, tiles, geom = conv_workload(
            Cin, H, Wd, M, 3, 3, 1, N, N, rng, zpa, zpw, min(ra, rw)
        )
        return tiles, {"x": x, "w": w, "geom": geom, "dims": (Cin, H, Wd, M)}
    # single tiles with a chosen program
    tiles = []
    for t in range(args.tiles):
        if args.program == "gemm":
            inst = gemm_insts(N)
        elif args.program.startswith("conv"):
            inst = conv_insts(N)[int(args.program[4:])]
        elif args.program == "pass":
            inst = np.zeros(birrd_shape(N), dtype=np.int8)
        elif args.program == "random":
            inst = rng.integers(0, 4, size=birrd_shape(N)).astype(np.int8)
        elif args.program == "mixed":
            # a different program per tile: the four conv programs in turn at
            # N = 4, the GEMM program alternating with a random one elsewhere
            if N == 4:
                inst = conv_insts(N)[t % 4]
            elif t % 2 == 0:
                inst = gemm_insts(N)
            else:
                inst = rng.integers(0, 4, size=birrd_shape(N)).astype(np.int8)
        else:
            raise SystemExit(f"unknown program {args.program}")
        if args.pattern == "sparse":
            iActs = np.where(rng.random((N, N)) < 0.2, rng.integers(0, 256 - zpa, size=(N, N)), 0)
            weights = np.where(rng.random((N, N, N)) < 0.2, rng.integers(0, 256 - zpw, size=(N, N, N)), 0)
        else:
            iActs = rng.integers(0, ra, size=(N, N))
            weights = rng.integers(0, rw, size=(N, N, N))
        iActs = ((iActs + zpa) % 256).astype(np.uint8)
        weights = ((weights + zpw) % 256).astype(np.uint8)
        tiles.append((iActs, weights, inst, (t,)))
    return tiles, {}


def generate(args):
    N = args.N
    # a feed is N^3 cycles over the published one-PE-a-cycle select, N^2 over
    # the row-wise one; every span below is in feeds, so they all follow
    WLEN = feed_len(N, args.loader)
    out = args.out
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tiles, host = make_tiles(args, rng)
    T = len(tiles)
    G = args.gap
    # Row 0 of tile t leaves the activation SRAM at cycle A0 + a(t, 0) + 1 and
    # must be the cycle after a "last index" cycle of the PEs' counter, which
    # starts at F0 + 1 (the feed's first cycle raises weights_to_use) and runs
    # every cycle from then on: so a(t, 0) + 1 - G must be 1 mod N, i.e.
    # a(t, 0) = G + WLEN for tile 0 (the pass of tile 0 follows feed 0; WLEN is
    # a multiple of N either way), then strides of WLEN (a feed a tile) or N
    # (passes back to back).
    A_BASE = G + WLEN
    A_STRIDE = WLEN if args.mode == 0 else N
    span = A_BASE + (T - 1) * A_STRIDE + 2 * N + 8 + WLEN
    IA = max(int(np.ceil(np.log2(span + 1))), 4)
    WA = max(int(np.ceil(np.log2((T if args.mode == 0 else 1) * WLEN))), 2)
    # MODE 1 feeds one weight set -- tile 0's -- and streams every tile's
    # activations and program through it: the datapath's throughput with the
    # weights resident. The expectation uses the same weights.
    if args.mode == 1:
        tiles = [(ia, tiles[0][1], inst, coord) for (ia, _, inst, coord) in tiles]
    files = np.stack([pe_files(w, N) for (_, w, _, _) in tiles])  # [T, N^2, N]
    if args.mode == 1:
        files = files[:1]
    rows, stale = weight_image(files, N, args.loader)
    wrow0 = rows[0].copy()
    rows[0] = stale
    write_rows_hex(os.path.join(out, "weights.hex"), rows)
    write_rows_hex(os.path.join(out, "wrow0.hex"), wrow0[None, :])
    with open(os.path.join(out, "pefiles.hex"), "w", encoding="utf-8") as f:
        for v in files[0].reshape(-1):
            f.write(f"{int(v):02x}\n")
    # activations, one sparse file per bank (column); instructions, one word per output row
    banks = [[] for _ in range(N)]
    instr = []
    S = rtl_stages(N)
    digits = (2 * S * (N // 2) + 3) // 4
    expect = np.zeros((T, N, N), dtype=np.uint32)
    signed_ok = True
    for t, (iActs, weights, inst, _) in enumerate(tiles):
        base = A_BASE + t * A_STRIDE
        for j in range(N):
            banks[j].append((base, [int(v) for v in iActs[:, j]]))
        codes = rtl_codes(inst, N, swap_arl=not args.no_swap_arl)
        word = rtl_word(codes, N)
        instr.append((base + N + 3, [word] * N))
        expect[t] = rtl_tile(iActs, weights, codes, args.zpa, args.zpw, N)
        if signed_ok:
            sc = signed_cols(iActs, weights, args.zpa, args.zpw)
            rc = rtl_cols(iActs, weights, args.zpa, args.zpw).astype(np.int64)
            rc[rc >= (1 << 31)] -= 1 << 32
            signed_ok = bool(np.array_equal(sc, rc))
    for j in range(N):
        write_sparse_hex(os.path.join(out, f"iacts_bank{j}.hex"), banks[j], 2)
    write_sparse_hex(os.path.join(out, "instr.hex"), instr, digits)
    np.save(os.path.join(out, "expect.npy"), expect)
    np.savez_compressed(
        os.path.join(out, "operands.npz"),
        iacts=np.stack([t[0] for t in tiles]),
        weights=np.stack([t[1] for t in tiles]),
        insts=np.stack([t[2] for t in tiles]),
        coords=np.array([t[3] for t in tiles]),
        **{k: v for k, v in host.items() if isinstance(v, np.ndarray)},
    )
    meta = {
        "N": N,
        "T": T,
        "MODE": args.mode,
        "WA": WA,
        "IA": IA,
        "A_BASE": A_BASE,
        "A_STRIDE": A_STRIDE,
        "G": G,
        "WLEN": WLEN,
        "loader": args.loader,
        "zpa": args.zpa,
        "zpw": args.zpw,
        "seed": args.seed,
        "pattern": args.pattern,
        "workload": args.workload,
        "program": args.program,
        "gemm": args.gemm,
        "conv": args.conv,
        "swap_arl": not args.no_swap_arl,
        "signed_equals_rtl": signed_ok,
        "host_dims": host.get("dims"),
        "conv_geom": host.get("geom"),
        "weight_rows": int(rows.shape[0]),
    }
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


# -- checking ------------------------------------------------------------------


def check(out, xsim_log):
    with open(os.path.join(out, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    N, T = meta["N"], meta["T"]
    expect = np.load(os.path.join(out, "expect.npy"))
    S = rtl_stages(N)
    LAT = N + 5 + S
    rows = {}
    with open(os.path.join(out, "bus.log"), encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == N + 1:
                rows[int(parts[0])] = np.array([int(v) for v in parts[1:]], dtype=np.uint32)
    info = {"A0": None, "F0": None, "toggles": [], "buffers": {}, "end": None, "verdict_lines": []}
    with open(xsim_log, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("E4 "):
                continue
            info["verdict_lines"].append(line.strip())
            parts = line.split()
            if parts[1] == "START":
                for kv in parts[2:]:
                    k, v = kv.split("=")
                    info[k] = int(v)
            elif parts[1] == "TOGGLE":
                d = dict(kv.split("=") for kv in parts[2:])
                info["toggles"].append((int(d["n"]), int(d["cycle"]), int(d["sel"])))
            elif parts[1] == "BUF":
                d = dict(kv.split("=") for kv in parts[2:])
                info["buffers"][int(d["k"])] = (int(d["ping"]), int(d["pong"]))
            elif parts[1] == "END":
                d = dict(kv.split("=") for kv in parts[2:])
                info["end"] = int(d["cycle"])
    A0 = info.get("A0")
    if A0 is None:
        return {"status": "no_start", "info": info}
    found = np.zeros(T, dtype=bool)
    first_cycle = np.full(T, -1, dtype=np.int64)
    offset = np.zeros(T, dtype=np.int64)
    bad_elems = 0
    first_bad = None
    for t in range(T):
        t0 = A0 + meta["A_BASE"] + t * meta["A_STRIDE"] + LAT
        hit = -1
        # the predicted cycle first, then the nearest offsets: a tile whose
        # expected rows are all zero (a padding-only conv window) also matches
        # the idle bus before its slot, which must not be read as an early output
        for x in sorted(range(t0 - 4, t0 + 5), key=lambda v: (abs(v - t0), v)):
            ok = True
            for r in range(N):
                row = rows.get(x + r)
                if row is None or not np.array_equal(row, expect[t, r]):
                    ok = False
                    break
            if ok:
                hit = x
                break
        if hit >= 0:
            found[t] = True
            first_cycle[t] = hit
            offset[t] = hit - t0
        else:
            got = np.stack([rows.get(t0 + r, np.zeros(N, dtype=np.uint32)) for r in range(N)])
            nb = int((got != expect[t]).sum())
            bad_elems += nb
            if first_bad is None:
                first_bad = {"tile": t, "expected_row0": expect[t, 0].tolist(), "got_row0": got[0].tolist(), "bad_elements": nb}
    last_cycle = first_cycle + N - 1
    # MODE 1 dumps every PE's ping buffer against pefiles.hex at the end of the
    # feed: N^2 PEs must hold the right byte at every one of the N indices. A
    # weight image that lands the wrong bytes -- or none -- still produces
    # well-formed bus rows with the right handshaking and the right timing, so
    # this is checked, not inferred, and a shortfall fails the run.
    pe_files_ok = None
    if meta["MODE"] == 1 and info["buffers"]:
        pe_files_ok = all(ping == N * N for (ping, _) in info["buffers"].values())
        pe_files_ok = pe_files_ok and len(info["buffers"]) == N
    ok = bool(found.all()) and pe_files_ok is not False
    result = {
        "status": "pass" if ok else "fail",
        "pe_files_ok": pe_files_ok,
        "pe_files_wrong": None if pe_files_ok is None else sum(N * N - ping for (ping, _) in info["buffers"].values()),
        "expect_nonzero": int(np.count_nonzero(expect)),
        "tiles": T,
        "tiles_ok": int(found.sum()),
        "bad_elements": bad_elems,
        "first_bad": first_bad,
        "A0": A0,
        "F0": info.get("F0"),
        "G": info.get("G"),
        "offsets": sorted(set(offset[found].tolist())),
        "toggles": info["toggles"][:8],
        "toggle_count": len(info["toggles"]),
        "buffers": info["buffers"],
        "end_cycle": info["end"],
        "signed_equals_rtl": meta["signed_equals_rtl"],
    }
    if found.all():
        F0 = info["F0"]
        done = last_cycle  # cycle of each tile's last bus row
        result["tile_done_cycles_rel_F0"] = (done - F0 + 1).tolist() if T <= 64 else (done[:8] - F0 + 1).tolist() + (done[-8:] - F0 + 1).tolist()
        result["first_output_cycles"] = int(done[0] - F0 + 1)
        result["completion_cycles"] = int(done[-1] - F0 + 1)
        if T > 1:
            gaps = np.diff(done)
            result["steady_interval_cycles"] = {"min": int(gaps.min()), "median": float(np.median(gaps)), "max": int(gaps.max()), "mean": float(gaps.mean())}
            result["steady_interval_last32"] = float(np.mean(gaps[-32:]))
        if len(info["toggles"]) >= 2:
            tg = [c for (_, c, _) in info["toggles"]]
            result["feed_cycles_between_toggles"] = sorted(set(np.diff(tg).tolist()))
            result["first_toggle_rel_F0"] = tg[0] - F0
            result["drain_cycles_after_last_toggle"] = int(done[-1] - tg[-1]) if meta["MODE"] == 0 else None
    # host-level reduction against the signed reference (MODE 0 only: MODE 1
    # runs every tile through tile 0's weights, so the workload's result is
    # not what it computes)
    if found.all() and meta["workload"] in ("gemm", "conv") and meta["MODE"] == 0:
        op = np.load(os.path.join(out, "operands.npz"))
        signed_rows = []
        for t in range(T):
            got = np.stack([rows[first_cycle[t] + r] for r in range(N)]).astype(np.int64)
            got[got >= (1 << 31)] -= 1 << 32
            # back to the drivers' column order
            drv = np.zeros_like(got)
            for q in range(N):
                drv[:, q] = got[:, rtl_column_of(q, N)]
            signed_rows.append(drv)
        coords = [tuple(c) for c in op["coords"]]
        if meta["workload"] == "gemm":
            M, K, Nn = meta["host_dims"]
            C = gemm_reduce(signed_rows, coords, M, K, Nn, N, N)
            A = op["A"].astype(np.int64) - meta["zpa"]
            B = op["B"].astype(np.int64) - meta["zpw"]
            ref = A @ B
            result["host_check"] = "pass" if np.array_equal(C, ref) else "fail"
            result["host_bad"] = int((C != ref).sum())
        else:
            P, Q, _ = meta["conv_geom"]
            Cin, H, Wd, M = meta["host_dims"]
            got = conv_reduce(signed_rows, coords, P, Q, M, N, N)
            ref = conv_reference(op["x"], op["w"], meta["zpa"], meta["zpw"], 1)
            result["host_check"] = "pass" if np.array_equal(got, ref) else "fail"
            result["host_bad"] = int((got != ref).sum())
    with open(os.path.join(out, "check.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=int)
    return result


def selftest():
    """The mapping against feather_ref on tiles, for every program the drivers ship."""
    rng = np.random.default_rng(0)
    for N in (4, 8, 16):
        # every conv program sums all N column sums into its output column
        for pos, inst in enumerate(conv_insts(N)):
            iActs = rng.integers(-128, 128, size=(N, N)).astype(np.int8)
            weights = rng.integers(-128, 128, size=(N, N, N)).astype(np.int8)
            ref = feather_ref(iActs, weights, inst, N, N).astype(np.int64)
            cols = np.einsum("kj,ijk->ij", iActs.astype(np.int64), weights.astype(np.int64))
            assert np.array_equal(ref[:, pos], cols.sum(axis=1)), ("conv program", N, pos)
        progs = [gemm_insts(N)] + list(conv_insts(N)) + [
            rng.integers(0, 4, size=birrd_shape(N)).astype(np.int8) for _ in range(5)
        ]
        for inst in progs:
            # legal operands: non-negative differences from the zero points (a
            # negative one is where the RTL's unsigned arithmetic parts from the
            # signed mathematics, which `signed_equals_rtl` reports per run)
            iActs = rng.integers(0, 8, size=(N, N)).astype(np.int8)
            weights = rng.integers(0, 8, size=(N, N, N)).astype(np.int8)
            ref = feather_ref(iActs, weights, inst, N, N).astype(np.int64)
            codes = rtl_codes(inst, N)
            zpa, zpw = 100, 50
            got = rtl_tile((iActs.astype(np.int64) + zpa).astype(np.uint8),
                           (weights.astype(np.int64) + zpw).astype(np.uint8), codes, zpa, zpw, N).astype(np.int64)
            got[got >= (1 << 31)] -= 1 << 32
            perm = np.zeros_like(got)
            for q in range(N):
                perm[:, q] = got[:, rtl_column_of(q, N)]
            assert np.array_equal(perm, ref), (N, inst)
            # the other mapping must differ for programs with adds
            if (inst == AR).any() or (inst == AL).any():
                other = rtl_tile((iActs.astype(np.int64) + zpa).astype(np.uint8),
                                 (weights.astype(np.int64) + zpw).astype(np.uint8), rtl_codes(inst, N, False), zpa, zpw, N).astype(np.int64)
                other[other >= (1 << 31)] -= 1 << 32
                perm2 = np.zeros_like(other)
                for q in range(N):
                    perm2[:, q] = other[:, rtl_column_of(q, N)]
                assert not np.array_equal(perm2, ref), "the two AR/AL mappings agree; the test cannot tell them apart"
    # the weight image round trip: every PE's file is recoverable by the feed rule
    for N in (4, 8, 16):
        T = 3
        files = np.random.default_rng(1).integers(0, 256, size=(T, N * N, N)).astype(np.uint8)
        # `pe`: feed cycle c has pe_sel = c mod N^2 and the read register shows
        # the stale row at c = 0 (address 0 before the start pulse rewrote it)
        # and mem[c-1] after; the PE it names takes lane (pe_sel // N).
        rows, stale = weight_image(files, N, "pe")
        N2, N3 = N * N, N ** 3
        assert rows.shape[0] == T * N3 == T * feed_len(N, "pe")
        got = np.zeros_like(files)
        for c in range(T * N3):
            q = c % N2
            k = (c % N3) // N2
            t = c // N3
            reg = stale if c == 0 else rows[c - 1]
            got[t, q, k] = reg[q // N]
        assert np.array_equal(got, files), "weight image round trip (pe)"
        # `row`: pe_sel = c mod N names a row, and every column's PE in it takes
        # its own lane, so N bytes land per cycle and a feed is N^2 cycles.
        rows_r, stale_r = weight_image(files, N, "row")
        assert rows_r.shape[0] == T * N2 == T * feed_len(N, "row")
        got_r = np.zeros_like(files)
        for c in range(T * N2):
            r = c % N
            k = (c % N2) // N
            t = c // N2
            reg = stale_r if c == 0 else rows_r[c - 1]
            for col in range(N):
                got_r[t, N * col + r, k] = reg[col]
        assert np.array_equal(got_r, files), "weight image round trip (row)"
        # and the point of the exercise: the same bytes in the same slots
        assert np.array_equal(got_r, got), "the two loaders disagree on the PE files"
        assert rows_r.shape[0] * N == rows.shape[0], "the row image is not N times shorter"
    print("selftest ok")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--N", type=int, required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--workload", choices=("tile", "gemm", "conv"), default="tile")
    g.add_argument("--tiles", type=int, default=1, help="tiles (workload=tile)")
    g.add_argument("--program", default="gemm", help="gemm|conv0..3|pass|random (workload=tile)")
    g.add_argument("--gemm", type=lambda s: [int(v) for v in s.split(",")], default=[128, 128, 128])
    g.add_argument("--conv", type=lambda s: [int(v) for v in s.split(",")], default=[64, 16, 16, 64], help="Cin,H,W,M")
    g.add_argument("--mode", type=int, default=1, help="0: a feed per tile; 1: one feed, passes back to back")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--zpa", type=int, default=0)
    g.add_argument("--zpw", type=int, default=0)
    g.add_argument("--pattern", default="small", help="small|mid|full|sparse")
    g.add_argument("--gap", type=int, default=8, help="weight feed starts this many cycles after the activation feed")
    g.add_argument("--loader", choices=("pe", "row"), default="pe",
                   help="pe: the published select, one PE a cycle, a feed of N^3; "
                        "row: the row-wise select (scripts/loader/apply_row_loader.py), "
                        "a row of PEs a cycle, a feed of N^2")
    g.add_argument("--no-swap-arl", action="store_true", help="map AR->01, AL->10 instead (the wrong way round; for the mapping test)")
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    c.add_argument("--log", required=True)
    sub.add_parser("selftest")
    args = p.parse_args()
    if args.cmd == "gen":
        meta = generate(args)
        print(json.dumps(meta))
    elif args.cmd == "check":
        print(json.dumps(check(args.out, args.log), indent=1, default=int))
    else:
        selftest()


if __name__ == "__main__":
    main()
