# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""FEATHER on SPMW: NEST as a weight-file mesh, BIRRD as a butterfly grid.

FEATHER (Tong et al., ISCA 2024, https://arxiv.org/abs/2405.13170) is a
reconfigurable accelerator: NEST, an AH x AW array of PEs with a stationary
input activation and a local weight file each, and BIRRD, a butterfly
network of 2 log2(AW) stages of AW/2 two-by-two switches that reduces the
columns' partial sums and reorders them on the way out, so the output lands
in the layout the next layer wants. `examples/feather` is the Allo dataflow
version; this is the same engine as an SPMW fabric:

- `pe`: a pair of columns' worth of NEST -- two stationary activations, two
  weight files of AH entries, two partial-sum chains down the column. One
  unit per column *pair* so the bottom row's outputs pair up with BIRRD's
  first stage (a switch has a left and a right input); it is still one MAC
  per column per cycle.
- `switch`: BIRRD's egg. `cmd` is PS (pass), AR (add right), AL (add left)
  or SW (swap); the link rule between stages is the paper's bit-reversal
  butterfly, exactly as `examples/feather/feather.py` writes it.

Per launch the engine does one tile: AH output rows, each the AW column sums
of AH products, through BIRRD. The GEMM and convolution tilings are the Allo
drivers' (`examples/feather/gemm.py`, `convolution.py`), reimplemented here
as functions of (AW, AH) so a launch can be checked against numpy exactly.
"""

from math import log2

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import int8, int32

PS, AR, AL, SW = 0, 1, 2, 3


def reverse_bits(data, bit_range):
    mask = (1 << bit_range) - 1
    out = 0
    for i in range(bit_range):
        if data & (1 << i):
            out |= 1 << (bit_range - 1 - i)
    return (data & ~mask) | out


def birrd_shape(AW):
    """(stages, switches per stage) of the BIRRD for an AW-wide NEST."""
    lg = int(log2(AW))
    return (2 * lg if AW > 4 else 2 * lg - 1), AW // 2


def stage_bits(stage, AW):
    lg = int(log2(AW))
    return 2 if stage == 0 else min(lg, 2 + stage, 2 * lg - stage)


def feather(AW, AH):
    """The fabric: NEST (AH x AW/2 pair-PEs) into BIRRD (P0 x P1 switches).

    Arguments of a launch:
      X[AH, AW/2, 2]        the stationary input activations, one per column
      W[AH, AW/2, 2 * AH]   each PE's two weight files, AH entries a column
      Inst[P0, P1, 2]       one BIRRD command per switch (entry 0)
      YL[AH, P1], YR[AH, P1]  the last stage's left and right outputs:
                            output column 2j is YL[:, j], 2j + 1 is YR[:, j]
    """
    P0, P1 = birrd_shape(AW)
    W2 = 2 * AH

    class PEIO(spmw.Interface):
        p0_in = spmw.In(int32)
        p0_out = spmw.Out(int32)
        p1_in = spmw.In(int32)
        p1_out = spmw.Out(int32)
        x = spmw.MemIn(int8[2])
        w = spmw.MemIn(int8[W2])

    class SwIO(spmw.Interface):
        in_l = spmw.In(int32)
        in_r = spmw.In(int32)
        out_l = spmw.Out(int32)
        out_r = spmw.Out(int32)
        # Two entries, the second unused: a one-entry file arrives as a scalar.
        cmd = spmw.MemIn(int8[2])

    nest = spmw.Topology(
        PEIO,
        grid=(AH, AW // 2),
        link=lambda k, j: {
            PEIO.p0_out: spmw.to((k + 1, j), PEIO.p0_in),
            PEIO.p1_out: spmw.to((k + 1, j), PEIO.p1_in),
        },
    )

    def wiring(stage, j):
        if stage == P0 - 1:
            return {}
        bits = stage_bits(stage, AW)
        out = {}
        for port, q in ((SwIO.out_l, 2 * j), (SwIO.out_r, 2 * j + 1)):
            pos = reverse_bits(q, bits)
            target = SwIO.in_l if pos % 2 == 0 else SwIO.in_r
            out[port] = spmw.to((stage + 1, pos // 2), target)
        return out

    birrd = spmw.Topology(SwIO, grid=(P0, P1), link=wiring)

    @spmw.unit
    def pe(io: PEIO):
        # One tile: AH output rows. Row i takes weight i of each file; the
        # column's partial sum comes down from the row above and leaves below.
        for i in range(AH):
            p0: int32 = io.p0_in.get()
            p1: int32 = io.p1_in.get()
            x0: int32 = io.x[0]
            x1: int32 = io.x[1]
            w0: int32 = io.w[i]
            w1: int32 = io.w[AH + i]
            io.p0_out.put(p0 + x0 * w0)
            io.p1_out.put(p1 + x1 * w1)

    @spmw.unit
    def switch(io: SwIO):
        c: int32 = io.cmd[0]
        for _i in range(AH):
            l: int32 = io.in_l.get()
            r: int32 = io.in_r.get()
            ol: int32 = l
            orr: int32 = r
            if c == 1:  # add right
                orr = l + r
            elif c == 2:  # add left
                ol = l + r
            elif c == 3:  # swap
                ol = r
                orr = l
            io.out_l.put(ol)
            io.out_r.put(orr)

    @spmw.fabric
    def engine(
        X: int8[AH, AW // 2, 2],
        W: int8[AH, AW // 2, W2],
        Inst: int8[P0, P1, 2],
        YL: int32[AH, P1],
        YR: int32[AH, P1],
    ):
        P = spmw.place(pe, on=nest)
        B = spmw.place(switch, on=birrd)
        spmw.shard(X, into=P.x)
        spmw.shard(W, into=P.w)
        spmw.shard(Inst, into=B.cmd)
        spmw.stream_in(0, into=P.p0_in)
        spmw.stream_in(0, into=P.p1_in)
        spmw.link(P.p0_out, to=B.in_l)
        spmw.link(P.p1_out, to=B.in_r)
        (_stage, sw) = B.axes
        spmw.gather(YL, from_=B.out_l, index=(..., sw))
        spmw.gather(YR, from_=B.out_r, index=(..., sw))

    engine.spmw_parts = (pe, switch, AW, AH, P0, P1)
    # The array cosim drives random operands; BIRRD's program has to be a
    # real one, so the GEMM layout program is what the cosim runs.
    inst = np.zeros((P0, P1, 2), dtype=np.int8)
    inst[:, :, 0] = gemm_insts(AW)
    engine.spmw_operands = {"Inst": inst}
    return engine


def feather_stream(AW, AH, NT):
    """The same engine taking `NT` tiles per launch, operands streamed.

    Per tile each PE takes its two activations and two weight files from a
    stream, each switch its command; then the tile's AH rows. What this
    measures in the cosim is throughput: the fabric's cycles per tile once
    tiles follow each other, rather than one tile's latency.

      X[NT * 2, AH, AW/2], W[NT * 2AH, AH, AW/2], Inst[NT, P0, P1]
      YL[NT * AH, P1], YR[NT * AH, P1]
    """
    P0, P1 = birrd_shape(AW)
    W2 = 2 * AH

    class PEIO(spmw.Interface):
        p0_in = spmw.In(int32)
        p0_out = spmw.Out(int32)
        p1_in = spmw.In(int32)
        p1_out = spmw.Out(int32)
        # Streamed as words: the fabric simulator's loaders for an int8 stream
        # indexed by two site axes fail to lower (a store type mismatch), and
        # the token's width does not change the cycle count being measured.
        x_in = spmw.In(int32)
        w_in = spmw.In(int32)

    class SwIO(spmw.Interface):
        in_l = spmw.In(int32)
        in_r = spmw.In(int32)
        out_l = spmw.Out(int32)
        out_r = spmw.Out(int32)
        cmd_in = spmw.In(int32)

    nest = spmw.Topology(
        PEIO,
        grid=(AH, AW // 2),
        link=lambda k, j: {
            PEIO.p0_out: spmw.to((k + 1, j), PEIO.p0_in),
            PEIO.p1_out: spmw.to((k + 1, j), PEIO.p1_in),
        },
    )

    def wiring(stage, j):
        if stage == P0 - 1:
            return {}
        bits = stage_bits(stage, AW)
        out = {}
        for port, q in ((SwIO.out_l, 2 * j), (SwIO.out_r, 2 * j + 1)):
            pos = reverse_bits(q, bits)
            target = SwIO.in_l if pos % 2 == 0 else SwIO.in_r
            out[port] = spmw.to((stage + 1, pos // 2), target)
        return out

    birrd = spmw.Topology(SwIO, grid=(P0, P1), link=wiring)

    @spmw.unit
    def pe(io: PEIO):
        # The file holds int32 words: the fabric simulator carries stream
        # tokens as words, and a store into an int8 array is a type mismatch.
        wf: int32[W2]
        for _t in range(NT):
            x0: int32 = io.x_in.get()
            x1: int32 = io.x_in.get()
            for n in range(W2):
                wn: int32 = io.w_in.get()
                wf[n] = wn
            for i in range(AH):
                p0: int32 = io.p0_in.get()
                p1: int32 = io.p1_in.get()
                w0: int32 = wf[i]
                w1: int32 = wf[AH + i]
                io.p0_out.put(p0 + x0 * w0)
                io.p1_out.put(p1 + x1 * w1)

    @spmw.unit
    def switch(io: SwIO):
        for _t in range(NT):
            c: int32 = io.cmd_in.get()
            for _i in range(AH):
                l: int32 = io.in_l.get()
                r: int32 = io.in_r.get()
                ol: int32 = l
                orr: int32 = r
                if c == 1:
                    orr = l + r
                elif c == 2:
                    ol = l + r
                elif c == 3:
                    ol = r
                    orr = l
                io.out_l.put(ol)
                io.out_r.put(orr)

    @spmw.fabric
    def engine(
        X: int32[NT * 2, AH, AW // 2],
        W: int32[NT * W2, AH, AW // 2],
        Inst: int32[NT, P0, P1],
        YL: int32[NT * AH, P1],
        YR: int32[NT * AH, P1],
    ):
        P = spmw.place(pe, on=nest)
        B = spmw.place(switch, on=birrd)
        (row, col) = P.axes
        (stage, sw) = B.axes
        spmw.stream_in(X, into=P.x_in, index=(..., row, col))
        spmw.stream_in(W, into=P.w_in, index=(..., row, col))
        spmw.stream_in(Inst, into=B.cmd_in, index=(..., stage, sw))
        spmw.stream_in(0, into=P.p0_in)
        spmw.stream_in(0, into=P.p1_in)
        spmw.link(P.p0_out, to=B.in_l)
        spmw.link(P.p1_out, to=B.in_r)
        spmw.gather(YL, from_=B.out_l, index=(..., sw))
        spmw.gather(YR, from_=B.out_r, index=(..., sw))

    engine.spmw_parts = (pe, switch, AW, AH, P0, P1, NT)
    inst = np.zeros((NT, P0, P1), dtype=np.int32)
    inst[:] = gemm_insts(AW)
    engine.spmw_operands = {"Inst": inst}
    return engine


def stream_operands(tiles, AW, AH):
    """`tiles` = [(iActs, weights, inst), ...] into `feather_stream`'s arguments."""
    T = len(tiles)
    P0, P1 = birrd_shape(AW)
    X = np.zeros((T * 2, AH, AW // 2), dtype=np.int32)
    W = np.zeros((T * 2 * AH, AH, AW // 2), dtype=np.int32)
    I = np.zeros((T, P0, P1), dtype=np.int32)
    for t, (iActs, weights, inst) in enumerate(tiles):
        Xt, Wt, It = tile_operands(iActs, weights, inst, AW, AH)
        X[2 * t : 2 * t + 2] = Xt.transpose(2, 0, 1)
        W[2 * AH * t : 2 * AH * (t + 1)] = Wt.transpose(2, 0, 1)
        I[t] = It[:, :, 0]
    return X, W, I


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_stream_two_tiles(target):
    """Two GEMM tiles in one launch of the streaming engine, 4x4."""
    AW = AH = 4
    rng = np.random.default_rng(9)
    inst = gemm_insts(AW)
    tiles = []
    for _ in range(2):
        A = rng.integers(-4, 4, size=(AW // 2, 2 * AH)).astype(np.int8)
        B = rng.integers(-4, 4, size=(2 * AH, AH)).astype(np.int8)
        iActs, weights = gemm_tile(A, B, AW, AH)
        tiles.append((iActs, weights, inst))
    X, W, I = stream_operands(tiles, AW, AH)
    P0, P1 = birrd_shape(AW)
    YL = np.zeros((2 * AH, P1), dtype=np.int32)
    YR = np.zeros((2 * AH, P1), dtype=np.int32)
    spmw.build(feather_stream(AW, AH, 2), target=target)(X, W, I, YL, YR)
    Y = merge_outputs(YL, YR)
    for t, (iActs, weights, inst_t) in enumerate(tiles):
        np.testing.assert_array_equal(
            Y[t * AH : (t + 1) * AH], feather_ref(iActs, weights, inst_t, AW, AH)
        )


# -- the tile in the Allo drivers' terms --------------------------------------


def tile_operands(iActs, weights, inst, AW, AH):
    """The engine's arguments for one tile, from the drivers' tile tensors.

    ``iActs``  [AH, AW] as `examples/feather` lays a tile out (row k, column j)
    ``weights`` [AH, AW, AH]: weights[i, j, k] multiplies iActs[k, j] for
    output row i -- so PE (k, j) holds weights[:, j, k].
    ``inst``   [P0, P1] BIRRD commands.
    """
    P0, P1 = birrd_shape(AW)
    X = np.ascontiguousarray(iActs.reshape(AH, AW // 2, 2)).astype(np.int8)
    W = np.zeros((AH, AW // 2, 2 * AH), dtype=np.int8)
    for k in range(AH):
        for j in range(AW):
            W[k, j // 2, (j % 2) * AH : (j % 2) * AH + AH] = weights[:, j, k]
    I = np.zeros((P0, P1, 2), dtype=np.int8)
    I[:, :, 0] = inst
    return X, W, I


def feather_ref(iActs, weights, inst, AW, AH):
    """NEST then BIRRD in numpy, int32 -- the drivers' arithmetic, unwrapped."""
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


def merge_outputs(YL, YR):
    AH, P1 = YL.shape
    Y = np.zeros((AH, 2 * P1), dtype=np.int32)
    Y[:, 0::2] = YL
    Y[:, 1::2] = YR
    return Y


def gemm_insts(AW):
    """The drivers' BIRRD programs for 'Workload A, change oAct layout'."""
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
    # No layout program shipped for this width: pass everything through, so
    # the outputs are the plain column sums (a legal program for a cosim).
    P0, P1 = birrd_shape(AW)
    return np.zeros((P0, P1), dtype=np.int8)


def conv_insts():
    """The drivers' four programs (AW = 4), one per output position in a line."""
    return [
        np.array([[AL, AL], [AL, PS], [PS, PS]], dtype=np.int8),
        np.array([[AL, AL], [AL, PS], [SW, PS]], dtype=np.int8),
        np.array([[AL, AL], [AR, PS], [PS, PS]], dtype=np.int8),
        np.array([[AL, AL], [AR, PS], [PS, SW]], dtype=np.int8),
    ]


def gemm_tile(A_tile, B_tile, AW, AH):
    """The drivers' layouts: A_tile [Mt, Kt] and B_tile [Kt, Nt] into the
    engine's iActs [AH, AW] and weights [AH, AW, AH], Mt = AW/2, Kt = 2AH,
    Nt = AH. Each K-half goes to one half of the columns."""
    Mt, Kt, Nt = AW // 2, 2 * AH, AH
    assert A_tile.shape == (Mt, Kt) and B_tile.shape == (Kt, Nt)
    left, right = np.hsplit(A_tile, 2)
    iActs = np.ascontiguousarray(np.hstack([left.T, right.T]))  # [AH, AW]
    bl, br = np.vsplit(B_tile, 2)
    cl = np.array([bl.T] * (AW // 2))
    cr = np.array([br.T] * (AW // 2))
    weights = np.ascontiguousarray(np.vstack([cl, cr]).transpose(1, 0, 2))
    return iActs, weights


def gemm_extract(Y, AW):
    """Which output columns carry the Mt results, per the drivers."""
    order = {16: [8, 10, 11, 9, 5, 6, 7, 4], 8: [6, 5, 2, 1], 4: [2, 0]}[AW]
    return Y[:, order]  # [Nt, Mt]


def run_gemm(M, K, N, AW, AH, target="ref", seed=0):
    """A whole GEMM as tile launches, checked exactly. Returns the tile count."""
    rng = np.random.default_rng(seed)
    Mt, Kt, Nt = AW // 2, 2 * AH, AH
    A = rng.integers(-4, 4, size=(M, K)).astype(np.int8)
    B = rng.integers(-4, 4, size=(K, N)).astype(np.int8)
    inst = gemm_insts(AW)
    P0, P1 = birrd_shape(AW)
    run = spmw.build(feather(AW, AH), target=target)
    out = np.zeros((M, N), dtype=np.int64)
    tiles = 0
    for n in range(N // Nt):
        for m in range(M // Mt):
            for k in range(K // Kt):
                iActs, weights = gemm_tile(
                    A[m * Mt : (m + 1) * Mt, k * Kt : (k + 1) * Kt],
                    B[k * Kt : (k + 1) * Kt, n * Nt : (n + 1) * Nt],
                    AW,
                    AH,
                )
                X, W, I = tile_operands(iActs, weights, inst, AW, AH)
                YL = np.zeros((AH, P1), dtype=np.int32)
                YR = np.zeros((AH, P1), dtype=np.int32)
                run(X, W, I, YL, YR)
                Y = merge_outputs(YL, YR)
                np.testing.assert_array_equal(Y, feather_ref(iActs, weights, inst, AW, AH))
                part = gemm_extract(Y, AW)  # [Nt, Mt]
                out[m * Mt : (m + 1) * Mt, n * Nt : (n + 1) * Nt] += part.T
                tiles += 1
    np.testing.assert_array_equal(out, A.astype(np.int64) @ B.astype(np.int64))
    return tiles


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_gemm_4x4(target):
    assert run_gemm(4, 16, 8, 4, 4, target) == 2 * 2 * 2


@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_gemm_8x8(target):
    assert run_gemm(8, 16, 16, 8, 8, target) == 2 * 2 * 1


def test_conv_4x4_matches_numpy():
    """The convolution driver's tiling: a 4x4 engine, channel-last input,
    row-major output, one program per output position within a line."""
    AW = AH = 4
    N, C, H, Wd, M, R, S = 1, 8, 8, 7, 4, 4, 4
    P, Q = H - R + 1, Wd - S + 1
    rng = np.random.default_rng(3)
    x = rng.integers(-2, 2, size=(N, C, H, Wd)).astype(np.int8)
    wts = rng.integers(-2, 2, size=(M, C, R, S)).astype(np.int8)
    ref = np.zeros((N, M, P, Q), dtype=np.int64)
    for n in range(N):
        for m in range(M):
            for p in range(P):
                for q in range(Q):
                    ref[n, m, p, q] = int(
                        (x[n, :, p : p + R, q : q + S].astype(np.int64) * wts[m].astype(np.int64)).sum()
                    )
    x_cl = np.ascontiguousarray(x.transpose(0, 2, 3, 1))  # NHWC
    w_flat = wts.reshape(M, C, R * S)
    insts = conv_insts()
    run = spmw.build(feather(AW, AH), target="ref")
    P0, P1 = birrd_shape(AW)
    got = np.zeros((N, M * P * Q // AW, AW), dtype=np.int64)
    tiles = 0
    for n in range(N):
        for p in range(P):
            for q in range(Q):
                win = x_cl[n, p : p + R, q : q + S, :].reshape(R * S, C)
                for mt in range(0, M, AH):
                    acc = np.zeros((AH, AW), dtype=np.int64)
                    line, pos = (p * Q + q) // AW, (p * Q + q) % AW
                    for ct in range(0, C, AW):
                        for vn in range(0, R * S, AH):
                            iActs = np.ascontiguousarray(win[vn : vn + AH, ct : ct + AW])
                            weights = np.ascontiguousarray(w_flat[mt : mt + AH, ct : ct + AW, vn : vn + AH])
                            X, W, I = tile_operands(iActs, weights, insts[pos], AW, AH)
                            YL = np.zeros((AH, P1), dtype=np.int32)
                            YR = np.zeros((AH, P1), dtype=np.int32)
                            run(X, W, I, YL, YR)
                            acc += merge_outputs(YL, YR)
                            tiles += 1
                    for m in range(mt, mt + AH):
                        got[n, m * P * Q // AW + line, pos] = acc[m - mt, pos]
    np.testing.assert_array_equal(got.reshape(N * M, P * Q), ref.reshape(N * M, P * Q))
    assert tiles == P * Q * (M // AH) * (C // AW) * (R * S // AH)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
