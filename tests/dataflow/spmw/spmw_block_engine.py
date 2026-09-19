# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW's transformer-block engine: the mirror of Gemmini's `MxuVpuNorm`.

Gemmini's block engine is three things -- a `MeshWithDelays`, a `Normalizer`
and an `AccumulatorScale` -- and this is those three things in SPMW, operation
for operation. Every constant, every rounding mode and every truncation below
was read out of Gemmini's source at commit 8c3f992 rather than inferred, and
the places where the two could still differ are named in the comments.

**The accumulator is outside both engines.** `MxuVpuNorm` wires the mesh to
the scale path directly and has no accumulator SRAM; real Gemmini's normaliser
is driven by *accumulator reads*, which is what lets it make three passes over
a row without recomputing the matmul. So this fabric has the same two disjoint
pipelines Gemmini's scope does, and the host holds the accumulator for both:

    A, W, Pin  -> [16x16 mac mesh] -> Psum      (int32, to the accumulator)
    Acc x3     -> [red1 -> sum1 -> sca1 -> red2 -> sum2 -> sca2 -> scl] -> Y

Comparing them is then like for like: neither side counts an SRAM the other
one has.

**Why seven roles and not two.** Gemmini's `Normalizer` keeps `mean`, `max`,
`inv_stddev` and `sum` in registers that its own accumulation lanes read back,
which is a register loop inside one module. A SPMW fabric is a dataflow graph,
and the same shape would be a *cycle* -- lanes to statistics to lanes. It is
unrolled in space instead: `red1` reduces across the lanes, `sum1` finishes the
reduction, `sca1` turns it into a scalar, `red2` reduces the second pass
against that scalar, `sum2` and `sca2` do the same again, and `scl` applies
both. Feed-forward, no cycle, and every stage is flat enough to pipeline.

The cost of that choice is 32 reduce lanes where Gemmini has 16, and the
buffering that lets a lane hold its data while the scalar it needs is still
being computed. The saving is that Gemmini's state machine -- fifteen states,
two `Stats` bundles, and the arbitration that picks which of them owns each
functional unit -- is not there at all: the sequence is the topology.
"""

import os

import numpy as np

import allo.spmw as spmw
from allo.ir.types import int8, int32, float32

#: `Activation` in Gemmini, same encoding.
M_NONE, M_RELU, M_LN, M_GELU, M_SM = 0, 1, 2, 3, 4

#: Per-site integer constants.  `nbeat` and `total` are runtime rather than
#: folded because Gemmini's `len` is a runtime port; folding them would make
#: SPMW's loop bounds constants and Gemmini's not, and the comparison would be
#: between a design that knows its shape and one that does not.
K_MODE, K_NBEAT, K_TOTAL, K_QB, K_QC, K_QLN2, K_QLN2I, K_NLEN = range(8)
NK = 8
NF = 2  # the float32 scale, and one spare


def _log2(n):
    bits = n.bit_length() - 1
    if 1 << bits != n:
        raise ValueError(f"the array side must be a power of two, got {n}")
    return bits


def block_engine(dim=16, tiles=4, nacc=64, nrow=4, link_depth=2,
                 scalar_depth=None):
    """The mesh and the normalise/scale path, as one fabric.

    `nacc` is how many accumulator beats a launch carries and `nrow` how
    many logical rows they make up, so `nacc / nrow` is Gemmini's `len`
    divided by the mesh width.

    `scalar_depth` is the depth of the links that carry **one token a row**
    rather than one a beat.  Two is not enough: `sca1` takes 39 cycles an
    iteration and `sca2` takes 65, while a row is only `nbeat` beats long, so
    a stage has to run two or three rows ahead of the one after it to keep the
    latency hidden, and a depth-2 register slice does not let it.  Measured at
    4x4, `nbeat = 16`: the cost of a row is `75 + 1.0 * nbeat` cycles, the
    beats exactly one cycle each and 75 of fixed cost that is almost exactly
    `sca2`'s own latency.  `SPMW_SCALAR_DEPTH` overrides it so the depth stays
    measurable rather than asserted.

    Each of the three passes reads the accumulator for itself -- `Acc1`,
    `Acc2`, `Acc3` are the same rows three times -- so no lane holds a row
    while the stage ahead of it computes a scalar.  That is Gemmini's
    discipline and not a simplification of it: its normaliser makes three
    accumulator reads over a row, one to sum, one to accumulate the variance
    against the mean, and one for the activated output.

    Forwarding the row down the lanes instead was tried first, and it
    **deadlocks**: a `red1` lane blocks on `d_out` before it reaches
    `r_out`, so a full data link stalls the reduction chain, the statistics
    sees the row's last value, the scalar never arrives and the data link
    never drains.  Surviving it needs a link deeper than a row plus the
    divider's 36 cycles -- past `BRAM_FIFO_DEPTH`, into block RAM, on 32
    lanes -- to buy a pass Gemmini pays for anyway.
    """
    scalar_depth = int(os.environ.get("SPMW_SCALAR_DEPTH", "16")) \
        if scalar_depth is None else scalar_depth
    kw = tiles // 4
    if tiles % 4:
        raise ValueError(f"the packed weight file needs a multiple of 4, got {tiles}")
    outs = tiles * dim
    sbits = _log2(dim)

    # -- the mesh -----------------------------------------------------------
    # E3's fixed-function cell, unchanged: the comparison of the GEMM half is
    # the one already measured, and changing the cell here would silently make
    # it a different measurement.

    class CellIO(spmw.Interface):
        a_in = spmw.In(int8, depth=link_depth)
        a_out = spmw.Out(int8)
        p_in = spmw.In(int32, depth=link_depth)
        p_out = spmw.Out(int32)
        w_in = spmw.In(int32, depth=link_depth)
        w_out = spmw.Out(int32)

    mxu = spmw.Topology(
        CellIO,
        grid=(dim, dim),
        link=lambda i, j: {
            CellIO.a_out: spmw.to((i, j + 1), CellIO.a_in),
            CellIO.w_out: spmw.to((i, j + 1), CellIO.w_in),
            CellIO.p_out: spmw.to((i + 1, j), CellIO.p_in),
        },
    )

    @spmw.unit
    def mac(io: CellIO):
        wf: int32[kw]
        n: int32 = io.w_in.get()
        io.w_out.put(n - kw)
        for i in range(kw):
            wf[i] = io.w_in.get()
        for _j in range(n - kw):
            fwd: int32 = io.w_in.get()
            io.w_out.put(fwd)
        for r in range(outs):
            a = io.a_in.get()
            p = io.p_in.get()
            io.a_out.put(a)
            idx: int32 = r >> sbits
            packed: int32 = wf[idx >> 2]
            byte: int32 = (packed >> ((idx & 3) * 8)) & 255
            wt: int32 = (byte ^ 128) - 128
            io.p_out.put(p + a * wt)

    # -- the reduce lanes ---------------------------------------------------
    # Gemmini's `AccumulationLanes` and `MaxLanes` are a reduction *tree* of 16
    # with `latency = 4`; these are chains of 16, one adder deep per lane --
    # the same 16 adders with 12 more cycles of latency and no tree wiring.
    # Latency is amortised over a row, so the chain costs nothing the interval
    # can see, and it is the shape SPMW writes naturally.
    #
    # Two roles rather than one placed twice.  A shared role would put a
    # maximum, a square *and* an `iexp` in all 32 lanes; Gemmini splits them
    # the same way, `MaxLanes` holding only a comparator.

    class Red1IO(spmw.Interface):
        d_in = spmw.In(int32, depth=link_depth)
        r_in = spmw.In(int32, depth=link_depth)
        r_out = spmw.Out(int32)
        k = spmw.MemIn(int32[NK])

    class Red2IO(spmw.Interface):
        d_in = spmw.In(int32, depth=link_depth)
        r_in = spmw.In(int32, depth=link_depth)
        r_out = spmw.Out(int32)
        m_in = spmw.In(int32, depth=scalar_depth)
        m_out = spmw.Out(int32)
        k = spmw.MemIn(int32[NK])

    chain1 = spmw.Topology(
        Red1IO, grid=(dim,),
        link=lambda j: {Red1IO.r_out: spmw.to((j + 1,), Red1IO.r_in)},
    )
    chain2 = spmw.Topology(
        Red2IO, grid=(dim,),
        link=lambda j: {
            Red2IO.r_out: spmw.to((j + 1,), Red2IO.r_in),
            Red2IO.m_out: spmw.to((j + 1,), Red2IO.m_in),
        },
    )

    @spmw.unit
    def red1(io: Red1IO):
        """Pass one: the sum for LayerNorm, the maximum for softmax.

        `AccumulationLanes` and `MaxLanes` in one lane, selected by a constant,
        because the two are an add and a compare against the same chain and
        splitting them into two placements would double the links to save one
        comparator.
        """
        mode: int32 = io.k[K_MODE]
        total: int32 = io.k[K_TOTAL]
        for _i in range(total):
            d: int32 = io.d_in.get()
            acc: int32 = io.r_in.get()
            if mode == M_SM:
                if d > acc:
                    acc = d
                io.r_out.put(acc)
            else:
                io.r_out.put(acc + d)

    @spmw.unit
    def red2(io: Red2IO):
        """Pass two: `(d - mean)^2` for LayerNorm, `iexp(d - max)` for softmax.

        The scalar from pass one arrives once a row and is forwarded down the
        chain, which is why `d_in` is the deep link: lane 15 sees `mean`
        fifteen cycles after lane 0, and its row has to wait somewhere.  In
        Gemmini it waits in the accumulator SRAM and is read again.
        """
        mode: int32 = io.k[K_MODE]
        nbeat: int32 = io.k[K_NBEAT]
        total: int32 = io.k[K_TOTAL]
        qb: int32 = io.k[K_QB]
        qc: int32 = io.k[K_QC]
        qln2: int32 = io.k[K_QLN2]
        qln2i: int32 = io.k[K_QLN2I]

        # Flat, not nested: E3 measured that a loop whose trip count is a
        # runtime field cannot be pipelined when it sits inside another loop,
        # and that the steps under it pay for the whole nest.  One loop with a
        # beat counter has the same control and pipelines at II=1.
        b: int32 = 0
        m: int32 = 0
        for _i in range(total):
            if b == 0:
                m = io.m_in.get()
                io.m_out.put(m)
            d: int32 = io.d_in.get()
            acc: int32 = io.r_in.get()

            v: int32 = 0
            if mode == M_LN:
                t: int32 = d - m
                v = t * t
            elif mode == M_SM:
                q: int32 = d - m
                nq: int32 = -q
                # `(-q * qln2_inv) >> 16`, truncated to 32 bits.  The product
                # is taken wide because Gemmini's `*` on SInt(32) is 64 bits
                # before the shift, and psum magnitudes overflow the narrow
                # form.
                z: int32 = int((int(nq) * int(qln2i)) >> 16)
                zs: int32 = z
                if ((z >> 5) & 2047) != 0:   # bits 5..15, exactly as Gemmini
                    zs = 32
                qp: int32 = q + z * qln2
                u: int32 = qp + qb
                poly: int32 = u * u + qc
                if zs < 32:   # a 32-bit shift is 0 in Chisel and UB in C
                    v = poly >> zs
            io.r_out.put(acc + v)

            b = b + 1
            if b == nbeat:
                b = 0

    # -- the scalar units ---------------------------------------------------
    # Four sites, not two, and the split is what makes them keep up.
    #
    # With the reduction and the scalar arithmetic in one unit the body is a
    # per-row loop wrapping a reduce whose trip count is a runtime field, and
    # HLS reports that outer loop `Pipelined no` -- so a row's divide, square
    # root and reciprocal all sit in front of the next row's first beat.  At
    # 4x4 that measured **94 cycles a row** against 16 beats of actual work.
    #
    # Split, both loops are flat: `sum*` reduces at II=1 and `sca*` sees one
    # value a row with only a compile-time loop inside it, so HLS can pipeline
    # it and rows overlap.  It costs a pipelined divider where the folded form
    # had a sequential one.
    #
    # Gemmini cannot do this and the reason is structural, not an oversight:
    # its divider, square root and reciprocal are *one each*, shared across
    # the accumulator and arbitrated by the `Stats` state machine, so its rows
    # serialise on them.  That is why its LayerNorm row costs 130 cycles of
    # which almost all is scalar latency.  Two statistics banks give it
    # two-way overlap and no more.

    class Sum1IO(spmw.Interface):
        t_in = spmw.In(int32, depth=scalar_depth)
        u_out = spmw.Out(int32)
        k = spmw.MemIn(int32[NK])

    class Sca1IO(spmw.Interface):
        x_in = spmw.In(int32, depth=scalar_depth)
        s_out = spmw.Out(int32)
        k = spmw.MemIn(int32[NK])

    class Sum2IO(spmw.Interface):
        t_in = spmw.In(int32, depth=scalar_depth)
        u_out = spmw.Out(int32)
        k = spmw.MemIn(int32[NK])

    class Sca2IO(spmw.Interface):
        x_in = spmw.In(int32, depth=scalar_depth)
        v_out = spmw.Out(float32)
        k = spmw.MemIn(int32[NK])
        f = spmw.MemIn(float32[NF])

    one = spmw.Grid((1,))

    @spmw.unit
    def sum1(io: Sum1IO):
        """Pass one's reduction across the lane chain: a sum, or a maximum."""
        mode: int32 = io.k[K_MODE]
        nbeat: int32 = io.k[K_NBEAT]
        total: int32 = io.k[K_TOTAL]
        b: int32 = 0
        acc: int32 = 0
        for _i in range(total):
            t: int32 = io.t_in.get()
            if b == 0:
                acc = 0
                if mode == M_SM:
                    acc = -2147483647
            if mode == M_SM:
                if t > acc:
                    acc = t
            else:
                acc = acc + t
            b = b + 1
            if b == nbeat:
                b = 0
                io.u_out.put(acc)

    @spmw.unit
    def sca1(io: Sca1IO):
        """`mean` or `max`: the first pass's scalar.

        `mean` is Gemmini's `Arithmetic.divider`, and that divider is *not*
        IEEE single -- it is `Float(expWidth = log2Up(32)+1, sigWidth = 32)`
        with `round_minMag` on both conversions and on the divide.  A 32-bit
        significand holds any int32 exactly and `round_minMag` truncates
        towards zero, so the whole float round trip is integer division with C
        semantics.  Written as a sign-corrected division of magnitudes so that
        it means the same under Python's floor `//` in the reference simulator
        and `arith.divsi`'s truncation in the hardware.
        """
        mode: int32 = io.k[K_MODE]
        nrow: int32 = io.k[K_TOTAL]
        cnt: int32 = io.k[K_NLEN]
        for _r in range(nrow):
            x: int32 = io.x_in.get()
            # Only LayerNorm wants a mean.  Softmax wants the maximum
            # unchanged, and a pointwise activation wants neither -- putting
            # them all through the divider measured **31 cycles a row** for
            # IGELU where Gemmini's state machine, which skips the passes it
            # does not need, takes 16.
            if mode == M_LN:
                mag: int32 = x
                sgn: int32 = 1
                if x < 0:
                    mag = -x
                    sgn = -1
                io.s_out.put(sgn * (mag // cnt))
            else:
                io.s_out.put(x)

    @spmw.unit
    def sum2(io: Sum2IO):
        """Pass two's reduction: always a sum, of squares or of exponentials."""
        nbeat: int32 = io.k[K_NBEAT]
        total: int32 = io.k[K_TOTAL]
        b: int32 = 0
        acc: int32 = 0
        for _i in range(total):
            t: int32 = io.t_in.get()
            if b == 0:
                acc = 0
            acc = acc + t
            b = b + 1
            if b == nbeat:
                b = 0
                io.u_out.put(acc)

    @spmw.unit
    def sca2(io: Sca2IO):
        """`inv_stddev` or `inv_sum_exp`: the second pass's scalar.

        LayerNorm divides the sum of squared deviations by the count, takes
        the integer square root of that and then a *float* reciprocal --
        Gemmini's divider, `IntSqrt` and `Arithmetic.reciprocal` in that
        order, the last being IEEE single.  Softmax divides 127 by the sum of
        exponentials, and the 127 is Gemmini's own: "softmax maximum is 127
        for signed int8".  Both are then multiplied by the requantisation
        scale in float32, which is Gemmini's `MulPipe`.
        """
        mode: int32 = io.k[K_MODE]
        nrow: int32 = io.k[K_TOTAL]
        cnt: int32 = io.k[K_NLEN]
        sc: float32 = io.f[0]
        for _r in range(nrow):
            tot: int32 = io.x_in.get()
            out: float32 = sc
            if mode == M_LN:
                # The **mean** of the squared deviations.  Gemmini runs its
                # one divider a second time here -- `get_sum` ->
                # `get_variance` -- and the square root sees `sum/count`.
                vmag: int32 = tot
                vsgn: int32 = 1
                if tot < 0:
                    vmag = -tot
                    vsgn = -1
                v: int32 = vsgn * (vmag // cnt)
                # Gemmini's `IntSqrt`: restoring, two bits a step, sixteen
                # steps for a 32-bit input, exact floor(sqrt(x)).  Written out
                # rather than approximated through `sqrtf`, because the float
                # square root of a 32-bit variance is not the same integer.
                # The trip count is a compile-time constant, which is what
                # lets the enclosing per-row loop pipeline.
                x: int32 = v
                a: int32 = 0
                qq: int32 = 0
                for _s in range(16):
                    hi: int32 = (x >> 30) & 3
                    ac: int32 = (a << 2) | hi
                    tt: int32 = ac - ((qq << 2) | 1)
                    neg: int32 = 0
                    if tt < 0:
                        neg = 1
                    if neg != 0:
                        a = ac
                    else:
                        a = tt
                    qq = ((qq << 1) | (1 - neg)) & 65535
                    x = x << 2
                sd: int32 = qq
                if sd == 0:   # Gemmini's own fallback for a zero deviation
                    sd = 1
                out = sc / float(sd)
            elif mode == M_SM:
                den: float32 = float(tot)
                out = sc * (127.0 / den)
            io.v_out.put(out)

    # -- the scale lanes ----------------------------------------------------

    class SclIO(spmw.Interface):
        d_in = spmw.In(int32, depth=link_depth)
        m_in = spmw.In(int32, depth=scalar_depth)
        m_out = spmw.Out(int32)
        v_in = spmw.In(float32, depth=scalar_depth)
        v_out = spmw.Out(float32)
        y_out = spmw.Out(int8)
        k = spmw.MemIn(int32[NK])

    chain3 = spmw.Topology(
        SclIO, grid=(dim,),
        link=lambda j: {
            SclIO.m_out: spmw.to((j + 1,), SclIO.m_in),
            SclIO.v_out: spmw.to((j + 1,), SclIO.v_in),
        },
    )

    @spmw.unit
    def scale(io: SclIO):
        """`AccumulatorScale`: the activation, the float scale, the clip.

        `scale_func` in Gemmini is `INToRecFN` -> `MulAddRecFN` -> `RecFNToIN`,
        all `round_near_even`, then a saturating clip to int8.  The
        int-to-float and the multiply are what C gives for free; the
        float-to-int is not -- C truncates -- so the round-half-to-even is
        written out.  It is not a detail that can be skipped: the
        requantisation scale is a power of two, so an exact half falls out of
        one value in every 256.
        """
        mode: int32 = io.k[K_MODE]
        nbeat: int32 = io.k[K_NBEAT]
        total: int32 = io.k[K_TOTAL]
        qb: int32 = io.k[K_QB]
        qc: int32 = io.k[K_QC]
        qln2: int32 = io.k[K_QLN2]
        qln2i: int32 = io.k[K_QLN2I]

        b: int32 = 0
        m: int32 = 0
        sv: float32 = 0.0
        for _i in range(total):
            if b == 0:
                m = io.m_in.get()
                sv = io.v_in.get()
                io.m_out.put(m)
                io.v_out.put(sv)
            d: int32 = io.d_in.get()

            e: int32 = d
            if mode == M_LN:
                e = d - m
            elif mode == M_SM:
                q: int32 = d - m
                nq: int32 = -q
                z: int32 = int((int(nq) * int(qln2i)) >> 16)
                zs: int32 = z
                if ((z >> 5) & 2047) != 0:
                    zs = 32
                qp: int32 = q + z * qln2
                u: int32 = qp + qb
                poly: int32 = u * u + qc
                ee: int32 = 0
                if zs < 32:
                    ee = poly >> zs
                e = ee
            elif mode == M_GELU:
                sgn: int32 = 1
                qa: int32 = d
                if d < 0:
                    sgn = -1
                    qa = -d
                lim: int32 = -qb
                if qa > lim:
                    qa = lim
                g: int32 = qa + qb
                gp: int32 = g * g + qc
                e = d * (sgn * gp + qc)
            elif mode == M_RELU:
                if d < 0:
                    e = 0

            # The float32 scale.  Clamped first: `int()` of a float outside
            # int32 is undefined in C where Chisel saturates, and everything
            # outside the int8 window clips to the same edge either way, so
            # the clamp changes no result and removes the undefined case.
            fv: float32 = float(e) * sv
            if fv > 4096.0:
                fv = 4096.0
            if fv < -4096.0:
                fv = -4096.0
            iv: int32 = int(fv)          # C truncates towards zero
            fr: float32 = fv - float(iv)
            if fr > 0.5:
                iv = iv + 1
            elif fr < -0.5:
                iv = iv - 1
            elif fr == 0.5:
                if (iv & 1) != 0:
                    iv = iv + 1
            elif fr == -0.5:
                if (iv & 1) != 0:
                    iv = iv - 1
            if iv > 127:
                iv = 127
            if iv < -128:
                iv = -128
            io.y_out.put(iv)

            b = b + 1
            if b == nbeat:
                b = 0

    # -- the fabric ---------------------------------------------------------

    @spmw.fabric
    def engine(
        A: int8[outs, dim],
        W: int32[dim * kw + 1, dim],
        Pin: int32[outs, dim],
        Psum: int32[outs, dim],
        Acc1: int32[nacc, dim],
        Acc2: int32[nacc, dim],
        Acc3: int32[nacc, dim],
        Kr1: int32[dim, NK],
        Kr2: int32[dim, NK],
        Ksu1: int32[1, NK],
        Ksc1: int32[1, NK],
        Ksu2: int32[1, NK],
        Ksc2: int32[1, NK],
        Fsc2: float32[1, NF],
        Kscl: int32[dim, NK],
        Z1: int32[nacc],
        Z2: int32[nacc],
        Tail: int32[nrow],
        Vail: float32[nrow],
        Y: int8[nacc, dim],
    ):
        P = spmw.place(mac, on=mxu)
        R1 = spmw.place(red1, on=chain1)
        U1 = spmw.place(sum1, on=one)
        C1 = spmw.place(sca1, on=one)
        R2 = spmw.place(red2, on=chain2)
        U2 = spmw.place(sum2, on=one)
        C2 = spmw.place(sca2, on=one)
        V = spmw.place(scale, on=chain3)

        # The mesh.  Psums go straight out to the host's accumulator, which is
        # where Gemmini's go too -- `MxuVpuNorm` has no accumulator SRAM, and
        # neither does this.
        spmw.stream_in(A, into=P.a_in, index=(..., P.rows))
        spmw.stream_in(W, into=P.w_in, index=(..., P.rows))
        spmw.stream_in(Pin, into=P.p_in, index=(..., P.cols))
        spmw.gather(Psum, from_=P.p_out, index=(..., P.cols))

        # The normalise/scale path, fed from that accumulator.
        spmw.shard(Kr1, into=R1.k)
        spmw.shard(Kr2, into=R2.k)
        spmw.shard(Ksu1, into=U1.k)
        spmw.shard(Ksc1, into=C1.k)
        spmw.shard(Ksu2, into=U2.k)
        spmw.shard(Ksc2, into=C2.k)
        spmw.shard(Fsc2, into=C2.f)
        spmw.shard(Kscl, into=V.k)
        (l1,) = R1.axes
        (l2,) = R2.axes
        (l3,) = V.axes
        # Three reads of the accumulator, one per pass, which is exactly
        # what Gemmini's normaliser does: `NormCmd.SUM` then
        # `INV_STDDEV` then the activated read, each an accumulator
        # read of the same row.  The lanes hold nothing between them.
        spmw.stream_in(Acc1, into=R1.d_in, index=(..., l1))
        spmw.stream_in(Acc2, into=R2.d_in, index=(..., l2))
        spmw.stream_in(Acc3, into=V.d_in, index=(..., l3))
        spmw.stream_in(Z1, into=R1.r_in, index=(...,))
        spmw.stream_in(Z2, into=R2.r_in, index=(...,))
        spmw.link(R1.r_out, to=U1.t_in)
        spmw.link(U1.u_out, to=C1.x_in)
        # `mean`/`max` has two consumers -- the second reduction and the
        # scale -- so it goes down `red2`'s chain and out the far end into
        # the scale lanes' own chain, rather than being duplicated.
        spmw.link(C1.s_out, to=R2.m_in)
        spmw.link(R2.m_out, to=V.m_in)
        spmw.link(R2.r_out, to=U2.t_in)
        spmw.link(U2.u_out, to=C2.x_in)
        spmw.link(C2.v_out, to=V.v_in)
        spmw.gather(Tail, from_=V.m_out, index=(...,))
        spmw.gather(Vail, from_=V.v_out, index=(...,))
        spmw.gather(Y, from_=V.y_out, index=(..., l3))

    engine.spmw_bind_mul_fabric = os.environ.get("SPMW_BIND_MUL", "1") != "0"
    engine.spmw_shape = dict(dim=dim, tiles=tiles, outs=outs, nacc=nacc, nrow=nrow)
    return engine


# -- the measured configurations ---------------------------------------------
#
# One engine, three launches.  The RTL does not depend on which activation a
# launch runs -- the mode is a constant in the lane's memory, exactly as
# Gemmini's is a CSR field -- so the resource numbers come from one build and
# the cycle numbers from one cosimulation per activation.


def block_of(size, mode=None, nrow=None, ln=None, seed=0):
    """The block engine with a launch's operands attached, for the array build.

    `ln` defaults to the block's own row lengths: 256 for LayerNorm, which is
    `d_model`; 64 for softmax, which is the sequence length; and 256 for the
    pointwise activations, where the row length only sets how often the scalar
    chain turns over.
    """
    from spmw_block_drive import SPMW_BLOCK_ORDER, launch_operands

    mode = M_LN if mode is None else mode
    # `E8_NROW` and `E8_LEN` sweep the launch shape without a new design name.
    # The marginal cost of a row is what the comparison needs -- both engines
    # are measured at two row counts and the difference taken -- and the fill
    # is not separable any other way.
    nrow = int(os.environ.get("E8_NROW", "4")) if nrow is None else nrow
    # `E8_TILES` varies the mesh's work while the scale path's stays fixed,
    # which is how the mesh's own rate is separated: the two pipelines share
    # no channel, so the marginal cost of a tile is the mesh's alone.
    tiles = int(os.environ.get("E8_TILES", "4"))
    ln = int(os.environ.get("E8_LEN", "0")) or ln
    ln = (64 if mode == M_SM else 256) if not ln else ln
    nacc = nrow * (ln // size)
    eng = block_engine(dim=size, tiles=tiles, nacc=nacc, nrow=nrow)
    ops, want, _ = launch_operands(size, tiles, mode, nrow, ln, seed)
    eng.spmw_operands = {n: ops[n] for n in SPMW_BLOCK_ORDER if n != "Y"}
    eng.spmw_tokens_per_transform = nacc * size
    eng.spmw_block = dict(mode=mode, nrow=nrow, ln=ln, nacc=nacc, expected=want)
    print(f"E8 BLOCK design: mode={mode} nrow={nrow} len={ln} "
          f"nbeat={ln // size} nacc={nacc} tiles={tiles} "
          f"steps={tiles * size}")
    return eng
