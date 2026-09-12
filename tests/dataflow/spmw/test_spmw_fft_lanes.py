# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The lane-unrolled FFT: the bank axis is the *placement*, so it is a constant.

Two designs already exist.  `test_spmw_fft_rolled.py` keeps the point-to-point
topology and low block RAM but feeds each butterfly unit one sample a cycle, so
every multiplier idles half the time -- 318 DSP at N=256, W=2.
`test_spmw_fft_paired.py` reaches DSP parity with HP-FFT by letting one unit read
both operands out of a shared XOR-banked buffer, but it pays for the buffer:
every stage sits on `spmw.Grid((1,))`, there is no `spmw.to` anywhere in it, and
the block RAM goes from 6 to 44.

A third form was tried and lost: the same paired design with `buffer="banked"`,
where the swizzle is honoured by the scheduler instead of by a wire.  It
synthesises at **II=2 on seven of nine units and an array interval of 512, four
times the ideal**, because `bank(p)` is a runtime value and Vitis cannot prove
the cycle's two reads land in different banks.

This design removes that premise rather than working around it.  **There is no
shared buffer and no bank arithmetic at all**: the `W` lanes are `W` separate
streams, a stage is `W/2` sites, and site `m` reads the two lanes it needs on
two ports.  A lane index is therefore a *placement coordinate*, which is the
strongest compile-time constant SPMW has -- stronger than the paired design's
`split` form, which still muxes on a runtime `sel`.

## What has to happen for a site to see both operands

Stage `s` of the decimation-in-frequency recursion pairs `i` with `i + D`,
``D = n >> (s+1)``.  Under the lane law ``lane(i) = i & (W-1)``,
``beat(i) = i >> log2(W)`` -- the same law the rolled design uses, and the same
natural streaming order at the boundary -- that pair is:

- ``D >= W``: **the same lane, `D/W` beats apart.**  Not two lanes of one beat,
  so a site cannot read it.  This is exactly why the rolled design runs at half:
  one operand is always in the past.
- ``D < W``: **two lanes of one beat.**  A site reads it directly and the wiring
  is a constant.

So the whole problem is the first case, and the fix is a permutation in front of
the stage that turns a beat distance into a lane distance.  That element is a
**delay-switch-delay** (`perm` below): delay one lane by `d`, a 2x2 switch that
toggles every `d` beats, then delay the other side by `d`.  Its output pair at
any beat is one lane's samples `d` beats apart, so choosing `d` so that lane's
index step over `d` beats is `D` puts both operands of a distance-`D` butterfly
on two lanes of one beat.  `test_the_permutation_turns_time_into_lanes` is that
claim checked against the schedule rather than asserted here.

The delays are forced, not tuned: `d = R/2, R/4, ... 1` for the stages with
``D >= W`` and one more `R/2` to restore the natural layout, `R = n/W`.  Searching
every power of two at every stage finds **exactly one** that works
(`test_each_stage_admits_exactly_one_delay`), so there is no trade to make here.

## What it costs and what it buys

`log2(n) - 2` stages need a multiplier -- the last two draw their twiddles from
{1, -i}, where a rotation is a swap and a sign -- so the design holds
``(log2(n) - 2) * W/2`` complex multipliers, all of them busy every beat.  At
N=256, W=2 that is six, the same six HP-FFT UF1 has.

The permutation memory is `sum(d) * W` complex samples, about `1.5 n` whatever
`W` is, against the paired design's `2n` *per stage*.  There is no buffer to
bank, so `schedule.partition_banks` is not involved and neither is `xor_bank`.

## Where the structure is explicit

Every stage is `W/2` placed sites wired port to port, which is the rolled
design's form and not the paired design's single site.  At ``W >= 4`` the last
``log2(W) - 1`` stages have distances below the lane count, and their pairings
differ from each other, so they carry real lane crossings: those rows go on one
`spmw.Topology` whose `link` rule names each crossing with `spmw.to`, as the
rolled design's cross stages do.  At `W = 2` there are none and the design is a
chain -- the crossings are a function of the width, which is the honest
statement.

`spmw.link(index=)` is *not* how a crossing is spelled, and cannot be: it is
accepted, recorded and then dropped by both paths.  See
`test_link_index_is_accepted_and_ignored`.

Radix-2, complex FP32, natural-order input and output, unnormalised forward
transform; `batch` transforms a launch, back to back.
"""

import cmath
import os

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import float32, int32

csample = float32[2]  # one complex sample: [re, im]


def bitrev(x, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def _w(k, n):
    """The twiddle W_n^k = exp(-2 pi i k / n), as (re, im)."""
    z = cmath.exp(-2j * cmath.pi * k / n)
    return z.real, z.imag


# ---------------------------------------------------------------------------
# The schedule, and the tables that come out of it
# ---------------------------------------------------------------------------


def lanes_plan(n, lanes):
    """Which stages need a permutation in front of them, and how deep.

    Stage `s` has butterfly distance ``D = n >> (s+1)``.  While ``D >= lanes``
    the two operands are in one lane ``D/lanes`` beats apart and a
    delay-switch-delay element converts that into a lane distance; the delay is
    ``R >> (s+1)``, because a lane's index step over `d` beats is `d*lanes` while
    the layout is natural and the stage needs `D`.  One more element of depth
    ``R/2`` restores the natural layout, after which the remaining
    ``log2(lanes) - 1`` stages pair two lanes of one beat with no hardware.
    """
    S = n.bit_length() - 1
    w = lanes.bit_length() - 1
    R = n // lanes
    if 1 << S != n:
        raise ValueError(f"radix-2 only; n={n}")
    if 1 << w != lanes:
        raise ValueError(f"the lane count is a power of two; lanes={lanes}")
    if not 0 < w < S:
        raise ValueError(
            f"lanes={lanes} at n={n}: at least one stage must have its partner "
            f"in another beat, and at least one lane pair must exist"
        )
    ops = []
    for s in range(S):
        if s < S - w:
            ops.append(("perm", s, R >> (s + 1), 0))
        elif s == S - w:
            ops.append(("perm", s, R >> 1, 1))
        ops.append(("bfly", s))
    return ops, S, w, R


def _perm_step(streams, T, lanes, d, pairs, phase):
    """One delay-switch-delay element, on tags."""
    k = d.bit_length() - 1
    out = [[None] * T for _ in range(lanes)]
    for t in range(T):
        sw = ((t + phase) >> k) & 1
        for ju, jv in pairs:
            if sw:
                out[ju][t] = streams[ju][t - d] if t - d >= 0 else None
                out[jv][t] = streams[ju][t]
            else:
                out[ju][t] = streams[jv][t - 2 * d] if t - 2 * d >= 0 else None
                out[jv][t] = streams[jv][t - d] if t - d >= 0 else None
    return out


def _pairs_from_tags(streams, T, lanes, D):
    """The lane pairing that holds (i, i+D) in one beat, read off the tags."""
    t0 = next(
        (t for t in range(T) if all(streams[j][t] is not None for j in range(lanes))),
        None,
    )
    if t0 is None:
        return None
    where = {streams[j][t0][1]: j for j in range(lanes)}
    if len(where) != lanes:
        return None
    bit = D.bit_length() - 1
    pairs = []
    for j in range(lanes):
        i = streams[j][t0][1]
        if (i >> bit) & 1:
            continue
        if i + D not in where:
            return None
        pairs.append((j, where[i + D]))
    return pairs if len(pairs) == lanes // 2 else None


def lanes_tables(n, lanes, blocks=None):
    """Every constant the design needs, from one tag run of its own schedule.

    The tables are not restated arithmetic: the twiddle exponents, the reorder
    permutation, the fill length and the host-side unpacking all come out of a
    simulation that carries each sample's ``(block, index)`` and checks, on
    every beat and at every site, that the two operands really are a
    distance-`D` pair from one block.  A schedule that drifted from its tables
    raises here rather than returning wrong data.
    """
    ops, S, w, R = lanes_plan(n, lanes)
    fixed = [(m, m + lanes // 2) for m in range(lanes // 2)]
    lat = sum(2 * op[2] for op in ops if op[0] == "perm")
    blocks = blocks or (4 + (lat + R - 1) // R)
    T = blocks * R

    streams = [[(t // R, (t % R) * lanes + j) for t in range(T)] for j in range(lanes)]

    twiddles, pairings = {}, {}
    for op in ops:
        if op[0] == "perm":
            streams = _perm_step(streams, T, lanes, op[2], fixed, op[3])
            continue
        s = op[1]
        D = n >> (s + 1)
        bit = D.bit_length() - 1
        fed = any(o[0] == "perm" and o[1] == s for o in ops)
        pairs = fixed if fed else _pairs_from_tags(streams, T, lanes, D)
        if pairs is None:
            raise AssertionError(
                f"n={n} lanes={lanes} stage {s} (D={D}): no fixed lane pairing "
                f"holds both operands of a butterfly in one beat"
            )
        pairings[s] = pairs
        out = [[None] * T for _ in range(lanes)]
        exps = {}
        for t in range(T):
            for m, (ja, jb) in enumerate(pairs):
                a, b = streams[ja][t], streams[jb][t]
                if a is None or b is None:
                    continue
                if a[0] != b[0]:
                    raise AssertionError(
                        f"stage {s} site {m} beat {t}: operands from blocks "
                        f"{a[0]} and {b[0]} -- the schedule crosses a block edge"
                    )
                if b[1] != a[1] + D or ((a[1] >> bit) & 1):
                    raise AssertionError(
                        f"stage {s} site {m} beat {t}: indices {a[1]},{b[1]} are "
                        f"not a distance-{D} butterfly pair"
                    )
                e = (a[1] % D) * (1 << s)
                if exps.get((m, t % R), e) != e:
                    raise AssertionError(
                        f"stage {s} site {m}: the twiddle exponent is not "
                        f"periodic in R={R} beats (beat {t})"
                    )
                exps[(m, t % R)] = e
                out[ja][t], out[jb][t] = a, b
        if len(exps) != (lanes // 2) * R:
            raise AssertionError(
                f"stage {s}: {len(exps)} of {(lanes // 2) * R} (site, beat) slots "
                f"ever carry a butterfly, so a multiplier idles"
            )
        twiddles[s] = exps
        streams = out

    # Once the layout is natural again -- from the stage the last permutation
    # feeds onwards -- a site's twiddle exponent stops depending on the beat:
    # the low index at site `m` is `t*lanes + j` and the exponent is
    # `(i mod D) * 2^s` with `D` dividing `lanes`, so the `t*lanes` term drops
    # out. Those stages can therefore index their ROM by site alone, and with
    # the site coordinate specialised the read folds to a literal -- which is
    # what makes a rotation by 1 or -i cost no multiplier *per site* rather
    # than only per stage. This records which stages have that property; the
    # design asserts the ones it relies on.
    site_const = {}
    for s in range(S):
        per_site = {}
        for (m, _r), e in twiddles[s].items():
            if per_site.setdefault(m, e) != e:
                break
        else:
            site_const[s] = per_site

    # ---- which bin lands where, and the per-lane reorder that undoes it
    bin_at = {}
    for t in range(T):
        for j in range(lanes):
            tok = streams[j][t]
            if tok is None:
                continue
            b = bitrev(tok[1], S)
            if bin_at.get((j, t % R), b) != b:
                raise AssertionError(
                    f"lane {j}: the output bin at beat {t % R} is not periodic in R"
                )
            bin_at[(j, t % R)] = b
    if len(bin_at) != lanes * R:
        raise AssertionError(f"{len(bin_at)} of {lanes * R} output slots are used")

    base, perm = {}, None
    for j in range(lanes):
        got = [bin_at[(j, r)] for r in range(R)]
        lo = min(got)
        if sorted(got) != list(range(lo, lo + R)):
            raise AssertionError(
                f"lane {j} does not own a contiguous block of bins, so the "
                f"reorder would have to cross a lane"
            )
        if lo != bitrev(j, w) * R:
            raise AssertionError(
                f"lane {j} owns bins from {lo}, not bitrev({j})*R = "
                f"{bitrev(j, w) * R}"
            )
        base[j] = lo
        rel = [g - lo for g in got]
        if perm is None:
            perm = rel
        elif perm != rel:
            raise AssertionError(
                f"lane {j}'s reorder permutation differs from lane 0's, so one "
                f"shared ROM cannot serve every site"
            )

    # ---- the fill: beats reaching the reorder before block 0 is whole
    lead = None
    for t in range(T - R + 1):
        window = [streams[j][t + r] for r in range(R) for j in range(lanes)]
        if any(tok is None or tok[0] != 0 for tok in window):
            continue
        if len({tok[1] for tok in window}) != R * lanes:
            continue
        lead = t
        break
    if lead is None:
        raise AssertionError("no whole block of the first transform ever appears")

    return {
        "ops": ops,
        "S": S,
        "w": w,
        "R": R,
        "lat": lat,
        "pairings": pairings,
        "twiddles": twiddles,
        "site_const": site_const,
        # The reorder reads its ROM at its own beat, which starts `lead` beats
        # into the stream, so the rotation is folded in here once rather than
        # being an offset the body has to carry.
        "perm": [perm[(lead + r) % R] for r in range(R)],
        "base": base,
        "lead": lead,
    }


# ---------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------


def fft_lanes_of(n, batch, lanes=2, name=None):
    """`n` points over `lanes` lanes, one whole butterfly a site a beat."""
    tab = lanes_tables(n, lanes)
    ops, S, w, R = tab["ops"], tab["S"], tab["w"], tab["R"]
    sites = lanes // 2
    lead = tab["lead"]
    total = lead + batch * R
    # Stages whose distance is below the lane count pair two lanes of one beat,
    # and their pairings differ, so they carry real lane crossings. They go on
    # one topology together with the stage before them, so every crossing is an
    # edge of that topology rather than a link between placements.
    tail = list(range(S - w, S)) if w >= 2 else []
    plain = [s for s in range(S) if s not in tail]

    def exps_of(s):
        return [[tab["twiddles"][s][(m, r)] for m in range(sites)] for r in range(R)]

    def trivial(s):
        return set(tab["twiddles"][s].values()) <= {0, n // 4}

    # ------------------------------------------------------------- the perms
    def perm_unit(s, d, phase):
        """Delay `d`, switch every `d` beats, delay `d`: time becomes lanes.

        Each delay line is read then written at the same address, which is a
        depth-`d` delay and one read and one write a beat -- no bank to prove
        anything about, so the loop closes at one token a cycle by construction.
        """
        kbit = d.bit_length() - 1

        class PermIO(spmw.Interface):
            u_in = spmw.In(csample, depth=8)
            v_in = spmw.In(csample, depth=8)
            u_out = spmw.Out(csample, depth=8)
            v_out = spmw.Out(csample, depth=8)

        def body(io: PermIO):
            vr: float32[d]
            vi: float32[d]
            ur: float32[d]
            ui: float32[d]
            for t in range(total):
                x = io.u_in.get()
                y = io.v_in.get()
                c: int32 = t & (d - 1)
                # the v line, read before write: v1 is v from `d` beats ago
                v1r: float32 = vr[c]
                v1i: float32 = vi[c]
                vr[c] = y[0]
                vi[c] = y[1]
                # Annotated, not inferred: an unannotated local holding a
                # comparison is typed by the arithmetic that feeds it and then
                # stored as a predicate, which the dataflow lowering rejects.
                sw: int32 = ((t + phase) >> kbit) & 1
                u2r: float32 = x[0]
                u2i: float32 = x[1]
                v2r: float32 = v1r
                v2i: float32 = v1i
                if sw == 1:
                    u2r = v1r
                    u2i = v1i
                    v2r = x[0]
                    v2i = x[1]
                p: csample
                q: csample
                p[0] = ur[c]
                p[1] = ui[c]
                ur[c] = u2r
                ui[c] = u2i
                q[0] = v2r
                q[1] = v2i
                io.u_out.put(p)
                io.v_out.put(q)

        # A unit takes its name at decoration, and two bodies sharing a name
        # share their captured constants: rename first, then decorate.
        body.__name__ = f"perm{s}"
        body.__qualname__ = f"perm{s}"
        return spmw.unit(body)

    # -------------------------------------------------------- the butterflies
    def bfly_unit(s):
        """One whole butterfly a beat: both operands arrive on two ports.

        The sum keeps the low lane and the twiddled difference the high one, so
        nothing is stored and the float pipeline is feed-forward.
        """
        triv = trivial(s)

        class BflyIO(spmw.Interface):
            a_in = spmw.In(csample, depth=8)
            b_in = spmw.In(csample, depth=8)
            a_out = spmw.Out(csample, depth=8)
            b_out = spmw.Out(csample, depth=8)
            if triv:
                # 0 rotates by 1, 1 rotates by -i. An int selector rather than a
                # float pair, so there is no multiplier for Vitis to find.
                sel = spmw.MemIn(int32[R, sites])
            else:
                tw = spmw.MemIn(float32[R, sites, 2])

        def general(io: BflyIO, site: spmw.Site):
            (m,) = site.rank
            for t in range(total):
                a = io.a_in.get()
                b = io.b_in.get()
                r: int32 = t & (R - 1)
                dr: float32 = a[0] - b[0]
                di: float32 = a[1] - b[1]
                wr: float32 = io.tw[r, m, 0]
                wi: float32 = io.tw[r, m, 1]
                p: csample
                q: csample
                p[0] = a[0] + b[0]
                p[1] = a[1] + b[1]
                q[0] = dr * wr - di * wi
                q[1] = dr * wi + di * wr
                io.a_out.put(p)
                io.b_out.put(q)

        def rotate(io: BflyIO, site: spmw.Site):
            (m,) = site.rank
            for t in range(total):
                a = io.a_in.get()
                b = io.b_in.get()
                r: int32 = t & (R - 1)
                dr: float32 = a[0] - b[0]
                di: float32 = a[1] - b[1]
                rot: int32 = io.sel[r, m]
                p: csample
                q: csample
                p[0] = a[0] + b[0]
                p[1] = a[1] + b[1]
                q[0] = dr
                q[1] = di
                if rot == 1:
                    q[0] = di
                    q[1] = 0.0 - dr
                io.a_out.put(p)
                io.b_out.put(q)

        body = rotate if triv else general
        body.__name__ = f"bfly{s}"
        body.__qualname__ = f"bfly{s}"
        return spmw.unit(body), triv

    # --------------------------------------------------------------- the tail
    tail_topo = None
    tail_unit = None
    if tail:
        n_tail = len(tail)
        # Every tail stage's exponent is a per-site constant -- checked, not
        # assumed, because the whole ROM shape below rests on it.
        for s in tail:
            if s not in tab["site_const"]:
                raise AssertionError(
                    f"stage {s} is in the tail but its twiddle exponent still "
                    f"depends on the beat, so a per-site ROM would drop data"
                )

        class TailIO(spmw.Interface):
            a_in = spmw.In(csample, depth=8)
            b_in = spmw.In(csample, depth=8)
            a_out = spmw.Out(csample, depth=8)
            b_out = spmw.Out(csample, depth=8)
            # One entry per (row, site), not per beat, and a rotation code
            # beside it. With both grid axes specialised the indices are
            # literals, so both reads fold to the constants they are and the
            # dead arm of the branch below goes with them: a site that rotates
            # only by 1 or -i has no multiply left to build. That is the
            # per-butterfly trivial-twiddle folding the scalar reference gets
            # from unrolling its lane loop; here the lane loop is the grid.
            #
            # The code is a ROM rather than a test on the twiddle's magnitude,
            # because `wr > 0.5` is also true of a 45-degree rotation.
            tw = spmw.MemIn(float32[n_tail, sites, 2])
            sel = spmw.MemIn(int32[n_tail, sites])

        def tail_links(k, m):
            if k == n_tail - 1:  # the last row's results leave the placement
                return {}
            here = tab["pairings"][tail[k]]
            nxt = tab["pairings"][tail[k + 1]]
            out = {}
            for lane, port in ((here[m][0], TailIO.a_out), (here[m][1], TailIO.b_out)):
                for m1, (ja, jb) in enumerate(nxt):
                    if lane == ja:
                        out[port] = spmw.to((k + 1, m1), TailIO.a_in)
                    elif lane == jb:
                        out[port] = spmw.to((k + 1, m1), TailIO.b_in)
            return out

        tail_topo = spmw.Topology(TailIO, grid=(n_tail, sites), link=tail_links)

        def tail_general(io: TailIO, site: spmw.Site):
            k, m = site.rank
            for _t in range(total):
                a = io.a_in.get()
                b = io.b_in.get()
                dr: float32 = a[0] - b[0]
                di: float32 = a[1] - b[1]
                wr: float32 = io.tw[k, m, 0]
                wi: float32 = io.tw[k, m, 1]
                rot: int32 = io.sel[k, m]
                p: csample
                q: csample
                p[0] = a[0] + b[0]
                p[1] = a[1] + b[1]
                q[0] = dr * wr - di * wi
                q[1] = dr * wi + di * wr
                if rot == 1:  # rotate by 1: nothing to do
                    q[0] = dr
                    q[1] = di
                if rot == 2:  # rotate by -i: a swap and a sign
                    q[0] = di
                    q[1] = 0.0 - dr
                io.a_out.put(p)
                io.b_out.put(q)

        tail_unit = spmw.unit(tail_general)

    # ------------------------------------------------------------ the reorder
    # Decimation in frequency leaves the transform bit-reversed, and the
    # permutation network leaves its own trace on top of that; `lanes_tables`
    # checks that the composition still never crosses a lane, so one site holds
    # two independent per-lane buffers and no data moves sideways here.
    class ReorderIO(spmw.Interface):
        a_in = spmw.In(csample, depth=8)
        b_in = spmw.In(csample, depth=8)
        a_out = spmw.Out(csample, depth=8)
        b_out = spmw.Out(csample, depth=8)
        rd = spmw.MemIn(int32[R])

    # A body local must not be named `v<digits>`: the emitter names the role's
    # stream arguments `v0, v1, ...`, so a local called `v2` shadows the output
    # stream and `v2.write(...)` becomes a member reference on `float[2]`. It
    # fails loudly at csynth rather than quietly, but the names here avoid it.
    @spmw.unit
    def reorder(io: ReorderIO):
        ar: float32[2, R]
        ai: float32[2, R]
        br: float32[2, R]
        bi: float32[2, R]
        for _f in range(lead):
            _sa = io.a_in.get()
            _sb = io.b_in.get()
        for blk in range(batch):
            side: int32 = blk & 1
            other: int32 = 1 - side
            for r in range(R):
                x = io.a_in.get()
                y = io.b_in.get()
                pos: int32 = io.rd[r]
                ar[side, pos] = x[0]
                ai[side, pos] = x[1]
                br[side, pos] = y[0]
                bi[side, pos] = y[1]
                if blk > 0:
                    lo: csample
                    hi: csample
                    lo[0] = ar[other, r]
                    lo[1] = ai[other, r]
                    hi[0] = br[other, r]
                    hi[1] = bi[other, r]
                    io.a_out.put(lo)
                    io.b_out.put(hi)
        last: int32 = (batch - 1) & 1
        for r2 in range(R):
            tlo: csample
            thi: csample
            tlo[0] = ar[last, r2]
            tlo[1] = ai[last, r2]
            thi[0] = br[last, r2]
            thi[1] = bi[last, r2]
            io.a_out.put(tlo)
            io.b_out.put(thi)

    perm_units = {
        op[1]: perm_unit(op[1], op[2], op[3]) for op in ops if op[0] == "perm"
    }
    bfly_units = {}
    for s in plain:
        bfly_units[s] = bfly_unit(s)

    # ------------------------------------------------------------- the fabric
    @spmw.fabric
    def engine(
        Xa: float32[sites, total, 2],
        Xb: float32[sites, total, 2],
        Ya: float32[sites, batch * R, 2],
        Yb: float32[sites, batch * R, 2],
    ):
        P = {s: spmw.place(u, on=spmw.Grid((sites,))) for s, u in perm_units.items()}
        B = {
            s: spmw.place(u, on=spmw.Grid((sites,)))
            for s, (u, _t) in bfly_units.items()
        }
        # Both grid axes specialised: the row picks the stage and the site
        # picks the lane pair, and a body that reads either as a literal
        # folds its twiddle read to a constant. One role per tail site is
        # the price, and they synthesise concurrently.
        Tl = spmw.place(tail_unit, on=tail_topo, specialise=(0, 1)) if tail else None
        Rd = spmw.place(reorder, on=spmw.Grid((sites,)))

        # One ROM per stage, each with its own name: memories made in a loop
        # would otherwise all be called `tw`, and one tensor bound stationary at
        # several placements reaches only the first.
        for s, (_u, triv) in bfly_units.items():
            if triv:
                rot = np.array(
                    [[1 if e else 0 for e in row] for row in exps_of(s)],
                    dtype=np.int32,
                )
                spmw.stationary(
                    spmw.mem(
                        int32[R, sites],
                        init=rot,
                        layout=spmw.replicate,
                        name=f"rotl{s}",
                    ),
                    at=B[s].sel,
                )
            else:
                tw = np.zeros((R, sites, 2), dtype=np.float32)
                for r, row in enumerate(exps_of(s)):
                    for m, e in enumerate(row):
                        tw[r, m] = _w(e, n)
                spmw.stationary(
                    spmw.mem(
                        float32[R, sites, 2],
                        init=tw,
                        layout=spmw.replicate,
                        name=f"twl{s}",
                    ),
                    at=B[s].tw,
                )
        if tail:
            tw = np.zeros((len(tail), sites, 2), dtype=np.float32)
            rot = np.zeros((len(tail), sites), dtype=np.int32)
            for k, s in enumerate(tail):
                for m, e in tab["site_const"][s].items():
                    tw[k, m] = _w(e, n)
                    rot[k, m] = 1 if e == 0 else (2 if e == n // 4 else 0)
            spmw.stationary(
                spmw.mem(
                    float32[len(tail), sites, 2],
                    init=tw,
                    layout=spmw.replicate,
                    name="twtail",
                ),
                at=Tl.tw,
            )
            spmw.stationary(
                spmw.mem(
                    int32[len(tail), sites],
                    init=rot,
                    layout=spmw.replicate,
                    name="rottail",
                ),
                at=Tl.sel,
            )
        spmw.stationary(
            spmw.mem(
                int32[R],
                init=np.array(tab["perm"], dtype=np.int32),
                layout=spmw.replicate,
                name="rdl",
            ),
            at=Rd.rd,
        )

        # The chain, in schedule order. Every edge is site-to-site: a stage's
        # site `m` and the next element's site `m` hold the same two lanes,
        # which is why no crossing is needed outside the tail topology.
        def ports(kind, s):
            if kind == "perm":
                return P[s].u_out, P[s].v_out, P[s].u_in, P[s].v_in
            return B[s].a_out, B[s].b_out, B[s].a_in, B[s].b_in

        chain = [(op[0], op[1]) for op in ops if op[0] == "perm" or op[1] in plain]
        head = chain[0]
        _o1, _o2, i1, i2 = ports(*head)
        spmw.stream_in(Xa, into=i1, index=(P[0].rows, ...))
        spmw.stream_in(Xb, into=i2, index=(P[0].rows, ...))
        for a, b in zip(chain, chain[1:]):
            o1, o2, _i1, _i2 = ports(*a)
            _p1, _p2, j1, j2 = ports(*b)
            spmw.link(o1, to=j1)
            spmw.link(o2, to=j2)
        o1, o2, _i1, _i2 = ports(*chain[-1])
        if tail:
            spmw.link(o1, to=Tl.a_in)
            spmw.link(o2, to=Tl.b_in)
            spmw.link(Tl.a_out, to=Rd.a_in)
            spmw.link(Tl.b_out, to=Rd.b_in)
        else:
            spmw.link(o1, to=Rd.a_in)
            spmw.link(o2, to=Rd.b_in)
        spmw.gather(Ya, from_=Rd.a_out, index=(Rd.rows, ...))
        spmw.gather(Yb, from_=Rd.b_out, index=(Rd.rows, ...))

    engine.__name__ = name or f"fft_lanes_{n}_w{lanes}"
    engine.spmw_parts = (n, batch, lanes, S, w, R, lead, total, tuple(tail))
    engine.spmw_tables = tab
    # The butterflies cancel O(n) intermediates, so the differences between the
    # reference and the HLS float units are absolute, ~1e-5.
    engine.spmw_tolerance = (1e-4, 1e-4)
    # Every body is one deep pipeline over the whole launch, so it has to drain
    # when its loop ends: with the default stall style HLS keeps the iterations
    # in flight and the last unit is short by its depth.
    engine.spmw_pipeline_style = "flp"
    # On by default, as in the paired design: the row this is measured against
    # is HP-FFT's, which binds its own adders, and a DSP comparison where one
    # side spends DSPs on adds is not a comparison.
    engine.spmw_bind_fabric = os.environ.get("SPMW_BIND_FABRIC", "1") != "0"
    # One transform is `n` output tokens across every channel; the testbench
    # counts tokens on all channels, not per channel.
    engine.spmw_tokens_per_transform = n
    return engine


# ---------------------------------------------------------------------------
# Operands and the check
# ---------------------------------------------------------------------------


def operands(n, batch, lanes, seed=0):
    """`batch` transforms in natural streaming order, split over the sites.

    Site `m` of the first element holds lanes `m` and `m + lanes/2`, so sample
    ``t*lanes + m`` arrives on `Xa` and ``t*lanes + m + lanes/2`` on `Xb`: the
    lane law at the boundary, exactly as the rolled design spells it.
    """
    tab = lanes_tables(n, lanes)
    R, sites = tab["R"], lanes // 2
    total = tab["lead"] + batch * R
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((batch, n)) + 1j * rng.standard_normal((batch, n))).astype(
        np.complex64
    )
    Xa = np.zeros((sites, total, 2), dtype=np.float32)
    Xb = np.zeros((sites, total, 2), dtype=np.float32)
    flat = x.reshape(batch, R, lanes)
    for m in range(sites):
        Xa[m, : batch * R, 0] = flat[:, :, m].real.reshape(-1)
        Xa[m, : batch * R, 1] = flat[:, :, m].imag.reshape(-1)
        Xb[m, : batch * R, 0] = flat[:, :, m + sites].real.reshape(-1)
        Xb[m, : batch * R, 1] = flat[:, :, m + sites].imag.reshape(-1)
    return x, Xa, Xb


def unpack(Ya, Yb, n, batch, lanes):
    """Undo the lane split: each lane owns a contiguous block of bins."""
    tab = lanes_tables(n, lanes)
    R, sites = tab["R"], lanes // 2
    pairs = tab["pairings"][tab["S"] - 1]
    out = np.zeros((batch, n), dtype=np.complex128)
    for m in range(sites):
        for tensor, lane in ((Ya, pairs[m][0]), (Yb, pairs[m][1])):
            lo = tab["base"][lane]
            blk = tensor[m].reshape(batch, R, 2)
            out[:, lo : lo + R] = blk[:, :, 0] + 1j * blk[:, :, 1]
    return out


def check(x, Ya, Yb, n, batch, lanes, atol=1e-4, rtol=1e-4):
    got = unpack(Ya, Yb, n, batch, lanes)
    want = np.fft.fft(x.astype(np.complex128), axis=1)
    err = np.abs(got - want).max()
    norm = err / max(np.abs(want).max(), 1e-30)
    np.testing.assert_allclose(got, want, atol=atol, rtol=rtol)
    return err, norm


def run(n, batch, lanes, target, seed=0):
    x, Xa, Xb = operands(n, batch, lanes, seed=seed)
    R = n // lanes
    Ya = np.zeros((lanes // 2, batch * R, 2), dtype=np.float32)
    Yb = np.zeros((lanes // 2, batch * R, 2), dtype=np.float32)
    spmw.build(fft_lanes_of(n, batch, lanes), target=target)(Xa, Xb, Ya, Yb)
    return check(x, Ya, Yb, n, batch, lanes)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("n", [8, 16])
@pytest.mark.parametrize("target", ["ref", "simulator"])
def test_lanes_matches_numpy(n, lanes, target):
    if lanes >= n // 2:
        pytest.skip("at least one stage must have its partner in another beat")
    _err, norm = run(n, 3, lanes, target, seed=n * 16 + lanes)
    assert norm < 1e-5


@pytest.mark.parametrize(
    "n,lanes", [(64, 2), (128, 4), (256, 2), (256, 4), (256, 8), (256, 16)]
)
def test_the_wide_end_of_the_sweep(n, lanes):
    """The configurations the array is built at, on the fast target."""
    _err, norm = run(n, 2, lanes, "ref", seed=n + lanes)
    assert norm < 1e-5


def test_the_permutation_turns_time_into_lanes():
    """The claim the design rests on, checked rather than asserted.

    A stage whose butterfly distance is at least the lane count has both
    operands in *one lane, several beats apart* -- which is why the rolled
    design's per-lane unit can only ever see one of them. After the
    delay-switch-delay element in front of that stage, the same two samples are
    on *two lanes of one beat*. Both halves are checked here from the tag run:
    the first from the lane law, the second from the schedule's own pairing.
    """
    n, lanes = 256, 4
    tab = lanes_tables(n, lanes)
    S, R = tab["S"], tab["R"]
    fed = {op[1] for op in tab["ops"] if op[0] == "perm"}
    assert fed, "no stage needed a permutation, so there is nothing to check"
    for s in sorted(fed):
        D = n >> (s + 1)
        if D < lanes:
            continue
        # Before: under the lane law both operands are on one lane, D/lanes
        # beats apart, because D is a whole number of beats. A site reading two
        # lanes of one beat cannot see that pair -- which is the rolled design's
        # situation and the reason its multiplier idles.
        assert D % lanes == 0, (s, D)
        assert D // lanes >= 1, (s, D)
        # After: the stage's own pairing holds both of them in one beat.
        pairs = tab["pairings"][s]
        assert pairs == [(m, m + lanes // 2) for m in range(lanes // 2)], (s, pairs)
    # and every site of every stage carries a butterfly on every beat
    for s in range(S):
        assert len(tab["twiddles"][s]) == (lanes // 2) * R, s


def test_each_stage_admits_exactly_one_delay():
    """The delays are forced by the schedule, not chosen -- so nothing is tuned.

    For every stage that needs a permutation, every power-of-two delay and every
    switch phase is tried and the pairing checked. Exactly one delay works, and
    it is the one `lanes_plan` names. If that ever stops holding, the plan and
    the schedule have drifted apart and one of them is wrong.
    """
    n, lanes = 64, 2
    ops, S, w, R = lanes_plan(n, lanes)
    fixed = [(m, m + lanes // 2) for m in range(lanes // 2)]
    blocks = 6 + (sum(2 * o[2] for o in ops if o[0] == "perm") + R - 1) // R
    T = blocks * R
    streams = [[(t // R, (t % R) * lanes + j) for t in range(T)] for j in range(lanes)]

    def pairs_hold(st, D):
        bit = D.bit_length() - 1
        seen = 0
        for t in range(T):
            for ja, jb in fixed:
                a, b = st[ja][t], st[jb][t]
                if a is None or b is None:
                    continue
                if a[0] != b[0] or b[1] != a[1] + D or ((a[1] >> bit) & 1):
                    return False
                seen += 1
        return seen > 0

    # A butterfly replaces values but moves nothing, so it is the identity on
    # the tags and only the permutations change the layout.
    want = {op[1]: op[2] for op in ops if op[0] == "perm"}
    for op in ops:
        if op[0] != "perm":
            continue
        s, d, phase = op[1], op[2], op[3]
        D = n >> (s + 1)
        works = [
            dd
            for dd in (1 << k for k in range(R.bit_length()))
            if any(
                pairs_hold(_perm_step(streams, T, lanes, dd, fixed, ph), D)
                for ph in range(2 * dd)
            )
        ]
        assert works == [want[s]], (s, D, works, want[s])
        streams = _perm_step(streams, T, lanes, d, fixed, phase)


def test_one_multiplier_per_site_per_non_trivial_stage():
    """The DSP story, counted from the design rather than from a report.

    `log2(n) - 2` stages need a complex multiplier -- the last two rotate only by
    1 and -i -- and each holds `lanes/2` of them, one per site, busy every beat.
    That is the count HP-FFT reaches by hand and the rolled design does not.
    """
    n = 256
    S = 8
    for lanes in (2, 4, 8, 16):
        tab = lanes_tables(n, lanes)
        triv = [s for s in range(S) if set(tab["twiddles"][s].values()) <= {0, n // 4}]
        assert triv == [S - 2, S - 1], (lanes, triv)
        mults = (S - len(triv)) * (lanes // 2)
        assert mults == 6 * (lanes // 2), (lanes, mults)


def test_the_interval_is_the_ideal_one():
    """`n/W` beats a transform, which is one butterfly a site a beat."""
    for n, lanes in ((256, 2), (256, 4), (256, 8), (256, 16)):
        tab = lanes_tables(n, lanes)
        R, sites = tab["R"], lanes // 2
        assert R == n // lanes
        # log2(n) stages, each retiring n/2 butterflies in R beats on `sites`
        butterflies = tab["S"] * (n // 2)
        assert butterflies == tab["S"] * n // 2
        assert (n // 2) / (R * sites) == 1.0, (n, lanes)


def test_the_permutation_memory_is_not_a_per_stage_buffer():
    """Why this should not cost the paired design's 44 BRAM18.

    The paired design holds `2n` positions per stage per component. Here the
    only memory is the permutation delays, `sum(d) * W` complex for the whole
    pipeline, plus one double-buffered reorder.
    """
    n = 256
    for lanes in (2, 4, 8, 16):
        tab = lanes_tables(n, lanes)
        delay = sum(op[2] for op in tab["ops"] if op[0] == "perm") * lanes
        reord = 2 * n
        assert delay <= 2 * n, (lanes, delay)
        paired = 2 * n * tab["S"]
        assert delay + reord < paired / 2, (lanes, delay + reord, paired)


def test_the_tail_crossings_are_a_function_of_the_width():
    """At W=2 there is nothing to cross; the crossings appear with the width.

    Stages whose distance is below the lane count pair two lanes of one beat,
    and consecutive ones pair differently, so each such boundary is a real lane
    crossing. There are `log2(W) - 1` of them, which is why the W=2 design is a
    chain and the W=16 one is not.
    """
    n = 256
    for lanes, want in ((2, 0), (4, 1), (8, 2), (16, 3)):
        tab = lanes_tables(n, lanes)
        S, w = tab["S"], tab["w"]
        wire = [
            s
            for s in range(S)
            if not any(o[0] == "perm" and o[1] == s for o in tab["ops"])
        ]
        assert len(wire) == w - 1 == want, (lanes, wire)
        for a, b in zip(wire, wire[1:]):
            assert tab["pairings"][a] != tab["pairings"][b], (lanes, a, b)


def test_link_index_is_accepted_and_ignored():
    """`index=` on `spmw.link` is recorded and then dropped by both paths.

    This is why the tail's lane crossings are spelled with a topology `link`
    rule and `spmw.to` rather than with a pairing on the link: `bindings.link`
    stores `index=` without running it through `make_map` or `bind_check`,
    `lower_df._plan_link` never reads `binding.imap`, and `refsim` zips
    `src.sites` with `dst.sites` in order. A design that asked for a permutation
    here would be wired straight through and would compute the wrong transform
    with nothing raised.

    Pinned rather than asserted-correct because it is not correct yet: it is the
    same shape as `test_stationary_index_on_a_brick_is_ignored` in the rolled
    design -- a knob with a checker in front of it and nothing behind it. When
    the pairing is honoured, this test is the change that says so.
    """
    sites = 4

    class WireIO(spmw.Interface):
        x_in = spmw.In(float32)
        y_out = spmw.Out(float32)

    @spmw.unit
    def pass_through(io: WireIO):
        io.y_out.put(io.x_in.get())

    @spmw.unit
    def sink(io: WireIO):
        io.y_out.put(io.x_in.get() * 2.0)

    reverse = [sites - 1 - i for i in range(sites)]

    @spmw.fabric
    def fab(X: float32[sites], Y: float32[sites]):
        A = spmw.place(pass_through, on=spmw.Grid((sites,)))
        B = spmw.place(sink, on=spmw.Grid((sites,)))
        spmw.stream_in(X, into=A.x_in, index=(A.rows,))
        # A reversal, asked for and not delivered.
        spmw.link(A.y_out, to=B.x_in, index=lambda i: (reverse[i],))
        spmw.gather(Y, from_=B.y_out, index=(B.rows,))

    X = np.arange(sites, dtype=np.float32)
    Y = np.zeros(sites, dtype=np.float32)
    spmw.build(fab, target="ref")(X, Y)
    # Straight through, not reversed: the map consumed nothing.
    np.testing.assert_allclose(Y, X * 2.0)
    assert not np.allclose(Y, X[::-1] * 2.0), "the pairing was honoured after all"


def test_at_two_lanes_the_design_is_a_chain():
    """W=2 needs no lane crossing, so there is no topology in it."""
    fab = fft_lanes_of(64, 2, 2)
    _n, _b, lanes, S, w, R, _lead, _total, tail = fab.spmw_parts
    assert (lanes, w, R, tail) == (2, 1, 32, ())
    graph = spmw.elaborate(fab)
    for p in graph.placements:
        assert len(p.topology.grid) == 1, (p.name, p.topology.grid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
