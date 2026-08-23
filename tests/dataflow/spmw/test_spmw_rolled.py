# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the rolling holds, and where it is lost.

The design's load-bearing claim is that the number of *bodies* a spatial design
compiles to tracks its role count, not its grid: a 2-D mesh has nine site
signatures at any size, so it should emit nine bodies at any size.

The frontend delivers that, and the first two tests pin it.  The third records
where it is currently lost -- the dataflow builder expands one kernel instance
per grid point, so HLS sees one function per site.  Closing that gap is what the
rolled path is for, and this test is the number it has to move.
"""

import re

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import float32

SIZES = [2, 3, 4, 6]


def gemm_of(size):
    """A systolic GEMM at a chosen size; the grid is a parameter, not structure."""

    class IO(spmw.Interface):
        west = spmw.In(float32)
        north = spmw.In(float32)
        east = spmw.Out(float32)
        south = spmw.Out(float32)
        c = spmw.MemOut(float32)

    @spmw.unit
    def pe(io: IO):
        acc: float32 = 0
        for k in range(size):
            a = io.west.get()
            b = io.north.get()
            acc += a * b
            io.east.put(a)
            io.south.put(b)
        io.c = acc

    @spmw.fabric
    def g(A: float32[size, size], B: float32[size, size], C: float32[size, size]):
        P = spmw.place(pe, on=spmw.mesh(IO, (size, size)))
        spmw.stream_in(A, into=P.west, index=(P.rows, ...))
        spmw.stream_in(B, into=P.north, index=(..., P.cols))
        spmw.gather(C, from_=P.c)

    return g


@pytest.mark.parametrize("size", SIZES)
def test_signature_count_goes_flat(size):
    """Interior, four edges, four corners -- nine, once the mesh is big enough."""
    graph = spmw.elaborate(gemm_of(size))
    expected = 9 if size >= 3 else 4
    assert len(graph.placements[0].topology.signatures()) == expected


@pytest.mark.parametrize("size", SIZES)
def test_emitted_arms_track_signatures_not_sites(size):
    """One arm per signature class in the emitted program, whatever the grid."""
    text = spmw.source(gemm_of(size))
    arms = text.count("meta_if") + text.count("meta_elif") + text.count("meta_else")
    expected = 9 if size >= 3 else 4
    assert arms == expected, f"{size}x{size}: {arms} arms for {expected} signatures"


@pytest.mark.parametrize("size", SIZES)
def test_hls_body_count_is_the_gap(size):
    """The count HLS actually sees, which is still one body per site.

    Pinned rather than asserted-flat because it is not flat yet: the dataflow
    builder expands `mapping=[R, C]` into one kernel instance per grid point.
    When the rolled path lands this becomes an assertion that the count does not
    grow with the grid, and the change in this test is the evidence.
    """
    mod = spmw.build(gemm_of(size), target="vhls")
    bodies = len(re.findall(r"^void\s+\w+\(", mod.hls_code, re.M))
    # sites + one loader per edge column + one per edge row + the top.
    assert bodies == size * size + 2 * size + 1


def test_the_gap_is_quadratic_while_the_frontend_is_flat():
    """State the two curves side by side, since that contrast is the whole point."""
    arms, bodies = {}, {}
    for size in SIZES:
        fab = gemm_of(size)
        text = spmw.source(fab)
        arms[size] = (
            text.count("meta_if") + text.count("meta_elif") + text.count("meta_else")
        )
        mod = spmw.build(fab, target="vhls")
        bodies[size] = len(re.findall(r"^void\s+\w+\(", mod.hls_code, re.M))

    big = [s for s in SIZES if s >= 3]
    assert len({arms[s] for s in big}) == 1, f"arms should be flat, got {arms}"
    assert (
        bodies[max(big)] > bodies[min(big)]
    ), f"bodies should still grow, got {bodies}"


# --------------------------------------------------------------------------
# The rolled form, where the count actually goes flat
# --------------------------------------------------------------------------

ROLLED_SIZES = [3, 4, 6, 8, 16]


@pytest.mark.parametrize("size", ROLLED_SIZES)
def test_the_rolled_form_verifies(size):
    """What the frontend computes is a shape the dialect accepts."""
    from allo._mlir.dialects import allo as allo_d
    from allo._mlir.ir import Context, Module
    from allo.spmw.lower_mlir import render_module

    with Context() as ctx:
        allo_d.register_dialect(ctx)
        Module.parse(render_module(spmw.elaborate(gemm_of(size))))


def test_the_rolled_body_count_does_not_grow():
    """Nine roles at nine sites and at two hundred and fifty-six.

    This is the design's load-bearing claim, and the contrast with
    `test_hls_body_count_is_the_gap` above is the whole point: the same designs
    expand to one body per site through the dataflow path and stay flat here.
    """
    from allo._mlir.dialects import allo as allo_d
    from allo._mlir.ir import Context, Module
    from allo.spmw.lower_df import _wiring_classes
    from allo.spmw.lower_mlir import RolledEmitter, render_module

    roles, funcs = {}, {}
    with Context() as ctx:
        allo_d.register_dialect(ctx)
        for size in ROLLED_SIZES:
            graph = spmw.elaborate(gemm_of(size))
            emitter = RolledEmitter(graph)
            placement = emitter.placements()[0]
            roles[size] = len(
                _wiring_classes(placement, emitter.low.resolutions[placement])
            )
            text = render_module(graph)
            Module.parse(text)
            funcs[size] = text.count("func.func @")

    assert set(roles.values()) == {9}, f"roles should be flat at 9, got {roles}"
    assert len(set(funcs.values())) == 1, f"functions should be flat, got {funcs}"
    # The largest grid here has 256 sites; the dataflow path would emit 289.
    assert max(ROLLED_SIZES) ** 2 // roles[max(ROLLED_SIZES)] > 25


def test_the_rolled_form_emits_flat_hls():
    """The number that the whole exercise is about.

    Same designs, same nine roles, and now the HLS function count does not grow
    with the array: ten functions whether the mesh has nine sites or two hundred
    and fifty-six. Compare `test_hls_body_count_is_the_gap`, which measures the
    dataflow path emitting one body per site for exactly these fabrics.
    """
    import io

    from allo._mlir.dialects import allo as allo_d
    from allo._mlir.ir import Context, Module
    from allo.spmw.lower_mlir import render_module

    counts = {}
    with Context() as ctx:
        allo_d.register_dialect(ctx)
        for size in ROLLED_SIZES:
            module = Module.parse(render_module(spmw.elaborate(gemm_of(size))))
            buf = io.StringIO()
            allo_d.emit_vhls(module, buf)
            code = buf.getvalue()
            counts[size] = len(re.findall(r"^void\s+\w+\(", code, re.M))
            # The channels and the instantiation the emitter is responsible for.
            assert "#pragma HLS stream variable=" in code
            assert "#pragma HLS unroll" in code

    assert len(set(counts.values())) == 1, f"HLS bodies should be flat, got {counts}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
