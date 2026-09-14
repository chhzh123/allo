# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the rolling holds, all the way down.

The design's load-bearing claim is that the number of *bodies* a spatial design
compiles to tracks its role count, not its grid: a 2-D mesh has nine site
signatures at any size, so it should emit nine bodies at any size.

The frontend delivers that, the program keeps it -- one function per role and a
``spmw.map`` carrying the instantiation -- and the HLS code the backend emits
keeps it too, because the expanded form calls the same nine functions from
every site rather than copying them.
"""

import re

import numpy as np
import pytest

import allo.spmw as spmw
from allo.ir.types import float32

SIZES = [2, 3, 4, 6]


def role_functions(text):
    """The role bodies in a rolled program: the functions a map names."""
    return len(re.findall(r"^\s*func\.func @\w+_r\d+\(", text, re.M))


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
def test_emitted_bodies_track_signatures_not_sites(size):
    """One function per signature class in the program, whatever the grid."""
    text = spmw.source(gemm_of(size))
    bodies = role_functions(text)
    expected = 9 if size >= 3 else 4
    assert bodies == expected, f"{size}x{size}: {bodies} bodies for {expected}"
    # The instantiation is an attribute on the map, not more functions.
    assert text.count("spmw.map") == 3  # the array and its two loaders


@pytest.mark.parametrize("size", SIZES)
def test_hls_body_count_is_flat(size):
    """The count HLS actually sees: the roles, the loaders, and the top.

    This used to be one body per site -- the dataflow builder expanded one
    kernel instance per grid point -- and this test pinned that gap.  The
    expanded program now calls the same role function from every site, so the
    count is bounded by the role count rather than by the grid.
    """
    mod = spmw.build(gemm_of(size), target="vhls")
    bodies = len(re.findall(r"^void\s+\w+\(", mod.hls_code, re.M))
    roles = 9 if size >= 3 else 4
    # roles + one loader per operand + the top.
    assert bodies == roles + 2 + 1


def test_the_hls_body_count_is_as_flat_as_the_frontend():
    """State the two curves side by side, since that contrast was the whole point."""
    bodies, functions = {}, {}
    for size in SIZES:
        fab = gemm_of(size)
        bodies[size] = role_functions(spmw.source(fab))
        mod = spmw.build(fab, target="vhls")
        functions[size] = len(re.findall(r"^void\s+\w+\(", mod.hls_code, re.M))

    big = [s for s in SIZES if s >= 3]
    assert len({bodies[s] for s in big}) == 1, f"bodies should be flat, got {bodies}"
    assert (
        len({functions[s] for s in big}) == 1
    ), f"HLS functions should be flat too, got {functions}"


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

    This is the design's load-bearing claim: the same designs used to expand to
    one body per site through the dataflow path, and stay flat here.
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
    # The largest grid here has 256 sites; one body per site would be 289.
    assert max(ROLLED_SIZES) ** 2 // roles[max(ROLLED_SIZES)] > 25


def test_the_rolled_bodies_are_real():
    """The role functions carry the unit's arithmetic, not an empty shell.

    Tensors, then one index per grid axis, then one stream per wired port -- and
    the body stores its result through the map's tensor at its own coordinates.
    """
    text = spmw.source(gemm_of(4))
    interior = re.search(r"func\.func @pe_r0\(([^)]*)\)", text).group(1)
    assert interior.count("memref<4x4xf32>") == 1
    assert interior.count("index") == 2
    assert interior.count("!allo.stream<f32, 2>") == 4
    assert "arith.mulf" in text and "arith.addf" in text
    assert "allo.stream_get" in text and "allo.stream_put" in text


def test_the_rolled_form_emits_flat_hls():
    """The number that the whole exercise is about.

    Same designs, same nine roles, and the HLS function count does not grow
    with the array whether the mesh has nine sites or two hundred and fifty-six.
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
