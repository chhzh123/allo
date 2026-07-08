# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo.spmw as spmw
from allo.ir.types import float32


def _systolic_twin(M, N, K):
    grid = spmw.mesh((M, N))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(K):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm


def test_interior_role_func_carries_real_datapath():
    # the rolled spmw.map interior role is no longer an empty stub: lower() transcribes the unit
    # body into the same allo ops the dataflow frontend emits, once, parameterized by the writer PID
    text = str(spmw.lower(_systolic_twin(2, 2, 2)))
    assert "func.func @gemm_pe_interior(" in text
    for op in (
        "allo.stream_get",
        "arith.mulf",
        "arith.addf",
        "allo.stream_put",
        "memref.store",
    ):
        assert op in text
    # parameterized by the writer position (index args) and streaming over typed FIFOs whose depth
    # matches the topology's port depth (default 2), not a hard-coded constant
    assert "index" in text
    assert "!allo.stream<f32, 2>" in text
    # still one rolled map and one interior func regardless -- no per-PID clones
    assert text.count("spmw.map") == 1
    assert text.count("func.func @gemm_pe_interior(") == 1


def test_halo_loader_drain_datapaths_are_separate_boundary_funcs():
    # the boundary datapaths implied by the stream flows are synthesized as real funcs:
    # a loader reads its operand row/col and streams it in, a drain consumes at the exit edge
    text = str(spmw.lower(_systolic_twin(2, 2, 2)))
    for sym in (
        "gemm_pe_load_A",
        "gemm_pe_load_B",
        "gemm_pe_drain_A",
        "gemm_pe_drain_B",
    ):
        assert f"@{sym}(" in text
    assert text.count("memref.load") == 2  # exactly the two loaders read an operand
    assert "allo.stream_get" in text and "allo.stream_put" in text
    # but they are boundary tasks around the grid, NOT compute-grid roles: the map carries only the
    # interior compute role, so no edge PE loses its compute and corners stay unambiguous
    assert text.count("spmw.role") == 1
    assert "@gemm_pe_interior" in text
    assert text.count("spmw.map") == 1


def test_map_carries_real_tensor_operands():
    # spmw.map now passes the region's real A/B/C memrefs (matching the role tensor ABI), not a
    # memref<1xf32> placeholder
    text = str(spmw.lower(_systolic_twin(2, 2, 2)))
    assert (
        "func.func @gemm(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, "
        "%arg2: memref<2x2xf32>)" in text
    )
    assert "spmw.map(%arg0, %arg1, %arg2)" in text
    assert "memref<1xf32>" not in text


def test_explicit_role_body_fails_closed():
    # an explicit @pe.role carrying real work would otherwise be silently dropped to an empty stub;
    # lowering rejects it until explicit-role datapath transcription lands
    grid = spmw.mesh((4, 4))

    @spmw.unit
    def pe(ctx):
        pass

    @pe.role("west")
    def load(ctx):
        ctx.east.put(1)

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid)

    with pytest.raises(spmw.SPMWError, match="cannot transcribe"):
        spmw.lower(r)


def test_lower_carries_halo_tasks():
    # the rolled spmw.map now records its loader/drain boundary tasks as #spmw.halo attributes so the
    # spmw-unroll pass can wire them to the edge channels, instead of leaving them orphan siblings
    text = str(spmw.lower(_systolic_twin(2, 2, 2)))
    assert "#spmw.halo<unit = @gemm_pe_load_A" in text
    assert "#spmw.halo<unit = @gemm_pe_load_B" in text
    assert 'kind = "load"' in text
    assert 'kind = "drain"' in text


def test_untranscribable_systolic_body_fails_closed():
    # a systolic region whose interior uses a construct the transcriber does not handle must raise,
    # not silently lower to an empty stub
    M, N, K = 2, 2, 2
    grid = spmw.mesh((M, N))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(K):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            if a:  # a control-flow construct the datapath transcriber does not handle
                c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    with pytest.raises(spmw.SPMWError):
        spmw.lower(gemm)


def _mesh_region_with_roles(shape):
    grid = spmw.mesh(shape)

    @spmw.unit
    def pe(ctx):
        a = ctx.west.get()
        ctx.east.put(a)

    @pe.role("west")
    def pe_load(ctx):
        pass

    @pe.role("east", "south")
    def pe_drain(ctx):
        pass

    @spmw.region()
    def gemm(A):
        spmw.map(pe, grid=grid)

    return gemm


def test_lower_one_map_and_role_funcs():
    text = str(spmw.lower(_mesh_region_with_roles((4, 4))))
    # exactly one rolled map op
    assert text.count("spmw.map") == 1
    # one role func per predicate tag: interior + west + (east, south) = 3
    assert text.count("func.func @gemm_pe_") == 3
    # the typed topology attribute prints positionally on the op
    assert "topology = <" in text
    assert "grid = [4, 4]" in text


def test_role_count_independent_of_grid_size():
    small = str(spmw.lower(_mesh_region_with_roles((4, 4))))
    large = str(spmw.lower(_mesh_region_with_roles((8, 8))))
    # the role-func count and map count do NOT grow with the grid (no P0*P1 blow-up)
    assert small.count("func.func @gemm_pe_") == 3
    assert large.count("func.func @gemm_pe_") == 3
    assert small.count("spmw.map") == 1
    assert large.count("spmw.map") == 1
    assert "grid = [4, 4]" in small
    assert "grid = [8, 8]" in large


def test_topology_carries_four_peer_links():
    text = str(spmw.lower(_mesh_region_with_roles((4, 4))))
    assert text.count("spmw.peer_link") == 4


def test_build_ir_target_returns_module():
    module = spmw.build(_mesh_region_with_roles((4, 4)), target="ir")
    assert "spmw.map" in str(module)


def test_build_execution_target_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        spmw.build(_mesh_region_with_roles((4, 4)), target="simulator")


def test_grid_only_region_lowers_with_empty_links():
    @spmw.unit
    def pe(ctx):
        pass

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=spmw.Grid((3, 3)))

    text = str(spmw.lower(r))
    assert "links = []" in text
    assert text.count("func.func @r_pe_") == 1  # interior role only


def test_interior_role_declares_stream_ports():
    # the interior role now declares its stream ABI (one port per stream arg) so the pass binds
    # channels by port name, not by a sorted-port positional convention
    text = str(spmw.lower(_systolic_twin(2, 2, 2)))
    assert 'ports = ["east", "north", "south", "west"]' in text
