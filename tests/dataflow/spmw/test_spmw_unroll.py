# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re

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


def _region(shape):
    grid = spmw.mesh(shape)

    @spmw.unit
    def pe(ctx):
        ctx.west.get()

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


def test_unroll_one_call_per_grid_point():
    text = str(spmw.unroll(_region((4, 4))))
    assert text.count("call @gemm_pe_") == 16  # P0*P1 = 16 per-PID calls


def test_role_func_count_is_constant_across_grid_sizes():
    small = str(spmw.unroll(_region((4, 4))))
    large = str(spmw.unroll(_region((6, 6))))
    # the role func BODIES stay O(#roles) = 3, never cloned per grid point
    assert small.count("func.func @gemm_pe_") == 3
    assert large.count("func.func @gemm_pe_") == 3
    # ... while the per-PID call count scales with the grid
    assert small.count("call @gemm_pe_") == 16
    assert large.count("call @gemm_pe_") == 36


def test_unroll_orders_loaders_before_and_drains_after_compute():
    """Per grid point, `spmw.unroll` emits LOADER halos (producers) BEFORE the compute-role call and
    DRAIN halos (consumers) AFTER it, so the unrolled call order is safe even under a sequential
    lowering: a west/north edge PE never `stream_get`s a boundary FIFO before its loader has run, and a
    drain never `stream_get`s a FIFO before the PE's `stream_put`. The sequence opens with a loader
    (producer first) and ends on a drain (consumer last)."""
    calls = re.findall(
        r"call @(\w+)", str(spmw.build(_systolic_twin(2, 2, 2), target="unroll"))
    )
    assert calls, "no calls in the unrolled IR"
    assert "load" in calls[0]  # opens with a loader (producer before compute)
    assert "drain" in calls[-1]  # ends on a drain (consumer after compute)
    # no drain precedes the FIRST compute call, and no loader follows the LAST compute call
    first_compute = next(i for i, c in enumerate(calls) if "interior" in c)
    last_compute = max(i for i, c in enumerate(calls) if "interior" in c)
    assert not any("drain" in c for c in calls[:first_compute])
    assert not any("load" in c for c in calls[last_compute + 1 :])


def test_role_assignment_by_missing_links():
    text = str(spmw.unroll(_region((4, 4))))
    # the west column (4 PIDs) is missing its west link -> the load role;
    # the south-east corner is missing east and south -> the drain role;
    # every other grid point -> interior.
    assert text.count("call @gemm_pe_west") == 4
    assert text.count("call @gemm_pe_east_south") == 1
    assert text.count("call @gemm_pe_interior") == 11


def test_build_unroll_target():
    text = str(spmw.build(_region((4, 4)), target="unroll"))
    assert text.count("call @gemm_pe_") == 16


def test_unroll_consumes_the_rolled_map():
    # unroll now runs the spmw-unroll MLIR pass over the rolled spmw.map rather than re-deriving
    # the expansion in Python: the map op is genuinely consumed and no longer present afterwards.
    text = str(spmw.unroll(_region((4, 4))))
    assert "spmw.map" not in text
    assert text.count("call @gemm_pe_") == 16


def test_unroll_streaming_interior_wires_channels():
    # the systolic twin's interior role carries the real streaming datapath (A/B/C memrefs, %pi/%pj
    # PID indices, and one !allo.stream per port). The pass instantiates it across the grid with
    # per-PID index constants and allo.stream_construct channels -- and the result verifies.
    text = str(spmw.unroll(_systolic_twin(3, 3, 2)))
    assert "spmw.map" not in text
    # only the interior role exists, so every one of the 9 grid points calls it
    assert text.count("call @gemm_pe_interior") == 9
    # channels are materialized as allo.stream_construct and PID indices as arith.constant
    assert "allo.stream_construct" in text
    assert "arith.constant" in text


def test_unroll_wires_halo_loaders_and_drains():
    text = str(spmw.unroll(_systolic_twin(3, 3, 2)))
    # the halo loader/drain funcs are now CALLED (not orphaned siblings): the W->E flow gives 3
    # west-edge A-loaders + 3 east-edge A-drains, the N->S flow 3 north-edge B-loaders + 3 south-edge
    # B-drains (one per edge point of the 3x3 grid)
    assert text.count("call @gemm_pe_load_A") == 3
    assert text.count("call @gemm_pe_load_B") == 3
    assert text.count("call @gemm_pe_drain_A") == 3
    assert text.count("call @gemm_pe_drain_B") == 3
    # the loaders/drains REUSE the edge PEs' boundary channels rather than allocating new ones: the
    # stream_construct count is unchanged (12 shared peer + 12 boundary = 24) -- no channel is left
    # dangling with only one endpoint
    assert text.count("allo.stream_construct") == 24


def test_ambiguous_role_rejected():
    grid = spmw.mesh((4, 4))

    @spmw.unit
    def pe(ctx):
        pass

    @pe.role("west")
    def west_role(ctx):
        pass

    @pe.role("north")
    def north_role(ctx):
        pass

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=grid)

    # the NW corner is missing both west and north; two size-1 roles fit -> ambiguous
    with pytest.raises(spmw.SPMWError, match="ambiguous role"):
        spmw.unroll(r)


def _checkerboard_twin(n):
    grid = spmw.mesh((n, n))

    @spmw.unit
    def pe(ctx):
        c: float32 = 0
        for k in range(4):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a * b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @pe.variant("(d0 + d1) mod 2")
    def pe_odd(ctx):
        c: float32 = 0
        for k in range(4):
            a: float32 = ctx.west.get()
            b: float32 = ctx.north.get()
            c += a + b
            ctx.east.put(a)
            ctx.south.put(b)
        ctx.c_local[0] = c

    @spmw.region()
    def gemm(A: float32[n, 4], B: float32[4, n], C: float32[n, n]):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm


def test_unroll_selects_predicate_variant_per_point():
    # spmw-unroll is predicate-aware: a checkerboard variant map expands to per-PID calls that select
    # the base interior body on even cells and the variant body on odd cells -- not an ambiguous tie,
    # so the predicated spmw.map is consumable by the simulator lowering, not only the HLS emitter
    text = str(spmw.unroll(_checkerboard_twin(4)))
    assert text.count("call @gemm_pe_interior(") == 8  # even cells
    assert text.count("call @gemm_pe_variant0(") == 8  # odd cells
