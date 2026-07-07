# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo.spmw as spmw


def _systolic_twin():
    grid = spmw.mesh((4, 4))

    @spmw.unit
    def pe(ctx):
        a = ctx.west.get()
        b = ctx.north.get()
        ctx.east.put(a)
        ctx.south.put(b)

    @spmw.region()
    def gemm(A, B, C):
        spmw.map(pe, grid=grid)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm


def test_halo_roles_synthesized_from_flows():
    roles = spmw.halo_roles(_systolic_twin())
    # A streams W->E: loaded at the west edge, drained at the east edge
    assert ("west", "loader", "A") in roles
    assert ("east", "drain", "A") in roles
    # B streams N->S: loaded at the north edge, drained at the south edge
    assert ("north", "loader", "B") in roles
    assert ("south", "drain", "B") in roles
    # the where="local" output carries no flow, so it adds no halo role
    assert len(roles) == 4


def test_no_flow_streams_synthesize_no_halo():
    @spmw.unit
    def pe(ctx):
        pass

    @spmw.region()
    def r(A):
        spmw.map(pe, grid=spmw.mesh((3, 3)))

    assert spmw.halo_roles(r) == []
