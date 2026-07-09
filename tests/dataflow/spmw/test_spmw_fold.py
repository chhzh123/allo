# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo.spmw as spmw
from allo.ir.types import float32


def _twin(M, N, K, **map_kwargs):
    """A systolic GEMM twin whose spmw.map forwards fold=/unroll= (or nothing)."""
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
        spmw.map(pe, grid=grid, **map_kwargs)
        spmw.stream_in(A, into=pe, flow="W->E")
        spmw.stream_in(B, into=pe, flow="N->S")
        spmw.stream_out(C, from_=pe, where="local", as_="c_local")

    return gemm


# --- the fold/unroll factors reach the spmw.map op, in full-rank form ---


def test_fold_dict_sets_map_attr():
    # {"row": 2} names axis 0; the other axis defaults to 1
    text = str(spmw.lower(_twin(4, 4, 2, fold={"row": 2})))
    assert "fold = array<i64: 2, 1>" in text
    assert text.count("spmw.map") == 1


def test_unroll_full_rank_list_sets_map_attr():
    text = str(spmw.lower(_twin(4, 4, 2, unroll=[2, 4])))
    assert "unroll = array<i64: 2, 4>" in text


def test_fold_and_unroll_together_int_and_named_axes():
    text = str(spmw.lower(_twin(4, 4, 2, fold={1: 2}, unroll={"row": 2})))
    assert "fold = array<i64: 1, 2>" in text
    assert "unroll = array<i64: 2, 1>" in text


def test_unfolded_map_carries_no_fold_or_unroll_attr():
    # a plain map is byte-for-byte unchanged: neither optional attr appears
    text = str(spmw.lower(_twin(4, 4, 2)))
    assert "fold = array<i64" not in text
    assert "unroll = array<i64" not in text


def test_fold_unroll_survive_the_passes():
    # the factors are not consumed yet (folding lands later), but they must ride through the
    # existing passes without breaking them
    region = _twin(4, 4, 2, fold={"col": 2}, unroll={"row": 2})
    module = spmw.lower(region)
    spmw._run_module_pass(module, "spmw-role-partition")
    spmw._run_module_pass(module, "spmw-resolve-channels")
    passed = str(module)
    assert "spmw.partition" in passed
    assert "spmw.channel_families" in passed
    # and the whole-grid unroll still expands to one call per compute point (a 3x3 grid is all
    # interior roles, so nine calls to the interior role func)
    text = str(spmw.unroll(_twin(3, 3, 2, fold={"row": 3})))
    assert text.count("call @gemm_pe_interior") >= 9


# --- the frontend rejects malformed factors at build (trace) time, before any IR ---


def test_fold_rank_mismatch_rejected():
    with pytest.raises(spmw.SPMWError, match="rank"):
        spmw.lower(_twin(4, 4, 2, fold=[2, 2, 2]))


def test_fold_non_dividing_factor_rejected():
    # 3 does not divide the extent 4
    with pytest.raises(spmw.SPMWError, match="divide"):
        spmw.lower(_twin(4, 4, 2, fold={"row": 3}))


def test_unroll_non_positive_factor_rejected():
    with pytest.raises(spmw.SPMWError, match="positive"):
        spmw.lower(_twin(4, 4, 2, unroll={"col": 0}))


def test_fold_unknown_named_axis_rejected():
    with pytest.raises(spmw.SPMWError, match="axis"):
        spmw.lower(_twin(4, 4, 2, fold={"diag": 2}))


def test_fold_axis_out_of_range_rejected():
    with pytest.raises(spmw.SPMWError, match="out of range"):
        spmw.lower(_twin(4, 4, 2, fold={5: 2}))
