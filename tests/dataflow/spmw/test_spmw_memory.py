# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import allo.spmw as spmw
from allo.ir.types import float32


def test_shared_space_resolves_to_memory():
    buf = spmw.shared(float32, space="L2")
    assert buf.kind == "shared"
    assert buf.memory.resource == "URAM"


def test_space_levels_resolve_distinctly():
    # each logical level resolves to a concrete on-chip resource
    assert spmw.shared(float32, space="L1").memory.resource == "LUTRAM"
    assert spmw.shared(float32, space="L2").memory.resource == "URAM"
    assert spmw.shared(float32, space="L3").memory.resource == "BRAM"


def test_resource_escape_hatch_overrides_space():
    # resource= pins the concrete resource, overriding any logical level
    buf = spmw.shared(float32, space="L2", resource="BRAM")
    assert buf.memory.resource == "BRAM"


def test_shared_defaults_to_auto():
    assert spmw.shared(float32).memory.resource == "AUTO"


def test_banked_carries_axis_and_memory():
    buf = spmw.banked(float32, on="col", space="L2")
    assert buf.kind == "banked"
    assert buf.memory.resource == "URAM"
    # banked on "col" -> the tensor axis to partition along is 1
    assert buf.bank_axis == 1


def test_banked_on_int_axis():
    assert spmw.banked(float32, on=0, space="L3").bank_axis == 0


def test_view_shares_the_source_memory():
    src = spmw.shared(float32, space="L2")
    v = spmw.view(src, shape=(8, 8))
    assert v.kind == "view"
    assert v.memory is src.memory  # a view aliases the source's resolved memory
    assert v.shape == (8, 8)


def test_unknown_space_rejected():
    with pytest.raises(spmw.SPMWError, match="unknown memory space"):
        spmw.shared(float32, space="L9")


def test_unknown_resource_rejected():
    with pytest.raises(spmw.SPMWError, match="unknown memory resource"):
        spmw.shared(float32, resource="MRAM")


def test_banked_requires_on():
    with pytest.raises(spmw.SPMWError, match="requires on="):
        spmw.banked(float32, space="L2")


def test_banked_unknown_axis_rejected():
    with pytest.raises(spmw.SPMWError, match="unknown banking axis"):
        spmw.banked(float32, on="diagonal", space="L2")


def test_view_of_non_buffer_rejected():
    with pytest.raises(spmw.SPMWError, match="expects a spmw.shared"):
        spmw.view(float32)
