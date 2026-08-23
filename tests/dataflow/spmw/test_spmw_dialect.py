# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The spmw dialect: the rolled form a spatial design keeps to code generation.

These exercise the IR directly and touch nothing in the frontend, so a change to
either side cannot quietly drift the other.
"""

import pytest

from allo._mlir.dialects import allo as allo_d
from allo._mlir.ir import Context, Module

TOP = """
func.func @pe(%%A: memref<4x4xf32>, %%i: index, %%j: index,
              %%w: !allo.stream<f32, 2>, %%e: !allo.stream<f32, 2>) {
  return
}
func.func @top(%%A: memref<4x4xf32>) attributes {dataflow} {
  spmw.map (%%A)
    topology = #spmw.topology<grid = [4, 4],
      families = [#spmw.family<name = "ew", type = f32, block = [], depth = 2,
                               shape = [4, 4]>],
      ports = [#spmw.port_map<port = "west", family = "ew", kind = "affine",
                              offset = [0, 0]>,
               #spmw.port_map<port = "east", family = "ew", kind = "affine",
                              offset = [0, 1]>%s]>
    roles = [#spmw.role<unit = @pe, missing = [], ports = ["west", "east"]>]
    classes = dense<%s> : tensor<4x4xi32>
    : memref<4x4xf32>
  return
}
"""


def build(classes="0", extra_port=""):
    return TOP % (", " + extra_port if extra_port else "", classes)


@pytest.fixture(name="ctx")
def _ctx():
    with Context() as ctx:
        allo_d.register_dialect(ctx)
        yield ctx


def test_round_trips(ctx):
    text = str(Module.parse(build()))
    assert "spmw.map" in text
    assert "#spmw.family" in text
    assert "#spmw.role" in text


def test_a_class_must_name_a_declared_role(ctx):
    with pytest.raises(Exception, match="classes names role 7"):
        Module.parse(build(classes="7"))


def test_a_table_port_must_carry_slots(ctx):
    """A port with no closed-form index needs the table that gives it one."""
    with pytest.raises(Exception, match="table-addressed but carries no slots"):
        Module.parse(
            build(
                extra_port='#spmw.port_map<port = "n", family = "ew", kind = "table">'
            )
        )


def test_a_port_must_address_a_declared_family(ctx):
    with pytest.raises(Exception, match="does not declare"):
        Module.parse(
            build(
                extra_port='#spmw.port_map<port = "n", family = "zz", '
                'kind = "affine", offset = [0, 0]>'
            )
        )


def test_an_affine_offset_must_match_the_grid_rank(ctx):
    with pytest.raises(Exception, match="rank-1 offset on a rank-2 grid"):
        Module.parse(
            build(
                extra_port='#spmw.port_map<port = "n", family = "ew", '
                'kind = "affine", offset = [1]>'
            )
        )


def test_the_role_calling_convention_is_checked(ctx):
    """Tensors, then one pid per grid axis, then one stream per declared port."""
    bad = build().replace(
        'ports = ["west", "east"]', 'ports = ["west", "east", "north"]'
    )
    with pytest.raises(Exception, match="parameters; the map implies"):
        Module.parse(bad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
