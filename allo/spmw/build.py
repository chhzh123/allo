# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiling an SPMW fabric.

Elaborate the fabric, lower it to a dataflow program, and hand that to the
existing backends.  The targets are the dataflow ones, unchanged:
``simulator`` for functional checks, ``vitis_hls`` and friends for hardware.
"""

from .errors import SPMWPlacementError
from .graph import elaborate
from .lower_df import build_dataflow, render_source


def customize(fabric_fn, tensor_specs=None, keep=None, verbose=False):
    """Elaborate a fabric and return its dataflow schedule.

    The intermediate program is kept on the returned object so a failure in the
    backend can be read against the source that produced it.
    """
    import allo.dataflow as df  # pylint: disable=import-outside-toplevel

    graph = elaborate(fabric_fn, tensor_specs=tensor_specs)
    top = build_dataflow(graph, keep=keep)
    if verbose:
        print(top._spmw_source)  # pylint: disable=protected-access
    schedule = df.customize(top)
    schedule.spmw_graph = graph
    schedule.spmw_source = top._spmw_source  # pylint: disable=protected-access
    return schedule


def build(
    fabric_fn, target="simulator", tensor_specs=None, keep=None, verbose=False, **kwargs
):
    """Elaborate, lower, and compile a fabric for ``target``."""
    import allo.dataflow as df  # pylint: disable=import-outside-toplevel

    graph = elaborate(fabric_fn, tensor_specs=tensor_specs)
    top = build_dataflow(graph, keep=keep)
    if verbose:
        print(top._spmw_source)  # pylint: disable=protected-access
    module = df.build(top, target=target, **kwargs)
    module.spmw_graph = graph
    module.spmw_source = top._spmw_source  # pylint: disable=protected-access
    return module


def _check_realised(graph):
    """Refuse to build a placement whose knobs this path does not honour.

    ``fold`` and ``unroll`` decide how much of the grid is space and how much is
    time, so a build that quietly ignored them would hand back a different design
    from the one asked for.
    """
    for placement in graph.placements:
        for name in ("fold", "unroll", "layout"):
            value = getattr(placement, name, None)
            if value:
                raise SPMWPlacementError(
                    f"`{placement.name}` was placed with {name}={value!r}, which "
                    f"this path does not realise yet. It elaborates and checks, "
                    f"but the build would silently ignore the knob -- drop it to "
                    f"build the fully spatial design."
                )


def source(fabric_fn, tensor_specs=None):
    """The dataflow program a fabric lowers to, without compiling it."""
    return render_source(elaborate(fabric_fn, tensor_specs=tensor_specs))


__all__ = ["build", "customize", "source"]
