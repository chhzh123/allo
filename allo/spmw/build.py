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
    _check_realised(graph)
    top = build_dataflow(graph, keep=keep)
    if verbose:
        print(top._spmw_source)  # pylint: disable=protected-access
    schedule = df.customize(top)
    schedule.spmw_graph = graph
    schedule.spmw_source = top._spmw_source  # pylint: disable=protected-access
    # The built module takes its arguments in this order, which is the union of
    # the kernels' arguments in first-seen order rather than the fabric's own.
    schedule.spmw_arg_order = top._spmw_arg_order  # pylint: disable=protected-access
    return schedule


def build(
    fabric_fn, target="simulator", tensor_specs=None, keep=None, verbose=False, **kwargs
):
    """Elaborate, lower, and compile a fabric for ``target``.

    ``target="ref"`` runs the graph directly -- one task per site over bounded
    channels -- and never touches the compiler, which is the fastest way to ask
    whether a fabric computes the right thing and whether it deadlocks.
    """
    graph = elaborate(fabric_fn, tensor_specs=tensor_specs)
    _check_realised(graph)
    if target == "ref":
        from .refsim import build_ref  # pylint: disable=import-outside-toplevel

        return build_ref(graph, **kwargs)

    import allo.dataflow as df  # pylint: disable=import-outside-toplevel

    top = build_dataflow(graph, keep=keep)
    if verbose:
        print(top._spmw_source)  # pylint: disable=protected-access
    module = df.build(top, target=target, **kwargs)
    module.spmw_graph = graph
    module.spmw_source = top._spmw_source  # pylint: disable=protected-access
    return _Callable(module, graph, top)


class _Callable:
    """The built module, called with the fabric's own argument order.

    The dataflow backend builds its top signature from the union of the kernels'
    arguments in first-seen order, which is the order the kernels happen to be
    emitted in. The fabric declares its own order, and that is the one the caller
    wrote against, so the permutation between them is applied here rather than
    left as a trap.
    """

    def __init__(self, module, graph, top):
        self._module = module
        self.spmw_graph = graph
        self.spmw_source = top._spmw_source  # pylint: disable=protected-access
        declared = [t.name for t in graph.tensors.values()]
        backend = top._spmw_arg_order  # pylint: disable=protected-access
        missing = [name for name in declared if name not in backend]
        if missing:
            raise SPMWPlacementError(
                f"`{graph.fabric.name}` declares {', '.join(missing)} but nothing "
                f"in the fabric reads or writes it, so the built module has no "
                f"argument for it."
            )
        self._order = [declared.index(name) for name in backend]
        self._declared = declared

    def __call__(self, *arrays):
        if len(arrays) != len(self._declared):
            raise SPMWPlacementError(
                f"expected {len(self._declared)} arrays "
                f"({', '.join(self._declared)}), got {len(arrays)}."
            )
        return self._module(*[arrays[i] for i in self._order])

    def __getattr__(self, name):
        return getattr(self.__dict__["_module"], name)

    def __repr__(self):
        return f"<spmw module {self.spmw_graph.fabric.name}>"


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
