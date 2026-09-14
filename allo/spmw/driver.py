# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiling an SPMW fabric.

Elaborate the fabric, build its program as MLIR -- one ``spmw.map`` per
placement and per mover, one function per role -- and hand the expanded form
to the existing backends.  The targets are the dataflow ones, unchanged:
``simulator`` for functional checks, ``vitis_hls`` and friends for hardware.
"""

from . import schedule as sched
from .errors import SPMWPlacementError
from .graph import elaborate
from .lower_mlir import TOP, build_program


def _top_of(module):
    for op in module.body.operations:
        name = getattr(getattr(op, "name", None), "value", None)
        if name == TOP:
            return op
    raise SPMWPlacementError(f"the built module has no `{TOP}` function.")


def _schedule(built):
    """Wrap a built program as an Allo schedule, ready for the HLS backends."""
    # pylint: disable=import-outside-toplevel
    from allo._mlir.ir import InsertionPoint
    from allo.customize import Schedule
    from allo.ir.utils import MockBuffer

    top = _top_of(built.module)
    schedule = Schedule(
        built.module,
        top,
        built.func_args,
        InsertionPoint.at_block_terminator(top.entry_block),
        ext_libs=built.ext_libs,
        inst_list=[],
        func_instances={},
    )
    schedule.stateful_var_map = {}
    # The top's tensors, addressable the way `customize` exposes a function's
    # arguments, so `partition` can name them.
    for idx, name in enumerate(built.arg_order):
        setattr(schedule, name, MockBuffer(TOP, name, idx))
    schedule.spmw_graph = built.graph
    schedule.spmw_source = built.rolled
    schedule.spmw_arg_order = list(built.arg_order)
    return schedule


def customize(fabric_fn, tensor_specs=None, verbose=False):
    """Elaborate a fabric and return its schedule, before any backend runs.

    The rolled program is kept on the returned object as ``spmw_source`` so a
    failure in the backend can be read against the IR that produced it.
    """
    graph = elaborate(fabric_fn, tensor_specs=tensor_specs)
    _check_realised(graph)
    built = build_program(graph)
    if verbose:
        print(built.rolled)
    return _schedule(built)


def build(
    fabric_fn,
    target="simulator",
    tensor_specs=None,
    verbose=False,
    partition=True,
    **kwargs,
):
    """Elaborate, lower, and compile a fabric for ``target``.

    ``target="ref"`` runs the graph directly -- one task per site over bounded
    channels -- and never touches the compiler, which is the fastest way to ask
    whether a fabric computes the right thing and whether it deadlocks.

    ``partition`` completely partitions every tensor argument on the FPGA
    targets, which a spatial design needs to *synthesise* at all: the sites read
    and write one array between them, and HLS dataflow allows a single reader
    and a single writer per interface array unless its elements are
    independently addressable.  Without it ``csynth`` stops at

        [HLS 200-979] Argument 'v185' failed dataflow checking:
                      it can only be written in one process function.

    Pass ``partition=False`` to emit the unpartitioned program -- worth doing to
    see that error, and for a design whose arrays are too large to scalarise.
    """
    graph = elaborate(fabric_fn, tensor_specs=tensor_specs)
    _check_realised(graph)
    if target == "ref":
        from .refsim import build_ref  # pylint: disable=import-outside-toplevel

        return build_ref(graph, **kwargs)

    built = build_program(graph)
    if verbose:
        print(built.rolled)
    if target == "simulator":
        # pylint: disable=import-outside-toplevel
        from allo.backend.simulator import LLVMOMPModule

        module = LLVMOMPModule(built.module, TOP)
    else:
        schedule = _schedule(built)
        if partition:
            module = _build_partitioned(schedule, graph, target, **kwargs)
        else:
            module = schedule.build(target=target, **kwargs)
    module.spmw_graph = graph
    module.spmw_source = built.rolled
    return _Callable(module, graph, built.arg_order)


def _build_partitioned(schedule, graph, target, **kwargs):
    """Pipeline every body, partition every tensor, then build."""
    # Without this each site's loop is scheduled sequentially. One interval
    # for the whole program -- the bodies are not separable here, unlike a
    # unit, which is built on its own and takes its placement's own.
    intervals = {sched.interval(p) for p in graph.placements} or {1}
    sched.apply(schedule, sched.function_names(schedule), min(intervals))
    for tensor in graph.tensors.values():
        buffer = getattr(schedule, tensor.name, None)
        if buffer is None:
            raise SPMWPlacementError(
                f"`{tensor.name}` is a fabric tensor but the built schedule has "
                f"no buffer for it, so it cannot be partitioned."
            )
        schedule.partition(buffer)
    return schedule.build(
        target=target,
        mode=kwargs.pop("mode", "csim"),
        project=kwargs.pop("project", "top.prj"),
        **kwargs,
    )


class _Callable:
    """The built module, called with the fabric's own argument order.

    The program's top takes its tensors inputs-first in the order the maps use
    them, which is what the HLS backend requires.  The fabric declares its own
    order, and that is the one the caller wrote against, so the permutation
    between them is applied here rather than left as a trap.
    """

    def __init__(self, module, graph, backend_order):
        self._module = module
        self.spmw_graph = graph
        self.spmw_source = getattr(module, "spmw_source", None)
        declared = [t.name for t in graph.tensors.values()]
        missing = [name for name in declared if name not in backend_order]
        if missing:
            raise SPMWPlacementError(
                f"`{graph.fabric.name}` declares {', '.join(missing)} but nothing "
                f"in the fabric reads or writes it, so the built module has no "
                f"argument for it."
            )
        self._order = [declared.index(name) for name in backend_order]
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
    """The rolled program a fabric lowers to, as MLIR text."""
    return build_program(elaborate(fabric_fn, tensor_specs=tensor_specs)).rolled


__all__ = ["build", "customize", "source"]
