# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a unit is compiled, as opposed to what it computes.

The frontend is the specification: ports, links, placements, bindings.  Nothing
here changes what a design *means* -- it decides how the loops inside one unit
are scheduled, which is a compilation choice and belongs on its own axis.

The default matters more than the knob.  A unit body's inner loop is a spatial
design's entire inner loop, and left alone Vitis HLS schedules it sequentially:
every SPMW design emitted before this ran its PEs one operation at a time, with
no ``#pragma HLS pipeline`` anywhere in the generated code.  So loops are
pipelined at II=1 by default and the primitive is there to override that.

``pipeline(P, ii=n)`` asks for a different interval; ``pipeline(P, ii=0)`` turns
it off.  A design that cannot meet II=1 -- a float accumulator's recurrence is
bounded by the adder's latency -- still benefits from being pipelined at the
interval it *can* meet, which is what HLS falls back to on its own.
"""

from .errors import SPMWMemoryError, SPMWPlacementError

PIPELINE = "pipeline"


class Directive:
    """One scheduling request against a placement's unit."""

    __slots__ = ("kind", "value")

    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"<{self.kind} {self.value}>"


def pipeline(target, ii=1):
    """Pipeline the unit's loops at initiation interval ``ii``.

    ``target`` is a placement -- what :func:`allo.spmw.place` returned.  ``ii=0``
    leaves the loops alone, which is worth having to measure what the pipelining
    is buying.
    """
    if not hasattr(target, "schedule"):
        raise SPMWPlacementError(
            f"pipeline() applies to a placement; got {type(target).__name__}."
        )
    if not isinstance(ii, int) or ii < 0:
        raise SPMWPlacementError(
            f"pipeline(ii=) must be a non-negative int, got {ii!r}"
        )
    target.schedule = [d for d in target.schedule if d.kind != PIPELINE]
    target.schedule.append(Directive(PIPELINE, ii))
    return target


def interval(placement, default=1):
    """The initiation interval asked for at this placement."""
    for directive in getattr(placement, "schedule", ()):
        if directive.kind == PIPELINE:
            return directive.value
    return default


def function_names(schedule):
    """The functions the built module actually has."""
    names = []
    for op in schedule.module.body.operations:
        name = getattr(op, "name", None)
        value = getattr(name, "value", None)
        if value is not None:
            names.append(value)
    return names


def apply(schedule, functions, ii):
    """Pipeline the innermost loop of every band in ``functions``.

    The innermost loop is the one worth pipelining: pipelining an outer loop
    instead would flatten the nest, which is a different design.

    A requested function that the module does not have is skipped. That is
    checked against the module's own symbols rather than by catching what
    ``get_loops`` throws -- a unit is looked up under two names and only one
    exists, and guessing which exception stands for "no such function" got it
    wrong once already.

    Returns the loops it pipelined, for the caller to report or check.
    """
    if not ii:
        return []
    have = set(function_names(schedule))
    done = []
    for name in functions:
        if name not in have:
            continue
        for band in schedule.get_loops(name).loops.values():
            loops = list(getattr(band, "loops", {}).values())
            if not loops:
                continue
            schedule.pipeline(loops[-1], initiation_interval=ii)
            done.append(loops[-1].name)
    return done


def accumulators(tree, io_name=None):
    """Names the body carries across iterations of a loop.

    A name assigned inside a loop, read inside it, and live before it is an
    accumulator: its update is a *recurrence*, and a recurrence through a
    floating-point add is what stops a unit reaching II=1.  Finding them is a
    property of the body, so it is read off the source rather than guessed at
    from the generated code.
    """
    import ast  # pylint: disable=import-outside-toplevel

    before = set()
    found = []
    for stmt in tree.body:
        loops = [n for n in ast.walk(stmt) if isinstance(n, (ast.For, ast.While))]
        if not loops:
            for node in ast.walk(stmt):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    for target in _targets(node):
                        before.add(target)
            continue
        for loop in loops:
            written, read = set(), set()
            for node in ast.walk(loop):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    written.update(_targets(node))
                if isinstance(node, ast.AugAssign):
                    # `acc += x` reads acc, but its target carries a Store
                    # context, so the walk below never sees the read.
                    read.update(_targets(node))
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    read.add(node.id)
            for name in sorted(written & read & before):
                if name != io_name and name not in found:
                    found.append(name)
    return found


def _targets(node):
    """The plain names one assignment writes."""
    import ast  # pylint: disable=import-outside-toplevel

    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [t.id for t in targets if isinstance(t, ast.Name)]


def bind_recurrences(code, names, latency):
    """Bind each accumulator's adder to ``latency``, in generated HLS C++.

    A recurrence through a float add costs II = latency + 1, and Vitis picks a
    deeply pipelined adder by default -- II=7 on the systolic GEMM's PE. Binding
    a shorter one trades combinational delay for interval, and unlike
    reassociating the sum it does not change a single rounding.

    The accumulator keeps its own name in the generated code, so its update
    ``acc = vNN;`` names the value whose adder to bind. Matching is line-based:
    Allo puts a ``// L17`` provenance comment after every statement, so a regex
    anchored at end-of-line silently matches nothing.

    Anything not found is skipped -- the binding is an optimisation, so missing
    it costs speed rather than correctness.

    Returns the rewritten code and the values bound.
    """
    import re  # pylint: disable=import-outside-toplevel

    lines = code.splitlines(True)
    bound = []
    for name in names:
        value = None
        for line in lines:
            match = re.match(rf"\s*{re.escape(name)}\s*=\s*(v\d+);", line)
            if match:
                value = match.group(1)
                break
        if value is None:
            continue
        for index, line in enumerate(lines):
            if re.match(rf"(\s*)float\s+{re.escape(value)}\s*=", line):
                indent = re.match(r"\s*", line).group(0)
                lines.insert(
                    index + 1,
                    f"{indent}#pragma HLS bind_op variable={value} op=fadd "
                    f"impl=fabric latency={latency}\n",
                )
                bound.append(value)
                break
    return "".join(lines), bound


def bind_fabric_arith(code):
    """Bind every float add and subtract in generated HLS C++ to fabric.

    `bind_recurrences` binds one adder per accumulator, because a recurrence
    through a float add sets the interval. This binds the *feed-forward* ones,
    which cost nothing in interval and everything in DSP blocks: Vitis
    implements a float add in DSPs by default, so a design that never says
    otherwise spends two of them on every butterfly add it performs.

    HP-FFT, the hand-written baseline this is measured against, carries six
    `bind_op ... impl=fabric` pragmas in its butterfly for exactly this reason
    and keeps only its multiplies in DSPs. Matching that is what makes the DSP
    columns comparable rather than a comparison of who remembered the pragma.

    Multiplies are deliberately left alone: the multiplier is the arithmetic
    the DSP exists for, and moving it to fabric would cost lookup tables to no
    purpose.

    Matching is line-based and mirrors `bind_recurrences`: Allo emits one
    operation per statement as ``float vNN = vA + vB;`` and puts a ``// L17``
    provenance comment *after* the semicolon, so the pattern stops at the
    semicolon rather than at end-of-line.

    Returns the rewritten code and the values bound. Anything unmatched is
    skipped -- a binding is an implementation directive, so missing one costs
    DSP blocks rather than correctness, and it changes no rounding.
    """
    import re  # pylint: disable=import-outside-toplevel

    out, bound = [], []
    for line in code.splitlines(True):
        out.append(line)
        match = re.match(
            r"(\s*)float\s+(v\d+)\s*=\s*[^;]*?([-+])\s*v\d+\s*;", line
        )
        if match:
            indent, value, op = match.group(1), match.group(2), match.group(3)
            out.append(
                f"{indent}#pragma HLS bind_op variable={value} "
                f"op={'fadd' if op == '+' else 'fsub'} impl=fabric\n"
            )
            bound.append(value)
    return "".join(out), bound


def partition_banks(code, banked):
    """Give every banked memory one set of ports per bank.

    Banking is a promise about *ports*, not about addresses: the point of the
    swizzle is that a butterfly's two operands sit in different banks so both
    can be read in one cycle. Vitis keeps a `[banks][rows]` array in a single
    memory unless told otherwise, so without this the swizzle costs its address
    arithmetic and buys nothing -- the accesses queue on the same two ports and
    the loop's interval multiplies by however many of them there are.

    Measured on the paired FFT at N=256, two samples a cycle, four accesses a
    bank a cycle: **II=3 without this pragma and II=1 with it**, same design,
    same source. That is the whole difference between a banked memory and a
    banked memory that works.

    Anything unmatched raises. `bind_recurrences` and `bind_fabric_arith` skip
    what they cannot find because a missing binding costs DSP blocks; a missing
    partition costs the interval, which is the thing the design exists to hold.
    """
    import re  # pylint: disable=import-outside-toplevel

    out, placed = [], set()
    for line in code.splitlines(True):
        match = re.match(r"(\s*)//\s*placeholder for .*?\b(_st_\w+)\b", line)
        if match and match.group(2) in banked and match.group(2) not in placed:
            local = match.group(2)
            out.append(
                f"{match.group(1)}#pragma HLS array_partition variable={local} "
                f"complete dim=1\n"
            )
            placed.add(local)
        out.append(line)
    missing = sorted(set(banked) - placed)
    if missing:
        raise SPMWMemoryError(
            f"banked memor{'y' if len(missing) == 1 else 'ies'} "
            f"{', '.join(missing)} could not be partitioned: no declaration was "
            f"found to attach the pragma to. Unpartitioned, the banks share one "
            f"memory's ports and the swizzle buys nothing, so this is refused "
            f"rather than emitted as a design that is banked in name only."
        )
    return "".join(out), sorted(placed)


__all__ = [
    "Directive",
    "PIPELINE",
    "accumulators",
    "bind_fabric_arith",
    "apply",
    "bind_recurrences",
    "interval",
    "partition_banks",
    "pipeline",
]
