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

``pipeline(P, ii=1, registered_links=True)`` asks for no pipeline *depth* the
links already provide.  Vitis charges each FIFO access 1.2-1.4 ns, but an SPMW
link is a register slice -- a flop on either side of the unit -- so left alone
it registers a multiply-add cell three times over on top of its links: a
systolic cell that is its own pipeline and then the array's again.  A stage
that begins at a link read or ends at a link write is credited the access it
touches, and the unit keeps only the registers its own logic needs.

``combinational=True`` says more: the body fits one clock, so its one stage
touches *both* links and is credited both -- Gemmini's ``tile_latency = 0``.

HLS schedules every stage against one clock, so a credit is only right if
*every* stage touches that many links: two for a one-stage body, one for two
stages, and none once a middle stage touches neither.  Over-crediting packs a
multiply and two adder levels into one stage.  On E3's 16x16 microbenchmark: a
4x4 block of cells closed at 260 MHz credited both and 333 MHz credited one;
8x8 and 16x16 blocks, three stages deep, at 258 and 262 MHz credited one and
322 and 311 credited none.

``ram(P, "buf")`` keeps an array a unit declares as the memory it is.  Allo
zeroes a declared array with a loop ahead of the body, and in hardware that
loop is cycles after reset in which the unit takes no token: 2,048 of them for
a 2,048-entry accumulator, while its neighbours on bare-register links have
nowhere to put theirs.  ``distance=n`` adds what a read-modify-write needs to
hold II=1: that an element written is not read again for ``n`` iterations.
"""

from .errors import SPMWMemoryError, SPMWPlacementError

PIPELINE = "pipeline"
LINK_CREDITS = "link_credits"
RAM = "ram"
#: What bounds a loop body at one state in generated HLS C++.
ONE_STAGE = "#pragma HLS latency max=0"


class Directive:
    """One scheduling request against a placement's unit."""

    __slots__ = ("kind", "value")

    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"<{self.kind} {self.value}>"


def pipeline(target, ii=1, registered_links=False, combinational=False):
    """Pipeline the unit's loops at initiation interval ``ii``.

    ``target`` is a placement -- what :func:`allo.spmw.place` returned.  ``ii=0``
    leaves the loops alone, which is worth having to measure what the pipelining
    is buying.  ``registered_links=True`` schedules each stage that touches a
    link as if the access were free, because the link is a register;
    ``combinational=True`` asserts the body fits one clock between its links,
    and credits that one stage both.  Place and route at the real clock is the
    check either way.
    """
    if not hasattr(target, "schedule"):
        raise SPMWPlacementError(
            f"pipeline() applies to a placement; got {type(target).__name__}."
        )
    if not isinstance(ii, int) or ii < 0:
        raise SPMWPlacementError(
            f"pipeline(ii=) must be a non-negative int, got {ii!r}"
        )
    given = 2 if combinational else (1 if registered_links else 0)
    if given and not ii:
        raise SPMWPlacementError(
            "pipeline(registered_links=/combinational=) needs a pipelined loop, "
            "got ii=0"
        )
    target.schedule = [
        d for d in target.schedule if d.kind not in (PIPELINE, LINK_CREDITS)
    ]
    target.schedule.append(Directive(PIPELINE, ii))
    if given:
        target.schedule.append(Directive(LINK_CREDITS, given))
    return target


def ram(target, name, distance=None):
    """Keep the unit's local array ``name`` as a RAM.

    ``target`` is a placement and ``name`` an array its unit declares.  The
    array then starts as zeros because it was configured that way, not because
    a loop filled it: the unit takes its first token the cycle after reset, and
    a second run finds what the first one left.

    ``distance`` is a promise about the unit's own addressing: an element
    written in one iteration is not read again for at least that many.  A
    read-modify-write is a recurrence through the memory -- the read, the
    arithmetic and the write of one iteration before the read of the next --
    unless the tool is told how far apart two visits to one element are, and
    with that it may take ``distance`` stages over them at II=1.
    """
    if not hasattr(target, "schedule"):
        raise SPMWPlacementError(
            f"ram() applies to a placement; got {type(target).__name__}."
        )
    if not isinstance(name, str) or not name.isidentifier():
        raise SPMWPlacementError(
            f"ram() names an array of the unit's body, got {name!r}"
        )
    if distance is not None and (
        isinstance(distance, bool) or not isinstance(distance, int) or distance < 1
    ):
        raise SPMWPlacementError(
            f"ram(distance=) must be a positive int, got {distance!r}"
        )
    target.schedule = [
        d for d in target.schedule if not (d.kind == RAM and d.value[0] == name)
    ]
    target.schedule.append(Directive(RAM, (name, distance)))
    return target


def rams(placement):
    """The arrays this placement's unit keeps as RAMs, each with its distance."""
    return {
        d.value[0]: d.value[1]
        for d in getattr(placement, "schedule", ())
        if d.kind == RAM
    }


def link_credits(placement):
    """How many link accesses each of the unit's stages is credited: 0, 1, 2."""
    for directive in getattr(placement, "schedule", ()):
        if directive.kind == LINK_CREDITS:
            return directive.value
    return 0


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
        match = re.match(r"(\s*)float\s+(v\d+)\s*=\s*[^;]*?([-+])\s*v\d+\s*;", line)
        if match:
            indent, value, op = match.group(1), match.group(2), match.group(3)
            out.append(
                f"{indent}#pragma HLS bind_op variable={value} "
                f"op={'fadd' if op == '+' else 'fsub'} impl=fabric\n"
            )
            bound.append(value)
    return "".join(out), bound


def pipeline_whiles(code, ii, one_stage=False):
    """Pipeline every ``while`` loop of generated HLS C++ at interval ``ii``.

    `apply` reaches a loop through the schedule's bands, and a ``while`` has
    none. A unit that stops on a token rather than on a count -- a cell that
    runs until its stream's last beat -- is one ``while``, emitted as
    ``while (true) {`` with the exit test first, and the pragma goes at the top
    of that body, where Vitis reads it.

    ``one_stage`` holds the body to a single state, which is what
    ``combinational=True`` asserts and what a unit on bare-register links
    needs. A loop that stops on a token carries a recurrence from the read of
    that token to its own exit test, so Vitis schedules that one read in the
    first state and, left alone, everything else in the second -- whatever the
    clock. The unit then holds its other links' values a cycle longer than its
    neighbour takes to send the next, and a bare register has no way to say so.

    A body left to take several states is made a *flushing* pipeline. Vitis's
    default pipeline advances only while its first stage does, and a loop that
    ends on a token ends with its last iterations still in flight: the unit
    restarts, waits in its first stage for a token that never comes, and the
    rows behind it never leave. Each stage of a flushing pipeline moves on its
    own.

    Returns the rewritten code and how many loops it pipelined.
    """
    import re  # pylint: disable=import-outside-toplevel

    out, count = [], 0
    for line in code.split("\n"):
        out.append(line)
        opened = re.match(r"(\s*)while \(true\) \{", line)
        if opened:
            pad = opened.group(1)
            if one_stage:
                out += [f"{pad}  #pragma HLS pipeline II={ii}", f"{pad}  {ONE_STAGE}"]
            else:
                out.append(f"{pad}  #pragma HLS pipeline II={ii} style=flp")
            count += 1
    return "\n".join(out), count


def hold_rams(code, held):
    """Make each array of ``held`` a RAM in generated HLS C++.

    Allo declares an array and zeroes it in a loop, ``int32_t buf[2048];`` and
    three lines after it.  The loop goes and the declaration becomes ``static``
    with its initial value, which is how Vitis is told that a memory's contents
    are part of the configuration.  A ``distance`` becomes a ``dependence``
    pragma in the body of every ``while``, where the accesses are.

    An array that is not found raises, as an unpartitioned bank does: what is
    lost is not speed.  The fill loop of an accumulator runs for thousands of
    cycles after reset, and a unit that has not started loses its neighbours'
    tokens on bare-register links.
    """
    import re  # pylint: disable=import-outside-toplevel

    missing = []
    for name, distance in held.items():
        fill = re.compile(
            rf"^(?P<pad>[ \t]*)(?P<type>\w+) {re.escape(name)}(?P<dims>\[\d+\]);"
            rf"(?P<note>[^\n]*)\n"
            rf"[ \t]*for \(int (?P<i>v\d+) = 0; (?P=i) < \d+; (?P=i)\+\+\) \{{[^\n]*\n"
            rf"[ \t]*{re.escape(name)}\[(?P=i)\] = 0;[^\n]*\n"
            rf"[ \t]*\}}[^\n]*\n",
            re.M,
        )
        code, count = fill.subn(
            rf"\g<pad>static \g<type> {name}\g<dims> = {{0}};\g<note>\n", code
        )
        if count != 1:
            missing.append(name)
        elif distance:
            code = re.sub(
                r"^([ \t]*)while \(true\) \{[^\n]*\n",
                lambda m, n=name, d=distance: (
                    f"{m.group(0)}{m.group(1)}  #pragma HLS dependence "
                    f"variable={n} type=inter direction=RAW distance={d} true\n"
                ),
                code,
                flags=re.M,
            )
    if missing:
        raise SPMWMemoryError(
            f"`ram()` names {', '.join(sorted(missing))}, and the unit declares no "
            f"such array for a loop to fill. It would be built as it was, a loop "
            f"of cycles ahead of its first token, so this is refused."
        )
    return code


def iteration_latency(report):
    """The deepest iteration latency among the loops of a ``csynth`` report.

    One means every loop body is a single state. None means the report lists
    no loop.
    """
    import re  # pylint: disable=import-outside-toplevel

    rows = re.findall(r"^\s*\|\s*[-+]+\s*[^|]*\|[^|]*\|[^|]*\|\s*(\d+)\|", report, re.M)
    return max(int(row) for row in rows) if rows else None


def bind_fabric_mul(code):
    """Bind every integer multiply in generated HLS C++ to fabric.

    `bind_fabric_arith` moves float *adds* out of DSP blocks because Vitis puts
    them there by default. This is the opposite direction and it exists for a
    comparison rather than for area: when the baseline implements its
    multiplier in lookup tables, a port that lets Vivado infer a DSP is not
    being measured against it.

    FEATHER is the case. Its published RTL routes with **zero** DSPs at every
    size -- a plain `*` in Verilog that Vivado maps to fabric -- while the SPMW
    port's identical multiply is inferred into one DSP per element: 16, 64 and
    256. The lookup-table columns then sit beside each other as though they
    meant the same thing, and they do not, because one of the two designs has
    moved its arithmetic off the fabric being counted. This pragma puts it
    back, so both designs spend the same kind of resource and the LUT column
    becomes a comparison.

    It is not free and is not meant to be: an `int8 x int8` multiply in fabric
    is lookup tables the DSP was doing for nothing, so the port's LUT count
    goes *up*. That is the honest number.

    Matching mirrors `bind_fabric_arith`: Allo emits one operation per
    statement, ``int64_t v30 = v28 * v29;``, with a ``// L33`` provenance
    comment after the semicolon, so the pattern stops at the semicolon. Only
    integer types are matched -- a float multiply is what a DSP is for, and
    the FFT designs depend on keeping theirs there.

    Returns the rewritten code and the values bound.
    """
    import re  # pylint: disable=import-outside-toplevel

    out, bound = [], []
    for line in code.splitlines(True):
        out.append(line)
        match = re.match(
            r"(\s*)(?:ap_u?int<\d+>|u?int\d+_t|unsigned\s+\w+|int|long)\s+"
            r"(v\d+)\s*=\s*v\d+\s*\*\s*v\d+\s*;",
            line,
        )
        if match:
            indent, value = match.group(1), match.group(2)
            out.append(
                f"{indent}#pragma HLS bind_op variable={value} op=mul impl=fabric\n"
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
    "hold_rams",
    "interval",
    "partition_banks",
    "pipeline",
    "pipeline_whiles",
    "ram",
    "rams",
    "iteration_latency",
    "ONE_STAGE",
]
