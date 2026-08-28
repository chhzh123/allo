# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One role as a standalone streaming kernel -- the thing HLS synthesises once.

:mod:`allo.spmw.rtl` builds the array; this builds the unit that goes in it.  A
role is emitted as its own single-kernel dataflow program in which *every* port,
including a memory port, is an ``hls::stream``, so the synthesised IP is
free-running and drops onto the fabric's FIFOs.

Turning a memory port into a stream is what lets the unit stand alone.  In the
array every site stores into the same result tensor, which HLS dataflow accepts
only because ``spmw.build`` completely partitions that tensor -- see
:func:`allo.spmw.driver.build`.  A unit has no such tensor to partition: its
result leaves on a port, which is also what makes the IP free-running.

The cost argument for building units is scaling and parallelism, not
feasibility.  Whole-array ``csynth`` works too, but it is one monolithic run --
39s at 9 sites, 280s at 144, 807s at 256 -- while roles are independent by
construction and so synthesise concurrently: nine of them in 40.5s of wall clock
whatever the grid.  Decomposing is what creates that parallelism.

A role stands for many sites, so a unit is only well defined when its sites are
interchangeable.  That is checked rather than assumed: the body is rewritten for
two different sites and refused if they disagree.
"""

import ast
import os
import re
import sys
import tempfile
import types

from .errors import SPMWBindingError
from .lower_df import Lowering, _BodyRewriter, _is_site_rank, _wiring_classes
from .ports import IN, MEMORY, OUT
from . import schedule as sched
from .abi import AXI_ADDR_WIDTH, CoordPort, _port_signals, _width, _WIDTH, axi_signals
from .rtl import StructuralEmitter


# The coordinate ports' name stem. `rtl.CoordPort` builds `_pid<k>`, and the
# unit holds each in `_st__pid<k>` -- the same `_st_` prefix a stationary weight
# gets, because it is the same thing: read once, then used.
_COORD = "_pid"


class _UnitRewriter(_BodyRewriter):
    """Rewrites a unit body against bare streams rather than channel arrays.

    In the array program a port touch becomes ``fam[i, j].get()``; here it
    becomes ``west[0].get()``, because the unit's ports are its own arguments.
    Everything else -- the arithmetic, the loops -- goes through the base class
    untouched, so the unit computes what the array's site computes.
    """

    def __init__(self, *args, connected=(), **kwargs):
        super().__init__(*args, **kwargs)
        # What the *site* connects, not what the signature holds: a port fed by
        # a loader is a real input to the unit but is absent from the signature,
        # which holds only peer links.
        self.connected = set(connected)
        # Memory ports the unit reads. In the array a site reads the parent's
        # tensor at its own coordinates -- `local_W[i, j]` -- which is per-site
        # *data*, not coordinate-dependent computation. A unit takes it on its
        # own port and holds it, which is what `stationary` already means.
        self.residents = {}
        # Which grid axes the body reads. A role that reads its position needs it
        # as an input, and the fabric must drive it per site.
        self.coords = set()

    def _bound(self, port):
        return port in self.connected

    def _subscript(self, port):
        if not self._bound(port):
            return None
        # A one-element array rather than a bare Stream: it is the declaration
        # shape the rest of the pipeline already handles.
        return ast.Subscript(
            value=ast.Name(id=port.name, ctx=ast.Load()),
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        )

    def _resident(self, port):
        """The local this unit holds a memory port's value in."""
        self.residents[port.name] = port
        return ast.Name(id=f"_st_{port.name}", ctx=ast.Load())

    def visit_Subscript(self, node):
        if self.site and _is_site_rank(node.value, self.site):
            if isinstance(node.slice, ast.Constant) and isinstance(
                node.slice.value, int
            ):
                self.coords.add(node.slice.value)
        port = self._port_of(node.value)
        if port is not None and port.protocol == MEMORY:
            return ast.Subscript(
                value=self._resident(port),
                slice=node.slice,
                ctx=node.ctx,
            )
        return super().visit_Subscript(node)

    def visit_Attribute(self, node):
        if (
            self.site
            and isinstance(node.value, ast.Name)
            and node.value.id == self.site
            and node.attr == "rank"
        ):
            self.coords.update(range(len(self.pids)))
        port = self._port_of(node)
        if port is not None and port.protocol == MEMORY:
            return self._resident(port)
        return super().visit_Attribute(node)

    def visit_Assign(self, node):
        """``io.c = acc`` writes to storage in the array, and out here."""
        # `s, b = site.rank` restates the coordinates. In the array they come
        # from df.get_pid(); a single-instance kernel's pid is always zero, so
        # here they come from the unit's own coordinate inputs.
        if self.site and _is_site_rank(node.value, self.site):
            self.coords.update(range(len(self.pids)))
            names = [ast.Name(id=p, ctx=ast.Load()) for p in self.pids]
            # Shape the value like the *target*, not like the grid: `(slot,) =
            # site.rank` on a 1-D placement unpacks a one-tuple, and handing it a
            # bare name emits `slot, = _st__pid0`, which is a scalar unpack.
            unpacking = len(node.targets) == 1 and isinstance(
                node.targets[0], (ast.Tuple, ast.List)
            )
            value = (
                ast.Tuple(elts=names, ctx=ast.Load())
                if unpacking or len(names) > 1
                else names[0]
            )
            return ast.Assign(targets=node.targets, value=value)
        target = node.targets[0] if len(node.targets) == 1 else None
        port = self._port_of(target) if target is not None else None
        if port is not None and port.protocol == MEMORY:
            value = self.visit(node.value)
            self.residents.pop(port.name, None)  # written, not held
            sub = self._subscript(port)
            if sub is None:
                raise SPMWBindingError(
                    f"`{self.placement.name}.{port.name}` is written but is not "
                    f"bound at this site, so the result has nowhere to go."
                )
            return ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(value=sub, attr="put", ctx=ast.Load()),
                    args=[value],
                    keywords=[],
                )
            )
        return super().visit_Assign(node)


class UnitEmitter:
    """Renders each role of a placement as its own dataflow program."""

    def __init__(self, graph):
        self.graph = graph
        self.low = Lowering(graph)
        self.struct = StructuralEmitter(graph)

    def placements(self):
        """The placements whose roles can be built as units."""
        return self.low.placements

    def classes(self, placement):
        """The wiring classes: one role each."""
        return _wiring_classes(placement, self.low.resolutions[placement])

    def role_name(self, placement, order):
        """The unit's name -- the same one the fabric instantiates."""
        return f"{self.low.kernel_names[placement]}_r{order}"

    def ports(self, placement, order):
        """The unit's ports, in the order its arguments take.

        Taken from the structural emitter so the IP and the fabric's stub agree
        by construction rather than by two orderings that happen to match.
        """
        return self.struct.unit_ports(placement, order)

    def body_for(self, placement, order, site):
        """The role's body rewritten for one concrete site.

        Public because the coordinate and accumulator analyses both need it:
        each is a question about the body, and the body is what this returns.
        """
        signature, routing, sites = self.classes(placement)[order]
        body = placement.roles.get(site)
        if body is None:
            raise SPMWBindingError(
                f"`{placement.name}` is a placed fabric; a unit is defined for a "
                f"placed *unit*, so inline the sub-fabric first."
            )
        pids = [f"_st_{_COORD}{k}" for k in range(len(placement.grid))]
        rewriter = _UnitRewriter(
            self.low,
            placement,
            signature,
            routing,
            [site] + [s for s in sites if s != site],
            pids,
            _io_name(placement, site),
            _site_name(placement, site),
            connected=[p for p, _f in self.struct.site_ports(placement, site)],
        )
        out = []
        for stmt in body.tree.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue  # a docstring
            visited = rewriter.visit(ast.fix_missing_locations(_copy(stmt)))
            if visited is not None:
                out.append(visited)
        return (out or [ast.Pass()]), pids, rewriter

    def _check_uniform(self, placement, order, source):
        """A role stands for its whole class, so its sites must be alike.

        Rewriting a second site and comparing is a check rather than an
        assumption: a body that reads its own coordinates would otherwise be
        frozen to whichever site happened to be listed first, and every other
        instance would silently run the wrong one.
        """
        name = self.role_name(placement, order)
        _sig, _routing, sites = self.classes(placement)[order]
        if len(sites) > 1:
            other, _pids, _rw = self.body_for(placement, order, sites[1])
            if _unparse(other) != source:
                raise SPMWBindingError(
                    f"`{name}` covers {len(sites)} sites whose bodies differ, so "
                    f"one IP cannot stand for them: the body reads its own "
                    f"coordinates, which a unit would have to take as inputs."
                )
        # `df.get_pid()` would be a silent wrong answer: a single-instance
        # kernel's pid is always zero, so the unit would run every site as if it
        # were the origin. Coordinates must arrive as inputs instead, and any
        # route that still reaches get_pid() has escaped that.
        if "df.get_pid()" in source:
            raise SPMWBindingError(
                f"`{name}` still reads df.get_pid(), which is zero in a "
                f"single-instance kernel. Its coordinates should have become "
                f"inputs; this is an emission bug."
            )

    def _held(self, rewriter):
        """Declarations for everything this unit reads once and holds.

        A stationary weight and a grid coordinate are the same shape of thing:
        per-site data that arrives on a port and does not change. Reading each
        once keeps the unit free-running rather than re-reading a value the array
        would have kept in a register.
        """
        out = []
        resident = {
            ast.unparse(decl.target): ast.unparse(decl)
            for decl in self.low.stationary_locals(rewriter.placement)
        }
        for name, port in rewriter.residents.items():
            local = f"_st_{name}"
            if local in resident:
                # Compile-time contents: the same ROM the array gives each site.
                out.append(resident[local])
                continue
            ann = ast.unparse(self.low.type_ann(port.dtype, port.shape))
            out.append(f"{local}: {ann} = {name}[0].get()")
        for axis in sorted(rewriter.coords):
            port = CoordPort(axis)
            ann = ast.unparse(self.low.type_ann(port.dtype, port.shape))
            out.append(f"_st_{_COORD}{axis}: {ann} = {_COORD}{axis}[0].get()")
        return out

    def _declarations(self, placement, order):
        """The region's stream declarations, and the types they need in scope."""
        decls, extras = [], {}
        for port, _fam in self.ports(placement, order):
            ann = self.low.type_ann(port.dtype, port.shape)
            decls.append(
                f"    {port.name}: Stream[{ast.unparse(ann)}, "
                f"{placement.depths.get(port, port.depth)}][1]"
            )
            extras[_base_name(ann)] = port.dtype
        return decls, extras

    def program(self, placement, order):
        """One role as a complete, self-contained dataflow program.

        Every port is a stream, memory ports included -- that is what makes the
        unit synthesisable. The whole-array program is not: each site stores into
        the same result tensor and HLS dataflow allows one writer per array.
        """
        _sig, _routing, sites = self.classes(placement)[order]
        name = self.role_name(placement, order)
        body, _pids, rewriter = self.body_for(placement, order, sites[0])
        plain = _unparse(body)
        self._check_uniform(placement, order, plain)
        source = "\n".join(self._held(rewriter) + [plain])
        decls, extras = self._declarations(placement, order)
        indented = "\n".join("        " + line for line in source.splitlines())
        text = (
            "import allo\n"
            "import allo.dataflow as df\n"
            "from allo.ir.types import Stream\n\n"
            "@df.region()\n"
            "def top():\n" + "\n".join(decls) + "\n\n"
            "    @df.kernel(mapping=[1])\n"
            f"    def {name}():\n{indented}\n"
        )
        return text, extras

    def programs(self):
        """Every role in the design, as `(name, source, extras)`."""
        out = []
        for placement in self.placements():
            for order in range(len(self.classes(placement))):
                text, extras = self.program(placement, order)
                out.append((self.role_name(placement, order), text, extras))
        return out


class _MoverRewriter(ast.NodeTransformer):
    """Rewrites a mover's body to talk to a bare stream and its own coordinates.

    A mover is the loader or drain a binding synthesises: it walks a tensor and
    moves elements to or from one channel of a family. In the array program it
    addresses `fam[member]` and gets its position from `df.get_pid()`; standing
    alone it addresses a port and takes its position as an input, exactly as a
    role does.
    """

    def __init__(self, family_name, pids):
        super().__init__()
        self.family = family_name
        self.pids = list(pids)
        self.coords = set()

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == self.family:
            # One channel, and this instance owns it.
            return ast.Subscript(
                value=ast.Name(id="chan", ctx=ast.Load()),
                slice=ast.Constant(value=0),
                ctx=node.ctx,
            )
        return node

    def visit_Assign(self, node):
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "get_pid"
        ):
            self.coords.update(range(len(self.pids)))
            names = [ast.Name(id=p, ctx=ast.Load()) for p in self.pids]
            unpacking = len(node.targets) == 1 and isinstance(
                node.targets[0], (ast.Tuple, ast.List)
            )
            if not names:
                # A mover whose bundle is a single site -- the head of a
                # distribution chain -- has no coordinate to be told: there is
                # one instance and its position is 0. It takes no `_pid` input
                # at all, which is what lets one AXI master feed a whole array.
                value = ast.Constant(value=0)
            elif unpacking or len(names) > 1:
                value = ast.Tuple(elts=names, ctx=ast.Load())
            else:
                value = names[0]
            return ast.Assign(targets=node.targets, value=value)
        self.generic_visit(node)
        return node


class MoverEmitter:
    """Renders each of a graph's movers as its own dataflow program.

    There is one mover per *binding*, not per site -- two for a plain mesh,
    three for the matched design, and the same number at 3x3 as at 16x16 -- so
    giving the fabric a memory interface costs a fixed number of extra HLS runs
    however large the array is.
    """

    def __init__(self, graph):
        self.graph = graph
        self.low = Lowering(graph)
        # The order `program` declares this mover's streams in, which is what
        # the call in `top` indexes -- the same positional recovery the unit
        # path uses, and the only thing that says which parameter is which.
        self._orders = {}

    def movers(self):
        return list(self.low.movers)

    def name(self, index):
        mover = self.low.movers[index]
        return f"{mover.name}_io"

    def family(self, index):
        """The boundary family this mover feeds or drains."""
        return self.low.movers[index].family

    def coords(self, index):
        """How many coordinate inputs this mover takes.

        A mover whose bundle is one site -- the head of a distribution chain --
        takes none, which is the whole point of chaining: one instance, one AXI
        master, however large the array behind it.
        """
        return len(self.low.movers[index].bundle.shape or ())

    def instances(self, index):
        """One instance per member of the bundle."""
        return max(1, len(self.low.movers[index].bundle.sites))

    def port_order(self, index):
        """This mover's stream ports, in the order `top` declares them."""
        if index not in self._orders:
            self.program(index)
        return list(self._orders[index])

    def stream_ports(self, index):
        """The mover's stream ports: their widths, and which way each runs.

        A loader *writes* the family the sites read, so the directions here are
        the mirror of the family's boundary direction.
        """
        # pylint: disable=import-outside-toplevel
        from .rtl import StructuralEmitter as _SE

        mover = self.low.movers[index]
        edge = _SE.boundary_direction(mover.family)
        widths = {"chan": _width(mover.family)}
        directions = {"chan": OUT if edge == IN else IN}
        for axis in range(self.coords(index)):
            port = CoordPort(axis)
            widths[port.name] = 32
            directions[port.name] = IN
        return widths, directions

    def axi_width(self, index, achieved=None):
        """The AXI master's data width.

        Vitis widens a burst up to ``max_widen_bitwidth`` when the accesses are
        contiguous, and how far it gets depends on the tensor as well as the
        request -- a 16-byte operand cannot reach 512 bits.  So the achieved
        width is read back off the exported netlist and passed in; the element
        width is only the fallback that lets the fabric elaborate beforehand.
        """
        if achieved is not None:
            return int(achieved)
        mover = self.low.movers[index]
        return _WIDTH[str(mover.tensor.dtype)]

    def program(self, index):
        """One mover as a self-contained program: a tensor in, a stream out."""
        mover = self.low.movers[index]
        kernel = self.low._mover_kernel(mover)
        # A mover's coordinates are its *bundle's*, not the placement's: the
        # west loader walks one column, so it has one index however many axes
        # the mesh has.
        rank = len(mover.bundle.shape or ())
        pids = [f"_st_{_COORD}{k}" for k in range(rank)]
        rewriter = _MoverRewriter(mover.family.name, pids)
        body = [rewriter.visit(stmt) for stmt in kernel.body]
        body = [b for b in body if b is not None] or [ast.Pass()]

        held = [
            f"_st_{_COORD}{axis}: int32 = {_COORD}{axis}[0].get()"
            for axis in sorted(rewriter.coords)
        ]
        source = "\n".join(held + [_unparse(body)])
        indented = "\n".join("        " + line for line in source.splitlines())

        fam = mover.family
        elem = ast.unparse(self.low.type_ann(fam.dtype, fam.block))
        tensor = mover.tensor
        tensor_ann = ast.unparse(self.low.type_ann(tensor.dtype, tensor.base.shape))
        decls = [f"    chan: Stream[{elem}, {fam.depth}][1]"]
        for axis in sorted(rewriter.coords):
            ann = ast.unparse(self.low.type_ann(CoordPort(axis).dtype, ()))
            decls.append(f"    {_COORD}{axis}: Stream[{ann}, 1][1]")
        name = self.name(index)
        self._orders[index] = ["chan"] + [
            f"{_COORD}{axis}" for axis in sorted(rewriter.coords)
        ]
        extras = {}
        coord = CoordPort(0)
        extras[_base_name(self.low.type_ann(coord.dtype, ()))] = coord.dtype
        extras[_base_name(self.low.type_ann(tensor.dtype, ()))] = tensor.dtype
        extras[_base_name(self.low.type_ann(fam.dtype, ()))] = fam.dtype

        text = (
            "import allo\n"
            "import allo.dataflow as df\n"
            "from allo.ir.types import Stream\n\n"
            "@df.region()\n"
            # The region argument and the kernel parameter must differ: Allo
            # rejects a kernel parameter that shadows a region symbol, which is
            # why the array path names them `A` and `local_A`.
            f"def top({tensor.name}: {tensor_ann}):\n" + "\n".join(decls) + "\n\n"
            f"    @df.kernel(mapping=[1], args=[{tensor.name}])\n"
            f"    def {name}(local_{tensor.name}: {tensor_ann}):\n{indented}\n"
        )
        return text, extras


def build_mover(graph, index, target="vhls", keep=None, **kwargs):
    """Compile one mover on its own: the loader or drain a binding asks for.

    This is what gives the structural fabric a memory interface. The tensor
    argument becomes an AXI master at the top of the synthesised IP, so the
    array can be fed from DRAM rather than from a testbench holding its edge
    streams.
    """
    import allo.dataflow as df  # pylint: disable=import-outside-toplevel

    emitter = MoverEmitter(graph)
    name = emitter.name(index)
    src, extras = emitter.program(index)
    namespace = dict(emitter.low.injected)
    namespace.update(extras)
    module = _import_program(f"_spmw_mover_{name}", src, namespace, keep)
    built = df.build(module.top, target=target, **kwargs)  # pylint: disable=no-member
    built.spmw_mover_source = src
    built.spmw_mover_name = name
    return built


def build_unit(graph, placement, order, target="vhls", keep=None, ii=None, **kwargs):
    """Compile one role on its own.

    Imported the way :func:`allo.spmw.lower_df.build_dataflow` imports the array
    program -- from a real file, with the body's captured names in the module's
    namespace -- because the tracer reads the source back off disk.

    The unit's loops are pipelined before it is built. Without that Vitis
    schedules a PE's inner loop sequentially, which is a spatial design's entire
    inner loop; see :mod:`allo.spmw.schedule`.
    """
    import allo.dataflow as df  # pylint: disable=import-outside-toplevel

    emitter = UnitEmitter(graph)
    name = emitter.role_name(placement, order)
    src, extras = emitter.program(placement, order)
    namespace = dict(emitter.low.injected)
    namespace.update(extras)
    module = _import_program(f"_spmw_unit_{name}", src, namespace, keep)
    # The requested interval is a *recurrence budget*: it buys interval by
    # spending combinational delay on the accumulator's adder. A unit that
    # carries nothing has no recurrence to trade against, so peak is free and
    # asking for a wider interval would only slow it down -- which is exactly
    # what forcing ii=4 on attention did, taking it from II=1 to II=4.
    carried = _accumulators(emitter, placement, order)
    want = sched.interval(placement, default=0) if ii is None else ii
    if not carried and want:
        want = 1
    schedule = df.customize(module.top)  # pylint: disable=no-member
    pipelined = sched.apply(schedule, [name, f"{name}_0"], want)
    built = schedule.build(target=target, **kwargs)
    built.spmw_unit_source = src
    built.spmw_unit_name = name
    built.spmw_pipelined = pipelined
    built.spmw_accumulators = carried
    built.spmw_interval = want
    return built


def _accumulators(emitter, placement, order):
    """The unit body's loop-carried names, from its own source."""
    _sig, _routing, sites = emitter.classes(placement)[order]
    body, _pids, _rw = emitter.body_for(placement, order, sites[0])
    tree = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    return sched.accumulators(tree)


def optimise(code, built):
    """Apply the schedule's recurrence budget to generated HLS C++.

    A unit that asks for II=N is asking for its accumulator's adder to fit in
    N-1 cycles, because a recurrence through a float add costs II = latency + 1.
    Vitis picks a deeply pipelined adder by default, which is why the systolic
    GEMM's PE sits at II=7 while the mini-TPU's -- whose partial sum arrives on a
    *stream*, so nothing is carried -- is already at 1.

    Returns the code and the values bound; without an interval it is a no-op.
    """
    interval_ = getattr(built, "spmw_interval", 0)
    names = getattr(built, "spmw_accumulators", ())
    if not interval_ or not names:
        return code, []
    return sched.bind_recurrences(code, names, max(interval_ - 1, 0))


def _import_program(stem, src, namespace, keep):
    """Import generated source from a real file, with the body's names in scope.

    A file on disk rather than a string, because the tracer reads the body back
    with ``inspect.getsourcelines`` rather than from the code object -- the same
    reason :func:`allo.spmw.lower_df.build_dataflow` does it this way.
    """
    seq = _MODULE_SEQ[0] = _MODULE_SEQ[0] + 1
    modname = f"{stem}_{seq}"
    directory = keep or tempfile.mkdtemp(prefix="spmw_unit_")
    path = os.path.join(directory, f"{modname}.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(src)
    module = types.ModuleType(modname)
    module.__file__ = path
    module.__dict__.update(namespace)
    sys.modules[modname] = module
    exec(compile(src, path, "exec"), module.__dict__)  # pylint: disable=exec-used
    return module


_MODULE_SEQ = [0]


def _base_name(node):
    """The root name of a type annotation: `float32[2]` -> `float32`."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id


_COORD_CACHE = {}


def coord_axes(graph, placement, order):
    """Which grid axes this role's body reads, as a sorted tuple.

    A fact about the body, so it is answered by rewriting the body -- the same
    rewrite the unit is emitted from, so the ports the fabric drives and the
    ports the unit declares cannot disagree.

    Memoised because the fabric asks once per site.
    """
    key = (id(graph), id(placement), order)
    if key not in _COORD_CACHE:
        emitter = UnitEmitter(graph)
        _sig, _routing, sites = emitter.classes(placement)[order]
        _body, _pids, rewriter = emitter.body_for(placement, order, sites[0])
        _COORD_CACHE[key] = tuple(sorted(rewriter.coords))
    return _COORD_CACHE[key]


def _unparse(body):
    """Unparse a rewritten body; constructed nodes carry no source locations."""
    return ast.unparse(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    )


def _copy(node):
    return ast.parse(ast.unparse(ast.Module(body=[node], type_ignores=[]))).body[0]


def _io_name(placement, site):
    args = placement.roles[site].tree.args.args
    return args[0].arg if args else "io"


def _site_name(placement, site):
    args = placement.roles[site].tree.args.args
    return args[1].arg if len(args) > 1 else None


# What each header of Allo's fixed preamble is actually needed for. Parsing the
# whole block costs ~23s of every csynth -- against ~0.5s of scheduling, binding
# and RTL for one PE -- so a unit's synthesis time is almost entirely header
# parsing, and the nine-role total is mostly that cost paid nine times.
#
# Patterns, not substrings: `hls::` alone matches `hls::stream` and would keep
# `hls_math.h` in every design, and a bare `exp` or `log` matches any identifier
# containing them.
_HEADER_USES = {
    "<algorithm>": r"\bstd::",
    "<ap_axi_sdata.h>": r"\bap_axi",
    "<ap_fixed.h>": r"\bap_u?fixed\b",
    "<ap_int.h>": r"\bap_u?int<",
    "<hls_math.h>": r"\bhls::(?!stream|vector)\w",
    "<hls_stream.h>": r"\bhls::stream\b",
    "<hls_vector.h>": r"\bhls::vector\b",
    "<math.h>": r"\b(?:sqrtf?|expf?|logf?|sinf?|cosf?|tanf?|tanhf?|fabsf?|powf?|"
    r"floorf?|ceilf?)\s*\(",
    "<stdint.h>": r"\b(?:u?int(?:8|16|32|64)_t)\b",
}


def trim_includes(code):
    """Drop the headers this unit does not use.

    Measured on one role: 31.6s to 8.8s of ``csynth_design``, with "Source Code
    Analysis and Preprocessing" falling from 23.2s to 0.6s. Nothing about the
    hardware changes -- scheduling saw the same 0.5s of work either way.

    A header is kept when the body matches any pattern it provides, which errs
    toward keeping: an unnecessary header costs seconds, a missing one costs the
    build.
    """
    lines = code.splitlines(True)
    body = "".join(line for line in lines if not line.startswith("#include"))
    out = []
    for line in lines:
        if not line.startswith("#include"):
            out.append(line)
            continue
        header = line.split(None, 1)[1].strip()
        pattern = _HEADER_USES.get(header)
        if pattern is None or re.search(pattern, body):
            out.append(line)
    return "".join(out)


def unit_interface(code, name, ports):
    """Map the synthesised unit's parameters back to port names.

    Allo renames every value, so the generated unit is ``pe_r2_0(v0, v1, v2,
    v3)`` and nothing in it says which parameter is ``west``.  The mapping is
    recovered from the program's own structure -- the region declares the
    streams in the order this module emitted them, and the call in ``top`` says
    which declaration reaches which parameter.

    That is a positional argument, so it is *checked* against a second,
    independent signal: whether the body reads or writes each parameter has to
    agree with the port's declared direction.  A mapping that silently slipped
    would wire a reader to a writer's FIFO, which is the kind of error that
    produces wrong numbers rather than a failure.

    Returns ``[(param, port), ...]`` in parameter order.
    """
    kernel = re.search(rf"^void ({re.escape(name)}\w*)\(([^)]*)\)", code, re.M)
    if kernel is None:
        raise SPMWBindingError(
            f"the synthesised unit has no function named `{name}`; the emitter "
            f"and the backend disagree about the kernel's name."
        )
    params = re.findall(r"&\s*(\w+)", kernel.group(2))
    if len(params) != len(ports):
        raise SPMWBindingError(
            f"`{name}` was synthesised with {len(params)} stream parameters but "
            f"the role has {len(ports)} ports."
        )
    # A block-carrying port is `hls::stream< hls::vector< float, 2 > >`, so the
    # element type nests -- one level of angle brackets has to be allowed for.
    decls = re.findall(r"hls::stream<(?:[^<>]|<[^<>]*>)*>\s*(\w+);", code)
    call = re.search(rf"{re.escape(kernel.group(1))}\(([^)]*)\);", code)
    if call is None or len(decls) != len(ports):
        raise SPMWBindingError(
            f"cannot read `{name}`'s wiring out of the generated program."
        )
    order = [decls.index(a.strip()) for a in call.group(1).split(",")]
    body = code[kernel.end() :]
    body = body[: body.index("\n}")]
    mapping = []
    for pos, param in enumerate(params):
        port = ports[order[pos]][0]
        _confirm_direction(name, param, port, body)
        mapping.append((param, port))
    return mapping


def _confirm_direction(name, param, port, body):
    """Cross-check a positional mapping against how the body uses the parameter."""
    reads = f"{param}.read()" in body
    writes = f"{param}.write(" in body
    if reads == writes:
        raise SPMWBindingError(
            f"`{name}` parameter {param} is neither only read nor only written, "
            f"so its direction cannot be confirmed."
        )
    want = IN if reads else OUT
    if port.direction != want and port.protocol != MEMORY:
        raise SPMWBindingError(
            f"`{name}` parameter {param} maps to `{port.name}`, declared "
            f"{port.direction}, but the body {'reads' if reads else 'writes'} it. "
            f"The parameter mapping is wrong."
        )


def _rename(mapping, widths):
    """The wrapper's own ports, and the connections onto the IP's names."""
    # A free-running (ap_ctrl_none) Vitis HLS IP presents an active-high
    # synchronous `ap_rst`, while the fabric carries `ap_rst_n`; `check_wrapper`
    # confirms this against the exported netlist rather than trusting it.
    decls, conns = [], [".ap_clk(ap_clk)", ".ap_rst(~ap_rst_n)"]
    for param, port in mapping:
        for sig, kind, width in _port_signals(
            port.name, port.direction, widths[port.name]
        ):
            span = f"[{width - 1}:0] " if width > 1 else ""
            decls.append(
                f"  {'input ' if kind == 'input' else 'output'} wire {span}{sig}"
            )
            conns.append(f".{param}_{sig[len(port.name) + 1:]}({sig})")
    return decls, conns


def mover_interface(code, name, ports=None):
    """Map a synthesised mover's parameters back to what they carry.

    A mover has one parameter the others do not: the tensor, passed as an array
    rather than a stream reference, which the ``m_axi`` directive turns into an
    AXI master.  The streams are recovered exactly as a unit's are -- from the
    region's declaration order and the call in ``top``.

    Returns ``(tensor_param, [(param, port_name), ...])``.
    """
    kernel = re.search(rf"^void ({re.escape(name)}\w*)\(([^)]*)\)", code, re.M)
    if kernel is None:
        raise SPMWBindingError(
            f"the synthesised mover has no function named `{name}`; the emitter "
            f"and the backend disagree about the kernel's name."
        )
    # Matched rather than split on commas: a block-carrying stream is
    # `hls::stream< hls::vector< int8_t, 4 > >`, whose own comma is not a
    # parameter separator.
    signature = kernel.group(2)
    streams = re.findall(r"&\s*(\w+)", signature)
    tensors = re.findall(r"(\w+)\s*(?:\[\s*\d+\s*\])+", signature)
    if len(tensors) != 1:
        raise SPMWBindingError(
            f"`{name}` was synthesised with {len(tensors)} array parameters; a "
            f"mover walks exactly one tensor."
        )
    decls = re.findall(r"hls::stream<(?:[^<>]|<[^<>]*>)*>\s*(\w+);", code)
    call = re.search(rf"{re.escape(kernel.group(1))}\(([^)]*)\);", code)
    if call is None:
        raise SPMWBindingError(
            f"cannot read `{name}`'s wiring out of the generated program."
        )
    args = [a.strip() for a in call.group(1).split(",")]
    order = [decls.index(a) for a in args if a in decls]
    if len(order) != len(streams):
        raise SPMWBindingError(
            f"`{name}` takes {len(streams)} streams but `top` passes "
            f"{len(order)} of its {len(decls)} declarations."
        )
    # Allo renames the region's declarations too, so `decls` holds `v9`, not
    # `chan`. The names come from the emitter that wrote the program.
    named = list(ports) if ports is not None else decls
    if len(named) != len(decls):
        raise SPMWBindingError(
            f"`{name}` declares {len(decls)} streams but the emitter named "
            f"{len(named)}."
        )
    return tensors[0], [(param, named[order[i]]) for i, param in enumerate(streams)]


def mover_wrapper_sv(
    graph, index, code, bundle="gmem", ip_suffix="_0", data_width=None
):
    """A shim giving a synthesised mover the fabric's names, AXI included.

    Unlike a role, a mover is not free-running: it reads or writes a whole
    tensor and then stops, so it keeps ``ap_ctrl_hs`` and the fabric starts it.
    Its tensor argument becomes an AXI master plus a base-address input, and
    both are forwarded whole -- this is the port through which the array reaches
    DRAM.
    """
    emitter = MoverEmitter(graph)
    name = emitter.name(index)
    tensor, streams = mover_interface(code, name, emitter.port_order(index))
    widths, directions = emitter.stream_ports(index)
    decls = [
        "  input  wire ap_clk",
        "  input  wire ap_rst_n",
        "  input  wire ap_start",
        "  output wire ap_done",
        "  output wire ap_idle",
        "  output wire ap_ready",
        f"  input  wire [{AXI_ADDR_WIDTH - 1}:0] offset",
    ]
    conns = [
        ".ap_clk(ap_clk)",
        ".ap_rst_n(ap_rst_n)",
        ".ap_start(ap_start)",
        ".ap_done(ap_done)",
        ".ap_idle(ap_idle)",
        ".ap_ready(ap_ready)",
        f".{tensor}(offset)",
    ]
    for sig, kind, width in axi_signals(bundle, emitter.axi_width(index, data_width)):
        span = f"[{width - 1}:0] " if width > 1 else ""
        decls.append(f"  {'input ' if kind == 'input' else 'output'} wire {span}{sig}")
        conns.append(f".{sig}({sig})")
    for param, port in streams:
        for sig, kind, width in _port_signals(port, directions[port], widths[port]):
            span = f"[{width - 1}:0] " if width > 1 else ""
            decls.append(
                f"  {'input ' if kind == 'input' else 'output'} wire {span}{sig}"
            )
            conns.append(f".{param}_{sig[len(port) + 1:]}({sig})")
    return (
        "`timescale 1ns/1ps\n\n"
        + f"module {name} (\n"
        + ",\n".join(decls)
        + "\n);\n"
        + f"  {name}{ip_suffix} u (\n      "
        + ",\n      ".join(conns)
        + ");\nendmodule\n"
    )


def wrapper_sv(graph, placement, order, code, ip_suffix="_0"):
    """A SystemVerilog shim giving the synthesised IP the fabric's port names.

    The fabric instantiates ``pe_r2`` with ports called ``west_dout``; HLS
    exports ``pe_r2_0`` with ports called ``v0_dout``.  Rather than teach the
    fabric the backend's names, the rename is written down once, here, where it
    can be read.
    """
    emitter = UnitEmitter(graph)
    name = emitter.role_name(placement, order)
    ports = emitter.ports(placement, order)
    widths = {p.name: _width(f) for p, f in ports}
    decls, conns = _rename(unit_interface(code, name, ports), widths)
    head = ["  input  wire ap_clk", "  input  wire ap_rst_n"] + decls
    return (
        "`timescale 1ns/1ps\n\n"
        + f"module {name} (\n"
        + ",\n".join(head)
        + "\n);\n"
        + f"  {name}{ip_suffix} u (\n      "
        + ",\n      ".join(conns)
        + ");\nendmodule\n"
    )


def axi_data_width(exported, bundle="gmem"):
    """How wide an AXI master Vitis actually built, from the exported netlist.

    Requested widening is a ceiling, not a promise: a 16-byte operand cannot
    reach 512 bits however wide the request.  Reading the achieved width back is
    what lets the wrapper declare the ports the IP really has.
    """
    found = re.search(
        rf"parameter\s+C_M_AXI_{bundle.upper()}_DATA_WIDTH\s*=\s*(\d+)", exported
    )
    if found is None:
        raise SPMWBindingError(
            f"the exported netlist declares no `C_M_AXI_{bundle.upper()}_DATA_"
            f"WIDTH`; it may not have an AXI master at all."
        )
    return int(found.group(1))


def check_wrapper(wrapper, exported):
    """Confirm the wrapper connects ports the exported IP actually has.

    The wrapper is written from the *generated C++*, before synthesis; the IP's
    real port list only exists afterwards.  Comparing them closes that gap --
    a port the wrapper invents would otherwise surface as an elaboration error
    deep in a Vivado run, or, for a reset polarity, not at all.

    Returns the number of connections checked.
    """
    module = re.search(r"module\s+(\w+)\s*\(([^)]*)\);", exported)
    if module is None:
        raise SPMWBindingError("the exported netlist declares no module")
    have = {p.strip() for p in module.group(2).split(",") if p.strip()}
    want = set(re.findall(r"\.(\w+)\(", wrapper.split("u (", 1)[-1]))
    missing = sorted(want - have)
    if missing:
        raise SPMWBindingError(
            f"the wrapper connects {missing} on `{module.group(1)}`, which the "
            f"exported IP does not have; it has {sorted(have)}."
        )
    return len(want)


__all__ = [
    "MoverEmitter",
    "axi_data_width",
    "mover_interface",
    "mover_wrapper_sv",
    "UnitEmitter",
    "build_mover",
    "optimise",
    "coord_axes",
    "trim_includes",
    "build_unit",
    "check_wrapper",
    "unit_interface",
    "wrapper_sv",
]
