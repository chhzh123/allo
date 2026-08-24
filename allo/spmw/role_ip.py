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

The cost argument for building units is scaling, not feasibility.  Whole-array
``csynth`` works and is cheaper below roughly 200 sites (280s at 144 sites
against 544s for nine roles); it grows superlinearly while the role count, and
so the per-role cost, stays flat.

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
from .lower_df import Lowering, _BodyRewriter, _wiring_classes
from .ports import IN, MEMORY, OUT
from .rtl import StructuralEmitter, _port_signals, _width


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
        port = self._port_of(node.value)
        if port is not None and port.protocol == MEMORY:
            return ast.Subscript(
                value=self._resident(port),
                slice=node.slice,
                ctx=node.ctx,
            )
        return super().visit_Subscript(node)

    def visit_Attribute(self, node):
        port = self._port_of(node)
        if port is not None and port.protocol == MEMORY:
            return self._resident(port)
        return super().visit_Attribute(node)

    def visit_Assign(self, node):
        """``io.c = acc`` writes to storage in the array, and out here."""
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

    def _body(self, placement, order, site):
        """The role's body rewritten for one concrete site."""
        signature, routing, sites = self.classes(placement)[order]
        body = placement.roles.get(site)
        if body is None:
            raise SPMWBindingError(
                f"`{placement.name}` is a placed fabric; a unit is defined for a "
                f"placed *unit*, so inline the sub-fabric first."
            )
        pids = [f"_p{k}" for k in range(len(placement.grid))]
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

    def _check_uniform(self, placement, order, source, pids):
        """A role stands for its whole class, so its sites must be alike.

        Rewriting a second site and comparing is a check rather than an
        assumption: a body that reads its own coordinates would otherwise be
        frozen to whichever site happened to be listed first, and every other
        instance would silently run the wrong one.
        """
        name = self.role_name(placement, order)
        _sig, _routing, sites = self.classes(placement)[order]
        if len(sites) > 1:
            other, _pids, _rw = self._body(placement, order, sites[1])
            if _unparse(other) != source:
                raise SPMWBindingError(
                    f"`{name}` covers {len(sites)} sites whose bodies differ, so "
                    f"one IP cannot stand for them: the body reads its own "
                    f"coordinates, which a unit would have to take as inputs."
                )
        # Coordinates reach a body two ways: through the pid names this
        # emitter substitutes, and through `site.rank`, which the base rewriter
        # turns into `df.get_pid()`. A single-instance kernel's pid is always
        # zero, so letting the second through would compile and silently run
        # every site as if it were the origin.
        for marker in list(pids) + ["df.get_pid()"]:
            if marker in source:
                raise SPMWBindingError(
                    f"`{name}` reads its own grid coordinates, which a "
                    f"standalone unit does not receive. The role is real -- its "
                    f"sites differ by position, not just by wiring -- so the "
                    f"unit would need them as inputs."
                )

    def _held(self, rewriter):
        """Declarations for the memory ports this unit reads once and holds.

        That is what ``stationary`` already means, and it keeps the unit
        free-running rather than re-reading a value the array kept in a
        register.
        """
        out = []
        for name, port in rewriter.residents.items():
            ann = ast.unparse(self.low.type_ann(port.dtype, port.shape))
            out.append(f"_st_{name}: {ann} = {name}[0].get()")
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
        body, pids, rewriter = self._body(placement, order, sites[0])
        plain = _unparse(body)
        self._check_uniform(placement, order, plain, pids)
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


def build_unit(graph, placement, order, target="vhls", keep=None, **kwargs):
    """Compile one role on its own.

    Imported the way :func:`allo.spmw.lower_df.build_dataflow` imports the array
    program -- from a real file, with the body's captured names in the module's
    namespace -- because the tracer reads the source back off disk.
    """
    import allo.dataflow as df  # pylint: disable=import-outside-toplevel

    emitter = UnitEmitter(graph)
    name = emitter.role_name(placement, order)
    src, extras = emitter.program(placement, order)
    namespace = dict(emitter.low.injected)
    namespace.update(extras)
    module = _import_program(f"_spmw_unit_{name}", src, namespace, keep)
    built = df.build(module.top, target=target, **kwargs)  # pylint: disable=no-member
    built.spmw_unit_source = src
    built.spmw_unit_name = name
    return built


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
    decls = re.findall(r"hls::stream<[^>]*>\s*(\w+);", code)
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
    "UnitEmitter",
    "build_unit",
    "check_wrapper",
    "unit_interface",
    "wrapper_sv",
]
