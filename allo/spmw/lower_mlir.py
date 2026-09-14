# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emitting an elaborated SPMW graph as MLIR, directly.

The structure of a spatial design -- its grid, the channel families over it,
how each port addresses them, and which body runs where -- becomes one
``spmw.map`` op per placement, and one per loader or drain.  Each body becomes a
``func.func`` built from the unit's own AST through Allo's IR builder, with the
site's coordinates as ``index`` parameters and its ports as ``!allo.stream``
parameters.  Nothing is rendered as Python source, written to disk, executed or
re-parsed: the graph goes to the IR in one step.

The module is kept in two forms:

* the **rolled** form, which is the IR -- one body per role, nine for a mesh at
  any size, with the instantiation carried as attributes on the map;
* the **expanded** form the backends consume, in which each map is replaced by
  one ``func.call`` per site and each channel by an ``allo.stream_construct``.
  The bodies are the same functions; only the instantiation is spelled out.

The frontend already computes every table -- site signatures, per-site routing,
family shapes -- so this is a transcription rather than an analysis, and the
dialect's verifier is a check on it rather than the source of it.
"""

import ast
import builtins

from . import channels as ch
from .component import _io_param_name
from .errors import SPMWBindingError
from .lower_df import (
    Lowering,
    _BodyRewriter,
    _copy,
    _fill_empty_suites,
    _geometry,
    _is_docstring,
    _site_param_name,
    _wiring_classes,
    pid_type,
)
from .ports import OUT, STREAM

#: The name of the function holding the maps, and of the expanded program's top.
TOP = "top"


# ---------------------------------------------------------------------------
# Spelling types and tables as MLIR text
# ---------------------------------------------------------------------------


def _mlir_type(dtype, shape=()):
    """The MLIR spelling of a declared type.

    Allo's scalar types already print as their MLIR names -- ``f32``, ``i8`` --
    so a scalar needs no translation and a block becomes a memref of them.
    """
    base = str(dtype)
    if not shape:
        return base
    extents = "x".join(str(int(s)) for s in shape)
    return f"memref<{extents}x{base}>"


def _stream_type(family):
    return f"!allo.stream<{_mlir_type(family.dtype, family.block)}, {family.depth}>"


def _ints(values):
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def _strs(values):
    return "[" + ", ".join(f'"{v}"' for v in values) + "]"


def _dense(values, elem="i32"):
    """A per-site table, flat and row-major.

    Kept one-dimensional rather than shaped like the grid: the reader indexes it
    by a linearised coordinate anyway, and a flat literal needs no nesting.
    """
    values = list(values)
    body = ", ".join(str(int(v)) for v in values)
    return f"dense<[{body}]> : tensor<{len(values)}x{elem}>"


def _family_attr(fam):
    return (
        f'#spmw.family<name = "{fam.name}", type = '
        f"{_mlir_type(fam.dtype, fam.block)}, block = {_ints(fam.block)}, "
        f"depth = {fam.depth}, shape = {_ints(fam.shape)}>"
    )


def _volume(shape):
    n = 1
    for extent in shape:
        n *= int(extent)
    return n


def _unravel(linear, shape):
    """The row-major coordinates of a flat position."""
    coords = []
    for extent in reversed(shape):
        coords.append(linear % extent)
        linear //= extent
    return tuple(reversed(coords))


# ---------------------------------------------------------------------------
# The tables a map carries
# ---------------------------------------------------------------------------


class RolledEmitter:
    """The ``spmw.map`` attributes for each placement, read off the graph.

    Which family a port addresses at a site, which sites a role stands for and
    which ports it is wired on are all answered here; the structural RTL
    emitter and the program builder both read them, so the netlist and the IR
    describe one design.
    """

    def __init__(self, graph):
        self.graph = graph
        self.low = Lowering(graph)

    def placements(self):
        return self.low.placements

    def classes(self, placement):
        """The wiring classes: one role each."""
        return _wiring_classes(placement, self.low.resolutions[placement])

    def role_name(self, placement, order):
        return f"{self.low.kernel_names[placement]}_r{order}"

    def channel_of(self, placement, site, port):
        """The family and channel this site's ``port`` attaches to.

        Three things can serve a port: a peer link addressed by the destination
        site, a peer link looked up by channel id, or a binding's family indexed
        by the site's position in the bundle.  ``(None, None)`` means the port
        connects to nothing here -- a rim site's unbound side.
        """
        res = self.low.resolutions[placement]
        fam = res.families.get(res.site_family.get((site, port)))
        if fam is not None:
            if fam.kind == ch.AFFINE:
                offs = fam.offset if port.direction == OUT else (0,) * len(site)
                return fam, tuple(int(c + o) for c, o in zip(site, offs))
            return fam, (int(fam.slots[(site, port)]),)
        fam = self.low.bind_families.get((placement, port))
        if fam is not None and (site, port) in fam.slots:
            return fam, (int(fam.slots[(site, port)]),)
        return None, None

    def role_ports(self, placement, order):
        """The stream ports a role is wired on, by name, each with its family.

        A role is one body with one calling convention, so every site it stands
        for must be wired alike.  The wiring classes already group sites by
        signature and routing; what they do not see is a *binding* covering
        only part of a class, so that is checked here rather than assumed.
        """
        _sig, _routing, sites = self.classes(placement)[order]
        first = None
        for site in sites:
            wired = []
            for port in placement.iface.ports():
                if port.protocol != STREAM:
                    continue
                fam, _idx = self.channel_of(placement, site, port)
                if fam is not None:
                    wired.append((port, fam))
            wired.sort(key=lambda pf: pf[0].name)
            if first is None:
                first = wired
                continue
            if [(p.name, f.name) for p, f in wired] != [
                (p.name, f.name) for p, f in first
            ]:
                raise SPMWBindingError(
                    f"`{self.role_name(placement, order)}` stands for {len(sites)} "
                    f"sites that are not wired alike: site {sites[0]} has "
                    f"{[p.name for p, _f in first]} and site {site} has "
                    f"{[p.name for p, _f in wired]}. A binding must cover every "
                    f"site of a class or none -- bind the whole rim, or place the "
                    f"odd sites as a role of their own."
                )
        return first or []

    def families(self, placement):
        """Every channel array the placement needs."""
        res = self.low.resolutions[placement]
        out = list(res.families.values())
        for (pl, _port), fam in self.low.bind_families.items():
            if pl is placement and fam not in out:
                out.append(fam)
        return out

    def family_attrs(self, placement):
        return [_family_attr(fam) for fam in self.families(placement)]

    def port_map_attrs(self, placement):
        """How each port reaches its family: a displacement, or a slot table.

        A port appears once per family it addresses, because routing is a
        per-site question and one port may take part in more than one.  Table
        maps come before affine ones: a reader resolving a port at a site takes
        the first map that is live there, and a binding's table is live only at
        the sites it covers, so it must be tried before the peer link that
        serves the interior.
        """
        res = self.low.resolutions[placement]
        grid = placement.grid
        tables, affine = [], []
        for fam in self.families(placement):
            ports = {
                port
                for (_site, port), name in res.site_family.items()
                if name == fam.name
            }
            ports |= {port for (_site, port) in fam.slots if port.protocol == STREAM}
            for port in sorted(ports, key=lambda p: p.name):
                if fam.kind == ch.AFFINE:
                    offset = fam.offset if port.direction == OUT else (0,) * len(grid)
                    affine.append(
                        f'#spmw.port_map<port = "{port.name}", family = '
                        f'"{fam.name}", kind = "affine", offset = {_ints(offset)}>'
                    )
                else:
                    slots = [
                        fam.slots.get((site, port), -1) for site in placement.sites()
                    ]
                    tables.append(
                        f'#spmw.port_map<port = "{port.name}", family = '
                        f'"{fam.name}", kind = "table", slots = {_dense(slots)}>'
                    )
        return tables + affine

    def role_attrs(self, placement, names):
        """One role per wiring class: the body, its signature, its stream order."""
        out = []
        for order, (signature, _routing, _sites) in enumerate(self.classes(placement)):
            missing = sorted(
                p.name
                for p in placement.iface.ports()
                if p.protocol == STREAM and p not in signature
            )
            ports = [p.name for p, _f in self.role_ports(placement, order)]
            out.append(
                f"#spmw.role<unit = @{names[order]}, missing = {_strs(missing)}, "
                f"ports = {_strs(ports)}>"
            )
        return out

    def class_table(self, placement):
        """Which role runs at each site, flattened row-major."""
        index = {}
        for order, (_sig, _routing, sites) in enumerate(self.classes(placement)):
            for site in sites:
                index[site] = order
        return [index[site] for site in placement.sites()]

    def map_attrs(self, placement, role_names):
        """The three attributes ``spmw.map`` carries, as MLIR text."""
        topology = (
            f"#spmw.topology<grid = {_ints(placement.grid)}, families = ["
            + ", ".join(self.family_attrs(placement))
            + "], ports = ["
            + ", ".join(self.port_map_attrs(placement))
            + "]>"
        )
        roles = "[" + ", ".join(self.role_attrs(placement, role_names)) + "]"
        classes = _dense(self.class_table(placement))
        return topology, roles, classes


# ---------------------------------------------------------------------------
# Function bodies
# ---------------------------------------------------------------------------


class _RoleRewriter(_BodyRewriter):
    """Rewrites a unit body for one role of a rolled map.

    Ports are the function's own stream parameters, so a port touch becomes
    ``west[0].get()`` rather than a subscript into a channel array.  Memory
    ports and coordinates go through the base class: a memory port is the
    tensor the map passes, addressed by the site's coordinates, which are the
    function's ``index`` parameters.
    """

    def __init__(self, *args, wired=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.wired = set(wired)

    def _subscript(self, port):
        if port not in self.wired:
            return None
        return ast.Subscript(
            value=ast.Name(id=port.name, ctx=ast.Load()),
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        )


def _function(name, params, body):
    return ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=n, annotation=ann) for n, ann in params],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=None,
        type_params=[],
    )


def stream_decl(name, elem_ann, depth):
    """``name: Stream[elem, depth][1]`` -- a one-element stream array.

    The builder needs a stream it constructs; declared this way it becomes an
    ``allo.stream_construct`` that :func:`hoist_streams` turns into a parameter.
    """
    stream = ast.Subscript(
        value=ast.Name(id="Stream", ctx=ast.Load()),
        slice=ast.Tuple(
            elts=[elem_ann, ast.Constant(value=int(depth))], ctx=ast.Load()
        ),
        ctx=ast.Load(),
    )
    return ast.AnnAssign(
        target=ast.Name(id=name, ctx=ast.Store()),
        annotation=ast.Subscript(
            value=stream, slice=ast.Constant(value=1), ctx=ast.Load()
        ),
        value=None,
        simple=1,
    )


def check_names(tree, known):
    """Every name the program reads must be one the builder can resolve.

    A lowering bug usually shows up as a name nobody defines, and that is far
    cheaper to diagnose here than as a builder failure three passes later.
    """
    known = set(known) | set(dir(builtins)) | {"Stream"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            known.add(node.name)
            known.update(a.arg for a in node.args.args)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            known.add(node.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            known.add(node.target.id)
    missing = sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in known
        }
    )
    if missing:
        raise SPMWBindingError(
            f"the lowered program reads {', '.join(missing)}, which nothing "
            f"defines. This is a lowering bug, not a program error."
        )


def build_functions(functions, global_vars):
    """Turn function ASTs into a module through Allo's IR builder.

    The AST is handed to the type inferer and the builder in memory: nothing is
    unparsed or read back from a file.  Returns the module and the builder's
    context, whose ``func_args`` a schedule needs.
    """
    # pylint: disable=import-outside-toplevel
    from allo._mlir.ir import Context
    from allo.ir.builder import ASTTransformer
    from allo.ir.infer import TypeInferer
    from allo.ir.types import Stream
    from allo.ir.visitor import ASTContext

    tree = ast.Module(body=list(functions), type_ignores=[])
    ast.fix_missing_locations(tree)
    env = dict(global_vars)
    env.setdefault("Stream", Stream)
    check_names(tree, env)
    mlir_ctx = Context()
    inferred = ASTContext(tree=tree, global_vars=dict(env), mlir_ctx=mlir_ctx)
    tree = TypeInferer()(inferred, tree)
    ctx = ASTContext(
        tree=tree,
        global_vars=dict(env),
        mlir_ctx=mlir_ctx,
        func_predicate_tags=inferred.func_predicate_tags,
        meta_fors_to_unroll=inferred.meta_fors_to_unroll,
    )
    module = ASTTransformer()(ctx, tree, "<spmw>")
    return module, ctx


def hoist_streams(fn, names, func_args=None):
    """Make a function's declared streams its trailing parameters, in order.

    The builder constructs one ``allo.stream_construct`` per declaration and
    clones it at every access; the clones are folded back onto the first, and
    the first is replaced by a block argument.  ``itypes`` grows with it, since
    the HLS emitter indexes that string by argument position.
    """
    # pylint: disable=import-outside-toplevel
    from allo._mlir.dialects import allo as allo_d
    from allo._mlir.ir import FunctionType, Location, StringAttr, TypeAttr

    block = fn.entry_block
    constructs = {}
    for op in list(block.operations):
        if isinstance(op, allo_d.StreamConstructOp):
            name = op.attributes["name"].value
            if name in constructs:
                op.result.replace_all_uses_with(constructs[name].result)
                op.operation.erase()
            else:
                constructs[name] = op
    signed = ""
    for name in names:
        op = constructs.pop(name, None)
        if op is None:
            raise SPMWBindingError(
                f"`{fn.name.value}` declares the stream `{name}` but the builder "
                f"constructed none by that name; this is an emission bug."
            )
        arg = block.add_argument(op.result.type, Location.unknown())
        op.result.replace_all_uses_with(arg)
        signed += "u" if "unsigned" in op.attributes else "_"
        op.operation.erase()
    if constructs:
        raise SPMWBindingError(
            f"`{fn.name.value}` constructs streams {sorted(constructs)} that are "
            f"not among its ports; this is an emission bug."
        )
    fn.attributes["function_type"] = TypeAttr.get(
        FunctionType.get([a.type for a in block.arguments], [])
    )
    if "itypes" in fn.attributes:
        fn.attributes["itypes"] = StringAttr.get(fn.attributes["itypes"].value + signed)
    if func_args is not None:
        func_args.setdefault(fn.name.value, []).extend(names)


# ---------------------------------------------------------------------------
# The program
# ---------------------------------------------------------------------------


class _MapPlan:
    """One ``spmw.map``: what it instantiates, and how each site is wired.

    ``roles`` lists, per role, the function's name and its stream ports in
    parameter order; ``wiring`` gives each site's channel for each of those
    ports; ``classes`` names the role at each site, row-major.  The attributes
    are the same facts as MLIR text.
    """

    __slots__ = (
        "name",
        "operands",
        "grid",
        "sites",
        "roles",
        "classes",
        "wiring",
        "attrs",
        "functions",
    )

    def __init__(self, name, operands, grid, sites, roles, classes, wiring, attrs):
        self.name = name
        self.operands = operands
        self.grid = tuple(grid)
        self.sites = sites
        self.roles = roles
        self.classes = classes
        self.wiring = wiring
        self.attrs = attrs
        self.functions = []


class Built:
    """What building a program leaves.

    ``rolled`` is the IR with its maps, as text; ``module`` is the expanded
    form the backends consume, whose ``top`` takes ``arg_order``.
    """

    __slots__ = ("graph", "rolled", "module", "func_args", "ext_libs", "arg_order")

    def __init__(self, graph, rolled, module, func_args, ext_libs, arg_order):
        self.graph = graph
        self.rolled = rolled
        self.module = module
        self.func_args = func_args
        self.ext_libs = ext_libs
        self.arg_order = arg_order


class Program:
    """The rolled program of one elaborated fabric.

    One function per role and per mover, and a ``top`` holding one
    ``spmw.map`` per placement and per mover -- loaders first, then the
    arrays, then drains, which is the order a dataflow region wants them in.
    """

    def __init__(self, graph):
        self.graph = graph
        self.rolled = RolledEmitter(graph)
        self.low = self.rolled.low
        self.arg_order = self.low.arg_order()
        self.maps = []
        for index, mover in enumerate(self.low.movers):
            if mover.role == "load":
                self.maps.append(self._mover_plan(index))
        for placement in self.rolled.placements():
            self.maps.append(self._placement_plan(placement))
        for index, mover in enumerate(self.low.movers):
            if mover.role != "load":
                self.maps.append(self._mover_plan(index))
        self.low.check_outputs_last()

    # -- roles --------------------------------------------------------------

    def role_function(self, placement, order):
        """One role as a function: tensors, one index per grid axis, streams.

        The streams are declared in the body and hoisted afterwards, so the
        returned names are the parameters the function ends up with.
        """
        signature, routing, sites = self.rolled.classes(placement)[order]
        body = placement.roles.get(sites[0])
        if body is None:
            raise SPMWBindingError(
                f"`{placement.name}` is a fabric placed on a topology. Hierarchical "
                f"placement elaborates, but it is not lowered yet -- inline the "
                f"sub-fabric, or place its unit directly."
            )
        tree = body.tree
        pids = [f"_p{a}" for a in range(len(placement.grid))]
        ports = self.rolled.role_ports(placement, order)
        fixed = {a: sites[0][a] for a in getattr(placement, "specialise", ()) or ()}
        rewriter = _RoleRewriter(
            self.low,
            placement,
            signature,
            routing,
            sites,
            pids,
            _io_param_name(tree),
            _site_param_name(tree),
            fixed=fixed,
            tree=tree,
            wired=[p for p, _f in ports],
        )
        stmts = []
        for stmt in tree.body:
            if _is_docstring(stmt):
                continue
            out = rewriter.visit(ast.fix_missing_locations(_copy(stmt)))
            if out is not None:
                stmts.append(out)
        for stmt in stmts:
            _fill_empty_suites(stmt)
        decls = [
            stream_decl(p.name, self.low.type_ann(fam.dtype, fam.block), fam.depth)
            for p, fam in ports
        ]
        body = decls + self.low.stationary_locals(placement) + (stmts or [ast.Pass()])
        params = [
            (f"local_{t.base.name}", self.low.type_ann(t.dtype, t.base.shape))
            for t in self.low.tensors_used(placement)
        ]
        params += [(pid, self.low.type_ann(pid_type(), ())) for pid in pids]
        fn = _function(self.rolled.role_name(placement, order), params, body)
        return fn, [f"{p.name}_0" for p, _f in ports]

    def _placement_plan(self, placement):
        classes = self.rolled.classes(placement)
        names = [self.rolled.role_name(placement, k) for k in range(len(classes))]
        roles, wiring, functions = [], {}, []
        for order, (_sig, _routing, sites) in enumerate(classes):
            ports = self.rolled.role_ports(placement, order)
            roles.append((names[order], [(p.name, fam) for p, fam in ports]))
            for site in sites:
                for port, _fam in ports:
                    wiring[(site, port.name)] = self.rolled.channel_of(
                        placement, site, port
                    )
            functions.append(self.role_function(placement, order))
        plan = _MapPlan(
            placement.name,
            [t.base.name for t in self.low.tensors_used(placement)],
            placement.grid,
            list(placement.sites()),
            roles,
            self.rolled.class_table(placement),
            wiring,
            self.rolled.map_attrs(placement, names),
        )
        plan.functions = functions
        return plan

    # -- movers -------------------------------------------------------------

    def _mover_plan(self, index):
        mover = self.low.movers[index]
        geom = _geometry(mover.bundle)
        grid = tuple(geom.dense)
        count = _volume(grid)
        fam = mover.family
        sites = [_unravel(q, grid) for q in range(count)]
        wiring = {(site, "chan"): (fam, (q,)) for q, site in enumerate(sites)}
        topology = (
            f"#spmw.topology<grid = {_ints(grid)}, families = [{_family_attr(fam)}], "
            f'ports = [#spmw.port_map<port = "chan", family = "{fam.name}", '
            f'kind = "table", slots = {_dense(range(count))}>]>'
        )
        roles = f'[#spmw.role<unit = @{mover.name}, missing = [], ports = ["chan"]>]'
        plan = _MapPlan(
            mover.name,
            [mover.tensor.base.name],
            grid,
            sites,
            [(mover.name, [("chan", fam)])],
            [0] * count,
            wiring,
            (topology, roles, _dense([0] * count)),
        )
        plan.functions = [self.low.mover_function(mover)]
        return plan

    # -- the top ------------------------------------------------------------

    def top_function(self):
        """The top, taking the tensors in the order the program uses them."""
        params = []
        for name in self.arg_order:
            tensor = self.graph.tensors[name]
            params.append((name, self.low.type_ann(tensor.dtype, tensor.shape)))
        return _function(TOP, params, [ast.Pass()])

    def global_vars(self):
        """What the bodies read: their captured names, and the types injected."""
        return dict(self.low.injected)

    def functions(self):
        """Every function AST in the program: the top, then one per role."""
        out = [self.top_function()]
        for plan in self.maps:
            out += [fn for fn, _streams in plan.functions]
        return out

    def build(self):
        """Build the module, verify its maps, and expand it for the backends."""
        # pylint: disable=import-outside-toplevel
        from allo._mlir.dialects import func as func_d
        from allo._mlir.ir import (
            Attribute,
            InsertionPoint,
            Location,
            Operation,
            UnitAttr,
        )

        # The top comes first so it is the builder's own top function; the
        # bodies are appended after it.
        module, ctx = build_functions(self.functions(), self.global_vars())
        with module.context, Location.unknown():
            fns = {
                op.name.value: op
                for op in module.body.operations
                if isinstance(op, func_d.FuncOp)
            }
            for plan in self.maps:
                for fn, streams in plan.functions:
                    hoist_streams(fns[fn.name], streams, ctx.func_args)
                    fns[fn.name].attributes["df.kernel"] = UnitAttr.get()
            top = fns[TOP]
            position = {name: i for i, name in enumerate(self.arg_order)}
            for plan in self.maps:
                topology, roles, classes = plan.attrs
                Operation.create(
                    "spmw.map",
                    results=[],
                    operands=[top.arguments[position[n]] for n in plan.operands],
                    attributes={
                        "topology": Attribute.parse(topology),
                        "roles": Attribute.parse(roles),
                        "classes": Attribute.parse(classes),
                    },
                    ip=InsertionPoint.at_block_terminator(top.entry_block),
                )
            top.attributes["dataflow"] = UnitAttr.get()
            module.operation.verify()
            rolled = str(module)
            self.expand(module)
        return Built(
            self.graph, rolled, module, ctx.func_args, ctx.ext_libs, self.arg_order
        )

    def expand(self, module):
        """Replace every map in ``top`` by one call per site.

        Each channel becomes an ``allo.stream_construct`` named by its family
        and coordinates, made once however many maps address it -- a loader's
        family and the rim it feeds are the same array.
        """
        # pylint: disable=import-outside-toplevel
        from allo._mlir.dialects import allo as allo_d
        from allo._mlir.dialects import arith as arith_d
        from allo._mlir.dialects import func as func_d
        from allo._mlir.ir import (
            FlatSymbolRefAttr,
            IndexType,
            InsertionPoint,
            Location,
            StringAttr,
            UnitAttr,
        )
        from allo.ir.types import UInt

        with module.context, Location.unknown():
            fns = {
                op.name.value: op
                for op in module.body.operations
                if isinstance(op, func_d.FuncOp)
            }
            top = fns[TOP]
            block = top.entry_block
            maps = [op for op in block.operations if op.operation.name == "spmw.map"]
            if len(maps) != len(self.maps):
                raise SPMWBindingError(
                    f"`{TOP}` holds {len(maps)} maps but the program planned "
                    f"{len(self.maps)}; this is an emission bug."
                )
            position = {name: i for i, name in enumerate(self.arg_order)}
            # Channels and coordinates are made once each and go in front of
            # whatever the block holds at that moment, so every definition
            # precedes every call: the simulator later moves each call into
            # its own OpenMP section, and a value defined between two calls
            # would no longer dominate the later one.
            streams, coords = {}, {}

            def front():
                return InsertionPoint(block.operations[0])

            def coordinate(value):
                if value not in coords:
                    # pylint: disable=too-many-function-args
                    coords[value] = arith_d.ConstantOp(
                        IndexType.get(), int(value), ip=front()
                    ).result
                return coords[value]

            for plan, op in zip(self.maps, maps):
                ip = InsertionPoint(op)
                for linear, site in enumerate(plan.sites):
                    role_name, ports = plan.roles[plan.classes[linear]]
                    callee = fns[role_name]
                    types = callee.type.inputs
                    args = [block.arguments[position[n]] for n in plan.operands]
                    args += [coordinate(coord) for coord in site]
                    for k, (port_name, _fam) in enumerate(ports):
                        fam, idx = plan.wiring[(site, port_name)]
                        key = (fam.name, idx)
                        if key not in streams:
                            construct = allo_d.StreamConstructOp(
                                types[len(plan.operands) + len(site) + k], ip=front()
                            )
                            construct.attributes["name"] = StringAttr.get(
                                f"{fam.name}_{'_'.join(str(i) for i in idx)}"
                            )
                            if isinstance(fam.dtype, UInt):
                                construct.attributes["unsigned"] = UnitAttr.get()
                            streams[key] = construct.result
                        args.append(streams[key])
                    func_d.CallOp([], FlatSymbolRefAttr.get(role_name), args, ip=ip)
            for op in maps:
                op.operation.erase()
        return module


def build_program(graph):
    """Build a graph's program: the rolled IR, and its expansion for backends."""
    return Program(graph).build()


def render_module(graph):
    """The rolled program as MLIR text."""
    return build_program(graph).rolled


def role_signature(emitter, placement, order, tensors=None):
    """The calling convention a role body has.

    Tensors the placement uses, then one index per grid axis, then one stream
    per port the role is wired on -- which is what the op's verifier checks.
    """
    low = emitter.low
    used = tensors if tensors is not None else low.tensors_used(placement)
    params = [_mlir_type(t.dtype, t.base.shape) for t in used]
    params += ["index"] * len(placement.grid)
    for _port, fam in emitter.role_ports(placement, order):
        params.append(_stream_type(fam))
    return params


__all__ = [
    "Built",
    "Program",
    "RolledEmitter",
    "TOP",
    "build_functions",
    "build_program",
    "check_names",
    "hoist_streams",
    "render_module",
    "role_signature",
    "stream_decl",
]
