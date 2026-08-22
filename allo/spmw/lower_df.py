# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lowering an elaborated SPMW graph to an ``allo.dataflow`` program.

This is a compiler pass over the graph, not a set of pattern recognisers:
placements become kernels, channel families become stream arrays, movers become
their own kernels, and memory bindings become subscripts.  Anything the
elaborator can represent, this can emit.

Role dispatch rides ``meta_if``, whose condition is evaluated at compile time
against the pids.  One arm per *site signature* -- nine for a mesh, at any size
-- so the emitted program has a number of bodies bounded by the role count
rather than by the grid.

The program is built as an AST and unparsed, never assembled from format
strings, so a malformed emission is a syntax error at generation time rather
than a mystery three passes later.
"""

import ast
import os
import sys
import tempfile
import types

from . import channels as ch
from .bricks import Brick, Tensor
from .component import _io_param_name, captured_env
from .errors import SPMWBindingError
from .index import IndexMap, SliceMap, TIME, to_source
from .placement import Bundle, MemGrid
from .ports import OUT, STREAM

_MODULE_SEQ = [0]


class Mover:
    """A synthesised loader or drain: one kernel over a bundle's members."""

    __slots__ = (
        "binding",
        "bundle",
        "family",
        "tensor",
        "imap",
        "extent",
        "name",
        "role",
    )

    def __init__(self, binding, bundle, family, tensor, imap, extent, name, role):
        self.binding = binding
        self.bundle = bundle
        self.family = family
        self.tensor = tensor
        self.imap = imap
        self.extent = extent
        self.name = name
        self.role = role  # "load" or "drain"


class Lowering:
    """Turns one elaborated fabric into an importable dataflow program."""

    def __init__(self, graph):
        self.graph = graph
        self.injected = {}
        self.consts = []
        self.resolutions = {}
        self.movers = []
        self.seeds = {}
        self.mem_reads = {}
        self.mem_writes = {}
        self.bind_families = {}
        self.kernel_names = {}
        self.tensor_users = {}
        self.fills = {}
        self._plan()

    # -- planning ----------------------------------------------------------

    def _plan(self):
        self._capture_bodies()
        for n, placement in enumerate(self.graph.placements):
            prefix = _ident(placement.name)
            if prefix in self.kernel_names.values():
                prefix = f"{prefix}_{n}"
            self.kernel_names[placement] = prefix
            self.resolutions[placement] = ch.resolve(placement, prefix)
        # Copies first: a shard may name a brick that a later copy fills.
        for binding in self.graph.bindings:
            if binding.kind == "copy":
                self._plan_binding(binding)
        for binding in self.graph.bindings:
            if binding.kind != "copy":
                self._plan_binding(binding)

    def _capture_bodies(self):
        """Carry every body's free names into the generated module.

        A unit body reads types, sizes and helpers from the module it was written
        in; re-emitting it elsewhere has to bring those along, or the tracer
        resolves them against the wrong scope.
        """
        for placement in self.graph.placements:
            component = placement.component
            bodies = [component] + list(getattr(component, "roles", []))
            for body in bodies:
                fn = getattr(body, "fn", None)
                tree = getattr(body, "tree", None)
                if fn is None or tree is None:
                    continue
                for name, value in captured_env(fn, tree).items():
                    self.injected.setdefault(name, value)
        self.reserved = set(self.injected)

    def _plan_binding(self, binding):
        kind = binding.kind
        if kind in ("stream_in", "scatter"):
            self._plan_mover(binding, binding.target, binding.source, "load")
        elif kind == "gather" and isinstance(binding.source, Bundle):
            self._plan_mover(binding, binding.source, binding.target, "drain")
        elif kind == "gather" and isinstance(binding.source, MemGrid):
            self.mem_writes[(binding.source.placement, binding.source.port)] = binding
        elif kind == "gather_mem":
            self.mem_writes[(binding.source.placement, binding.source.port)] = binding
        elif kind == "seed":
            self.seeds[(binding.target.placement, binding.target.port)] = binding.source
        elif kind in ("shard", "stationary"):
            side = binding.target
            self.mem_reads[(side.placement, side.port)] = binding
        elif kind == "link":
            self._plan_link(binding)
        elif kind == "copy":
            # Staging: record what fills the brick so a client reading it can be
            # traced back to the tensor, which is this path's desugaring of the
            # copy.
            self.fills[id(binding.target)] = binding.source

    def _plan_mover(self, binding, bundle, tensor, role):
        family = self._binding_family(bundle)
        name = f"{self.kernel_names[bundle.placement]}_{bundle.port.name}_{role}"
        self.movers.append(
            Mover(
                binding,
                bundle,
                family,
                tensor,
                binding.imap,
                binding.extras.get("extent"),
                name,
                role,
            )
        )
        self._note_user(tensor, name)

    def _plan_link(self, binding):
        src, dst = binding.source, binding.target
        family = self._binding_family(dst, writer=src)
        self.bind_families[(src.placement, src.port)] = family

    def _binding_family(self, bundle, writer=None):
        """One flat family per binding-fed or binding-drained bundle."""
        key = (bundle.placement, bundle.port)
        if key in self.bind_families:
            return self.bind_families[key]
        prefix = self.kernel_names[bundle.placement]
        port = bundle.port
        fam = ch.Family(
            f"{prefix}_{port.name}_bind",
            port.dtype,
            port.shape,
            bundle.placement.depths.get(port, port.depth),
            ch.TABLE,
            (len(bundle),),
        )
        for pos, site in enumerate(bundle.sites):
            fam.slots[(site, port)] = pos
        fam.count = len(bundle)
        if writer is not None:
            for pos, site in enumerate(writer.sites):
                fam.slots[(site, writer.port)] = pos
        self.bind_families[key] = fam
        return fam

    def resolve_storage(self, source):
        """Follow a staged brick back to the tensor a copy fills it from.

        A brick with contents of its own -- an init= ROM -- is already storage;
        one that a copy fills is a staging step, and on a functional path the
        client reads the copy's source directly.
        """
        seen = set()
        while isinstance(source, Brick) and source.init is None:
            filler = self.fills.get(id(source))
            if filler is None or id(filler) in seen:
                break
            seen.add(id(source))
            source = filler
        return source

    def _note_user(self, tensor, kernel):
        if isinstance(tensor, Tensor):
            self.tensor_users.setdefault(tensor.name, []).append(kernel)

    # -- rendering ---------------------------------------------------------

    def render(self):
        """The generated module's source."""
        body = []
        body += _parse_stmts(
            "import allo\n"
            "import allo.dataflow as df\n"
            "from allo.ir.types import Stream\n"
        )
        body += self.consts
        body.append(self._region())
        module = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(module)
        self._check_names(module)
        return ast.unparse(module)

    def _check_names(self, module):
        """Every name the emitted program reads must be one it can resolve.

        A lowering bug usually shows up as a name nobody defines, and that is far
        cheaper to diagnose here than as a tracer failure three passes later.
        """
        import builtins  # pylint: disable=import-outside-toplevel

        known = set(self.injected) | set(dir(builtins))
        known |= {"allo", "df", "Stream"}
        for node in ast.walk(module):
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
                for node in ast.walk(module)
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

    def _region(self):
        args = []
        for tensor in self.graph.tensors.values():
            args.append(
                ast.arg(
                    arg=tensor.name,
                    annotation=self._type_ann(tensor.dtype, tensor.shape),
                )
            )
        body = self._stream_decls()
        for placement in self.graph.placements:
            body.append(self._placement_kernel(placement))
        for mover in self.movers:
            body.append(self._mover_kernel(mover))
        fn = ast.FunctionDef(
            name="top",
            args=ast.arguments(
                posonlyargs=[],
                args=args,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[_call("df.region", [])],
            returns=None,
            type_params=[],
        )
        return fn

    def _stream_decls(self):
        decls = []
        for res in self.resolutions.values():
            for fam in res.families.values():
                decls.append(self._stream_decl(fam))
        for fam in dict.fromkeys(self.bind_families.values()):
            decls.append(self._stream_decl(fam))
        return decls

    def _stream_decl(self, fam):
        elem = self._type_ann(fam.dtype, fam.block)
        stream = ast.Subscript(
            value=ast.Name(id="Stream", ctx=ast.Load()),
            slice=ast.Tuple(elts=[elem, ast.Constant(value=fam.depth)], ctx=ast.Load()),
            ctx=ast.Load(),
        )
        ann = ast.Subscript(
            value=stream,
            slice=(
                ast.Tuple(
                    elts=[ast.Constant(value=int(s)) for s in fam.shape], ctx=ast.Load()
                )
                if len(fam.shape) > 1
                else ast.Constant(value=int(fam.shape[0]))
            ),
            ctx=ast.Load(),
        )
        return ast.AnnAssign(
            target=ast.Name(id=fam.name, ctx=ast.Store()),
            annotation=ann,
            value=None,
            simple=1,
        )

    def _type_ann(self, dtype, shape):
        name = self._inject_type(dtype)
        base = ast.Name(id=name, ctx=ast.Load())
        if not shape:
            return base
        idx = (
            ast.Tuple(elts=[ast.Constant(value=int(s)) for s in shape], ctx=ast.Load())
            if len(shape) > 1
            else ast.Constant(value=int(shape[0]))
        )
        return ast.Subscript(value=base, slice=idx, ctx=ast.Load())

    def _inject_type(self, dtype):
        for name, value in self.injected.items():
            if value is dtype or _same(value, dtype):
                return name
        name = f"_T{len(self.injected)}"
        self.injected[name] = dtype
        return name

    def _inject(self, prefix, value):
        for name, existing in self.injected.items():
            if name.startswith(f"_{prefix}") and existing is value:
                return name
            if name.startswith(f"_{prefix}") and _same(existing, value):
                return name
        name = f"_{prefix}{len(self.injected)}"
        self.injected[name] = value
        return name

    # -- placement kernels -------------------------------------------------

    def _placement_kernel(self, placement):
        name = self.kernel_names[placement]
        grid = placement.grid
        pids = [f"_p{i}" for i in range(len(grid))]
        used = self._tensors_used(placement)

        body = [self._pid_assign(pids)]
        body.extend(self.stationary_locals(placement))
        classes = _signature_classes(placement)
        arms = []
        for order, (sig, sites) in enumerate(classes):
            arm_body = self._transcribe(placement, sig, sites[0], pids)
            arms.append((sig, sites, arm_body or [ast.Pass()], order))

        if len(arms) == 1:
            # Every site runs the same body; there is nothing to dispatch on.
            body.extend(arms[0][2])
        else:
            for _sig, sites, arm_body, order in arms:
                if order == len(arms) - 1:
                    body.append(_with(_call("allo.meta_else", []), arm_body))
                    continue
                cond = self._class_condition(
                    placement, sites, pids, name, order, classes
                )
                verb = "meta_if" if order == 0 else "meta_elif"
                body.append(_with(_call(f"allo.{verb}", [cond]), arm_body))

        args = [
            ast.arg(arg=f"local_{t.name}", annotation=self._type_ann(t.dtype, t.shape))
            for t in used
        ]
        return ast.FunctionDef(
            name=name,
            args=ast.arguments(
                posonlyargs=[],
                args=args,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[self._kernel_decorator(grid, used)],
            returns=None,
            type_params=[],
        )

    def _kernel_decorator(self, grid, tensors):
        keywords = {
            "mapping": ast.List(
                elts=[ast.Constant(value=int(g)) for g in grid], ctx=ast.Load()
            )
        }
        if tensors:
            keywords["args"] = ast.List(
                elts=[ast.Name(id=t.name, ctx=ast.Load()) for t in tensors],
                ctx=ast.Load(),
            )
        return _call("df.kernel", [], keywords=keywords)

    def _class_condition(self, placement, sites, pids, name, order, classes):
        """A compile-time predicate selecting one signature class.

        A table indexed by the pids is exact for any topology; the pids are
        constants when ``meta_if`` evaluates, so the lookup folds away.
        """
        table = _nested_table(placement.grid, sites, order, classes)
        tname = self._inject(f"ROLE_{name}_", table)
        expr = ast.Name(id=tname, ctx=ast.Load())
        for pid in pids:
            expr = ast.Subscript(
                value=expr, slice=ast.Name(id=pid, ctx=ast.Load()), ctx=ast.Load()
            )
        return ast.Compare(
            left=expr, ops=[ast.Eq()], comparators=[ast.Constant(value=order)]
        )

    def _pid_assign(self, pids):
        target = (
            ast.Tuple(
                elts=[ast.Name(id=p, ctx=ast.Store()) for p in pids], ctx=ast.Store()
            )
            if len(pids) > 1
            else ast.Name(id=pids[0], ctx=ast.Store())
        )
        return ast.Assign(targets=[target], value=_call("df.get_pid", []))

    def _tensors_used(self, placement):
        used = []
        for (pl, _port), binding in list(self.mem_reads.items()) + list(
            self.mem_writes.items()
        ):
            if pl is not placement:
                continue
            for side in (binding.source, binding.target):
                side = self.resolve_storage(side)
                if isinstance(side, Tensor) and side not in used:
                    used.append(side)
        return used

    # -- body transcription ------------------------------------------------

    def _transcribe(self, placement, signature, sample_site, pids):
        """Rewrite one unit body for one signature class.

        Everything that is not a port touch is carried through verbatim, which
        is what keeps the arithmetic -- and so the numerics -- identical to the
        program the user wrote.
        """
        body = placement.roles.get(sample_site)
        if body is None:
            raise SPMWBindingError(
                f"`{placement.name}` is a fabric placed on a topology. Hierarchical "
                f"placement elaborates, but it is not lowered on the dataflow path "
                f"yet -- inline the sub-fabric, or place its unit directly."
            )
        tree = body.tree
        io_name = _io_param_name(tree)
        site_name = _site_param_name(tree)
        rewriter = _BodyRewriter(self, placement, signature, pids, io_name, site_name)
        stmts = [rewriter.visit(ast.fix_missing_locations(_copy(s))) for s in tree.body]
        return [s for s in stmts if s is not None]

    # -- addressing --------------------------------------------------------

    def port_subscript(self, placement, port, pids, bound=True):
        """The stream-array subscript naming this site's end of ``port``.

        Which family serves a port is a per-site question: where the link rule
        binds it, the topology's family; where it does not, the family the
        binding's loaders and drains write to.
        """
        res = self.resolutions[placement]
        fam = res.family_of(port) if bound else None
        if fam is None:
            fam = self.bind_families.get((placement, port))
            if fam is None:
                return None
        if fam.kind == ch.AFFINE:
            offs = fam.offset if port.direction == OUT else (0,) * len(pids)
            elts = []
            for pid, off in zip(pids, offs):
                elts.append(
                    ast.Name(id=pid, ctx=ast.Load())
                    if off == 0
                    else ast.BinOp(
                        left=ast.Name(id=pid, ctx=ast.Load()),
                        op=ast.Add() if off > 0 else ast.Sub(),
                        right=ast.Constant(value=abs(off)),
                    )
                )
            idx = ast.Tuple(elts=elts, ctx=ast.Load()) if len(elts) > 1 else elts[0]
        else:
            table = _slot_table(placement.grid, fam, port)
            tname = self._inject(f"CH_{fam.name}_{port.name}_", table)
            idx = ast.Name(id=tname, ctx=ast.Load())
            for pid in pids:
                idx = ast.Subscript(
                    value=idx, slice=ast.Name(id=pid, ctx=ast.Load()), ctx=ast.Load()
                )
        return ast.Subscript(
            value=ast.Name(id=fam.name, ctx=ast.Load()), slice=idx, ctx=ast.Load()
        )

    def mem_subscript(self, placement, port, pids, extra=None):
        """The tensor subscript this site's memory port reads or writes."""
        binding = self.mem_reads.get((placement, port)) or self.mem_writes.get(
            (placement, port)
        )
        if binding is None:
            raise SPMWBindingError(
                f"`{placement.name}.{port.name}` has no memory binding to lower."
            )
        source = self.resolve_storage(
            binding.source
            if binding.kind in ("shard", "stationary")
            else binding.target
        )
        if isinstance(source, Brick):
            return self._brick_subscript(source, extra, port)
        names = {
            axis: ast.Name(id=pid, ctx=ast.Load())
            for axis, pid in zip(placement.axes, pids)
        }
        subs = _map_subscripts(binding.imap, names, placement, extra)
        target = ast.Name(id=f"local_{source.name}", ctx=ast.Load())
        idx = ast.Tuple(elts=subs, ctx=ast.Load()) if len(subs) > 1 else subs[0]
        return ast.Subscript(value=target, slice=idx, ctx=ast.Load())

    def stationary_locals(self, placement):
        """Per-site constant declarations for this placement's stationary bricks."""
        decls = []
        for (pl, port), binding in self.mem_reads.items():
            if pl is not placement or binding.kind != "stationary":
                continue
            brick = binding.source
            if not isinstance(brick, Brick):
                continue
            if brick.init is None:
                raise SPMWBindingError(
                    f"`{brick.name}` is stationary but has no init= contents, so "
                    f"there is nothing to make resident. Give it init=, or fill it "
                    f"with a copy from a tensor."
                )
            rom = self._inject(f"ROM_{_ident(brick.name)}_", brick.init)
            decls.append(
                ast.AnnAssign(
                    target=ast.Name(id=self._station_name(port), ctx=ast.Store()),
                    annotation=self._type_ann(brick.dtype, brick.shape),
                    value=ast.Name(id=rom, ctx=ast.Load()),
                    simple=1,
                )
            )
        return decls

    @staticmethod
    def _station_name(port):
        return f"_st_{port.name}"

    def _brick_subscript(self, brick, extra, port=None):
        if brick.init is None:
            raise SPMWBindingError(
                f"`{brick.name}` is read by `{port.name if port else '?'}` but has "
                f"neither init= contents nor a copy filling it, so there is nothing "
                f"to read."
            )
        node = ast.Name(id=self._station_name(port), ctx=ast.Load())
        if extra:
            idx = (
                ast.Tuple(elts=list(extra), ctx=ast.Load())
                if len(extra) > 1
                else extra[0]
            )
            node = ast.Subscript(value=node, slice=idx, ctx=ast.Load())
        return node

    # -- mover kernels -----------------------------------------------------

    def _mover_kernel(self, mover):
        bundle = mover.bundle
        geom = _geometry(bundle)
        pids = [f"_q{i}" for i in range(len(geom.dense))]
        body = [self._pid_assign(pids)]

        site_exprs = geom.site_exprs(pids)
        member = geom.member_expr(pids)
        chan = ast.Subscript(
            value=ast.Name(id=mover.family.name, ctx=ast.Load()),
            slice=member,
            ctx=ast.Load(),
        )
        tensor = mover.tensor
        names = {axis: site_exprs[i] for i, axis in enumerate(bundle.placement.axes)}

        loop_var = "_t"
        extent = mover.extent if mover.extent is not None else 1
        subs = self._mover_subscripts(mover, names, loop_var, geom, pids)
        elem = ast.Subscript(
            value=ast.Name(id=f"local_{tensor.name}", ctx=ast.Load()),
            slice=ast.Tuple(elts=subs, ctx=ast.Load()) if len(subs) > 1 else subs[0],
            ctx=ast.Load(),
        )
        block = mover.binding.extras.get("block", ())
        inner = self._transfer(mover, chan, elem, block)
        loop = ast.For(
            target=ast.Name(id=loop_var, ctx=ast.Store()),
            iter=_call("range", [ast.Constant(value=int(extent))]),
            body=inner,
            orelse=[],
        )
        body.append(loop)
        args = [
            ast.arg(
                arg=f"local_{tensor.name}",
                annotation=self._type_ann(tensor.dtype, tensor.shape),
            )
        ]
        return ast.FunctionDef(
            name=mover.name,
            args=ast.arguments(
                posonlyargs=[],
                args=args,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[self._kernel_decorator(geom.dense, [tensor])],
            returns=None,
            type_params=[],
        )

    def _transfer(self, mover, chan, elem, block):
        """Move one token between a tensor element and a channel.

        A port carrying a block moves the whole block per token, so the transfer
        is a declared temporary filled element-wise -- the shape the dataflow
        model already uses for streams of blocks.
        """
        put = mover.role == "load"
        if not block:
            if put:
                return [
                    ast.Expr(
                        value=_call_node(
                            ast.Attribute(value=chan, attr="put", ctx=ast.Load()),
                            [elem],
                        )
                    )
                ]
            got = _call_node(ast.Attribute(value=chan, attr="get", ctx=ast.Load()), [])
            return [ast.Assign(targets=[_store(elem)], value=got)]

        tmp = "_blk"
        ann = self._type_ann(mover.bundle.port.dtype, block)
        ref = ast.Name(id=tmp, ctx=ast.Load())
        idx = [ast.Name(id=f"_b{k}", ctx=ast.Load()) for k in range(len(block))]
        cell = ast.Subscript(
            value=ref,
            slice=ast.Tuple(elts=idx, ctx=ast.Load()) if len(idx) > 1 else idx[0],
            ctx=ast.Load(),
        )
        if put:
            decl = ast.AnnAssign(
                target=ast.Name(id=tmp, ctx=ast.Store()),
                annotation=ann,
                value=ast.Constant(value=0),
                simple=1,
            )
            copy = [ast.Assign(targets=[_store(cell)], value=elem)]
            tail = [
                ast.Expr(
                    value=_call_node(
                        ast.Attribute(value=chan, attr="put", ctx=ast.Load()), [ref]
                    )
                )
            ]
        else:
            decl = ast.AnnAssign(
                target=ast.Name(id=tmp, ctx=ast.Store()),
                annotation=ann,
                value=_call_node(
                    ast.Attribute(value=chan, attr="get", ctx=ast.Load()), []
                ),
                simple=1,
            )
            copy = [ast.Assign(targets=[_store(elem)], value=cell)]
            tail = []
        loops = copy
        for k in reversed(range(len(block))):
            loops = [
                ast.For(
                    target=ast.Name(id=f"_b{k}", ctx=ast.Store()),
                    iter=_call("range", [ast.Constant(value=int(block[k]))]),
                    body=loops,
                    orelse=[],
                )
            ]
        return [decl] + loops + tail

    def _mover_subscripts(self, mover, names, loop_var, geom, pids):
        imap = mover.imap
        block = mover.binding.extras.get("block", ())
        if isinstance(imap, IndexMap) and not imap.is_lambda:
            subs = []
            for entry in imap.spec:
                if entry is TIME:
                    subs.append(ast.Name(id=loop_var, ctx=ast.Load()))
                elif isinstance(entry, int):
                    subs.append(ast.Constant(value=entry))
                else:
                    subs.append(
                        _parse_expr(to_source(entry, _src_names(names), loop_var))
                    )
        else:
            # A lambda is the escape hatch; evaluate it over the whole domain and
            # emit the result as a table, which is exact and stays small because
            # lambdas are reserved for maps affine arithmetic cannot state.
            table = _lambda_table(mover, geom)
            tname = self._inject(f"IX_{mover.name}_", table)
            node = ast.Name(id=tname, ctx=ast.Load())
            for pid in pids:
                node = ast.Subscript(
                    value=node, slice=ast.Name(id=pid, ctx=ast.Load()), ctx=ast.Load()
                )
            node = ast.Subscript(
                value=node, slice=ast.Name(id=loop_var, ctx=ast.Load()), ctx=ast.Load()
            )
            rank = mover.tensor.rank - len(block)
            subs = [
                ast.Subscript(value=node, slice=ast.Constant(value=k), ctx=ast.Load())
                for k in range(rank)
            ]
        for k in range(len(block)):
            subs.append(ast.Name(id=f"_b{k}", ctx=ast.Load()))
        return subs


# ---------------------------------------------------------------------------
# Body rewriting
# ---------------------------------------------------------------------------


class _BodyRewriter(ast.NodeTransformer):
    """Rewrites port touches into channel and tensor accesses."""

    def __init__(self, lowering, placement, signature, pids, io_name, site_name):
        super().__init__()
        self.low = lowering
        self.placement = placement
        self.signature = signature
        self.pids = pids
        self.io = io_name
        self.site = site_name
        self.iface = placement.iface

    # -- helpers

    def _port_of(self, node):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.io
        ):
            return self.iface.__ports__.get(node.attr)
        return None

    def _bound(self, port):
        return port in self.signature

    def _subscript(self, port):
        """This site's channel for ``port``, or None when it has none at all."""
        return self.low.port_subscript(
            self.placement, port, self.pids, bound=self._bound(port)
        )

    # -- visits

    def visit_Call(self, node):
        self.generic_visit(node)
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in ("get", "put"):
            port = self._port_of(fn.value)
            if port is not None and port.protocol == STREAM:
                return self._stream_access(node, port, fn.attr)
        return node

    def _stream_access(self, node, port, verb):
        if verb == "get" and not self._bound(port):
            seed = self.low.seeds.get((self.placement, port))
            if seed is not None:
                # A rank-0 source folds into the consuming site: zero hardware.
                return ast.Constant(value=seed)
        sub = self._subscript(port)
        if sub is None:
            if verb == "get":
                raise SPMWBindingError(
                    f"`{self.placement.name}.{port.name}` is read but nothing "
                    f"feeds it at this site."
                )
            return node
        return _call_node(
            ast.Attribute(value=sub, attr=verb, ctx=ast.Load()), node.args
        )

    def visit_Expr(self, node):
        # A put on an unbound Out is a discard, so the statement disappears.
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            if value.func.attr == "put":
                port = self._port_of(value.func.value)
                if (
                    port is not None
                    and port.protocol == STREAM
                    and self._subscript(port) is None
                ):
                    return None
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        # `s, b = site.rank` restates the pids the kernel already has.
        if self.site and _is_site_rank(node.value, self.site):
            return ast.Assign(targets=node.targets, value=_call("df.get_pid", []))
        node.value = self.visit(node.value)
        node.targets = [self._store_target(t) for t in node.targets]
        return node

    def _store_target(self, target):
        port = self._port_of(target)
        if port is not None and port.protocol != STREAM:
            sub = self.low.mem_subscript(self.placement, port, self.pids)
            return _store(sub)
        if isinstance(target, ast.Subscript):
            port = self._port_of(target.value)
            if port is not None and port.protocol != STREAM:
                extra = _index_elts(target.slice)
                sub = self.low.mem_subscript(self.placement, port, self.pids, extra)
                return _store(sub)
        return self.visit(target)

    def visit_Subscript(self, node):
        port = self._port_of(node.value)
        if port is not None and port.protocol != STREAM:
            extra = [self.visit(e) for e in _index_elts(node.slice)]
            return self.low.mem_subscript(self.placement, port, self.pids, extra)
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):
        port = self._port_of(node)
        if port is not None and port.protocol != STREAM:
            return self.low.mem_subscript(self.placement, port, self.pids)
        if (
            self.site
            and isinstance(node.value, ast.Name)
            and node.value.id == self.site
        ):
            if node.attr == "grid":
                return ast.Tuple(
                    elts=[ast.Constant(value=int(g)) for g in self.placement.grid],
                    ctx=ast.Load(),
                )
            if node.attr == "rank":
                return ast.Tuple(
                    elts=[ast.Name(id=p, ctx=ast.Load()) for p in self.pids],
                    ctx=ast.Load(),
                )
        self.generic_visit(node)
        return node


# ---------------------------------------------------------------------------
# Bundle geometry
# ---------------------------------------------------------------------------


class _Geometry:
    """How a bundle's members are addressed by a mover kernel's pids."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.rank = len(bundle.placement.grid)
        self.axes = []
        for a in range(self.rank):
            values = sorted({s[a] for s in bundle.sites})
            step = _progression(values)
            self.axes.append((values, step))
        self.varying = [a for a in range(self.rank) if len(self.axes[a][0]) > 1]
        self.dense = [len(self.axes[a][0]) for a in self.varying] or [1]

    def site_exprs(self, pids):
        """Per grid axis, the source expression giving this member's coordinate."""
        out = []
        for a in range(self.rank):
            values, step = self.axes[a]
            if len(values) == 1:
                out.append(ast.Constant(value=int(values[0])))
                continue
            pid = pids[self.varying.index(a)]
            node = ast.Name(id=pid, ctx=ast.Load())
            if step not in (1, None):
                node = ast.BinOp(
                    left=node, op=ast.Mult(), right=ast.Constant(value=int(step))
                )
            if values[0]:
                node = ast.BinOp(
                    left=node, op=ast.Add(), right=ast.Constant(value=int(values[0]))
                )
            out.append(node)
        return out

    def member_expr(self, pids):
        """The flat position of this member, row-major over the dense shape."""
        node = None
        for k, extent in enumerate(self.dense):
            term = (
                ast.Name(id=pids[k], ctx=ast.Load())
                if k < len(pids)
                else ast.Constant(0)
            )
            node = (
                term
                if node is None
                else ast.BinOp(
                    left=ast.BinOp(
                        left=node, op=ast.Mult(), right=ast.Constant(value=int(extent))
                    ),
                    op=ast.Add(),
                    right=term,
                )
            )
        return node if node is not None else ast.Constant(value=0)

    def site_of(self, position):
        """The site a flat member position names."""
        coords = []
        rem = position
        strides = []
        acc = 1
        for extent in reversed(self.dense):
            strides.insert(0, acc)
            acc *= extent
        picks = {}
        for k, a in enumerate(self.varying):
            picks[a] = rem // strides[k]
            rem = rem % strides[k]
        for a in range(self.rank):
            values, _ = self.axes[a]
            coords.append(values[picks[a]] if a in picks else values[0])
        return tuple(coords)


def _geometry(bundle):
    return _Geometry(bundle)


def _progression(values):
    if len(values) < 2:
        return 1
    step = values[1] - values[0]
    for a, b in zip(values, values[1:]):
        if b - a != step:
            return None
    return step


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _signature_classes(placement):
    """Signature classes in a deterministic order, biggest first.

    Ordering by size puts the interior arm first, which is what a reader of the
    emitted program expects to see.
    """
    classes = placement.topology.signatures()
    ordered = sorted(
        classes.items(), key=lambda kv: (-len(kv[1]), sorted(p.name for p in kv[0]))
    )
    return ordered


def _nested_table(grid, sites, order, classes):
    """A grid-shaped table naming each site's class index."""
    index = {}
    for k, (_sig, members) in enumerate(classes):
        for site in members:
            index[site] = k

    def build(prefix):
        depth = len(prefix)
        if depth == len(grid):
            return index.get(tuple(prefix), -1)
        return tuple(build(prefix + [v]) for v in range(grid[depth]))

    return build([])


def _slot_table(grid, fam, port):
    """A grid-shaped table naming the channel each site's port attaches to."""

    def build(prefix):
        depth = len(prefix)
        if depth == len(grid):
            return fam.slots.get((tuple(prefix), port), 0)
        return tuple(build(prefix + [v]) for v in range(grid[depth]))

    return build([])


def _lambda_table(mover, geom):
    """Evaluate a lambda index map over the whole (member, step) domain."""
    extent = mover.extent if mover.extent is not None else 1
    rows = []
    for pos in range(len(mover.bundle.sites)):
        site = geom.site_of(pos)
        env = dict(mover.bundle.placement.env(site), __coords__=site)
        steps = []
        for t in range(extent):
            idx = mover.imap.eval(env, step=t if mover.imap.has_time else None)
            steps.append(tuple(int(v) for v in idx))
        rows.append(tuple(steps))
    return _reshape(tuple(rows), geom.dense)


def _reshape(flat, dense):
    """Nest a flat per-member sequence into the mover's pid shape."""
    if len(dense) <= 1:
        return flat
    out = []
    stride = 1
    for extent in dense[1:]:
        stride *= extent
    for k in range(dense[0]):
        out.append(_reshape(flat[k * stride : (k + 1) * stride], dense[1:]))
    return tuple(out)


def _map_subscripts(imap, names, placement, extra):
    """Tensor subscripts for a memory binding at one site."""
    if isinstance(imap, SliceMap):
        subs = []
        block_pos = 0
        for t in range(imap.rank):
            g = imap.axis_of[t]
            if g is None:
                node = (
                    extra[block_pos]
                    if extra and block_pos < len(extra)
                    else ast.Constant(0)
                )
                block_pos += 1
                subs.append(node)
                continue
            pid = names[placement.axes[g]]
            size = imap.block[t] if t < len(imap.block) else 1
            if imap.kind == "blocked" and size > 1:
                node = ast.BinOp(
                    left=pid, op=ast.Mult(), right=ast.Constant(value=int(size))
                )
                if extra and block_pos < len(extra):
                    node = ast.BinOp(left=node, op=ast.Add(), right=extra[block_pos])
                    block_pos += 1
                subs.append(node)
            else:
                subs.append(pid)
        return subs
    if imap.is_lambda:
        raise SPMWBindingError(
            "a lambda index= on a memory binding is not lowered on this path; "
            "spell it as an axis expression."
        )
    subs = []
    for entry in imap.spec:
        if isinstance(entry, int):
            subs.append(ast.Constant(value=entry))
        else:
            subs.append(_parse_expr(to_source(entry, _src_names(names))))
    if extra:
        subs.extend(extra)
    return subs


def _src_names(names):
    return {axis: ast.unparse(node) for axis, node in names.items()}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _parse_stmts(src):
    return ast.parse(src).body


def _parse_expr(src):
    return ast.parse(src, mode="eval").body


def _copy(node):
    return ast.parse(ast.unparse(ast.Module(body=[node], type_ignores=[]))).body[0]


def _call(name, args, keywords=None):
    func = _parse_expr(name)
    return ast.Call(
        func=func,
        args=list(args),
        keywords=[ast.keyword(arg=k, value=v) for k, v in (keywords or {}).items()],
    )


def _call_node(func, args):
    return ast.Call(func=func, args=list(args), keywords=[])


def _with(ctx_expr, body):
    return ast.With(
        items=[ast.withitem(context_expr=ctx_expr, optional_vars=None)],
        body=body,
    )


def _store(node):
    new = _copy_expr(node)
    new.ctx = ast.Store()
    return new


def _copy_expr(node):
    return ast.parse(ast.unparse(node), mode="eval").body


def _index_elts(slice_node):
    if isinstance(slice_node, ast.Tuple):
        return list(slice_node.elts)
    return [slice_node]


def _is_site_rank(node, site_name):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == site_name
        and node.attr == "rank"
    )


def _site_param_name(tree):
    args = tree.args.args
    return args[1].arg if len(args) > 1 else None


def _same(a, b):
    """Structural equality for injected constants, tolerating unhashable values."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    try:
        return bool(a == b)
    except Exception:  # pylint: disable=broad-except
        return False


def _ident(name):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------


def build_dataflow(graph, keep=None):
    """Emit the dataflow program and import it, returning its ``top`` region."""
    low = Lowering(graph)
    src = low.render()
    seq = _MODULE_SEQ[0] = _MODULE_SEQ[0] + 1
    name = f"_spmw_{_ident(graph.fabric.name)}_{seq}"
    directory = keep or tempfile.mkdtemp(prefix="spmw_")
    path = os.path.join(directory, f"{name}.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(src)
    module = types.ModuleType(name)
    module.__file__ = path
    module.__dict__.update(low.injected)
    sys.modules[name] = module
    # A real file on disk, because the tracer reads the body back with
    # inspect.getsourcelines rather than from the code object.
    exec(compile(src, path, "exec"), module.__dict__)  # pylint: disable=exec-used
    top = module.top
    top._spmw_source = src
    top._spmw_path = path
    return top


def render_source(graph):
    """The dataflow program this graph lowers to, as source."""
    return Lowering(graph).render()


__all__ = ["Lowering", "build_dataflow", "render_source"]
