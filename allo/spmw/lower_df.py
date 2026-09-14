# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Planning the lowering of an elaborated SPMW graph.

This is the analysis half of the compiler: placements become roles, channel
families become stream arrays, movers -- the loaders and drains a binding asks
for -- become bodies of their own, and memory bindings become subscripts.
Everything here is a table or a rewrite over the unit's own AST; the IR is
built from these by :mod:`allo.spmw.lower_mlir`, which hands the rewritten
bodies to Allo's IR builder in memory.  No Python source is rendered, written
or executed on the way.

The rewrites produce ASTs rather than text, so a malformed emission is a
structural error at generation time rather than a mystery three passes later.
"""

import ast

from . import channels as ch
from .abi import EDGE_DEPTH
from .bricks import Brick, Tensor
from .component import captured_env, rename_free
from .errors import SPMWBindingError, SPMWMemoryError
from .index import IndexMap, SliceMap, TIME, to_source
from .placement import Bundle, MemGrid
from .ports import STREAM


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


def _banked_layout(brick):
    """The brick's XOR-swizzled layout, or None if it is stored plainly."""
    layout = getattr(brick, "layout", None)
    if layout is None or getattr(layout, "bank_fn", None) != "xor":
        return None
    if len(brick.shape) != 1:
        raise SPMWMemoryError(
            f"`{brick.name}` is {len(brick.shape)}-dimensional and asks for "
            f"xor_bank. Banking splits *one* linear address space across banks, "
            f"so the brick has to be one-dimensional; reshape it, or bank the "
            f"axis you mean by declaring it that way."
        )
    if layout.stride_bit is None:
        raise SPMWMemoryError(
            f"`{brick.name}` asks for xor_bank with no stride bit, so there is "
            f"no access pattern to be conflict-free *for*. Give the stride the "
            f"stage reads at, with `xor_bank(banks, stride_bit=s)`."
        )
    size = int(brick.shape[0])
    if size % layout.banks:
        raise SPMWMemoryError(
            f"`{brick.name}` holds {size} elements across {layout.banks} banks, "
            f"which does not divide. A ragged bank is not a layout."
        )
    return layout


def _bank_subscript(index, layout):
    """`(bank, row)` as AST, the swizzle written out.

    `bank = (i & (banks-1)) ^ (((i >> s) & 1) << (bits-1))`, `row = i >> bits`
    -- the same arithmetic `Layout.bank_of` does in Python, so the emitted
    design and the elaboration-time permutation cannot drift apart.
    """
    bits = layout.bank_bits

    def const(value):
        return ast.Constant(value=value)

    def op(left, operator, right):
        return ast.BinOp(left=left, op=operator, right=right)

    low = op(index, ast.BitAnd(), const(layout.banks - 1))
    picked = op(
        op(index, ast.RShift(), const(layout.stride_bit)), ast.BitAnd(), const(1)
    )
    bank = op(low, ast.BitXor(), op(picked, ast.LShift(), const(bits - 1)))
    row = op(index, ast.RShift(), const(bits))
    return bank, row


def _bank_init(data, layout):
    """The brick's contents rearranged into `[banks][rows]`.

    The swizzle is a bijection -- `test_spmw_banking.py` holds it to that -- so
    this is a permutation and nothing is lost or duplicated.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    flat = np.asarray(data).reshape(-1)
    rows = len(flat) // layout.banks
    out = np.zeros((layout.banks, rows), dtype=flat.dtype)
    for index, value in enumerate(flat):
        out[layout.bank_of(index), layout.row_of(index)] = value
    return out


class Lowering:
    """Turns one elaborated fabric into an importable dataflow program.

    The state is a set of small side tables keyed by placement, port or binding;
    splitting them into helper objects would add indirection without hiding
    anything, so the count stands.
    """

    # pylint: disable=too-many-instance-attributes

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

    @property
    def placements(self):
        """The placements that still instantiate something.

        A placed fabric was replaced by running its body per site, so it emits
        nothing itself.
        """
        return [p for p in self.graph.placements if not p.expanded]

    def _plan(self):
        self._capture_bodies()
        for n, placement in enumerate(self.placements):
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
        for placement in self.placements:
            component = placement.component
            bodies = [component] + list(getattr(component, "roles", []))
            for body in bodies:
                fn = getattr(body, "fn", None)
                tree = getattr(body, "tree", None)
                if fn is None or tree is None:
                    continue
                # Units made by a factory capture the same names with
                # different values (a stage's `span`); every body is emitted
                # into one program, so a name that is already injected with
                # another value is renamed in this body and injected afresh.
                renamed = dict(getattr(body, "spmw_renamed", {}))
                fresh = {}
                for name, value in captured_env(fn, tree, renamed).items():
                    if name in self.injected and not _same(self.injected[name], value):
                        new = f"{name}__{_ident(placement.name)}"
                        while new in self.injected:
                            new += "_"
                        fresh[name] = new
                        self.injected[new] = value
                    else:
                        self.injected.setdefault(name, value)
                if fresh:
                    rename_free(tree, fresh)
                    for name, new in fresh.items():
                        renamed[new] = renamed.get(name, name)
                    body.spmw_renamed = renamed
                    tree.spmw_renamed = renamed
        self.reserved = set(self.injected)

    def _plan_binding(self, binding):
        for side in (binding.source, binding.target):
            if isinstance(side, (Bundle, MemGrid)) and side.placement.expanded:
                return  # consumed by the expansion
        kind = binding.kind
        if kind in {"stream_in", "scatter"}:
            self._plan_mover(binding, binding.target, binding.source, "load")
        elif kind == "gather" and isinstance(binding.source, Bundle):
            self._plan_mover(binding, binding.source, binding.target, "drain")
        elif kind == "gather" and isinstance(binding.source, MemGrid):
            self.mem_writes[(binding.source.placement, binding.source.port)] = binding
        elif kind == "gather_mem":
            self.mem_writes[(binding.source.placement, binding.source.port)] = binding
        elif kind == "seed":
            self.seeds[(binding.target.placement, binding.target.port)] = binding.source
        elif kind in {"shard", "stationary"}:
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
        # An edge stream holds a launch's tokens for its site (up to
        # EDGE_DEPTH), as the kernel's edge FIFOs do: a drain reads its sites
        # in a fixed order, and at a depth of two a site that has produced its
        # rows blocks while the drain waits on another site whose inputs
        # are stuck behind the first -- the 8x8 FEATHER launch never returned.
        family.depth = max(
            family.depth, min(_tokens_per_site(tensor, bundle), EDGE_DEPTH)
        )
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
        fam.geometry[port] = _geometry(bundle)
        fam.count = len(bundle)
        if writer is not None:
            for pos, site in enumerate(writer.sites):
                fam.slots[(site, writer.port)] = pos
            fam.geometry[writer.port] = _geometry(writer)
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

    def written_tensors(self):
        """Tensors some binding writes -- the outputs, in declared order."""
        written = set()
        for binding in self.graph.bindings:
            target = binding.target
            if isinstance(target, Tensor):
                written.add(target.base.name)
            if binding.kind in {"gather", "gather_mem"} and isinstance(
                binding.target, Tensor
            ):
                written.add(binding.target.base.name)
        return written

    def check_outputs_last(self):
        """The HLS backend takes its outputs at the end of the top signature."""
        order = self.arg_order()
        written = self.written_tensors()
        seen_output = None
        for name in order:
            if name in written:
                seen_output = name
            elif seen_output is not None:
                raise SPMWBindingError(
                    f"the lowered program would take `{name}` after the output "
                    f"`{seen_output}`, and the backend requires output arguments "
                    f"at the end. This is a lowering bug: the kernels are emitted "
                    f"in an order that puts a read after a write."
                )

    def arg_order(self):
        """The top function's tensors: inputs first, outputs last.

        Loaders, then the arrays, then drains -- the order the maps are laid
        out in -- with each tensor at its first use.  A fabric may declare its
        tensors in any order; the built module takes this one, and the caller
        permutes.
        """
        seen = []

        def note(name):
            if name not in seen:
                seen.append(name)

        for mover in self.movers:
            if mover.role == "load":
                note(mover.tensor.base.name)
        for placement in self.placements:
            for tensor in self.tensors_used(placement):
                note(tensor.base.name)
        for mover in self.movers:
            if mover.role != "load":
                note(mover.tensor.base.name)
        return seen

    def canonical_annotation(self, node):
        """Rewrite a declaration's type into the subscript spelling.

        A shaped type bound to a name -- `csample = float32[2]`, then
        `u: csample` -- reads as a scalar to the tracer, which resolves a bare
        Name annotation without unwrapping it. Spelling the shape explicitly is
        what makes `u[0]` an element rather than a bit-slice.
        """
        try:
            value = eval(  # pylint: disable=eval-used
                compile(
                    ast.Expression(ast.fix_missing_locations(_copy_expr(node))),
                    "<annotation>",
                    "eval",
                ),
                dict(self.injected),
            )
        except Exception:  # pylint: disable=broad-except
            return node
        shape = tuple(getattr(value, "shape", ()) or ())
        if not shape:
            return node
        return self.type_ann(getattr(value, "dtype", value), shape)

    def type_ann(self, dtype, shape):
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

    def tensors_used(self, placement):
        """The tensors a placement's memory bindings reach, first-seen order."""
        used = []
        for (pl, _port), binding in list(self.mem_reads.items()) + list(
            self.mem_writes.items()
        ):
            if pl is not placement:
                continue
            for side in (binding.source, binding.target):
                side = self.resolve_storage(side)
                if isinstance(side, Tensor) and not any(
                    u.base is side.base for u in used
                ):
                    used.append(side)
        return used

    # -- body transcription ------------------------------------------------

    # -- addressing --------------------------------------------------------

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
            if binding.kind in {"shard", "stationary"}
            else binding.target
        )
        if isinstance(source, Brick):
            return self._brick_subscript(source, extra, port)
        names = {
            axis: ast.Name(id=pid, ctx=ast.Load())
            for axis, pid in zip(placement.axes, pids)
        }
        subs = _offset(_map_subscripts(binding.imap, names, placement, extra), source)
        target = ast.Name(id=f"local_{source.base.name}", ctx=ast.Load())
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
            layout = _banked_layout(brick)
            contents = brick.init if layout is None else _bank_init(brick.init, layout)
            shape = brick.shape if layout is None else contents.shape
            rom = self._inject(f"ROM_{_ident(brick.name)}_", contents)
            decls.append(
                ast.AnnAssign(
                    target=ast.Name(id=self._station_name(port), ctx=ast.Store()),
                    annotation=self.type_ann(brick.dtype, shape),
                    value=ast.Name(id=rom, ctx=ast.Load()),
                    simple=1,
                )
            )
        return decls

    @staticmethod
    def _station_name(port):
        return f"_st_{port.name}"

    def resident_layout(self, placement, port):
        """The XOR-banked layout behind a memory port, or None if it is plain.

        `stationary_locals` permutes a banked brick's contents into
        `[banks][rows]`, so *every* path that reads the resident has to address
        it that way. The array program does it in `_brick_subscript`; the unit
        path in :mod:`allo.spmw.role_ip` builds its own subscript and needs the
        same answer. Both asking one function is what stops them drifting --
        which they had: the unit declared the permuted ROM and then indexed it
        linearly, so a banked memory was stored banked and read straight.
        """
        binding = self.mem_reads.get((placement, port)) or self.mem_writes.get(
            (placement, port)
        )
        if binding is None:
            return None
        source = self.resolve_storage(
            binding.source
            if binding.kind in {"shard", "stationary"}
            else binding.target
        )
        return _banked_layout(source) if isinstance(source, Brick) else None

    def banked_subscript(self, target, index, layout):
        """`target[bank, row]` -- the one place a banked resident is addressed."""
        bank, row = _bank_subscript(index, layout)
        return ast.Subscript(
            value=target,
            slice=ast.Tuple(elts=[bank, row], ctx=ast.Load()),
            ctx=ast.Load(),
        )

    def _brick_subscript(self, brick, extra, port=None):
        if brick.init is None:
            raise SPMWBindingError(
                f"`{brick.name}` is read by `{port.name if port else '?'}` but has "
                f"neither init= contents nor a copy filling it, so there is nothing "
                f"to read."
            )
        node = ast.Name(id=self._station_name(port), ctx=ast.Load())
        layout = _banked_layout(brick)
        if layout is not None:
            # `extra is None` is the whole-brick read, `io.tab` with no index.
            # It used to reach `len(None)` and die as a TypeError, which is the
            # same bug in miniature: the check was there and could not say what
            # was wrong.
            if extra is None or len(extra) != 1:
                given = "no index" if extra is None else f"{len(extra)}"
                raise SPMWMemoryError(
                    f"`{brick.name}` is banked, so it takes one linear index; "
                    f"`{port.name if port else '?'}` gave {given}. Banking stores "
                    f"it as [banks][rows], so the whole brick and a multi-axis "
                    f"index no longer name anything the reader means."
                )
            return self.banked_subscript(node, extra[0], layout)
        if extra:
            idx = (
                ast.Tuple(elts=list(extra), ctx=ast.Load())
                if len(extra) > 1
                else extra[0]
            )
            node = ast.Subscript(value=node, slice=idx, ctx=ast.Load())
        return node

    # -- movers ------------------------------------------------------------

    def mover_function(self, mover):
        """One loader or drain as a function over its bundle.

        It takes the tensor it walks and one ``index`` per bundle axis, and
        declares its channel as a one-element stream array, which
        :func:`allo.spmw.lower_mlir.hoist_streams` turns into the trailing
        parameter.  Returns the function and its stream names in that order.
        """
        bundle = mover.bundle
        geom = _geometry(bundle)
        pids = [f"_q{i}" for i in range(len(geom.dense))]
        # A table the body indexes at runtime -- which site a member is, or
        # where a lambda index map sends it -- is declared as a constant local
        # of the function, the shape the builder loads from.
        prologue = []

        def declare(prefix, table):
            local = f"_tab{len(prologue)}"
            prologue.append(
                ast.AnnAssign(
                    target=ast.Name(id=local, ctx=ast.Store()),
                    annotation=self.type_ann(table_type(), tuple(table.shape)),
                    value=ast.Name(id=self._inject(prefix, table), ctx=ast.Load()),
                    simple=1,
                )
            )
            return local

        site_exprs = geom.site_exprs(pids, declare=declare)
        chan = ast.Subscript(
            value=ast.Name(id="chan", ctx=ast.Load()),
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        )
        tensor = mover.tensor
        names = {axis: site_exprs[i] for i, axis in enumerate(bundle.placement.axes)}
        loop_var = "_t"
        extent = mover.extent if mover.extent is not None else 1
        subs = _offset(
            self._mover_subscripts(mover, names, loop_var, geom, pids, declare),
            tensor,
        )
        elem = ast.Subscript(
            value=ast.Name(id=f"local_{tensor.base.name}", ctx=ast.Load()),
            slice=ast.Tuple(elts=subs, ctx=ast.Load()) if len(subs) > 1 else subs[0],
            ctx=ast.Load(),
        )
        block = mover.binding.extras.get("block", ())
        fam = mover.family
        stream = ast.Subscript(
            value=ast.Name(id="Stream", ctx=ast.Load()),
            slice=ast.Tuple(
                elts=[
                    self.type_ann(fam.dtype, fam.block),
                    ast.Constant(value=int(fam.depth)),
                ],
                ctx=ast.Load(),
            ),
            ctx=ast.Load(),
        )
        decl = ast.AnnAssign(
            target=ast.Name(id="chan", ctx=ast.Store()),
            annotation=ast.Subscript(
                value=stream, slice=ast.Constant(value=1), ctx=ast.Load()
            ),
            value=None,
            simple=1,
        )
        loop = ast.For(
            target=ast.Name(id=loop_var, ctx=ast.Store()),
            iter=_call("range", [ast.Constant(value=int(extent))]),
            body=self._transfer(mover, chan, elem, block),
            orelse=[],
        )
        params = [
            ast.arg(
                arg=f"local_{tensor.base.name}",
                annotation=self.type_ann(tensor.dtype, tensor.base.shape),
            )
        ]
        params += [
            ast.arg(arg=pid, annotation=self.type_ann(pid_type(), ())) for pid in pids
        ]
        fn = ast.FunctionDef(
            name=mover.name,
            args=ast.arguments(
                posonlyargs=[],
                args=params,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[decl] + prologue + [loop],
            decorator_list=[],
            returns=None,
            type_params=[],
        )
        return fn, ["chan_0"]

    def _mover_subscripts(self, mover, names, loop_var, geom, pids, declare):
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
            # A lambda is the escape hatch: it is evaluated over the whole
            # (member, step) domain at elaboration and carried as a constant
            # table the body indexes by its own position and step.
            tname = declare(f"IX_{mover.name}_", _lambda_table(mover, geom))
            rank = mover.tensor.rank - len(block)
            subs = [
                ast.Subscript(
                    value=ast.Name(id=tname, ctx=ast.Load()),
                    slice=ast.Tuple(
                        elts=[
                            _copy_expr(geom.member_expr(pids)),
                            ast.Name(id=loop_var, ctx=ast.Load()),
                            ast.Constant(value=k),
                        ],
                        ctx=ast.Load(),
                    ),
                    ctx=ast.Load(),
                )
                for k in range(rank)
            ]
        for k in range(len(block)):
            subs.append(ast.Name(id=f"_b{k}", ctx=ast.Load()))
        return subs

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
        ann = self.type_ann(mover.bundle.port.dtype, block)
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


# ---------------------------------------------------------------------------
# Body rewriting
# ---------------------------------------------------------------------------


class _BodyRewriter(ast.NodeTransformer):
    """Rewrites port touches into channel and tensor accesses."""

    # pylint: disable=too-many-instance-attributes,too-many-arguments

    def __init__(
        self,
        lowering,
        placement,
        signature,
        routing,
        sites,
        pids,
        io_name,
        site_name,
        fixed=None,
        tree=None,
    ):
        super().__init__()
        self.low = lowering
        self.placement = placement
        self.signature = signature
        self.routing = routing
        self.sites = sites
        self.pids = pids
        self.io = io_name
        self.site = site_name
        self.iface = placement.iface
        self.drops = 0
        # Coordinates known at compile time, because the placement specialises
        # those axes: the body sees a literal, so a loop bounded by one has a
        # constant trip count. The rest arrive as the function's parameters.
        self.fixed = dict(fixed or {})
        # Names a body binds to its coordinates -- `row, _col = site.rank` --
        # stand for the parameters directly wherever they are read, so a loop
        # bounded by one keeps an affine bound. A name the body also assigns
        # elsewhere is left as the variable it is.
        self.aliases = {}
        self._stores = {}
        for node in ast.walk(tree) if tree is not None else ():
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self._stores[node.id] = self._stores.get(node.id, 0) + 1

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
        """This site's end of ``port`` as an expression, or None if it has none."""
        raise NotImplementedError("a rewriter says how its ports are reached")

    def _coord(self, axis):
        """The site's coordinate on ``axis``: a literal if specialised, else a pid."""
        if axis in self.fixed:
            return ast.Constant(value=int(self.fixed[axis]))
        return ast.Name(id=self.pids[axis], ctx=ast.Load())

    def _rank_value(self, node):
        """What ``s, b = site.rank`` binds: the coordinates, shaped like the target.

        `(slot,) = site.rank` on a 1-D placement unpacks a one-tuple, and handing
        it a bare name would emit `slot, = _p0`, which is a scalar unpack.
        """
        names = [self._coord(a) for a in range(len(self.pids))]
        unpacking = len(node.targets) == 1 and isinstance(
            node.targets[0], (ast.Tuple, ast.List)
        )
        if unpacking or len(names) > 1:
            return ast.Tuple(elts=names, ctx=ast.Load())
        return names[0]

    # -- visits

    def visit_Call(self, node):
        self.generic_visit(node)
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in {"get", "put"}:
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
            raise SPMWBindingError(
                f"`{self.placement.name}.{port.name}` is unbound here, so the put "
                f"is a discard -- but it is used as a value, and a discard has "
                f"none. Write it as its own statement."
            )
        return _call_node(
            ast.Attribute(value=sub, attr=verb, ctx=ast.Load()), node.args
        )

    def visit_Expr(self, node):
        """A put on an unbound Out is a discard -- of the put, not the statement.

        The value being put may itself read a channel, and those reads consume
        tokens whether or not anyone wants the result. Dropping the whole
        statement would take them with it and quietly change what the program
        computes.
        """
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            if value.func.attr == "put":
                port = self._port_of(value.func.value)
                if (
                    port is not None
                    and port.protocol == STREAM
                    and self._subscript(port) is None
                ):
                    return self._discard(value)
        self.generic_visit(node)
        return node

    def _discard(self, call):
        """Keep whatever the discarded put was going to send, if it had effects."""
        if not call.args:
            return None
        block_port = self._block_get(_FakeAssign(call.args[0]))
        arg = self.visit(call.args[0])
        if not _reads_a_channel(arg):
            return None  # nothing observable to keep
        name = f"_drop{self.drops}"
        self.drops += 1
        target = ast.Name(id=name, ctx=ast.Store())
        if block_port is not None:
            return ast.AnnAssign(
                target=target,
                annotation=self.low.type_ann(block_port.dtype, block_port.shape),
                value=arg,
                simple=1,
            )
        return ast.Assign(targets=[target], value=arg)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.aliases:
            return _copy_expr(self.aliases[node.id])
        return node

    def visit_Assign(self, node):
        # `s, b = site.rank` restates the coordinates the function takes. Bound
        # once to plain names, those names become the coordinates themselves.
        if self.site and _is_site_rank(node.value, self.site):
            names = _plain_targets(node)
            if names is not None and all(self._stores.get(n, 0) == 1 for n in names):
                for axis, name in enumerate(names):
                    self.aliases[name] = self._coord(axis)
                return None
            return ast.Assign(targets=node.targets, value=self._rank_value(node))
        block_port = self._block_get(node)
        node.value = self.visit(node.value)
        node.targets = [self._store_target(t) for t in node.targets]
        if block_port is not None:
            # A block-valued read is declared, which is the shape the dataflow
            # model expects for a stream of blocks.
            return ast.AnnAssign(
                target=node.targets[0],
                annotation=self.low.type_ann(block_port.dtype, block_port.shape),
                value=node.value,
                simple=1,
            )
        return node

    def _block_get(self, node):
        """The port of a ``x = io.p.get()`` whose token is a block, if any."""
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return None
        value = node.value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)):
            return None
        if value.func.attr != "get":
            return None
        port = self._port_of(value.func.value)
        if port is None or port.protocol != STREAM or not port.shape:
            return None
        return port

    def _store_target(self, target):
        port = self._port_of(target)
        if port is not None and port.protocol != STREAM:
            sub = self.low.mem_subscript(self.placement, port, self.pids)
            return _store(sub)
        if isinstance(target, ast.Subscript):
            port = self._port_of(target.value)
            if port is not None and port.protocol != STREAM:
                extra = [self.visit(e) for e in _index_elts(target.slice)]
                sub = self.low.mem_subscript(self.placement, port, self.pids, extra)
                return _store(sub)
        return self.visit(target)

    def visit_AnnAssign(self, node):
        """A bare declaration allocates.

        `u: csample` reserves a buffer in the source language, but with no value
        the tracer takes `u` for a scalar and reads `u[0]` as a bit-slice. Zeroing
        it is also what the reference simulator does, so the two agree.
        """
        self.generic_visit(node)
        if node.value is None:
            node.value = ast.Constant(value=0)
        node.annotation = self.low.canonical_annotation(node.annotation)
        return node

    def visit_Subscript(self, node):
        if (
            self.site
            and _is_site_rank(node.value, self.site)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            # `site.rank[k]` is this site's k-th coordinate, which the function
            # already has a name for; a tuple subscript would leave the builder
            # to fold it.
            return self._coord(node.slice.value)
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
                    elts=[self._coord(a) for a in range(len(self.pids))],
                    ctx=ast.Load(),
                )
        self.generic_visit(node)
        return node


# ---------------------------------------------------------------------------
# Bundle geometry
# ---------------------------------------------------------------------------


class _Geometry:
    """How a bundle's members are addressed by a mover kernel's pids.

    When the bundle's sites are a product of arithmetic progressions, a member
    is addressed by its coordinates. Otherwise it is walked flatly, one pid over
    the member list, with the site read from a table -- which is the escape
    hatch the design reserves for an unbound set that has no dense shape.
    """

    def __init__(self, bundle):
        self.bundle = bundle
        self.rank = len(bundle.placement.grid)
        self.axes = []
        for a in range(self.rank):
            values = sorted({s[a] for s in bundle.sites})
            step = _progression(values)
            self.axes.append((values, step))
        self.flat = not bundle.is_dense or any(
            self.axes[a][1] is None
            for a in range(self.rank)
            if len(self.axes[a][0]) > 1
        )
        if self.flat:
            self.varying = []
            self.dense = [max(1, len(bundle.sites))]
        else:
            self.varying = [a for a in range(self.rank) if len(self.axes[a][0]) > 1]
            self.dense = [len(self.axes[a][0]) for a in self.varying] or [1]

    def site_exprs(self, pids, declare=None):
        """Per grid axis, the source expression giving this member's coordinate."""
        if self.flat:
            import numpy as np  # pylint: disable=import-outside-toplevel

            # A constant table the body indexes by its own position: a
            # numpy array rather than a tuple, because a numpy constant is a
            # memory the builder can load from at a runtime index.
            table = np.array(
                [[int(c) for c in site] for site in self.bundle.sites], dtype=np.int32
            ).reshape(len(self.bundle.sites), self.rank)
            name = declare("SITE_", table)
            return [
                ast.Subscript(
                    value=ast.Name(id=name, ctx=ast.Load()),
                    slice=ast.Tuple(
                        elts=[
                            ast.Name(id=pids[0], ctx=ast.Load()),
                            ast.Constant(value=a),
                        ],
                        ctx=ast.Load(),
                    ),
                    ctx=ast.Load(),
                )
                for a in range(self.rank)
            ]
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
        if self.flat:
            return ast.Name(id=pids[0], ctx=ast.Load())
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

    def member_from_pids(self, pids):
        """This site's member position, as an expression over the kernel's pids.

        Returns None when the bundle is not a product of arithmetic progressions,
        in which case a lookup is the honest spelling.
        """
        if not self.bundle.is_dense:
            return None
        strides, acc = [], 1
        for extent in reversed(self.dense):
            strides.insert(0, acc)
            acc *= extent
        node = None
        for k, axis in enumerate(self.varying):
            values, step = self.axes[axis]
            if step is None:
                return None
            term = ast.Name(id=pids[axis], ctx=ast.Load())
            if values[0]:
                term = ast.BinOp(
                    left=term, op=ast.Sub(), right=ast.Constant(value=int(values[0]))
                )
            if step != 1:
                term = ast.BinOp(
                    left=term, op=ast.FloorDiv(), right=ast.Constant(value=int(step))
                )
            if strides[k] != 1:
                term = ast.BinOp(
                    left=term, op=ast.Mult(), right=ast.Constant(value=int(strides[k]))
                )
            node = (
                term if node is None else ast.BinOp(left=node, op=ast.Add(), right=term)
            )
        return node if node is not None else ast.Constant(value=0)

    def site_of(self, position):
        """The site a flat member position names."""
        if self.flat:
            return self.bundle.sites[position]
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


def _tokens_per_site(tensor, bundle):
    """How many tokens a launch moves through one member of the bundle."""
    shape = getattr(tensor, "shape", None) or ()
    volume = 1
    for extent in shape:
        volume *= int(extent)
    return max(1, -(-volume // max(1, len(bundle))))


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


def _wiring_classes(placement, resolution):
    """Sites grouped by what an arm has to say about them, biggest group first.

    A site's *signature* decides which body runs, but two sites with the same
    signature can still be wired to different channel families -- a port can take
    part in more than one port pair, or reach its neighbour by a coordinate link
    at one site and by a key at another. One subscript cannot serve both, so the
    routing is part of the class as well.

    In the ordinary case every site with a signature shares its routing, so this
    is the signature partition and the arm count is unchanged.

    A *specialised* axis joins the key: sites that differ along it get different
    roles even when they are wired identically, which is what lets the body see
    its position as a literal. See :func:`allo.spmw.place`.
    """
    axes = getattr(placement, "specialise", ()) or ()
    groups = {}
    for (
        site,
        signature,
    ) in placement.topology._bound.items():
        fixed = tuple(int(site[a]) for a in axes)
        key = (signature, resolution.routing(site, signature), fixed)
        groups.setdefault(key, (signature, {}, []))[2].append(site)
    ordered = sorted(
        groups.items(),
        key=lambda kv: (-len(kv[1][2]), kv[0][2], sorted(p.name for p in kv[0][0])),
    )
    return [
        (signature, dict(routing), sites)
        for (signature, routing, _fixed), (signature_, _, sites) in ordered
    ]


def _lambda_table(mover, geom):
    """Evaluate a lambda index map over the whole (member, step) domain.

    Emitted as a numpy constant indexed by member, because a site slices it at
    compile time and the tracer accepts that where it rejects a bare tuple
    global.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    extent = mover.extent if mover.extent is not None else 1
    rows = []
    for pos in range(len(mover.bundle.sites)):
        site = geom.site_of(pos)
        env = dict(mover.bundle.placement.env(site), __coords__=site)
        steps = []
        for t in range(extent):
            idx = mover.imap.eval(env, step=t if mover.imap.has_time else None)
            steps.append([int(v) for v in idx])
        rows.append(steps)
    return np.array(rows, dtype=np.int32)


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


def _offset(subs, tensor):
    """Shift a view's own subscripts into the parent array they alias."""
    offsets = getattr(tensor, "offsets", None)
    if not offsets or not any(offsets):
        return subs
    out = []
    for k, sub in enumerate(subs):
        off = offsets[k] if k < len(offsets) else 0
        if off:
            sub = ast.BinOp(left=sub, op=ast.Add(), right=ast.Constant(value=int(off)))
        out.append(sub)
    return out


def _src_names(names):
    return {axis: ast.unparse(node) for axis, node in names.items()}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


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


class _FakeAssign:
    """Just enough of an Assign for the block-get test to read."""

    __slots__ = ("targets", "value")

    def __init__(self, value):
        self.targets = [ast.Name(id="_", ctx=ast.Store())]
        self.value = value


def _reads_a_channel(node):
    """Whether an expression consumes a token, and so must survive a discard."""
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "get"
        for child in ast.walk(node)
    )


def _fill_empty_suites(node):
    """Give any suite an elision emptied a `pass`.

    A loop whose only statement was a discarded put still has to be a loop; an
    empty body would unparse to source that cannot be compiled at all.
    """
    # Only `body`: an empty `orelse` means there is no else clause, which is
    # exactly what an elided else should become.
    for child in ast.walk(node):
        if isinstance(child, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
            if not child.body:
                child.body = [ast.Pass()]


def _is_docstring(node):
    """A bare string statement documents the source, not the emitted program."""
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _plain_targets(node):
    """The names an assignment binds, one per grid axis, or None.

    `s, b = site.rank` and `(slot,) = site.rank` bind names; `xs = site.rank`
    binds the tuple, which is not a coordinate and is left alone.
    """
    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, (ast.Tuple, ast.List)):
        return None
    if not all(isinstance(elt, ast.Name) for elt in target.elts):
        return None
    return [elt.id for elt in target.elts]


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


class _TypeStandIn:
    """A stand-in for an Allo scalar type when Allo is not importable.

    The elaboration core runs without a built compiler; only building needs
    the real type, and by then Allo is there.
    """

    shape = ()

    def __init__(self, spelling):
        self.spelling = spelling

    def __str__(self):
        return self.spelling


def pid_type():
    """The type a site coordinate has as a function parameter: MLIR's ``index``."""
    try:
        from allo.ir.types import index  # pylint: disable=import-outside-toplevel

        return index
    except Exception:  # pylint: disable=broad-except
        return _TypeStandIn("index")


def table_type():
    """The element type of a coordinate or index table: a 32-bit integer."""
    try:
        from allo.ir.types import int32  # pylint: disable=import-outside-toplevel

        return int32
    except Exception:  # pylint: disable=broad-except
        return _TypeStandIn("i32")


__all__ = ["Lowering", "pid_type", "table_type"]
