# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The index algebra bindings are written in.

Every binding is a function *from the bundle's world into the tensor's*: its
domain is the bundle's axes plus, for streams, time; its codomain is the tensor's
index space, which is what an ``index=`` tuple spells, one entry per tensor axis.

Positions belong to the tensor; names belong to the fabric.  Grid coordinates
appear only as symbols inside entries, so evaluation runs from site to element
and there is no positional grid-to-tensor mapping to get wrong.
"""

from .errors import SPMWBindingError

# The marker for the port side's own axes: a stream's token counter, a memory
# port's block axes.  ``...`` is Python's Ellipsis; NumPy already reads it as
# "the axes I am not naming", and here the unnamed axes are exactly the sequence.
TIME = Ellipsis


class Expr:
    """A node in an affine-ish index expression over axis symbols."""

    def __add__(self, other):
        return BinOp("+", self, _lift(other))

    def __radd__(self, other):
        return BinOp("+", _lift(other), self)

    def __sub__(self, other):
        return BinOp("-", self, _lift(other))

    def __rsub__(self, other):
        return BinOp("-", _lift(other), self)

    def __mul__(self, other):
        return BinOp("*", self, _lift(other))

    def __rmul__(self, other):
        return BinOp("*", _lift(other), self)

    def __floordiv__(self, other):
        return BinOp("//", self, _lift(other))

    def __rfloordiv__(self, other):
        return BinOp("//", _lift(other), self)

    def __mod__(self, other):
        return BinOp("%", self, _lift(other))

    def __rmod__(self, other):
        return BinOp("%", _lift(other), self)

    def __neg__(self):
        return BinOp("-", Const(0), self)

    def eval(self, env):
        raise NotImplementedError

    def axes(self):
        """Every axis symbol this expression reads."""
        raise NotImplementedError


def _lift(value):
    if isinstance(value, Expr):
        return value
    if isinstance(value, int):
        return Const(value)
    raise SPMWBindingError(
        f"{value!r} is not usable in an index expression; expected an axis "
        f"symbol or an integer."
    )


class Const(Expr):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = int(value)

    def eval(self, env):
        return self.value

    def axes(self):
        return frozenset()

    def __repr__(self):
        return str(self.value)


class Axis(Expr):
    """A named index-space axis with a known extent.

    Placement axes come straight from the grid (``P.rows``); ``split`` derives
    further ones.  Extents are always known, which is what keeps every binding
    check decidable and exact.
    """

    __slots__ = ("name", "extent", "source", "_resolve", "definition")

    def __init__(self, name, extent, source=None, resolve=None, definition=None):
        self.name = name
        self.extent = extent
        self.source = source
        self._resolve = resolve
        # A derived axis (from `split`) carries the expression that defines it,
        # so it can be rendered as source and not merely evaluated.
        self.definition = definition

    def eval(self, env):
        if self in env:
            return env[self]
        if self.definition is not None:
            return self.definition.eval(env)
        if self._resolve is not None:
            return self._resolve(env)
        raise SPMWBindingError(f"axis `{self.name}` is unbound in this evaluation")

    def axes(self):
        return frozenset({self})

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        return self.name


class BinOp(Expr):
    __slots__ = ("op", "lhs", "rhs")

    _OPS = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "//": lambda a, b: a // b,
        "%": lambda a, b: a % b,
    }

    def __init__(self, op, lhs, rhs):
        self.op, self.lhs, self.rhs = op, lhs, rhs

    def eval(self, env):
        return self._OPS[self.op](self.lhs.eval(env), self.rhs.eval(env))

    def axes(self):
        return self.lhs.axes() | self.rhs.axes()

    def __repr__(self):
        return f"({self.lhs} {self.op} {self.rhs})"


def split(axis, factor=None, size=None):
    """Name an axis's factors.

    ``factor=G`` gives the outer extent, ``size=d`` the inner; the other is
    derived, and giving both is a checked redundancy.  Splits compose, so no
    n-way form is needed.
    """
    if not isinstance(axis, Axis):
        raise SPMWBindingError(f"split expects an axis symbol, got {axis!r}")
    extent = axis.extent
    if factor is None and size is None:
        raise SPMWBindingError("split needs `factor=` or `size=`")
    if factor is None:
        factor = extent // size
    elif size is None:
        size = extent // factor
    if factor * size != extent:
        raise SPMWBindingError(
            f"split of `{axis.name}` (extent {extent}) into factor={factor} x "
            f"size={size} does not cover it: {factor} * {size} != {extent}."
        )
    outer = Axis(
        f"{axis.name}.o", factor, source=axis, definition=BinOp("//", axis, Const(size))
    )
    inner = Axis(
        f"{axis.name}.i", size, source=axis, definition=BinOp("%", axis, Const(size))
    )
    return outer, inner


class IndexMap:
    """A binding's site-to-element map.

    Either a tuple of expressions -- one entry per tensor axis, with ``...``
    marking the port side's own axes -- or a lambda, the escape hatch for maps
    that are not affine in the axes.
    """

    __slots__ = ("spec", "is_lambda", "time_pos", "rank", "wants_step")

    def __init__(self, spec, rank):
        self.rank = rank
        self.is_lambda = callable(spec)
        self.spec = spec
        self.time_pos = None
        # Set by the binding once the port side's rank is known: a lambda over
        # `rank` arguments is indexed by site alone, one over `1 + rank` by
        # (step, site).
        self.wants_step = False
        if not self.is_lambda:
            if not isinstance(spec, tuple):
                spec = (spec,)
                self.spec = spec
            if len(spec) != rank:
                raise SPMWBindingError(
                    f"index= has {len(spec)} entries but the tensor has rank "
                    f"{rank}. The tuple is shaped like the tensor: one entry per "
                    f"tensor axis."
                )
            marks = [i for i, e in enumerate(spec) if e is TIME]
            if len(marks) > 1:
                raise SPMWBindingError(
                    "index= names `...` more than once; exactly one axis is the "
                    "streamed sequence."
                )
            self.time_pos = marks[0] if marks else None

    @property
    def has_time(self):
        return self.time_pos is not None

    def eval(self, env, step=None):
        """Resolve to a concrete tensor subscript tuple."""
        if self.is_lambda:
            args = list(env["__coords__"])
            if self.wants_step:
                args = [step] + args
            out = self.spec(*args)
            return out if isinstance(out, tuple) else (out,)
        out = []
        for entry in self.spec:
            if entry is TIME:
                out.append(step)
            elif isinstance(entry, Expr):
                out.append(entry.eval(env))
            elif isinstance(entry, int):
                out.append(entry)
            else:
                raise SPMWBindingError(
                    f"index= entry {entry!r} is neither an axis expression, an "
                    f"integer, nor `...`."
                )
        return tuple(out)

    def __repr__(self):
        if self.is_lambda:
            return "index=<lambda>"
        return (
            "index=("
            + ", ".join("..." if e is TIME else repr(e) for e in self.spec)
            + ")"
        )


class SliceMap:
    """A site-to-slice map: the shape a bare (``index=``-less) binding may take.

    Three forms, all of them positional identities rather than searches:

    ``identity``
        grid axes address elements one-for-one -- §3.5's ``shard(w_tile, into=P.w)``.
    ``concat``
        site axes lead, block axes trail, extents slot-for-slot -- §3.1's
        ``gather(C, from_=P.c)``.
    ``blocked``
        site ``(i, j)`` owns the tensor's ``(i, j)``-th block -- §3.2's
        ``shard(C, from_=T.c)``.

    Coverage is checked arithmetically from the shape equation rather than by
    enumerating elements, which keeps the check exact and cheap at real sizes.
    """

    __slots__ = ("kind", "grid", "block", "shape", "axis_of", "rank")

    def __init__(self, kind, grid, block, shape, axis_of):
        self.kind = kind
        self.grid = tuple(grid)
        self.block = tuple(block)
        self.shape = tuple(shape)
        # axis_of[t] is the grid axis addressing tensor axis t, or None.
        self.axis_of = tuple(axis_of)
        self.rank = len(self.shape)

    @property
    def is_lambda(self):
        return False

    @property
    def has_time(self):
        return False

    def slice_for(self, site):
        """``(start, size)`` per tensor axis for one site."""
        out = []
        for t, bound in enumerate(self.shape):
            g = self.axis_of[t]
            size = self.block[t] if t < len(self.block) and self.block[t] else 1
            if g is None:
                out.append((0, bound))
            elif self.kind == "blocked":
                out.append((site[g] * size, size))
            else:
                out.append((site[g], 1))
        return tuple(out)

    def per_site_sizes(self):
        """The per-axis extent one site owns -- what the union check counts by."""
        sizes = []
        for t, bound in enumerate(self.shape):
            g = self.axis_of[t]
            if g is None:
                sizes.append(bound)
            elif self.kind == "blocked":
                sizes.append(
                    self.block[t] if t < len(self.block) and self.block[t] else 1
                )
            else:
                sizes.append(1)
        return tuple(sizes)

    def tiles_exactly(self, nsites):
        """Whether the slices partition the tensor with no gap and no overlap."""
        covered = 1
        for t, bound in enumerate(self.shape):
            g = self.axis_of[t]
            if g is None:
                covered *= bound
                continue
            size = self.block[t] if t < len(self.block) and self.block[t] else 1
            if self.kind == "blocked":
                if self.grid[g] * size != bound:
                    return False
                covered *= bound
            else:
                if self.grid[g] != bound:
                    return False
                covered *= bound
        total = 1
        for bound in self.shape:
            total *= bound
        return covered == total and nsites > 0

    def __repr__(self):
        return f"<slicemap {self.kind} grid={self.grid} block={self.block}>"


def check_bounds(imap, members, shape, extent, where, write=False, block=()):
    """Bounds-check every computed index over the whole (step, site) domain.

    A read map may name the same element for many (step, site) pairs -- broadcast
    is free -- while a write map must tile its tensor disjointly.
    """
    if isinstance(imap, SliceMap):
        return _check_slices(imap, members, shape, where, write)
    # A port carrying a block addresses only the tensor's leading axes; the
    # trailing ones are the block itself, which is why the tuple is shorter.
    lead = shape[: len(shape) - len(block)] if block else shape
    seen = {}
    steps = range(extent) if extent is not None else [None]
    for coords, env in members:
        for step in steps:
            idx = imap.eval(dict(env, __coords__=coords), step=step)
            if len(idx) != len(lead):
                raise SPMWBindingError(
                    f"{where}: index map produced {len(idx)} subscripts, but the "
                    f"tensor has {len(lead)} axis(es) left for it to address"
                    + (f" after the port's {list(block)} block" if block else "")
                    + "."
                )
            for axis, (val, bound) in enumerate(zip(idx, lead)):
                if not 0 <= val < bound:
                    raise SPMWBindingError(
                        f"{where}: index {idx} is out of bounds on axis {axis} "
                        f"(extent {bound}), reached at site {coords}"
                        + (f", step {step}" if step is not None else "")
                        + "."
                    )
            if write:
                if idx in seen:
                    raise SPMWBindingError(
                        f"{where}: element {idx} is written by two sources -- "
                        f"site {seen[idx]} and site {coords}. Every output element "
                        f"must have exactly one writer."
                    )
                seen[idx] = coords
    return seen


def _check_slices(imap, members, shape, where, write):
    """Validate a :class:`SliceMap` against the tensor it binds."""
    if tuple(shape) != imap.shape:
        raise SPMWBindingError(
            f"{where}: the binding covers {list(imap.shape)} but the tensor is "
            f"{list(shape)}."
        )
    for coords, _env in members:
        for axis, (start, size) in enumerate(imap.slice_for(coords)):
            if start < 0 or start + size > shape[axis]:
                raise SPMWBindingError(
                    f"{where}: site {coords} owns [{start}, {start + size}) on "
                    f"axis {axis}, which is outside the tensor's extent "
                    f"{shape[axis]}."
                )
    covered = {
        tuple(start for start, _ in imap.slice_for(coords)): coords
        for coords, _env in members
    }
    if write and not imap.tiles_exactly(len(members)):
        uncovered = _uncovered_range(imap, shape)
        raise SPMWBindingError(
            f"{where}: the writers do not tile the destination. {uncovered} "
            f"Every output element must have exactly one writer -- supply the "
            f"missing extent from another phase, a wider instantiation, or a "
            f"second placement owning the other slice."
        )
    return covered


def _uncovered_range(imap, shape):
    """Name the axis and range a partial write leaves unwritten."""
    for axis, bound in enumerate(shape):
        g = imap.axis_of[axis]
        if g is None:
            continue
        size = imap.block[axis] if axis < len(imap.block) and imap.block[axis] else 1
        reach = imap.grid[g] * size if imap.kind == "blocked" else imap.grid[g]
        if reach != bound:
            return f"Axis {axis} is covered up to {reach} of {bound}: [{reach}, {bound}) is unwritten."
    return "Some elements have no writer."


# ---------------------------------------------------------------------------
# Rendering expressions back to source
# ---------------------------------------------------------------------------


def to_source(expr, names, time=None):
    """Render an index expression as Python source.

    ``names`` maps each placement axis to the source expression naming it in the
    generated program -- a pid, or an arithmetic combination of pids.  This is
    what lets an affine binding stay an affine subscript in the emitted code
    instead of collapsing into a lookup table.
    """
    if expr is TIME:
        if time is None:
            raise SPMWBindingError("`...` appears where there is no token counter")
        return time
    if isinstance(expr, int):
        return str(expr)
    if isinstance(expr, Const):
        return str(expr.value)
    if isinstance(expr, Axis):
        if expr in names:
            return names[expr]
        if expr.definition is not None:
            return to_source(expr.definition, names, time)
        raise SPMWBindingError(
            f"axis `{expr.name}` has no source spelling in this scope"
        )
    if isinstance(expr, BinOp):
        lhs = to_source(expr.lhs, names, time)
        rhs = to_source(expr.rhs, names, time)
        return f"({lhs} {expr.op} {rhs})"
    raise SPMWBindingError(f"cannot render {expr!r} as source")


def renderable(expr, names):
    """Whether ``expr`` can be rendered with the axis spellings in ``names``."""
    try:
        to_source(expr, names, time="__t")
        return True
    except SPMWBindingError:
        return False
