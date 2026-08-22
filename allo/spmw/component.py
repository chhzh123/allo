# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Components: units execute, fabrics elaborate.

A ``unit`` body is imperative and runs per firing; the compiler schedules it.  A
``fabric`` body is declarative and runs once at compile time to build a graph;
the compiler elaborates it.  The two decorators are the mode markers, and the
compiler forks behind them.
"""

import ast
import inspect
import textwrap

from .errors import SPMWError, SPMWPlacementError
from .interface import Interface, unwrap
from .ports import STREAM, IN, OUT  # noqa: F401


class Site:
    """A placed unit's coordinates.

    Declared as a second parameter when a body needs them::

        def pe(io: MacIO, site: spmw.Site): ...

    Rank is deliberately not a port: its *absence* is a checkable property, since
    a rank-independent body folds without predication.
    """

    __slots__ = ("rank", "grid")

    def __init__(self, rank, grid):
        self.rank = tuple(rank)
        self.grid = tuple(grid)

    def __repr__(self):
        return f"Site(rank={self.rank}, grid={self.grid})"


class Role:
    """A variant body for sites missing a declared set of ports."""

    __slots__ = ("fn", "unbound", "tree", "wants_site", "name")

    def __init__(self, fn, unbound):
        self.fn = fn
        self.unbound = frozenset(unbound)
        self.name = fn.__name__
        self.tree = _parse_body(fn)
        self.wants_site = _wants_site(fn)


class Unit:
    """A leaf component: one program, written against an Interface."""

    def __init__(self, fn, state=None):
        self.fn = fn
        self.name = fn.__name__
        self.state = dict(state or {})
        self.iface = _iface_of(fn)
        self.tree = _parse_body(fn)
        self.wants_site = _wants_site(fn)
        self.roles = []
        _check_io_not_rebound(self.tree, self.name)
        _check_port_names(self.tree, self.iface, self.name)
        check_directions(self.tree, self.iface, self.name)

    def role(self, unbound=()):
        """Register a variant body for sites whose listed ports are unbound.

        A role body may not touch its declared-unbound ports; that is a
        compile-time error here, not a deadlock later.
        """
        unbound = tuple(unbound)
        for sym in unbound:
            if not unwrap(self.iface).owns(sym):
                raise SPMWError(
                    f"`{self.name}.role` names `{sym}`, which is not a port of "
                    f"`{unwrap(self.iface).__name__}`."
                )
            if sym.protocol != STREAM:
                raise SPMWError(
                    f"`{self.name}.role` declares `{sym}` unbound, but it is "
                    f"{sym.type_str()}. A site's signature is decided by its "
                    f"links, and a memory port is bound by a fabric binding "
                    f"instead -- such a role could never be selected."
                )

        def decorate(fn):
            role = Role(fn, unbound)
            _check_io_not_rebound(role.tree, role.name)
            _check_port_names(role.tree, self.iface, role.name)
            check_directions(role.tree, self.iface, role.name)
            _check_untouched(role.tree, unbound, role.name)
            self.roles.append(role)
            return role

        return decorate

    def body_for(self, signature):
        """The body serving a site whose *bound* port set is ``signature``.

        Among roles, the one declaring the largest matching unbound set wins.
        Otherwise the default body serves, because boundary differences that are
        only about *inputs* are bindings, not roles: a loader feeds the rim so
        the interior body stays total.  Whether such a binding actually exists is
        settled at build, where the diagnostic can name it.
        """
        iface = unwrap(self.iface)
        missing = frozenset(sym for sym in iface.ports() if sym not in signature)
        matching = [role for role in self.roles if role.unbound <= missing]
        if not matching:
            return self
        widest = max(len(role.unbound) for role in matching)
        best = [role for role in matching if len(role.unbound) == widest]
        if len(best) > 1:
            names = ", ".join(sorted(role.name for role in best))
            absent = ", ".join(sorted(sym.name for sym in missing))
            raise SPMWPlacementError(
                f"sites missing {{{absent}}} match {len(best)} equally specific "
                f"roles ({names}), so which one runs would come down to the order "
                f"they were declared in. Give one of them the wider unbound set it "
                f"actually serves."
            )
        return best[0]

    def reads(self, body=None):
        """The ``In`` and ``MemIn`` ports a body actually touches."""
        tree = (body or self).tree
        iface = unwrap(self.iface)
        touched = set()
        for sym, _ in port_uses(tree, iface):
            if sym.direction == IN:
                touched.add(sym)
        return frozenset(touched)

    def __repr__(self):
        return f"<unit {self.name}({unwrap(self.iface).__name__})>"


class Fabric:
    """A composite component: declares memories, places components, wires bindings."""

    def __init__(self, fn, io=None):
        self.fn = fn
        self.name = fn.__name__
        self.io = io
        self.signature = inspect.signature(fn)
        self.tree = _parse_body(fn)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __repr__(self):
        shown = f"({unwrap(self.io).__name__})" if self.io is not None else ""
        return f"<fabric {self.name}{shown}>"


def unit(fn=None, *, state=None):
    """Declare a leaf component.

    Usable bare (``@spmw.unit``) or with keywords (``@spmw.unit(state=...)``).
    """
    if fn is None:
        return lambda f: Unit(f, state=state)
    return Unit(fn, state=state)


def fabric(fn=None, *, io=None):
    """Declare a composite component.

    ``io=`` makes the fabric itself placeable, which is what closes the
    composition loop: what a placement exposes is the same kind of thing a unit
    declares.
    """
    if fn is None:
        return lambda f: Fabric(f, io=io)
    return Fabric(fn, io=io)


# ---------------------------------------------------------------------------
# Source introspection
# ---------------------------------------------------------------------------


def _parse_body(fn):
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    func = tree.body[0]
    # Drop the decorator so the tree can be re-emitted in another context.
    func.decorator_list = []
    return func


def _iface_of(fn):
    """The interface a unit body binds, read off its first parameter's annotation."""
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if not params:
        raise SPMWPlacementError(
            f"`{fn.__name__}` declares no parameters; a unit takes its Interface "
            f"as the first one, e.g. `def {fn.__name__}(io: MacIO)`."
        )
    ann = params[0].annotation
    if ann is inspect.Parameter.empty:
        raise SPMWPlacementError(
            f"`{fn.__name__}`'s first parameter `{params[0].name}` has no type "
            f"annotation. The annotation is the binding: it names the Interface "
            f"whose ports the body may touch."
        )
    if isinstance(ann, str):
        raise SPMWPlacementError(
            f"`{fn.__name__}`'s interface annotation is the string {ann!r}. "
            f"SPMW needs the Interface object itself, so `from __future__ import "
            f"annotations` cannot be in effect for this module."
        )
    base = unwrap(ann)
    if not (isinstance(base, type) and issubclass(base, Interface)):
        raise SPMWPlacementError(
            f"`{fn.__name__}`'s first parameter is annotated `{ann}`, which is not "
            f"an spmw.Interface."
        )
    return ann


def _wants_site(fn):
    params = list(inspect.signature(fn).parameters.values())
    return len(params) > 1 and _is_site_annotation(params[1].annotation)


def _is_site_annotation(ann):
    return ann is Site or (isinstance(ann, type) and issubclass(ann, Site))


def _io_param_name(tree):
    return tree.args.args[0].arg if tree.args.args else "io"


def _bound_names(target):
    """The names an assignment target binds, ignoring what it writes through."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for elt in target.elts:
            names.extend(_bound_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    return []


def _check_io_not_rebound(tree, where):
    """The io parameter names the contract for the whole body.

    Rebinding it would leave every later `io.x` reading as a port when it is not,
    so say so rather than silently rewriting the wrong thing.
    """
    io_name = _io_param_name(tree)
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For)):
            targets = [node.target]
        for target in targets:
            # Only a binding of the name itself counts: `io.c = x` and
            # `io.a[i] = x` write *through* it and are ordinary port writes.
            for bound in _bound_names(target):
                if bound == io_name:
                    raise SPMWPlacementError(
                        f"`{where}` rebinds `{io_name}` on line {node.lineno}. "
                        f"That name is the contract for the whole body, so every "
                        f"later `{io_name}.x` would read as a port it no longer is."
                    )


def _check_port_names(tree, iface, where):
    """Reject `io.wets` at declaration time rather than at first firing."""
    base = unwrap(iface)
    io_name = _io_param_name(tree)
    known = set(base.__ports__)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == io_name
            and node.attr not in known
        ):
            raise AttributeError(
                f"`{where}` touches `{io_name}.{node.attr}` (line {node.lineno}), "
                f"which `{base.__name__}` does not declare. "
                f"Declared ports: {', '.join(sorted(known))}"
            )


def _check_untouched(tree, unbound, where):
    """A role body may not touch the ports it declares unbound."""
    io_name = _io_param_name(tree)
    forbidden = {sym.name: sym for sym in unbound}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == io_name
            and node.attr in forbidden
        ):
            raise SPMWPlacementError(
                f"role `{where}` declares `{forbidden[node.attr]}` unbound but "
                f"touches it on line {node.lineno}."
            )


def free_names(tree):
    """Names a body reads but does not bind.

    These are the constants, types and helper functions the body closes over, and
    they must travel with it when the body is re-emitted in another module.
    """
    bound = {a.arg for a in tree.args.args}
    reads = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
            else:
                reads.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
    return reads - bound


def captured_env(fn, tree):
    """Resolve a body's free names against the module it was written in."""
    env = {}
    closure = {}
    if fn.__closure__:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                closure[name] = cell.cell_contents
            except ValueError:
                continue
    for name in free_names(tree):
        if name in closure:
            env[name] = closure[name]
        elif name in fn.__globals__:
            env[name] = fn.__globals__[name]
        elif name in getattr(__builtins__, "__dict__", __builtins__):
            continue
    return env


def port_uses(tree, iface):
    """Every ``io.<port>`` reference in a body, as ``(symbol, ast.Attribute)``."""
    base = unwrap(iface)
    io_name = _io_param_name(tree)
    uses = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == io_name
            and node.attr in base.__ports__
        ):
            uses.append((base.__ports__[node.attr], node))
    return uses


def check_directions(tree, iface, where):
    """`get` on an Out, `put` on an In, a write through a MemIn -- at trace time."""
    base = unwrap(iface)
    io_name = _io_param_name(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr in {"get", "put"}):
            continue
        target = fn.value
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == io_name
        ):
            continue
        sym = base.__ports__.get(target.attr)
        if sym is None:
            continue
        if sym.protocol != STREAM:
            raise SPMWError(
                f"`{where}` calls `{fn.attr}` on `{sym}` (line {node.lineno}), which "
                f"is {sym.type_str()} -- a memory port. Streams and memories never "
                f"mate by coercion; cross them with an adapter component."
            )
        want = IN if fn.attr == "get" else OUT
        if sym.direction != want:
            raise SPMWError(
                f"`{where}` calls `{fn.attr}` on `{sym}` (line {node.lineno}), which "
                f"is declared {sym.type_str()}."
            )
