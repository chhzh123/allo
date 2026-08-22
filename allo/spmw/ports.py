# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Port declarations for SPMW interfaces.

A port declaration is written as an assignment in an ``Interface`` body::

    class MacIO(spmw.Interface):
        west = spmw.In(float32)
        c    = spmw.MemOut(float32)

Each declaration constructs a :class:`PortSymbol` at class-creation time via the
``__set_name__`` descriptor protocol.  Class access yields that symbol -- hashable
and identity-comparable, so it can key a link dict.  Instance access yields an
accessor bound to one site, which is what a unit body calls ``get``/``put`` on.
"""

from .errors import SPMWDirectionError, SPMWTypeError

# Protocols.  Protocol is part of a port's type: a stream never mates with a
# memory by coercion, only through an adapter component.
STREAM = "stream"
MEMORY = "memory"

# Stream directions.
IN = "in"
OUT = "out"

# Memory access modes.
READ = "rd"
WRITE = "wr"
READWRITE = "rw"


class PortSymbol:
    """The runtime identity of one declared port.

    Carries the port's name, protocol, direction and element type.  Symbols are
    compared by identity, never by name, so that a port belonging to some other
    interface is rejected on ownership rather than silently matching a homonym.
    """

    __slots__ = (
        "name",
        "protocol",
        "direction",
        "dtype",
        "shape",
        "depth",
        "access",
        "owner",
    )

    def __init__(self, decl, name, owner):
        self.name = name
        self.protocol = decl.protocol
        self.direction = decl.direction
        self.dtype = decl.dtype
        self.shape = decl.shape
        self.depth = decl.depth
        self.access = decl.access
        self.owner = owner

    @property
    def is_stream(self):
        return self.protocol == STREAM

    @property
    def is_memory(self):
        return self.protocol == MEMORY

    @property
    def rank(self):
        return len(self.shape)

    def type_str(self):
        """Render the declared type the way a diagnostic should print it."""
        kind = {
            (STREAM, IN): "In",
            (STREAM, OUT): "Out",
            (MEMORY, READ): "MemIn",
            (MEMORY, WRITE): "MemOut",
            (MEMORY, READWRITE): "Mem",
        }[(self.protocol, self.direction if self.is_stream else self.access)]
        inner = str(self.dtype)
        if self.shape:
            inner += "[" + ", ".join(str(s) for s in self.shape) + "]"
        return f"{kind}({inner})"

    def __repr__(self):
        owner = self.owner.__name__ if self.owner is not None else "?"
        return f"{owner}.{self.name}"

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class _PortDecl:
    """Base class for the declarations written in an ``Interface`` body.

    The declaration is a descriptor.  ``__set_name__`` fires at class creation and
    constructs the :class:`PortSymbol`; from then on the declaration forwards all
    access to it.
    """

    protocol = None
    direction = None
    access = None

    def __init__(self, dtype, shape=(), depth=2):
        self.dtype, self.shape = _split_type(dtype)
        if shape:
            self.shape = tuple(shape)
        self.depth = depth
        self.symbol = None

    def __set_name__(self, owner, name):
        # Fires once per class-body assignment, and never for an inherited name
        # -- which is what makes ``MacIO.west is NSEW.west`` hold with no special
        # handling here.  Reuse of one declaration is caught by the metaclass
        # before this runs, so that the diagnostic is not buried in the
        # RuntimeError Python wraps __set_name__ failures in.
        self.symbol = PortSymbol(self, name, owner)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.symbol
        return obj._accessor(self.symbol)

    def __set__(self, obj, value):
        # ``io.c = acc`` on a MemOut scalar port is a write, not a rebind.
        obj._write(self.symbol, value)


def _split_type(dtype):
    """Accept both ``float32`` and ``float32[2]``, returning ``(elem, shape)``."""
    shape = getattr(dtype, "shape", None)
    if shape:
        elem = getattr(dtype, "dtype", None)
        if elem is not None:
            return elem, tuple(shape)
    return dtype, ()


class In(_PortDecl):
    """The reading end of a FIFO."""

    protocol = STREAM
    direction = IN


class Out(_PortDecl):
    """The writing end of a FIFO."""

    protocol = STREAM
    direction = OUT


class MemIn(_PortDecl):
    """An array the site is given -- a shard, a stationary weight."""

    protocol = MEMORY
    access = READ
    direction = IN


class MemOut(_PortDecl):
    """An array the site produces, backed by a per-site brick the parent gathers."""

    protocol = MEMORY
    access = WRITE
    direction = OUT


class Mem(_PortDecl):
    """Shared random access; capacity- and phase-checked at link time."""

    protocol = MEMORY
    access = READWRITE
    direction = None


class PortAccessor:
    """A port bound to one site.  What a unit body actually touches.

    This object exists so that direction errors are raised where they are
    written -- ``io.west.put(x)`` on an ``In`` port is an error at trace time, not
    a deadlock at run time.
    """

    __slots__ = ("symbol", "site")

    def __init__(self, symbol, site):
        self.symbol = symbol
        self.site = site

    def get(self):
        sym = self.symbol
        if not sym.is_stream:
            raise SPMWDirectionError(
                f"`{sym}` is {sym.type_str()}, a memory port; `get` is the stream "
                f"protocol. Read it by subscript, or cross protocols with an adapter."
            )
        if sym.direction != IN:
            raise SPMWDirectionError(
                f"`get` on `{sym}`, which is declared {sym.type_str()}. "
                f"Only In ports can be read."
            )
        raise SPMWTypeError(
            f"`{sym}.get()` was called outside a traced unit body. "
            f"Port access is only meaningful inside an @spmw.unit function."
        )

    def put(self, value):  # pylint: disable=unused-argument
        sym = self.symbol
        if not sym.is_stream:
            raise SPMWDirectionError(
                f"`{sym}` is {sym.type_str()}, a memory port; `put` is the stream "
                f"protocol. Write it by subscript, or cross protocols with an adapter."
            )
        if sym.direction != OUT:
            raise SPMWDirectionError(
                f"`put` on `{sym}`, which is declared {sym.type_str()}. "
                f"Only Out ports can be written."
            )
        raise SPMWTypeError(
            f"`{sym}.put()` was called outside a traced unit body. "
            f"Port access is only meaningful inside an @spmw.unit function."
        )
