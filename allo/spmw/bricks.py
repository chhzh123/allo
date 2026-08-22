# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory bricks -- components with interfaces, on the same board as everything else.

Memories are not a side system.  Every connection in a fabric -- compute to
compute, compute to memory, memory to memory -- is a port-to-port link on one
checked substrate, and the surface verbs desugar into it.

There are no memory level names.  The hierarchy *is* the structure: a plain array
in a unit body is the innermost storage, a brick in a tile fabric sits one level
further out, a top-level fabric argument is a brick at the host boundary.  A
brick's realization is the compiler's choice from the target resource model,
with ``resource=`` as the explicit binding.
"""

from .errors import SPMWMemoryError
from .ports import _split_type


class Layout:
    """A placement pattern for RAM bricks."""

    __slots__ = ("kind", "on", "banks", "bank_fn", "stride_bit")

    def __init__(self, kind, on=None, banks=None, bank_fn=None, stride_bit=None):
        self.kind = kind
        self.on = on
        self.banks = banks
        self.bank_fn = bank_fn
        self.stride_bit = stride_bit

    def __repr__(self):
        extra = f"(on={self.on!r})" if self.on is not None else ""
        return f"{self.kind}{extra}"


def banked(on=None, banks=None, bank="cyclic", stride_bit=None):
    """A grid of bricks, one per row/col, each with write capacity 1.

    Statically race-free: the single-writer rule holds by construction rather
    than by check.
    """
    return Layout("banked", on=on, banks=banks, bank_fn=bank, stride_bit=stride_bit)


def xor_bank(banks, stride_bit=None):
    """XOR-swizzled banking, the conflict-free layout for butterfly access sets."""
    return Layout("banked", banks=banks, bank_fn="xor", stride_bit=stride_bit)


#: One brick per site (or group), all write-linked to a single broadcast loader.
replicate = Layout("replicate")

#: One brick, N use clients on one serve port; legal iff capacity >= N.
shared = Layout("shared")

Layout.banked = staticmethod(banked)
Layout.replicate = replicate
Layout.shared = shared
Layout.xor_bank = staticmethod(xor_bank)


class Brick:
    """Addressed storage with serve ports."""

    _counter = 0

    def __init__(
        self,
        dtype,
        shape=(),
        layout=None,
        resource=None,
        double=False,
        pingpong=None,
        init=None,
        name=None,
    ):
        self.dtype, inferred = _split_type(dtype)
        self.shape = tuple(shape) if shape else inferred
        self.layout = layout
        self.resource = resource
        self.pingpong = pingpong if pingpong is not None else (2 if double else 1)
        self.init = init
        Brick._counter += 1
        self.name = name or f"brick{Brick._counter}"
        if init is not None:
            init_shape = tuple(getattr(init, "shape", ()))
            if init_shape and init_shape != self.shape:
                raise SPMWMemoryError(
                    f"`{self.name}` declares shape {self.shape} but its init= "
                    f"contents are {init_shape}."
                )

    @property
    def double(self):
        return self.pingpong > 1

    @property
    def is_rom(self):
        return self.init is not None

    def __repr__(self):
        return f"<brick {self.name} {self.dtype}{list(self.shape)}>"


class FIFO(Brick):
    """Stream storage: one writer, one reader.

    A coordinate edge is sugar for an anonymous FIFO whose single writer and
    single reader are the two linked ports, which is why one-writer/one-reader
    holds by construction.
    """

    def __init__(self, dtype, depth=2, name=None):
        super().__init__(dtype, name=name)
        self.depth = depth

    def __repr__(self):
        return f"<fifo {self.name} {self.dtype} depth={self.depth}>"


def RAM(
    dtype, shape=(), resource=None, double=False, init=None, name=None
):  # pylint: disable=invalid-name
    """Addressed storage; compiler-chosen realization unless bound."""
    return Brick(
        dtype, shape=shape, resource=resource, double=double, init=init, name=name
    )


def mem(
    dtype, shape=(), layout=None, resource=None, double=False, init=None, name=None
):
    """Declare and place a memory brick in one line.

    Each ``layout`` names a placement pattern that is also expressible longhand
    as a ``place`` of RAM bricks on a Grid.
    """
    brick = Brick(
        dtype,
        shape=shape,
        layout=layout,
        resource=resource,
        double=double,
        init=init,
        name=name,
    )
    from .graph import current_fabric  # pylint: disable=import-outside-toplevel

    fab = current_fabric(required=False)
    if fab is not None:
        fab.bricks.append(brick)
    return brick


class Tensor:
    """A brick at the host boundary -- a top-level fabric argument."""

    __slots__ = ("name", "dtype", "shape", "index")

    def __init__(self, name, dtype, shape, index):
        self.name = name
        self.dtype = dtype
        self.shape = tuple(shape)
        self.index = index

    @property
    def rank(self):
        return len(self.shape)

    def __repr__(self):
        return f"<tensor {self.name} {self.dtype}{list(self.shape)}>"
