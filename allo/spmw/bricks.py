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
        if self.bank_fn:
            extra = f"({self.bank_fn}, banks={self.banks}, s={self.stride_bit})"
        return f"{self.kind}{extra}"

    @property
    def bank_bits(self):
        """log2 of the bank count."""
        return (self.banks - 1).bit_length() if self.banks else 0

    def at_stride(self, stride_bit):
        """The same layout specialised to one stage's stride.

        A stride below the bank count is refused rather than swizzled. There is
        nothing to fix there -- the two operands already differ in their low
        bits -- and the swizzle would XOR a bank bit against itself, pinning it
        to zero and folding two indices onto one slot. Separating operands is
        easy if you may lose data; this is where that is ruled out.
        """
        if (
            self.bank_fn == "xor"
            and stride_bit is not None
            and stride_bit < self.bank_bits
        ):
            raise SPMWMemoryError(
                f"xor_bank at stride bit {stride_bit} with {self.banks} banks: "
                f"the stride is inside the bank index, where the two operands "
                f"already land in different banks and the swizzle would map two "
                f"indices onto one slot. Swizzle only from bit "
                f"{self.bank_bits} up."
            )
        return Layout(
            self.kind,
            on=self.on,
            banks=self.banks,
            bank_fn=self.bank_fn,
            stride_bit=stride_bit,
        )

    def bank_of(self, index):
        """Which bank a linear index lives in.

        Cyclic banking is the low bits alone. The XOR form additionally folds
        the stride's bit into the top of the bank index, which is what keeps a
        butterfly's two operands apart once the stride reaches the bank count.
        """
        if not self.banks:
            return 0
        low = index & (self.banks - 1)
        if self.bank_fn != "xor" or self.stride_bit is None:
            return low
        return low ^ (((index >> self.stride_bit) & 1) << (self.bank_bits - 1))

    def row_of(self, index):
        """Which row within its bank a linear index lives in."""
        return index >> self.bank_bits if self.banks else index

    def place(self, index):
        """(bank, row) for a linear index."""
        return self.bank_of(index), self.row_of(index)


def banked(on=None, banks=None, bank="cyclic", stride_bit=None):
    """A grid of bricks, one per row/col, each with write capacity 1.

    Statically race-free: the single-writer rule holds by construction rather
    than by check.
    """
    return Layout("banked", on=on, banks=banks, bank_fn=bank, stride_bit=stride_bit)


def xor_bank(banks, stride_bit=None):
    """XOR-swizzled banking, the conflict-free layout for butterfly access sets.

    A butterfly at stage ``s`` reads ``i`` and ``i + (1 << s)``. Once the stride
    reaches the bank count those two share their low bits, so plain
    (``cyclic``) banking puts them in the same bank and the pair serialises.
    The swizzle folds the stride's own bit into the top of the bank index::

        bank(i) = (i & (banks - 1)) ^ (((i >> s) & 1) << (log2(banks) - 1))

    which differs between the two operands precisely because bit ``s`` does.
    `Layout.bank_of` computes it and
    `test_spmw_banking.py` checks the conflict-freedom it claims, rather than
    leaving it as a remark in a docstring.

    ``stride_bit`` is the ``s`` above. Leaving it ``None`` means the layout is
    not yet specialised to a stage; `Layout.at_stride` returns one that is.
    """
    if banks & (banks - 1):
        raise SPMWMemoryError(
            f"xor_bank needs a power-of-two bank count, not {banks}: the swizzle "
            f"is defined on the low log2(banks) bits of the index."
        )
    layout = Layout("banked", banks=banks, bank_fn="xor")
    return layout if stride_bit is None else layout.at_stride(stride_bit)


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
    from .context import current_fabric  # pylint: disable=import-outside-toplevel

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

    @property
    def base(self):
        return self

    @property
    def offsets(self):
        return (0,) * len(self.shape)

    def __repr__(self):
        return f"<tensor {self.name} {self.dtype}{list(self.shape)}>"


class TensorView(Tensor):
    """One site's slice of a parent tensor -- a view, not a copy.

    A fabric placed on a topology receives, at each site, the piece of the
    parent's tensor that the parent's ``shard`` assigns it.  The piece is
    addressed in its own coordinates, so the sub-fabric is written as though it
    owned a whole tensor; the offset is added when the access is emitted.
    """

    __slots__ = ("_base", "_offsets", "site")

    def __init__(self, base, offsets, shape, site=None):
        super().__init__(base.name, base.dtype, shape, base.index)
        self._base = base.base
        parent = base.offsets
        self._offsets = tuple(p + o for p, o in zip(parent, offsets))
        self.site = site

    @property
    def base(self):
        return self._base

    @property
    def offsets(self):
        return self._offsets

    def __repr__(self):
        span = ", ".join(f"{o}:{o + n}" for o, n in zip(self._offsets, self.shape))
        return f"<view {self.name}[{span}]>"
