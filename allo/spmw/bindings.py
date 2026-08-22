# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Movers and views -- the verbs a fabric wires with.

Crossing a protocol or staging data is a component, not a verb: ``stream_in`` is
a column of loader bricks, ``gather`` a set of drains, ``copy`` a mover.  The
surface spellings here desugar onto the one checked substrate of port-to-port
links.

Every binding lambda's *domain* is the port side's index space.  A memory port is
indexed by its site alone -- stationed views are timeless -- so ``shard`` maps
take grid coordinates.  A stream port is a token sequence per site, so
``stream_in``/``gather`` maps take the flow step first, then the coordinates.
"""

import numbers

from .bricks import Brick, Tensor
from .errors import SPMWBindingError, SPMWMemoryError
from .graph import Binding, Phase, bind_check, current_fabric, make_map
from .index import IndexMap, SliceMap
from .placement import Bundle, MemGrid
from .ports import IN, OUT, READ, STREAM, WRITE


def phase(name):
    """Order a fabric's bindings into fill-then-use epochs."""
    return Phase(name)


# ---------------------------------------------------------------------------
# Stream movers
# ---------------------------------------------------------------------------


def stream_in(source, into, index=None):
    """Synthesise loaders at a bundle's unbound ports.

    ``into=`` names a bundle -- the port's *unbound* instances, wherever the link
    rule left them, never the instances links already feed -- and one loader is
    synthesised per member, with ``index=`` evaluated at that member's own site
    coordinates.  One map, many loaders: a single line feeds a physical west edge
    and every interior slab seam at once.

    A rank-0 source streams as the constant sequence and needs no brick and no
    loader: the literal folds into the consuming sites, zero hardware.
    """
    fab = current_fabric()
    bundle = _as_bundle(into, "stream_in(into=)")
    _expect_direction(bundle, IN, "stream_in")
    where = f"stream_in into `{bundle.placement.name}.{bundle.port.name}`"

    if _is_scalar(source):
        if index is not None:
            raise SPMWBindingError(
                f"{where}: a rank-0 source has no axes to name, so index= does "
                f"not apply."
            )
        return fab.record(Binding("seed", source, bundle))

    tensor = _as_source(source, where)
    if index is None:
        raise SPMWBindingError(
            f"{where}: index= is required on every tensor-source stream_in. The "
            f"tuple is always spelled, so nothing is derived from shapes behind "
            f"the reader's back -- a shape-derived default is correct until two "
            f"extents happen to coincide, then silently transposes."
        )
    block = _block_of(bundle, tensor, where)
    imap = _check_arity(make_map(index, tensor.rank - len(block), where), bundle, where)
    extent = _time_extent(imap, tensor, where, streamed=True)
    bind_check(imap, bundle, tensor, where, extent=extent, write=False, block=block)
    return fab.record(
        Binding("stream_in", tensor, bundle, imap=imap, extent=extent, block=block)
    )


def gather(dest, from_, index=None, pack=None):
    """Drain many per-site sources into one tensor.

    One verb for one meaning; the bundle's port kind picks the mechanism.  A
    ``MemOut`` bundle drains its per-site bricks; a stream ``Out`` bundle
    collects along links.  Being a write, the map's image must tile the
    destination exactly.
    """
    fab = current_fabric()
    side = _as_bundle(from_, "gather(from_=)")
    _expect_direction(side, OUT, "gather")
    tensor = _as_dest(dest, "gather")
    where = f"gather from `{side.placement.name}.{side.port.name}` into `{tensor.name}`"

    block = _block_of(side, tensor, where)
    if pack is not None:
        raise SPMWBindingError(
            f"{where}: pack= is not implemented. It decides how per-site values "
            f"are packed into a word, so accepting it and ignoring it would give "
            f"you a differently-shaped result than you asked for."
        )
    imap, extent = _resolve_drain_map(index, side, tensor, where, block)
    seen = bind_check(imap, side, tensor, where, extent=extent, write=True, block=block)
    if isinstance(imap, SliceMap):
        block = imap.per_site_sizes()
    return fab.record(
        Binding(
            "gather",
            side,
            tensor,
            imap=imap,
            extent=extent,
            pack=pack,
            block=block,
            writes=(seen, block),
        )
    )


def scatter(source, into, index=None, unpack=None):
    """One tensor to many sites -- ``gather``'s mirror."""
    fab = current_fabric()
    bundle = _as_bundle(into, "scatter(into=)")
    _expect_direction(bundle, IN, "scatter")
    tensor = _as_source(source, "scatter")
    where = f"scatter `{tensor.name}` into `{bundle.placement.name}.{bundle.port.name}`"
    block = _block_of(bundle, tensor, where)
    if unpack is not None:
        raise SPMWBindingError(
            f"{where}: unpack= is not implemented. It decides how a packed word "
            f"is split across sites, so accepting it and ignoring it would feed "
            f"the array something other than what you asked for."
        )
    imap, extent = _resolve_feed_map(index, bundle, tensor, where, block)
    bind_check(imap, bundle, tensor, where, extent=extent, write=False, block=block)
    return fab.record(
        Binding(
            "scatter",
            tensor,
            bundle,
            imap=imap,
            extent=extent,
            unpack=unpack,
            block=block,
        )
    )


def link(out_side, to, index=None):
    """Wire two placements' boundary bundles element-wise.

    Pure wiring *within* the stream protocol -- no adapter and no ordering
    choice, because the producer's token order is the order.  Lexicographic site
    order by default; ``index=`` supplies a different pairing.
    """
    fab = current_fabric()
    src = _as_bundle(out_side, "link(...)")
    dst = _as_bundle(to, "link(to=)")
    _expect_direction(src, OUT, "link")
    _expect_direction(dst, IN, "link")
    where = (
        f"link `{src.placement.name}.{src.port.name}` -> "
        f"`{dst.placement.name}.{dst.port.name}`"
    )
    if src.port.dtype != dst.port.dtype or src.port.shape != dst.port.shape:
        raise SPMWBindingError(
            f"{where}: element types differ -- {src.port.type_str()} vs "
            f"{dst.port.type_str()}."
        )
    if len(src) != len(dst):
        raise SPMWBindingError(
            f"{where}: extents differ -- {len(src)} open ports on one side, "
            f"{len(dst)} on the other."
        )
    pairing = index if index is not None else None
    return fab.record(Binding("link", src, dst, imap=pairing))


# ---------------------------------------------------------------------------
# Memory bindings
# ---------------------------------------------------------------------------


def shard(source, into=None, from_=None, index=None, dim=None):
    """Distribute a tensor's ownership across the sites of a placement.

    The aliasing counterpart of scatter: disjoint slice links -- views where the
    target serves them directly, staged copies where it cannot.  ``dim=`` maps
    tensor axes to grid axes; ``index=`` handles permuted mappings.
    """
    fab = current_fabric()
    if (into is None) == (from_ is None):
        raise SPMWBindingError("shard takes exactly one of into= or from_=.")
    side = _as_bundle(into if into is not None else from_, "shard")
    writing = from_ is not None
    tensor = _as_dest(source, "shard") if writing else _as_source(source, "shard")
    where = (
        f"shard `{tensor.name}` "
        f"{'from' if writing else 'into'} "
        f"`{side.placement.name}.{side.port.name}`"
    )
    if side.port.protocol == STREAM:
        raise SPMWMemoryError(
            f"{where}: `{side.port}` is {side.port.type_str()}, a stream. shard "
            f"aliases storage; use scatter/gather to cross into the stream "
            f"protocol."
        )
    imap = _resolve_shard_map(index, dim, side, tensor, where)
    block = tuple(side.port.shape) if index is not None else ()
    seen = bind_check(
        imap, side, tensor, where, extent=None, write=writing, block=block
    )
    if isinstance(imap, SliceMap):
        block = imap.per_site_sizes()
    kind = "gather_mem" if writing else "shard"
    src, tgt = (side, tensor) if writing else (tensor, side)
    extras = {"block": block}
    if writing and seen:
        extras["writes"] = (seen, block)
    return fab.record(Binding(kind, src, tgt, imap=imap, **extras))


def stationary(source, at, index=None):
    """Make a brick resident at every site of a bundle."""
    fab = current_fabric()
    side = _as_bundle(at, "stationary(at=)")
    where = f"stationary at `{side.placement.name}.{side.port.name}`"
    if side.port.protocol == STREAM:
        raise SPMWMemoryError(
            f"{where}: `{side.port}` is {side.port.type_str()}. Stationary data "
            f"lives on a memory port."
        )
    block = tuple(side.port.shape)
    imap = None
    if index is not None:
        shape = tuple(getattr(source, "shape", ()))
        imap = _check_arity(
            make_map(index, len(shape) - len(block), where), side, where
        )
        _reject_time(imap, where)
        bind_check(imap, side, source, where, extent=None, write=False, block=block)
    return fab.record(Binding("stationary", source, side, imap=imap, block=block))


def copy(src, into, how=None):
    """A bulk mover brick: a read client on ``src``, a write client on ``into``."""
    fab = current_fabric()
    src_shape = tuple(getattr(src, "shape", ()))
    dst_shape = tuple(getattr(into, "shape", ()))
    if src_shape and dst_shape and src_shape != dst_shape:
        raise SPMWMemoryError(
            f"copy `{_name_of(src)}` {list(src_shape)} into `{_name_of(into)}` "
            f"{list(dst_shape)}: shapes differ. A copy is a copy; slice one side "
            f"explicitly."
        )
    return fab.record(Binding("copy", src, into, how=how))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_scalar(value):
    return isinstance(value, numbers.Number)


def _name_of(obj):
    return getattr(obj, "name", repr(obj))


def _as_bundle(side, where):
    if isinstance(side, (Bundle, MemGrid)):
        return side
    raise SPMWBindingError(
        f"{where} expects a placement bundle such as `P.west` or `P.c`, got "
        f"{type(side).__name__}. Nothing crosses a placement boundary except "
        f"through a declared port."
    )


def _as_source(source, where):
    if isinstance(source, (Tensor, Brick)):
        return source
    raise SPMWBindingError(
        f"{where}: {type(source).__name__} is not a tensor or a memory brick."
    )


def _as_dest(dest, where):
    if isinstance(dest, (Tensor, Brick)):
        return dest
    raise SPMWBindingError(
        f"{where}: {type(dest).__name__} is not a tensor or a memory brick."
    )


def _expect_direction(side, direction, verb):
    port = side.port
    if port.protocol == STREAM and port.direction != direction:
        want = "In" if direction == IN else "Out"
        raise SPMWBindingError(
            f"{verb} needs an {want} bundle but `{port}` is {port.type_str()}."
        )
    if port.protocol != STREAM:
        access = READ if direction == IN else WRITE
        if port.access not in (access, "rw"):
            raise SPMWBindingError(
                f"{verb} needs a {'readable' if direction == IN else 'writable'} "
                f"port but `{port}` is {port.type_str()}."
            )


def _time_extent(imap, tensor, where, streamed):
    """The token count per port: the extent of the axis ``...`` marks."""
    if imap.is_lambda:
        return 1 if streamed else None
    if imap.time_pos is None:
        # No `...` fixes every axis: a single token per port.
        return 1 if streamed else None
    return tensor.shape[imap.time_pos]


def _check_arity(imap, side, where):
    """A lambda is indexed by site, or by (step, site) -- its arity says which."""
    if not imap.is_lambda:
        return imap
    import inspect  # pylint: disable=import-outside-toplevel

    rank = len(side.placement.grid)
    try:
        params = len(inspect.signature(imap.spec).parameters)
    except (TypeError, ValueError):
        return imap
    if params == rank:
        imap.wants_step = False
    elif params == rank + 1:
        raise SPMWBindingError(
            f"{where}: the lambda takes {params} arguments, so it is indexed by "
            f"(step, site) over a rank-{rank} grid. A stepping lambda is not "
            f"lowered on this path -- spell the tuple form with `...` marking the "
            f"streamed axis."
        )
    else:
        raise SPMWBindingError(
            f"{where}: the lambda takes {params} arguments, but the bundle's grid "
            f"has rank {rank}. A site-indexed map takes {rank}; a (step, site) one "
            f"takes {rank + 1}."
        )
    return imap


def _resolve_feed_map(index, bundle, tensor, where, block=()):
    if index is None:
        raise SPMWBindingError(
            f"{where}: index= is required; the tuple spells the source tensor's "
            f"subscripts."
        )
    imap = _check_arity(make_map(index, tensor.rank - len(block), where), bundle, where)
    return imap, _time_extent(imap, tensor, where, streamed=True)


def _resolve_drain_map(index, side, tensor, where, block=()):
    """A drain's map, or the full positional identity when it is omitted."""
    if index is not None:
        imap = _check_arity(
            make_map(index, tensor.rank - len(block), where), side, where
        )
        streamed = side.port.protocol == STREAM
        return imap, _time_extent(imap, tensor, where, streamed=streamed)
    identity = _positional_identity(side, tensor, where)
    return identity, None


def _reject_time(imap, where):
    """`...` marks a token sequence, and a stationed view has none."""
    if getattr(imap, "has_time", False):
        raise SPMWBindingError(
            f"{where}: index= names `...`, which marks the streamed axis. A memory "
            f"port is indexed by its site alone -- stationed views are timeless -- "
            f"so there is no sequence for `...` to stand for."
        )


def _block_of(side, tensor, where):
    """The port's own block, which covers the tensor's trailing axes.

    A port carrying ``float32[2]`` moves one complex sample per token, so the
    binding's tuple addresses only the axes the block does not.
    """
    block = tuple(side.port.shape)
    if not block:
        return ()
    if tensor.shape[-len(block) :] != block:
        raise SPMWBindingError(
            f"{where}: `{side.port}` carries a {list(block)} block, so the "
            f"tensor's trailing axes must be {list(block)}; `{tensor.name}` is "
            f"{list(tensor.shape)}."
        )
    return block


def _positional_identity(side, tensor, where):
    """The only shape a bare binding may take -- never a search.

    Three readings, tried in order and distinguished by shape alone: elements
    addressed one-for-one by the grid; site axes leading with block axes
    trailing; and site ``(i, j)`` owning the tensor's ``(i, j)``-th block.
    """
    grid = tuple(side.placement.grid)
    block = tuple(side.port.shape)
    shape = tuple(tensor.shape)

    if side.port.protocol == STREAM:
        # A stream drain moves one token per site, so the identity is an element
        # map over the site axes rather than a slice: `...` never appears, and
        # the block covers the trailing axes as everywhere else.
        if shape != grid + block:
            raise SPMWBindingError(
                f"{where}: index= was omitted, which means the full positional "
                f"identity -- site axes leading {list(grid)}, block axes trailing "
                f"{list(block)}, so {list(grid + block)}. `{tensor.name}` is "
                f"{list(shape)}. Spell the tuple, or match the shapes."
            )
        return IndexMap(tuple(side.placement.axes), len(grid))

    if not block and shape == grid:
        axis_of = tuple(range(len(grid)))
        return SliceMap("identity", grid, (1,) * len(grid), shape, axis_of)

    if shape == grid + block:
        axis_of = tuple(range(len(grid))) + (None,) * len(block)
        return SliceMap("concat", grid, (1,) * len(grid) + block, shape, axis_of)

    if len(block) == len(grid) == len(shape) and all(
        g * b == s for g, b, s in zip(grid, block, shape)
    ):
        return SliceMap("blocked", grid, block, shape, tuple(range(len(grid))))

    raise SPMWBindingError(
        f"{where}: index= was omitted, which means the full positional identity. "
        f"With a {list(grid)} grid and a {list(block) or '[]'} block that reads as "
        f"{list(grid + block)} (site axes leading, block axes trailing) or "
        f"{[g * b for g, b in zip(grid, block)] if len(block) == len(grid) else '-'} "
        f"(one block per site), but `{tensor.name}` is {list(shape)}. Spell the "
        f"tuple, or match the shapes."
    )


def _resolve_shard_map(index, dim, side, tensor, where):
    if index is not None:
        block = tuple(side.port.shape)
        imap = make_map(index, tensor.rank - len(block), where)
        _reject_time(imap, where)
        return imap
    if dim is not None:
        return _dim_map(dim, side, tensor, where)
    return _positional_identity(side, tensor, where)


def _dim_map(dim, side, tensor, where):
    """``dim=`` maps tensor axes to grid axes; unnamed axes take the whole extent."""
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    grid = tuple(side.placement.grid)
    block = tuple(side.port.shape)
    shape = tuple(tensor.shape)
    if len(dims) > len(grid):
        raise SPMWBindingError(
            f"{where}: dim={dim} names {len(dims)} axes but the placement grid "
            f"has rank {len(grid)}."
        )
    axis_of, sizes = [], []
    for t, bound in enumerate(shape):
        if t in dims:
            # dim= names tensor axes, each distributed along the grid axis of the
            # same index; a permuted mapping is index='s job, not this one's.
            g = t
            if g >= len(grid):
                raise SPMWBindingError(
                    f"{where}: dim={dim} names tensor axis {t}, but the placement "
                    f"grid has rank {len(grid)}. Use index= for a permuted mapping."
                )
            size = block[t] if t < len(block) else 1
            if grid[g] * size != bound:
                raise SPMWBindingError(
                    f"{where}: axis {t} has extent {bound}, but grid axis {g} "
                    f"({grid[g]} sites) times the port's block ({size}) is "
                    f"{grid[g] * size}."
                )
            axis_of.append(g)
            sizes.append(size)
        else:
            axis_of.append(None)
            sizes.append(bound)
    return SliceMap("blocked", grid, tuple(sizes), shape, tuple(axis_of))


__all__ = [
    "copy",
    "gather",
    "link",
    "phase",
    "scatter",
    "shard",
    "stationary",
    "stream_in",
]
