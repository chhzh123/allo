# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A reference simulator: one task per site, over the elaborated graph.

This executes an SPMW program directly -- no IR, no backend -- by giving each
site a thread and each channel a bounded queue.  Blocking on a full or empty
queue *is* the handshake, so the systolic wavefront emerges from the FIFOs
rather than being staggered by hand, exactly as it does in hardware.

Its purpose is to check the model rather than to be fast: it answers "does this
fabric compute the right thing, and does it deadlock?" without a built compiler,
and it is the oracle the lowered dataflow program is compared against.
"""

import ast
import queue
import threading

from .bricks import Brick, Tensor
from .errors import SPMWError
from .index import SliceMap
from .placement import Bundle, MemGrid
from .ports import IN, OUT, STREAM

# Keyed on how a type prints. Allo's own types render as `f32`, `i8`, `ui8`;
# the numpy spellings are here too so a stand-in type works as well.
_NUMPY_OF = {
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "ui8": "uint8",
    "ui16": "uint16",
    "ui32": "uint32",
    "ui64": "uint64",
    "float32": "float32",
    "float64": "float64",
    "float16": "float16",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "index": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
}


class SPMWDeadlock(SPMWError):
    """A run made no progress: some site is waiting for a token nobody sends."""


class _Chan:
    """One bounded channel, plus who is waiting on it."""

    __slots__ = ("q", "name", "waiting")

    def __init__(self, name, depth):
        self.q = queue.Queue(maxsize=max(1, depth))
        self.name = name
        self.waiting = set()


class _StreamIn:
    __slots__ = ("chan", "who", "seed", "sim")

    def __init__(self, chan, who, seed, sim):
        self.chan = chan
        self.who = who
        self.seed = seed
        self.sim = sim

    def get(self):
        if self.chan is None:
            # A rank-0 source is the constant sequence: it folds into the site.
            return self.seed
        return _scalar(self.sim.recv(self.chan, self.who))

    def put(self, value):
        raise SPMWError(f"{self.who}: put on an In port")


class _StreamOut:
    __slots__ = ("chans", "who", "sim")

    def __init__(self, chans, who, sim):
        self.chans = chans
        self.who = who
        self.sim = sim

    def put(self, value):
        # Writing to nowhere is a no-op; one writer with many readers is a
        # broadcast, so every reader gets its own copy. The copy is what makes a
        # token a value: a body that refills one block buffer per iteration would
        # otherwise have every reader see whatever it last wrote.
        import numpy as np  # pylint: disable=import-outside-toplevel

        for chan in self.chans:
            token = np.copy(value) if isinstance(value, np.ndarray) else value
            self.sim.send(chan, token, self.who)

    def get(self):
        raise SPMWError(f"{self.who}: get on an Out port")


class _IO:
    """A contract bound to one site: what the body's ``io`` parameter is."""

    def __init__(self, ports):
        object.__setattr__(self, "_ports", ports)

    def __getattr__(self, name):
        ports = object.__getattribute__(self, "_ports")
        if name in ports:
            return ports[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        ports = object.__getattribute__(self, "_ports")
        if name in ports:
            sink = ports[name]
            if callable(getattr(sink, "store", None)):
                sink.store(value)
                return
        raise AttributeError(f"cannot assign to `{name}`")


def _scalar(value):
    """Widen an integer read; leave floats and blocks alone.

    An int8 operand feeding an int32 accumulator is widened by the datapath, so
    the reference widens too rather than wrapping at the narrow width.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.integer):
        return int(value)
    return value


def _store_into(array, index, value):
    """Write a value at the destination's own width, truncating as hardware does."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    array[index] = np.asarray(value).astype(array.dtype, casting="unsafe")


class _Cell:
    """A rank-0 memory port: read as a plain value, written as one."""

    __slots__ = ("array", "index")

    def __init__(self, array, index):
        self.array = array
        self.index = index

    def value(self):
        return self.array[self.index]

    def store(self, value):
        _store_into(self.array, self.index, value)

    def __float__(self):
        return float(self.value())

    def __index__(self):
        return int(self.value())


def _prepare(body):
    """Compile a body for direct execution.

    A bare ``u: csample`` declares an array in Allo but nothing at all in Python,
    so it is materialised here; every other statement is executed verbatim, which
    is the point -- the reference must run the program as written.
    """
    tree = _copy_tree(body.tree)
    tree.body = [_Materialise().visit(s) for s in tree.body]
    tree.decorator_list = []
    tree.name = "_body"
    module = ast.Module(body=[tree], type_ignores=[])
    ast.fix_missing_locations(module)
    env = dict(getattr(body.fn, "__globals__", {}))
    env.setdefault("_zeros", _zeros)
    if body.fn.__closure__:
        for name, cell in zip(body.fn.__code__.co_freevars, body.fn.__closure__):
            try:
                env[name] = cell.cell_contents
            except ValueError:
                continue
    exec(compile(module, "<spmw-refsim>", "exec"), env)  # pylint: disable=exec-used
    return env["_body"]


class _Materialise(ast.NodeTransformer):
    """Turn a bare array declaration into a zero-filled buffer."""

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if node.value is not None:
            return node
        if not isinstance(node.target, ast.Name):
            return node
        # Whether the declaration is an array or a scalar is a property of the
        # annotation's value, not its syntax, so materialise either way.
        return ast.Assign(
            targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="_zeros", ctx=ast.Load()),
                args=[node.annotation],
                keywords=[],
            ),
        )


def _zeros(annotation):
    """The zero value a bare declaration stands for: an array, or a scalar."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    shape = tuple(getattr(annotation, "shape", ()) or ())
    dtype = _numpy_dtype(getattr(annotation, "dtype", annotation))
    if not shape:
        return dtype.type(0)
    return np.zeros(shape, dtype=dtype)


def _numpy_dtype(dtype):
    """The numpy type a declared type stands for.

    Guessing here would let the reference compute at wider precision than the
    hardware, so an unrecognised type is named rather than widened -- an integer
    accumulator that never wraps hides exactly the bug this exists to find.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    name = _NUMPY_OF.get(str(dtype))
    if name is None:
        raise SPMWError(
            f"the reference simulator does not know how to represent `{dtype}`. "
            f"Add it to the type table, or use a target that lowers through Allo."
        )
    return np.dtype(name)


def _copy_tree(tree):
    return ast.parse(ast.unparse(ast.Module(body=[tree], type_ignores=[]))).body[0]


class RefSim:
    """A callable that runs one elaborated fabric on numpy arrays."""

    def __init__(self, graph, timeout=20.0):
        self.graph = graph
        self.timeout = timeout
        self.spmw_graph = graph
        self._bodies = {}
        self._plan()

    # -- planning ----------------------------------------------------------

    def _plan(self):
        self.seeds = {}
        self.mem_reads = {}
        self.mem_writes = {}
        self.movers = []
        self.links = []
        self.fills = {}
        for binding in self.graph.bindings:
            if any(
                isinstance(side, (Bundle, MemGrid)) and side.placement.expanded
                for side in (binding.source, binding.target)
            ):
                continue  # consumed by the expansion
            kind = binding.kind
            if kind in ("stream_in", "scatter"):
                self.movers.append(("load", binding, binding.target, binding.source))
            elif kind == "gather" and isinstance(binding.source, Bundle):
                self.movers.append(("drain", binding, binding.source, binding.target))
            elif kind in ("gather", "gather_mem"):
                self.mem_writes[(binding.source.placement, binding.source.port)] = (
                    binding
                )
            elif kind == "seed":
                self.seeds[(binding.target.placement, binding.target.port)] = (
                    binding.source
                )
            elif kind in ("shard", "stationary"):
                self.mem_reads[(binding.target.placement, binding.target.port)] = (
                    binding
                )
            elif kind == "link":
                self.links.append(binding)
            elif kind == "copy":
                self.fills[id(binding.target)] = binding.source

    def resolve_storage(self, source):
        """Follow a staged brick back to the tensor a copy fills it from."""
        seen = set()
        while isinstance(source, Brick) and source.init is None:
            filler = self.fills.get(id(source))
            if filler is None or id(filler) in seen:
                break
            seen.add(id(source))
            source = filler
        return source

    def _body_for(self, placement, site):
        body = placement.roles.get(site)
        if body is None:
            raise SPMWError(
                f"`{placement.name}` is a fabric placed on a topology. Hierarchical "
                f"placement elaborates, but the reference simulator runs unit "
                f"bodies -- inline the sub-fabric, or place its unit directly."
            )
        key = id(body)
        if key not in self._bodies:
            self._bodies[key] = _prepare(body)
        return self._bodies[key], body

    # -- channel plumbing --------------------------------------------------

    def send(self, chan, value, who):
        chan.waiting.add(who)
        try:
            chan.q.put(value, timeout=self.timeout)
        except queue.Full as err:
            raise SPMWDeadlock(
                f"{who} blocked writing `{chan.name}`: the channel stayed full, so "
                f"its reader never consumed."
            ) from err
        finally:
            chan.waiting.discard(who)

    def recv(self, chan, who):
        chan.waiting.add(who)
        try:
            return chan.q.get(timeout=self.timeout)
        except queue.Empty as err:
            raise SPMWDeadlock(
                f"{who} blocked reading `{chan.name}`: the channel stayed empty, so "
                f"its writer never produced."
            ) from err
        finally:
            chan.waiting.discard(who)

    # -- running -----------------------------------------------------------

    def __call__(self, *arrays):
        tensors = list(self.graph.tensors.values())
        if len(arrays) != len(tensors):
            raise SPMWError(
                f"`{self.graph.fabric.name}` takes {len(tensors)} arrays "
                f"({', '.join(t.name for t in tensors)}), got {len(arrays)}."
            )
        data = {t.name: a for t, a in zip(tensors, arrays)}
        chans = self._build_channels()
        errors = []
        threads = []

        for placement in self.graph.placements:
            if placement.expanded:
                continue  # replaced by running its body per site
            for site in placement.sites():
                threads.append(
                    threading.Thread(
                        target=self._run_site,
                        args=(placement, site, chans, data, errors),
                        daemon=True,
                    )
                )
        for role, binding, bundle, tensor in self.movers:
            for pos, site in enumerate(bundle.sites):
                threads.append(
                    threading.Thread(
                        target=self._run_mover,
                        args=(
                            role,
                            binding,
                            bundle,
                            tensor,
                            pos,
                            site,
                            chans,
                            data,
                            errors,
                        ),
                        daemon=True,
                    )
                )
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=self.timeout + 5.0)
        stuck = [t for t in threads if t.is_alive()]
        if errors:
            raise errors[0]
        if stuck:
            raise SPMWDeadlock(
                f"{len(stuck)} of {len(threads)} tasks never finished; the fabric "
                f"deadlocked."
            )
        return None

    def _build_channels(self):
        chans = {"read": {}, "write": {}}
        for placement in self.graph.placements:
            topo = placement.topology
            for chan in topo.channels.values():
                if chan.writer is None or not chan.readers:
                    continue
                queues = []
                for rsite, rport in chan.readers:
                    node = _Chan(f"{placement.name}.{rport.name}@{rsite}", chan.depth)
                    chans["read"][(placement, rsite, rport)] = node
                    queues.append(node)
                wsite, wport = chan.writer
                chans["write"].setdefault((placement, wsite, wport), []).extend(queues)

        # Channels a binding creates: loaders and drains, and placement-to-
        # placement links.
        for role, _binding, bundle, _tensor in self.movers:
            for site in bundle.sites:
                node = _Chan(
                    f"{bundle.placement.name}.{bundle.port.name}@{site}",
                    bundle.port.depth,
                )
                if role == "load":
                    chans["read"][(bundle.placement, site, bundle.port)] = node
                    chans["write"].setdefault(("mover", id(bundle), site), []).append(
                        node
                    )
                else:
                    chans["write"].setdefault(
                        (bundle.placement, site, bundle.port), []
                    ).append(node)
                    chans["read"][("mover", id(bundle), site)] = node
        for binding in self.links:
            src, dst = binding.source, binding.target
            for ssite, dsite in zip(src.sites, dst.sites):
                node = _Chan(
                    f"{dst.placement.name}.{dst.port.name}@{dsite}", dst.port.depth
                )
                chans["read"][(dst.placement, dsite, dst.port)] = node
                chans["write"].setdefault((src.placement, ssite, src.port), []).append(
                    node
                )
        return chans

    def _run_site(self, placement, site, chans, data, errors):
        try:
            fn, body = self._body_for(placement, site)
            ports = {}
            who = f"{placement.name}{site}"
            for sym in placement.iface.ports():
                if sym.protocol == STREAM:
                    if sym.direction == IN:
                        chan = chans["read"].get((placement, site, sym))
                        seed = self.seeds.get((placement, sym))
                        ports[sym.name] = _StreamIn(chan, who, seed, self)
                    else:
                        ports[sym.name] = _StreamOut(
                            chans["write"].get((placement, site, sym), []), who, self
                        )
                else:
                    ports[sym.name] = self._memory_port(placement, site, sym, data)
            io = _IO(ports)
            from .component import Site  # pylint: disable=import-outside-toplevel

            args = [io]
            if getattr(body, "wants_site", False):
                args.append(Site(site, placement.grid))
            fn(*args)
        except BaseException as err:  # pylint: disable=broad-except
            errors.append(err)

    def _run_mover(self, role, binding, bundle, tensor, pos, site, chans, data, errors):
        """One loader or drain: the mover synthesised at an open port."""
        import numpy as np  # pylint: disable=import-outside-toplevel

        try:
            who = f"{bundle.placement.name}.{bundle.port.name}-{role}@{site}"
            array = _array_of(tensor, data)
            block = binding.extras.get("block", ())
            extent = binding.extras.get("extent") or 1
            env = dict(bundle.placement.env(site), __coords__=site)
            if role == "load":
                chan = chans["write"][("mover", id(bundle), site)][0]
                for step in range(extent):
                    idx = tuple(binding.imap.eval(env, step=step))
                    value = array[idx]
                    self.send(chan, np.copy(value) if block else value, who)
            else:
                chan = chans["read"][("mover", id(bundle), site)]
                for step in range(extent):
                    idx = tuple(binding.imap.eval(env, step=step))
                    value = self.recv(chan, who)
                    if block:
                        array[idx] = np.asarray(value).reshape(block)
                    else:
                        _store_into(array, idx, value)
        except BaseException as err:  # pylint: disable=broad-except
            errors.append(err)

    def _memory_port(self, placement, site, sym, data):
        binding = self.mem_reads.get((placement, sym)) or self.mem_writes.get(
            (placement, sym)
        )
        if binding is None:
            raise SPMWError(f"`{placement.name}.{sym.name}` has no memory binding")
        source = self.resolve_storage(
            binding.source
            if binding.kind in ("shard", "stationary")
            else binding.target
        )
        if isinstance(source, Brick):
            # Resident at every site means a copy per site: sharing one array
            # would let a write through one site's port reach every other, and
            # reach the caller's init= besides.
            import numpy as np  # pylint: disable=import-outside-toplevel

            return np.copy(source.init)
        array = _array_of(source, data) if isinstance(source, Tensor) else source
        view = _view(array, binding.imap, placement, site, sym)
        if sym.shape == ():
            holder, index = view
            if sym.direction == OUT:
                return _Cell(holder, index)
            return _scalar(holder[index])
        return view


def _array_of(tensor, data):
    """The numpy array a tensor names -- for a view, a real slice of its parent."""
    array = data[tensor.base.name]
    offsets = tensor.offsets
    if not any(offsets):
        return array
    return array[tuple(slice(o, o + n) for o, n in zip(offsets, tensor.shape))]


def _view(array, imap, placement, site, sym):
    """The slice of a tensor one site owns, per its binding's map."""
    if isinstance(imap, SliceMap):
        idx = tuple(
            slice(start, start + size) if size > 1 else start
            for start, size in imap.slice_for(site)
        )
        if sym.shape == ():
            return array, idx
        return array[idx]
    env = dict(placement.env(site), __coords__=site)
    idx = imap.eval(env)
    if sym.shape == ():
        return array, tuple(idx)
    return array[tuple(idx)]


def build_ref(graph, timeout=20.0):
    """A callable running this graph directly, without a compiler."""
    return RefSim(graph, timeout=timeout)


__all__ = ["RefSim", "SPMWDeadlock", "build_ref"]
