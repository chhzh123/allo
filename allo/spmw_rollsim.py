# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=cyclic-import

"""Coroutine functional simulator over the rolled SPMW form (M5 task5.1).

``spmw.build(region, target="rolled_simulator")`` runs a region on a **cooperative coroutine**
scheduler: one generator per logical PID over the resolved topology + boundary streams, bounded-FIFO
``put``/``get``, single-threaded round-robin scheduling. Unlike the desugar ``target="simulator"`` path
(which lowers to ``allo.dataflow`` and injects one OpenMP section per PE), this reads no
``OMP_NUM_THREADS`` and cannot explode a thread per PE: it is one Python thread cooperatively driving N
generators. A stuck schedule (an undersized FIFO, an unsatisfiable dependency) is **detected** -- when a
full scheduling pass makes no progress the blocked ``(pid, fifo)`` pairs are reported via
``SPMWDeadlockError`` -- never a host-thread hang.

Scope (task5.1): the 2-D systolic mesh and the 1-D producer/consumer pipeline -- reproducing the exact
per-PID dataflow of ``allo/spmw_datapath.py`` (the ``(M+2)x(N+2)`` halo ring for the mesh; channel FIFOs
for the pipeline). Folded/key-form coverage and the analytic perf model (task5.2/5.3) are follow-ups.
"""

import ast
import inspect
import textwrap
from collections import deque


class SPMWDeadlockError(Exception):
    """A rolled-simulator schedule made no progress with coroutines still blocked (e.g. an FIFO too
    small): carries the blocked ``(pid, fifo_key)`` pairs so the stall is diagnosable, not a hang.
    """


class _Fifo:
    """A bounded FIFO: ``put`` blocks (returns False) when full, ``get`` blocks when empty."""

    def __init__(self, depth):
        self.depth = depth
        self.q = deque()

    def can_put(self):
        return len(self.q) < self.depth

    def can_get(self):
        return len(self.q) > 0


def _run_scheduler(coros, fifos, max_steps):
    """Cooperatively drive ``coros`` (name -> generator yielding ``('get', key)`` / ``('put', key,
    v)``) against ``fifos`` (key -> :class:`_Fifo`). Round-robin passes; a pass with zero progress and
    live coroutines is a deadlock. ``max_steps`` bounds the run so a bug can never wedge CI.
    """
    state = {}  # name -> current request, or None once the coroutine is done
    for name, gen in coros.items():
        try:
            state[name] = next(gen)
        except StopIteration:
            state[name] = None
    active = [n for n in coros if state[n] is not None]
    steps = 0
    while active:
        progressed = False
        for name in active:
            op, key = state[name][0], state[name][1]
            fifo = fifos[key]
            if op == "get" and fifo.can_get():
                value = fifo.q.popleft()
                resume = value
            elif op == "put" and fifo.can_put():
                fifo.q.append(state[name][2])
                resume = None
            else:
                continue
            try:
                state[name] = coros[name].send(resume)
            except StopIteration:
                state[name] = None
            progressed = True
        active = [n for n in active if state[n] is not None]
        steps += 1
        if active and not progressed:
            blocked = [(n, state[n][0], state[n][1]) for n in active]
            raise SPMWDeadlockError(
                f"rolled simulator deadlocked: {len(blocked)} coroutine(s) blocked with no FIFO "
                f"progress (increase FIFO depth or fix the schedule); blocked (pid, op, fifo): "
                f"{blocked[:8]}"
            )
        if steps > max_steps:
            raise SPMWDeadlockError(
                f"rolled simulator exceeded its step budget ({max_steps}); likely a livelock/deadlock"
            )
    return steps


# ----------------------------------------------------------------------------------------------------
# AST transform: a work-unit body -> a generator that yields ``('get'/'put', ...)`` requests.
# ----------------------------------------------------------------------------------------------------
class _CoroRewriter(ast.NodeTransformer):
    """Rewrite a work-unit body into a cooperative generator body.

    ``ctx.<port>.get()``/``get_or(...)`` -> ``(yield ('get', PORTS['<port>']))``; ``ctx.<port>.put(v)``
    -> ``(yield ('put', PORTS['<port>'], v))``; ``ctx.rank()`` -> ``RANK``; ``ctx.<name>[...]`` ->
    ``OPS['<name>'][...]`` (a region operand / local buffer); annotated assigns drop their annotation.
    ``PORTS``/``OPS``/``RANK`` are names bound in the generator's namespace per PID.
    """

    def __init__(self, ctx_name, ports):
        self.ctx = ctx_name
        self.ports = ports  # set of port names that map through the FIFO PORTS table

    @staticmethod
    def _expr(text):
        return ast.parse(text, mode="eval").body

    def _ctx_attr(self, node):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.ctx
        ):
            return node.attr
        return None

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if node.value is None:
            return node
        return ast.copy_location(
            ast.Assign(targets=[node.target], value=node.value), node
        )

    def visit_Call(self, node):
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute):
            port = self._ctx_attr(func.value)
            if port in self.ports and func.attr in {"get", "get_or"}:
                return ast.copy_location(
                    self._expr(f"(yield ('get', PORTS[{port!r}]))"), node
                )
            if (
                isinstance(func.value, ast.Name)
                and func.value.id == self.ctx
                and func.attr == "rank"
            ):
                return ast.copy_location(ast.Name("RANK", ast.Load()), node)
        return node

    def visit_Expr(self, node):
        self.generic_visit(node)
        val = node.value
        if (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Attribute)
            and val.func.attr == "put"
            and self._ctx_attr(val.func.value) in self.ports
        ):
            port = self._ctx_attr(val.func.value)
            yield_expr = self._expr(f"(yield ('put', PORTS[{port!r}], _V))")
            # splice the real put argument in for the _V placeholder
            yield_expr.value.elts[2] = val.args[0]
            return ast.copy_location(ast.Expr(yield_expr), node)
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        name = self._ctx_attr(node.value)
        if name is not None and name not in self.ports:
            node.value = ast.copy_location(self._expr(f"OPS[{name!r}]"), node.value)
        return node


def _make_coro(unit_fn, ports, extra_globals):
    """Compile ``unit_fn``'s body into a generator function ``gen(PORTS, OPS, RANK)`` (a Python
    coroutine that yields FIFO requests). ``ports`` are the port names routed through ``PORTS``.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(unit_fn)))
    func = tree.body[0]
    ctx_name = func.args.args[0].arg
    _CoroRewriter(ctx_name, ports).visit(func)
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id == ctx_name:
            raise NotImplementedError(
                f"rolled simulator cannot transcribe unit {unit_fn.__name__!r}: it uses ctx in an "
                f"unsupported form"
            )
    func.args.args = [ast.arg(arg="PORTS"), ast.arg(arg="OPS"), ast.arg(arg="RANK")]
    func.name = "gen"
    func.decorator_list = []
    ast.fix_missing_locations(func)
    namespace = dict(getattr(unit_fn, "__globals__", {}))
    closure = inspect.getclosurevars(unit_fn)
    namespace.update(closure.nonlocals)
    namespace.update(closure.globals)
    namespace.update(extra_globals)
    module = ast.Module(body=[func], type_ignores=[])
    code = compile(module, filename="<spmw_rollsim>", mode="exec")
    exec(code, namespace)  # pylint: disable=exec-used
    return namespace["gen"]


# ----------------------------------------------------------------------------------------------------
# The 2-D systolic mesh: the (M+2)x(N+2) halo ring of allo/spmw_datapath.py:generate_source.
# ----------------------------------------------------------------------------------------------------
class _CellProxy:
    """A 1-element view that redirects any ``[idx]`` access to a fixed ``arr[i, j]`` -- the systolic
    interior writes ``ctx.c_local[0]``, which (like the desugar) lands in ``C[i-1, j-1]``.
    """

    def __init__(self, arr, i, j):
        self.arr, self.i, self.j = arr, i, j

    def __setitem__(self, _idx, value):
        self.arr[self.i, self.j] = value

    def __getitem__(self, _idx):
        return self.arr[self.i, self.j]


def _west_loader(i, a_arr, depth_k):
    for k in range(depth_k):
        yield ("put", ("A", i, 1), a_arr[i - 1, k])


def _north_loader(j, b_arr, depth_k):
    for k in range(depth_k):
        yield ("put", ("B", 1, j), b_arr[k, j - 1])


def _east_drain(i, cols_n, depth_k):
    for _k in range(depth_k):
        yield ("get", ("A", i, cols_n + 1))


def _south_drain(j, rows_m, depth_k):
    for _k in range(depth_k):
        yield ("get", ("B", rows_m + 1, j))


def _build_mesh(collection, operands, order):
    """Coroutines + FIFOs reproducing the ``(M+2)x(N+2)`` systolic halo ring for ``A@B -> C``."""
    decl = collection.maps[0]
    a_name, b_name, c_name = order[0], order[1], order[2]
    a_arr, b_arr, c_arr = operands[a_name], operands[b_name], operands[c_name]
    rows_m, depth_k = a_arr.shape
    cols_n = b_arr.shape[1]
    # The declared mesh must match the operand-derived M x N PE grid, or a stale spmw.mesh((...)) would
    # simulate a different topology than the tensors imply (the dataflow lowering rejects this too).
    # pylint: disable=import-outside-toplevel
    from .spmw_datapath import _check_mesh_grid

    _check_mesh_grid(decl.topology.grid, rows_m, cols_n)
    out = [s for s in collection.streams if s.direction == "out"][0]
    out_attr = out.extra.get("as_", "c_local")
    ports = set(decl.topology.port_names())
    gen_fn = _make_coro(decl.unit.interior, ports, {})

    # A flows west->east on fifo_A, B flows north->south on fifo_B: size each family's FIFOs from the
    # map's resolved per-port depths (the entry port of each flow), not a hard-coded constant.
    depth_a = decl.port_depths.get("west", decl.port_depths.get("east", 2))
    depth_b = decl.port_depths.get("north", decl.port_depths.get("south", 2))
    fifos = {}
    for i in range(1, rows_m + 1):
        for j in range(1, cols_n + 2):
            fifos[("A", i, j)] = _Fifo(depth_a)
    for i in range(1, rows_m + 2):
        for j in range(1, cols_n + 1):
            fifos[("B", i, j)] = _Fifo(depth_b)

    coros = {}
    for i in range(1, rows_m + 1):
        coros[f"wload_{i}"] = _west_loader(i, a_arr, depth_k)
        coros[f"edrain_{i}"] = _east_drain(i, cols_n, depth_k)
    for j in range(1, cols_n + 1):
        coros[f"nload_{j}"] = _north_loader(j, b_arr, depth_k)
        coros[f"sdrain_{j}"] = _south_drain(j, rows_m, depth_k)
    for i in range(1, rows_m + 1):
        for j in range(1, cols_n + 1):
            portmap = {
                "west": ("A", i, j),
                "north": ("B", i, j),
                "east": ("A", i, j + 1),
                "south": ("B", i + 1, j),
            }
            coros[f"pe_{i}_{j}"] = gen_fn(
                portmap, {out_attr: _CellProxy(c_arr, i - 1, j - 1)}, (i - 1, j - 1)
            )
    return coros, fifos


# ----------------------------------------------------------------------------------------------------
# The 1-D producer/consumer pipeline: units joined by spmw.channel FIFOs.
# ----------------------------------------------------------------------------------------------------
def _build_pipeline(collection, operands):
    """Coroutines + FIFOs for a singleton 1-D pipeline: each unit is one coroutine; a channel is one
    shared bounded FIFO both endpoints reference by name."""
    channels = {ch.name: ch for ch in collection.channels}
    for decl in collection.maps:
        if tuple(decl.topology.grid) != (1,):
            raise NotImplementedError(
                "rolled simulator pipeline (task5.1) supports singleton 1-D maps only"
            )
    fifos = {(name,): _Fifo(ch.depth) for name, ch in channels.items()}
    portmap = {name: (name,) for name in channels}
    coros = {}
    for idx, decl in enumerate(collection.maps):
        gen_fn = _make_coro(decl.unit.interior, set(channels), {})
        coros[f"{decl.unit.name}#{idx}"] = gen_fn(portmap, operands, 0)
    return coros, fifos


class _RolledSimModule:
    """Callable returned by ``spmw.build(region, target="rolled_simulator")``: run it with the region
    operands (outputs written in place), exactly like the desugar simulator module."""

    _MAX_STEPS = 5_000_000

    def __init__(self, region):
        self.region = region

    def __call__(self, *args):
        """Run the region on the coroutine scheduler (outputs written in place); returns the number of
        scheduling cycles it took (a cycle-count reference for the analytic model)."""
        # pylint: disable=import-outside-toplevel
        from .spmw import _collect, _validate_collection
        from .spmw_datapath import _recognize, _region_tensors

        collection = _validate_collection(_collect(self.region), strict_topology=True)
        order = [name for name, _, _ in _region_tensors(self.region)]
        if len(args) != len(order):
            raise ValueError(
                f"rolled simulator expected {len(order)} operands {order}, got {len(args)}"
            )
        operands = dict(zip(order, args))

        if collection.channels:
            coros, fifos = _build_pipeline(collection, operands)
        else:
            _recognize(collection)  # fail closed unless it is the 2-D systolic mesh
            coros, fifos = _build_mesh(collection, operands, order)
        return _run_scheduler(coros, fifos, self._MAX_STEPS)


def build_rolled_simulator(region):
    """Build a cooperative coroutine functional simulator for ``region`` (M5 task5.1)."""
    return _RolledSimModule(region)
