# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=cyclic-import

"""Datapath lowering for SPMW: run a systolic-mesh region on the dataflow backend.

This is the first datapath slice. It recognizes the systolic-mesh-GEMM pattern -- a single work
unit mapped over a 2-D mesh, with two ``stream_in`` operands flowing ``W->E`` and ``N->S`` and a
local ``stream_out`` -- and desugars it to the equivalent ``allo.dataflow`` program so the existing
backend can build it (the LLVM/OMP simulator, or the Vitis HLS toolchain via ``target="vitis_hls"``).
The work-unit body's arithmetic is transcribed verbatim (via AST), so the float accumulation order
matches its hand-written ``df`` twin and the result is bit-identical. Other patterns raise
``NotImplementedError`` rather than silently mis-lowering.
"""

import ast
import inspect
import textwrap

# The canonical two-operand systolic mesh: A streams west->east on fifo_A, B streams north->south
# on fifo_B. A get reads this grid point's fifo; a put writes the forwarding neighbor's fifo.
_GETS = {"west": "fifo_A[i, j]", "north": "fifo_B[i, j]"}
_PUTS = {"east": "fifo_A[i, j + 1]", "south": "fifo_B[i + 1, j]"}

_DTYPE_NAMES = {
    "f32": "float32",
    "f64": "float64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "index": "index",
}


class _SystolicRewriter(ast.NodeTransformer):
    """Rewrite ``ctx.<port>.get/put`` and ``ctx.<local>[...]`` into df fifo / local-tensor form."""

    def __init__(self, ctx_name, locals_):
        self.ctx = ctx_name
        self.locals_ = locals_  # attr -> ("local_C", "i - 1", "j - 1")

    @staticmethod
    def _expr(text):
        return ast.parse(text, mode="eval").body

    def _ctx_port(self, node):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.ctx
        ):
            return node.attr
        return None

    def visit_Call(self, node):
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"get", "put", "get_or"}:
            port = self._ctx_port(func.value)
            if port is not None:
                if func.attr in {"get", "get_or"} and port in _GETS:
                    func.value = self._expr(_GETS[port])
                elif func.attr == "put" and port in _PUTS:
                    func.value = self._expr(_PUTS[port])
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        port = self._ctx_port(node.value)
        if port is not None and port in self.locals_:
            tensor, i_expr, j_expr = self.locals_[port]
            node.value = ast.Name(tensor, ast.Load())
            node.slice = ast.Tuple([self._expr(i_expr), self._expr(j_expr)], ast.Load())
        return node


def _resolve_dims(region):
    annotations = getattr(region.fn, "__annotations__", {})
    tensors = [v for v in annotations.values() if getattr(v, "shape", None)]
    if len(tensors) < 2:
        raise NotImplementedError(
            "systolic datapath needs A, B, C tensor arguments on the region"
        )
    a_shape, b_shape = tensors[0].shape, tensors[1].shape  # (M, K), (K, N)
    dtype = repr(tensors[0].dtype)
    if dtype not in _DTYPE_NAMES:
        raise NotImplementedError(
            f"systolic datapath supports {sorted(_DTYPE_NAMES)} elements, not {dtype!r}"
        )
    return a_shape[0], b_shape[1], a_shape[1], _DTYPE_NAMES[dtype]


def _recognize(collection):
    """Check the region is the two-operand systolic mesh pattern; return its map decl."""
    if len(collection.maps) != 1:
        raise NotImplementedError("systolic datapath expects exactly one mapped unit")
    decl = collection.maps[0]
    if decl.topology.dims != 2:
        raise NotImplementedError("systolic datapath expects a 2-D mesh")
    in_flows = sorted(
        s.flow for s in collection.streams if s.direction == "in" and s.flow
    )
    if in_flows != ["N->S", "W->E"]:
        raise NotImplementedError(
            "systolic datapath expects stream_in operands flowing W->E and N->S"
        )
    return decl


def _transcribe_interior(unit, locals_):
    tree = ast.parse(textwrap.dedent(inspect.getsource(unit.interior)))
    func = tree.body[0]
    ctx_name = func.args.args[0].arg
    _SystolicRewriter(ctx_name, locals_).visit(func)
    ast.fix_missing_locations(func)
    return [ast.unparse(stmt) for stmt in func.body]


def generate_source(region, collection):
    """The equivalent allo.dataflow source for a recognized systolic-mesh region."""
    # pylint: disable=import-outside-toplevel
    from .spmw import SPMWError

    decl = _recognize(collection)
    rows, cols, depth, dtype = _resolve_dims(region)
    # The declared mesh must match the operand-derived PE grid, or a stale spmw.mesh((...)) would
    # silently compile a different topology than the one the tensors imply.
    if tuple(decl.topology.grid) != (rows, cols):
        raise SPMWError(
            f"declared mesh grid {tuple(decl.topology.grid)} does not match the operand-derived "
            f"PE grid {(rows, cols)}: A is [M, K] and B is [K, N], so the mesh must be M x N"
        )
    out = [s for s in collection.streams if s.direction == "out"][0]
    locals_ = {out.extra.get("as_", "c_local"): ("local_C", "i - 1", "j - 1")}
    interior = _transcribe_interior(decl.unit, locals_)
    body = "\n".join(" " * 12 + line for text in interior for line in text.splitlines())
    return f"""import allo
from allo.ir.types import {dtype}, Stream
import allo.dataflow as df

M, N, K = {rows}, {cols}, {depth}
P0, P1 = M + 2, N + 2


@df.region()
def {region.name}_df(A: {dtype}[M, K], B: {dtype}[K, N], C: {dtype}[M, N]):
    fifo_A: Stream[{dtype}, 4][P0, P1]
    fifo_B: Stream[{dtype}, 4][P0, P1]

    @df.kernel(mapping=[P0, P1], args=[A, B, C])
    def gemm(local_A: {dtype}[M, K], local_B: {dtype}[K, N], local_C: {dtype}[M, N]):
        i, j = df.get_pid()
        with allo.meta_if(i in {{0, M + 1}} and j in {{0, N + 1}}):
            pass
        with allo.meta_elif(j == 0):
            for k in range(K):
                fifo_A[i, j + 1].put(local_A[i - 1, k])
        with allo.meta_elif(i == 0):
            for k in range(K):
                fifo_B[i + 1, j].put(local_B[k, j - 1])
        with allo.meta_elif(i == M + 1 and j > 0):
            for k in range(K):
                _b: {dtype} = fifo_B[i, j].get()
        with allo.meta_elif(j == N + 1 and i > 0):
            for k in range(K):
                _a: {dtype} = fifo_A[i, j].get()
        with allo.meta_else():
{body}
"""


def _recognize_mini_tpu(collection):
    """Detect the §3.4 Mini-TPU: a 2-D systolic MXU stage feeding a 1-D activation stage.

    Returns ``(mesh_decl, act_decl, mesh_out, act_in, act_out)`` or raises ``NotImplementedError``.
    The two stages are connected by a buffer that the MXU writes (``stream_out``) and the activation
    reads (``stream_in``) -- recognized by identity of the shared tensor, not by name.
    """
    if len(collection.maps) != 2:
        raise NotImplementedError("mini-TPU expects exactly two mapped stages")
    mesh = [d for d in collection.maps if d.topology.dims == 2]
    vec = [d for d in collection.maps if d.topology.dims == 1]
    if len(mesh) != 1 or len(vec) != 1:
        raise NotImplementedError(
            "mini-TPU expects one 2-D mesh (MXU) stage and one 1-D vector (activation) stage"
        )
    mesh_decl, act_decl = mesh[0], vec[0]
    in_flows = sorted(
        s.flow
        for s in collection.streams
        if s.direction == "in" and s.unit is mesh_decl.unit and s.flow
    )
    if in_flows != ["N->S", "W->E"]:
        raise NotImplementedError(
            "mini-TPU MXU stage expects a W->E / N->S output-stationary systolic mesh"
        )
    mesh_out = [
        s
        for s in collection.streams
        if s.direction == "out" and s.unit is mesh_decl.unit
    ]
    act_in = [
        s for s in collection.streams if s.direction == "in" and s.unit is act_decl.unit
    ]
    act_out = [
        s
        for s in collection.streams
        if s.direction == "out" and s.unit is act_decl.unit
    ]
    if len(mesh_out) != 1 or len(act_in) != 1 or len(act_out) != 1:
        raise NotImplementedError(
            "mini-TPU expects the MXU to write one buffer that the activation reads and writes one out"
        )
    if mesh_out[0].tensor is not act_in[0].tensor:
        raise NotImplementedError(
            "mini-TPU activation must read the exact buffer the MXU writes (the psum connection)"
        )
    # The shape above IS a mini-TPU (a 2-D MXU writing a buffer a 1-D activation reads), so a wrong
    # psum banking is a hard configuration error (SPMWError), not a "try the next generator"
    # NotImplementedError. The psum must be declared `banked(on="col")`: the activation reads it one
    # column per lane, so per-column banking is what makes that access conflict-free. Requiring it here
    # makes the declaration load-bearing (the emitter realizes it as per-column conflict-free psum
    # streams -- each lane touches only its own column's FIFOs) rather than a cosmetic annotation.
    # pylint: disable=import-outside-toplevel
    from .spmw import SPMWError

    psum = mesh_out[0].tensor
    if getattr(psum, "kind", None) != "banked" or getattr(psum, "bank_axis", None) != 1:
        raise SPMWError(
            "mini-TPU psum connection must be spmw.banked(on='col') so per-column activation reads are "
            "conflict-free (each lane reads only its column's psum)"
        )
    return mesh_decl, act_decl, mesh_out[0], act_in[0], act_out[0]


def _mesh_store_to_psum_put(stmt_text):
    """Turn the systolic interior's ``local_C[i - 1, j - 1] = <rhs>`` store into a per-element psum
    stream ``psum_fifo[i - 1, j - 1].put(<rhs>)`` -- so the MXU streams its output to the activation
    stage (a sound one-put/one-get connection) instead of writing a shared array (which races).
    """
    node = ast.parse(stmt_text).body[0]
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "local_C"
    ):
        put = ast.Expr(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Subscript(
                        value=ast.Name("psum_fifo", ast.Load()),
                        slice=node.targets[0].slice,
                        ctx=ast.Load(),
                    ),
                    attr="put",
                    ctx=ast.Load(),
                ),
                args=[node.value],
                keywords=[],
            )
        )
        return ast.unparse(ast.fix_missing_locations(put))
    return stmt_text


class _ActRewriter(ast.NodeTransformer):
    """Rewrite the 1-D activation unit body to its dataflow-kernel form.

    ``ctx.rank()`` -> ``df.get_pid()``; the input port ``ctx.<in_port>.get()`` -> the per-row psum
    stream ``psum_fifo[<row>, j].get()``; ``ctx.<bias_attr>`` (a region operand) -> ``local_bias``;
    and the output port ``ctx.<out_port>.put(x)`` -> the per-row store ``local_OUT[<row>, j] = x``.
    ``<row>`` is the enclosing ``for`` loop variable, so the activation consumes its column in row
    order (each psum element is its own stream, so ordering is deterministic).
    """

    def __init__(self, ctx_name, in_port, out_port, bias_attr):
        self.ctx = ctx_name
        self.in_port = in_port
        self.out_port = out_port
        self.bias_attr = bias_attr
        self.rowvars = []

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

    def _row(self):
        return self.rowvars[-1] if self.rowvars else "r"

    def visit_For(self, node):
        rowvar = node.target.id if isinstance(node.target, ast.Name) else None
        self.rowvars.append(rowvar)
        self.generic_visit(node)
        self.rowvars.pop()
        # Unroll the row loop at compile time (``for r in range(n)`` -> ``with allo.meta_for(0, n)
        # as r``) so each per-element psum stream index ``psum_fifo[r, j]`` is a constant row + the
        # lane pid ``j`` -- a runtime loop variable is not a resolvable stream index.
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and rowvar is not None
        ):
            args = node.iter.args
            lb, ub = (
                (ast.Constant(0), args[0]) if len(args) == 1 else (args[0], args[1])
            )
            item = ast.withitem(
                context_expr=ast.Call(
                    func=ast.Attribute(
                        ast.Name("allo", ast.Load()), "meta_for", ast.Load()
                    ),
                    args=[lb, ub],
                    keywords=[],
                ),
                optional_vars=ast.Name(rowvar, ast.Store()),
            )
            return ast.copy_location(ast.With(items=[item], body=node.body), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "rank"
            and isinstance(func.value, ast.Name)
            and func.value.id == self.ctx
        ):
            return self._expr("df.get_pid()")
        if (
            isinstance(func, ast.Attribute)
            and func.attr in {"get", "get_or"}
            and self._ctx_attr(func.value) == self.in_port
        ):
            func.value = self._expr(f"psum_fifo[{self._row()}, j]")
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if self._ctx_attr(node) == self.bias_attr:
            return ast.copy_location(ast.Name("local_bias", ast.Load()), node)
        return node

    def visit_Expr(self, node):
        self.generic_visit(node)
        val = node.value
        if (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Attribute)
            and val.func.attr == "put"
            and self._ctx_attr(val.func.value) == self.out_port
        ):
            target = self._expr(f"local_OUT[{self._row()}, j]")
            target.ctx = ast.Store()
            return ast.copy_location(
                ast.Assign(targets=[target], value=val.args[0]), node
            )
        return node


def generate_mini_tpu_source(region, collection):
    """The allo.dataflow source for the §3.4 Mini-TPU: a systolic MXU stage streaming per-element
    psum to a 1-D bias+ReLU activation stage.

    The MXU is the §3.1 output-stationary systolic mesh verbatim (bit-identical accumulation), except
    each interior PE, instead of storing to a shared array, ``put``s its result onto a per-element
    ``psum_fifo`` stream. The activation is a distinct 1-D vector unit: lane ``j`` reads its column's
    psum in row order, adds the bias, applies ReLU, and writes ``OUT``. The two heterogeneous stages
    run concurrently, connected only by the stream -- exactly the memory/stream composition §3.4 shows.
    """
    # pylint: disable=import-outside-toplevel
    from .spmw import SPMWError

    mesh_decl, act_decl, mesh_out, act_in, act_out = _recognize_mini_tpu(collection)
    tensors = _region_tensors(region)
    shape_of = {name: shape for name, shape, _ in tensors}
    dtype = tensors[0][2]

    mesh_ins = [
        s
        for s in collection.streams
        if s.direction == "in" and s.unit is mesh_decl.unit
    ]
    act_name = next(s.tensor.name for s in mesh_ins if s.flow == "W->E")
    wgt_name = next(s.tensor.name for s in mesh_ins if s.flow == "N->S")
    out_name = act_out.tensor.name
    for needed in (act_name, wgt_name, out_name):
        if needed not in shape_of:
            raise NotImplementedError(
                f"mini-TPU operand {needed!r} is not a shaped region tensor"
            )
    rows, depth = shape_of[act_name]  # ACT is [M, K]
    cols = shape_of[wgt_name][1]  # WGT is [K, N]
    if tuple(mesh_decl.topology.grid) != (rows, cols):
        raise SPMWError(
            f"mini-TPU MXU mesh {tuple(mesh_decl.topology.grid)} does not match the operand-derived "
            f"grid {(rows, cols)} (ACT is [M, K], WGT is [K, N], so the mesh is M x N)"
        )
    if tuple(act_decl.topology.grid) != (cols,):
        raise SPMWError(
            f"mini-TPU activation grid {tuple(act_decl.topology.grid)} must be the output-column "
            f"count ({cols},)"
        )
    bias_name = next(
        (
            name
            for name, shape in shape_of.items()
            if shape == (cols,) and name not in (act_name, wgt_name, out_name)
        ),
        None,
    )
    if bias_name is None:
        raise NotImplementedError(
            f"mini-TPU needs a 1-D bias operand of length {cols} distinct from ACT/WGT/OUT"
        )

    out_attr = mesh_out.extra.get("as_", "c_local")
    interior = _transcribe_interior(
        mesh_decl.unit, {out_attr: ("local_C", "i - 1", "j - 1")}
    )
    interior = [_mesh_store_to_psum_put(stmt) for stmt in interior]
    mesh_body = "\n".join(
        " " * 12 + line for text in interior for line in text.splitlines()
    )

    in_port = act_in.extra.get("as_")
    out_port = act_out.extra.get("as_")
    if not isinstance(in_port, str) or not isinstance(out_port, str):
        raise NotImplementedError(
            "mini-TPU activation streams need as_= port names (the col_in/col_out ports)"
        )
    tree = ast.parse(textwrap.dedent(inspect.getsource(act_decl.unit.interior)))
    func = tree.body[0]
    ctx_name = func.args.args[0].arg
    _ActRewriter(ctx_name, in_port, out_port, "bias").visit(func)
    ast.fix_missing_locations(func)
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id == ctx_name:
            raise NotImplementedError(
                "mini-TPU activation body uses a ctx reference the rewriter does not handle"
            )
    act_body = "\n".join(
        " " * 8 + line for stmt in func.body for line in ast.unparse(stmt).splitlines()
    )

    return f"""import allo
from allo.ir.types import {dtype}, Stream
import allo.dataflow as df

Rt, Ct, K = {rows}, {cols}, {depth}
M, N = Rt, Ct
P0, P1 = M + 2, N + 2


@df.region()
def {region.name}_df(ACT: {dtype}[M, K], WGT: {dtype}[K, N], bias: {dtype}[N], OUT: {dtype}[M, N]):
    fifo_A: Stream[{dtype}, 4][P0, P1]
    fifo_B: Stream[{dtype}, 4][P0, P1]
    psum_fifo: Stream[{dtype}, 2][M, N]

    @df.kernel(mapping=[P0, P1], args=[ACT, WGT])
    def mxu(local_A: {dtype}[M, K], local_B: {dtype}[K, N]):
        i, j = df.get_pid()
        with allo.meta_if(i in {{0, M + 1}} and j in {{0, N + 1}}):
            pass
        with allo.meta_elif(j == 0):
            for k in range(K):
                fifo_A[i, j + 1].put(local_A[i - 1, k])
        with allo.meta_elif(i == 0):
            for k in range(K):
                fifo_B[i + 1, j].put(local_B[k, j - 1])
        with allo.meta_elif(i == M + 1 and j > 0):
            for k in range(K):
                _b: {dtype} = fifo_B[i, j].get()
        with allo.meta_elif(j == N + 1 and i > 0):
            for k in range(K):
                _a: {dtype} = fifo_A[i, j].get()
        with allo.meta_else():
{mesh_body}

    @df.kernel(mapping=[N], args=[bias, OUT])
    def act(local_bias: {dtype}[N], local_OUT: {dtype}[M, N]):
{act_body}
"""


def _region_tensors(region):
    """Ordered ``[(name, shape_tuple, dtype_name)]`` for the region's shaped operands."""
    annotations = getattr(region.fn, "__annotations__", {})
    out = []
    for name, typ in annotations.items():
        shape = getattr(typ, "shape", None)
        if shape is None:
            continue
        dtype = repr(typ.dtype)
        if dtype not in _DTYPE_NAMES:
            raise NotImplementedError(
                f"pipeline datapath supports {sorted(_DTYPE_NAMES)} elements, not {dtype!r}"
            )
        out.append((name, tuple(shape), _DTYPE_NAMES[dtype]))
    return out


class _StripeRewriter(ast.NodeTransformer):
    """Rewrite a 1-D systolic compute body to its dataflow-kernel form.

    ``ctx.west.get()`` reads this column's A fifo, ``ctx.east.put(a)`` forwards A to the next column,
    ``ctx.north.get()`` reads this column's B fifo, and ``ctx.<local>[m]`` stores to
    ``local_C[m, j - 1]`` (this column's output column).
    """

    def __init__(self, ctx_name, out_attr):
        self.ctx = ctx_name
        self.out_attr = out_attr

    @staticmethod
    def _expr(text):
        return ast.parse(text, mode="eval").body

    def _ctx_port(self, node):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.ctx
        ):
            return node.attr
        return None

    def visit_Call(self, node):
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"get", "put", "get_or"}:
            port = self._ctx_port(func.value)
            if port == "west" and func.attr in {"get", "get_or"}:
                func.value = self._expr("fifo_A[i, j]")
            elif port == "north" and func.attr in {"get", "get_or"}:
                func.value = self._expr("fifo_B[i, j]")
            elif port == "east" and func.attr == "put":
                func.value = self._expr("fifo_A[i, j + 1]")
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if self._ctx_port(node.value) == self.out_attr:
            node.value = ast.Name("local_C", ast.Load())
            node.slice = ast.Tuple([node.slice, self._expr("j - 1")], ast.Load())
        return node


def _recognize_1d_systolic(collection):
    """Detect a 1-D systolic stripe: a single unit on a 1-row mesh, A flowing W->E, B fed N->S.

    Returns ``(decl, out_attr)`` or raises ``NotImplementedError``. The unit reads A west / B north,
    forwards A east, and stores per output row -- the output-stationary GEMM stripe.
    """
    if len(collection.maps) != 1:
        raise NotImplementedError("1-D systolic expects exactly one mapped unit")
    decl = collection.maps[0]
    grid = tuple(decl.topology.grid)
    if len(grid) != 2 or grid[0] != 1:
        raise NotImplementedError("1-D systolic expects a 1-row (1 x cols) mesh")
    in_flows = sorted(
        s.flow for s in collection.streams if s.direction == "in" and s.flow
    )
    if in_flows != ["N->S", "W->E"]:
        raise NotImplementedError(
            "1-D systolic expects stream_in flowing W->E and N->S"
        )
    outs = [s for s in collection.streams if s.direction == "out"]
    if len(outs) != 1:
        raise NotImplementedError("1-D systolic expects exactly one stream_out")
    return decl, outs[0].extra.get("as_", "c_local")


def generate_1d_systolic_source(region, collection):
    """The allo.dataflow source for a 1-D output-stationary systolic stripe.

    The compute PE body is transcribed verbatim into the ``2 x (K + 2)`` stripe: row 0 feeds B down
    each column, column 0 feeds A across, the last column drains A, and the interior columns compute
    -- so the float accumulation order matches the hand-written df stripe and the result is
    bit-identical.
    """
    # pylint: disable=import-outside-toplevel
    from .spmw import SPMWError

    decl, out_attr = _recognize_1d_systolic(collection)
    tensors = _region_tensors(region)
    shape_of = {name: shape for name, shape, _ in tensors}
    if "A" not in shape_of or "B" not in shape_of or "C" not in shape_of:
        raise NotImplementedError("1-D systolic needs A, B, C tensor operands")
    rows, depth = shape_of["A"]  # A is [M, K]
    cols = shape_of["B"][1]  # B is [K, N]
    dtype = tensors[0][2]

    tree = ast.parse(textwrap.dedent(inspect.getsource(decl.unit.interior)))
    func = tree.body[0]
    ctx_name = func.args.args[0].arg
    _StripeRewriter(ctx_name, out_attr).visit(func)
    ast.fix_missing_locations(func)
    # Fall through to the 2-D mesh path if any ctx reference survives (e.g. a 1-row 2-D body using
    # ctx.south): the region is not this stripe pattern. This is a NotImplementedError (try next),
    # distinct from the shape check below (a terminal SPMWError for a genuine stripe).
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id == ctx_name:
            raise NotImplementedError(
                "1-D systolic body uses a port the stripe rewriter does not handle"
            )
    # The region IS a stripe. The copied template balances only when the compute-column count (K),
    # the output-column count (N), and the B-feed/compute counts (M vs N) all agree -- the square
    # M == N == K case -- so reject other shapes rather than compute too few columns / index out of
    # bounds / deadlock.
    if not rows == cols == depth:
        raise SPMWError(
            f"1-D systolic stripe currently supports only the square M == N == K case; "
            f"got M={rows}, N={cols}, K={depth}"
        )
    body = "\n".join(
        " " * 12 + line for stmt in func.body for line in ast.unparse(stmt).splitlines()
    )
    return f"""import allo
from allo.ir.types import {dtype}, Stream
import allo.dataflow as df

M, N, K = {rows}, {cols}, {depth}
P0 = K + 2


@df.region()
def {region.name}_df(A: {dtype}[M, K], B: {dtype}[K, N], C: {dtype}[M, N]):
    fifo_A: Stream[{dtype}, 4][2, P0]
    fifo_B: Stream[{dtype}, 4][2, P0]

    @df.kernel(mapping=[2, P0], args=[A, B, C])
    def gemm(local_A: {dtype}[M, K], local_B: {dtype}[K, N], local_C: {dtype}[M, N]):
        i, j = df.get_pid()
        with allo.meta_if(i == 0 and (j == 0 or j == P0 - 1)):
            pass
        with allo.meta_elif(i == 0):
            for _ in range(N):
                for k in range(K):
                    fifo_B[i + 1, j].put(local_B[k, j - 1])
        with allo.meta_elif(j == 0):
            for m in range(M):
                for k in range(K):
                    fifo_A[i, j + 1].put(local_A[m, k])
        with allo.meta_elif(j == P0 - 1):
            for m in range(M):
                for _ in range(K):
                    _a: {dtype} = fifo_A[i, j].get()
        with allo.meta_else():
{body}
"""


class _PipelineRewriter(ast.NodeTransformer):
    """Rewrite a pipeline unit body to its dataflow-kernel form.

    ``ctx.<tensor>[...]`` becomes the kernel's local operand ``local_<tensor>[...]`` and
    ``ctx.<channel>.put/get(...)`` becomes the region-level ``<channel>.put/get(...)``.
    """

    def __init__(self, ctx_name, tensors, channels, pid_var=None):
        self.ctx = ctx_name
        self.tensors = tensors
        self.channels = channels
        # For a replicated (extent>1) grid, ``pid_var`` names the df.get_pid() variable that indexes
        # rank() and the channel arrays. For a size-1 grid it is None: rank() is the constant 0 and
        # channels are scalar.
        self.pid_var = pid_var

    def _rank_node(self):
        if self.pid_var is not None:
            return ast.Name(id=self.pid_var, ctx=ast.Load())
        return ast.Constant(value=0)

    def _ctx_attr(self, node):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.ctx
        ):
            return node.attr
        return None

    def visit_Subscript(self, node):
        self.generic_visit(node)
        name = self._ctx_attr(node.value)
        if name in self.tensors:
            node.value = ast.Name(id=f"local_{name}", ctx=ast.Load())
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute):
            # ctx.rank() -> this replica's pid (a variable when replicated, else the constant 0).
            if self._ctx_attr(func) == "rank":
                return self._rank_node()
            # ctx.<channel>.put/get(...) -> <channel>[pid].put/get(...) (or scalar).
            if func.attr in {"put", "get", "get_or"}:
                chan = self._ctx_attr(func.value)
                if chan in self.channels:
                    base = ast.Name(id=chan, ctx=ast.Load())
                    if self.pid_var is not None:
                        base = ast.Subscript(
                            value=base,
                            slice=ast.Name(id=self.pid_var, ctx=ast.Load()),
                            ctx=ast.Load(),
                        )
                    node.func = ast.Attribute(
                        value=base, attr=func.attr, ctx=ast.Load()
                    )
        return node


def _unit_tensor_args(unit, tensor_names):
    """The region tensor operands a unit body accesses (``ctx.<tensor>``), in first-use order."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(unit.interior)))
    func = tree.body[0]
    ctx_name = func.args.args[0].arg
    used = []
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == ctx_name
            and node.attr in tensor_names
            and node.attr not in used
        ):
            used.append(node.attr)
    return used


def _transcribe_pipeline_unit(unit, tensor_names, channel_names, pid_var):
    tree = ast.parse(textwrap.dedent(inspect.getsource(unit.interior)))
    func = tree.body[0]
    ctx_name = func.args.args[0].arg
    _PipelineRewriter(ctx_name, tensor_names, channel_names, pid_var).visit(func)
    ast.fix_missing_locations(func)
    # Fail closed: every ctx.<...> use must have been rewritten. A leftover ctx reference means the
    # body used a form (a ctx alias, ctx.port("..."), an undeclared port) the transcriber does not
    # handle -- so the generated dataflow would reference an undefined ctx.
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id == ctx_name:
            raise NotImplementedError(
                f"pipeline datapath cannot transcribe unit {unit.name!r}: it uses ctx in a form "
                f"the transcriber does not support (use direct ctx.<tensor>/<channel> access)"
            )
    return [ast.unparse(stmt) for stmt in func.body]


def _validate_pipeline_channels(collection, channel_names):
    """Each channel must have exactly one producer map and one distinct consumer map.

    Endpoints are counted per *mapped declaration* (by index), not by unit name -- so the same unit
    mapped twice counts as two endpoints -- and the producer and consumer must be different maps (a
    channel a single map both puts and gets is a self-loop).
    """
    # pylint: disable=import-outside-toplevel
    from .spmw import SPMWError

    putters = {name: set() for name in channel_names}
    getters = {name: set() for name in channel_names}
    for idx, decl in enumerate(collection.maps):
        tree = ast.parse(textwrap.dedent(inspect.getsource(decl.unit.interior)))
        func = tree.body[0]
        ctx_name = func.args.args[0].arg
        for node in ast.walk(func):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                continue
            inner = node.func.value
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == ctx_name
                and inner.attr in channel_names
            ):
                if node.func.attr == "put":
                    putters[inner.attr].add(idx)
                elif node.func.attr in {"get", "get_or"}:
                    getters[inner.attr].add(idx)
    for name in channel_names:
        if len(putters[name]) != 1 or len(getters[name]) != 1:
            raise SPMWError(
                f"channel {name!r} must have exactly one producer map and one consumer map; got "
                f"{len(putters[name])} producer(s) and {len(getters[name])} consumer(s)"
            )
        if putters[name] == getters[name]:
            raise SPMWError(
                f"channel {name!r} is put and got by the same map (self-loop); a channel connects "
                f"two different maps"
            )


def _channel_payload(ch):
    """``(payload_type_text, element_dtype_name)`` for a channel's Stream payload.

    A channel may carry a scalar (``float32``) or a small vector (``float32[Mt]``).
    """
    shape = getattr(ch.dtype, "shape", None)
    if shape:
        elem = _DTYPE_NAMES.get(repr(ch.dtype.dtype))
        if elem is None:
            raise NotImplementedError(
                f"pipeline channel element {ch.dtype!r} unsupported"
            )
        dims = ", ".join(str(d) for d in shape)
        return f"{elem}[{dims}]", elem
    elem = _DTYPE_NAMES.get(repr(ch.dtype))
    if elem is None:
        raise NotImplementedError(f"pipeline channel dtype {ch.dtype!r} unsupported")
    return elem, elem


def _map_size(decl):
    size = 1
    for extent in decl.topology.grid:
        size *= extent
    return size


def generate_pipeline_source(region, collection):
    """The equivalent allo.dataflow source for a producer/consumer pipeline region.

    A region that declares two or more units connected by ``spmw.channel`` becomes one
    ``@df.kernel(mapping=[1])`` per unit plus a ``Stream`` per channel, with each unit body
    transcribed verbatim -- so the result is bit-identical to a hand-written dataflow pipeline.
    """
    if not collection.channels:
        raise NotImplementedError("pipeline datapath expects at least one spmw.channel")
    tensors = _region_tensors(region)
    if not tensors:
        raise NotImplementedError(
            "pipeline datapath needs typed tensor operands on the region"
        )
    tensor_names = {name for name, _, _ in tensors}
    channel_names = {ch.name for ch in collection.channels}
    _validate_pipeline_channels(collection, channel_names)
    shape_of = {name: (shape, dt) for name, shape, dt in tensors}

    # Constants (M, N, ...) and helper functions the unit bodies reference from their closure.
    consts = {}
    helpers = {}
    for decl in collection.maps:
        closure = inspect.getclosurevars(decl.unit.interior)
        for key, value in {**closure.nonlocals, **closure.globals}.items():
            if isinstance(value, int) and not isinstance(value, bool):
                consts.setdefault(key, value)
            elif inspect.isfunction(value):
                helpers.setdefault(key, value)

    def _sig(name, shape, dt):
        dims = ", ".join(str(d) for d in shape)
        return f"{name}: {dt}[{dims}]"

    dtypes_used = {dt for _, _, dt in tensors}

    # Replication: a unit mapped over a 1-D grid of extent P>1 becomes a mapping=[P] kernel whose
    # channels are P-wide arrays indexed by df.get_pid(); a size-1 grid is a plain singleton. A
    # multi-dimensional grid would give a tuple pid (not the scalar the channel indexing assumes), so
    # the pipeline path handles 1-D replication only -- multi-D systolic arrays are the mesh family.
    for decl in collection.maps:
        if decl.topology.dims != 1:
            raise NotImplementedError(
                "pipeline datapath handles 1-D replication only; "
                f"unit {decl.unit.name!r} has a {decl.topology.dims}-D grid"
            )
    sizes = {_map_size(decl) for decl in collection.maps}
    if sizes == {1}:
        replicated, pid_extent, pid_var = False, 1, None
    elif len(sizes) == 1:
        replicated, pid_extent, pid_var = True, sizes.pop(), "_pid"
    else:
        raise NotImplementedError(
            "pipeline datapath needs a uniform replication factor across the units"
        )

    chan_lines = []
    for ch in collection.channels:
        payload, elem = _channel_payload(ch)
        dtypes_used.add(elem)
        array = f"[{pid_extent}]" if replicated else ""
        chan_lines.append(f"    {ch.name}: Stream[{payload}, {ch.depth}]{array}")

    kernels = []
    name_counts = {}
    for decl in collection.maps:
        used = _unit_tensor_args(decl.unit, tensor_names)
        body = _transcribe_pipeline_unit(
            decl.unit, tensor_names, channel_names, pid_var
        )
        if replicated:
            body = ["_pid = df.get_pid()"] + body
        params = ", ".join(_sig(f"local_{t}", *shape_of[t]) for t in used)
        args = ", ".join(used)
        body_text = "\n".join(
            " " * 8 + line for stmt in body for line in stmt.splitlines()
        )
        # A unit mapped more than once needs a distinct kernel name per instance.
        seen = name_counts.get(decl.unit.name, 0)
        name_counts[decl.unit.name] = seen + 1
        kernel_name = decl.unit.name if seen == 0 else f"{decl.unit.name}_{seen}"
        kernels.append(
            f"    @df.kernel(mapping=[{_map_size(decl)}], args=[{args}])\n"
            f"    def {kernel_name}({params}):\n"
            f"{body_text}"
        )

    const_lines = "\n".join(f"{key} = {value}" for key, value in consts.items())
    # Helper functions the bodies call are emitted at module level; they may use `index`.
    helper_src = "\n\n\n".join(
        textwrap.dedent(inspect.getsource(fn)) for fn in helpers.values()
    )
    if helpers:
        dtypes_used.add("index")
    imports = ", ".join(sorted(dtypes_used))
    region_sig = ", ".join(_sig(name, shape, dt) for name, shape, dt in tensors)
    return (
        "import allo\n"
        f"from allo.ir.types import {imports}, Stream\n"
        "import allo.dataflow as df\n\n"
        f"{const_lines}\n\n\n"
        f"{helper_src}\n\n\n"
        "@df.region()\n"
        f"def {region.name}_df({region_sig}):\n"
        + "\n".join(chan_lines)
        + "\n\n"
        + "\n\n".join(kernels)
        + "\n"
    )


# --- FFT: key-form `lane` butterfly topology ---------------------------------------------------
#
# A radix-2 FFT is a permutation network: butterfly (s, b) reads two lanes at stage s and writes two
# lanes at stage s+1, wired by key-form `("lane_*", stage, slot)` links rather than mesh neighbors.
# The butterfly slot function is non-affine, so it is not a rolled representative-point link -- but
# the simulator desugar evaluates the topology `link` concretely at every (s, b), so it bakes the
# per-point (upper, lower) slots into compile-time tables and emits the equivalent dataflow FFT: an
# input loader (bit-reversed feed into stage 0), one rolled butterfly kernel over the (S, HALF) grid,
# and an output store draining stage S. The butterfly datapath (the twiddle math) is transcribed from
# the unit body, so the numerics match a hand-written FFT.


class _FFTRewriter(ast.NodeTransformer):
    """Rewrite an FFT butterfly unit body to its dataflow-kernel form.

    ``ctx.rank()`` becomes ``df.get_pid()``; ``ctx.<port>.get()`` / ``ctx.<port>.put(x)`` becomes a
    read/write of the stage lane array the port's key binds -- ``stage_<family>[<stage>, <slot>]``,
    where ``<stage>`` is the pid or pid+1 and ``<slot>`` is the injected ``upper``/``lower`` constant.
    """

    def __init__(self, ctx_name, pid0, ports):
        self.ctx = ctx_name
        self.pid0 = pid0  # the first grid pid name (the stage index), e.g. "s"
        self.ports = ports  # port -> (stage_array, stage_offset in {0,1}, slot_role)

    def _is_ctx(self, node):
        return isinstance(node, ast.Name) and node.id == self.ctx

    def visit_Call(self, node):
        self.generic_visit(node)
        func = node.func
        if not isinstance(func, ast.Attribute):
            return node
        if func.attr == "rank" and self._is_ctx(func.value):
            return ast.parse("df.get_pid()", mode="eval").body
        if func.attr in {"get", "put", "get_or"}:
            recv = func.value
            if (
                isinstance(recv, ast.Attribute)
                and self._is_ctx(recv.value)
                and recv.attr in self.ports
            ):
                array, offset, role = self.ports[recv.attr]
                stage = self.pid0 if offset == 0 else f"{self.pid0} + 1"
                slot = "upper" if role == "upper" else "lower"
                receiver = ast.parse(f"{array}[{stage}, {slot}]", mode="eval").body
                node.func = ast.Attribute(
                    value=receiver, attr=func.attr, ctx=ast.Load()
                )
        return node


def _stage_array(family):
    """A dataflow Stream-array identifier for a lane family (``lane_re`` -> ``stage_lane_re``)."""
    safe = "".join(c if c.isalnum() else "_" for c in family)
    return f"stage_{safe}"


def _is_rank_call(value, ctx_name):
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "rank"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == ctx_name
    )


def _recognize_fft(collection):
    """Recognize a key-form butterfly `lane` topology; return a desugar descriptor or raise.

    The topology is checked from first principles -- every link key-form ``(family, stage, slot)``,
    an in-place radix-2 butterfly (each butterfly reads and writes the same two lanes, at stage s then
    s+1), and each stage a permutation of the N lanes -- so an unrecognized topology raises rather
    than mis-lowering. Returns the grid (S, HALF), point count N, the two lane families, the per-(s,b)
    upper/lower slot tables, and each port's (stage array, stage offset, slot role).
    """
    # pylint: disable=import-outside-toplevel,too-many-locals
    from .spmw import _is_key_form

    if len(collection.maps) != 1:
        raise NotImplementedError("FFT datapath expects exactly one mapped unit")
    decl = collection.maps[0]
    topology = decl.topology
    if topology.dims != 2:
        raise NotImplementedError("FFT datapath expects a 2-D (stage, butterfly) grid")
    n_stages, half = int(topology.grid[0]), int(topology.grid[1])
    n_points = half * 2

    port_family, port_offset, port_dir = {}, {}, {}
    up_table, lo_table = {}, {}
    families = []
    for coord in topology.coords():
        stage_idx = coord[0]
        in_slots, out_slots = set(), set()
        links = topology.links_at(coord)
        if not links or not all(_is_key_form(t) for t in links.values()):
            raise NotImplementedError(
                "FFT datapath expects an all-key-form `lane` topology"
            )
        for port, (key, direction) in links.items():
            if not (isinstance(key, tuple) and len(key) == 3):
                raise NotImplementedError("FFT lane key must be (family, stage, slot)")
            family, stage, slot = key
            offset = stage - stage_idx
            if offset not in (0, 1):
                raise NotImplementedError("FFT lane key stage must be s or s+1")
            if port in port_family and (
                port_family[port] != family
                or port_offset[port] != offset
                or port_dir[port] != direction
            ):
                raise NotImplementedError(
                    "FFT port binds an inconsistent (family, stage, direction)"
                )
            port_family[port], port_offset[port], port_dir[port] = (
                family,
                offset,
                direction,
            )
            if family not in families:
                families.append(family)
            (in_slots if direction == "sink" else out_slots).add(slot)
        if len(in_slots) != 2 or in_slots != out_slots:
            raise NotImplementedError(
                "FFT butterfly must read and write the same two lanes"
            )
        up_table[coord], lo_table[coord] = sorted(in_slots)

    if len(families) != 2:
        raise NotImplementedError("FFT datapath expects two lane families (real, imag)")
    for stage_idx in range(n_stages):
        written = [up_table[(stage_idx, b)] for b in range(half)]
        written += [lo_table[(stage_idx, b)] for b in range(half)]
        if sorted(written) != list(range(n_points)):
            raise NotImplementedError("FFT stage is not a lane permutation")

    port_role = {}
    for port in port_family:
        roles = {
            "upper" if topology.links_at(c)[port][0][2] == up_table[c] else "lower"
            for c in topology.coords()
        }
        if len(roles) != 1:
            raise NotImplementedError(
                "FFT port slot role is not constant over the grid"
            )
        port_role[port] = roles.pop()

    ports = {
        p: (_stage_array(fam), port_offset[p], port_role[p])
        for p, fam in port_family.items()
    }
    return {
        "decl": decl,
        "S": n_stages,
        "HALF": half,
        "N": n_points,
        "families": families,
        "up_table": up_table,
        "lo_table": lo_table,
        "ports": ports,
    }


def _fft_helpers(unit_fn):
    """The module-level functions an FFT unit body references (twiddle helpers)."""
    closure = inspect.getclosurevars(unit_fn)
    return [
        v
        for v in {**closure.nonlocals, **closure.globals}.values()
        if inspect.isfunction(v)
    ]


def _fam_tensor_pairs(stream):
    """``[(family, tensor_name)]`` for a key-stage stream, matched positionally."""
    families = stream.unit if isinstance(stream.unit, tuple) else (stream.unit,)
    tensors = stream.tensor if isinstance(stream.tensor, tuple) else (stream.tensor,)
    if len(families) != len(tensors):
        raise NotImplementedError("FFT stream family/tensor count mismatch")
    return [(f, getattr(t, "name", None)) for f, t in zip(families, tensors)]


def generate_fft_source(region, collection):
    """The equivalent allo.dataflow source for a key-form `lane` butterfly FFT region."""
    # pylint: disable=too-many-locals
    desc = _recognize_fft(collection)
    decl = desc["decl"]
    n_stages, half, n_points = desc["S"], desc["HALF"], desc["N"]

    tensors = _region_tensors(region)
    for name, shape, dtype in tensors:
        if dtype != "float32" or tuple(shape) != (n_points,):
            raise NotImplementedError(
                f"FFT datapath expects float32[{n_points}] operands; {name} is {dtype}{list(shape)}"
            )
    stream_ins = [s for s in collection.streams if s.direction == "in"]
    stream_outs = [s for s in collection.streams if s.direction == "out"]
    if len(stream_ins) != 1 or len(stream_outs) != 1:
        raise NotImplementedError(
            "FFT datapath expects one stream_in and one stream_out"
        )
    if stream_ins[0].extra.get("at_stage") != 0:
        raise NotImplementedError("FFT stream_in must feed at_stage=0")
    if stream_outs[0].extra.get("at_stage") != n_stages:
        raise NotImplementedError("FFT stream_out must drain at_stage=S")

    # module-level helpers: twiddle functions from the unit body, plus the stream_in slot permutation
    helper_fns = {fn.__name__: fn for fn in _fft_helpers(decl.unit.interior)}
    index_fn = stream_ins[0].extra.get("index")
    if callable(index_fn):
        helper_fns[index_fn.__name__] = index_fn
    helper_src = "\n\n\n".join(
        textwrap.dedent(inspect.getsource(fn)) for fn in helper_fns.values()
    )
    reserved = {"N", "S", "HALF"}
    consts = {}
    closure = inspect.getclosurevars(decl.unit.interior)
    for key, value in {**closure.nonlocals, **closure.globals}.items():
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and key not in reserved
        ):
            consts.setdefault(key, value)
    const_lines = "".join(f"{key} = {value}\n" for key, value in consts.items())

    up_flat = tuple(
        desc["up_table"][(s, b)] for s in range(n_stages) for b in range(half)
    )
    lo_flat = tuple(
        desc["lo_table"][(s, b)] for s in range(n_stages) for b in range(half)
    )

    # transcribe the butterfly datapath from the unit body
    func = ast.parse(textwrap.dedent(inspect.getsource(decl.unit.interior))).body[0]
    ctx_name = func.args.args[0].arg
    first = func.body[0]
    if not (
        isinstance(first, ast.Assign)
        and isinstance(first.targets[0], ast.Tuple)
        and _is_rank_call(first.value, ctx_name)
    ):
        raise NotImplementedError("FFT unit body must start with `s, b = ctx.rank()`")
    pid_names = [t.id for t in first.targets[0].elts]
    if len(pid_names) != 2:
        raise NotImplementedError("FFT unit rank() must unpack two grid indices")
    pid0, pid1 = pid_names
    _FFTRewriter(ctx_name, pid0, desc["ports"]).visit(func)
    ast.fix_missing_locations(func)
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id == ctx_name:
            raise NotImplementedError(
                f"FFT datapath cannot transcribe unit {decl.unit.name!r}: it uses ctx in an "
                f"unsupported form"
            )
    stmts = [ast.unparse(s) for s in func.body]
    inject = [
        f"upper: ConstExpr[int32] = _FFT_UP[{pid0} * HALF + {pid1}]",
        f"lower: ConstExpr[int32] = _FFT_LO[{pid0} * HALF + {pid1}]",
    ]
    kernel_stmts = [stmts[0]] + inject + stmts[1:]
    kernel_body = "\n".join(
        " " * 8 + line for stmt in kernel_stmts for line in stmt.splitlines()
    )

    # input loader (bit-reversed feed into stage 0) and output store (drain stage S)
    in_pairs = _fam_tensor_pairs(stream_ins[0])
    out_pairs = _fam_tensor_pairs(stream_outs[0])
    idx_expr = f"{index_fn.__name__}(idx, S)" if callable(index_fn) else "idx"
    loader_lines = ["idx = df.get_pid()"]
    loader_lines += [f"v_{t}: float32 = local_{t}[idx]" for _, t in in_pairs]
    loader_lines += [
        f"{_stage_array(f)}[0, {idx_expr}].put(v_{t})" for f, t in in_pairs
    ]
    store_lines = ["idx = df.get_pid()"]
    store_lines += [
        f"local_{t}[idx] = {_stage_array(f)}[S, idx].get()" for f, t in out_pairs
    ]
    loader_body = "\n".join(" " * 8 + line for line in loader_lines)
    store_body = "\n".join(" " * 8 + line for line in store_lines)

    stage_decls = "\n".join(
        f"    {_stage_array(f)}: Stream[float32, 4][S + 1, N]" for f in desc["families"]
    )
    region_sig = ", ".join(f"{name}: float32[N]" for name, _, _ in tensors)
    in_args = ", ".join(t for _, t in in_pairs)
    out_args = ", ".join(t for _, t in out_pairs)
    in_params = ", ".join(f"local_{t}: float32[N]" for _, t in in_pairs)
    out_params = ", ".join(f"local_{t}: float32[N]" for _, t in out_pairs)
    return (
        "import allo\n"
        "from allo.ir.types import float32, int32, ConstExpr, Stream\n"
        "import allo.dataflow as df\n"
        "from math import cos, sin, pi, log2\n\n"
        f"N, S, HALF = {n_points}, {n_stages}, {half}\n"
        f"{const_lines}"
        f"_FFT_UP = {up_flat!r}\n"
        f"_FFT_LO = {lo_flat!r}\n\n\n"
        f"{helper_src}\n\n\n"
        "@df.region()\n"
        f"def {region.name}_df({region_sig}):\n"
        f"{stage_decls}\n\n"
        f"    @df.kernel(mapping=[N], args=[{in_args}])\n"
        f"    def input_loader({in_params}):\n"
        f"{loader_body}\n\n"
        "    @df.kernel(mapping=[S, HALF])\n"
        "    def butterfly():\n"
        f"{kernel_body}\n\n"
        f"    @df.kernel(mapping=[N], args=[{out_args}])\n"
        f"    def output_store({out_params}):\n"
        f"{store_body}\n"
    )


def build_dataflow(region, target="simulator", **kwargs):
    """Desugar a systolic-mesh region to allo.dataflow and build it for a dataflow target.

    ``target`` and ``kwargs`` pass straight through to ``allo.dataflow.build`` -- so ``"simulator"``
    runs the LLVM/OMP simulator, and ``"vitis_hls"`` (with ``mode="csim"``/``"csyn"``/``"hw_emu"``)
    drives the real HLS toolchain. The generated program is written to a temporary module and
    imported (rather than ``exec``'d) so the dataflow builder can read the kernel's source with
    ``inspect.getsource`` while walking it.
    """
    # pylint: disable=import-outside-toplevel
    import importlib.util
    import os
    import tempfile

    from .spmw import _collect, _validate_collection

    collection = _validate_collection(_collect(region), strict_topology=True)
    # Desugar families, tried in order: the 1-D systolic stripe (a strict 1-row mesh), the 2-D
    # systolic mesh, the key-form `lane` butterfly FFT, and a producer/consumer pipeline (units
    # joined by spmw.channel).
    # Each desugar family also declares which region operands to complete-partition for HLS dataflow
    # (single reader/writer per bank), or ``None`` when the family does not drive the csyn/hw_emu path.
    try:
        source = generate_mini_tpu_source(region, collection)
        hls_operands = ("ACT", "WGT", "bias", "OUT")
    except NotImplementedError:
        try:
            source = generate_1d_systolic_source(region, collection)
            hls_operands = ("A", "B", "C")
        except NotImplementedError:
            try:
                source = generate_source(region, collection)
                hls_operands = ("A", "B", "C")
            except NotImplementedError:
                try:
                    source = generate_fft_source(region, collection)
                    # The FFT input_loader/output_store kernels read/write each region operand per
                    # lane (mapping=[N]); complete-partition them so the HLS Makefile (hw_emu/hw) flow
                    # sees a single reader/writer per bank (else HLS 200-779, as the Mini-TPU hit).
                    hls_operands = tuple(name for name, _, _ in _region_tensors(region))
                except NotImplementedError:
                    source = generate_pipeline_source(region, collection)
                    hls_operands = None
    module_name = f"{region.name}_spmw_df"
    tmp_dir = tempfile.mkdtemp(prefix="spmw_df_")
    path = os.path.join(tmp_dir, module_name + ".py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location(module_name, path)
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)

    import allo.dataflow as df

    df_region = getattr(generated, f"{region.name}_df")
    if (
        hls_operands
        and target != "simulator"
        and kwargs.get("mode") in {"csyn", "hw_emu", "hw"}
    ):
        # HLS dataflow requires a single reader/writer per array. Each grid point / activation lane
        # reads and writes distinct operand banks (e.g. each activation lane writes one OUT column),
        # so complete-partition the operands into per-lane banks before synthesis.
        from allo.customize import Partition

        schedule = df.customize(df_region)
        top = f"{region.name}_df"
        for tensor in hls_operands:
            schedule.partition(
                f"{top}:{tensor}", partition_type=Partition.Complete, dim=0
            )
        return schedule.build(target=target, **kwargs)
    return df.build(df_region, target=target, **kwargs)
