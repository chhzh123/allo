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

    collection = _validate_collection(_collect(region))
    source = generate_source(region, collection)
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
    if target != "simulator" and kwargs.get("mode") in {"csyn", "hw_emu", "hw"}:
        # HLS dataflow requires a single writer per array. Each grid point writes a distinct C
        # element (and reads distinct A/B lanes), so complete-partition the operands into per-lane
        # banks before synthesis.
        from allo.customize import Partition

        schedule = df.customize(df_region)
        top = f"{region.name}_df"
        for tensor in ("A", "B", "C"):
            schedule.partition(
                f"{top}:{tensor}", partition_type=Partition.Complete, dim=0
            )
        return schedule.build(target=target, **kwargs)
    return df.build(df_region, target=target, **kwargs)
