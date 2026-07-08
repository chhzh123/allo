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
    # The systolic mesh is one desugar family; a producer/consumer pipeline (units joined by
    # spmw.channel) is another. Try the mesh recognizer, then fall back to the pipeline path.
    try:
        source = generate_source(region, collection)
        systolic = True
    except NotImplementedError:
        source = generate_pipeline_source(region, collection)
        systolic = False
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
        systolic
        and target != "simulator"
        and kwargs.get("mode") in {"csyn", "hw_emu", "hw"}
    ):
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
