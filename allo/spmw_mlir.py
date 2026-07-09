# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=cyclic-import

"""Lower an SPMW work-unit body to a real allo-dialect datapath inside a ``spmw.map`` role func.

``spmw.lower`` used to emit role funcs as empty ``func.func @role() { return }`` stubs. This module
transcribes the ``@spmw.unit`` interior into the same allo ops the dataflow frontend produces --
``allo.stream_get``/``allo.stream_put``, ``affine.for``, ``arith`` and ``memref`` -- so the rolled
``spmw.map`` carries the genuine computation, once per role, rather than a placeholder. The role
func is parameterized by the writer position (``%pi``, ``%pj``) so a single interior body serves
every interior grid point.

It handles the constructs the systolic PE uses (the pattern the datapath already recognizes);
anything else raises :class:`SPMWError`.
"""

import ast
import inspect
import textwrap

# MLIR element types keyed by ``repr(tensor.dtype)`` (already the MLIR-style name).
_FLOAT_ELEMS = {"f32", "f64"}
_INT_ELEMS = {"i8", "i16", "i32", "i64"}

# Which port each ``ctx.<port>.get()`` reads from and each ``ctx.<port>.put()`` writes to; the
# stream args of the role func are named after the ports.
_READ_PORTS = {"west", "north", "east", "south"}


def _mlir_elem(elem):
    """``(mlir element type, "float"|"int")`` for a ``repr(tensor.dtype)`` name like ``"f32"``."""
    if elem in _FLOAT_ELEMS:
        return elem, "float"
    if elem in _INT_ELEMS:
        return elem, "int"
    from .spmw import SPMWError  # pylint: disable=import-outside-toplevel

    raise SPMWError(
        f"spmw.map lowering supports {sorted(_FLOAT_ELEMS | _INT_ELEMS)}, not {elem!r}"
    )


class _Datapath:
    """Emit allo-MLIR statements for a systolic unit body, tracking SSA values and accumulators."""

    def __init__(self, elem, kind, depths, ports, out_memref, out_shape):
        self.elem = elem  # "f32"
        self.kind = kind  # "float" | "int"
        self.depths = depths  # port name -> FIFO depth of that port's stream
        self.ports = ports  # port name -> SSA arg (e.g. "west" -> "%west")
        self.out_memref = out_memref  # "%C"
        self.out_shape = out_shape  # "2x2xf32"
        self.lines = []
        self._n = 0
        self.mem = {}  # accumulator var -> alloc SSA
        self.val = {}  # single-assignment var -> SSA value

    def _fresh(self):
        v = f"%v{self._n}"
        self._n += 1
        return v

    def _stream_ty(self, port):
        # each port carries its own declared FIFO depth, so reciprocal families with different
        # depths (e.g. A east/west vs B north/south) produce distinct stream types
        return f"!allo.stream<{self.elem}, {self.depths[port]}>"

    def _emit(self, line):
        self.lines.append(line)

    def _eval(self, node):
        """Return an SSA value holding ``node``."""
        if isinstance(node, ast.Name):
            if node.id in self.val:
                return self.val[node.id]
            if node.id in self.mem:
                v = self._fresh()
                self._emit(
                    f"      {v} = affine.load {self.mem[node.id]}[] : memref<{self.elem}>"
                )
                return v
            raise self._unsupported(node)
        if isinstance(node, ast.Constant):
            v = self._fresh()
            lit = (
                f"{float(node.value):.6e}" if self.kind == "float" else int(node.value)
            )
            self._emit(f"      {v} = arith.constant {lit} : {self.elem}")
            return v
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            v = self._fresh()
            self._emit(
                f"      {v} = {self._arith(node.op)} {left}, {right} : {self.elem}"
            )
            return v
        if isinstance(node, ast.Call) and self._is_get(node):
            port = node.func.value.attr
            v = self._fresh()
            self._emit(
                f"      {v} = allo.stream_get({self.ports[port]}, []) : "
                f"{self._stream_ty(port)} -> {self.elem}"
            )
            return v
        raise self._unsupported(node)

    def _arith(self, op):
        table = (
            {
                ast.Add: "arith.addf",
                ast.Sub: "arith.subf",
                ast.Mult: "arith.mulf",
                ast.Div: "arith.divf",
            }
            if self.kind == "float"
            else {ast.Add: "arith.addi", ast.Sub: "arith.subi", ast.Mult: "arith.muli"}
        )
        if type(op) not in table:
            raise self._unsupported(op)
        return table[type(op)]

    @staticmethod
    def _is_get(node):
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get", "read"}
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr in _READ_PORTS
        )

    @staticmethod
    def _is_put(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"put", "write"}
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr in _READ_PORTS
        )

    @staticmethod
    def _unsupported(node):
        from .spmw import SPMWError  # pylint: disable=import-outside-toplevel

        return SPMWError(f"spmw.map lowering cannot transcribe {ast.dump(node)}")

    def stmt(self, node):
        if isinstance(node, ast.AnnAssign):
            # accumulator init (assigned before the loop, mutated inside) -> memref; otherwise SSA
            name = node.target.id
            value = self._eval(node.value)
            if name in self.mem:
                self._emit(
                    f"      affine.store {value}, {self.mem[name]}[] : memref<{self.elem}>"
                )
            else:
                self.val[name] = value
            return
        if isinstance(node, ast.AugAssign):
            name = node.target.id
            cur = self._eval(ast.Name(name, ast.Load()))
            rhs = self._eval(node.value)
            res = self._fresh()
            self._emit(
                f"      {res} = {self._arith(node.op)} {cur}, {rhs} : {self.elem}"
            )
            self._emit(
                f"      affine.store {res}, {self.mem[name]}[] : memref<{self.elem}>"
            )
            return
        if isinstance(node, ast.Expr) and self._is_put(node.value):
            port = node.value.func.value.attr
            arg = self._eval(node.value.args[0])
            self._emit(
                f"      allo.stream_put({self.ports[port]}, [], {arg}) : "
                f"{self._stream_ty(port)} contains {self.elem}"
            )
            return
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            # ctx.c_local[0] = c  ->  store the accumulator to C[%pi, %pj]
            value = self._eval(node.value)
            self._emit(
                f"      memref.store {value}, {self.out_memref}[%pi, %pj] : "
                f"memref<{self.out_shape}>"
            )
            return
        raise self._unsupported(node)

    def transcribe(self, body, trip):
        """Emit the datapath for a unit ``body`` split into pre-loop / k-loop / post-loop parts."""
        pre, loop, post = [], None, []
        for node in body:
            if isinstance(node, ast.For):
                loop = node
            elif loop is None:
                pre.append(node)
            else:
                post.append(node)
        if loop is None:
            raise self._unsupported(ast.Module(body=body, type_ignores=[]))
        for name in self.mem:
            alloc = self._fresh()
            self.mem[name] = alloc
            self._emit(f"      {alloc} = memref.alloc() : memref<{self.elem}>")
        for node in pre:
            self.stmt(node)
        bound = loop.iter.args[0]
        # ``for k in range(K)``: K is a closure symbol, so use the caller-supplied trip count; a
        # literal bound is used directly.
        count = trip if isinstance(bound, ast.Name) else int(bound.value)
        self._emit(f"    affine.for %k = 0 to {count} {{")
        for node in loop.body:
            self.stmt(node)
        self._emit("    }")
        for node in post:
            self.stmt(node)


def loader_role_func(
    sym, elem, trip, stream_depth, operand, shape, forward_port, p_first
):
    """Synthesize a halo loader role: read the operand row/col and stream it into the array.

    ``p_first`` selects ``operand[%p, %k]`` (a horizontal W->E flow reading A row ``%p``) versus
    ``operand[%k, %p]`` (a vertical N->S flow reading B column ``%p``); ``forward_port`` is the port
    it streams into the array.
    """
    rows, cols = shape
    index = "%p, %k" if p_first else "%k, %p"
    stream_ty = f"!allo.stream<{elem}, {stream_depth}>"
    signature = f"{operand}: memref<{rows}x{cols}x{elem}>, %p: index, %{forward_port}: {stream_ty}"
    body = (
        f"    affine.for %k = 0 to {trip} {{\n"
        f"      %v = memref.load {operand}[{index}] : memref<{rows}x{cols}x{elem}>\n"
        f"      allo.stream_put(%{forward_port}, [], %v) : {stream_ty} contains {elem}\n"
        "    }"
    )
    return f"  func.func @{sym}({signature}) {{\n{body}\n    return\n  }}"


def drain_role_func(sym, elem, trip, stream_depth, in_port):
    """Synthesize a halo drain role: consume the streamed operand at the exit edge."""
    stream_ty = f"!allo.stream<{elem}, {stream_depth}>"
    body = (
        f"    affine.for %k = 0 to {trip} {{\n"
        f"      %v = allo.stream_get(%{in_port}, []) : {stream_ty} -> {elem}\n"
        "    }"
    )
    return f"  func.func @{sym}(%{in_port}: {stream_ty}) {{\n{body}\n    return\n  }}"


def _accumulators(body):
    """Names that are AugAssign targets (loop-carried) -- these need a memref, not an SSA value."""
    names = set()
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def interior_role_func(sym, body_fn, dtype, trip, depths, shapes):
    """Emit a PID-parameterized interior role ``func.func`` carrying the real unit datapath.

    ``body_fn`` is the work-unit body to transcribe (the interior body, or a predicate-selected
    variant body). ``trip`` is the k-loop trip count, ``depths`` maps each port name to its stream
    FIFO depth (so reciprocal families with different declared depths stay distinct, as the op
    verifier requires), and ``shapes`` is ``(a_shape, b_shape, c_shape)`` as ``(rows, cols)`` tuples.
    The func takes the A/B/C memrefs, the writer position ``%pi``/``%pj``, and one ``!allo.stream``
    per port.
    """
    elem, kind = _mlir_elem(dtype)
    tree = ast.parse(textwrap.dedent(inspect.getsource(body_fn)))
    fn = tree.body[0]
    body = fn.body
    ports = {p: f"%{p}" for p in sorted(_READ_PORTS)}
    (am, ak), (bk, bn), (cm, cn) = shapes
    out_shape = f"{cm}x{cn}x{elem}"
    dp = _Datapath(elem, kind, depths, ports, "%C", out_shape)
    dp.mem = {name: None for name in _accumulators(body)}
    dp.transcribe(body, trip)

    stream_args = ", ".join(
        f"%{p}: !allo.stream<{elem}, {depths[p]}>" for p in sorted(_READ_PORTS)
    )
    signature = (
        f"%A: memref<{am}x{ak}x{elem}>, %B: memref<{bk}x{bn}x{elem}>, "
        f"%C: memref<{cm}x{cn}x{elem}>, %pi: index, %pj: index, {stream_args}"
    )
    body_text = "\n".join(dp.lines)
    return f"  func.func @{sym}({signature}) {{\n{body_text}\n    return\n  }}"
