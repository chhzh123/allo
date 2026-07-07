# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HLS role emission for SPMW: transcribe a work-unit body to a synthesizable HLS C++ role.

The synthesis-time win is that each role is synthesized *once*, so a 32x32 mesh csynths a handful
of role bodies rather than 1024 cloned ones. This module makes that concrete with a small
Python-to-C++ transpiler that turns the systolic PE body into a synthesizable ``pe_interior``
function (``ctx.<port>.get/put`` -> ``stream.read()/write()``, the ``c += a*b`` MAC):

- ``emit_role_hls`` emits that one role, which Vitis can csynth on its own to one role's resources.
- ``emit_rolled_hls`` emits a whole *rolled* systolic top -- ``pe_interior`` plus the
  ``load_a``/``load_b``/``drain`` boundary roles, each called in a ``#pragma HLS unroll`` grid loop
  over per-key FIFO arrays -- so a single csynth sees a constant number of distinct bodies (one per
  role) no matter the grid size.

It handles the constructs the systolic PE uses; anything else raises ``NotImplementedError``.
"""

import ast
import inspect
import textwrap

_CPP_TYPE = {
    "float32": "float",
    "float64": "double",
    "int8": "signed char",
    "int16": "short",
    "int32": "int",
    "int64": "long long",
}
_CPP_OP = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


def _cpp(node):  # pylint: disable=too-many-return-statements
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Attribute):
        # a work-unit body's attributes are all ctx.<port>/<local>; drop the ctx receiver
        return node.attr
    if isinstance(node, ast.BinOp) and type(node.op) in _CPP_OP:
        return f"({_cpp(node.left)} {_CPP_OP[type(node.op)]} {_cpp(node.right)})"
    if isinstance(node, ast.Subscript):
        return f"{_cpp(node.value)}[{_cpp(node.slice)}]"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        receiver = _cpp(node.func.value)
        method = node.func.attr
        if method in {"get", "read"}:
            return f"{receiver}.read()"
        if method in {"put", "write"}:
            return f"{receiver}.write({_cpp(node.args[0])})"
    raise NotImplementedError(f"cannot transcribe expression to C++: {ast.dump(node)}")


def _cpp_type(annotation):
    if isinstance(annotation, ast.Name) and annotation.id in _CPP_TYPE:
        return _CPP_TYPE[annotation.id]
    raise NotImplementedError(f"unsupported element type: {ast.dump(annotation)}")


def _cpp_stmt(node):
    if isinstance(node, ast.AnnAssign):
        return f"{_cpp_type(node.annotation)} {_cpp(node.target)} = {_cpp(node.value)};"
    if isinstance(node, ast.Assign):
        return f"{_cpp(node.targets[0])} = {_cpp(node.value)};"
    if isinstance(node, ast.AugAssign) and type(node.op) in _CPP_OP:
        return f"{_cpp(node.target)} {_CPP_OP[type(node.op)]}= {_cpp(node.value)};"
    if isinstance(node, ast.Expr):
        return f"{_cpp(node.value)};"
    if isinstance(node, ast.For):
        var = node.target.id
        bound = _cpp(node.iter.args[0])
        inner = "\n".join(
            "  " + line for stmt in node.body for line in _cpp_stmt(stmt).splitlines()
        )
        return f"for (int {var} = 0; {var} < {bound}; {var}++) {{\n{inner}\n}}"
    raise NotImplementedError(f"cannot transcribe statement to C++: {ast.dump(node)}")


def transcribe_pe_cpp(unit):
    """The C++ statements for a work-unit's interior body."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(unit.interior)))
    return [_cpp_stmt(stmt) for stmt in tree.body[0].body]


# The rolled top wires each PE port to a per-key FIFO array element: A streams west->east
# along a row, B streams north->south down a column. Anything but this systolic port set is
# out of scope for the rolled emitter (as with the single-role emitter).
_ROLLED_WIRING = {
    "west": "fa[i][j]",
    "east": "fa[i][j + 1]",
    "north": "fb[i][j]",
    "south": "fb[i + 1][j]",
}


def _pe_interior_fn(decl, elem):
    """(sorted port names, ``pe_interior`` function text) for a systolic work unit.

    ``inline off`` keeps the interior a distinct synthesized module so one role datapath is
    scheduled once and stamped across the grid, rather than inlined per grid point.
    """
    ports = sorted(decl.topology.port_names())
    body = "\n".join(
        "  " + line
        for stmt in transcribe_pe_cpp(decl.unit)
        for line in stmt.splitlines()
    )
    args = (
        ", ".join(f"hls::stream<{elem}> &{port}" for port in ports)
        + f", {elem} c_local[1]"
    )
    return ports, (f"void pe_interior({args}) {{\n#pragma HLS inline off\n{body}\n}}\n")


def emit_role_hls(region):
    """A synthesizable single-role HLS C++ file (the interior PE) for a systolic-mesh region."""
    # pylint: disable=import-outside-toplevel
    from .spmw import _collect, _validate_collection
    from .spmw_datapath import _resolve_dims

    collection = _validate_collection(_collect(region))
    decl = collection.maps[0]
    _rows, _cols, depth, dtype = _resolve_dims(region)
    elem = _CPP_TYPE[dtype]
    _ports, pe_fn = _pe_interior_fn(decl, elem)
    return "#include <hls_stream.h>\n" f"#define K {depth}\n\n" + pe_fn


def emit_rolled_hls(region):
    """A synthesizable *rolled* systolic HLS top: role bodies called in unrolled grid loops.

    The interior ``pe_interior`` and the ``load_a``/``load_b``/``drain`` boundary roles are each
    defined once; ``top`` instantiates them across the grid with ``#pragma HLS unroll`` over
    per-key FIFO arrays under ``#pragma HLS dataflow``. So a single csynth of ``top`` schedules a
    number of distinct function bodies that is constant in the grid size (one per role), not one
    per grid point -- the synthesis-time win, made visible to the tool in one design.
    """
    # pylint: disable=import-outside-toplevel
    from .spmw import _collect, _validate_collection
    from .spmw_datapath import _resolve_dims

    collection = _validate_collection(_collect(region))
    decl = collection.maps[0]
    rows, cols, depth, dtype = _resolve_dims(region)
    elem = _CPP_TYPE[dtype]
    ports, pe_fn = _pe_interior_fn(decl, elem)
    if set(ports) != set(_ROLLED_WIRING):
        raise NotImplementedError(
            f"rolled emitter handles the systolic port set {sorted(_ROLLED_WIRING)}; "
            f"got {ports}"
        )
    wired = ", ".join(_ROLLED_WIRING[port] for port in ports)
    return (
        "#include <hls_stream.h>\n"
        f"#define M {rows}\n"
        f"#define N {cols}\n"
        f"#define K {depth}\n\n"
        f"{pe_fn}\n"
        f"void load_a({elem} A[M][K], int i, hls::stream<{elem}> &out) {{\n"
        "  for (int k = 0; k < K; k++)\n"
        "    out.write(A[i][k]);\n"
        "}\n\n"
        f"void load_b({elem} B[K][N], int j, hls::stream<{elem}> &out) {{\n"
        "  for (int k = 0; k < K; k++)\n"
        "    out.write(B[k][j]);\n"
        "}\n\n"
        f"void drain(hls::stream<{elem}> &in) {{\n"
        "  for (int k = 0; k < K; k++)\n"
        "    in.read();\n"
        "}\n\n"
        f"void top({elem} A[M][K], {elem} B[K][N], {elem} C[M][N]) {{\n"
        "#pragma HLS dataflow\n"
        # Each boundary role reads a distinct slice of A/B and each PE writes one C element;
        # complete partition gives every element its own memory so no interface is read by more
        # than one dataflow process.
        "#pragma HLS array_partition variable=A complete dim=0\n"
        "#pragma HLS array_partition variable=B complete dim=0\n"
        "#pragma HLS array_partition variable=C complete dim=0\n"
        f"  hls::stream<{elem}> fa[M][N + 1];\n"
        f"  hls::stream<{elem}> fb[M + 1][N];\n"
        "#pragma HLS stream variable=fa depth=K\n"
        "#pragma HLS stream variable=fb depth=K\n"
        "  for (int i = 0; i < M; i++) {\n"
        "#pragma HLS unroll\n"
        "    load_a(A, i, fa[i][0]);\n"
        "  }\n"
        "  for (int j = 0; j < N; j++) {\n"
        "#pragma HLS unroll\n"
        "    load_b(B, j, fb[0][j]);\n"
        "  }\n"
        "  for (int i = 0; i < M; i++) {\n"
        "    for (int j = 0; j < N; j++) {\n"
        "#pragma HLS unroll\n"
        f"      pe_interior({wired}, &C[i][j]);\n"
        "    }\n"
        "  }\n"
        "  for (int i = 0; i < M; i++) {\n"
        "#pragma HLS unroll\n"
        "    drain(fa[i][N]);\n"
        "  }\n"
        "  for (int j = 0; j < N; j++) {\n"
        "#pragma HLS unroll\n"
        "    drain(fb[M][j]);\n"
        "  }\n"
        "}\n"
    )
