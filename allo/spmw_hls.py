# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HLS role emission for SPMW: transcribe a work-unit body to a synthesizable HLS C++ role.

The synthesis-time win is that each role is synthesized *once*, so a 32x32 mesh csynths 9 role
bodies rather than 1024 cloned ones. This module makes that concrete: a small Python-to-C++
transpiler turns the systolic PE body into a synthesizable ``pe_interior`` function
(``ctx.<port>.get/put`` -> ``stream.read()/write()``, the ``c += a*b`` MAC), which Vitis can csynth
on its own to one role's resources. It handles the constructs the systolic PE uses; anything else
raises ``NotImplementedError``.
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


def _cpp(node):
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


def emit_role_hls(region):
    """A synthesizable single-role HLS C++ file (the interior PE) for a systolic-mesh region."""
    # pylint: disable=import-outside-toplevel
    from .spmw import _collect, _validate_collection
    from .spmw_datapath import _resolve_dims

    collection = _validate_collection(_collect(region))
    decl = collection.maps[0]
    _rows, _cols, depth, dtype = _resolve_dims(region)
    elem = _CPP_TYPE[dtype]
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
    return (
        "#include <hls_stream.h>\n"
        f"#define K {depth}\n\n"
        f"void pe_interior({args}) {{\n{body}\n}}\n"
    )
