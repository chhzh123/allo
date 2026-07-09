# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=cyclic-import

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
import os
import re
import textwrap

# The default synthesis target for a stand-alone rolled project (matches the flagship's U280 runs).
_DEFAULT_PART = "xcu280-fsvh2892-2L-e"
_DEFAULT_FREQUENCY_MHZ = 300

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
    # Honor the declared per-family FIFO depth, floored at K (a PE streams K elements per channel):
    # the fa (A) family is east/west, the fb (B) family north/south, and they may differ.
    fa_depth = max(
        decl.port_depths.get("east", depth), decl.port_depths.get("west", depth), depth
    )
    fb_depth = max(
        decl.port_depths.get("north", depth),
        decl.port_depths.get("south", depth),
        depth,
    )
    wired = ", ".join(_ROLLED_WIRING[port] for port in ports)
    dispatch = f"      pe_interior({wired}, &C[i][j]);"
    return _rolled_top_cpp(rows, cols, depth, elem, pe_fn, dispatch, fa_depth, fb_depth)


def _rolled_top_cpp(rows, cols, depth, elem, pe_defs, pe_dispatch, fa_depth, fb_depth):
    """The rolled systolic HLS top for the compute-role body definitions ``pe_defs``.

    Shared by the frontend-transcribed emitter and the IR-driven one: given the compute-role body
    definitions and the per-grid-point ``pe_dispatch`` statement (a single call, or a
    coordinate-predicate ``if/else`` chain that selects among predicate-variant bodies), it wires
    the ``load_a``/``load_b``/``drain`` boundary roles and stamps the PE across the whole ``M x N``
    grid (both loops unrolled) over the two FIFO families -- so a single csynth sees one body per
    role (never one per grid point). ``fa_depth``/``fb_depth`` are the per-family FIFO depths the
    resolver recorded (the ``east/west`` A family and the ``north/south`` B family may differ; each
    PE streams ``K`` elements, so a depth must admit that buffering).
    """
    return (
        "#include <hls_stream.h>\n"
        f"#define M {rows}\n"
        f"#define N {cols}\n"
        f"#define K {depth}\n\n"
        f"{pe_defs}\n"
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
        f"#pragma HLS stream variable=fa depth={fa_depth}\n"
        f"#pragma HLS stream variable=fb depth={fb_depth}\n"
        "  for (int i = 0; i < M; i++) {\n"
        "#pragma HLS unroll\n"
        "    load_a(A, i, fa[i][0]);\n"
        "  }\n"
        "  for (int j = 0; j < N; j++) {\n"
        "#pragma HLS unroll\n"
        "    load_b(B, j, fb[0][j]);\n"
        "  }\n"
        "  for (int i = 0; i < M; i++) {\n"
        "#pragma HLS unroll\n"
        "    for (int j = 0; j < N; j++) {\n"
        "#pragma HLS unroll\n"
        f"{pe_dispatch}\n"
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


def _rolled_csynth_tcl(part, frequency, with_testbench=False):
    """A Vitis HLS script for a stand-alone rolled ``top`` project.

    With ``with_testbench`` it registers ``tb.cpp`` and runs ``csim_design`` (the correctness gate:
    does the rolled top compute A@B); otherwise it runs ``csynth_design`` (the synthesis gate).
    """
    if with_testbench:
        action = "add_files -tb tb.cpp\n"
        design = "csim_design\n"
    else:
        action = ""
        design = "csynth_design\n"
    return (
        "open_project rolled.prj\n"
        "set_top top\n"
        "add_files kernel.cpp\n"
        f"{action}"
        "open_solution solution1\n"
        f"set_part {part}\n"
        f"create_clock -period {1000 / frequency:.2f} -name default\n"
        f"{design}"
        "exit\n"
    )


def _rolled_testbench(region):
    """A self-checking C++ testbench that runs the rolled ``top`` and compares against A@B."""
    # pylint: disable=import-outside-toplevel
    from .spmw_datapath import _resolve_dims

    rows, cols, depth, dtype = _resolve_dims(region)
    elem = _CPP_TYPE[dtype]
    check = (
        "if (std::fabs(ref - C[i][j]) > 1e-3) bad = 1;"
        if dtype.startswith("float")
        else "if (ref != C[i][j]) bad = 1;"
    )
    return (
        "#include <cmath>\n"
        "#include <cstdio>\n"
        f"#define M {rows}\n"
        f"#define N {cols}\n"
        f"#define K {depth}\n"
        f"void top({elem} A[M][K], {elem} B[K][N], {elem} C[M][N]);\n"
        "int main() {\n"
        f"  {elem} A[M][K], B[K][N], C[M][N];\n"
        "  for (int i = 0; i < M; i++)\n"
        "    for (int k = 0; k < K; k++)\n"
        f"      A[i][k] = ({elem})((i + 1) + k);\n"
        "  for (int k = 0; k < K; k++)\n"
        "    for (int j = 0; j < N; j++)\n"
        f"      B[k][j] = ({elem})((j + 1) - k);\n"
        "  top(A, B, C);\n"
        "  int bad = 0;\n"
        "  for (int i = 0; i < M; i++)\n"
        "    for (int j = 0; j < N; j++) {\n"
        f"      {elem} ref = 0;\n"
        "      for (int k = 0; k < K; k++)\n"
        "        ref += A[i][k] * B[k][j];\n"
        f"      {check}\n"
        "    }\n"
        '  printf(bad ? "CSIM MISMATCH\\n" : "CSIM MATCH\\n");\n'
        "  return bad;\n"
        "}\n"
    )


class RolledHLSProject:
    """A rolled O(#roles) systolic top emitted as a self-contained Vitis HLS project.

    ``hls_code`` is the synthesizable C++; when a ``project`` directory is given it also holds
    ``kernel.cpp`` and a ``run.tcl`` that csynth it (``vitis_hls -f run.tcl``), so the rolled
    synthesis path is reproducible through ``spmw.build`` rather than hand-assembled.
    """

    def __init__(self, hls_code, project=None):
        self.hls_code = hls_code
        self.project = project


def emit_rolled_project(
    region,
    project=None,
    part=_DEFAULT_PART,
    frequency=_DEFAULT_FREQUENCY_MHZ,
    testbench=False,
):
    """Emit the rolled systolic top for ``region`` as a :class:`RolledHLSProject`.

    With ``testbench`` the project also holds a self-checking ``tb.cpp`` and its ``run.tcl`` runs
    ``csim_design`` (correctness vs A@B) before ``csynth_design``.
    """
    # The public rolled path emits by consuming the rolled spmw.map IR (grid/partition/families +
    # the interior role func's datapath), so target="rolled" and its Vitis csim exercise the
    # IR-driven emitter.
    hls_code = emit_rolled_hls_ir(region)
    if project is not None:
        os.makedirs(project, exist_ok=True)
        with open(os.path.join(project, "kernel.cpp"), "w", encoding="utf-8") as handle:
            handle.write(hls_code)
        with open(os.path.join(project, "run.tcl"), "w", encoding="utf-8") as handle:
            handle.write(_rolled_csynth_tcl(part, frequency, with_testbench=testbench))
        if testbench:
            with open(os.path.join(project, "tb.cpp"), "w", encoding="utf-8") as handle:
                handle.write(_rolled_testbench(region))
    return RolledHLSProject(hls_code, project)


# --- IR-driven rolled emitter: lower to spmw.map, run the M2 passes, and emit the rolled HLS by
# reading the grid/families off the rolled IR and translating the interior role func's transcribed
# datapath (allo ops) to HLS C++ -- so the emission genuinely consumes spmw.map, not the frontend.

# The systolic interior op set the datapath translator handles, mapped to their C++ operators.
_MLIR_ARITH = {"mulf": "*", "addf": "+", "subf": "-", "divf": "/"}


def _interior_body_from_ir(func_text, ports, elem):
    """Translate the interior role func's transcribed MLIR body to a ``pe_interior`` C++ body.

    ``ports`` is the sorted local-port name per stream parameter (the role's declared stream ABI);
    the body's ``allo.stream_get/put`` on those args become ``port.read()/write()``, the ``arith``
    ops become C operators, the accumulator ``memref`` becomes a local, and the final store to the
    output tensor becomes ``c_local[0] = ...``. Only the systolic op set is handled.
    """
    lines = [ln.strip() for ln in func_text.splitlines()]
    # The stream parameters, in signature order, are the declared ports.
    stream_args = re.findall(r"(%\w+): !allo\.stream", lines[0])
    if len(stream_args) != len(ports):
        raise NotImplementedError(
            "interior role stream ABI does not match its declared ports"
        )
    port_of = dict(zip(stream_args, ports))

    val = {}  # SSA name -> C expression (a temp name or a literal)
    acc = None  # the accumulator memref SSA
    out = []
    indent = "  "

    def emit(stmt):
        out.append(indent + stmt)

    for ln in lines[1:]:
        if not ln or (ln == "}" and acc is None):
            continue
        m = re.match(r"(%\w+) = memref\.alloc\(\) :", ln)
        if m:
            acc = m.group(1)
            emit(f"{elem} acc;")  # initialized by the affine.store that follows
            continue
        m = re.match(r"(%\w+) = arith\.constant (\S+) :", ln)
        if m:
            val[m.group(1)] = m.group(2)
            continue
        m = re.match(r"affine\.store (%\w+), " + re.escape(acc or "") + r"\[\]", ln)
        if acc and m:
            emit(f"acc = {val[m.group(1)]};")
            continue
        m = re.match(r"affine\.for (%\w+) = 0 to (\d+)", ln)
        if m:
            emit(
                f"for (int {m.group(1)[1:]} = 0; {m.group(1)[1:]} < {m.group(2)}; {m.group(1)[1:]}++) {{"
            )
            indent = "    "
            continue
        m = re.match(r"(%\w+) = affine\.load " + re.escape(acc or "") + r"\[\]", ln)
        if acc and m:
            val[m.group(1)] = "acc"
            continue
        m = re.match(r"(%\w+) = allo\.stream_get\((%\w+),", ln)
        if m:
            v = f"v{m.group(1)[1:]}"
            emit(f"{elem} {v} = {port_of[m.group(2)]}.read();")
            val[m.group(1)] = v
            continue
        m = re.match(r"(%\w+) = arith\.(mulf|addf|subf|divf) (%\w+), (%\w+)", ln)
        if m:
            v = f"v{m.group(1)[1:]}"
            emit(
                f"{elem} {v} = {val[m.group(3)]} {_MLIR_ARITH[m.group(2)]} {val[m.group(4)]};"
            )
            val[m.group(1)] = v
            continue
        m = re.match(r"allo\.stream_put\((%\w+), \[\], (%\w+)\)", ln)
        if m:
            emit(f"{port_of[m.group(1)]}.write({val[m.group(2)]});")
            continue
        if ln == "}":
            indent = "  "
            emit("}")
            continue
        m = re.match(r"memref\.store (%\w+), %\w+\[", ln)
        if m:
            emit(f"c_local[0] = {val[m.group(1)]};")
            continue
        if ln.startswith("return"):
            break
        raise NotImplementedError(f"IR datapath translator does not handle: {ln}")
    return "\n".join(out)


def emit_rolled_hls_ir(region):
    """The rolled systolic HLS top emitted by *consuming the rolled ``spmw.map`` IR*.

    Lowers ``region`` to the rolled ``spmw.map``, runs ``spmw-role-partition`` and
    ``spmw-resolve-channels``, then reads the grid extents and channel families off the IR and
    translates the interior role func's transcribed datapath to the ``pe_interior`` body. So the
    emitted O(#roles) top is derived from the rolled IR + its partition/family attributes, not
    re-derived from the frontend collection. Bit-for-bit equivalent to :func:`emit_rolled_hls` for
    the systolic twin, but IR-driven.
    """
    # pylint: disable=import-outside-toplevel
    from .spmw import lower, _run_module_pass

    module = lower(region)
    _run_module_pass(module, "spmw-role-partition")
    _run_module_pass(module, "spmw-resolve-channels")
    ir = str(module)

    # The channel families the resolver produced are the FIFO arrays the rolled top declares; the
    # rolled emitter wires the two systolic families (A along east/west, B along north/south).
    families = re.findall(
        r'"([^"]+)"',
        re.search(r"spmw\.channel_families = \[([^\]]*)\]", ir).group(1),
    )
    if sorted(families) != ["east/west", "north/south"]:
        raise NotImplementedError(
            f"IR-driven rolled emitter handles the systolic east/west + north/south families; "
            f"got {families}"
        )
    # The compute roles (the link-presence interior plus any predicate-selected variants) must
    # together tile the grid; the emitter stamps one distinct body per role, never one per point.
    partition = [
        int(x)
        for x in re.search(r"spmw\.partition = array<i64: ([^>]*)>", ir)
        .group(1)
        .split(",")
    ]
    grid = re.search(r"grid = \[(\d+), (\d+)\]", ir)
    rows, cols = int(grid.group(1)), int(grid.group(2))
    if sum(partition) != rows * cols:
        raise NotImplementedError(
            f"IR-driven rolled emitter handles a systolic map whose compute roles tile the grid; "
            f"got partition {partition} for a {rows}x{cols} grid"
        )
    # A is the first tensor operand, memref<rows x K x elem>: its second extent is the contraction K.
    a_shape = re.search(r"memref<(\d+)x(\d+)x(\w+)>", ir)
    depth = int(a_shape.group(2))
    elem = _CPP_TYPE[
        {
            "f32": "float32",
            "f64": "float64",
            "i8": "int8",
            "i16": "int16",
            "i32": "int32",
            "i64": "int64",
        }[a_shape.group(3)]
    ]

    roles = _compute_roles_from_ir(ir, elem)
    # every compute role streams over the same systolic port set, so the wiring is shared
    ports = roles[0]["ports"]
    if set(ports) != set(_ROLLED_WIRING):
        raise NotImplementedError(
            f"IR-driven rolled emitter handles the systolic port set {sorted(_ROLLED_WIRING)}; "
            f"got {ports}"
        )
    pe_defs = "\n".join(role["definition"] for role in roles)
    wired = ", ".join(_ROLLED_WIRING[port] for port in ports)
    pe_dispatch = _pe_dispatch(roles, wired)
    # The per-family FIFO depths the resolver recorded, in the canonical family order
    # (channel_families sorted: "east/west" then "north/south"), each floored at K.
    depths = [
        int(x)
        for x in re.search(r"spmw\.channel_family_depths = array<i64: ([^>]*)>", ir)
        .group(1)
        .split(",")
    ]
    fa_depth, fb_depth = max(depths[0], depth), max(depths[1], depth)
    return _rolled_top_cpp(
        rows, cols, depth, elem, pe_defs, pe_dispatch, fa_depth, fb_depth
    )


# Grid coordinate d0, d1, ... map to the unrolled C++ loop variables i, j for predicate conditions.
_COORD_C = ("i", "j")


def _role_cpp_name(sym):
    """The C++ body name for a compute-role symbol: ``..._interior`` -> ``pe_interior`` etc."""
    match = re.search(r"_(interior|variant\d+)$", sym)
    return f"pe_{match.group(1)}" if match else "pe_interior"


def _extract_affine_map(text, start):
    """The full ``affine_map<...>`` token in ``text`` beginning at ``start`` (paren-balanced)."""
    i = start + len("affine_map<")
    depth = 0
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ">" and depth == 0 and text[i - 1] != "-":
            return text[start : i + 1]
        i += 1
    raise NotImplementedError("unterminated affine_map in role predicate")


def _compute_roles_from_ir(ir, elem):
    """The compute roles of the rolled map, each a dict with its C++ body definition + predicate.

    A compute role is a ``#spmw.role`` with an empty missing set: the interior (no predicate, the
    default) and any predicate-selected variants, in IR order. Each role's transcribed func datapath
    becomes its own distinct ``pe_<role>`` body, so distinct predicate tags stay distinct HLS bodies
    rather than being merged.
    """
    # MLIR hoists affine maps to top-level aliases (`#map4 = affine_map<...>`) and references them,
    # so a role predicate prints as `predicate = #map4`; resolve those aliases back to their text.
    aliases = dict(re.findall(r"^(#\w+) = (affine_map<.*>)$", ir, re.MULTILINE))
    roles = []
    for match in re.finditer(r"#spmw\.role<unit = @(\w+), missing = \[\s*\]", ir):
        sym = match.group(1)
        tail = ir[match.end() :]
        ports_match = re.match(r", ports = \[([^\]]*)\]", tail)
        role_ports = (
            re.findall(r'"([^"]+)"', ports_match.group(1)) if ports_match else []
        )
        rest = tail[ports_match.end() :] if ports_match else tail
        predicate = None
        pred_match = re.match(r", predicate = (#\w+|affine_map<)", rest)
        if pred_match:
            token = pred_match.group(1)
            predicate = (
                aliases[token]
                if token.startswith("#")
                else _extract_affine_map(rest, rest.index("affine_map<"))
            )
        func = re.search(rf"(func\.func @{sym}\(.*?\n  \}})", ir, re.DOTALL).group(1)
        body = _interior_body_from_ir(func, role_ports, elem)
        name = _role_cpp_name(sym)
        args = (
            ", ".join(f"hls::stream<{elem}> &{p}" for p in role_ports)
            + f", {elem} c_local[1]"
        )
        definition = f"void {name}({args}) {{\n#pragma HLS inline off\n{body}\n}}\n"
        roles.append(
            {
                "name": name,
                "ports": role_ports,
                "predicate": predicate,
                "definition": definition,
            }
        )
    return roles


def _predicate_c_condition(affine_map_token):
    """Translate an ``affine_map<(d0,...) -> (expr)>`` indicator to a C++ ``expr != 0`` condition."""
    result = affine_map_token.split("-> ", 1)[1]  # e.g. "((d0 + d1) mod 2)>"
    expr = result.rsplit(">", 1)[0]  # strip the closing map delimiter
    if "ceildiv" in expr:
        raise NotImplementedError(
            "ceildiv predicates are not supported by the rolled emitter"
        )
    expr = re.sub(r"\bd(\d+)\b", lambda m: _COORD_C[int(m.group(1))], expr)
    expr = expr.replace(" mod ", " % ").replace(" floordiv ", " / ")
    return f"{expr} != 0"


def _pe_dispatch(roles, wired):
    """The per-grid-point PE statement: one call, or an if/else chain over the variant predicates.

    Because the instantiation loops are fully unrolled, ``i``/``j`` are compile-time constants, so
    the coordinate predicate folds and each grid point is bound to exactly one variant body.
    """
    base = [role for role in roles if role["predicate"] is None]
    variants = [role for role in roles if role["predicate"] is not None]

    def call(name):
        return f"{name}({wired}, &C[i][j]);"

    if not variants:
        return "      " + call(base[0]["name"])
    if len(base) != 1:
        raise NotImplementedError(
            "a predicate-variant map needs exactly one default (unpredicated) compute role"
        )
    lines = []
    for idx, role in enumerate(variants):
        keyword = "if" if idx == 0 else "} else if"
        lines.append(
            f"      {keyword} ({_predicate_c_condition(role['predicate'])}) {{"
        )
        lines.append(f"        {call(role['name'])}")
    lines.append("      } else {")
    lines.append(f"        {call(base[0]['name'])}")
    lines.append("      }")
    return "\n".join(lines)
