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


_CONCRETE_STORAGE_IMPL = {"URAM", "BRAM", "LUTRAM"}


def _memory_pragmas(memory_placements):
    """The array-partition (+ optional bind_storage) pragmas for the rolled top's A/B/C arrays.

    Each array defaults to a full ``complete dim=0`` partition (the streaming top needs every
    element independently addressable, so no interface is read by more than one dataflow process).
    A placement overrides that: ``bank_axis >= 0`` partitions only that (banking) axis, and a
    concrete ``impl`` adds a ``bind_storage`` pinning the storage resource.
    """
    placements = memory_placements or {}
    lines = []
    for var in ("A", "B", "C"):
        place = placements.get(var)
        bank_axis = place["bank_axis"] if place else None
        dim = bank_axis + 1 if (bank_axis is not None and bank_axis >= 0) else 0
        lines.append(f"#pragma HLS array_partition variable={var} complete dim={dim}\n")
        if place and place["impl"]:
            lines.append(
                f"#pragma HLS bind_storage variable={var} type=RAM_2P impl={place['impl']}\n"
            )
    return "".join(lines)


def _rolled_top_cpp(  # pylint: disable=too-many-arguments
    rows,
    cols,
    depth,
    elem,
    pe_defs,
    pe_dispatch,
    fa_depth,
    fb_depth,
    memory_placements=None,
    bank_decl="",
    bank_writeback="",
):
    """The rolled systolic HLS top for the compute-role body definitions ``pe_defs``.

    Shared by the frontend-transcribed emitter and the IR-driven one: given the compute-role body
    definitions and the per-grid-point ``pe_dispatch`` statement (a single call, or a
    coordinate-predicate ``if/else`` chain that selects among predicate-variant bodies), it wires
    the ``load_a``/``load_b``/``drain`` boundary roles and stamps the PE across the whole ``M x N``
    grid (both loops unrolled) over the two FIFO families -- so a single csynth sees one body per
    role (never one per grid point). ``fa_depth``/``fb_depth`` are the per-family FIFO depths the
    resolver recorded (the ``east/west`` A family and the ``north/south`` B family may differ; each
    PE streams ``K`` elements, so a depth must admit that buffering).

    ``memory_placements`` maps a top array (``"A"``/``"B"``/``"C"``) to its resolved logical-memory
    placement ``{"impl": <resource or None>, "bank_axis": <int>}``: a concrete ``impl`` emits a
    ``bind_storage`` pragma pinning that array's storage, and ``bank_axis >= 0`` partitions along
    that axis (banking) instead of the default full ``complete`` partition. An array with no
    placement keeps the default ``complete dim=0`` -- so an unplaced design is byte-for-byte
    unchanged.
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
        # Each boundary role reads a distinct slice of A/B and each PE writes one C element; the
        # array-partition pragmas (default full partition, or the placement's banking) keep no
        # interface read by more than one dataflow process, plus any bind_storage the memory model
        # pinned.
        f"{_memory_pragmas(memory_placements)}"
        f"{bank_decl}"
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
        f"{bank_writeback}"
        "}\n"
    )


def _folded_pe_interior(roles, elem):
    """The folded interior PE: it reads its A operand from the banked on-chip buffer (a ``K``-element
    row ``A_row``) instead of the east/west FIFO family, and does not forward A (broadcast from the
    buffer). Returns ``(definition, stream_ports)`` -- the remaining stream ports (north/south) keep
    their FIFO wiring. Derived from the transcribed interior role so the compute datapath is exact.
    """
    if len(roles) != 1 or roles[0]["predicate"] is not None:
        raise NotImplementedError(
            "folded rolled emitter realizes a single (no-variant) interior systolic role"
        )
    role = roles[0]
    if set(role["ports"]) != set(_ROLLED_WIRING):
        raise NotImplementedError(
            f"folded rolled emitter needs the systolic port set {sorted(_ROLLED_WIRING)}; "
            f"got {role['ports']}"
        )
    body = (
        role["definition"]
        .split("{\n#pragma HLS inline off\n", 1)[1]
        .rsplit("\n}", 1)[0]
    )
    loop = re.search(r"for \(int (\w+) = 0;", body)
    if loop is None:
        raise NotImplementedError(
            "folded rolled emitter needs a contraction loop in the interior role"
        )
    loop_var = loop.group(1)
    folded = [
        line.replace("west.read()", f"A_row[{loop_var}]")
        for line in body.splitlines()
        if "east.write("
        not in line  # A is broadcast from the buffer; no FIFO forwarding
    ]
    stream_ports = [p for p in role["ports"] if p not in ("east", "west")]
    args = (
        f"{elem} A_row[K], "
        + ", ".join(f"hls::stream<{elem}> &{p}" for p in stream_ports)
        + f", {elem} c_local[1]"
    )
    definition = (
        f"void {role['name']}({args}) {{\n#pragma HLS inline off\n"
        + "\n".join(folded)
        + "\n}\n"
    )
    return definition, stream_ports


def _folded_top_cpp(
    rows,
    cols,
    depth,
    elem,
    roles,
    c_ref,
    bank_decl,
    bank_writeback,
    fb_depth,
    a_banks,
    memory_placements=None,
):
    """A *folded* systolic top: the east/west (A) family, reclassified to a buffer under fold, is a
    real F2-banked on-chip buffer that the PE reads directly by ``(bank, offset)`` -- there is **no**
    ``fa`` stream, load, or drain for it. B (north/south) stays a FIFO family. Realizes the AC-6
    folded channel->buffer->banking path so the reclassified family is a genuine buffer in the
    datapath, not a mislabeled stream. If the output ``C`` is also banked, its ``C_bank`` storage is
    threaded in (``bank_decl``/``bank_writeback``).
    """
    # pylint: disable=import-outside-toplevel,too-many-arguments
    from .transform.f2_layout import F2LayoutSolver

    n_bits = rows.bit_length() - 1
    bank_bits = a_banks.bit_length() - 1
    helper = F2LayoutSolver(n_bits, bank_bits).solve(
        []
    )  # cyclic banking of the row axis
    num_banks, a_depth = helper.dims()
    bank_i, off_i = helper.bank_expr("i"), helper.offset_expr("i")
    pe_def, stream_ports = _folded_pe_interior(roles, elem)
    wired = ", ".join(_ROLLED_WIRING[p] for p in stream_ports)  # fb[i][j], fb[i + 1][j]
    dispatch = f"      {roles[0]['name']}(A_bank[{bank_i}][{off_i}], {wired}, {c_ref});"
    return (
        "#include <hls_stream.h>\n"
        f"#define M {rows}\n"
        f"#define N {cols}\n"
        f"#define K {depth}\n\n"
        f"{pe_def}\n"
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
        # top-level operand placements (resource=/bank axis) honored on the folded top too
        f"{_memory_pragmas(memory_placements)}"
        f"{bank_decl}"
        # materialize the folded A family as a real F2-banked on-chip buffer, filled from A
        f"  {elem} A_bank[{num_banks}][{a_depth}][K];\n"
        "#pragma HLS array_partition variable=A_bank complete dim=1\n"
        "  for (int i = 0; i < M; i++) {\n"
        "#pragma HLS unroll\n"
        "    for (int k = 0; k < K; k++) {\n"
        "#pragma HLS unroll\n"
        f"      A_bank[{bank_i}][{off_i}][k] = A[i][k];\n"
        "    }\n"
        "  }\n"
        f"  hls::stream<{elem}> fb[M + 1][N];\n"
        f"#pragma HLS stream variable=fb depth={fb_depth}\n"
        "  for (int j = 0; j < N; j++) {\n"
        "#pragma HLS unroll\n"
        "    load_b(B, j, fb[0][j]);\n"
        "  }\n"
        "  for (int i = 0; i < M; i++) {\n"
        "#pragma HLS unroll\n"
        "    for (int j = 0; j < N; j++) {\n"
        "#pragma HLS unroll\n"
        f"{dispatch}\n"
        "    }\n"
        "  }\n"
        "  for (int j = 0; j < N; j++) {\n"
        "#pragma HLS unroll\n"
        "    drain(fb[M][j]);\n"
        "  }\n"
        f"{bank_writeback}"
        "}\n"
    )


def _pingpong_top_cpp(rows, cols, depth, elem, place):
    """A two-epoch K-tiled GEMM with genuine ping-pong double buffering of the B operand.

    ``spmw.shared(B, double=True)`` requests double buffering: two physical on-chip copies of a
    K-tile of B (``B_buf[2][KT][N]``, partitioned on the ping/pong axis) alternate by ``epoch & 1``.
    Each epoch preloads the *next* tile into the alternate copy while it consumes the current copy
    (the overlap), computes a partial GEMM over its K-tile, and accumulates into ``C``. This is a
    tiled compute -- a pure systolic streaming GEMM has no epoch boundary to double-buffer -- so a
    ``double=True`` placement lowers to this ping-pong top rather than the streaming systolic one.
    ``place['impl']`` (a concrete resource) pins the ping-pong buffer's storage.
    """
    kt = depth // 2
    impl = f" impl={place['impl']}" if place.get("impl") else ""
    return (
        "#include <hls_stream.h>\n"
        f"#define M {rows}\n"
        f"#define N {cols}\n"
        f"#define K {depth}\n"
        f"#define KT {kt}\n\n"
        f"void top({elem} A[M][K], {elem} B[K][N], {elem} C[M][N]) {{\n"
        # two physical copies of a K-tile of B (ping + pong); partition the ping/pong axis so the two
        # copies are independent RAMs, and bind them to real dual-port storage
        f"  {elem} B_buf[2][KT][N];\n"
        "#pragma HLS array_partition variable=B_buf complete dim=1\n"
        f"#pragma HLS bind_storage variable=B_buf type=RAM_2P{impl}\n"
        f"  {elem} acc[M][N];\n"
        "  for (int i = 0; i < M; i++)\n"
        "    for (int j = 0; j < N; j++)\n"
        "      acc[i][j] = 0;\n"
        # preload epoch 0's tile into the ping copy
        "  for (int k = 0; k < KT; k++)\n"
        "    for (int j = 0; j < N; j++)\n"
        "      B_buf[0][k][j] = B[k][j];\n"
        "  for (int e = 0; e < 2; e++) {\n"
        "    int cur = e & 1;\n"
        "    int nxt = (e + 1) & 1;\n"
        # preload the next epoch's tile into the alternate copy (overlaps this epoch's compute)
        "    if (e + 1 < 2)\n"
        "      for (int k = 0; k < KT; k++)\n"
        "        for (int j = 0; j < N; j++)\n"
        "          B_buf[nxt][k][j] = B[(e + 1) * KT + k][j];\n"
        # consume: partial GEMM over this K-tile using the current copy, accumulate into acc
        "    for (int i = 0; i < M; i++)\n"
        "      for (int j = 0; j < N; j++) {\n"
        f"        {elem} s = 0;\n"
        "        for (int k = 0; k < KT; k++)\n"
        "          s += A[i][e * KT + k] * B_buf[cur][k][j];\n"
        "        acc[i][j] += s;\n"
        "      }\n"
        "  }\n"
        "  for (int i = 0; i < M; i++)\n"
        "    for (int j = 0; j < N; j++)\n"
        "      C[i][j] = acc[i][j];\n"
        "}\n"
    )


def _transcribe_butterfly(unit_fn, elem):
    """Transcribe a key-form FFT butterfly unit body to a C++ ``bfly`` function body.

    ``ctx.<port>.get()``/``put()`` become ``st_<port>.read()``/``write()``; each ``ConstExpr``
    assignment (the twiddle factors, which depend on the grid point) is lifted to a scalar function
    *parameter* so the one body serves every butterfly, with the per-``(s, b)`` value supplied by the
    instantiation loop. Returns the ordered in/out ports, the const-parameter names, the C++ body, and
    a callable that evaluates the constants for a given grid point (reusing the unit's own twiddle
    helpers, exactly as the frontend evaluates a ``ConstExpr``).
    """
    # pylint: disable=import-outside-toplevel,eval-used
    from .spmw_datapath import _fft_helpers, _is_rank_call

    func = ast.parse(textwrap.dedent(inspect.getsource(unit_fn))).body[0]
    ctx = func.args.args[0].arg

    def cexpr(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.BinOp) and type(node.op) in _CPP_OP:
            return f"({cexpr(node.left)} {_CPP_OP[type(node.op)]} {cexpr(node.right)})"
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return repr(node.value)
        raise NotImplementedError(
            f"FFT butterfly transcriber cannot translate expression {ast.dump(node)}"
        )

    def _ctx_port(node):
        # a `ctx.<port>` attribute -> the port name, else None
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == ctx
        ):
            return node.attr
        return None

    in_ports, out_ports, const_specs, events = [], [], [], []
    pid_names = None
    for stmt in func.body:
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Tuple)
            and _is_rank_call(stmt.value, ctx)
        ):
            pid_names = [t.id for t in stmt.targets[0].elts]
            continue
        if isinstance(stmt, ast.AnnAssign):
            name, ann, value = stmt.target.id, stmt.annotation, stmt.value
            if (
                isinstance(ann, ast.Subscript)
                and getattr(ann.value, "id", None) == "ConstExpr"
            ):
                const_specs.append((name, value))
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr in {"get", "get_or"}
                and _ctx_port(value.func.value) is not None
            ):
                port = _ctx_port(value.func.value)
                in_ports.append(port)
                events.append(("read", port))
                continue
            events.append(("compute", name, cexpr(value)))
            continue
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == "put"
            and _ctx_port(stmt.value.func.value) is not None
        ):
            port = _ctx_port(stmt.value.func.value)
            out_ports.append(port)
            events.append(("write", port, cexpr(stmt.value.args[0])))
            continue
        raise NotImplementedError(
            f"FFT butterfly transcriber cannot translate statement {ast.dump(stmt)}"
        )

    if pid_names is None or len(pid_names) != 2:
        raise NotImplementedError("FFT butterfly must start with `s, b = ctx.rank()`")

    # Two ABIs from the same ordered events: the streaming body (spatial top -- ports are
    # hls::stream args, `.read()`/`.write()`) and the value body (folded top -- in-ports are scalar
    # value args and out-ports reference args written by assignment into the addressed lane buffers).
    stream_lines, value_lines = [], []
    for ev in events:
        if ev[0] == "read":
            stream_lines.append(f"  {elem} {ev[1]} = st_{ev[1]}.read();")
        elif ev[0] == "compute":
            stream_lines.append(f"  {elem} {ev[1]} = {ev[2]};")
            value_lines.append(f"  {elem} {ev[1]} = {ev[2]};")
        else:  # write
            stream_lines.append(f"  st_{ev[1]}.write({ev[2]});")
            value_lines.append(f"  {ev[1]} = {ev[2]};")

    helpers = {fn.__name__: fn for fn in _fft_helpers(unit_fn)}
    closure = inspect.getclosurevars(unit_fn)
    scope_consts = {
        k: v
        for k, v in {**closure.nonlocals, **closure.globals}.items()
        if isinstance(v, int) and not isinstance(v, bool)
    }

    def eval_consts(s_val, b_val):
        env = {**scope_consts, **helpers, pid_names[0]: s_val, pid_names[1]: b_val}
        return [
            float(
                eval(
                    compile(ast.Expression(e), "<tw>", "eval"),
                    {"__builtins__": {}},
                    env,
                )
            )
            for _, e in const_specs
        ]

    return {
        "in_ports": in_ports,
        "out_ports": out_ports,
        "const_names": [n for n, _ in const_specs],
        "stream_body": "\n".join(stream_lines),
        "value_body": "\n".join(value_lines),
        "eval_consts": eval_consts,
    }


def _keyform_fft_common(region):
    """Shared setup for the key-form FFT rolled tops: recognition, element type, ports, and streams.

    Returns a dict with the frontend-derived wiring/datapath the resolved IR does not carry (the
    per-``(s, b)`` slot tables, the transcribed butterfly, the boundary stream ``index`` and
    family/tensor pairs) plus the element type derived from the region operands.
    """
    # pylint: disable=import-outside-toplevel
    from .spmw import _collect
    from .spmw_datapath import _recognize_fft, _fam_tensor_pairs, _region_tensors

    collection = _collect(region)
    desc = _recognize_fft(collection)
    tensors = _region_tensors(region)
    dtypes = {dt for _, _, dt in tensors}
    if len(dtypes) != 1 or next(iter(dtypes)) not in _CPP_TYPE:
        raise NotImplementedError(
            f"key-form FFT rolled top needs uniform supported operands; got {sorted(dtypes)}"
        )
    elem = _CPP_TYPE[next(iter(dtypes))]

    stream_ins = [s for s in collection.streams if s.direction == "in"]
    stream_outs = [s for s in collection.streams if s.direction == "out"]
    if len(stream_ins) != 1 or len(stream_outs) != 1:
        raise NotImplementedError(
            "FFT rolled top expects one stream_in and one stream_out"
        )

    bfly = _transcribe_butterfly(desc["decl"].unit.interior, elem)
    sig_ports = bfly["in_ports"] + bfly["out_ports"]
    if set(sig_ports) != set(desc["ports"]):
        raise NotImplementedError(
            f"FFT butterfly ports {sorted(sig_ports)} do not match the topology ports "
            f"{sorted(desc['ports'])}"
        )
    return {
        "desc": desc,
        "elem": elem,
        "bfly": bfly,
        "sig_ports": sig_ports,
        "index_fn": stream_ins[0].extra.get("index"),
        "in_pairs": _fam_tensor_pairs(stream_ins[0]),
        "out_pairs": _fam_tensor_pairs(stream_outs[0]),
        "top_sig": ", ".join(f"{elem} {name}[N]" for name, _, _ in tensors),
    }


def _validate_key_links(key_links, ports, families):
    """Fail-closed check of the resolved-IR ``#spmw.key_link`` endpoints against the recognized ports.

    The IR records each port's ``(family, end)`` (the per-rank up/lo slot is not in the IR -- it comes
    from evaluating the topology link). Requires the parsed endpoint set to be present and **exactly**
    the recognized FFT port set (no missing / duplicate / unknown), each endpoint to agree with the
    recognition on family + direction, and each endpoint family to be a resolved channel/buffer
    family. Any deviation raises rather than silently skipping validation.
    """
    # pylint: disable=import-outside-toplevel
    from .spmw_datapath import _stage_array

    if not key_links:
        raise NotImplementedError(
            "no resolved #spmw.key_link endpoints parsed from the map: cannot validate the "
            "key-form FFT wiring (fail closed)"
        )
    endpoints = {}
    for port, family, end in key_links:
        if port in endpoints:
            raise NotImplementedError(
                f"duplicate resolved #spmw.key_link endpoint for port {port!r}"
            )
        endpoints[port] = (family, end)
    if set(endpoints) != set(ports):
        raise NotImplementedError(
            f"resolved #spmw.key_link ports {sorted(endpoints)} are not exactly the recognized "
            f"FFT ports {sorted(ports)}"
        )
    resolved_families = set(families or [])
    if not resolved_families:
        raise NotImplementedError(
            "no resolved channel/buffer families parsed for the key-form FFT (fail closed)"
        )
    for port, (family, end) in endpoints.items():
        base, offset, _role = ports[port]
        if base != _stage_array(family) or (end == "sink") != (offset == 0):
            raise NotImplementedError(
                f"resolved #spmw.key_link {port!r} -> ({family}, {end}) disagrees with the "
                f"recognition ({base}, offset={offset})"
            )
        if family not in resolved_families:
            raise NotImplementedError(
                f"resolved #spmw.key_link family {family!r} for port {port!r} is not a resolved "
                f"channel/buffer family {sorted(resolved_families)}"
            )


def _keyform_fft_top_cpp(region, ir_info):
    """The rolled O(#roles) HLS top for a key-form ``lane`` butterfly FFT region.

    ``ir_info`` carries the resolved ``spmw.map`` facts the IR is authoritative for -- the ``(S,
    HALF)`` grid, the lane families, their FIFO/buffer depths, and ``folded``. The per-``(s, b)`` slot
    wiring, the butterfly datapath, the twiddle values, and the boundary ``index`` come from the
    frontend recognition (:func:`_recognize_fft` / :func:`_transcribe_butterfly`), which the IR does
    not carry. The single butterfly datapath is transcribed once (twiddle lifted to parameters), so
    the top has ONE ``bfly`` compute body -- O(#roles)=1, not one body per butterfly. On the spatial
    map each lane ``(stage, slot)`` is a FIFO; a folded map materializes each lane stage as an
    addressed on-chip buffer (partitioned on the slot axis for conflict-free access) read/written by a
    pipelined butterfly loop. Stage 0 is loaded bit-reversed and stage S drained.
    """
    # pylint: disable=import-outside-toplevel,too-many-locals
    from .spmw_datapath import _stage_array

    setup = _keyform_fft_common(region)
    desc, elem, bfly, sig_ports = (
        setup["desc"],
        setup["elem"],
        setup["bfly"],
        setup["sig_ports"],
    )
    S, HALF, N = desc["S"], desc["HALF"], desc["N"]
    if ir_info.get("grid") not in (None, (S, HALF)):
        raise NotImplementedError(
            f"resolved IR grid {ir_info['grid']} != recognized FFT grid {(S, HALF)}"
        )
    up_table, lo_table, ports = desc["up_table"], desc["lo_table"], desc["ports"]
    # Fail-closed check of the resolved-IR key-link endpoints against the recognition.
    _validate_key_links(ir_info.get("key_links"), ports, ir_info.get("families"))
    arrays = [_stage_array(f) for f in desc["families"]]
    index_fn, in_pairs, out_pairs, top_sig = (
        setup["index_fn"],
        setup["in_pairs"],
        setup["out_pairs"],
        setup["top_sig"],
    )

    def slot(role, up, lo):
        return up if role == "upper" else lo

    header = (
        "#include <hls_stream.h>\n"
        f"#define N {N}\n"
        f"#define S {S}\n"
        f"#define HALF {HALF}\n\n"
    )

    if ir_info.get("folded"):
        # Folded: each lane stage is a small addressed on-chip buffer and the butterfly axis is a
        # pipelined loop (one folded body). The stage index is *unrolled* into compile-time array
        # names so each iteration's read (stage s) and write (stage s + 1) land on DISTINCT
        # fully-partitioned register arrays -- the runtime (up, lo) slot access is then a conflict-free
        # register mux, so the folded butterfly loop schedules at II=1. (A single [stage][slot] buffer
        # serialises because HLS cannot disambiguate a same-array read/write on a runtime slot.)
        vsig = (
            ", ".join(f"{elem} {p}" for p in bfly["in_ports"])
            + "".join(f", {elem} {c}" for c in bfly["const_names"])
            + "".join(f", {elem} &{p}" for p in bfly["out_ports"])
        )
        bfly_def = (
            f"void bfly({vsig}) {{\n#pragma HLS inline off\n{bfly['value_body']}\n}}\n"
        )
        points = [(s, b) for s in range(S) for b in range(HALF)]
        tw = [bfly["eval_consts"](s, b) for s, b in points]

        def table(name, values, ctype):
            return f"static const {ctype} {name}[{len(values)}] = {{{', '.join(values)}}};\n"

        tables = table("UP", [str(up_table[p]) for p in points], "int")
        tables += table("LO", [str(lo_table[p]) for p in points], "int")
        for ci, _ in enumerate(bfly["const_names"]):
            tables += table(
                f"TW{ci}", [f"{tw[k][ci]:.9e}f" for k in range(len(points))], elem
            )

        def stage_buf(base, stage):
            return f"{base}_{stage}"

        buf_decls = "".join(
            f"  {elem} {stage_buf(a, s)}[N];\n" for a in arrays for s in range(S + 1)
        )
        buf_part = "".join(
            f"#pragma HLS array_partition variable={stage_buf(a, s)} complete dim=0\n"
            for a in arrays
            for s in range(S + 1)
        )
        load = [
            f"  {stage_buf(_stage_array(fam), 0)}"
            f"[{index_fn(idx, S) if callable(index_fn) else idx}] = {tensor}[{idx}];"
            for fam, tensor in in_pairs
            for idx in range(N)
        ]
        # Fold-factor faithful schedule: fold[1]=F runs F logical butterflies per physical PE, so the
        # butterfly axis (extent HALF) becomes P = HALF/F physical PEs (unrolled, parallel), each
        # time-multiplexing F butterflies in a pipelined II=1 fold loop. Butterfly b = p*F + i (PE p,
        # fold-iteration i), so distinct partial/full folds emit distinct schedules. The per-stage
        # register arrays keep the P parallel (up, lo) accesses conflict-free.
        # Only butterfly-axis (dim 1) folding is implemented. Any fold factor > 1 on any axis makes
        # the map's lane families reclassify to buffers and enter this branch, so a stage-axis fold
        # (e.g. fold={0:3}) would otherwise be emitted as if the butterfly axis were unfolded -- reject
        # it fail-closed rather than mis-scheduling.
        fold_dims = ir_info.get("fold") or []
        if len(fold_dims) != 2 or fold_dims[0] != 1 or fold_dims[1] <= 1:
            raise NotImplementedError(
                "folded key-form FFT rolled top only implements butterfly-axis folding "
                f"(a rank-2 fold with fold[0] == 1 and fold[1] > 1); got fold={fold_dims or None}"
            )
        fold_f = fold_dims[1]
        if HALF % fold_f != 0:
            raise NotImplementedError(
                f"fold factor {fold_f} must divide the butterfly-axis extent {HALF}"
            )
        n_pe = HALF // fold_f

        def wire(port, stage, idx_expr):
            base, offset, role = ports[port]
            return (
                f"{stage_buf(base, stage + offset)}"
                f"[{'UP' if role == 'upper' else 'LO'}[{idx_expr}]]"
            )

        loops = ""
        for s in range(S):
            body = ""
            for pe in range(n_pe):
                idx = f"{s} * HALF + {pe} * {fold_f} + i"
                args = [wire(p, s, idx) for p in bfly["in_ports"]]
                args += [f"TW{ci}[{idx}]" for ci in range(len(bfly["const_names"]))]
                args += [wire(p, s, idx) for p in bfly["out_ports"]]
                body += f"    bfly({', '.join(args)});\n"
            loops += (
                f"  for (int i = 0; i < {fold_f}; i++) {{\n"
                "#pragma HLS pipeline II=1\n"
                f"{body}"
                "  }\n"
            )
        drain = [
            f"  {tensor}[{idx}] = {stage_buf(_stage_array(fam), S)}[{idx}];"
            for fam, tensor in out_pairs
            for idx in range(N)
        ]
        return (
            header
            + f"{tables}\n{bfly_def}\n"
            + f"void top({top_sig}) {{\n"
            + f"{buf_decls}{buf_part}"
            + "\n".join(load)
            + "\n"
            + loops
            + "\n".join(drain)
            + "\n}\n"
        )

    # Spatial: one FIFO per lane (stage, slot); the butterfly body is stamped per (s, b) with
    # compile-time constant twiddle args and the topology's slot wiring under #pragma HLS dataflow.
    depth_of = dict(zip(ir_info.get("families", []), ir_info.get("depths", [])))
    sig = ", ".join(f"hls::stream<{elem}> &st_{p}" for p in sig_ports)
    sig += "".join(f", {elem} {c}" for c in bfly["const_names"])
    bfly_def = (
        f"void bfly({sig}) {{\n#pragma HLS inline off\n{bfly['stream_body']}\n}}\n"
    )
    decls = "".join(f"  hls::stream<{elem}> {a}[S + 1][N];\n" for a in arrays)
    depths = "".join(
        f"#pragma HLS stream variable={a} depth={max(depth_of.get(fam, 2), 2)}\n"
        for fam, a in zip(desc["families"], arrays)
    )
    load = [
        f"  {_stage_array(fam)}[0][{index_fn(idx, S) if callable(index_fn) else idx}]"
        f".write({tensor}[{idx}]);"
        for fam, tensor in in_pairs
        for idx in range(N)
    ]
    calls = []
    for s in range(S):
        for b in range(HALF):
            up, lo = up_table[(s, b)], lo_table[(s, b)]
            wires = [
                f"{ports[p][0]}[{s + ports[p][1]}][{slot(ports[p][2], up, lo)}]"
                for p in sig_ports
            ]
            wires += [f"{v:.9e}f" for v in bfly["eval_consts"](s, b)]
            calls.append(f"  bfly({', '.join(wires)});")
    drain = [
        f"  {tensor}[{idx}] = {_stage_array(fam)}[S][{idx}].read();"
        for fam, tensor in out_pairs
        for idx in range(N)
    ]
    return (
        header
        + f"{bfly_def}\n"
        + f"void top({top_sig}) {{\n"
        + "#pragma HLS dataflow\n"
        + f"{decls}"
        + f"{depths}"
        + "\n".join(load)
        + "\n"
        + "\n".join(calls)
        + "\n"
        + "\n".join(drain)
        + "\n}\n"
    )


def _is_keyform_fft_region(region):
    """True if ``region`` is a recognizable key-form ``lane`` butterfly FFT (vs a systolic mesh)."""
    # pylint: disable=import-outside-toplevel,broad-except
    from .spmw import _collect
    from .spmw_datapath import _recognize_fft

    try:
        _recognize_fft(_collect(region))
        return True
    except Exception:
        return False


def _fft_rolled_testbench(region):
    """A self-checking C++ testbench that runs the rolled FFT ``top`` vs a naive DFT reference."""
    # pylint: disable=import-outside-toplevel
    from .spmw import _collect
    from .spmw_datapath import _recognize_fft, _fam_tensor_pairs, _region_tensors

    collection = _collect(region)
    N = _recognize_fft(collection)["N"]
    stream_ins = [s for s in collection.streams if s.direction == "in"]
    stream_outs = [s for s in collection.streams if s.direction == "out"]
    xr, xi = (t for _, t in _fam_tensor_pairs(stream_ins[0]))
    yr, yi = (t for _, t in _fam_tensor_pairs(stream_outs[0]))
    tensors = _region_tensors(region)
    names = [name for name, _, _ in tensors]
    # match the kernel's operand element type (derived from the region operands, not hardcoded float)
    elem = _CPP_TYPE[tensors[0][2]]
    top_sig = ", ".join(f"{elem} {n}[N]" for n in names)
    return (
        "#include <cmath>\n#include <cstdio>\n"
        f"#define N {N}\n"
        f"void top({top_sig});\n"
        "int main(){\n"
        f"  {elem} {', '.join(f'{n}[N]' for n in names)};\n"
        f"  for(int i=0;i<N;i++){{ {xr}[i]=(float)((i*7+3)%N)/N; {xi}[i]=0.0f; }}\n"
        f"  top({', '.join(names)});\n"
        "  int bad=0;\n"
        "  for(int k=0;k<N;k++){\n"
        "    float rr=0,ii=0;\n"
        "    for(int n=0;n<N;n++){\n"
        "      float ang=-2.0f*3.14159265358979f*k*n/N;\n"
        f"      rr+={xr}[n]*cosf(ang)-{xi}[n]*sinf(ang);\n"
        f"      ii+={xr}[n]*sinf(ang)+{xi}[n]*cosf(ang);\n"
        "    }\n"
        f"    if(std::fabs(rr-{yr}[k])>1e-2||std::fabs(ii-{yi}[k])>1e-2) bad=1;\n"
        "  }\n"
        '  printf(bad?"CSIM MISMATCH\\n":"CSIM MATCH\\n");\n'
        "  return bad;\n}\n"
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
            tb = (
                _fft_rolled_testbench(region)
                if _is_keyform_fft_region(region)
                else _rolled_testbench(region)
            )
            with open(os.path.join(project, "tb.cpp"), "w", encoding="utf-8") as handle:
                handle.write(tb)
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


def _memory_placements_from_ir(ir, ordered_operands):
    """Map each top-level ``#spmw.memory`` placement onto its rolled-top C++ array (``A``/``B``/``C``).

    The rolled top names its arrays A, B, C by tensor-operand order, while a placement names its
    target by the region operand's name; ``ordered_operands`` (the region's shaped operands, in
    order) bridges the two. A resolved resource of ``AUTO``/``SRL`` carries no ``bind_storage``.
    """
    cpp_vars = ("A", "B", "C")
    # Double-buffered (ping-pong) operands ride on a separate `spmw.double_buffers` string list (a
    # builtin ArrayAttr) rather than on the typed `#spmw.memory` attr, so no dialect change is needed.
    double_match = re.search(r"spmw\.double_buffers = \[([^\]]*)\]", ir)
    double_tensors = (
        set(re.findall(r'"([^"]+)"', double_match.group(1))) if double_match else set()
    )
    placements = {}
    for tensor, resource, bank_axis in re.findall(
        r'#spmw\.memory<tensor = "([^"]+)", resource = "([^"]+)", bank_axis = (-?\d+)>',
        ir,
    ):
        if tensor not in ordered_operands:
            continue
        idx = ordered_operands.index(tensor)
        if idx >= len(cpp_vars):
            continue
        resource = resource.upper()
        impl = resource if resource in _CONCRETE_STORAGE_IMPL else None
        placements[cpp_vars[idx]] = {
            "impl": impl,
            "bank_axis": int(bank_axis),
            "double": tensor in double_tensors,
        }
    return placements


def _bank_functions_from_ir(ir):
    """Parse the top func's ``spmw.bank_functions`` -> ``{tensor: {banks, stride, mode, axis}}``."""
    match = re.search(r"spmw\.bank_functions = \[([^\]]*)\]", ir, re.DOTALL)
    if not match:
        return {}
    functions = {}
    for entry in re.findall(r'"([^"]+)"', match.group(1)):
        tensor, banks, stride, mode, axis = entry.split(":")
        functions[tensor] = {
            "banks": int(banks),
            "stride": int(stride),
            "mode": mode,
            "axis": int(axis),
        }
    return functions


def _banked_c_storage(bank_functions, rows, elem):
    """Realize F2 banking of the output operand ``C`` as real 2D ``[banks][depth]`` storage.

    Returns ``(c_ref, decl, writeback)``: the swizzled write target the PE dispatch writes to
    (``&C_bank[bank(i)][offset(i)][j]``), the local banked-array declaration, and the writeback loop
    copying the banked buffer back to the host ``C[M][N]`` (so ``mod(A,B,C)`` is unchanged). The
    rolled systolic emitter realizes banking of the output ``C`` along the row axis; a banked ``A``/
    ``B`` or a non-row axis is rejected rather than silently ignored.
    """
    if not bank_functions:
        return "&C[i][j]", "", ""
    for tensor, spec in bank_functions.items():
        if tensor != "C" or spec["axis"] != 0:
            raise NotImplementedError(
                f"the rolled emitter realizes F2 banking of the output C along the row axis; "
                f"got a banked {tensor} on axis {spec['axis']}"
            )
    # pylint: disable=import-outside-toplevel
    from .transform.f2_layout import F2LayoutSolver

    spec = bank_functions["C"]
    n_bits = rows.bit_length() - 1
    bank_bits = spec["banks"].bit_length() - 1
    strides = [spec["stride"]] if spec["mode"] == "xor" else []
    helper = F2LayoutSolver(n_bits, bank_bits).solve(strides)
    num_banks, depth = helper.dims()
    bank_i, off_i = helper.bank_expr("i"), helper.offset_expr("i")
    c_ref = f"&C_bank[{bank_i}][{off_i}][j]"
    decl = (
        f"  {elem} C_bank[{num_banks}][{depth}][N];\n"
        f"#pragma HLS array_partition variable=C_bank complete dim=1\n"
    )
    writeback = (
        "  for (int i = 0; i < M; i++) {\n#pragma HLS unroll\n"
        "    for (int j = 0; j < N; j++) {\n#pragma HLS unroll\n"
        f"      C[i][j] = C_bank[{bank_i}][{off_i}][j];\n"
        "    }\n  }\n"
    )
    return c_ref, decl, writeback


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

    # The channel families the resolver produced are FIFO arrays the rolled top declares; a family
    # reclassified to a buffer under fold (spmw.buffer_families) is materialized as a real F2-banked
    # on-chip buffer instead. The systolic emitter wires A along east/west and B along north/south.
    def _families(attr):
        match = re.search(re.escape(attr) + r" = \[([^\]]*)\]", ir)
        return re.findall(r'"([^"]+)"', match.group(1)) if match else []

    stream_families = _families("spmw.channel_families")
    buffer_families = _families("spmw.buffer_families")
    # A key-form `lane` topology (the radix-2 FFT butterfly permutation network) is a different
    # rolled top than the systolic mesh: one transcribed butterfly body instantiated across the
    # (stage, butterfly) grid with per-(s,b) constant twiddles and the topology's slot wiring. The
    # systolic east/west + north/south shapes (including their folded variants) keep the path below.
    # The resolved IR is authoritative for grid / families / depths / fold; the frontend recognition
    # supplies only what the IR does not carry (the datapath, per-rank slot wiring, and twiddle).
    if not (set(stream_families) | set(buffer_families)) <= {
        "east/west",
        "north/south",
    }:
        folded = bool(buffer_families)
        grid_m = re.search(r"grid = \[(\d+), (\d+)\]", ir)
        depth_attr = (
            "spmw.buffer_family_depths" if folded else "spmw.channel_family_depths"
        )
        depth_m = re.search(re.escape(depth_attr) + r" = array<i64: ([^>]*)>", ir)
        fold_m = re.search(r"fold = array<i64: ([^>]*)>", ir)
        # The resolved map records each port's channel endpoint: port -> (key/family, end). The
        # per-rank (s, b) slot is not in the IR (it comes from evaluating the topology link), so the
        # emitter validates the frontend recognition against these IR endpoints rather than deriving
        # the slots from them.
        key_links = re.findall(
            r'#spmw\.key_link<port = "([^"]+)", key = "([^"]+)", end = "([^"]+)"', ir
        )
        ir_info = {
            "grid": (int(grid_m.group(1)), int(grid_m.group(2))) if grid_m else None,
            "folded": folded,
            "families": buffer_families if folded else stream_families,
            "depths": (
                [int(x) for x in depth_m.group(1).split(",") if x.strip()]
                if depth_m
                else []
            ),
            "fold": (
                [int(x) for x in fold_m.group(1).split(",") if x.strip()]
                if fold_m
                else None
            ),
            "key_links": key_links,
        }
        return _keyform_fft_top_cpp(region, ir_info)
    if buffer_families:
        # A folded systolic mesh: the A-forwarding (east/west) is reclassified to a banked on-chip
        # buffer; B (north/south) stays a FIFO. Only this shape is realized so far.
        if sorted(buffer_families) != ["east/west"] or sorted(stream_families) != [
            "north/south"
        ]:
            raise NotImplementedError(
                f"folded rolled emitter realizes an east/west buffer + north/south stream; got "
                f"buffer families {buffer_families}, stream families {stream_families}"
            )
    elif sorted(stream_families) != ["east/west", "north/south"]:
        raise NotImplementedError(
            f"IR-driven rolled emitter handles the systolic east/west + north/south families; "
            f"got {stream_families}"
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
    # F2-banked output: the PE writes to a swizzled slot of a real 2D [banks][depth] C_bank, which
    # is written back to C at the end (host interface unchanged).
    c_ref, bank_decl, bank_writeback = _banked_c_storage(
        _bank_functions_from_ir(ir), rows, elem
    )
    pe_dispatch = _pe_dispatch(roles, wired, c_ref)
    # Honor the logical-memory placements the region pinned on top-level operands (resolved
    # resource/bank axis -> bind_storage / partition pragmas); BOTH the spatial and folded tops
    # thread these, so a folded map does not silently drop a placed operand's resource pragmas.
    annotations = getattr(region.fn, "__annotations__", {})
    ordered_operands = [n for n, t in annotations.items() if getattr(t, "shape", None)]
    memory_placements = _memory_placements_from_ir(ir, ordered_operands)
    # A `double=True` placement requests real ping-pong double buffering: emit a two-epoch K-tiled
    # GEMM with two alternating on-chip copies of the operand's tile (preload-next while consuming
    # current), not the streaming systolic top -- fail closed on a shape the schedule cannot cover.
    double_ops = sorted(
        v for v, place in memory_placements.items() if place.get("double")
    )
    if double_ops:
        if double_ops != ["B"]:
            raise NotImplementedError(
                f"ping-pong double buffering is implemented for the B operand only; got {double_ops}"
            )
        if buffer_families or depth % 2 != 0:
            raise NotImplementedError(
                f"ping-pong double buffering needs a non-folded systolic GEMM whose contraction "
                f"K={depth} splits into two epochs (folded={bool(buffer_families)})"
            )
        return _pingpong_top_cpp(rows, cols, depth, elem, memory_placements["B"])
    if buffer_families:
        # Folded map: A (east/west) is reclassified to a banked on-chip buffer; B (north/south) stays
        # a FIFO. A is banked on the row axis with an F2 bit-swizzle, so the row extent and the fold
        # factor must both be powers of two.
        if rows & (rows - 1) != 0:
            raise NotImplementedError(
                f"folded rolled emitter banks A on the row axis with an F2 bit-swizzle, which needs "
                f"a power-of-two row extent; got M={rows}"
            )
        fold = re.search(r"fold = array<i64: ([^>]*)>", ir).group(1).split(",")
        a_banks = int(fold[1])
        if a_banks & (a_banks - 1) != 0:
            raise NotImplementedError(
                f"folded rolled emitter needs a power-of-two fold factor for banking; got {a_banks}"
            )
        b_depth = int(
            re.search(r"spmw\.channel_family_depths = array<i64: ([^>]*)>", ir)
            .group(1)
            .split(",")[0]
        )
        return _folded_top_cpp(
            rows,
            cols,
            depth,
            elem,
            roles,
            c_ref,
            bank_decl,
            bank_writeback,
            max(b_depth, depth),
            a_banks,
            memory_placements,
        )
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
        rows,
        cols,
        depth,
        elem,
        pe_defs,
        pe_dispatch,
        fa_depth,
        fb_depth,
        memory_placements,
        bank_decl,
        bank_writeback,
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
    """Translate an ``affine_map<(d0,...) -> (expr)>`` indicator to a C++ ``expr != 0`` condition.

    Only affine indicators whose ``mod``/``floordiv`` operate on non-negative subexpressions are
    accepted: C++ ``%`` and ``/`` truncate toward zero, so they diverge from MLIR's floor-based
    ``mod``/``floordiv`` when an operand can be negative -- which would make the emitted dispatch
    select a different variant than ``spmw-role-partition``/``spmw-unroll`` (which evaluate with
    floor semantics). A predicate mixing ``mod``/``floordiv`` with subtraction/negation is rejected
    (fail closed) rather than silently mistranslated. ``ceildiv`` has no C++ operator and is rejected.
    """
    result = affine_map_token.split("-> ", 1)[1]  # e.g. "((d0 + d1) mod 2)>"
    expr = result.rsplit(">", 1)[0]  # strip the closing map delimiter
    if "ceildiv" in expr:
        raise NotImplementedError(
            "ceildiv predicates are not supported by the rolled emitter"
        )
    if ("mod" in expr or "floordiv" in expr) and "-" in expr:
        raise NotImplementedError(
            "the rolled emitter only supports mod/floordiv predicates over non-negative "
            f"subexpressions (C++ % and / diverge from MLIR floor semantics on negatives); got {expr}"
        )
    expr = re.sub(r"\bd(\d+)\b", lambda m: _COORD_C[int(m.group(1))], expr)
    expr = expr.replace(" mod ", " % ").replace(" floordiv ", " / ")
    return f"{expr} != 0"


def _pe_dispatch(roles, wired, c_ref="&C[i][j]"):
    """The per-grid-point PE statement: one call, or an if/else chain over the variant predicates.

    Because the instantiation loops are fully unrolled, ``i``/``j`` are compile-time constants, so
    the coordinate predicate folds and each grid point is bound to exactly one variant body.
    ``c_ref`` is where the PE writes its output element -- ``&C[i][j]`` by default, or the swizzled
    ``&C_bank[bank(i)][offset(i)][j]`` when the output operand is F2-banked.
    """
    base = [role for role in roles if role["predicate"] is None]
    variants = [role for role in roles if role["predicate"] is not None]

    def call(name):
        return f"{name}({wired}, {c_ref});"

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
