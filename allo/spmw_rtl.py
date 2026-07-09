# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=cyclic-import

"""Structural RTL emission for SPMW: walk the rolled ``spmw.map`` and generate a structural top.

This is the RTL counterpart of the rolled HLS emitter (``spmw_hls.emit_rolled_hls_ir``). Instead of a
C++ dataflow top it emits a structural SystemVerilog ``spmw_top`` that instantiates *one role module
per role* inside ``generate`` loops over the grid, wired by *one FIFO per channel* -- so the module
text is O(#roles), constant in the grid size, while the instance count is O(P0*P1). The synthesis-
time-win regularity that the HLS path shows as a constant body count shows here as a constant number
of instantiated module *types* and ``generate``-loop nests.

The role bodies (``pe_interior``/``load_a``/``load_b``/``drain``) are the free-running ``ap_ctrl_none``
IPs the role-IP export produces; here they are declared as black boxes with the ap_fifo stream ABI
(each ``hls::stream<T>&`` port becomes ``<p>_dout/_empty_n/_read`` for an input or
``<p>_din/_full_n/_write`` for an output), so the exported IP drops straight in. A generic
synchronous ``spmw_fifo`` is emitted inline.
"""

import re
from collections import namedtuple

# The ap_fifo signal width per Allo element type (bits).
_DW = {
    "f32": 32,
    "f64": 64,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
}

# One PE local port's structural wiring: the FIFO family, the row/col index expressions of the FIFO
# array element it connects to, and whether it takes that FIFO's read or write side.
_Wire = namedtuple("_Wire", ["fam", "row", "col", "side"])

# The systolic wiring: A streams west->east along a row (fa), B streams north->south down a column
# (fb). ``read`` ports take a FIFO's read side, ``write`` ports its write side.
_RTL_WIRING = {
    "west": _Wire("fa", "i", "j", "read"),
    "east": _Wire("fa", "i", "j + 1", "write"),
    "north": _Wire("fb", "i", "j", "read"),
    "south": _Wire("fb", "i + 1", "j", "write"),
}


def emit_structural_verilog(region):
    """Emit a structural SystemVerilog ``spmw_top`` for the systolic ``region`` from its ``spmw.map``.

    Lowers to the rolled ``spmw.map``, runs ``spmw-role-partition`` + ``spmw-resolve-channels``, and
    reads the grid extents, channel families, and compute-role ports off the IR -- so the structure
    is derived from the rolled IR, not the frontend collection (as with the HLS emitter).
    """
    # pylint: disable=import-outside-toplevel
    from .spmw import lower, _run_module_pass
    from .spmw_hls import _ROLLED_WIRING

    module = lower(region)
    _run_module_pass(module, "spmw-role-partition")
    _run_module_pass(module, "spmw-resolve-channels")
    ir = str(module)

    families = re.findall(
        r'"([^"]+)"',
        re.search(r"spmw\.channel_families = \[([^\]]*)\]", ir).group(1),
    )
    if sorted(families) != ["east/west", "north/south"]:
        raise NotImplementedError(
            f"structural RTL emitter handles the systolic east/west + north/south families; "
            f"got {families}"
        )
    partition = [
        int(x)
        for x in re.search(r"spmw\.partition = array<i64: ([^>]*)>", ir)
        .group(1)
        .split(",")
    ]
    grid = re.search(r"grid = \[(\d+), (\d+)\]", ir)
    rows, cols = int(grid.group(1)), int(grid.group(2))
    if len(partition) != 1 or partition[0] != rows * cols:
        raise NotImplementedError(
            f"structural RTL emitter handles a single-compute-role systolic map; "
            f"got partition {partition} for a {rows}x{cols} grid"
        )
    a_shape = re.search(r"memref<(\d+)x(\d+)x(\w+)>", ir)
    depth = int(a_shape.group(2))
    dw = _DW[a_shape.group(3)]

    role_attr = re.search(
        r"#spmw\.role<unit = @\w+_interior, missing = \[[^\]]*\], ports = \[([^\]]*)\]>",
        ir,
    )
    ports = re.findall(r'"([^"]+)"', role_attr.group(1))
    if set(ports) != set(_ROLLED_WIRING):
        raise NotImplementedError(
            f"structural RTL emitter handles the systolic port set {sorted(_ROLLED_WIRING)}; "
            f"got {ports}"
        )
    depths = [
        int(x)
        for x in re.search(r"spmw\.channel_family_depths = array<i64: ([^>]*)>", ir)
        .group(1)
        .split(",")
    ]
    fa_depth, fb_depth = max(depths[0], depth), max(depths[1], depth)
    return _structural_top_sv(rows, cols, depth, dw, fa_depth, fb_depth, ports)


def _stream_port_decls(port, direction):
    """The ap_fifo module-port declarations for one stream ``port`` (``in`` reads, ``out`` writes)."""
    if direction == "in":
        return [
            f"  input  wire [DW-1:0] {port}_dout",
            f"  input  wire          {port}_empty_n",
            f"  output wire          {port}_read",
        ]
    return [
        f"  output wire [DW-1:0] {port}_din",
        f"  input  wire          {port}_full_n",
        f"  output wire          {port}_write",
    ]


def _fifo_module():
    """A generic synchronous ap_fifo FIFO (write: din/full_n/write; read: dout/empty_n/read)."""
    return (
        "module spmw_fifo #(parameter DW = 32, parameter DEPTH = 2) (\n"
        "  input  wire          clk,\n"
        "  input  wire          rst_n,\n"
        "  input  wire [DW-1:0] din,\n"
        "  output wire          full_n,\n"
        "  input  wire          write,\n"
        "  output wire [DW-1:0] dout,\n"
        "  output wire          empty_n,\n"
        "  input  wire          read\n"
        ");\n"
        "  localparam AW = (DEPTH <= 1) ? 1 : $clog2(DEPTH);\n"
        "  reg [DW-1:0] mem [0:DEPTH-1];\n"
        "  reg [AW:0]   count;\n"
        "  reg [AW-1:0] rptr, wptr;\n"
        "  assign full_n  = (count != DEPTH);\n"
        "  assign empty_n = (count != 0);\n"
        "  assign dout    = mem[rptr];\n"
        "  wire do_wr = write & full_n;\n"
        "  wire do_rd = read  & empty_n;\n"
        "  always @(posedge clk) begin\n"
        "    if (!rst_n) begin\n"
        "      count <= 0; rptr <= 0; wptr <= 0;\n"
        "    end else begin\n"
        "      if (do_wr) begin mem[wptr] <= din; wptr <= (wptr == DEPTH-1) ? 0 : wptr + 1; end\n"
        "      if (do_rd) rptr <= (rptr == DEPTH-1) ? 0 : rptr + 1;\n"
        "      count <= count + (do_wr ? 1 : 0) - (do_rd ? 1 : 0);\n"
        "    end\n"
        "  end\n"
        "endmodule\n"
    )


def _role_blackbox(name, ports, kind):
    """A black-box declaration for a role IP (filled by the ap_ctrl_none role-IP export).

    ``kind`` selects the non-stream ABI: ``"pe"`` takes ``A``/``B`` row/col operands and a ``c_out``;
    ``"load_a"``/``"load_b"`` take an operand row/column; ``"drain"`` takes nothing extra.
    """
    lines = ["  input  wire          clk", "  input  wire          rst_n"]
    if kind == "pe":
        for port in ports:
            lines += _stream_port_decls(
                port, "in" if port in {"west", "north"} else "out"
            )
        lines += ["  output wire [DW-1:0] c_out", "  output wire          c_out_valid"]
    elif kind == "load_a":
        lines += ["  input  wire [DW-1:0] a_row [0:K-1]"] + _stream_port_decls(
            "out", "out"
        )
    elif kind == "load_b":
        lines += ["  input  wire [DW-1:0] b_col [0:K-1]"] + _stream_port_decls(
            "out", "out"
        )
    else:  # drain
        lines += _stream_port_decls("in", "in")
    body = ",\n".join(lines)
    return f"module {name} #(parameter DW = 32, parameter K = 4) (\n{body}\n);\n// role IP body: ap_ctrl_none export\nendmodule\n"


def _structural_top_sv(rows, cols, k, dw, fa_depth, fb_depth, ports):
    """The structural ``spmw_top``: one role instance per grid point + one FIFO per channel."""
    del ports  # the systolic wiring is fixed (validated by the caller)

    def conn(port, wire):
        read = wire.side == "read"
        data = "dout" if read else "din"
        ok = "empty_n" if read else "full_n"
        req = "rd" if read else "wr"
        elem = f"[{wire.row}][{wire.col}]"
        return (
            f"      .{port}_{data}({wire.fam}_{data}{elem}),\n"
            f"      .{port}_{ok}({wire.fam}_{'empty' if read else 'full'}{elem}),\n"
            f"      .{port}_{'read' if read else 'write'}({wire.fam}_{req}{elem})"
        )

    pe_conn = ",\n".join(conn(port, wire) for port, wire in _RTL_WIRING.items())
    return (
        _fifo_module()
        + "\n"
        + _role_blackbox("pe_interior", ["west", "east", "north", "south"], "pe")
        + "\n"
        + _role_blackbox("load_a", [], "load_a")
        + "\n"
        + _role_blackbox("load_b", [], "load_b")
        + "\n"
        + _role_blackbox("drain", [], "drain")
        + "\n"
        f"module spmw_top #(parameter M = {rows}, parameter N = {cols}, parameter K = {k},\n"
        f"                  parameter DW = {dw}, parameter DA = {fa_depth}, parameter DB = {fb_depth}) (\n"
        "  input  wire          clk,\n"
        "  input  wire          rst_n,\n"
        "  input  wire [DW-1:0] A [0:M-1][0:K-1],\n"
        "  input  wire [DW-1:0] B [0:K-1][0:N-1],\n"
        "  output wire [DW-1:0] C [0:M-1][0:N-1]\n"
        ");\n"
        # A family (east/west): (N+1) FIFOs per row; B family (north/south): (M+1) per column.
        "  wire [DW-1:0] fa_din  [0:M-1][0:N];   wire [DW-1:0] fa_dout [0:M-1][0:N];\n"
        "  wire          fa_full [0:M-1][0:N];   wire          fa_empty[0:M-1][0:N];\n"
        "  wire          fa_wr   [0:M-1][0:N];   wire          fa_rd   [0:M-1][0:N];\n"
        "  wire [DW-1:0] fb_din  [0:M][0:N-1];   wire [DW-1:0] fb_dout [0:M][0:N-1];\n"
        "  wire          fb_full [0:M][0:N-1];   wire          fb_empty[0:M][0:N-1];\n"
        "  wire          fb_wr   [0:M][0:N-1];   wire          fb_rd   [0:M][0:N-1];\n"
        "  genvar i, j;\n"
        "  generate\n"
        # one FIFO per A-family channel
        "    for (i = 0; i < M; i = i + 1) begin : fa_row\n"
        "      for (j = 0; j < N + 1; j = j + 1) begin : fa_col\n"
        "        spmw_fifo #(.DW(DW), .DEPTH(DA)) u_fa (.clk(clk), .rst_n(rst_n),\n"
        "          .din(fa_din[i][j]), .full_n(fa_full[i][j]), .write(fa_wr[i][j]),\n"
        "          .dout(fa_dout[i][j]), .empty_n(fa_empty[i][j]), .read(fa_rd[i][j]));\n"
        "      end\n"
        "    end\n"
        # one FIFO per B-family channel
        "    for (i = 0; i < M + 1; i = i + 1) begin : fb_row\n"
        "      for (j = 0; j < N; j = j + 1) begin : fb_col\n"
        "        spmw_fifo #(.DW(DW), .DEPTH(DB)) u_fb (.clk(clk), .rst_n(rst_n),\n"
        "          .din(fb_din[i][j]), .full_n(fb_full[i][j]), .write(fb_wr[i][j]),\n"
        "          .dout(fb_dout[i][j]), .empty_n(fb_empty[i][j]), .read(fb_rd[i][j]));\n"
        "      end\n"
        "    end\n"
        # one loader per row/column at the array edge
        "    for (i = 0; i < M; i = i + 1) begin : load_a_row\n"
        "      load_a #(.DW(DW), .K(K)) u_la (.clk(clk), .rst_n(rst_n), .a_row(A[i]),\n"
        "        .out_din(fa_din[i][0]), .out_full_n(fa_full[i][0]), .out_write(fa_wr[i][0]));\n"
        "    end\n"
        "    for (j = 0; j < N; j = j + 1) begin : load_b_col\n"
        "      load_b #(.DW(DW), .K(K)) u_lb (.clk(clk), .rst_n(rst_n), .b_col(B[j]),\n"
        "        .out_din(fb_din[0][j]), .out_full_n(fb_full[0][j]), .out_write(fb_wr[0][j]));\n"
        "    end\n"
        # one PE per grid point -- the single compute-role module, instantiated over the whole mesh
        "    for (i = 0; i < M; i = i + 1) begin : pe_row\n"
        "      for (j = 0; j < N; j = j + 1) begin : pe_col\n"
        "        pe_interior #(.DW(DW), .K(K)) u_pe (.clk(clk), .rst_n(rst_n),\n"
        f"{pe_conn},\n"
        "          .c_out(C[i][j]), .c_out_valid());\n"
        "      end\n"
        "    end\n"
        # one drain per row/column at the far edge
        "    for (i = 0; i < M; i = i + 1) begin : drain_a_row\n"
        "      drain #(.DW(DW)) u_da (.clk(clk), .rst_n(rst_n),\n"
        "        .in_dout(fa_dout[i][N]), .in_empty_n(fa_empty[i][N]), .in_read(fa_rd[i][N]));\n"
        "    end\n"
        "    for (j = 0; j < N; j = j + 1) begin : drain_b_col\n"
        "      drain #(.DW(DW)) u_db (.clk(clk), .rst_n(rst_n),\n"
        "        .in_dout(fb_dout[M][j]), .in_empty_n(fb_empty[M][j]), .in_read(fb_rd[M][j]));\n"
        "    end\n"
        "  endgenerate\n"
        "endmodule\n"
    )
