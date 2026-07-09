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


def emit_structural_verilog(region, mode="blackbox"):
    """Emit a structural SystemVerilog ``spmw_top`` for the systolic ``region`` from its ``spmw.map``.

    Lowers to the rolled ``spmw.map``, runs ``spmw-role-partition`` + ``spmw-resolve-channels``, and
    reads the grid extents, channel families, and compute-role ports off the IR -- so the structure
    is derived from the rolled IR, not the frontend collection (as with the HLS emitter).

    ``mode`` selects the role bodies (the wiring is identical): ``"blackbox"`` (default) emits ap_fifo
    black boxes -- the integration template; ``"behavioral"`` emits an FP MAC PE + synthesizable
    loaders/drains/collect so the top is a self-contained cosim; ``"synth"`` emits the synthesizable
    loaders/drains/collect and leaves the PE to the exported ap_ctrl_none IP (the ``vitis_rtl`` path).
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
    if a_shape is None:
        raise NotImplementedError(
            "structural RTL emitter handles the systolic A[M,K] @ B[K,N] -> C[M,N] shape; "
            "the region has no 2-D tensor operands"
        )
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
    del ports  # the systolic wiring is fixed (validated above)
    return _structural_top_sv(rows, cols, depth, dw, fa_depth, fb_depth, mode)


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


def _pe_port_decls():
    """The PE module port list, identical to the exported ap_ctrl_none IP (ap_clk/ap_rst + ap_fifo).

    The result leaves on a ``c_out`` ap_fifo output stream, so this is exactly the ABI Vitis emits for
    the role IP -- the exported IP drops into this slot with no adapter beyond the top's clk/rst map.
    """
    lines = ["  input  wire          ap_clk", "  input  wire          ap_rst"]
    for port in ("west", "east", "north", "south"):
        lines += _stream_port_decls(port, "in" if port in {"west", "north"} else "out")
    lines += _stream_port_decls("c_out", "out")
    return lines


def _role_blackbox(name, kind):
    """A black-box declaration for a role module (filled by the ap_ctrl_none IP or a synth body).

    ``kind`` selects the ABI: ``"pe"`` matches the exported IP (ap_clk/ap_rst + ap_fifo streams incl.
    ``c_out``); ``"load_a"``/``"load_b"`` take an operand row/column; ``"drain"`` consumes a stream;
    ``"collect"`` reads a result stream and latches the ``C`` element.
    """
    if kind == "pe":
        lines = _pe_port_decls()
    elif kind == "load_a":
        lines = ["  input  wire          clk", "  input  wire          rst_n"]
        lines += ["  input  wire [DW-1:0] a_row [0:K-1]"] + _stream_port_decls(
            "out", "out"
        )
    elif kind == "load_b":
        lines = ["  input  wire          clk", "  input  wire          rst_n"]
        lines += ["  input  wire [DW-1:0] b_col [0:K-1]"] + _stream_port_decls(
            "out", "out"
        )
    elif kind == "collect":
        lines = ["  input  wire          clk", "  input  wire          rst_n"]
        lines += _stream_port_decls("in", "in") + ["  output wire [DW-1:0] c_val"]
    else:  # drain
        lines = ["  input  wire          clk", "  input  wire          rst_n"]
        lines += _stream_port_decls("in", "in")
    body = ",\n".join(lines)
    return f"module {name} #(parameter DW = 32, parameter K = 4) (\n{body}\n);\n// role IP body: ap_ctrl_none export / synth\nendmodule\n"


def _synth_support_modules():
    """Synthesizable loader/drain/collect bodies (plain RTL, no ``shortreal``).

    ``load_a``/``load_b`` stream a ``K``-vector into an ap_fifo; ``drain`` consumes an edge stream;
    ``collect`` latches a PE's ``c_out`` result FIFO into a ``C`` element. These are real RTL and are
    reused by both the cosim top and the synthesizable ``vitis_rtl`` hierarchy.
    """
    loader = (
        "module {name} #(parameter DW = 32, parameter K = 4) (\n"
        "  input  wire          clk,\n"
        "  input  wire          rst_n,\n"
        "  input  wire [DW-1:0] {vec} [0:K-1],\n"
        "  output wire [DW-1:0] out_din,\n"
        "  input  wire          out_full_n,\n"
        "  output wire          out_write\n"
        ");\n"
        "  reg [31:0] k; reg active;\n"
        "  assign out_write = active && out_full_n;\n"
        "  assign out_din   = {vec}[k];\n"
        "  always @(posedge clk) begin\n"
        "    if (!rst_n) begin k <= 0; active <= 1; end\n"
        "    else if (out_write) begin if (k == K - 1) active <= 0; else k <= k + 1; end\n"
        "  end\n"
        "endmodule\n"
    )
    return (
        loader.format(name="load_a", vec="a_row")
        + "\n"
        + loader.format(name="load_b", vec="b_col")
        + "\n"
        "module drain #(parameter DW = 32, parameter K = 4) (\n"
        "  input  wire          clk,\n"
        "  input  wire          rst_n,\n"
        "  input  wire [DW-1:0] in_dout,\n"
        "  input  wire          in_empty_n,\n"
        "  output wire          in_read\n"
        ");\n"
        "  assign in_read = in_empty_n;\n"
        "endmodule\n"
        "\n"
        "module collect #(parameter DW = 32, parameter K = 4) (\n"
        "  input  wire          clk,\n"
        "  input  wire          rst_n,\n"
        "  input  wire [DW-1:0] in_dout,\n"
        "  input  wire          in_empty_n,\n"
        "  output wire          in_read,\n"
        "  output wire [DW-1:0] c_val\n"
        ");\n"
        "  reg [DW-1:0] val;\n"
        "  assign in_read = in_empty_n;\n"
        "  assign c_val   = val;\n"
        "  always @(posedge clk) begin\n"
        "    if (!rst_n) val <= 0;\n"
        "    else if (in_empty_n) val <= in_dout;\n"
        "  end\n"
        "endmodule\n"
    )


def _behavioral_pe():
    """A simulation FP-MAC PE with the exported-IP ABI (ap_clk/ap_rst + ap_fifo, ``c_out`` stream).

    Accumulates in ``shortreal`` in the numpy k-order and writes the result on the ``c_out`` stream,
    so the self-contained top computes ``A @ B``. Simulation only (``shortreal`` is not synthesizable);
    the synthesizable path uses the exported ap_ctrl_none IP in this same slot.
    """
    return (
        "module pe_interior #(parameter DW = 32, parameter K = 4) (\n"
        "  input  wire          ap_clk,\n"
        "  input  wire          ap_rst,\n"
        "  input  wire [DW-1:0] west_dout,\n"
        "  input  wire          west_empty_n,\n"
        "  output wire          west_read,\n"
        "  output wire [DW-1:0] east_din,\n"
        "  input  wire          east_full_n,\n"
        "  output wire          east_write,\n"
        "  input  wire [DW-1:0] north_dout,\n"
        "  input  wire          north_empty_n,\n"
        "  output wire          north_read,\n"
        "  output wire [DW-1:0] south_din,\n"
        "  input  wire          south_full_n,\n"
        "  output wire          south_write,\n"
        "  output wire [DW-1:0] c_out_din,\n"
        "  input  wire          c_out_full_n,\n"
        "  output wire          c_out_write\n"
        ");\n"
        "  localparam READ = 2'd0, WRITE = 2'd1, EMIT = 2'd2, DONE = 2'd3;\n"
        "  reg [1:0]    state;\n"
        "  reg [31:0]   k;\n"
        "  shortreal    acc;\n"
        "  reg [DW-1:0] a_reg, b_reg;\n"
        "  assign west_read    = (state == READ) && west_empty_n && north_empty_n;\n"
        "  assign north_read   = west_read;\n"
        "  assign east_write   = (state == WRITE) && east_full_n && south_full_n;\n"
        "  assign south_write  = east_write;\n"
        "  assign east_din     = a_reg;\n"
        "  assign south_din    = b_reg;\n"
        "  assign c_out_write  = (state == EMIT) && c_out_full_n;\n"
        "  assign c_out_din    = $shortrealtobits(acc);\n"
        "  always @(posedge ap_clk) begin\n"
        "    if (ap_rst) begin state <= READ; k <= 0; acc <= 0.0; end\n"
        "    else case (state)\n"
        "      READ: if (west_empty_n && north_empty_n) begin\n"
        "              a_reg <= west_dout; b_reg <= north_dout;\n"
        "              acc <= acc + $bitstoshortreal(west_dout) * $bitstoshortreal(north_dout);\n"
        "              state <= WRITE;\n"
        "            end\n"
        "      WRITE: if (east_full_n && south_full_n) begin\n"
        "               if (k == K - 1) state <= EMIT;\n"
        "               else begin k <= k + 1; state <= READ; end\n"
        "             end\n"
        "      EMIT: if (c_out_full_n) state <= DONE;\n"
        "      DONE: ;\n"
        "    endcase\n"
        "  end\n"
        "endmodule\n"
    )


def _role_modules(mode):
    """The five role module definitions for the requested ``mode``.

    ``"blackbox"`` -> empty declarations (the ``build(target="rtl")`` template); ``"behavioral"`` ->
    the sim FP-MAC PE + synthesizable loaders/drains/collect (self-contained cosim); ``"synth"`` ->
    synthesizable loaders/drains/collect only (the PE is supplied by the exported ap_ctrl_none IP).
    """
    if mode == "blackbox":
        return (
            _role_blackbox("pe_interior", "pe")
            + "\n"
            + _role_blackbox("load_a", "load_a")
            + "\n"
            + _role_blackbox("load_b", "load_b")
            + "\n"
            + _role_blackbox("drain", "drain")
            + "\n"
            + _role_blackbox("collect", "collect")
        )
    if mode == "behavioral":
        return _behavioral_pe() + "\n" + _synth_support_modules()
    if mode == "synth":
        return _synth_support_modules()
    raise ValueError(f"unknown role mode {mode!r}")


def _structural_top_sv(rows, cols, k, dw, fa_depth, fb_depth, mode):
    """The structural ``spmw_top``: one role instance per grid point + one FIFO per channel.

    ``mode`` selects the role bodies (``blackbox``/``behavioral``/``synth``). The PE carries the
    exported IP ABI (ap_clk/ap_rst + a ``c_out`` ap_fifo result stream); each PE's result FIFO is
    read by a ``collect`` into ``C[i][j]``. ``load_b`` is fed column ``j`` of ``B`` (gathered from
    ``B[*][j]``), not row ``j``.
    """

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
    pe_params = "" if mode == "synth" else "#(.DW(DW), .K(K)) "
    return (
        "`timescale 1ns/1ps\n\n" + _fifo_module() + "\n" + _role_modules(mode) + "\n"
        f"module spmw_top #(parameter M = {rows}, parameter N = {cols}, parameter K = {k},\n"
        f"                  parameter DW = {dw}, parameter DA = {fa_depth}, parameter DB = {fb_depth}) (\n"
        "  input  wire          clk,\n"
        "  input  wire          rst_n,\n"
        "  input  wire [DW-1:0] A [0:M-1][0:K-1],\n"
        "  input  wire [DW-1:0] B [0:K-1][0:N-1],\n"
        "  output wire [DW-1:0] C [0:M-1][0:N-1]\n"
        ");\n"
        # A family (east/west): (N+1)/row; B family (north/south): (M+1)/column; results: M*N.
        "  wire [DW-1:0] fa_din  [0:M-1][0:N];   wire [DW-1:0] fa_dout [0:M-1][0:N];\n"
        "  wire          fa_full [0:M-1][0:N];   wire          fa_empty[0:M-1][0:N];\n"
        "  wire          fa_wr   [0:M-1][0:N];   wire          fa_rd   [0:M-1][0:N];\n"
        "  wire [DW-1:0] fb_din  [0:M][0:N-1];   wire [DW-1:0] fb_dout [0:M][0:N-1];\n"
        "  wire          fb_full [0:M][0:N-1];   wire          fb_empty[0:M][0:N-1];\n"
        "  wire          fb_wr   [0:M][0:N-1];   wire          fb_rd   [0:M][0:N-1];\n"
        "  wire [DW-1:0] fc_din  [0:M-1][0:N-1]; wire [DW-1:0] fc_dout [0:M-1][0:N-1];\n"
        "  wire          fc_full [0:M-1][0:N-1]; wire          fc_empty[0:M-1][0:N-1];\n"
        "  wire          fc_wr   [0:M-1][0:N-1]; wire          fc_rd   [0:M-1][0:N-1];\n"
        "  genvar i, j, g;\n"
        "  generate\n"
        # one FIFO per A-family channel, per B-family channel, and per PE result
        "    for (i = 0; i < M; i = i + 1) begin : fa_row\n"
        "      for (j = 0; j < N + 1; j = j + 1) begin : fa_col\n"
        "        spmw_fifo #(.DW(DW), .DEPTH(DA)) u_fa (.clk(clk), .rst_n(rst_n),\n"
        "          .din(fa_din[i][j]), .full_n(fa_full[i][j]), .write(fa_wr[i][j]),\n"
        "          .dout(fa_dout[i][j]), .empty_n(fa_empty[i][j]), .read(fa_rd[i][j]));\n"
        "      end\n"
        "    end\n"
        "    for (i = 0; i < M + 1; i = i + 1) begin : fb_row\n"
        "      for (j = 0; j < N; j = j + 1) begin : fb_col\n"
        "        spmw_fifo #(.DW(DW), .DEPTH(DB)) u_fb (.clk(clk), .rst_n(rst_n),\n"
        "          .din(fb_din[i][j]), .full_n(fb_full[i][j]), .write(fb_wr[i][j]),\n"
        "          .dout(fb_dout[i][j]), .empty_n(fb_empty[i][j]), .read(fb_rd[i][j]));\n"
        "      end\n"
        "    end\n"
        "    for (i = 0; i < M; i = i + 1) begin : fc_row\n"
        "      for (j = 0; j < N; j = j + 1) begin : fc_col\n"
        "        spmw_fifo #(.DW(DW), .DEPTH(2)) u_fc (.clk(clk), .rst_n(rst_n),\n"
        "          .din(fc_din[i][j]), .full_n(fc_full[i][j]), .write(fc_wr[i][j]),\n"
        "          .dout(fc_dout[i][j]), .empty_n(fc_empty[i][j]), .read(fc_rd[i][j]));\n"
        "      end\n"
        "    end\n"
        # loaders: load_a feeds row i of A; load_b feeds COLUMN j of B (gathered from B[*][j])
        "    for (i = 0; i < M; i = i + 1) begin : load_a_row\n"
        "      load_a #(.DW(DW), .K(K)) u_la (.clk(clk), .rst_n(rst_n), .a_row(A[i]),\n"
        "        .out_din(fa_din[i][0]), .out_full_n(fa_full[i][0]), .out_write(fa_wr[i][0]));\n"
        "    end\n"
        "    for (j = 0; j < N; j = j + 1) begin : load_b_col\n"
        "      wire [DW-1:0] bcol [0:K-1];\n"
        "      for (g = 0; g < K; g = g + 1) begin : bcol_gather\n"
        "        assign bcol[g] = B[g][j];\n"
        "      end\n"
        "      load_b #(.DW(DW), .K(K)) u_lb (.clk(clk), .rst_n(rst_n), .b_col(bcol),\n"
        "        .out_din(fb_din[0][j]), .out_full_n(fb_full[0][j]), .out_write(fb_wr[0][j]));\n"
        "    end\n"
        # one PE per grid point (exported-IP ABI: ap_clk/ap_rst, result on the c_out stream). The
        # exported ap_ctrl_none IP bakes its width/trip-count as literals (no DW/K parameters), so the
        # synth path instantiates it bare; the blackbox/behavioral PEs are parameterized.
        "    for (i = 0; i < M; i = i + 1) begin : pe_row\n"
        "      for (j = 0; j < N; j = j + 1) begin : pe_col\n"
        f"        pe_interior {pe_params}u_pe (.ap_clk(clk), .ap_rst(~rst_n),\n"
        f"{pe_conn},\n"
        "          .c_out_din(fc_din[i][j]), .c_out_full_n(fc_full[i][j]), .c_out_write(fc_wr[i][j]));\n"
        "      end\n"
        "    end\n"
        # one collector per PE: read its result FIFO into C[i][j]
        "    for (i = 0; i < M; i = i + 1) begin : collect_row\n"
        "      for (j = 0; j < N; j = j + 1) begin : collect_col\n"
        "        collect #(.DW(DW)) u_col (.clk(clk), .rst_n(rst_n),\n"
        "          .in_dout(fc_dout[i][j]), .in_empty_n(fc_empty[i][j]), .in_read(fc_rd[i][j]),\n"
        "          .c_val(C[i][j]));\n"
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


def emit_role_ip(region):
    """The interior PE as a free-running ``ap_ctrl_none`` HLS IP (fills the structural top's PE block).

    The role top carries ``#pragma HLS interface ap_ctrl_none port=return`` -- no ap_start/ap_done
    block handshake, so it runs as a persistent dataflow block -- and every port is a
    ``hls::stream<T>&`` (the compute result leaves on a ``c_out`` stream instead of a memref), so all
    ports synthesize as ap_fifo: the exact ``<p>_dout/_empty_n/_read`` and ``<p>_din/_full_n/_write``
    ABI the structural top's black boxes expect. csynth/export turns it into RTL that drops in.
    """
    # pylint: disable=import-outside-toplevel
    from .spmw import _collect, _validate_collection
    from .spmw_datapath import _resolve_dims
    from .spmw_hls import _CPP_TYPE, _ROLLED_WIRING, transcribe_pe_cpp

    collection = _validate_collection(_collect(region))
    decl = collection.maps[0]
    _rows, _cols, depth, dtype = _resolve_dims(region)
    elem = _CPP_TYPE[dtype]
    ports = sorted(decl.topology.port_names())
    if set(ports) != set(_ROLLED_WIRING):
        raise NotImplementedError(
            f"role-IP export handles the systolic port set {sorted(_ROLLED_WIRING)}; got {ports}"
        )
    body_lines = []
    for stmt in transcribe_pe_cpp(decl.unit):
        # the compute result is streamed out (c_local[0] = X -> c_out.write(X)) so the whole PE is a
        # pure ap_fifo dataflow block with no memory interface
        stmt = re.sub(r"c_local\[0\] = (.+);", r"c_out.write(\1);", stmt)
        for line in stmt.splitlines():
            body_lines.append("  " + line)
    body = "\n".join(body_lines)
    args = (
        ", ".join(f"hls::stream<{elem}> &{port}" for port in ports)
        + f", hls::stream<{elem}> &c_out"
    )
    fn = (
        f"void pe_interior({args}) {{\n"
        "#pragma HLS interface ap_ctrl_none port=return\n"
        f"{body}\n}}\n"
    )
    return f"#include <hls_stream.h>\n#define K {depth}\n\n{fn}"


def _role_ip_tcl(part, frequency):
    """A Vitis HLS script that csynths the free-running ``pe_interior`` role IP."""
    return (
        "open_project role_ip.prj\n"
        "set_top pe_interior\n"
        "add_files kernel.cpp\n"
        "open_solution solution1\n"
        f"set_part {part}\n"
        f"create_clock -period {1000 / frequency:.2f} -name default\n"
        "csynth_design\n"
        "exit\n"
    )


def emit_role_ip_project(region, project=None, part=None, frequency=None):
    """Write the free-running role IP as a stand-alone Vitis HLS project (``kernel.cpp`` + ``run.tcl``)."""
    # pylint: disable=import-outside-toplevel
    import os

    from .spmw_hls import _DEFAULT_FREQUENCY_MHZ, _DEFAULT_PART

    part = part or _DEFAULT_PART
    frequency = frequency or _DEFAULT_FREQUENCY_MHZ
    hls_code = emit_role_ip(region)
    if project is not None:
        os.makedirs(project, exist_ok=True)
        with open(os.path.join(project, "kernel.cpp"), "w", encoding="utf-8") as handle:
            handle.write(hls_code)
        with open(os.path.join(project, "run.tcl"), "w", encoding="utf-8") as handle:
            handle.write(_role_ip_tcl(part, frequency))
    return hls_code


def emit_cosim_testbench(region):
    """A self-checking testbench that drives A/B, runs ``spmw_top``, and checks ``C`` against A@B.

    Both the DUT PE and the reference sum ``sum_k A[i][k]*B[k][j]`` in ``shortreal`` in the same
    k-order, so the comparison is against the same oracle the M1 twins match numpy on.
    """
    # pylint: disable=import-outside-toplevel
    from .spmw_datapath import _resolve_dims

    rows, cols, k, _dtype = _resolve_dims(region)
    return (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        f"  localparam M = {rows}, N = {cols}, K = {k}, DW = 32;\n"
        "  reg clk = 0, rst_n = 0;\n"
        "  reg  [DW-1:0] A [0:M-1][0:K-1];\n"
        "  reg  [DW-1:0] B [0:K-1][0:N-1];\n"
        "  wire [DW-1:0] C [0:M-1][0:N-1];\n"
        "  spmw_top dut (.clk(clk), .rst_n(rst_n), .A(A), .B(B), .C(C));\n"
        "  always #5 clk = ~clk;\n"
        "  integer i, j, kk, errors;\n"
        "  shortreal exp_c, got_c;\n"
        "  initial begin\n"
        "    for (i = 0; i < M; i = i + 1) for (kk = 0; kk < K; kk = kk + 1)\n"
        "      A[i][kk] = $shortrealtobits(shortreal'(i + kk + 1) * 0.5);\n"
        "    for (kk = 0; kk < K; kk = kk + 1) for (j = 0; j < N; j = j + 1)\n"
        "      B[kk][j] = $shortrealtobits(shortreal'(kk + j + 1) * 0.25);\n"
        "    rst_n = 0; #20; rst_n = 1;\n"
        "    #5000;\n"
        "    errors = 0;\n"
        "    for (i = 0; i < M; i = i + 1) for (j = 0; j < N; j = j + 1) begin\n"
        "      exp_c = 0.0;\n"
        "      for (kk = 0; kk < K; kk = kk + 1)\n"
        "        exp_c = exp_c + $bitstoshortreal(A[i][kk]) * $bitstoshortreal(B[kk][j]);\n"
        "      got_c = $bitstoshortreal(C[i][j]);\n"
        "      if ((got_c - exp_c > 0.001) || (exp_c - got_c > 0.001)) begin\n"
        "        errors = errors + 1;\n"
        '        $display("MISMATCH C[%0d][%0d] exp=%f got=%f", i, j, exp_c, got_c);\n'
        "      end\n"
        "    end\n"
        '    if (errors == 0) $display("COSIM PASS");\n'
        '    else $display("COSIM FAIL %0d", errors);\n'
        "    $finish;\n"
        "  end\n"
        "endmodule\n"
    )


def emit_cosim_project(region, project=None):
    """Write the self-contained cosim project: the behavioral ``spmw_top`` + testbench + xsim script."""
    # pylint: disable=import-outside-toplevel
    import os

    dut = emit_structural_verilog(region, mode="behavioral")
    testbench = emit_cosim_testbench(region)
    if project is not None:
        os.makedirs(project, exist_ok=True)
        with open(os.path.join(project, "dut.sv"), "w", encoding="utf-8") as handle:
            handle.write(dut)
        with open(os.path.join(project, "tb.sv"), "w", encoding="utf-8") as handle:
            handle.write(testbench)
        with open(os.path.join(project, "run.sh"), "w", encoding="utf-8") as handle:
            handle.write(
                "#!/bin/bash\nset -e\n"
                "xvlog -sv dut.sv tb.sv\nxelab tb -s sim\nxsim sim -R\n"
            )
    return dut, testbench


def _role_ip_export_tcl(part, frequency):
    """A Vitis HLS script that csynths *and* exports the role as a reusable Vivado IP (ip_catalog)."""
    return (
        "open_project role_ip.prj\n"
        "set_top pe_interior\n"
        "add_files kernel.cpp\n"
        "open_solution solution1\n"
        f"set_part {part}\n"
        f"create_clock -period {1000 / frequency:.2f} -name default\n"
        "csynth_design\n"
        "export_design -format ip_catalog\n"
        "exit\n"
    )


def _vitis_rtl_package_tcl(part):
    """A Vivado batch script that assembles the synthesizable hierarchy: the reused PE IP + the top.

    Hierarchical IP reuse: the single ``pe_interior`` RTL (synthesized once from the ap_ctrl_none IP)
    plus the reusable ``spmw_fifo`` are added to the project, and ``spmw_top`` instantiates ``pe_interior``
    ``M*N`` times over a generate nest -- never re-synthesized per grid point. The synthesizable
    loaders/drains/collect complete the hierarchy; the PE body is the exported IP RTL. The final
    ``package_xo``/``v++`` link into an ``.xo`` + XRT bitstream is the out-of-band hardware build.
    """
    return (
        "# assemble the reused PE IP RTL + the synthesizable structural top\n"
        f"create_project -force spmw_rtl ./spmw_rtl_vivado -part {part}\n"
        "add_files -norecurse [glob ./role_ip.prj/solution1/syn/verilog/*.v]\n"
        "add_files -norecurse spmw_top.sv\n"
        "set_property top spmw_top [current_fileset]\n"
        "update_compile_order -fileset sources_1\n"
        "# also register the exported IP-catalog package so the PE can be reused as an IP\n"
        "set_property ip_repo_paths ./role_ip.prj/solution1/impl/ip [current_project]\n"
        "update_ip_catalog\n"
        "# next (out-of-band, needs the platform): package_xo -xo spmw.xo ...; v++ -l --platform ...\n"
    )


def emit_vitis_rtl_project(region, project=None, part=None, frequency=None):
    """Emit the ``vitis_rtl`` packaging project: one reusable role IP + the structural top + scripts.

    The synthesizable ``spmw_top`` (``mode="synth"``: real loaders/drains/collect, the PE left to the
    IP) instantiates a *single* ``pe_interior`` across the whole grid (hierarchical IP reuse).
    ``build.sh`` synthesizes and exports that one IP (its RTL fills the PE slot), then assembles the
    real synthesizable hierarchy for the ``.xo``/``v++``/XRT packaging flow (the final hardware link
    is out of band). Returns the ``{filename: contents}`` map that was written.
    """
    # pylint: disable=import-outside-toplevel
    import os

    from .spmw_hls import _DEFAULT_FREQUENCY_MHZ, _DEFAULT_PART

    part = part or _DEFAULT_PART
    frequency = frequency or _DEFAULT_FREQUENCY_MHZ
    files = {
        "spmw_top.sv": emit_structural_verilog(region, mode="synth"),
        "kernel.cpp": emit_role_ip(region),
        "synth_ip.tcl": _role_ip_export_tcl(part, frequency),
        "package.tcl": _vitis_rtl_package_tcl(part),
        "build.sh": (
            "#!/bin/bash\nset -e\n"
            "# 1. synthesize + export the single reusable pe_interior role IP (RTL fills the PE slot)\n"
            "vitis_hls -f synth_ip.tcl\n"
            "# 2. assemble the synthesizable hierarchy (PE IP RTL + structural top) for packaging\n"
            "vivado -mode batch -source package.tcl\n"
        ),
    }
    if project is not None:
        os.makedirs(project, exist_ok=True)
        for name, contents in files.items():
            with open(os.path.join(project, name), "w", encoding="utf-8") as handle:
                handle.write(contents)
    return files
