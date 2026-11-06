`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] perm0_u_in_bind_dout [0:7],
  input  wire perm0_u_in_bind_empty_n [0:7],
  output wire perm0_u_in_bind_read [0:7],
  input  wire [63:0] perm0_v_in_bind_dout [0:7],
  input  wire perm0_v_in_bind_empty_n [0:7],
  output wire perm0_v_in_bind_read [0:7],
  output wire [63:0] reorder_a_out_bind_din [0:7],
  output wire reorder_a_out_bind_write [0:7],
  input  wire reorder_a_out_bind_full_n [0:7],
  output wire [63:0] reorder_b_out_bind_din [0:7],
  output wire reorder_b_out_bind_write [0:7],
  input  wire reorder_b_out_bind_full_n [0:7]
);
  // family bfly0_a_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] bfly0_a_in_bind_din [0:7];
  wire [63:0] bfly0_a_in_bind_dout [0:7];
  wire bfly0_a_in_bind_full_n [0:7];
  wire bfly0_a_in_bind_write [0:7];
  wire bfly0_a_in_bind_empty_n [0:7];
  wire bfly0_a_in_bind_read [0:7];
  genvar bfly0_a_in_bind_i;
  generate
    for (bfly0_a_in_bind_i = 0; bfly0_a_in_bind_i < 8; bfly0_a_in_bind_i = bfly0_a_in_bind_i + 1) begin : g_bfly0_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly0_a_in_bind_din[bfly0_a_in_bind_i]), .full_n(bfly0_a_in_bind_full_n[bfly0_a_in_bind_i]), .write(bfly0_a_in_bind_write[bfly0_a_in_bind_i]), .dout(bfly0_a_in_bind_dout[bfly0_a_in_bind_i]), .empty_n(bfly0_a_in_bind_empty_n[bfly0_a_in_bind_i]), .read(bfly0_a_in_bind_read[bfly0_a_in_bind_i]));
    end
  endgenerate
  // family bfly0_b_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] bfly0_b_in_bind_din [0:7];
  wire [63:0] bfly0_b_in_bind_dout [0:7];
  wire bfly0_b_in_bind_full_n [0:7];
  wire bfly0_b_in_bind_write [0:7];
  wire bfly0_b_in_bind_empty_n [0:7];
  wire bfly0_b_in_bind_read [0:7];
  genvar bfly0_b_in_bind_i;
  generate
    for (bfly0_b_in_bind_i = 0; bfly0_b_in_bind_i < 8; bfly0_b_in_bind_i = bfly0_b_in_bind_i + 1) begin : g_bfly0_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly0_b_in_bind_din[bfly0_b_in_bind_i]), .full_n(bfly0_b_in_bind_full_n[bfly0_b_in_bind_i]), .write(bfly0_b_in_bind_write[bfly0_b_in_bind_i]), .dout(bfly0_b_in_bind_dout[bfly0_b_in_bind_i]), .empty_n(bfly0_b_in_bind_empty_n[bfly0_b_in_bind_i]), .read(bfly0_b_in_bind_read[bfly0_b_in_bind_i]));
    end
  endgenerate
  // family perm1_u_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] perm1_u_in_bind_din [0:7];
  wire [63:0] perm1_u_in_bind_dout [0:7];
  wire perm1_u_in_bind_full_n [0:7];
  wire perm1_u_in_bind_write [0:7];
  wire perm1_u_in_bind_empty_n [0:7];
  wire perm1_u_in_bind_read [0:7];
  genvar perm1_u_in_bind_i;
  generate
    for (perm1_u_in_bind_i = 0; perm1_u_in_bind_i < 8; perm1_u_in_bind_i = perm1_u_in_bind_i + 1) begin : g_perm1_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm1_u_in_bind_din[perm1_u_in_bind_i]), .full_n(perm1_u_in_bind_full_n[perm1_u_in_bind_i]), .write(perm1_u_in_bind_write[perm1_u_in_bind_i]), .dout(perm1_u_in_bind_dout[perm1_u_in_bind_i]), .empty_n(perm1_u_in_bind_empty_n[perm1_u_in_bind_i]), .read(perm1_u_in_bind_read[perm1_u_in_bind_i]));
    end
  endgenerate
  // family perm1_v_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] perm1_v_in_bind_din [0:7];
  wire [63:0] perm1_v_in_bind_dout [0:7];
  wire perm1_v_in_bind_full_n [0:7];
  wire perm1_v_in_bind_write [0:7];
  wire perm1_v_in_bind_empty_n [0:7];
  wire perm1_v_in_bind_read [0:7];
  genvar perm1_v_in_bind_i;
  generate
    for (perm1_v_in_bind_i = 0; perm1_v_in_bind_i < 8; perm1_v_in_bind_i = perm1_v_in_bind_i + 1) begin : g_perm1_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm1_v_in_bind_din[perm1_v_in_bind_i]), .full_n(perm1_v_in_bind_full_n[perm1_v_in_bind_i]), .write(perm1_v_in_bind_write[perm1_v_in_bind_i]), .dout(perm1_v_in_bind_dout[perm1_v_in_bind_i]), .empty_n(perm1_v_in_bind_empty_n[perm1_v_in_bind_i]), .read(perm1_v_in_bind_read[perm1_v_in_bind_i]));
    end
  endgenerate
  // family bfly1_a_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] bfly1_a_in_bind_din [0:7];
  wire [63:0] bfly1_a_in_bind_dout [0:7];
  wire bfly1_a_in_bind_full_n [0:7];
  wire bfly1_a_in_bind_write [0:7];
  wire bfly1_a_in_bind_empty_n [0:7];
  wire bfly1_a_in_bind_read [0:7];
  genvar bfly1_a_in_bind_i;
  generate
    for (bfly1_a_in_bind_i = 0; bfly1_a_in_bind_i < 8; bfly1_a_in_bind_i = bfly1_a_in_bind_i + 1) begin : g_bfly1_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly1_a_in_bind_din[bfly1_a_in_bind_i]), .full_n(bfly1_a_in_bind_full_n[bfly1_a_in_bind_i]), .write(bfly1_a_in_bind_write[bfly1_a_in_bind_i]), .dout(bfly1_a_in_bind_dout[bfly1_a_in_bind_i]), .empty_n(bfly1_a_in_bind_empty_n[bfly1_a_in_bind_i]), .read(bfly1_a_in_bind_read[bfly1_a_in_bind_i]));
    end
  endgenerate
  // family bfly1_b_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] bfly1_b_in_bind_din [0:7];
  wire [63:0] bfly1_b_in_bind_dout [0:7];
  wire bfly1_b_in_bind_full_n [0:7];
  wire bfly1_b_in_bind_write [0:7];
  wire bfly1_b_in_bind_empty_n [0:7];
  wire bfly1_b_in_bind_read [0:7];
  genvar bfly1_b_in_bind_i;
  generate
    for (bfly1_b_in_bind_i = 0; bfly1_b_in_bind_i < 8; bfly1_b_in_bind_i = bfly1_b_in_bind_i + 1) begin : g_bfly1_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly1_b_in_bind_din[bfly1_b_in_bind_i]), .full_n(bfly1_b_in_bind_full_n[bfly1_b_in_bind_i]), .write(bfly1_b_in_bind_write[bfly1_b_in_bind_i]), .dout(bfly1_b_in_bind_dout[bfly1_b_in_bind_i]), .empty_n(bfly1_b_in_bind_empty_n[bfly1_b_in_bind_i]), .read(bfly1_b_in_bind_read[bfly1_b_in_bind_i]));
    end
  endgenerate
  // family perm2_u_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] perm2_u_in_bind_din [0:7];
  wire [63:0] perm2_u_in_bind_dout [0:7];
  wire perm2_u_in_bind_full_n [0:7];
  wire perm2_u_in_bind_write [0:7];
  wire perm2_u_in_bind_empty_n [0:7];
  wire perm2_u_in_bind_read [0:7];
  genvar perm2_u_in_bind_i;
  generate
    for (perm2_u_in_bind_i = 0; perm2_u_in_bind_i < 8; perm2_u_in_bind_i = perm2_u_in_bind_i + 1) begin : g_perm2_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm2_u_in_bind_din[perm2_u_in_bind_i]), .full_n(perm2_u_in_bind_full_n[perm2_u_in_bind_i]), .write(perm2_u_in_bind_write[perm2_u_in_bind_i]), .dout(perm2_u_in_bind_dout[perm2_u_in_bind_i]), .empty_n(perm2_u_in_bind_empty_n[perm2_u_in_bind_i]), .read(perm2_u_in_bind_read[perm2_u_in_bind_i]));
    end
  endgenerate
  // family perm2_v_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] perm2_v_in_bind_din [0:7];
  wire [63:0] perm2_v_in_bind_dout [0:7];
  wire perm2_v_in_bind_full_n [0:7];
  wire perm2_v_in_bind_write [0:7];
  wire perm2_v_in_bind_empty_n [0:7];
  wire perm2_v_in_bind_read [0:7];
  genvar perm2_v_in_bind_i;
  generate
    for (perm2_v_in_bind_i = 0; perm2_v_in_bind_i < 8; perm2_v_in_bind_i = perm2_v_in_bind_i + 1) begin : g_perm2_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm2_v_in_bind_din[perm2_v_in_bind_i]), .full_n(perm2_v_in_bind_full_n[perm2_v_in_bind_i]), .write(perm2_v_in_bind_write[perm2_v_in_bind_i]), .dout(perm2_v_in_bind_dout[perm2_v_in_bind_i]), .empty_n(perm2_v_in_bind_empty_n[perm2_v_in_bind_i]), .read(perm2_v_in_bind_read[perm2_v_in_bind_i]));
    end
  endgenerate
  // family bfly2_a_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] bfly2_a_in_bind_din [0:7];
  wire [63:0] bfly2_a_in_bind_dout [0:7];
  wire bfly2_a_in_bind_full_n [0:7];
  wire bfly2_a_in_bind_write [0:7];
  wire bfly2_a_in_bind_empty_n [0:7];
  wire bfly2_a_in_bind_read [0:7];
  genvar bfly2_a_in_bind_i;
  generate
    for (bfly2_a_in_bind_i = 0; bfly2_a_in_bind_i < 8; bfly2_a_in_bind_i = bfly2_a_in_bind_i + 1) begin : g_bfly2_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly2_a_in_bind_din[bfly2_a_in_bind_i]), .full_n(bfly2_a_in_bind_full_n[bfly2_a_in_bind_i]), .write(bfly2_a_in_bind_write[bfly2_a_in_bind_i]), .dout(bfly2_a_in_bind_dout[bfly2_a_in_bind_i]), .empty_n(bfly2_a_in_bind_empty_n[bfly2_a_in_bind_i]), .read(bfly2_a_in_bind_read[bfly2_a_in_bind_i]));
    end
  endgenerate
  // family bfly2_b_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] bfly2_b_in_bind_din [0:7];
  wire [63:0] bfly2_b_in_bind_dout [0:7];
  wire bfly2_b_in_bind_full_n [0:7];
  wire bfly2_b_in_bind_write [0:7];
  wire bfly2_b_in_bind_empty_n [0:7];
  wire bfly2_b_in_bind_read [0:7];
  genvar bfly2_b_in_bind_i;
  generate
    for (bfly2_b_in_bind_i = 0; bfly2_b_in_bind_i < 8; bfly2_b_in_bind_i = bfly2_b_in_bind_i + 1) begin : g_bfly2_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly2_b_in_bind_din[bfly2_b_in_bind_i]), .full_n(bfly2_b_in_bind_full_n[bfly2_b_in_bind_i]), .write(bfly2_b_in_bind_write[bfly2_b_in_bind_i]), .dout(bfly2_b_in_bind_dout[bfly2_b_in_bind_i]), .empty_n(bfly2_b_in_bind_empty_n[bfly2_b_in_bind_i]), .read(bfly2_b_in_bind_read[bfly2_b_in_bind_i]));
    end
  endgenerate
  // family perm3_u_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] perm3_u_in_bind_din [0:7];
  wire [63:0] perm3_u_in_bind_dout [0:7];
  wire perm3_u_in_bind_full_n [0:7];
  wire perm3_u_in_bind_write [0:7];
  wire perm3_u_in_bind_empty_n [0:7];
  wire perm3_u_in_bind_read [0:7];
  genvar perm3_u_in_bind_i;
  generate
    for (perm3_u_in_bind_i = 0; perm3_u_in_bind_i < 8; perm3_u_in_bind_i = perm3_u_in_bind_i + 1) begin : g_perm3_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm3_u_in_bind_din[perm3_u_in_bind_i]), .full_n(perm3_u_in_bind_full_n[perm3_u_in_bind_i]), .write(perm3_u_in_bind_write[perm3_u_in_bind_i]), .dout(perm3_u_in_bind_dout[perm3_u_in_bind_i]), .empty_n(perm3_u_in_bind_empty_n[perm3_u_in_bind_i]), .read(perm3_u_in_bind_read[perm3_u_in_bind_i]));
    end
  endgenerate
  // family perm3_v_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] perm3_v_in_bind_din [0:7];
  wire [63:0] perm3_v_in_bind_dout [0:7];
  wire perm3_v_in_bind_full_n [0:7];
  wire perm3_v_in_bind_write [0:7];
  wire perm3_v_in_bind_empty_n [0:7];
  wire perm3_v_in_bind_read [0:7];
  genvar perm3_v_in_bind_i;
  generate
    for (perm3_v_in_bind_i = 0; perm3_v_in_bind_i < 8; perm3_v_in_bind_i = perm3_v_in_bind_i + 1) begin : g_perm3_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm3_v_in_bind_din[perm3_v_in_bind_i]), .full_n(perm3_v_in_bind_full_n[perm3_v_in_bind_i]), .write(perm3_v_in_bind_write[perm3_v_in_bind_i]), .dout(perm3_v_in_bind_dout[perm3_v_in_bind_i]), .empty_n(perm3_v_in_bind_empty_n[perm3_v_in_bind_i]), .read(perm3_v_in_bind_read[perm3_v_in_bind_i]));
    end
  endgenerate
  // family bfly3_a_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] bfly3_a_in_bind_din [0:7];
  wire [63:0] bfly3_a_in_bind_dout [0:7];
  wire bfly3_a_in_bind_full_n [0:7];
  wire bfly3_a_in_bind_write [0:7];
  wire bfly3_a_in_bind_empty_n [0:7];
  wire bfly3_a_in_bind_read [0:7];
  genvar bfly3_a_in_bind_i;
  generate
    for (bfly3_a_in_bind_i = 0; bfly3_a_in_bind_i < 8; bfly3_a_in_bind_i = bfly3_a_in_bind_i + 1) begin : g_bfly3_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly3_a_in_bind_din[bfly3_a_in_bind_i]), .full_n(bfly3_a_in_bind_full_n[bfly3_a_in_bind_i]), .write(bfly3_a_in_bind_write[bfly3_a_in_bind_i]), .dout(bfly3_a_in_bind_dout[bfly3_a_in_bind_i]), .empty_n(bfly3_a_in_bind_empty_n[bfly3_a_in_bind_i]), .read(bfly3_a_in_bind_read[bfly3_a_in_bind_i]));
    end
  endgenerate
  // family bfly3_b_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] bfly3_b_in_bind_din [0:7];
  wire [63:0] bfly3_b_in_bind_dout [0:7];
  wire bfly3_b_in_bind_full_n [0:7];
  wire bfly3_b_in_bind_write [0:7];
  wire bfly3_b_in_bind_empty_n [0:7];
  wire bfly3_b_in_bind_read [0:7];
  genvar bfly3_b_in_bind_i;
  generate
    for (bfly3_b_in_bind_i = 0; bfly3_b_in_bind_i < 8; bfly3_b_in_bind_i = bfly3_b_in_bind_i + 1) begin : g_bfly3_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly3_b_in_bind_din[bfly3_b_in_bind_i]), .full_n(bfly3_b_in_bind_full_n[bfly3_b_in_bind_i]), .write(bfly3_b_in_bind_write[bfly3_b_in_bind_i]), .dout(bfly3_b_in_bind_dout[bfly3_b_in_bind_i]), .empty_n(bfly3_b_in_bind_empty_n[bfly3_b_in_bind_i]), .read(bfly3_b_in_bind_read[bfly3_b_in_bind_i]));
    end
  endgenerate
  // family perm4_u_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] perm4_u_in_bind_din [0:7];
  wire [63:0] perm4_u_in_bind_dout [0:7];
  wire perm4_u_in_bind_full_n [0:7];
  wire perm4_u_in_bind_write [0:7];
  wire perm4_u_in_bind_empty_n [0:7];
  wire perm4_u_in_bind_read [0:7];
  genvar perm4_u_in_bind_i;
  generate
    for (perm4_u_in_bind_i = 0; perm4_u_in_bind_i < 8; perm4_u_in_bind_i = perm4_u_in_bind_i + 1) begin : g_perm4_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm4_u_in_bind_din[perm4_u_in_bind_i]), .full_n(perm4_u_in_bind_full_n[perm4_u_in_bind_i]), .write(perm4_u_in_bind_write[perm4_u_in_bind_i]), .dout(perm4_u_in_bind_dout[perm4_u_in_bind_i]), .empty_n(perm4_u_in_bind_empty_n[perm4_u_in_bind_i]), .read(perm4_u_in_bind_read[perm4_u_in_bind_i]));
    end
  endgenerate
  // family perm4_v_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] perm4_v_in_bind_din [0:7];
  wire [63:0] perm4_v_in_bind_dout [0:7];
  wire perm4_v_in_bind_full_n [0:7];
  wire perm4_v_in_bind_write [0:7];
  wire perm4_v_in_bind_empty_n [0:7];
  wire perm4_v_in_bind_read [0:7];
  genvar perm4_v_in_bind_i;
  generate
    for (perm4_v_in_bind_i = 0; perm4_v_in_bind_i < 8; perm4_v_in_bind_i = perm4_v_in_bind_i + 1) begin : g_perm4_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm4_v_in_bind_din[perm4_v_in_bind_i]), .full_n(perm4_v_in_bind_full_n[perm4_v_in_bind_i]), .write(perm4_v_in_bind_write[perm4_v_in_bind_i]), .dout(perm4_v_in_bind_dout[perm4_v_in_bind_i]), .empty_n(perm4_v_in_bind_empty_n[perm4_v_in_bind_i]), .read(perm4_v_in_bind_read[perm4_v_in_bind_i]));
    end
  endgenerate
  // family bfly4_a_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] bfly4_a_in_bind_din [0:7];
  wire [63:0] bfly4_a_in_bind_dout [0:7];
  wire bfly4_a_in_bind_full_n [0:7];
  wire bfly4_a_in_bind_write [0:7];
  wire bfly4_a_in_bind_empty_n [0:7];
  wire bfly4_a_in_bind_read [0:7];
  genvar bfly4_a_in_bind_i;
  generate
    for (bfly4_a_in_bind_i = 0; bfly4_a_in_bind_i < 8; bfly4_a_in_bind_i = bfly4_a_in_bind_i + 1) begin : g_bfly4_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly4_a_in_bind_din[bfly4_a_in_bind_i]), .full_n(bfly4_a_in_bind_full_n[bfly4_a_in_bind_i]), .write(bfly4_a_in_bind_write[bfly4_a_in_bind_i]), .dout(bfly4_a_in_bind_dout[bfly4_a_in_bind_i]), .empty_n(bfly4_a_in_bind_empty_n[bfly4_a_in_bind_i]), .read(bfly4_a_in_bind_read[bfly4_a_in_bind_i]));
    end
  endgenerate
  // family bfly4_b_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] bfly4_b_in_bind_din [0:7];
  wire [63:0] bfly4_b_in_bind_dout [0:7];
  wire bfly4_b_in_bind_full_n [0:7];
  wire bfly4_b_in_bind_write [0:7];
  wire bfly4_b_in_bind_empty_n [0:7];
  wire bfly4_b_in_bind_read [0:7];
  genvar bfly4_b_in_bind_i;
  generate
    for (bfly4_b_in_bind_i = 0; bfly4_b_in_bind_i < 8; bfly4_b_in_bind_i = bfly4_b_in_bind_i + 1) begin : g_bfly4_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly4_b_in_bind_din[bfly4_b_in_bind_i]), .full_n(bfly4_b_in_bind_full_n[bfly4_b_in_bind_i]), .write(bfly4_b_in_bind_write[bfly4_b_in_bind_i]), .dout(bfly4_b_in_bind_dout[bfly4_b_in_bind_i]), .empty_n(bfly4_b_in_bind_empty_n[bfly4_b_in_bind_i]), .read(bfly4_b_in_bind_read[bfly4_b_in_bind_i]));
    end
  endgenerate
  // family perm5_u_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] perm5_u_in_bind_din [0:7];
  wire [63:0] perm5_u_in_bind_dout [0:7];
  wire perm5_u_in_bind_full_n [0:7];
  wire perm5_u_in_bind_write [0:7];
  wire perm5_u_in_bind_empty_n [0:7];
  wire perm5_u_in_bind_read [0:7];
  genvar perm5_u_in_bind_i;
  generate
    for (perm5_u_in_bind_i = 0; perm5_u_in_bind_i < 8; perm5_u_in_bind_i = perm5_u_in_bind_i + 1) begin : g_perm5_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm5_u_in_bind_din[perm5_u_in_bind_i]), .full_n(perm5_u_in_bind_full_n[perm5_u_in_bind_i]), .write(perm5_u_in_bind_write[perm5_u_in_bind_i]), .dout(perm5_u_in_bind_dout[perm5_u_in_bind_i]), .empty_n(perm5_u_in_bind_empty_n[perm5_u_in_bind_i]), .read(perm5_u_in_bind_read[perm5_u_in_bind_i]));
    end
  endgenerate
  // family perm5_v_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] perm5_v_in_bind_din [0:7];
  wire [63:0] perm5_v_in_bind_dout [0:7];
  wire perm5_v_in_bind_full_n [0:7];
  wire perm5_v_in_bind_write [0:7];
  wire perm5_v_in_bind_empty_n [0:7];
  wire perm5_v_in_bind_read [0:7];
  genvar perm5_v_in_bind_i;
  generate
    for (perm5_v_in_bind_i = 0; perm5_v_in_bind_i < 8; perm5_v_in_bind_i = perm5_v_in_bind_i + 1) begin : g_perm5_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm5_v_in_bind_din[perm5_v_in_bind_i]), .full_n(perm5_v_in_bind_full_n[perm5_v_in_bind_i]), .write(perm5_v_in_bind_write[perm5_v_in_bind_i]), .dout(perm5_v_in_bind_dout[perm5_v_in_bind_i]), .empty_n(perm5_v_in_bind_empty_n[perm5_v_in_bind_i]), .read(perm5_v_in_bind_read[perm5_v_in_bind_i]));
    end
  endgenerate
  // family bfly5_a_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] bfly5_a_in_bind_din [0:7];
  wire [63:0] bfly5_a_in_bind_dout [0:7];
  wire bfly5_a_in_bind_full_n [0:7];
  wire bfly5_a_in_bind_write [0:7];
  wire bfly5_a_in_bind_empty_n [0:7];
  wire bfly5_a_in_bind_read [0:7];
  genvar bfly5_a_in_bind_i;
  generate
    for (bfly5_a_in_bind_i = 0; bfly5_a_in_bind_i < 8; bfly5_a_in_bind_i = bfly5_a_in_bind_i + 1) begin : g_bfly5_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly5_a_in_bind_din[bfly5_a_in_bind_i]), .full_n(bfly5_a_in_bind_full_n[bfly5_a_in_bind_i]), .write(bfly5_a_in_bind_write[bfly5_a_in_bind_i]), .dout(bfly5_a_in_bind_dout[bfly5_a_in_bind_i]), .empty_n(bfly5_a_in_bind_empty_n[bfly5_a_in_bind_i]), .read(bfly5_a_in_bind_read[bfly5_a_in_bind_i]));
    end
  endgenerate
  // family bfly5_b_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] bfly5_b_in_bind_din [0:7];
  wire [63:0] bfly5_b_in_bind_dout [0:7];
  wire bfly5_b_in_bind_full_n [0:7];
  wire bfly5_b_in_bind_write [0:7];
  wire bfly5_b_in_bind_empty_n [0:7];
  wire bfly5_b_in_bind_read [0:7];
  genvar bfly5_b_in_bind_i;
  generate
    for (bfly5_b_in_bind_i = 0; bfly5_b_in_bind_i < 8; bfly5_b_in_bind_i = bfly5_b_in_bind_i + 1) begin : g_bfly5_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly5_b_in_bind_din[bfly5_b_in_bind_i]), .full_n(bfly5_b_in_bind_full_n[bfly5_b_in_bind_i]), .write(bfly5_b_in_bind_write[bfly5_b_in_bind_i]), .dout(bfly5_b_in_bind_dout[bfly5_b_in_bind_i]), .empty_n(bfly5_b_in_bind_empty_n[bfly5_b_in_bind_i]), .read(bfly5_b_in_bind_read[bfly5_b_in_bind_i]));
    end
  endgenerate
  // family perm6_u_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] perm6_u_in_bind_din [0:7];
  wire [63:0] perm6_u_in_bind_dout [0:7];
  wire perm6_u_in_bind_full_n [0:7];
  wire perm6_u_in_bind_write [0:7];
  wire perm6_u_in_bind_empty_n [0:7];
  wire perm6_u_in_bind_read [0:7];
  genvar perm6_u_in_bind_i;
  generate
    for (perm6_u_in_bind_i = 0; perm6_u_in_bind_i < 8; perm6_u_in_bind_i = perm6_u_in_bind_i + 1) begin : g_perm6_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm6_u_in_bind_din[perm6_u_in_bind_i]), .full_n(perm6_u_in_bind_full_n[perm6_u_in_bind_i]), .write(perm6_u_in_bind_write[perm6_u_in_bind_i]), .dout(perm6_u_in_bind_dout[perm6_u_in_bind_i]), .empty_n(perm6_u_in_bind_empty_n[perm6_u_in_bind_i]), .read(perm6_u_in_bind_read[perm6_u_in_bind_i]));
    end
  endgenerate
  // family perm6_v_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] perm6_v_in_bind_din [0:7];
  wire [63:0] perm6_v_in_bind_dout [0:7];
  wire perm6_v_in_bind_full_n [0:7];
  wire perm6_v_in_bind_write [0:7];
  wire perm6_v_in_bind_empty_n [0:7];
  wire perm6_v_in_bind_read [0:7];
  genvar perm6_v_in_bind_i;
  generate
    for (perm6_v_in_bind_i = 0; perm6_v_in_bind_i < 8; perm6_v_in_bind_i = perm6_v_in_bind_i + 1) begin : g_perm6_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm6_v_in_bind_din[perm6_v_in_bind_i]), .full_n(perm6_v_in_bind_full_n[perm6_v_in_bind_i]), .write(perm6_v_in_bind_write[perm6_v_in_bind_i]), .dout(perm6_v_in_bind_dout[perm6_v_in_bind_i]), .empty_n(perm6_v_in_bind_empty_n[perm6_v_in_bind_i]), .read(perm6_v_in_bind_read[perm6_v_in_bind_i]));
    end
  endgenerate
  // family tail_general_a_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] tail_general_a_in_bind_din [0:7];
  wire [63:0] tail_general_a_in_bind_dout [0:7];
  wire tail_general_a_in_bind_full_n [0:7];
  wire tail_general_a_in_bind_write [0:7];
  wire tail_general_a_in_bind_empty_n [0:7];
  wire tail_general_a_in_bind_read [0:7];
  genvar tail_general_a_in_bind_i;
  generate
    for (tail_general_a_in_bind_i = 0; tail_general_a_in_bind_i < 8; tail_general_a_in_bind_i = tail_general_a_in_bind_i + 1) begin : g_tail_general_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tail_general_a_in_bind_din[tail_general_a_in_bind_i]), .full_n(tail_general_a_in_bind_full_n[tail_general_a_in_bind_i]), .write(tail_general_a_in_bind_write[tail_general_a_in_bind_i]), .dout(tail_general_a_in_bind_dout[tail_general_a_in_bind_i]), .empty_n(tail_general_a_in_bind_empty_n[tail_general_a_in_bind_i]), .read(tail_general_a_in_bind_read[tail_general_a_in_bind_i]));
    end
  endgenerate
  // family tail_general_b_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] tail_general_b_in_bind_din [0:7];
  wire [63:0] tail_general_b_in_bind_dout [0:7];
  wire tail_general_b_in_bind_full_n [0:7];
  wire tail_general_b_in_bind_write [0:7];
  wire tail_general_b_in_bind_empty_n [0:7];
  wire tail_general_b_in_bind_read [0:7];
  genvar tail_general_b_in_bind_i;
  generate
    for (tail_general_b_in_bind_i = 0; tail_general_b_in_bind_i < 8; tail_general_b_in_bind_i = tail_general_b_in_bind_i + 1) begin : g_tail_general_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tail_general_b_in_bind_din[tail_general_b_in_bind_i]), .full_n(tail_general_b_in_bind_full_n[tail_general_b_in_bind_i]), .write(tail_general_b_in_bind_write[tail_general_b_in_bind_i]), .dout(tail_general_b_in_bind_dout[tail_general_b_in_bind_i]), .empty_n(tail_general_b_in_bind_empty_n[tail_general_b_in_bind_i]), .read(tail_general_b_in_bind_read[tail_general_b_in_bind_i]));
    end
  endgenerate
  // family tail_general_a_out_a_in: 32 channel(s), 64-bit, depth 8
  wire [63:0] tail_general_a_out_a_in_din [0:31];
  wire [63:0] tail_general_a_out_a_in_dout [0:31];
  wire tail_general_a_out_a_in_full_n [0:31];
  wire tail_general_a_out_a_in_write [0:31];
  wire tail_general_a_out_a_in_empty_n [0:31];
  wire tail_general_a_out_a_in_read [0:31];
  genvar tail_general_a_out_a_in_i;
  generate
    for (tail_general_a_out_a_in_i = 0; tail_general_a_out_a_in_i < 32; tail_general_a_out_a_in_i = tail_general_a_out_a_in_i + 1) begin : g_tail_general_a_out_a_in
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tail_general_a_out_a_in_din[tail_general_a_out_a_in_i]), .full_n(tail_general_a_out_a_in_full_n[tail_general_a_out_a_in_i]), .write(tail_general_a_out_a_in_write[tail_general_a_out_a_in_i]), .dout(tail_general_a_out_a_in_dout[tail_general_a_out_a_in_i]), .empty_n(tail_general_a_out_a_in_empty_n[tail_general_a_out_a_in_i]), .read(tail_general_a_out_a_in_read[tail_general_a_out_a_in_i]));
    end
  endgenerate
  // family tail_general_b_out_a_in: 12 channel(s), 64-bit, depth 8
  wire [63:0] tail_general_b_out_a_in_din [0:11];
  wire [63:0] tail_general_b_out_a_in_dout [0:11];
  wire tail_general_b_out_a_in_full_n [0:11];
  wire tail_general_b_out_a_in_write [0:11];
  wire tail_general_b_out_a_in_empty_n [0:11];
  wire tail_general_b_out_a_in_read [0:11];
  genvar tail_general_b_out_a_in_i;
  generate
    for (tail_general_b_out_a_in_i = 0; tail_general_b_out_a_in_i < 12; tail_general_b_out_a_in_i = tail_general_b_out_a_in_i + 1) begin : g_tail_general_b_out_a_in
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tail_general_b_out_a_in_din[tail_general_b_out_a_in_i]), .full_n(tail_general_b_out_a_in_full_n[tail_general_b_out_a_in_i]), .write(tail_general_b_out_a_in_write[tail_general_b_out_a_in_i]), .dout(tail_general_b_out_a_in_dout[tail_general_b_out_a_in_i]), .empty_n(tail_general_b_out_a_in_empty_n[tail_general_b_out_a_in_i]), .read(tail_general_b_out_a_in_read[tail_general_b_out_a_in_i]));
    end
  endgenerate
  // family tail_general_a_out_b_in: 12 channel(s), 64-bit, depth 8
  wire [63:0] tail_general_a_out_b_in_din [0:11];
  wire [63:0] tail_general_a_out_b_in_dout [0:11];
  wire tail_general_a_out_b_in_full_n [0:11];
  wire tail_general_a_out_b_in_write [0:11];
  wire tail_general_a_out_b_in_empty_n [0:11];
  wire tail_general_a_out_b_in_read [0:11];
  genvar tail_general_a_out_b_in_i;
  generate
    for (tail_general_a_out_b_in_i = 0; tail_general_a_out_b_in_i < 12; tail_general_a_out_b_in_i = tail_general_a_out_b_in_i + 1) begin : g_tail_general_a_out_b_in
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tail_general_a_out_b_in_din[tail_general_a_out_b_in_i]), .full_n(tail_general_a_out_b_in_full_n[tail_general_a_out_b_in_i]), .write(tail_general_a_out_b_in_write[tail_general_a_out_b_in_i]), .dout(tail_general_a_out_b_in_dout[tail_general_a_out_b_in_i]), .empty_n(tail_general_a_out_b_in_empty_n[tail_general_a_out_b_in_i]), .read(tail_general_a_out_b_in_read[tail_general_a_out_b_in_i]));
    end
  endgenerate
  // family tail_general_b_out_b_in: 32 channel(s), 64-bit, depth 8
  wire [63:0] tail_general_b_out_b_in_din [0:31];
  wire [63:0] tail_general_b_out_b_in_dout [0:31];
  wire tail_general_b_out_b_in_full_n [0:31];
  wire tail_general_b_out_b_in_write [0:31];
  wire tail_general_b_out_b_in_empty_n [0:31];
  wire tail_general_b_out_b_in_read [0:31];
  genvar tail_general_b_out_b_in_i;
  generate
    for (tail_general_b_out_b_in_i = 0; tail_general_b_out_b_in_i < 32; tail_general_b_out_b_in_i = tail_general_b_out_b_in_i + 1) begin : g_tail_general_b_out_b_in
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tail_general_b_out_b_in_din[tail_general_b_out_b_in_i]), .full_n(tail_general_b_out_b_in_full_n[tail_general_b_out_b_in_i]), .write(tail_general_b_out_b_in_write[tail_general_b_out_b_in_i]), .dout(tail_general_b_out_b_in_dout[tail_general_b_out_b_in_i]), .empty_n(tail_general_b_out_b_in_empty_n[tail_general_b_out_b_in_i]), .read(tail_general_b_out_b_in_read[tail_general_b_out_b_in_i]));
    end
  endgenerate
  // family reorder_a_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] reorder_a_in_bind_din [0:7];
  wire [63:0] reorder_a_in_bind_dout [0:7];
  wire reorder_a_in_bind_full_n [0:7];
  wire reorder_a_in_bind_write [0:7];
  wire reorder_a_in_bind_empty_n [0:7];
  wire reorder_a_in_bind_read [0:7];
  genvar reorder_a_in_bind_i;
  generate
    for (reorder_a_in_bind_i = 0; reorder_a_in_bind_i < 8; reorder_a_in_bind_i = reorder_a_in_bind_i + 1) begin : g_reorder_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(reorder_a_in_bind_din[reorder_a_in_bind_i]), .full_n(reorder_a_in_bind_full_n[reorder_a_in_bind_i]), .write(reorder_a_in_bind_write[reorder_a_in_bind_i]), .dout(reorder_a_in_bind_dout[reorder_a_in_bind_i]), .empty_n(reorder_a_in_bind_empty_n[reorder_a_in_bind_i]), .read(reorder_a_in_bind_read[reorder_a_in_bind_i]));
    end
  endgenerate
  // family reorder_b_in_bind: 8 channel(s), 64-bit, depth 8
  wire [63:0] reorder_b_in_bind_din [0:7];
  wire [63:0] reorder_b_in_bind_dout [0:7];
  wire reorder_b_in_bind_full_n [0:7];
  wire reorder_b_in_bind_write [0:7];
  wire reorder_b_in_bind_empty_n [0:7];
  wire reorder_b_in_bind_read [0:7];
  genvar reorder_b_in_bind_i;
  generate
    for (reorder_b_in_bind_i = 0; reorder_b_in_bind_i < 8; reorder_b_in_bind_i = reorder_b_in_bind_i + 1) begin : g_reorder_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(reorder_b_in_bind_din[reorder_b_in_bind_i]), .full_n(reorder_b_in_bind_full_n[reorder_b_in_bind_i]), .write(reorder_b_in_bind_write[reorder_b_in_bind_i]), .dout(reorder_b_in_bind_dout[reorder_b_in_bind_i]), .empty_n(reorder_b_in_bind_empty_n[reorder_b_in_bind_i]), .read(reorder_b_in_bind_read[reorder_b_in_bind_i]));
    end
  endgenerate
  // role perm0_r0: 8 instance(s)
  perm0_r0 u_perm0_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm0_u_in_bind_dout[0]),
      .u_in_empty_n(perm0_u_in_bind_empty_n[0]),
      .u_in_read(perm0_u_in_bind_read[0]),
      .u_out_din(bfly0_a_in_bind_din[0]),
      .u_out_full_n(bfly0_a_in_bind_full_n[0]),
      .u_out_write(bfly0_a_in_bind_write[0]),
      .v_in_dout(perm0_v_in_bind_dout[0]),
      .v_in_empty_n(perm0_v_in_bind_empty_n[0]),
      .v_in_read(perm0_v_in_bind_read[0]),
      .v_out_din(bfly0_b_in_bind_din[0]),
      .v_out_full_n(bfly0_b_in_bind_full_n[0]),
      .v_out_write(bfly0_b_in_bind_write[0]));
  perm0_r0 u_perm0_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm0_u_in_bind_dout[1]),
      .u_in_empty_n(perm0_u_in_bind_empty_n[1]),
      .u_in_read(perm0_u_in_bind_read[1]),
      .u_out_din(bfly0_a_in_bind_din[1]),
      .u_out_full_n(bfly0_a_in_bind_full_n[1]),
      .u_out_write(bfly0_a_in_bind_write[1]),
      .v_in_dout(perm0_v_in_bind_dout[1]),
      .v_in_empty_n(perm0_v_in_bind_empty_n[1]),
      .v_in_read(perm0_v_in_bind_read[1]),
      .v_out_din(bfly0_b_in_bind_din[1]),
      .v_out_full_n(bfly0_b_in_bind_full_n[1]),
      .v_out_write(bfly0_b_in_bind_write[1]));
  perm0_r0 u_perm0_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm0_u_in_bind_dout[2]),
      .u_in_empty_n(perm0_u_in_bind_empty_n[2]),
      .u_in_read(perm0_u_in_bind_read[2]),
      .u_out_din(bfly0_a_in_bind_din[2]),
      .u_out_full_n(bfly0_a_in_bind_full_n[2]),
      .u_out_write(bfly0_a_in_bind_write[2]),
      .v_in_dout(perm0_v_in_bind_dout[2]),
      .v_in_empty_n(perm0_v_in_bind_empty_n[2]),
      .v_in_read(perm0_v_in_bind_read[2]),
      .v_out_din(bfly0_b_in_bind_din[2]),
      .v_out_full_n(bfly0_b_in_bind_full_n[2]),
      .v_out_write(bfly0_b_in_bind_write[2]));
  perm0_r0 u_perm0_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm0_u_in_bind_dout[3]),
      .u_in_empty_n(perm0_u_in_bind_empty_n[3]),
      .u_in_read(perm0_u_in_bind_read[3]),
      .u_out_din(bfly0_a_in_bind_din[3]),
      .u_out_full_n(bfly0_a_in_bind_full_n[3]),
      .u_out_write(bfly0_a_in_bind_write[3]),
      .v_in_dout(perm0_v_in_bind_dout[3]),
      .v_in_empty_n(perm0_v_in_bind_empty_n[3]),
      .v_in_read(perm0_v_in_bind_read[3]),
      .v_out_din(bfly0_b_in_bind_din[3]),
      .v_out_full_n(bfly0_b_in_bind_full_n[3]),
      .v_out_write(bfly0_b_in_bind_write[3]));
  perm0_r0 u_perm0_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm0_u_in_bind_dout[4]),
      .u_in_empty_n(perm0_u_in_bind_empty_n[4]),
      .u_in_read(perm0_u_in_bind_read[4]),
      .u_out_din(bfly0_a_in_bind_din[4]),
      .u_out_full_n(bfly0_a_in_bind_full_n[4]),
      .u_out_write(bfly0_a_in_bind_write[4]),
      .v_in_dout(perm0_v_in_bind_dout[4]),
      .v_in_empty_n(perm0_v_in_bind_empty_n[4]),
      .v_in_read(perm0_v_in_bind_read[4]),
      .v_out_din(bfly0_b_in_bind_din[4]),
      .v_out_full_n(bfly0_b_in_bind_full_n[4]),
      .v_out_write(bfly0_b_in_bind_write[4]));
  perm0_r0 u_perm0_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm0_u_in_bind_dout[5]),
      .u_in_empty_n(perm0_u_in_bind_empty_n[5]),
      .u_in_read(perm0_u_in_bind_read[5]),
      .u_out_din(bfly0_a_in_bind_din[5]),
      .u_out_full_n(bfly0_a_in_bind_full_n[5]),
      .u_out_write(bfly0_a_in_bind_write[5]),
      .v_in_dout(perm0_v_in_bind_dout[5]),
      .v_in_empty_n(perm0_v_in_bind_empty_n[5]),
      .v_in_read(perm0_v_in_bind_read[5]),
      .v_out_din(bfly0_b_in_bind_din[5]),
      .v_out_full_n(bfly0_b_in_bind_full_n[5]),
      .v_out_write(bfly0_b_in_bind_write[5]));
  perm0_r0 u_perm0_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm0_u_in_bind_dout[6]),
      .u_in_empty_n(perm0_u_in_bind_empty_n[6]),
      .u_in_read(perm0_u_in_bind_read[6]),
      .u_out_din(bfly0_a_in_bind_din[6]),
      .u_out_full_n(bfly0_a_in_bind_full_n[6]),
      .u_out_write(bfly0_a_in_bind_write[6]),
      .v_in_dout(perm0_v_in_bind_dout[6]),
      .v_in_empty_n(perm0_v_in_bind_empty_n[6]),
      .v_in_read(perm0_v_in_bind_read[6]),
      .v_out_din(bfly0_b_in_bind_din[6]),
      .v_out_full_n(bfly0_b_in_bind_full_n[6]),
      .v_out_write(bfly0_b_in_bind_write[6]));
  perm0_r0 u_perm0_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm0_u_in_bind_dout[7]),
      .u_in_empty_n(perm0_u_in_bind_empty_n[7]),
      .u_in_read(perm0_u_in_bind_read[7]),
      .u_out_din(bfly0_a_in_bind_din[7]),
      .u_out_full_n(bfly0_a_in_bind_full_n[7]),
      .u_out_write(bfly0_a_in_bind_write[7]),
      .v_in_dout(perm0_v_in_bind_dout[7]),
      .v_in_empty_n(perm0_v_in_bind_empty_n[7]),
      .v_in_read(perm0_v_in_bind_read[7]),
      .v_out_din(bfly0_b_in_bind_din[7]),
      .v_out_full_n(bfly0_b_in_bind_full_n[7]),
      .v_out_write(bfly0_b_in_bind_write[7]));
  // role perm1_r0: 8 instance(s)
  perm1_r0 u_perm1_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm1_u_in_bind_dout[0]),
      .u_in_empty_n(perm1_u_in_bind_empty_n[0]),
      .u_in_read(perm1_u_in_bind_read[0]),
      .u_out_din(bfly1_a_in_bind_din[0]),
      .u_out_full_n(bfly1_a_in_bind_full_n[0]),
      .u_out_write(bfly1_a_in_bind_write[0]),
      .v_in_dout(perm1_v_in_bind_dout[0]),
      .v_in_empty_n(perm1_v_in_bind_empty_n[0]),
      .v_in_read(perm1_v_in_bind_read[0]),
      .v_out_din(bfly1_b_in_bind_din[0]),
      .v_out_full_n(bfly1_b_in_bind_full_n[0]),
      .v_out_write(bfly1_b_in_bind_write[0]));
  perm1_r0 u_perm1_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm1_u_in_bind_dout[1]),
      .u_in_empty_n(perm1_u_in_bind_empty_n[1]),
      .u_in_read(perm1_u_in_bind_read[1]),
      .u_out_din(bfly1_a_in_bind_din[1]),
      .u_out_full_n(bfly1_a_in_bind_full_n[1]),
      .u_out_write(bfly1_a_in_bind_write[1]),
      .v_in_dout(perm1_v_in_bind_dout[1]),
      .v_in_empty_n(perm1_v_in_bind_empty_n[1]),
      .v_in_read(perm1_v_in_bind_read[1]),
      .v_out_din(bfly1_b_in_bind_din[1]),
      .v_out_full_n(bfly1_b_in_bind_full_n[1]),
      .v_out_write(bfly1_b_in_bind_write[1]));
  perm1_r0 u_perm1_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm1_u_in_bind_dout[2]),
      .u_in_empty_n(perm1_u_in_bind_empty_n[2]),
      .u_in_read(perm1_u_in_bind_read[2]),
      .u_out_din(bfly1_a_in_bind_din[2]),
      .u_out_full_n(bfly1_a_in_bind_full_n[2]),
      .u_out_write(bfly1_a_in_bind_write[2]),
      .v_in_dout(perm1_v_in_bind_dout[2]),
      .v_in_empty_n(perm1_v_in_bind_empty_n[2]),
      .v_in_read(perm1_v_in_bind_read[2]),
      .v_out_din(bfly1_b_in_bind_din[2]),
      .v_out_full_n(bfly1_b_in_bind_full_n[2]),
      .v_out_write(bfly1_b_in_bind_write[2]));
  perm1_r0 u_perm1_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm1_u_in_bind_dout[3]),
      .u_in_empty_n(perm1_u_in_bind_empty_n[3]),
      .u_in_read(perm1_u_in_bind_read[3]),
      .u_out_din(bfly1_a_in_bind_din[3]),
      .u_out_full_n(bfly1_a_in_bind_full_n[3]),
      .u_out_write(bfly1_a_in_bind_write[3]),
      .v_in_dout(perm1_v_in_bind_dout[3]),
      .v_in_empty_n(perm1_v_in_bind_empty_n[3]),
      .v_in_read(perm1_v_in_bind_read[3]),
      .v_out_din(bfly1_b_in_bind_din[3]),
      .v_out_full_n(bfly1_b_in_bind_full_n[3]),
      .v_out_write(bfly1_b_in_bind_write[3]));
  perm1_r0 u_perm1_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm1_u_in_bind_dout[4]),
      .u_in_empty_n(perm1_u_in_bind_empty_n[4]),
      .u_in_read(perm1_u_in_bind_read[4]),
      .u_out_din(bfly1_a_in_bind_din[4]),
      .u_out_full_n(bfly1_a_in_bind_full_n[4]),
      .u_out_write(bfly1_a_in_bind_write[4]),
      .v_in_dout(perm1_v_in_bind_dout[4]),
      .v_in_empty_n(perm1_v_in_bind_empty_n[4]),
      .v_in_read(perm1_v_in_bind_read[4]),
      .v_out_din(bfly1_b_in_bind_din[4]),
      .v_out_full_n(bfly1_b_in_bind_full_n[4]),
      .v_out_write(bfly1_b_in_bind_write[4]));
  perm1_r0 u_perm1_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm1_u_in_bind_dout[5]),
      .u_in_empty_n(perm1_u_in_bind_empty_n[5]),
      .u_in_read(perm1_u_in_bind_read[5]),
      .u_out_din(bfly1_a_in_bind_din[5]),
      .u_out_full_n(bfly1_a_in_bind_full_n[5]),
      .u_out_write(bfly1_a_in_bind_write[5]),
      .v_in_dout(perm1_v_in_bind_dout[5]),
      .v_in_empty_n(perm1_v_in_bind_empty_n[5]),
      .v_in_read(perm1_v_in_bind_read[5]),
      .v_out_din(bfly1_b_in_bind_din[5]),
      .v_out_full_n(bfly1_b_in_bind_full_n[5]),
      .v_out_write(bfly1_b_in_bind_write[5]));
  perm1_r0 u_perm1_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm1_u_in_bind_dout[6]),
      .u_in_empty_n(perm1_u_in_bind_empty_n[6]),
      .u_in_read(perm1_u_in_bind_read[6]),
      .u_out_din(bfly1_a_in_bind_din[6]),
      .u_out_full_n(bfly1_a_in_bind_full_n[6]),
      .u_out_write(bfly1_a_in_bind_write[6]),
      .v_in_dout(perm1_v_in_bind_dout[6]),
      .v_in_empty_n(perm1_v_in_bind_empty_n[6]),
      .v_in_read(perm1_v_in_bind_read[6]),
      .v_out_din(bfly1_b_in_bind_din[6]),
      .v_out_full_n(bfly1_b_in_bind_full_n[6]),
      .v_out_write(bfly1_b_in_bind_write[6]));
  perm1_r0 u_perm1_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm1_u_in_bind_dout[7]),
      .u_in_empty_n(perm1_u_in_bind_empty_n[7]),
      .u_in_read(perm1_u_in_bind_read[7]),
      .u_out_din(bfly1_a_in_bind_din[7]),
      .u_out_full_n(bfly1_a_in_bind_full_n[7]),
      .u_out_write(bfly1_a_in_bind_write[7]),
      .v_in_dout(perm1_v_in_bind_dout[7]),
      .v_in_empty_n(perm1_v_in_bind_empty_n[7]),
      .v_in_read(perm1_v_in_bind_read[7]),
      .v_out_din(bfly1_b_in_bind_din[7]),
      .v_out_full_n(bfly1_b_in_bind_full_n[7]),
      .v_out_write(bfly1_b_in_bind_write[7]));
  // role perm2_r0: 8 instance(s)
  perm2_r0 u_perm2_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm2_u_in_bind_dout[0]),
      .u_in_empty_n(perm2_u_in_bind_empty_n[0]),
      .u_in_read(perm2_u_in_bind_read[0]),
      .u_out_din(bfly2_a_in_bind_din[0]),
      .u_out_full_n(bfly2_a_in_bind_full_n[0]),
      .u_out_write(bfly2_a_in_bind_write[0]),
      .v_in_dout(perm2_v_in_bind_dout[0]),
      .v_in_empty_n(perm2_v_in_bind_empty_n[0]),
      .v_in_read(perm2_v_in_bind_read[0]),
      .v_out_din(bfly2_b_in_bind_din[0]),
      .v_out_full_n(bfly2_b_in_bind_full_n[0]),
      .v_out_write(bfly2_b_in_bind_write[0]));
  perm2_r0 u_perm2_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm2_u_in_bind_dout[1]),
      .u_in_empty_n(perm2_u_in_bind_empty_n[1]),
      .u_in_read(perm2_u_in_bind_read[1]),
      .u_out_din(bfly2_a_in_bind_din[1]),
      .u_out_full_n(bfly2_a_in_bind_full_n[1]),
      .u_out_write(bfly2_a_in_bind_write[1]),
      .v_in_dout(perm2_v_in_bind_dout[1]),
      .v_in_empty_n(perm2_v_in_bind_empty_n[1]),
      .v_in_read(perm2_v_in_bind_read[1]),
      .v_out_din(bfly2_b_in_bind_din[1]),
      .v_out_full_n(bfly2_b_in_bind_full_n[1]),
      .v_out_write(bfly2_b_in_bind_write[1]));
  perm2_r0 u_perm2_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm2_u_in_bind_dout[2]),
      .u_in_empty_n(perm2_u_in_bind_empty_n[2]),
      .u_in_read(perm2_u_in_bind_read[2]),
      .u_out_din(bfly2_a_in_bind_din[2]),
      .u_out_full_n(bfly2_a_in_bind_full_n[2]),
      .u_out_write(bfly2_a_in_bind_write[2]),
      .v_in_dout(perm2_v_in_bind_dout[2]),
      .v_in_empty_n(perm2_v_in_bind_empty_n[2]),
      .v_in_read(perm2_v_in_bind_read[2]),
      .v_out_din(bfly2_b_in_bind_din[2]),
      .v_out_full_n(bfly2_b_in_bind_full_n[2]),
      .v_out_write(bfly2_b_in_bind_write[2]));
  perm2_r0 u_perm2_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm2_u_in_bind_dout[3]),
      .u_in_empty_n(perm2_u_in_bind_empty_n[3]),
      .u_in_read(perm2_u_in_bind_read[3]),
      .u_out_din(bfly2_a_in_bind_din[3]),
      .u_out_full_n(bfly2_a_in_bind_full_n[3]),
      .u_out_write(bfly2_a_in_bind_write[3]),
      .v_in_dout(perm2_v_in_bind_dout[3]),
      .v_in_empty_n(perm2_v_in_bind_empty_n[3]),
      .v_in_read(perm2_v_in_bind_read[3]),
      .v_out_din(bfly2_b_in_bind_din[3]),
      .v_out_full_n(bfly2_b_in_bind_full_n[3]),
      .v_out_write(bfly2_b_in_bind_write[3]));
  perm2_r0 u_perm2_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm2_u_in_bind_dout[4]),
      .u_in_empty_n(perm2_u_in_bind_empty_n[4]),
      .u_in_read(perm2_u_in_bind_read[4]),
      .u_out_din(bfly2_a_in_bind_din[4]),
      .u_out_full_n(bfly2_a_in_bind_full_n[4]),
      .u_out_write(bfly2_a_in_bind_write[4]),
      .v_in_dout(perm2_v_in_bind_dout[4]),
      .v_in_empty_n(perm2_v_in_bind_empty_n[4]),
      .v_in_read(perm2_v_in_bind_read[4]),
      .v_out_din(bfly2_b_in_bind_din[4]),
      .v_out_full_n(bfly2_b_in_bind_full_n[4]),
      .v_out_write(bfly2_b_in_bind_write[4]));
  perm2_r0 u_perm2_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm2_u_in_bind_dout[5]),
      .u_in_empty_n(perm2_u_in_bind_empty_n[5]),
      .u_in_read(perm2_u_in_bind_read[5]),
      .u_out_din(bfly2_a_in_bind_din[5]),
      .u_out_full_n(bfly2_a_in_bind_full_n[5]),
      .u_out_write(bfly2_a_in_bind_write[5]),
      .v_in_dout(perm2_v_in_bind_dout[5]),
      .v_in_empty_n(perm2_v_in_bind_empty_n[5]),
      .v_in_read(perm2_v_in_bind_read[5]),
      .v_out_din(bfly2_b_in_bind_din[5]),
      .v_out_full_n(bfly2_b_in_bind_full_n[5]),
      .v_out_write(bfly2_b_in_bind_write[5]));
  perm2_r0 u_perm2_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm2_u_in_bind_dout[6]),
      .u_in_empty_n(perm2_u_in_bind_empty_n[6]),
      .u_in_read(perm2_u_in_bind_read[6]),
      .u_out_din(bfly2_a_in_bind_din[6]),
      .u_out_full_n(bfly2_a_in_bind_full_n[6]),
      .u_out_write(bfly2_a_in_bind_write[6]),
      .v_in_dout(perm2_v_in_bind_dout[6]),
      .v_in_empty_n(perm2_v_in_bind_empty_n[6]),
      .v_in_read(perm2_v_in_bind_read[6]),
      .v_out_din(bfly2_b_in_bind_din[6]),
      .v_out_full_n(bfly2_b_in_bind_full_n[6]),
      .v_out_write(bfly2_b_in_bind_write[6]));
  perm2_r0 u_perm2_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm2_u_in_bind_dout[7]),
      .u_in_empty_n(perm2_u_in_bind_empty_n[7]),
      .u_in_read(perm2_u_in_bind_read[7]),
      .u_out_din(bfly2_a_in_bind_din[7]),
      .u_out_full_n(bfly2_a_in_bind_full_n[7]),
      .u_out_write(bfly2_a_in_bind_write[7]),
      .v_in_dout(perm2_v_in_bind_dout[7]),
      .v_in_empty_n(perm2_v_in_bind_empty_n[7]),
      .v_in_read(perm2_v_in_bind_read[7]),
      .v_out_din(bfly2_b_in_bind_din[7]),
      .v_out_full_n(bfly2_b_in_bind_full_n[7]),
      .v_out_write(bfly2_b_in_bind_write[7]));
  // role perm3_r0: 8 instance(s)
  perm3_r0 u_perm3_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm3_u_in_bind_dout[0]),
      .u_in_empty_n(perm3_u_in_bind_empty_n[0]),
      .u_in_read(perm3_u_in_bind_read[0]),
      .u_out_din(bfly3_a_in_bind_din[0]),
      .u_out_full_n(bfly3_a_in_bind_full_n[0]),
      .u_out_write(bfly3_a_in_bind_write[0]),
      .v_in_dout(perm3_v_in_bind_dout[0]),
      .v_in_empty_n(perm3_v_in_bind_empty_n[0]),
      .v_in_read(perm3_v_in_bind_read[0]),
      .v_out_din(bfly3_b_in_bind_din[0]),
      .v_out_full_n(bfly3_b_in_bind_full_n[0]),
      .v_out_write(bfly3_b_in_bind_write[0]));
  perm3_r0 u_perm3_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm3_u_in_bind_dout[1]),
      .u_in_empty_n(perm3_u_in_bind_empty_n[1]),
      .u_in_read(perm3_u_in_bind_read[1]),
      .u_out_din(bfly3_a_in_bind_din[1]),
      .u_out_full_n(bfly3_a_in_bind_full_n[1]),
      .u_out_write(bfly3_a_in_bind_write[1]),
      .v_in_dout(perm3_v_in_bind_dout[1]),
      .v_in_empty_n(perm3_v_in_bind_empty_n[1]),
      .v_in_read(perm3_v_in_bind_read[1]),
      .v_out_din(bfly3_b_in_bind_din[1]),
      .v_out_full_n(bfly3_b_in_bind_full_n[1]),
      .v_out_write(bfly3_b_in_bind_write[1]));
  perm3_r0 u_perm3_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm3_u_in_bind_dout[2]),
      .u_in_empty_n(perm3_u_in_bind_empty_n[2]),
      .u_in_read(perm3_u_in_bind_read[2]),
      .u_out_din(bfly3_a_in_bind_din[2]),
      .u_out_full_n(bfly3_a_in_bind_full_n[2]),
      .u_out_write(bfly3_a_in_bind_write[2]),
      .v_in_dout(perm3_v_in_bind_dout[2]),
      .v_in_empty_n(perm3_v_in_bind_empty_n[2]),
      .v_in_read(perm3_v_in_bind_read[2]),
      .v_out_din(bfly3_b_in_bind_din[2]),
      .v_out_full_n(bfly3_b_in_bind_full_n[2]),
      .v_out_write(bfly3_b_in_bind_write[2]));
  perm3_r0 u_perm3_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm3_u_in_bind_dout[3]),
      .u_in_empty_n(perm3_u_in_bind_empty_n[3]),
      .u_in_read(perm3_u_in_bind_read[3]),
      .u_out_din(bfly3_a_in_bind_din[3]),
      .u_out_full_n(bfly3_a_in_bind_full_n[3]),
      .u_out_write(bfly3_a_in_bind_write[3]),
      .v_in_dout(perm3_v_in_bind_dout[3]),
      .v_in_empty_n(perm3_v_in_bind_empty_n[3]),
      .v_in_read(perm3_v_in_bind_read[3]),
      .v_out_din(bfly3_b_in_bind_din[3]),
      .v_out_full_n(bfly3_b_in_bind_full_n[3]),
      .v_out_write(bfly3_b_in_bind_write[3]));
  perm3_r0 u_perm3_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm3_u_in_bind_dout[4]),
      .u_in_empty_n(perm3_u_in_bind_empty_n[4]),
      .u_in_read(perm3_u_in_bind_read[4]),
      .u_out_din(bfly3_a_in_bind_din[4]),
      .u_out_full_n(bfly3_a_in_bind_full_n[4]),
      .u_out_write(bfly3_a_in_bind_write[4]),
      .v_in_dout(perm3_v_in_bind_dout[4]),
      .v_in_empty_n(perm3_v_in_bind_empty_n[4]),
      .v_in_read(perm3_v_in_bind_read[4]),
      .v_out_din(bfly3_b_in_bind_din[4]),
      .v_out_full_n(bfly3_b_in_bind_full_n[4]),
      .v_out_write(bfly3_b_in_bind_write[4]));
  perm3_r0 u_perm3_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm3_u_in_bind_dout[5]),
      .u_in_empty_n(perm3_u_in_bind_empty_n[5]),
      .u_in_read(perm3_u_in_bind_read[5]),
      .u_out_din(bfly3_a_in_bind_din[5]),
      .u_out_full_n(bfly3_a_in_bind_full_n[5]),
      .u_out_write(bfly3_a_in_bind_write[5]),
      .v_in_dout(perm3_v_in_bind_dout[5]),
      .v_in_empty_n(perm3_v_in_bind_empty_n[5]),
      .v_in_read(perm3_v_in_bind_read[5]),
      .v_out_din(bfly3_b_in_bind_din[5]),
      .v_out_full_n(bfly3_b_in_bind_full_n[5]),
      .v_out_write(bfly3_b_in_bind_write[5]));
  perm3_r0 u_perm3_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm3_u_in_bind_dout[6]),
      .u_in_empty_n(perm3_u_in_bind_empty_n[6]),
      .u_in_read(perm3_u_in_bind_read[6]),
      .u_out_din(bfly3_a_in_bind_din[6]),
      .u_out_full_n(bfly3_a_in_bind_full_n[6]),
      .u_out_write(bfly3_a_in_bind_write[6]),
      .v_in_dout(perm3_v_in_bind_dout[6]),
      .v_in_empty_n(perm3_v_in_bind_empty_n[6]),
      .v_in_read(perm3_v_in_bind_read[6]),
      .v_out_din(bfly3_b_in_bind_din[6]),
      .v_out_full_n(bfly3_b_in_bind_full_n[6]),
      .v_out_write(bfly3_b_in_bind_write[6]));
  perm3_r0 u_perm3_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm3_u_in_bind_dout[7]),
      .u_in_empty_n(perm3_u_in_bind_empty_n[7]),
      .u_in_read(perm3_u_in_bind_read[7]),
      .u_out_din(bfly3_a_in_bind_din[7]),
      .u_out_full_n(bfly3_a_in_bind_full_n[7]),
      .u_out_write(bfly3_a_in_bind_write[7]),
      .v_in_dout(perm3_v_in_bind_dout[7]),
      .v_in_empty_n(perm3_v_in_bind_empty_n[7]),
      .v_in_read(perm3_v_in_bind_read[7]),
      .v_out_din(bfly3_b_in_bind_din[7]),
      .v_out_full_n(bfly3_b_in_bind_full_n[7]),
      .v_out_write(bfly3_b_in_bind_write[7]));
  // role perm4_r0: 8 instance(s)
  perm4_r0 u_perm4_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm4_u_in_bind_dout[0]),
      .u_in_empty_n(perm4_u_in_bind_empty_n[0]),
      .u_in_read(perm4_u_in_bind_read[0]),
      .u_out_din(bfly4_a_in_bind_din[0]),
      .u_out_full_n(bfly4_a_in_bind_full_n[0]),
      .u_out_write(bfly4_a_in_bind_write[0]),
      .v_in_dout(perm4_v_in_bind_dout[0]),
      .v_in_empty_n(perm4_v_in_bind_empty_n[0]),
      .v_in_read(perm4_v_in_bind_read[0]),
      .v_out_din(bfly4_b_in_bind_din[0]),
      .v_out_full_n(bfly4_b_in_bind_full_n[0]),
      .v_out_write(bfly4_b_in_bind_write[0]));
  perm4_r0 u_perm4_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm4_u_in_bind_dout[1]),
      .u_in_empty_n(perm4_u_in_bind_empty_n[1]),
      .u_in_read(perm4_u_in_bind_read[1]),
      .u_out_din(bfly4_a_in_bind_din[1]),
      .u_out_full_n(bfly4_a_in_bind_full_n[1]),
      .u_out_write(bfly4_a_in_bind_write[1]),
      .v_in_dout(perm4_v_in_bind_dout[1]),
      .v_in_empty_n(perm4_v_in_bind_empty_n[1]),
      .v_in_read(perm4_v_in_bind_read[1]),
      .v_out_din(bfly4_b_in_bind_din[1]),
      .v_out_full_n(bfly4_b_in_bind_full_n[1]),
      .v_out_write(bfly4_b_in_bind_write[1]));
  perm4_r0 u_perm4_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm4_u_in_bind_dout[2]),
      .u_in_empty_n(perm4_u_in_bind_empty_n[2]),
      .u_in_read(perm4_u_in_bind_read[2]),
      .u_out_din(bfly4_a_in_bind_din[2]),
      .u_out_full_n(bfly4_a_in_bind_full_n[2]),
      .u_out_write(bfly4_a_in_bind_write[2]),
      .v_in_dout(perm4_v_in_bind_dout[2]),
      .v_in_empty_n(perm4_v_in_bind_empty_n[2]),
      .v_in_read(perm4_v_in_bind_read[2]),
      .v_out_din(bfly4_b_in_bind_din[2]),
      .v_out_full_n(bfly4_b_in_bind_full_n[2]),
      .v_out_write(bfly4_b_in_bind_write[2]));
  perm4_r0 u_perm4_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm4_u_in_bind_dout[3]),
      .u_in_empty_n(perm4_u_in_bind_empty_n[3]),
      .u_in_read(perm4_u_in_bind_read[3]),
      .u_out_din(bfly4_a_in_bind_din[3]),
      .u_out_full_n(bfly4_a_in_bind_full_n[3]),
      .u_out_write(bfly4_a_in_bind_write[3]),
      .v_in_dout(perm4_v_in_bind_dout[3]),
      .v_in_empty_n(perm4_v_in_bind_empty_n[3]),
      .v_in_read(perm4_v_in_bind_read[3]),
      .v_out_din(bfly4_b_in_bind_din[3]),
      .v_out_full_n(bfly4_b_in_bind_full_n[3]),
      .v_out_write(bfly4_b_in_bind_write[3]));
  perm4_r0 u_perm4_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm4_u_in_bind_dout[4]),
      .u_in_empty_n(perm4_u_in_bind_empty_n[4]),
      .u_in_read(perm4_u_in_bind_read[4]),
      .u_out_din(bfly4_a_in_bind_din[4]),
      .u_out_full_n(bfly4_a_in_bind_full_n[4]),
      .u_out_write(bfly4_a_in_bind_write[4]),
      .v_in_dout(perm4_v_in_bind_dout[4]),
      .v_in_empty_n(perm4_v_in_bind_empty_n[4]),
      .v_in_read(perm4_v_in_bind_read[4]),
      .v_out_din(bfly4_b_in_bind_din[4]),
      .v_out_full_n(bfly4_b_in_bind_full_n[4]),
      .v_out_write(bfly4_b_in_bind_write[4]));
  perm4_r0 u_perm4_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm4_u_in_bind_dout[5]),
      .u_in_empty_n(perm4_u_in_bind_empty_n[5]),
      .u_in_read(perm4_u_in_bind_read[5]),
      .u_out_din(bfly4_a_in_bind_din[5]),
      .u_out_full_n(bfly4_a_in_bind_full_n[5]),
      .u_out_write(bfly4_a_in_bind_write[5]),
      .v_in_dout(perm4_v_in_bind_dout[5]),
      .v_in_empty_n(perm4_v_in_bind_empty_n[5]),
      .v_in_read(perm4_v_in_bind_read[5]),
      .v_out_din(bfly4_b_in_bind_din[5]),
      .v_out_full_n(bfly4_b_in_bind_full_n[5]),
      .v_out_write(bfly4_b_in_bind_write[5]));
  perm4_r0 u_perm4_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm4_u_in_bind_dout[6]),
      .u_in_empty_n(perm4_u_in_bind_empty_n[6]),
      .u_in_read(perm4_u_in_bind_read[6]),
      .u_out_din(bfly4_a_in_bind_din[6]),
      .u_out_full_n(bfly4_a_in_bind_full_n[6]),
      .u_out_write(bfly4_a_in_bind_write[6]),
      .v_in_dout(perm4_v_in_bind_dout[6]),
      .v_in_empty_n(perm4_v_in_bind_empty_n[6]),
      .v_in_read(perm4_v_in_bind_read[6]),
      .v_out_din(bfly4_b_in_bind_din[6]),
      .v_out_full_n(bfly4_b_in_bind_full_n[6]),
      .v_out_write(bfly4_b_in_bind_write[6]));
  perm4_r0 u_perm4_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm4_u_in_bind_dout[7]),
      .u_in_empty_n(perm4_u_in_bind_empty_n[7]),
      .u_in_read(perm4_u_in_bind_read[7]),
      .u_out_din(bfly4_a_in_bind_din[7]),
      .u_out_full_n(bfly4_a_in_bind_full_n[7]),
      .u_out_write(bfly4_a_in_bind_write[7]),
      .v_in_dout(perm4_v_in_bind_dout[7]),
      .v_in_empty_n(perm4_v_in_bind_empty_n[7]),
      .v_in_read(perm4_v_in_bind_read[7]),
      .v_out_din(bfly4_b_in_bind_din[7]),
      .v_out_full_n(bfly4_b_in_bind_full_n[7]),
      .v_out_write(bfly4_b_in_bind_write[7]));
  // role perm5_r0: 8 instance(s)
  perm5_r0 u_perm5_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm5_u_in_bind_dout[0]),
      .u_in_empty_n(perm5_u_in_bind_empty_n[0]),
      .u_in_read(perm5_u_in_bind_read[0]),
      .u_out_din(bfly5_a_in_bind_din[0]),
      .u_out_full_n(bfly5_a_in_bind_full_n[0]),
      .u_out_write(bfly5_a_in_bind_write[0]),
      .v_in_dout(perm5_v_in_bind_dout[0]),
      .v_in_empty_n(perm5_v_in_bind_empty_n[0]),
      .v_in_read(perm5_v_in_bind_read[0]),
      .v_out_din(bfly5_b_in_bind_din[0]),
      .v_out_full_n(bfly5_b_in_bind_full_n[0]),
      .v_out_write(bfly5_b_in_bind_write[0]));
  perm5_r0 u_perm5_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm5_u_in_bind_dout[1]),
      .u_in_empty_n(perm5_u_in_bind_empty_n[1]),
      .u_in_read(perm5_u_in_bind_read[1]),
      .u_out_din(bfly5_a_in_bind_din[1]),
      .u_out_full_n(bfly5_a_in_bind_full_n[1]),
      .u_out_write(bfly5_a_in_bind_write[1]),
      .v_in_dout(perm5_v_in_bind_dout[1]),
      .v_in_empty_n(perm5_v_in_bind_empty_n[1]),
      .v_in_read(perm5_v_in_bind_read[1]),
      .v_out_din(bfly5_b_in_bind_din[1]),
      .v_out_full_n(bfly5_b_in_bind_full_n[1]),
      .v_out_write(bfly5_b_in_bind_write[1]));
  perm5_r0 u_perm5_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm5_u_in_bind_dout[2]),
      .u_in_empty_n(perm5_u_in_bind_empty_n[2]),
      .u_in_read(perm5_u_in_bind_read[2]),
      .u_out_din(bfly5_a_in_bind_din[2]),
      .u_out_full_n(bfly5_a_in_bind_full_n[2]),
      .u_out_write(bfly5_a_in_bind_write[2]),
      .v_in_dout(perm5_v_in_bind_dout[2]),
      .v_in_empty_n(perm5_v_in_bind_empty_n[2]),
      .v_in_read(perm5_v_in_bind_read[2]),
      .v_out_din(bfly5_b_in_bind_din[2]),
      .v_out_full_n(bfly5_b_in_bind_full_n[2]),
      .v_out_write(bfly5_b_in_bind_write[2]));
  perm5_r0 u_perm5_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm5_u_in_bind_dout[3]),
      .u_in_empty_n(perm5_u_in_bind_empty_n[3]),
      .u_in_read(perm5_u_in_bind_read[3]),
      .u_out_din(bfly5_a_in_bind_din[3]),
      .u_out_full_n(bfly5_a_in_bind_full_n[3]),
      .u_out_write(bfly5_a_in_bind_write[3]),
      .v_in_dout(perm5_v_in_bind_dout[3]),
      .v_in_empty_n(perm5_v_in_bind_empty_n[3]),
      .v_in_read(perm5_v_in_bind_read[3]),
      .v_out_din(bfly5_b_in_bind_din[3]),
      .v_out_full_n(bfly5_b_in_bind_full_n[3]),
      .v_out_write(bfly5_b_in_bind_write[3]));
  perm5_r0 u_perm5_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm5_u_in_bind_dout[4]),
      .u_in_empty_n(perm5_u_in_bind_empty_n[4]),
      .u_in_read(perm5_u_in_bind_read[4]),
      .u_out_din(bfly5_a_in_bind_din[4]),
      .u_out_full_n(bfly5_a_in_bind_full_n[4]),
      .u_out_write(bfly5_a_in_bind_write[4]),
      .v_in_dout(perm5_v_in_bind_dout[4]),
      .v_in_empty_n(perm5_v_in_bind_empty_n[4]),
      .v_in_read(perm5_v_in_bind_read[4]),
      .v_out_din(bfly5_b_in_bind_din[4]),
      .v_out_full_n(bfly5_b_in_bind_full_n[4]),
      .v_out_write(bfly5_b_in_bind_write[4]));
  perm5_r0 u_perm5_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm5_u_in_bind_dout[5]),
      .u_in_empty_n(perm5_u_in_bind_empty_n[5]),
      .u_in_read(perm5_u_in_bind_read[5]),
      .u_out_din(bfly5_a_in_bind_din[5]),
      .u_out_full_n(bfly5_a_in_bind_full_n[5]),
      .u_out_write(bfly5_a_in_bind_write[5]),
      .v_in_dout(perm5_v_in_bind_dout[5]),
      .v_in_empty_n(perm5_v_in_bind_empty_n[5]),
      .v_in_read(perm5_v_in_bind_read[5]),
      .v_out_din(bfly5_b_in_bind_din[5]),
      .v_out_full_n(bfly5_b_in_bind_full_n[5]),
      .v_out_write(bfly5_b_in_bind_write[5]));
  perm5_r0 u_perm5_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm5_u_in_bind_dout[6]),
      .u_in_empty_n(perm5_u_in_bind_empty_n[6]),
      .u_in_read(perm5_u_in_bind_read[6]),
      .u_out_din(bfly5_a_in_bind_din[6]),
      .u_out_full_n(bfly5_a_in_bind_full_n[6]),
      .u_out_write(bfly5_a_in_bind_write[6]),
      .v_in_dout(perm5_v_in_bind_dout[6]),
      .v_in_empty_n(perm5_v_in_bind_empty_n[6]),
      .v_in_read(perm5_v_in_bind_read[6]),
      .v_out_din(bfly5_b_in_bind_din[6]),
      .v_out_full_n(bfly5_b_in_bind_full_n[6]),
      .v_out_write(bfly5_b_in_bind_write[6]));
  perm5_r0 u_perm5_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm5_u_in_bind_dout[7]),
      .u_in_empty_n(perm5_u_in_bind_empty_n[7]),
      .u_in_read(perm5_u_in_bind_read[7]),
      .u_out_din(bfly5_a_in_bind_din[7]),
      .u_out_full_n(bfly5_a_in_bind_full_n[7]),
      .u_out_write(bfly5_a_in_bind_write[7]),
      .v_in_dout(perm5_v_in_bind_dout[7]),
      .v_in_empty_n(perm5_v_in_bind_empty_n[7]),
      .v_in_read(perm5_v_in_bind_read[7]),
      .v_out_din(bfly5_b_in_bind_din[7]),
      .v_out_full_n(bfly5_b_in_bind_full_n[7]),
      .v_out_write(bfly5_b_in_bind_write[7]));
  // role perm6_r0: 8 instance(s)
  perm6_r0 u_perm6_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm6_u_in_bind_dout[0]),
      .u_in_empty_n(perm6_u_in_bind_empty_n[0]),
      .u_in_read(perm6_u_in_bind_read[0]),
      .u_out_din(tail_general_a_in_bind_din[0]),
      .u_out_full_n(tail_general_a_in_bind_full_n[0]),
      .u_out_write(tail_general_a_in_bind_write[0]),
      .v_in_dout(perm6_v_in_bind_dout[0]),
      .v_in_empty_n(perm6_v_in_bind_empty_n[0]),
      .v_in_read(perm6_v_in_bind_read[0]),
      .v_out_din(tail_general_b_in_bind_din[0]),
      .v_out_full_n(tail_general_b_in_bind_full_n[0]),
      .v_out_write(tail_general_b_in_bind_write[0]));
  perm6_r0 u_perm6_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm6_u_in_bind_dout[1]),
      .u_in_empty_n(perm6_u_in_bind_empty_n[1]),
      .u_in_read(perm6_u_in_bind_read[1]),
      .u_out_din(tail_general_a_in_bind_din[1]),
      .u_out_full_n(tail_general_a_in_bind_full_n[1]),
      .u_out_write(tail_general_a_in_bind_write[1]),
      .v_in_dout(perm6_v_in_bind_dout[1]),
      .v_in_empty_n(perm6_v_in_bind_empty_n[1]),
      .v_in_read(perm6_v_in_bind_read[1]),
      .v_out_din(tail_general_b_in_bind_din[1]),
      .v_out_full_n(tail_general_b_in_bind_full_n[1]),
      .v_out_write(tail_general_b_in_bind_write[1]));
  perm6_r0 u_perm6_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm6_u_in_bind_dout[2]),
      .u_in_empty_n(perm6_u_in_bind_empty_n[2]),
      .u_in_read(perm6_u_in_bind_read[2]),
      .u_out_din(tail_general_a_in_bind_din[2]),
      .u_out_full_n(tail_general_a_in_bind_full_n[2]),
      .u_out_write(tail_general_a_in_bind_write[2]),
      .v_in_dout(perm6_v_in_bind_dout[2]),
      .v_in_empty_n(perm6_v_in_bind_empty_n[2]),
      .v_in_read(perm6_v_in_bind_read[2]),
      .v_out_din(tail_general_b_in_bind_din[2]),
      .v_out_full_n(tail_general_b_in_bind_full_n[2]),
      .v_out_write(tail_general_b_in_bind_write[2]));
  perm6_r0 u_perm6_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm6_u_in_bind_dout[3]),
      .u_in_empty_n(perm6_u_in_bind_empty_n[3]),
      .u_in_read(perm6_u_in_bind_read[3]),
      .u_out_din(tail_general_a_in_bind_din[3]),
      .u_out_full_n(tail_general_a_in_bind_full_n[3]),
      .u_out_write(tail_general_a_in_bind_write[3]),
      .v_in_dout(perm6_v_in_bind_dout[3]),
      .v_in_empty_n(perm6_v_in_bind_empty_n[3]),
      .v_in_read(perm6_v_in_bind_read[3]),
      .v_out_din(tail_general_b_in_bind_din[3]),
      .v_out_full_n(tail_general_b_in_bind_full_n[3]),
      .v_out_write(tail_general_b_in_bind_write[3]));
  perm6_r0 u_perm6_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm6_u_in_bind_dout[4]),
      .u_in_empty_n(perm6_u_in_bind_empty_n[4]),
      .u_in_read(perm6_u_in_bind_read[4]),
      .u_out_din(tail_general_a_in_bind_din[4]),
      .u_out_full_n(tail_general_a_in_bind_full_n[4]),
      .u_out_write(tail_general_a_in_bind_write[4]),
      .v_in_dout(perm6_v_in_bind_dout[4]),
      .v_in_empty_n(perm6_v_in_bind_empty_n[4]),
      .v_in_read(perm6_v_in_bind_read[4]),
      .v_out_din(tail_general_b_in_bind_din[4]),
      .v_out_full_n(tail_general_b_in_bind_full_n[4]),
      .v_out_write(tail_general_b_in_bind_write[4]));
  perm6_r0 u_perm6_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm6_u_in_bind_dout[5]),
      .u_in_empty_n(perm6_u_in_bind_empty_n[5]),
      .u_in_read(perm6_u_in_bind_read[5]),
      .u_out_din(tail_general_a_in_bind_din[5]),
      .u_out_full_n(tail_general_a_in_bind_full_n[5]),
      .u_out_write(tail_general_a_in_bind_write[5]),
      .v_in_dout(perm6_v_in_bind_dout[5]),
      .v_in_empty_n(perm6_v_in_bind_empty_n[5]),
      .v_in_read(perm6_v_in_bind_read[5]),
      .v_out_din(tail_general_b_in_bind_din[5]),
      .v_out_full_n(tail_general_b_in_bind_full_n[5]),
      .v_out_write(tail_general_b_in_bind_write[5]));
  perm6_r0 u_perm6_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm6_u_in_bind_dout[6]),
      .u_in_empty_n(perm6_u_in_bind_empty_n[6]),
      .u_in_read(perm6_u_in_bind_read[6]),
      .u_out_din(tail_general_a_in_bind_din[6]),
      .u_out_full_n(tail_general_a_in_bind_full_n[6]),
      .u_out_write(tail_general_a_in_bind_write[6]),
      .v_in_dout(perm6_v_in_bind_dout[6]),
      .v_in_empty_n(perm6_v_in_bind_empty_n[6]),
      .v_in_read(perm6_v_in_bind_read[6]),
      .v_out_din(tail_general_b_in_bind_din[6]),
      .v_out_full_n(tail_general_b_in_bind_full_n[6]),
      .v_out_write(tail_general_b_in_bind_write[6]));
  perm6_r0 u_perm6_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm6_u_in_bind_dout[7]),
      .u_in_empty_n(perm6_u_in_bind_empty_n[7]),
      .u_in_read(perm6_u_in_bind_read[7]),
      .u_out_din(tail_general_a_in_bind_din[7]),
      .u_out_full_n(tail_general_a_in_bind_full_n[7]),
      .u_out_write(tail_general_a_in_bind_write[7]),
      .v_in_dout(perm6_v_in_bind_dout[7]),
      .v_in_empty_n(perm6_v_in_bind_empty_n[7]),
      .v_in_read(perm6_v_in_bind_read[7]),
      .v_out_din(tail_general_b_in_bind_din[7]),
      .v_out_full_n(tail_general_b_in_bind_full_n[7]),
      .v_out_write(tail_general_b_in_bind_write[7]));
  // coordinate axis 0: 8 constant source(s)
  wire [31:0] bfly0_pid0_dout [0:7];
  wire bfly0_pid0_empty_n [0:7];
  wire bfly0_pid0_read [0:7];
  spmw_const #(.DW(32), .VAL(0)) u_bfly0_pid0_0 (.dout(bfly0_pid0_dout[0]), .empty_n(bfly0_pid0_empty_n[0]), .read(bfly0_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly0_pid0_1 (.dout(bfly0_pid0_dout[1]), .empty_n(bfly0_pid0_empty_n[1]), .read(bfly0_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_bfly0_pid0_2 (.dout(bfly0_pid0_dout[2]), .empty_n(bfly0_pid0_empty_n[2]), .read(bfly0_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_bfly0_pid0_3 (.dout(bfly0_pid0_dout[3]), .empty_n(bfly0_pid0_empty_n[3]), .read(bfly0_pid0_read[3]));
  spmw_const #(.DW(32), .VAL(4)) u_bfly0_pid0_4 (.dout(bfly0_pid0_dout[4]), .empty_n(bfly0_pid0_empty_n[4]), .read(bfly0_pid0_read[4]));
  spmw_const #(.DW(32), .VAL(5)) u_bfly0_pid0_5 (.dout(bfly0_pid0_dout[5]), .empty_n(bfly0_pid0_empty_n[5]), .read(bfly0_pid0_read[5]));
  spmw_const #(.DW(32), .VAL(6)) u_bfly0_pid0_6 (.dout(bfly0_pid0_dout[6]), .empty_n(bfly0_pid0_empty_n[6]), .read(bfly0_pid0_read[6]));
  spmw_const #(.DW(32), .VAL(7)) u_bfly0_pid0_7 (.dout(bfly0_pid0_dout[7]), .empty_n(bfly0_pid0_empty_n[7]), .read(bfly0_pid0_read[7]));
  // role bfly0_r0: 8 instance(s)
  bfly0_r0 u_bfly0_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly0_a_in_bind_dout[0]),
      .a_in_empty_n(bfly0_a_in_bind_empty_n[0]),
      .a_in_read(bfly0_a_in_bind_read[0]),
      .a_out_din(perm1_u_in_bind_din[0]),
      .a_out_full_n(perm1_u_in_bind_full_n[0]),
      .a_out_write(perm1_u_in_bind_write[0]),
      .b_in_dout(bfly0_b_in_bind_dout[0]),
      .b_in_empty_n(bfly0_b_in_bind_empty_n[0]),
      .b_in_read(bfly0_b_in_bind_read[0]),
      .b_out_din(perm1_v_in_bind_din[0]),
      .b_out_full_n(perm1_v_in_bind_full_n[0]),
      .b_out_write(perm1_v_in_bind_write[0]),
      ._pid0_dout(bfly0_pid0_dout[0]),
      ._pid0_empty_n(bfly0_pid0_empty_n[0]),
      ._pid0_read(bfly0_pid0_read[0]));
  bfly0_r0 u_bfly0_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly0_a_in_bind_dout[1]),
      .a_in_empty_n(bfly0_a_in_bind_empty_n[1]),
      .a_in_read(bfly0_a_in_bind_read[1]),
      .a_out_din(perm1_u_in_bind_din[1]),
      .a_out_full_n(perm1_u_in_bind_full_n[1]),
      .a_out_write(perm1_u_in_bind_write[1]),
      .b_in_dout(bfly0_b_in_bind_dout[1]),
      .b_in_empty_n(bfly0_b_in_bind_empty_n[1]),
      .b_in_read(bfly0_b_in_bind_read[1]),
      .b_out_din(perm1_v_in_bind_din[1]),
      .b_out_full_n(perm1_v_in_bind_full_n[1]),
      .b_out_write(perm1_v_in_bind_write[1]),
      ._pid0_dout(bfly0_pid0_dout[1]),
      ._pid0_empty_n(bfly0_pid0_empty_n[1]),
      ._pid0_read(bfly0_pid0_read[1]));
  bfly0_r0 u_bfly0_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly0_a_in_bind_dout[2]),
      .a_in_empty_n(bfly0_a_in_bind_empty_n[2]),
      .a_in_read(bfly0_a_in_bind_read[2]),
      .a_out_din(perm1_u_in_bind_din[2]),
      .a_out_full_n(perm1_u_in_bind_full_n[2]),
      .a_out_write(perm1_u_in_bind_write[2]),
      .b_in_dout(bfly0_b_in_bind_dout[2]),
      .b_in_empty_n(bfly0_b_in_bind_empty_n[2]),
      .b_in_read(bfly0_b_in_bind_read[2]),
      .b_out_din(perm1_v_in_bind_din[2]),
      .b_out_full_n(perm1_v_in_bind_full_n[2]),
      .b_out_write(perm1_v_in_bind_write[2]),
      ._pid0_dout(bfly0_pid0_dout[2]),
      ._pid0_empty_n(bfly0_pid0_empty_n[2]),
      ._pid0_read(bfly0_pid0_read[2]));
  bfly0_r0 u_bfly0_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly0_a_in_bind_dout[3]),
      .a_in_empty_n(bfly0_a_in_bind_empty_n[3]),
      .a_in_read(bfly0_a_in_bind_read[3]),
      .a_out_din(perm1_u_in_bind_din[3]),
      .a_out_full_n(perm1_u_in_bind_full_n[3]),
      .a_out_write(perm1_u_in_bind_write[3]),
      .b_in_dout(bfly0_b_in_bind_dout[3]),
      .b_in_empty_n(bfly0_b_in_bind_empty_n[3]),
      .b_in_read(bfly0_b_in_bind_read[3]),
      .b_out_din(perm1_v_in_bind_din[3]),
      .b_out_full_n(perm1_v_in_bind_full_n[3]),
      .b_out_write(perm1_v_in_bind_write[3]),
      ._pid0_dout(bfly0_pid0_dout[3]),
      ._pid0_empty_n(bfly0_pid0_empty_n[3]),
      ._pid0_read(bfly0_pid0_read[3]));
  bfly0_r0 u_bfly0_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly0_a_in_bind_dout[4]),
      .a_in_empty_n(bfly0_a_in_bind_empty_n[4]),
      .a_in_read(bfly0_a_in_bind_read[4]),
      .a_out_din(perm1_u_in_bind_din[4]),
      .a_out_full_n(perm1_u_in_bind_full_n[4]),
      .a_out_write(perm1_u_in_bind_write[4]),
      .b_in_dout(bfly0_b_in_bind_dout[4]),
      .b_in_empty_n(bfly0_b_in_bind_empty_n[4]),
      .b_in_read(bfly0_b_in_bind_read[4]),
      .b_out_din(perm1_v_in_bind_din[4]),
      .b_out_full_n(perm1_v_in_bind_full_n[4]),
      .b_out_write(perm1_v_in_bind_write[4]),
      ._pid0_dout(bfly0_pid0_dout[4]),
      ._pid0_empty_n(bfly0_pid0_empty_n[4]),
      ._pid0_read(bfly0_pid0_read[4]));
  bfly0_r0 u_bfly0_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly0_a_in_bind_dout[5]),
      .a_in_empty_n(bfly0_a_in_bind_empty_n[5]),
      .a_in_read(bfly0_a_in_bind_read[5]),
      .a_out_din(perm1_u_in_bind_din[5]),
      .a_out_full_n(perm1_u_in_bind_full_n[5]),
      .a_out_write(perm1_u_in_bind_write[5]),
      .b_in_dout(bfly0_b_in_bind_dout[5]),
      .b_in_empty_n(bfly0_b_in_bind_empty_n[5]),
      .b_in_read(bfly0_b_in_bind_read[5]),
      .b_out_din(perm1_v_in_bind_din[5]),
      .b_out_full_n(perm1_v_in_bind_full_n[5]),
      .b_out_write(perm1_v_in_bind_write[5]),
      ._pid0_dout(bfly0_pid0_dout[5]),
      ._pid0_empty_n(bfly0_pid0_empty_n[5]),
      ._pid0_read(bfly0_pid0_read[5]));
  bfly0_r0 u_bfly0_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly0_a_in_bind_dout[6]),
      .a_in_empty_n(bfly0_a_in_bind_empty_n[6]),
      .a_in_read(bfly0_a_in_bind_read[6]),
      .a_out_din(perm1_u_in_bind_din[6]),
      .a_out_full_n(perm1_u_in_bind_full_n[6]),
      .a_out_write(perm1_u_in_bind_write[6]),
      .b_in_dout(bfly0_b_in_bind_dout[6]),
      .b_in_empty_n(bfly0_b_in_bind_empty_n[6]),
      .b_in_read(bfly0_b_in_bind_read[6]),
      .b_out_din(perm1_v_in_bind_din[6]),
      .b_out_full_n(perm1_v_in_bind_full_n[6]),
      .b_out_write(perm1_v_in_bind_write[6]),
      ._pid0_dout(bfly0_pid0_dout[6]),
      ._pid0_empty_n(bfly0_pid0_empty_n[6]),
      ._pid0_read(bfly0_pid0_read[6]));
  bfly0_r0 u_bfly0_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly0_a_in_bind_dout[7]),
      .a_in_empty_n(bfly0_a_in_bind_empty_n[7]),
      .a_in_read(bfly0_a_in_bind_read[7]),
      .a_out_din(perm1_u_in_bind_din[7]),
      .a_out_full_n(perm1_u_in_bind_full_n[7]),
      .a_out_write(perm1_u_in_bind_write[7]),
      .b_in_dout(bfly0_b_in_bind_dout[7]),
      .b_in_empty_n(bfly0_b_in_bind_empty_n[7]),
      .b_in_read(bfly0_b_in_bind_read[7]),
      .b_out_din(perm1_v_in_bind_din[7]),
      .b_out_full_n(perm1_v_in_bind_full_n[7]),
      .b_out_write(perm1_v_in_bind_write[7]),
      ._pid0_dout(bfly0_pid0_dout[7]),
      ._pid0_empty_n(bfly0_pid0_empty_n[7]),
      ._pid0_read(bfly0_pid0_read[7]));
  // coordinate axis 0: 8 constant source(s)
  wire [31:0] bfly1_pid0_dout [0:7];
  wire bfly1_pid0_empty_n [0:7];
  wire bfly1_pid0_read [0:7];
  spmw_const #(.DW(32), .VAL(0)) u_bfly1_pid0_0 (.dout(bfly1_pid0_dout[0]), .empty_n(bfly1_pid0_empty_n[0]), .read(bfly1_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly1_pid0_1 (.dout(bfly1_pid0_dout[1]), .empty_n(bfly1_pid0_empty_n[1]), .read(bfly1_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_bfly1_pid0_2 (.dout(bfly1_pid0_dout[2]), .empty_n(bfly1_pid0_empty_n[2]), .read(bfly1_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_bfly1_pid0_3 (.dout(bfly1_pid0_dout[3]), .empty_n(bfly1_pid0_empty_n[3]), .read(bfly1_pid0_read[3]));
  spmw_const #(.DW(32), .VAL(4)) u_bfly1_pid0_4 (.dout(bfly1_pid0_dout[4]), .empty_n(bfly1_pid0_empty_n[4]), .read(bfly1_pid0_read[4]));
  spmw_const #(.DW(32), .VAL(5)) u_bfly1_pid0_5 (.dout(bfly1_pid0_dout[5]), .empty_n(bfly1_pid0_empty_n[5]), .read(bfly1_pid0_read[5]));
  spmw_const #(.DW(32), .VAL(6)) u_bfly1_pid0_6 (.dout(bfly1_pid0_dout[6]), .empty_n(bfly1_pid0_empty_n[6]), .read(bfly1_pid0_read[6]));
  spmw_const #(.DW(32), .VAL(7)) u_bfly1_pid0_7 (.dout(bfly1_pid0_dout[7]), .empty_n(bfly1_pid0_empty_n[7]), .read(bfly1_pid0_read[7]));
  // role bfly1_r0: 8 instance(s)
  bfly1_r0 u_bfly1_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly1_a_in_bind_dout[0]),
      .a_in_empty_n(bfly1_a_in_bind_empty_n[0]),
      .a_in_read(bfly1_a_in_bind_read[0]),
      .a_out_din(perm2_u_in_bind_din[0]),
      .a_out_full_n(perm2_u_in_bind_full_n[0]),
      .a_out_write(perm2_u_in_bind_write[0]),
      .b_in_dout(bfly1_b_in_bind_dout[0]),
      .b_in_empty_n(bfly1_b_in_bind_empty_n[0]),
      .b_in_read(bfly1_b_in_bind_read[0]),
      .b_out_din(perm2_v_in_bind_din[0]),
      .b_out_full_n(perm2_v_in_bind_full_n[0]),
      .b_out_write(perm2_v_in_bind_write[0]),
      ._pid0_dout(bfly1_pid0_dout[0]),
      ._pid0_empty_n(bfly1_pid0_empty_n[0]),
      ._pid0_read(bfly1_pid0_read[0]));
  bfly1_r0 u_bfly1_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly1_a_in_bind_dout[1]),
      .a_in_empty_n(bfly1_a_in_bind_empty_n[1]),
      .a_in_read(bfly1_a_in_bind_read[1]),
      .a_out_din(perm2_u_in_bind_din[1]),
      .a_out_full_n(perm2_u_in_bind_full_n[1]),
      .a_out_write(perm2_u_in_bind_write[1]),
      .b_in_dout(bfly1_b_in_bind_dout[1]),
      .b_in_empty_n(bfly1_b_in_bind_empty_n[1]),
      .b_in_read(bfly1_b_in_bind_read[1]),
      .b_out_din(perm2_v_in_bind_din[1]),
      .b_out_full_n(perm2_v_in_bind_full_n[1]),
      .b_out_write(perm2_v_in_bind_write[1]),
      ._pid0_dout(bfly1_pid0_dout[1]),
      ._pid0_empty_n(bfly1_pid0_empty_n[1]),
      ._pid0_read(bfly1_pid0_read[1]));
  bfly1_r0 u_bfly1_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly1_a_in_bind_dout[2]),
      .a_in_empty_n(bfly1_a_in_bind_empty_n[2]),
      .a_in_read(bfly1_a_in_bind_read[2]),
      .a_out_din(perm2_u_in_bind_din[2]),
      .a_out_full_n(perm2_u_in_bind_full_n[2]),
      .a_out_write(perm2_u_in_bind_write[2]),
      .b_in_dout(bfly1_b_in_bind_dout[2]),
      .b_in_empty_n(bfly1_b_in_bind_empty_n[2]),
      .b_in_read(bfly1_b_in_bind_read[2]),
      .b_out_din(perm2_v_in_bind_din[2]),
      .b_out_full_n(perm2_v_in_bind_full_n[2]),
      .b_out_write(perm2_v_in_bind_write[2]),
      ._pid0_dout(bfly1_pid0_dout[2]),
      ._pid0_empty_n(bfly1_pid0_empty_n[2]),
      ._pid0_read(bfly1_pid0_read[2]));
  bfly1_r0 u_bfly1_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly1_a_in_bind_dout[3]),
      .a_in_empty_n(bfly1_a_in_bind_empty_n[3]),
      .a_in_read(bfly1_a_in_bind_read[3]),
      .a_out_din(perm2_u_in_bind_din[3]),
      .a_out_full_n(perm2_u_in_bind_full_n[3]),
      .a_out_write(perm2_u_in_bind_write[3]),
      .b_in_dout(bfly1_b_in_bind_dout[3]),
      .b_in_empty_n(bfly1_b_in_bind_empty_n[3]),
      .b_in_read(bfly1_b_in_bind_read[3]),
      .b_out_din(perm2_v_in_bind_din[3]),
      .b_out_full_n(perm2_v_in_bind_full_n[3]),
      .b_out_write(perm2_v_in_bind_write[3]),
      ._pid0_dout(bfly1_pid0_dout[3]),
      ._pid0_empty_n(bfly1_pid0_empty_n[3]),
      ._pid0_read(bfly1_pid0_read[3]));
  bfly1_r0 u_bfly1_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly1_a_in_bind_dout[4]),
      .a_in_empty_n(bfly1_a_in_bind_empty_n[4]),
      .a_in_read(bfly1_a_in_bind_read[4]),
      .a_out_din(perm2_u_in_bind_din[4]),
      .a_out_full_n(perm2_u_in_bind_full_n[4]),
      .a_out_write(perm2_u_in_bind_write[4]),
      .b_in_dout(bfly1_b_in_bind_dout[4]),
      .b_in_empty_n(bfly1_b_in_bind_empty_n[4]),
      .b_in_read(bfly1_b_in_bind_read[4]),
      .b_out_din(perm2_v_in_bind_din[4]),
      .b_out_full_n(perm2_v_in_bind_full_n[4]),
      .b_out_write(perm2_v_in_bind_write[4]),
      ._pid0_dout(bfly1_pid0_dout[4]),
      ._pid0_empty_n(bfly1_pid0_empty_n[4]),
      ._pid0_read(bfly1_pid0_read[4]));
  bfly1_r0 u_bfly1_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly1_a_in_bind_dout[5]),
      .a_in_empty_n(bfly1_a_in_bind_empty_n[5]),
      .a_in_read(bfly1_a_in_bind_read[5]),
      .a_out_din(perm2_u_in_bind_din[5]),
      .a_out_full_n(perm2_u_in_bind_full_n[5]),
      .a_out_write(perm2_u_in_bind_write[5]),
      .b_in_dout(bfly1_b_in_bind_dout[5]),
      .b_in_empty_n(bfly1_b_in_bind_empty_n[5]),
      .b_in_read(bfly1_b_in_bind_read[5]),
      .b_out_din(perm2_v_in_bind_din[5]),
      .b_out_full_n(perm2_v_in_bind_full_n[5]),
      .b_out_write(perm2_v_in_bind_write[5]),
      ._pid0_dout(bfly1_pid0_dout[5]),
      ._pid0_empty_n(bfly1_pid0_empty_n[5]),
      ._pid0_read(bfly1_pid0_read[5]));
  bfly1_r0 u_bfly1_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly1_a_in_bind_dout[6]),
      .a_in_empty_n(bfly1_a_in_bind_empty_n[6]),
      .a_in_read(bfly1_a_in_bind_read[6]),
      .a_out_din(perm2_u_in_bind_din[6]),
      .a_out_full_n(perm2_u_in_bind_full_n[6]),
      .a_out_write(perm2_u_in_bind_write[6]),
      .b_in_dout(bfly1_b_in_bind_dout[6]),
      .b_in_empty_n(bfly1_b_in_bind_empty_n[6]),
      .b_in_read(bfly1_b_in_bind_read[6]),
      .b_out_din(perm2_v_in_bind_din[6]),
      .b_out_full_n(perm2_v_in_bind_full_n[6]),
      .b_out_write(perm2_v_in_bind_write[6]),
      ._pid0_dout(bfly1_pid0_dout[6]),
      ._pid0_empty_n(bfly1_pid0_empty_n[6]),
      ._pid0_read(bfly1_pid0_read[6]));
  bfly1_r0 u_bfly1_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly1_a_in_bind_dout[7]),
      .a_in_empty_n(bfly1_a_in_bind_empty_n[7]),
      .a_in_read(bfly1_a_in_bind_read[7]),
      .a_out_din(perm2_u_in_bind_din[7]),
      .a_out_full_n(perm2_u_in_bind_full_n[7]),
      .a_out_write(perm2_u_in_bind_write[7]),
      .b_in_dout(bfly1_b_in_bind_dout[7]),
      .b_in_empty_n(bfly1_b_in_bind_empty_n[7]),
      .b_in_read(bfly1_b_in_bind_read[7]),
      .b_out_din(perm2_v_in_bind_din[7]),
      .b_out_full_n(perm2_v_in_bind_full_n[7]),
      .b_out_write(perm2_v_in_bind_write[7]),
      ._pid0_dout(bfly1_pid0_dout[7]),
      ._pid0_empty_n(bfly1_pid0_empty_n[7]),
      ._pid0_read(bfly1_pid0_read[7]));
  // coordinate axis 0: 8 constant source(s)
  wire [31:0] bfly2_pid0_dout [0:7];
  wire bfly2_pid0_empty_n [0:7];
  wire bfly2_pid0_read [0:7];
  spmw_const #(.DW(32), .VAL(0)) u_bfly2_pid0_0 (.dout(bfly2_pid0_dout[0]), .empty_n(bfly2_pid0_empty_n[0]), .read(bfly2_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly2_pid0_1 (.dout(bfly2_pid0_dout[1]), .empty_n(bfly2_pid0_empty_n[1]), .read(bfly2_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_bfly2_pid0_2 (.dout(bfly2_pid0_dout[2]), .empty_n(bfly2_pid0_empty_n[2]), .read(bfly2_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_bfly2_pid0_3 (.dout(bfly2_pid0_dout[3]), .empty_n(bfly2_pid0_empty_n[3]), .read(bfly2_pid0_read[3]));
  spmw_const #(.DW(32), .VAL(4)) u_bfly2_pid0_4 (.dout(bfly2_pid0_dout[4]), .empty_n(bfly2_pid0_empty_n[4]), .read(bfly2_pid0_read[4]));
  spmw_const #(.DW(32), .VAL(5)) u_bfly2_pid0_5 (.dout(bfly2_pid0_dout[5]), .empty_n(bfly2_pid0_empty_n[5]), .read(bfly2_pid0_read[5]));
  spmw_const #(.DW(32), .VAL(6)) u_bfly2_pid0_6 (.dout(bfly2_pid0_dout[6]), .empty_n(bfly2_pid0_empty_n[6]), .read(bfly2_pid0_read[6]));
  spmw_const #(.DW(32), .VAL(7)) u_bfly2_pid0_7 (.dout(bfly2_pid0_dout[7]), .empty_n(bfly2_pid0_empty_n[7]), .read(bfly2_pid0_read[7]));
  // role bfly2_r0: 8 instance(s)
  bfly2_r0 u_bfly2_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly2_a_in_bind_dout[0]),
      .a_in_empty_n(bfly2_a_in_bind_empty_n[0]),
      .a_in_read(bfly2_a_in_bind_read[0]),
      .a_out_din(perm3_u_in_bind_din[0]),
      .a_out_full_n(perm3_u_in_bind_full_n[0]),
      .a_out_write(perm3_u_in_bind_write[0]),
      .b_in_dout(bfly2_b_in_bind_dout[0]),
      .b_in_empty_n(bfly2_b_in_bind_empty_n[0]),
      .b_in_read(bfly2_b_in_bind_read[0]),
      .b_out_din(perm3_v_in_bind_din[0]),
      .b_out_full_n(perm3_v_in_bind_full_n[0]),
      .b_out_write(perm3_v_in_bind_write[0]),
      ._pid0_dout(bfly2_pid0_dout[0]),
      ._pid0_empty_n(bfly2_pid0_empty_n[0]),
      ._pid0_read(bfly2_pid0_read[0]));
  bfly2_r0 u_bfly2_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly2_a_in_bind_dout[1]),
      .a_in_empty_n(bfly2_a_in_bind_empty_n[1]),
      .a_in_read(bfly2_a_in_bind_read[1]),
      .a_out_din(perm3_u_in_bind_din[1]),
      .a_out_full_n(perm3_u_in_bind_full_n[1]),
      .a_out_write(perm3_u_in_bind_write[1]),
      .b_in_dout(bfly2_b_in_bind_dout[1]),
      .b_in_empty_n(bfly2_b_in_bind_empty_n[1]),
      .b_in_read(bfly2_b_in_bind_read[1]),
      .b_out_din(perm3_v_in_bind_din[1]),
      .b_out_full_n(perm3_v_in_bind_full_n[1]),
      .b_out_write(perm3_v_in_bind_write[1]),
      ._pid0_dout(bfly2_pid0_dout[1]),
      ._pid0_empty_n(bfly2_pid0_empty_n[1]),
      ._pid0_read(bfly2_pid0_read[1]));
  bfly2_r0 u_bfly2_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly2_a_in_bind_dout[2]),
      .a_in_empty_n(bfly2_a_in_bind_empty_n[2]),
      .a_in_read(bfly2_a_in_bind_read[2]),
      .a_out_din(perm3_u_in_bind_din[2]),
      .a_out_full_n(perm3_u_in_bind_full_n[2]),
      .a_out_write(perm3_u_in_bind_write[2]),
      .b_in_dout(bfly2_b_in_bind_dout[2]),
      .b_in_empty_n(bfly2_b_in_bind_empty_n[2]),
      .b_in_read(bfly2_b_in_bind_read[2]),
      .b_out_din(perm3_v_in_bind_din[2]),
      .b_out_full_n(perm3_v_in_bind_full_n[2]),
      .b_out_write(perm3_v_in_bind_write[2]),
      ._pid0_dout(bfly2_pid0_dout[2]),
      ._pid0_empty_n(bfly2_pid0_empty_n[2]),
      ._pid0_read(bfly2_pid0_read[2]));
  bfly2_r0 u_bfly2_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly2_a_in_bind_dout[3]),
      .a_in_empty_n(bfly2_a_in_bind_empty_n[3]),
      .a_in_read(bfly2_a_in_bind_read[3]),
      .a_out_din(perm3_u_in_bind_din[3]),
      .a_out_full_n(perm3_u_in_bind_full_n[3]),
      .a_out_write(perm3_u_in_bind_write[3]),
      .b_in_dout(bfly2_b_in_bind_dout[3]),
      .b_in_empty_n(bfly2_b_in_bind_empty_n[3]),
      .b_in_read(bfly2_b_in_bind_read[3]),
      .b_out_din(perm3_v_in_bind_din[3]),
      .b_out_full_n(perm3_v_in_bind_full_n[3]),
      .b_out_write(perm3_v_in_bind_write[3]),
      ._pid0_dout(bfly2_pid0_dout[3]),
      ._pid0_empty_n(bfly2_pid0_empty_n[3]),
      ._pid0_read(bfly2_pid0_read[3]));
  bfly2_r0 u_bfly2_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly2_a_in_bind_dout[4]),
      .a_in_empty_n(bfly2_a_in_bind_empty_n[4]),
      .a_in_read(bfly2_a_in_bind_read[4]),
      .a_out_din(perm3_u_in_bind_din[4]),
      .a_out_full_n(perm3_u_in_bind_full_n[4]),
      .a_out_write(perm3_u_in_bind_write[4]),
      .b_in_dout(bfly2_b_in_bind_dout[4]),
      .b_in_empty_n(bfly2_b_in_bind_empty_n[4]),
      .b_in_read(bfly2_b_in_bind_read[4]),
      .b_out_din(perm3_v_in_bind_din[4]),
      .b_out_full_n(perm3_v_in_bind_full_n[4]),
      .b_out_write(perm3_v_in_bind_write[4]),
      ._pid0_dout(bfly2_pid0_dout[4]),
      ._pid0_empty_n(bfly2_pid0_empty_n[4]),
      ._pid0_read(bfly2_pid0_read[4]));
  bfly2_r0 u_bfly2_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly2_a_in_bind_dout[5]),
      .a_in_empty_n(bfly2_a_in_bind_empty_n[5]),
      .a_in_read(bfly2_a_in_bind_read[5]),
      .a_out_din(perm3_u_in_bind_din[5]),
      .a_out_full_n(perm3_u_in_bind_full_n[5]),
      .a_out_write(perm3_u_in_bind_write[5]),
      .b_in_dout(bfly2_b_in_bind_dout[5]),
      .b_in_empty_n(bfly2_b_in_bind_empty_n[5]),
      .b_in_read(bfly2_b_in_bind_read[5]),
      .b_out_din(perm3_v_in_bind_din[5]),
      .b_out_full_n(perm3_v_in_bind_full_n[5]),
      .b_out_write(perm3_v_in_bind_write[5]),
      ._pid0_dout(bfly2_pid0_dout[5]),
      ._pid0_empty_n(bfly2_pid0_empty_n[5]),
      ._pid0_read(bfly2_pid0_read[5]));
  bfly2_r0 u_bfly2_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly2_a_in_bind_dout[6]),
      .a_in_empty_n(bfly2_a_in_bind_empty_n[6]),
      .a_in_read(bfly2_a_in_bind_read[6]),
      .a_out_din(perm3_u_in_bind_din[6]),
      .a_out_full_n(perm3_u_in_bind_full_n[6]),
      .a_out_write(perm3_u_in_bind_write[6]),
      .b_in_dout(bfly2_b_in_bind_dout[6]),
      .b_in_empty_n(bfly2_b_in_bind_empty_n[6]),
      .b_in_read(bfly2_b_in_bind_read[6]),
      .b_out_din(perm3_v_in_bind_din[6]),
      .b_out_full_n(perm3_v_in_bind_full_n[6]),
      .b_out_write(perm3_v_in_bind_write[6]),
      ._pid0_dout(bfly2_pid0_dout[6]),
      ._pid0_empty_n(bfly2_pid0_empty_n[6]),
      ._pid0_read(bfly2_pid0_read[6]));
  bfly2_r0 u_bfly2_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly2_a_in_bind_dout[7]),
      .a_in_empty_n(bfly2_a_in_bind_empty_n[7]),
      .a_in_read(bfly2_a_in_bind_read[7]),
      .a_out_din(perm3_u_in_bind_din[7]),
      .a_out_full_n(perm3_u_in_bind_full_n[7]),
      .a_out_write(perm3_u_in_bind_write[7]),
      .b_in_dout(bfly2_b_in_bind_dout[7]),
      .b_in_empty_n(bfly2_b_in_bind_empty_n[7]),
      .b_in_read(bfly2_b_in_bind_read[7]),
      .b_out_din(perm3_v_in_bind_din[7]),
      .b_out_full_n(perm3_v_in_bind_full_n[7]),
      .b_out_write(perm3_v_in_bind_write[7]),
      ._pid0_dout(bfly2_pid0_dout[7]),
      ._pid0_empty_n(bfly2_pid0_empty_n[7]),
      ._pid0_read(bfly2_pid0_read[7]));
  // coordinate axis 0: 8 constant source(s)
  wire [31:0] bfly3_pid0_dout [0:7];
  wire bfly3_pid0_empty_n [0:7];
  wire bfly3_pid0_read [0:7];
  spmw_const #(.DW(32), .VAL(0)) u_bfly3_pid0_0 (.dout(bfly3_pid0_dout[0]), .empty_n(bfly3_pid0_empty_n[0]), .read(bfly3_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly3_pid0_1 (.dout(bfly3_pid0_dout[1]), .empty_n(bfly3_pid0_empty_n[1]), .read(bfly3_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_bfly3_pid0_2 (.dout(bfly3_pid0_dout[2]), .empty_n(bfly3_pid0_empty_n[2]), .read(bfly3_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_bfly3_pid0_3 (.dout(bfly3_pid0_dout[3]), .empty_n(bfly3_pid0_empty_n[3]), .read(bfly3_pid0_read[3]));
  spmw_const #(.DW(32), .VAL(4)) u_bfly3_pid0_4 (.dout(bfly3_pid0_dout[4]), .empty_n(bfly3_pid0_empty_n[4]), .read(bfly3_pid0_read[4]));
  spmw_const #(.DW(32), .VAL(5)) u_bfly3_pid0_5 (.dout(bfly3_pid0_dout[5]), .empty_n(bfly3_pid0_empty_n[5]), .read(bfly3_pid0_read[5]));
  spmw_const #(.DW(32), .VAL(6)) u_bfly3_pid0_6 (.dout(bfly3_pid0_dout[6]), .empty_n(bfly3_pid0_empty_n[6]), .read(bfly3_pid0_read[6]));
  spmw_const #(.DW(32), .VAL(7)) u_bfly3_pid0_7 (.dout(bfly3_pid0_dout[7]), .empty_n(bfly3_pid0_empty_n[7]), .read(bfly3_pid0_read[7]));
  // role bfly3_r0: 8 instance(s)
  bfly3_r0 u_bfly3_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly3_a_in_bind_dout[0]),
      .a_in_empty_n(bfly3_a_in_bind_empty_n[0]),
      .a_in_read(bfly3_a_in_bind_read[0]),
      .a_out_din(perm4_u_in_bind_din[0]),
      .a_out_full_n(perm4_u_in_bind_full_n[0]),
      .a_out_write(perm4_u_in_bind_write[0]),
      .b_in_dout(bfly3_b_in_bind_dout[0]),
      .b_in_empty_n(bfly3_b_in_bind_empty_n[0]),
      .b_in_read(bfly3_b_in_bind_read[0]),
      .b_out_din(perm4_v_in_bind_din[0]),
      .b_out_full_n(perm4_v_in_bind_full_n[0]),
      .b_out_write(perm4_v_in_bind_write[0]),
      ._pid0_dout(bfly3_pid0_dout[0]),
      ._pid0_empty_n(bfly3_pid0_empty_n[0]),
      ._pid0_read(bfly3_pid0_read[0]));
  bfly3_r0 u_bfly3_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly3_a_in_bind_dout[1]),
      .a_in_empty_n(bfly3_a_in_bind_empty_n[1]),
      .a_in_read(bfly3_a_in_bind_read[1]),
      .a_out_din(perm4_u_in_bind_din[1]),
      .a_out_full_n(perm4_u_in_bind_full_n[1]),
      .a_out_write(perm4_u_in_bind_write[1]),
      .b_in_dout(bfly3_b_in_bind_dout[1]),
      .b_in_empty_n(bfly3_b_in_bind_empty_n[1]),
      .b_in_read(bfly3_b_in_bind_read[1]),
      .b_out_din(perm4_v_in_bind_din[1]),
      .b_out_full_n(perm4_v_in_bind_full_n[1]),
      .b_out_write(perm4_v_in_bind_write[1]),
      ._pid0_dout(bfly3_pid0_dout[1]),
      ._pid0_empty_n(bfly3_pid0_empty_n[1]),
      ._pid0_read(bfly3_pid0_read[1]));
  bfly3_r0 u_bfly3_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly3_a_in_bind_dout[2]),
      .a_in_empty_n(bfly3_a_in_bind_empty_n[2]),
      .a_in_read(bfly3_a_in_bind_read[2]),
      .a_out_din(perm4_u_in_bind_din[2]),
      .a_out_full_n(perm4_u_in_bind_full_n[2]),
      .a_out_write(perm4_u_in_bind_write[2]),
      .b_in_dout(bfly3_b_in_bind_dout[2]),
      .b_in_empty_n(bfly3_b_in_bind_empty_n[2]),
      .b_in_read(bfly3_b_in_bind_read[2]),
      .b_out_din(perm4_v_in_bind_din[2]),
      .b_out_full_n(perm4_v_in_bind_full_n[2]),
      .b_out_write(perm4_v_in_bind_write[2]),
      ._pid0_dout(bfly3_pid0_dout[2]),
      ._pid0_empty_n(bfly3_pid0_empty_n[2]),
      ._pid0_read(bfly3_pid0_read[2]));
  bfly3_r0 u_bfly3_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly3_a_in_bind_dout[3]),
      .a_in_empty_n(bfly3_a_in_bind_empty_n[3]),
      .a_in_read(bfly3_a_in_bind_read[3]),
      .a_out_din(perm4_u_in_bind_din[3]),
      .a_out_full_n(perm4_u_in_bind_full_n[3]),
      .a_out_write(perm4_u_in_bind_write[3]),
      .b_in_dout(bfly3_b_in_bind_dout[3]),
      .b_in_empty_n(bfly3_b_in_bind_empty_n[3]),
      .b_in_read(bfly3_b_in_bind_read[3]),
      .b_out_din(perm4_v_in_bind_din[3]),
      .b_out_full_n(perm4_v_in_bind_full_n[3]),
      .b_out_write(perm4_v_in_bind_write[3]),
      ._pid0_dout(bfly3_pid0_dout[3]),
      ._pid0_empty_n(bfly3_pid0_empty_n[3]),
      ._pid0_read(bfly3_pid0_read[3]));
  bfly3_r0 u_bfly3_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly3_a_in_bind_dout[4]),
      .a_in_empty_n(bfly3_a_in_bind_empty_n[4]),
      .a_in_read(bfly3_a_in_bind_read[4]),
      .a_out_din(perm4_u_in_bind_din[4]),
      .a_out_full_n(perm4_u_in_bind_full_n[4]),
      .a_out_write(perm4_u_in_bind_write[4]),
      .b_in_dout(bfly3_b_in_bind_dout[4]),
      .b_in_empty_n(bfly3_b_in_bind_empty_n[4]),
      .b_in_read(bfly3_b_in_bind_read[4]),
      .b_out_din(perm4_v_in_bind_din[4]),
      .b_out_full_n(perm4_v_in_bind_full_n[4]),
      .b_out_write(perm4_v_in_bind_write[4]),
      ._pid0_dout(bfly3_pid0_dout[4]),
      ._pid0_empty_n(bfly3_pid0_empty_n[4]),
      ._pid0_read(bfly3_pid0_read[4]));
  bfly3_r0 u_bfly3_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly3_a_in_bind_dout[5]),
      .a_in_empty_n(bfly3_a_in_bind_empty_n[5]),
      .a_in_read(bfly3_a_in_bind_read[5]),
      .a_out_din(perm4_u_in_bind_din[5]),
      .a_out_full_n(perm4_u_in_bind_full_n[5]),
      .a_out_write(perm4_u_in_bind_write[5]),
      .b_in_dout(bfly3_b_in_bind_dout[5]),
      .b_in_empty_n(bfly3_b_in_bind_empty_n[5]),
      .b_in_read(bfly3_b_in_bind_read[5]),
      .b_out_din(perm4_v_in_bind_din[5]),
      .b_out_full_n(perm4_v_in_bind_full_n[5]),
      .b_out_write(perm4_v_in_bind_write[5]),
      ._pid0_dout(bfly3_pid0_dout[5]),
      ._pid0_empty_n(bfly3_pid0_empty_n[5]),
      ._pid0_read(bfly3_pid0_read[5]));
  bfly3_r0 u_bfly3_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly3_a_in_bind_dout[6]),
      .a_in_empty_n(bfly3_a_in_bind_empty_n[6]),
      .a_in_read(bfly3_a_in_bind_read[6]),
      .a_out_din(perm4_u_in_bind_din[6]),
      .a_out_full_n(perm4_u_in_bind_full_n[6]),
      .a_out_write(perm4_u_in_bind_write[6]),
      .b_in_dout(bfly3_b_in_bind_dout[6]),
      .b_in_empty_n(bfly3_b_in_bind_empty_n[6]),
      .b_in_read(bfly3_b_in_bind_read[6]),
      .b_out_din(perm4_v_in_bind_din[6]),
      .b_out_full_n(perm4_v_in_bind_full_n[6]),
      .b_out_write(perm4_v_in_bind_write[6]),
      ._pid0_dout(bfly3_pid0_dout[6]),
      ._pid0_empty_n(bfly3_pid0_empty_n[6]),
      ._pid0_read(bfly3_pid0_read[6]));
  bfly3_r0 u_bfly3_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly3_a_in_bind_dout[7]),
      .a_in_empty_n(bfly3_a_in_bind_empty_n[7]),
      .a_in_read(bfly3_a_in_bind_read[7]),
      .a_out_din(perm4_u_in_bind_din[7]),
      .a_out_full_n(perm4_u_in_bind_full_n[7]),
      .a_out_write(perm4_u_in_bind_write[7]),
      .b_in_dout(bfly3_b_in_bind_dout[7]),
      .b_in_empty_n(bfly3_b_in_bind_empty_n[7]),
      .b_in_read(bfly3_b_in_bind_read[7]),
      .b_out_din(perm4_v_in_bind_din[7]),
      .b_out_full_n(perm4_v_in_bind_full_n[7]),
      .b_out_write(perm4_v_in_bind_write[7]),
      ._pid0_dout(bfly3_pid0_dout[7]),
      ._pid0_empty_n(bfly3_pid0_empty_n[7]),
      ._pid0_read(bfly3_pid0_read[7]));
  // coordinate axis 0: 8 constant source(s)
  wire [31:0] bfly4_pid0_dout [0:7];
  wire bfly4_pid0_empty_n [0:7];
  wire bfly4_pid0_read [0:7];
  spmw_const #(.DW(32), .VAL(0)) u_bfly4_pid0_0 (.dout(bfly4_pid0_dout[0]), .empty_n(bfly4_pid0_empty_n[0]), .read(bfly4_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly4_pid0_1 (.dout(bfly4_pid0_dout[1]), .empty_n(bfly4_pid0_empty_n[1]), .read(bfly4_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_bfly4_pid0_2 (.dout(bfly4_pid0_dout[2]), .empty_n(bfly4_pid0_empty_n[2]), .read(bfly4_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_bfly4_pid0_3 (.dout(bfly4_pid0_dout[3]), .empty_n(bfly4_pid0_empty_n[3]), .read(bfly4_pid0_read[3]));
  spmw_const #(.DW(32), .VAL(4)) u_bfly4_pid0_4 (.dout(bfly4_pid0_dout[4]), .empty_n(bfly4_pid0_empty_n[4]), .read(bfly4_pid0_read[4]));
  spmw_const #(.DW(32), .VAL(5)) u_bfly4_pid0_5 (.dout(bfly4_pid0_dout[5]), .empty_n(bfly4_pid0_empty_n[5]), .read(bfly4_pid0_read[5]));
  spmw_const #(.DW(32), .VAL(6)) u_bfly4_pid0_6 (.dout(bfly4_pid0_dout[6]), .empty_n(bfly4_pid0_empty_n[6]), .read(bfly4_pid0_read[6]));
  spmw_const #(.DW(32), .VAL(7)) u_bfly4_pid0_7 (.dout(bfly4_pid0_dout[7]), .empty_n(bfly4_pid0_empty_n[7]), .read(bfly4_pid0_read[7]));
  // role bfly4_r0: 8 instance(s)
  bfly4_r0 u_bfly4_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly4_a_in_bind_dout[0]),
      .a_in_empty_n(bfly4_a_in_bind_empty_n[0]),
      .a_in_read(bfly4_a_in_bind_read[0]),
      .a_out_din(perm5_u_in_bind_din[0]),
      .a_out_full_n(perm5_u_in_bind_full_n[0]),
      .a_out_write(perm5_u_in_bind_write[0]),
      .b_in_dout(bfly4_b_in_bind_dout[0]),
      .b_in_empty_n(bfly4_b_in_bind_empty_n[0]),
      .b_in_read(bfly4_b_in_bind_read[0]),
      .b_out_din(perm5_v_in_bind_din[0]),
      .b_out_full_n(perm5_v_in_bind_full_n[0]),
      .b_out_write(perm5_v_in_bind_write[0]),
      ._pid0_dout(bfly4_pid0_dout[0]),
      ._pid0_empty_n(bfly4_pid0_empty_n[0]),
      ._pid0_read(bfly4_pid0_read[0]));
  bfly4_r0 u_bfly4_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly4_a_in_bind_dout[1]),
      .a_in_empty_n(bfly4_a_in_bind_empty_n[1]),
      .a_in_read(bfly4_a_in_bind_read[1]),
      .a_out_din(perm5_u_in_bind_din[1]),
      .a_out_full_n(perm5_u_in_bind_full_n[1]),
      .a_out_write(perm5_u_in_bind_write[1]),
      .b_in_dout(bfly4_b_in_bind_dout[1]),
      .b_in_empty_n(bfly4_b_in_bind_empty_n[1]),
      .b_in_read(bfly4_b_in_bind_read[1]),
      .b_out_din(perm5_v_in_bind_din[1]),
      .b_out_full_n(perm5_v_in_bind_full_n[1]),
      .b_out_write(perm5_v_in_bind_write[1]),
      ._pid0_dout(bfly4_pid0_dout[1]),
      ._pid0_empty_n(bfly4_pid0_empty_n[1]),
      ._pid0_read(bfly4_pid0_read[1]));
  bfly4_r0 u_bfly4_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly4_a_in_bind_dout[2]),
      .a_in_empty_n(bfly4_a_in_bind_empty_n[2]),
      .a_in_read(bfly4_a_in_bind_read[2]),
      .a_out_din(perm5_u_in_bind_din[2]),
      .a_out_full_n(perm5_u_in_bind_full_n[2]),
      .a_out_write(perm5_u_in_bind_write[2]),
      .b_in_dout(bfly4_b_in_bind_dout[2]),
      .b_in_empty_n(bfly4_b_in_bind_empty_n[2]),
      .b_in_read(bfly4_b_in_bind_read[2]),
      .b_out_din(perm5_v_in_bind_din[2]),
      .b_out_full_n(perm5_v_in_bind_full_n[2]),
      .b_out_write(perm5_v_in_bind_write[2]),
      ._pid0_dout(bfly4_pid0_dout[2]),
      ._pid0_empty_n(bfly4_pid0_empty_n[2]),
      ._pid0_read(bfly4_pid0_read[2]));
  bfly4_r0 u_bfly4_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly4_a_in_bind_dout[3]),
      .a_in_empty_n(bfly4_a_in_bind_empty_n[3]),
      .a_in_read(bfly4_a_in_bind_read[3]),
      .a_out_din(perm5_u_in_bind_din[3]),
      .a_out_full_n(perm5_u_in_bind_full_n[3]),
      .a_out_write(perm5_u_in_bind_write[3]),
      .b_in_dout(bfly4_b_in_bind_dout[3]),
      .b_in_empty_n(bfly4_b_in_bind_empty_n[3]),
      .b_in_read(bfly4_b_in_bind_read[3]),
      .b_out_din(perm5_v_in_bind_din[3]),
      .b_out_full_n(perm5_v_in_bind_full_n[3]),
      .b_out_write(perm5_v_in_bind_write[3]),
      ._pid0_dout(bfly4_pid0_dout[3]),
      ._pid0_empty_n(bfly4_pid0_empty_n[3]),
      ._pid0_read(bfly4_pid0_read[3]));
  bfly4_r0 u_bfly4_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly4_a_in_bind_dout[4]),
      .a_in_empty_n(bfly4_a_in_bind_empty_n[4]),
      .a_in_read(bfly4_a_in_bind_read[4]),
      .a_out_din(perm5_u_in_bind_din[4]),
      .a_out_full_n(perm5_u_in_bind_full_n[4]),
      .a_out_write(perm5_u_in_bind_write[4]),
      .b_in_dout(bfly4_b_in_bind_dout[4]),
      .b_in_empty_n(bfly4_b_in_bind_empty_n[4]),
      .b_in_read(bfly4_b_in_bind_read[4]),
      .b_out_din(perm5_v_in_bind_din[4]),
      .b_out_full_n(perm5_v_in_bind_full_n[4]),
      .b_out_write(perm5_v_in_bind_write[4]),
      ._pid0_dout(bfly4_pid0_dout[4]),
      ._pid0_empty_n(bfly4_pid0_empty_n[4]),
      ._pid0_read(bfly4_pid0_read[4]));
  bfly4_r0 u_bfly4_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly4_a_in_bind_dout[5]),
      .a_in_empty_n(bfly4_a_in_bind_empty_n[5]),
      .a_in_read(bfly4_a_in_bind_read[5]),
      .a_out_din(perm5_u_in_bind_din[5]),
      .a_out_full_n(perm5_u_in_bind_full_n[5]),
      .a_out_write(perm5_u_in_bind_write[5]),
      .b_in_dout(bfly4_b_in_bind_dout[5]),
      .b_in_empty_n(bfly4_b_in_bind_empty_n[5]),
      .b_in_read(bfly4_b_in_bind_read[5]),
      .b_out_din(perm5_v_in_bind_din[5]),
      .b_out_full_n(perm5_v_in_bind_full_n[5]),
      .b_out_write(perm5_v_in_bind_write[5]),
      ._pid0_dout(bfly4_pid0_dout[5]),
      ._pid0_empty_n(bfly4_pid0_empty_n[5]),
      ._pid0_read(bfly4_pid0_read[5]));
  bfly4_r0 u_bfly4_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly4_a_in_bind_dout[6]),
      .a_in_empty_n(bfly4_a_in_bind_empty_n[6]),
      .a_in_read(bfly4_a_in_bind_read[6]),
      .a_out_din(perm5_u_in_bind_din[6]),
      .a_out_full_n(perm5_u_in_bind_full_n[6]),
      .a_out_write(perm5_u_in_bind_write[6]),
      .b_in_dout(bfly4_b_in_bind_dout[6]),
      .b_in_empty_n(bfly4_b_in_bind_empty_n[6]),
      .b_in_read(bfly4_b_in_bind_read[6]),
      .b_out_din(perm5_v_in_bind_din[6]),
      .b_out_full_n(perm5_v_in_bind_full_n[6]),
      .b_out_write(perm5_v_in_bind_write[6]),
      ._pid0_dout(bfly4_pid0_dout[6]),
      ._pid0_empty_n(bfly4_pid0_empty_n[6]),
      ._pid0_read(bfly4_pid0_read[6]));
  bfly4_r0 u_bfly4_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly4_a_in_bind_dout[7]),
      .a_in_empty_n(bfly4_a_in_bind_empty_n[7]),
      .a_in_read(bfly4_a_in_bind_read[7]),
      .a_out_din(perm5_u_in_bind_din[7]),
      .a_out_full_n(perm5_u_in_bind_full_n[7]),
      .a_out_write(perm5_u_in_bind_write[7]),
      .b_in_dout(bfly4_b_in_bind_dout[7]),
      .b_in_empty_n(bfly4_b_in_bind_empty_n[7]),
      .b_in_read(bfly4_b_in_bind_read[7]),
      .b_out_din(perm5_v_in_bind_din[7]),
      .b_out_full_n(perm5_v_in_bind_full_n[7]),
      .b_out_write(perm5_v_in_bind_write[7]),
      ._pid0_dout(bfly4_pid0_dout[7]),
      ._pid0_empty_n(bfly4_pid0_empty_n[7]),
      ._pid0_read(bfly4_pid0_read[7]));
  // coordinate axis 0: 8 constant source(s)
  wire [31:0] bfly5_pid0_dout [0:7];
  wire bfly5_pid0_empty_n [0:7];
  wire bfly5_pid0_read [0:7];
  spmw_const #(.DW(32), .VAL(0)) u_bfly5_pid0_0 (.dout(bfly5_pid0_dout[0]), .empty_n(bfly5_pid0_empty_n[0]), .read(bfly5_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly5_pid0_1 (.dout(bfly5_pid0_dout[1]), .empty_n(bfly5_pid0_empty_n[1]), .read(bfly5_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_bfly5_pid0_2 (.dout(bfly5_pid0_dout[2]), .empty_n(bfly5_pid0_empty_n[2]), .read(bfly5_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_bfly5_pid0_3 (.dout(bfly5_pid0_dout[3]), .empty_n(bfly5_pid0_empty_n[3]), .read(bfly5_pid0_read[3]));
  spmw_const #(.DW(32), .VAL(4)) u_bfly5_pid0_4 (.dout(bfly5_pid0_dout[4]), .empty_n(bfly5_pid0_empty_n[4]), .read(bfly5_pid0_read[4]));
  spmw_const #(.DW(32), .VAL(5)) u_bfly5_pid0_5 (.dout(bfly5_pid0_dout[5]), .empty_n(bfly5_pid0_empty_n[5]), .read(bfly5_pid0_read[5]));
  spmw_const #(.DW(32), .VAL(6)) u_bfly5_pid0_6 (.dout(bfly5_pid0_dout[6]), .empty_n(bfly5_pid0_empty_n[6]), .read(bfly5_pid0_read[6]));
  spmw_const #(.DW(32), .VAL(7)) u_bfly5_pid0_7 (.dout(bfly5_pid0_dout[7]), .empty_n(bfly5_pid0_empty_n[7]), .read(bfly5_pid0_read[7]));
  // role bfly5_r0: 8 instance(s)
  bfly5_r0 u_bfly5_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly5_a_in_bind_dout[0]),
      .a_in_empty_n(bfly5_a_in_bind_empty_n[0]),
      .a_in_read(bfly5_a_in_bind_read[0]),
      .a_out_din(perm6_u_in_bind_din[0]),
      .a_out_full_n(perm6_u_in_bind_full_n[0]),
      .a_out_write(perm6_u_in_bind_write[0]),
      .b_in_dout(bfly5_b_in_bind_dout[0]),
      .b_in_empty_n(bfly5_b_in_bind_empty_n[0]),
      .b_in_read(bfly5_b_in_bind_read[0]),
      .b_out_din(perm6_v_in_bind_din[0]),
      .b_out_full_n(perm6_v_in_bind_full_n[0]),
      .b_out_write(perm6_v_in_bind_write[0]),
      ._pid0_dout(bfly5_pid0_dout[0]),
      ._pid0_empty_n(bfly5_pid0_empty_n[0]),
      ._pid0_read(bfly5_pid0_read[0]));
  bfly5_r0 u_bfly5_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly5_a_in_bind_dout[1]),
      .a_in_empty_n(bfly5_a_in_bind_empty_n[1]),
      .a_in_read(bfly5_a_in_bind_read[1]),
      .a_out_din(perm6_u_in_bind_din[1]),
      .a_out_full_n(perm6_u_in_bind_full_n[1]),
      .a_out_write(perm6_u_in_bind_write[1]),
      .b_in_dout(bfly5_b_in_bind_dout[1]),
      .b_in_empty_n(bfly5_b_in_bind_empty_n[1]),
      .b_in_read(bfly5_b_in_bind_read[1]),
      .b_out_din(perm6_v_in_bind_din[1]),
      .b_out_full_n(perm6_v_in_bind_full_n[1]),
      .b_out_write(perm6_v_in_bind_write[1]),
      ._pid0_dout(bfly5_pid0_dout[1]),
      ._pid0_empty_n(bfly5_pid0_empty_n[1]),
      ._pid0_read(bfly5_pid0_read[1]));
  bfly5_r0 u_bfly5_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly5_a_in_bind_dout[2]),
      .a_in_empty_n(bfly5_a_in_bind_empty_n[2]),
      .a_in_read(bfly5_a_in_bind_read[2]),
      .a_out_din(perm6_u_in_bind_din[2]),
      .a_out_full_n(perm6_u_in_bind_full_n[2]),
      .a_out_write(perm6_u_in_bind_write[2]),
      .b_in_dout(bfly5_b_in_bind_dout[2]),
      .b_in_empty_n(bfly5_b_in_bind_empty_n[2]),
      .b_in_read(bfly5_b_in_bind_read[2]),
      .b_out_din(perm6_v_in_bind_din[2]),
      .b_out_full_n(perm6_v_in_bind_full_n[2]),
      .b_out_write(perm6_v_in_bind_write[2]),
      ._pid0_dout(bfly5_pid0_dout[2]),
      ._pid0_empty_n(bfly5_pid0_empty_n[2]),
      ._pid0_read(bfly5_pid0_read[2]));
  bfly5_r0 u_bfly5_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly5_a_in_bind_dout[3]),
      .a_in_empty_n(bfly5_a_in_bind_empty_n[3]),
      .a_in_read(bfly5_a_in_bind_read[3]),
      .a_out_din(perm6_u_in_bind_din[3]),
      .a_out_full_n(perm6_u_in_bind_full_n[3]),
      .a_out_write(perm6_u_in_bind_write[3]),
      .b_in_dout(bfly5_b_in_bind_dout[3]),
      .b_in_empty_n(bfly5_b_in_bind_empty_n[3]),
      .b_in_read(bfly5_b_in_bind_read[3]),
      .b_out_din(perm6_v_in_bind_din[3]),
      .b_out_full_n(perm6_v_in_bind_full_n[3]),
      .b_out_write(perm6_v_in_bind_write[3]),
      ._pid0_dout(bfly5_pid0_dout[3]),
      ._pid0_empty_n(bfly5_pid0_empty_n[3]),
      ._pid0_read(bfly5_pid0_read[3]));
  bfly5_r0 u_bfly5_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly5_a_in_bind_dout[4]),
      .a_in_empty_n(bfly5_a_in_bind_empty_n[4]),
      .a_in_read(bfly5_a_in_bind_read[4]),
      .a_out_din(perm6_u_in_bind_din[4]),
      .a_out_full_n(perm6_u_in_bind_full_n[4]),
      .a_out_write(perm6_u_in_bind_write[4]),
      .b_in_dout(bfly5_b_in_bind_dout[4]),
      .b_in_empty_n(bfly5_b_in_bind_empty_n[4]),
      .b_in_read(bfly5_b_in_bind_read[4]),
      .b_out_din(perm6_v_in_bind_din[4]),
      .b_out_full_n(perm6_v_in_bind_full_n[4]),
      .b_out_write(perm6_v_in_bind_write[4]),
      ._pid0_dout(bfly5_pid0_dout[4]),
      ._pid0_empty_n(bfly5_pid0_empty_n[4]),
      ._pid0_read(bfly5_pid0_read[4]));
  bfly5_r0 u_bfly5_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly5_a_in_bind_dout[5]),
      .a_in_empty_n(bfly5_a_in_bind_empty_n[5]),
      .a_in_read(bfly5_a_in_bind_read[5]),
      .a_out_din(perm6_u_in_bind_din[5]),
      .a_out_full_n(perm6_u_in_bind_full_n[5]),
      .a_out_write(perm6_u_in_bind_write[5]),
      .b_in_dout(bfly5_b_in_bind_dout[5]),
      .b_in_empty_n(bfly5_b_in_bind_empty_n[5]),
      .b_in_read(bfly5_b_in_bind_read[5]),
      .b_out_din(perm6_v_in_bind_din[5]),
      .b_out_full_n(perm6_v_in_bind_full_n[5]),
      .b_out_write(perm6_v_in_bind_write[5]),
      ._pid0_dout(bfly5_pid0_dout[5]),
      ._pid0_empty_n(bfly5_pid0_empty_n[5]),
      ._pid0_read(bfly5_pid0_read[5]));
  bfly5_r0 u_bfly5_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly5_a_in_bind_dout[6]),
      .a_in_empty_n(bfly5_a_in_bind_empty_n[6]),
      .a_in_read(bfly5_a_in_bind_read[6]),
      .a_out_din(perm6_u_in_bind_din[6]),
      .a_out_full_n(perm6_u_in_bind_full_n[6]),
      .a_out_write(perm6_u_in_bind_write[6]),
      .b_in_dout(bfly5_b_in_bind_dout[6]),
      .b_in_empty_n(bfly5_b_in_bind_empty_n[6]),
      .b_in_read(bfly5_b_in_bind_read[6]),
      .b_out_din(perm6_v_in_bind_din[6]),
      .b_out_full_n(perm6_v_in_bind_full_n[6]),
      .b_out_write(perm6_v_in_bind_write[6]),
      ._pid0_dout(bfly5_pid0_dout[6]),
      ._pid0_empty_n(bfly5_pid0_empty_n[6]),
      ._pid0_read(bfly5_pid0_read[6]));
  bfly5_r0 u_bfly5_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly5_a_in_bind_dout[7]),
      .a_in_empty_n(bfly5_a_in_bind_empty_n[7]),
      .a_in_read(bfly5_a_in_bind_read[7]),
      .a_out_din(perm6_u_in_bind_din[7]),
      .a_out_full_n(perm6_u_in_bind_full_n[7]),
      .a_out_write(perm6_u_in_bind_write[7]),
      .b_in_dout(bfly5_b_in_bind_dout[7]),
      .b_in_empty_n(bfly5_b_in_bind_empty_n[7]),
      .b_in_read(bfly5_b_in_bind_read[7]),
      .b_out_din(perm6_v_in_bind_din[7]),
      .b_out_full_n(perm6_v_in_bind_full_n[7]),
      .b_out_write(perm6_v_in_bind_write[7]),
      ._pid0_dout(bfly5_pid0_dout[7]),
      ._pid0_empty_n(bfly5_pid0_empty_n[7]),
      ._pid0_read(bfly5_pid0_read[7]));
  // role tail_general_r0: 1 instance(s)
  tail_general_r0 u_tail_general_r0_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_in_bind_dout[0]),
      .a_in_empty_n(tail_general_a_in_bind_empty_n[0]),
      .a_in_read(tail_general_a_in_bind_read[0]),
      .a_out_din(tail_general_a_out_a_in_din[8]),
      .a_out_full_n(tail_general_a_out_a_in_full_n[8]),
      .a_out_write(tail_general_a_out_a_in_write[8]),
      .b_in_dout(tail_general_b_in_bind_dout[0]),
      .b_in_empty_n(tail_general_b_in_bind_empty_n[0]),
      .b_in_read(tail_general_b_in_bind_read[0]),
      .b_out_din(tail_general_b_out_a_in_din[0]),
      .b_out_full_n(tail_general_b_out_a_in_full_n[0]),
      .b_out_write(tail_general_b_out_a_in_write[0]));
  // role tail_general_r1: 1 instance(s)
  tail_general_r1 u_tail_general_r1_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_in_bind_dout[1]),
      .a_in_empty_n(tail_general_a_in_bind_empty_n[1]),
      .a_in_read(tail_general_a_in_bind_read[1]),
      .a_out_din(tail_general_a_out_a_in_din[9]),
      .a_out_full_n(tail_general_a_out_a_in_full_n[9]),
      .a_out_write(tail_general_a_out_a_in_write[9]),
      .b_in_dout(tail_general_b_in_bind_dout[1]),
      .b_in_empty_n(tail_general_b_in_bind_empty_n[1]),
      .b_in_read(tail_general_b_in_bind_read[1]),
      .b_out_din(tail_general_b_out_a_in_din[1]),
      .b_out_full_n(tail_general_b_out_a_in_full_n[1]),
      .b_out_write(tail_general_b_out_a_in_write[1]));
  // role tail_general_r2: 1 instance(s)
  tail_general_r2 u_tail_general_r2_0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_in_bind_dout[2]),
      .a_in_empty_n(tail_general_a_in_bind_empty_n[2]),
      .a_in_read(tail_general_a_in_bind_read[2]),
      .a_out_din(tail_general_a_out_a_in_din[10]),
      .a_out_full_n(tail_general_a_out_a_in_full_n[10]),
      .a_out_write(tail_general_a_out_a_in_write[10]),
      .b_in_dout(tail_general_b_in_bind_dout[2]),
      .b_in_empty_n(tail_general_b_in_bind_empty_n[2]),
      .b_in_read(tail_general_b_in_bind_read[2]),
      .b_out_din(tail_general_b_out_a_in_din[2]),
      .b_out_full_n(tail_general_b_out_a_in_full_n[2]),
      .b_out_write(tail_general_b_out_a_in_write[2]));
  // role tail_general_r3: 1 instance(s)
  tail_general_r3 u_tail_general_r3_0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_in_bind_dout[3]),
      .a_in_empty_n(tail_general_a_in_bind_empty_n[3]),
      .a_in_read(tail_general_a_in_bind_read[3]),
      .a_out_din(tail_general_a_out_a_in_din[11]),
      .a_out_full_n(tail_general_a_out_a_in_full_n[11]),
      .a_out_write(tail_general_a_out_a_in_write[11]),
      .b_in_dout(tail_general_b_in_bind_dout[3]),
      .b_in_empty_n(tail_general_b_in_bind_empty_n[3]),
      .b_in_read(tail_general_b_in_bind_read[3]),
      .b_out_din(tail_general_b_out_a_in_din[3]),
      .b_out_full_n(tail_general_b_out_a_in_full_n[3]),
      .b_out_write(tail_general_b_out_a_in_write[3]));
  // role tail_general_r4: 1 instance(s)
  tail_general_r4 u_tail_general_r4_0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_in_bind_dout[4]),
      .a_in_empty_n(tail_general_a_in_bind_empty_n[4]),
      .a_in_read(tail_general_a_in_bind_read[4]),
      .a_out_din(tail_general_a_out_b_in_din[0]),
      .a_out_full_n(tail_general_a_out_b_in_full_n[0]),
      .a_out_write(tail_general_a_out_b_in_write[0]),
      .b_in_dout(tail_general_b_in_bind_dout[4]),
      .b_in_empty_n(tail_general_b_in_bind_empty_n[4]),
      .b_in_read(tail_general_b_in_bind_read[4]),
      .b_out_din(tail_general_b_out_b_in_din[12]),
      .b_out_full_n(tail_general_b_out_b_in_full_n[12]),
      .b_out_write(tail_general_b_out_b_in_write[12]));
  // role tail_general_r5: 1 instance(s)
  tail_general_r5 u_tail_general_r5_0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_in_bind_dout[5]),
      .a_in_empty_n(tail_general_a_in_bind_empty_n[5]),
      .a_in_read(tail_general_a_in_bind_read[5]),
      .a_out_din(tail_general_a_out_b_in_din[1]),
      .a_out_full_n(tail_general_a_out_b_in_full_n[1]),
      .a_out_write(tail_general_a_out_b_in_write[1]),
      .b_in_dout(tail_general_b_in_bind_dout[5]),
      .b_in_empty_n(tail_general_b_in_bind_empty_n[5]),
      .b_in_read(tail_general_b_in_bind_read[5]),
      .b_out_din(tail_general_b_out_b_in_din[13]),
      .b_out_full_n(tail_general_b_out_b_in_full_n[13]),
      .b_out_write(tail_general_b_out_b_in_write[13]));
  // role tail_general_r6: 1 instance(s)
  tail_general_r6 u_tail_general_r6_0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_in_bind_dout[6]),
      .a_in_empty_n(tail_general_a_in_bind_empty_n[6]),
      .a_in_read(tail_general_a_in_bind_read[6]),
      .a_out_din(tail_general_a_out_b_in_din[2]),
      .a_out_full_n(tail_general_a_out_b_in_full_n[2]),
      .a_out_write(tail_general_a_out_b_in_write[2]),
      .b_in_dout(tail_general_b_in_bind_dout[6]),
      .b_in_empty_n(tail_general_b_in_bind_empty_n[6]),
      .b_in_read(tail_general_b_in_bind_read[6]),
      .b_out_din(tail_general_b_out_b_in_din[14]),
      .b_out_full_n(tail_general_b_out_b_in_full_n[14]),
      .b_out_write(tail_general_b_out_b_in_write[14]));
  // role tail_general_r7: 1 instance(s)
  tail_general_r7 u_tail_general_r7_0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_in_bind_dout[7]),
      .a_in_empty_n(tail_general_a_in_bind_empty_n[7]),
      .a_in_read(tail_general_a_in_bind_read[7]),
      .a_out_din(tail_general_a_out_b_in_din[3]),
      .a_out_full_n(tail_general_a_out_b_in_full_n[3]),
      .a_out_write(tail_general_a_out_b_in_write[3]),
      .b_in_dout(tail_general_b_in_bind_dout[7]),
      .b_in_empty_n(tail_general_b_in_bind_empty_n[7]),
      .b_in_read(tail_general_b_in_bind_read[7]),
      .b_out_din(tail_general_b_out_b_in_din[15]),
      .b_out_full_n(tail_general_b_out_b_in_full_n[15]),
      .b_out_write(tail_general_b_out_b_in_write[15]));
  // role tail_general_r8: 1 instance(s)
  tail_general_r8 u_tail_general_r8_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_out_a_in_dout[8]),
      .a_in_empty_n(tail_general_a_out_a_in_empty_n[8]),
      .a_in_read(tail_general_a_out_a_in_read[8]),
      .a_out_din(tail_general_a_out_a_in_din[16]),
      .a_out_full_n(tail_general_a_out_a_in_full_n[16]),
      .a_out_write(tail_general_a_out_a_in_write[16]),
      .b_in_dout(tail_general_a_out_b_in_dout[0]),
      .b_in_empty_n(tail_general_a_out_b_in_empty_n[0]),
      .b_in_read(tail_general_a_out_b_in_read[0]),
      .b_out_din(tail_general_b_out_a_in_din[4]),
      .b_out_full_n(tail_general_b_out_a_in_full_n[4]),
      .b_out_write(tail_general_b_out_a_in_write[4]));
  // role tail_general_r9: 1 instance(s)
  tail_general_r9 u_tail_general_r9_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_out_a_in_dout[9]),
      .a_in_empty_n(tail_general_a_out_a_in_empty_n[9]),
      .a_in_read(tail_general_a_out_a_in_read[9]),
      .a_out_din(tail_general_a_out_a_in_din[17]),
      .a_out_full_n(tail_general_a_out_a_in_full_n[17]),
      .a_out_write(tail_general_a_out_a_in_write[17]),
      .b_in_dout(tail_general_a_out_b_in_dout[1]),
      .b_in_empty_n(tail_general_a_out_b_in_empty_n[1]),
      .b_in_read(tail_general_a_out_b_in_read[1]),
      .b_out_din(tail_general_b_out_a_in_din[5]),
      .b_out_full_n(tail_general_b_out_a_in_full_n[5]),
      .b_out_write(tail_general_b_out_a_in_write[5]));
  // role tail_general_r10: 1 instance(s)
  tail_general_r10 u_tail_general_r10_1_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_out_a_in_dout[10]),
      .a_in_empty_n(tail_general_a_out_a_in_empty_n[10]),
      .a_in_read(tail_general_a_out_a_in_read[10]),
      .a_out_din(tail_general_a_out_b_in_din[4]),
      .a_out_full_n(tail_general_a_out_b_in_full_n[4]),
      .a_out_write(tail_general_a_out_b_in_write[4]),
      .b_in_dout(tail_general_a_out_b_in_dout[2]),
      .b_in_empty_n(tail_general_a_out_b_in_empty_n[2]),
      .b_in_read(tail_general_a_out_b_in_read[2]),
      .b_out_din(tail_general_b_out_b_in_din[18]),
      .b_out_full_n(tail_general_b_out_b_in_full_n[18]),
      .b_out_write(tail_general_b_out_b_in_write[18]));
  // role tail_general_r11: 1 instance(s)
  tail_general_r11 u_tail_general_r11_1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_out_a_in_dout[11]),
      .a_in_empty_n(tail_general_a_out_a_in_empty_n[11]),
      .a_in_read(tail_general_a_out_a_in_read[11]),
      .a_out_din(tail_general_a_out_b_in_din[5]),
      .a_out_full_n(tail_general_a_out_b_in_full_n[5]),
      .a_out_write(tail_general_a_out_b_in_write[5]),
      .b_in_dout(tail_general_a_out_b_in_dout[3]),
      .b_in_empty_n(tail_general_a_out_b_in_empty_n[3]),
      .b_in_read(tail_general_a_out_b_in_read[3]),
      .b_out_din(tail_general_b_out_b_in_din[19]),
      .b_out_full_n(tail_general_b_out_b_in_full_n[19]),
      .b_out_write(tail_general_b_out_b_in_write[19]));
  // role tail_general_r12: 1 instance(s)
  tail_general_r12 u_tail_general_r12_1_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_b_out_a_in_dout[0]),
      .a_in_empty_n(tail_general_b_out_a_in_empty_n[0]),
      .a_in_read(tail_general_b_out_a_in_read[0]),
      .a_out_din(tail_general_a_out_a_in_din[20]),
      .a_out_full_n(tail_general_a_out_a_in_full_n[20]),
      .a_out_write(tail_general_a_out_a_in_write[20]),
      .b_in_dout(tail_general_b_out_b_in_dout[12]),
      .b_in_empty_n(tail_general_b_out_b_in_empty_n[12]),
      .b_in_read(tail_general_b_out_b_in_read[12]),
      .b_out_din(tail_general_b_out_a_in_din[6]),
      .b_out_full_n(tail_general_b_out_a_in_full_n[6]),
      .b_out_write(tail_general_b_out_a_in_write[6]));
  // role tail_general_r13: 1 instance(s)
  tail_general_r13 u_tail_general_r13_1_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_b_out_a_in_dout[1]),
      .a_in_empty_n(tail_general_b_out_a_in_empty_n[1]),
      .a_in_read(tail_general_b_out_a_in_read[1]),
      .a_out_din(tail_general_a_out_a_in_din[21]),
      .a_out_full_n(tail_general_a_out_a_in_full_n[21]),
      .a_out_write(tail_general_a_out_a_in_write[21]),
      .b_in_dout(tail_general_b_out_b_in_dout[13]),
      .b_in_empty_n(tail_general_b_out_b_in_empty_n[13]),
      .b_in_read(tail_general_b_out_b_in_read[13]),
      .b_out_din(tail_general_b_out_a_in_din[7]),
      .b_out_full_n(tail_general_b_out_a_in_full_n[7]),
      .b_out_write(tail_general_b_out_a_in_write[7]));
  // role tail_general_r14: 1 instance(s)
  tail_general_r14 u_tail_general_r14_1_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_b_out_a_in_dout[2]),
      .a_in_empty_n(tail_general_b_out_a_in_empty_n[2]),
      .a_in_read(tail_general_b_out_a_in_read[2]),
      .a_out_din(tail_general_a_out_b_in_din[6]),
      .a_out_full_n(tail_general_a_out_b_in_full_n[6]),
      .a_out_write(tail_general_a_out_b_in_write[6]),
      .b_in_dout(tail_general_b_out_b_in_dout[14]),
      .b_in_empty_n(tail_general_b_out_b_in_empty_n[14]),
      .b_in_read(tail_general_b_out_b_in_read[14]),
      .b_out_din(tail_general_b_out_b_in_din[22]),
      .b_out_full_n(tail_general_b_out_b_in_full_n[22]),
      .b_out_write(tail_general_b_out_b_in_write[22]));
  // role tail_general_r15: 1 instance(s)
  tail_general_r15 u_tail_general_r15_1_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_b_out_a_in_dout[3]),
      .a_in_empty_n(tail_general_b_out_a_in_empty_n[3]),
      .a_in_read(tail_general_b_out_a_in_read[3]),
      .a_out_din(tail_general_a_out_b_in_din[7]),
      .a_out_full_n(tail_general_a_out_b_in_full_n[7]),
      .a_out_write(tail_general_a_out_b_in_write[7]),
      .b_in_dout(tail_general_b_out_b_in_dout[15]),
      .b_in_empty_n(tail_general_b_out_b_in_empty_n[15]),
      .b_in_read(tail_general_b_out_b_in_read[15]),
      .b_out_din(tail_general_b_out_b_in_din[23]),
      .b_out_full_n(tail_general_b_out_b_in_full_n[23]),
      .b_out_write(tail_general_b_out_b_in_write[23]));
  // role tail_general_r16: 1 instance(s)
  tail_general_r16 u_tail_general_r16_2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_out_a_in_dout[16]),
      .a_in_empty_n(tail_general_a_out_a_in_empty_n[16]),
      .a_in_read(tail_general_a_out_a_in_read[16]),
      .a_out_din(tail_general_a_out_a_in_din[24]),
      .a_out_full_n(tail_general_a_out_a_in_full_n[24]),
      .a_out_write(tail_general_a_out_a_in_write[24]),
      .b_in_dout(tail_general_a_out_b_in_dout[4]),
      .b_in_empty_n(tail_general_a_out_b_in_empty_n[4]),
      .b_in_read(tail_general_a_out_b_in_read[4]),
      .b_out_din(tail_general_b_out_a_in_din[8]),
      .b_out_full_n(tail_general_b_out_a_in_full_n[8]),
      .b_out_write(tail_general_b_out_a_in_write[8]));
  // role tail_general_r17: 1 instance(s)
  tail_general_r17 u_tail_general_r17_2_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_out_a_in_dout[17]),
      .a_in_empty_n(tail_general_a_out_a_in_empty_n[17]),
      .a_in_read(tail_general_a_out_a_in_read[17]),
      .a_out_din(tail_general_a_out_b_in_din[8]),
      .a_out_full_n(tail_general_a_out_b_in_full_n[8]),
      .a_out_write(tail_general_a_out_b_in_write[8]),
      .b_in_dout(tail_general_a_out_b_in_dout[5]),
      .b_in_empty_n(tail_general_a_out_b_in_empty_n[5]),
      .b_in_read(tail_general_a_out_b_in_read[5]),
      .b_out_din(tail_general_b_out_b_in_din[25]),
      .b_out_full_n(tail_general_b_out_b_in_full_n[25]),
      .b_out_write(tail_general_b_out_b_in_write[25]));
  // role tail_general_r18: 1 instance(s)
  tail_general_r18 u_tail_general_r18_2_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_b_out_a_in_dout[4]),
      .a_in_empty_n(tail_general_b_out_a_in_empty_n[4]),
      .a_in_read(tail_general_b_out_a_in_read[4]),
      .a_out_din(tail_general_a_out_a_in_din[26]),
      .a_out_full_n(tail_general_a_out_a_in_full_n[26]),
      .a_out_write(tail_general_a_out_a_in_write[26]),
      .b_in_dout(tail_general_b_out_b_in_dout[18]),
      .b_in_empty_n(tail_general_b_out_b_in_empty_n[18]),
      .b_in_read(tail_general_b_out_b_in_read[18]),
      .b_out_din(tail_general_b_out_a_in_din[9]),
      .b_out_full_n(tail_general_b_out_a_in_full_n[9]),
      .b_out_write(tail_general_b_out_a_in_write[9]));
  // role tail_general_r19: 1 instance(s)
  tail_general_r19 u_tail_general_r19_2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_b_out_a_in_dout[5]),
      .a_in_empty_n(tail_general_b_out_a_in_empty_n[5]),
      .a_in_read(tail_general_b_out_a_in_read[5]),
      .a_out_din(tail_general_a_out_b_in_din[9]),
      .a_out_full_n(tail_general_a_out_b_in_full_n[9]),
      .a_out_write(tail_general_a_out_b_in_write[9]),
      .b_in_dout(tail_general_b_out_b_in_dout[19]),
      .b_in_empty_n(tail_general_b_out_b_in_empty_n[19]),
      .b_in_read(tail_general_b_out_b_in_read[19]),
      .b_out_din(tail_general_b_out_b_in_din[27]),
      .b_out_full_n(tail_general_b_out_b_in_full_n[27]),
      .b_out_write(tail_general_b_out_b_in_write[27]));
  // role tail_general_r20: 1 instance(s)
  tail_general_r20 u_tail_general_r20_2_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_out_a_in_dout[20]),
      .a_in_empty_n(tail_general_a_out_a_in_empty_n[20]),
      .a_in_read(tail_general_a_out_a_in_read[20]),
      .a_out_din(tail_general_a_out_a_in_din[28]),
      .a_out_full_n(tail_general_a_out_a_in_full_n[28]),
      .a_out_write(tail_general_a_out_a_in_write[28]),
      .b_in_dout(tail_general_a_out_b_in_dout[6]),
      .b_in_empty_n(tail_general_a_out_b_in_empty_n[6]),
      .b_in_read(tail_general_a_out_b_in_read[6]),
      .b_out_din(tail_general_b_out_a_in_din[10]),
      .b_out_full_n(tail_general_b_out_a_in_full_n[10]),
      .b_out_write(tail_general_b_out_a_in_write[10]));
  // role tail_general_r21: 1 instance(s)
  tail_general_r21 u_tail_general_r21_2_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_out_a_in_dout[21]),
      .a_in_empty_n(tail_general_a_out_a_in_empty_n[21]),
      .a_in_read(tail_general_a_out_a_in_read[21]),
      .a_out_din(tail_general_a_out_b_in_din[10]),
      .a_out_full_n(tail_general_a_out_b_in_full_n[10]),
      .a_out_write(tail_general_a_out_b_in_write[10]),
      .b_in_dout(tail_general_a_out_b_in_dout[7]),
      .b_in_empty_n(tail_general_a_out_b_in_empty_n[7]),
      .b_in_read(tail_general_a_out_b_in_read[7]),
      .b_out_din(tail_general_b_out_b_in_din[29]),
      .b_out_full_n(tail_general_b_out_b_in_full_n[29]),
      .b_out_write(tail_general_b_out_b_in_write[29]));
  // role tail_general_r22: 1 instance(s)
  tail_general_r22 u_tail_general_r22_2_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_b_out_a_in_dout[6]),
      .a_in_empty_n(tail_general_b_out_a_in_empty_n[6]),
      .a_in_read(tail_general_b_out_a_in_read[6]),
      .a_out_din(tail_general_a_out_a_in_din[30]),
      .a_out_full_n(tail_general_a_out_a_in_full_n[30]),
      .a_out_write(tail_general_a_out_a_in_write[30]),
      .b_in_dout(tail_general_b_out_b_in_dout[22]),
      .b_in_empty_n(tail_general_b_out_b_in_empty_n[22]),
      .b_in_read(tail_general_b_out_b_in_read[22]),
      .b_out_din(tail_general_b_out_a_in_din[11]),
      .b_out_full_n(tail_general_b_out_a_in_full_n[11]),
      .b_out_write(tail_general_b_out_a_in_write[11]));
  // role tail_general_r23: 1 instance(s)
  tail_general_r23 u_tail_general_r23_2_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_b_out_a_in_dout[7]),
      .a_in_empty_n(tail_general_b_out_a_in_empty_n[7]),
      .a_in_read(tail_general_b_out_a_in_read[7]),
      .a_out_din(tail_general_a_out_b_in_din[11]),
      .a_out_full_n(tail_general_a_out_b_in_full_n[11]),
      .a_out_write(tail_general_a_out_b_in_write[11]),
      .b_in_dout(tail_general_b_out_b_in_dout[23]),
      .b_in_empty_n(tail_general_b_out_b_in_empty_n[23]),
      .b_in_read(tail_general_b_out_b_in_read[23]),
      .b_out_din(tail_general_b_out_b_in_din[31]),
      .b_out_full_n(tail_general_b_out_b_in_full_n[31]),
      .b_out_write(tail_general_b_out_b_in_write[31]));
  // role tail_general_r24: 1 instance(s)
  tail_general_r24 u_tail_general_r24_3_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_out_a_in_dout[24]),
      .a_in_empty_n(tail_general_a_out_a_in_empty_n[24]),
      .a_in_read(tail_general_a_out_a_in_read[24]),
      .a_out_din(reorder_a_in_bind_din[0]),
      .a_out_full_n(reorder_a_in_bind_full_n[0]),
      .a_out_write(reorder_a_in_bind_write[0]),
      .b_in_dout(tail_general_a_out_b_in_dout[8]),
      .b_in_empty_n(tail_general_a_out_b_in_empty_n[8]),
      .b_in_read(tail_general_a_out_b_in_read[8]),
      .b_out_din(reorder_b_in_bind_din[0]),
      .b_out_full_n(reorder_b_in_bind_full_n[0]),
      .b_out_write(reorder_b_in_bind_write[0]));
  // role tail_general_r25: 1 instance(s)
  tail_general_r25 u_tail_general_r25_3_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_b_out_a_in_dout[8]),
      .a_in_empty_n(tail_general_b_out_a_in_empty_n[8]),
      .a_in_read(tail_general_b_out_a_in_read[8]),
      .a_out_din(reorder_a_in_bind_din[1]),
      .a_out_full_n(reorder_a_in_bind_full_n[1]),
      .a_out_write(reorder_a_in_bind_write[1]),
      .b_in_dout(tail_general_b_out_b_in_dout[25]),
      .b_in_empty_n(tail_general_b_out_b_in_empty_n[25]),
      .b_in_read(tail_general_b_out_b_in_read[25]),
      .b_out_din(reorder_b_in_bind_din[1]),
      .b_out_full_n(reorder_b_in_bind_full_n[1]),
      .b_out_write(reorder_b_in_bind_write[1]));
  // role tail_general_r26: 1 instance(s)
  tail_general_r26 u_tail_general_r26_3_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_out_a_in_dout[26]),
      .a_in_empty_n(tail_general_a_out_a_in_empty_n[26]),
      .a_in_read(tail_general_a_out_a_in_read[26]),
      .a_out_din(reorder_a_in_bind_din[2]),
      .a_out_full_n(reorder_a_in_bind_full_n[2]),
      .a_out_write(reorder_a_in_bind_write[2]),
      .b_in_dout(tail_general_a_out_b_in_dout[9]),
      .b_in_empty_n(tail_general_a_out_b_in_empty_n[9]),
      .b_in_read(tail_general_a_out_b_in_read[9]),
      .b_out_din(reorder_b_in_bind_din[2]),
      .b_out_full_n(reorder_b_in_bind_full_n[2]),
      .b_out_write(reorder_b_in_bind_write[2]));
  // role tail_general_r27: 1 instance(s)
  tail_general_r27 u_tail_general_r27_3_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_b_out_a_in_dout[9]),
      .a_in_empty_n(tail_general_b_out_a_in_empty_n[9]),
      .a_in_read(tail_general_b_out_a_in_read[9]),
      .a_out_din(reorder_a_in_bind_din[3]),
      .a_out_full_n(reorder_a_in_bind_full_n[3]),
      .a_out_write(reorder_a_in_bind_write[3]),
      .b_in_dout(tail_general_b_out_b_in_dout[27]),
      .b_in_empty_n(tail_general_b_out_b_in_empty_n[27]),
      .b_in_read(tail_general_b_out_b_in_read[27]),
      .b_out_din(reorder_b_in_bind_din[3]),
      .b_out_full_n(reorder_b_in_bind_full_n[3]),
      .b_out_write(reorder_b_in_bind_write[3]));
  // role tail_general_r28: 1 instance(s)
  tail_general_r28 u_tail_general_r28_3_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_out_a_in_dout[28]),
      .a_in_empty_n(tail_general_a_out_a_in_empty_n[28]),
      .a_in_read(tail_general_a_out_a_in_read[28]),
      .a_out_din(reorder_a_in_bind_din[4]),
      .a_out_full_n(reorder_a_in_bind_full_n[4]),
      .a_out_write(reorder_a_in_bind_write[4]),
      .b_in_dout(tail_general_a_out_b_in_dout[10]),
      .b_in_empty_n(tail_general_a_out_b_in_empty_n[10]),
      .b_in_read(tail_general_a_out_b_in_read[10]),
      .b_out_din(reorder_b_in_bind_din[4]),
      .b_out_full_n(reorder_b_in_bind_full_n[4]),
      .b_out_write(reorder_b_in_bind_write[4]));
  // role tail_general_r29: 1 instance(s)
  tail_general_r29 u_tail_general_r29_3_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_b_out_a_in_dout[10]),
      .a_in_empty_n(tail_general_b_out_a_in_empty_n[10]),
      .a_in_read(tail_general_b_out_a_in_read[10]),
      .a_out_din(reorder_a_in_bind_din[5]),
      .a_out_full_n(reorder_a_in_bind_full_n[5]),
      .a_out_write(reorder_a_in_bind_write[5]),
      .b_in_dout(tail_general_b_out_b_in_dout[29]),
      .b_in_empty_n(tail_general_b_out_b_in_empty_n[29]),
      .b_in_read(tail_general_b_out_b_in_read[29]),
      .b_out_din(reorder_b_in_bind_din[5]),
      .b_out_full_n(reorder_b_in_bind_full_n[5]),
      .b_out_write(reorder_b_in_bind_write[5]));
  // role tail_general_r30: 1 instance(s)
  tail_general_r30 u_tail_general_r30_3_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_out_a_in_dout[30]),
      .a_in_empty_n(tail_general_a_out_a_in_empty_n[30]),
      .a_in_read(tail_general_a_out_a_in_read[30]),
      .a_out_din(reorder_a_in_bind_din[6]),
      .a_out_full_n(reorder_a_in_bind_full_n[6]),
      .a_out_write(reorder_a_in_bind_write[6]),
      .b_in_dout(tail_general_a_out_b_in_dout[11]),
      .b_in_empty_n(tail_general_a_out_b_in_empty_n[11]),
      .b_in_read(tail_general_a_out_b_in_read[11]),
      .b_out_din(reorder_b_in_bind_din[6]),
      .b_out_full_n(reorder_b_in_bind_full_n[6]),
      .b_out_write(reorder_b_in_bind_write[6]));
  // role tail_general_r31: 1 instance(s)
  tail_general_r31 u_tail_general_r31_3_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_b_out_a_in_dout[11]),
      .a_in_empty_n(tail_general_b_out_a_in_empty_n[11]),
      .a_in_read(tail_general_b_out_a_in_read[11]),
      .a_out_din(reorder_a_in_bind_din[7]),
      .a_out_full_n(reorder_a_in_bind_full_n[7]),
      .a_out_write(reorder_a_in_bind_write[7]),
      .b_in_dout(tail_general_b_out_b_in_dout[31]),
      .b_in_empty_n(tail_general_b_out_b_in_empty_n[31]),
      .b_in_read(tail_general_b_out_b_in_read[31]),
      .b_out_din(reorder_b_in_bind_din[7]),
      .b_out_full_n(reorder_b_in_bind_full_n[7]),
      .b_out_write(reorder_b_in_bind_write[7]));
  // role reorder_r0: 8 instance(s)
  reorder_r0 u_reorder_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(reorder_a_in_bind_dout[0]),
      .a_in_empty_n(reorder_a_in_bind_empty_n[0]),
      .a_in_read(reorder_a_in_bind_read[0]),
      .a_out_din(reorder_a_out_bind_din[0]),
      .a_out_full_n(reorder_a_out_bind_full_n[0]),
      .a_out_write(reorder_a_out_bind_write[0]),
      .b_in_dout(reorder_b_in_bind_dout[0]),
      .b_in_empty_n(reorder_b_in_bind_empty_n[0]),
      .b_in_read(reorder_b_in_bind_read[0]),
      .b_out_din(reorder_b_out_bind_din[0]),
      .b_out_full_n(reorder_b_out_bind_full_n[0]),
      .b_out_write(reorder_b_out_bind_write[0]));
  reorder_r0 u_reorder_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(reorder_a_in_bind_dout[1]),
      .a_in_empty_n(reorder_a_in_bind_empty_n[1]),
      .a_in_read(reorder_a_in_bind_read[1]),
      .a_out_din(reorder_a_out_bind_din[1]),
      .a_out_full_n(reorder_a_out_bind_full_n[1]),
      .a_out_write(reorder_a_out_bind_write[1]),
      .b_in_dout(reorder_b_in_bind_dout[1]),
      .b_in_empty_n(reorder_b_in_bind_empty_n[1]),
      .b_in_read(reorder_b_in_bind_read[1]),
      .b_out_din(reorder_b_out_bind_din[1]),
      .b_out_full_n(reorder_b_out_bind_full_n[1]),
      .b_out_write(reorder_b_out_bind_write[1]));
  reorder_r0 u_reorder_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(reorder_a_in_bind_dout[2]),
      .a_in_empty_n(reorder_a_in_bind_empty_n[2]),
      .a_in_read(reorder_a_in_bind_read[2]),
      .a_out_din(reorder_a_out_bind_din[2]),
      .a_out_full_n(reorder_a_out_bind_full_n[2]),
      .a_out_write(reorder_a_out_bind_write[2]),
      .b_in_dout(reorder_b_in_bind_dout[2]),
      .b_in_empty_n(reorder_b_in_bind_empty_n[2]),
      .b_in_read(reorder_b_in_bind_read[2]),
      .b_out_din(reorder_b_out_bind_din[2]),
      .b_out_full_n(reorder_b_out_bind_full_n[2]),
      .b_out_write(reorder_b_out_bind_write[2]));
  reorder_r0 u_reorder_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(reorder_a_in_bind_dout[3]),
      .a_in_empty_n(reorder_a_in_bind_empty_n[3]),
      .a_in_read(reorder_a_in_bind_read[3]),
      .a_out_din(reorder_a_out_bind_din[3]),
      .a_out_full_n(reorder_a_out_bind_full_n[3]),
      .a_out_write(reorder_a_out_bind_write[3]),
      .b_in_dout(reorder_b_in_bind_dout[3]),
      .b_in_empty_n(reorder_b_in_bind_empty_n[3]),
      .b_in_read(reorder_b_in_bind_read[3]),
      .b_out_din(reorder_b_out_bind_din[3]),
      .b_out_full_n(reorder_b_out_bind_full_n[3]),
      .b_out_write(reorder_b_out_bind_write[3]));
  reorder_r0 u_reorder_r0_4 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(reorder_a_in_bind_dout[4]),
      .a_in_empty_n(reorder_a_in_bind_empty_n[4]),
      .a_in_read(reorder_a_in_bind_read[4]),
      .a_out_din(reorder_a_out_bind_din[4]),
      .a_out_full_n(reorder_a_out_bind_full_n[4]),
      .a_out_write(reorder_a_out_bind_write[4]),
      .b_in_dout(reorder_b_in_bind_dout[4]),
      .b_in_empty_n(reorder_b_in_bind_empty_n[4]),
      .b_in_read(reorder_b_in_bind_read[4]),
      .b_out_din(reorder_b_out_bind_din[4]),
      .b_out_full_n(reorder_b_out_bind_full_n[4]),
      .b_out_write(reorder_b_out_bind_write[4]));
  reorder_r0 u_reorder_r0_5 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(reorder_a_in_bind_dout[5]),
      .a_in_empty_n(reorder_a_in_bind_empty_n[5]),
      .a_in_read(reorder_a_in_bind_read[5]),
      .a_out_din(reorder_a_out_bind_din[5]),
      .a_out_full_n(reorder_a_out_bind_full_n[5]),
      .a_out_write(reorder_a_out_bind_write[5]),
      .b_in_dout(reorder_b_in_bind_dout[5]),
      .b_in_empty_n(reorder_b_in_bind_empty_n[5]),
      .b_in_read(reorder_b_in_bind_read[5]),
      .b_out_din(reorder_b_out_bind_din[5]),
      .b_out_full_n(reorder_b_out_bind_full_n[5]),
      .b_out_write(reorder_b_out_bind_write[5]));
  reorder_r0 u_reorder_r0_6 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(reorder_a_in_bind_dout[6]),
      .a_in_empty_n(reorder_a_in_bind_empty_n[6]),
      .a_in_read(reorder_a_in_bind_read[6]),
      .a_out_din(reorder_a_out_bind_din[6]),
      .a_out_full_n(reorder_a_out_bind_full_n[6]),
      .a_out_write(reorder_a_out_bind_write[6]),
      .b_in_dout(reorder_b_in_bind_dout[6]),
      .b_in_empty_n(reorder_b_in_bind_empty_n[6]),
      .b_in_read(reorder_b_in_bind_read[6]),
      .b_out_din(reorder_b_out_bind_din[6]),
      .b_out_full_n(reorder_b_out_bind_full_n[6]),
      .b_out_write(reorder_b_out_bind_write[6]));
  reorder_r0 u_reorder_r0_7 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(reorder_a_in_bind_dout[7]),
      .a_in_empty_n(reorder_a_in_bind_empty_n[7]),
      .a_in_read(reorder_a_in_bind_read[7]),
      .a_out_din(reorder_a_out_bind_din[7]),
      .a_out_full_n(reorder_a_out_bind_full_n[7]),
      .a_out_write(reorder_a_out_bind_write[7]),
      .b_in_dout(reorder_b_in_bind_dout[7]),
      .b_in_empty_n(reorder_b_in_bind_empty_n[7]),
      .b_in_read(reorder_b_in_bind_read[7]),
      .b_out_din(reorder_b_out_bind_din[7]),
      .b_out_full_n(reorder_b_out_bind_full_n[7]),
      .b_out_write(reorder_b_out_bind_write[7]));
endmodule
