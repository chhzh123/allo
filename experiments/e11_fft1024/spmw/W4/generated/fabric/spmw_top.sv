`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] perm0_u_in_bind_dout [0:1],
  input  wire perm0_u_in_bind_empty_n [0:1],
  output wire perm0_u_in_bind_read [0:1],
  input  wire [63:0] perm0_v_in_bind_dout [0:1],
  input  wire perm0_v_in_bind_empty_n [0:1],
  output wire perm0_v_in_bind_read [0:1],
  output wire [63:0] reorder_a_out_bind_din [0:1],
  output wire reorder_a_out_bind_write [0:1],
  input  wire reorder_a_out_bind_full_n [0:1],
  output wire [63:0] reorder_b_out_bind_din [0:1],
  output wire reorder_b_out_bind_write [0:1],
  input  wire reorder_b_out_bind_full_n [0:1]
);
  // family bfly0_a_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly0_a_in_bind_din [0:1];
  wire [63:0] bfly0_a_in_bind_dout [0:1];
  wire bfly0_a_in_bind_full_n [0:1];
  wire bfly0_a_in_bind_write [0:1];
  wire bfly0_a_in_bind_empty_n [0:1];
  wire bfly0_a_in_bind_read [0:1];
  genvar bfly0_a_in_bind_i;
  generate
    for (bfly0_a_in_bind_i = 0; bfly0_a_in_bind_i < 2; bfly0_a_in_bind_i = bfly0_a_in_bind_i + 1) begin : g_bfly0_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly0_a_in_bind_din[bfly0_a_in_bind_i]), .full_n(bfly0_a_in_bind_full_n[bfly0_a_in_bind_i]), .write(bfly0_a_in_bind_write[bfly0_a_in_bind_i]), .dout(bfly0_a_in_bind_dout[bfly0_a_in_bind_i]), .empty_n(bfly0_a_in_bind_empty_n[bfly0_a_in_bind_i]), .read(bfly0_a_in_bind_read[bfly0_a_in_bind_i]));
    end
  endgenerate
  // family bfly0_b_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly0_b_in_bind_din [0:1];
  wire [63:0] bfly0_b_in_bind_dout [0:1];
  wire bfly0_b_in_bind_full_n [0:1];
  wire bfly0_b_in_bind_write [0:1];
  wire bfly0_b_in_bind_empty_n [0:1];
  wire bfly0_b_in_bind_read [0:1];
  genvar bfly0_b_in_bind_i;
  generate
    for (bfly0_b_in_bind_i = 0; bfly0_b_in_bind_i < 2; bfly0_b_in_bind_i = bfly0_b_in_bind_i + 1) begin : g_bfly0_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly0_b_in_bind_din[bfly0_b_in_bind_i]), .full_n(bfly0_b_in_bind_full_n[bfly0_b_in_bind_i]), .write(bfly0_b_in_bind_write[bfly0_b_in_bind_i]), .dout(bfly0_b_in_bind_dout[bfly0_b_in_bind_i]), .empty_n(bfly0_b_in_bind_empty_n[bfly0_b_in_bind_i]), .read(bfly0_b_in_bind_read[bfly0_b_in_bind_i]));
    end
  endgenerate
  // family perm1_u_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm1_u_in_bind_din [0:1];
  wire [63:0] perm1_u_in_bind_dout [0:1];
  wire perm1_u_in_bind_full_n [0:1];
  wire perm1_u_in_bind_write [0:1];
  wire perm1_u_in_bind_empty_n [0:1];
  wire perm1_u_in_bind_read [0:1];
  genvar perm1_u_in_bind_i;
  generate
    for (perm1_u_in_bind_i = 0; perm1_u_in_bind_i < 2; perm1_u_in_bind_i = perm1_u_in_bind_i + 1) begin : g_perm1_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm1_u_in_bind_din[perm1_u_in_bind_i]), .full_n(perm1_u_in_bind_full_n[perm1_u_in_bind_i]), .write(perm1_u_in_bind_write[perm1_u_in_bind_i]), .dout(perm1_u_in_bind_dout[perm1_u_in_bind_i]), .empty_n(perm1_u_in_bind_empty_n[perm1_u_in_bind_i]), .read(perm1_u_in_bind_read[perm1_u_in_bind_i]));
    end
  endgenerate
  // family perm1_v_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm1_v_in_bind_din [0:1];
  wire [63:0] perm1_v_in_bind_dout [0:1];
  wire perm1_v_in_bind_full_n [0:1];
  wire perm1_v_in_bind_write [0:1];
  wire perm1_v_in_bind_empty_n [0:1];
  wire perm1_v_in_bind_read [0:1];
  genvar perm1_v_in_bind_i;
  generate
    for (perm1_v_in_bind_i = 0; perm1_v_in_bind_i < 2; perm1_v_in_bind_i = perm1_v_in_bind_i + 1) begin : g_perm1_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm1_v_in_bind_din[perm1_v_in_bind_i]), .full_n(perm1_v_in_bind_full_n[perm1_v_in_bind_i]), .write(perm1_v_in_bind_write[perm1_v_in_bind_i]), .dout(perm1_v_in_bind_dout[perm1_v_in_bind_i]), .empty_n(perm1_v_in_bind_empty_n[perm1_v_in_bind_i]), .read(perm1_v_in_bind_read[perm1_v_in_bind_i]));
    end
  endgenerate
  // family bfly1_a_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly1_a_in_bind_din [0:1];
  wire [63:0] bfly1_a_in_bind_dout [0:1];
  wire bfly1_a_in_bind_full_n [0:1];
  wire bfly1_a_in_bind_write [0:1];
  wire bfly1_a_in_bind_empty_n [0:1];
  wire bfly1_a_in_bind_read [0:1];
  genvar bfly1_a_in_bind_i;
  generate
    for (bfly1_a_in_bind_i = 0; bfly1_a_in_bind_i < 2; bfly1_a_in_bind_i = bfly1_a_in_bind_i + 1) begin : g_bfly1_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly1_a_in_bind_din[bfly1_a_in_bind_i]), .full_n(bfly1_a_in_bind_full_n[bfly1_a_in_bind_i]), .write(bfly1_a_in_bind_write[bfly1_a_in_bind_i]), .dout(bfly1_a_in_bind_dout[bfly1_a_in_bind_i]), .empty_n(bfly1_a_in_bind_empty_n[bfly1_a_in_bind_i]), .read(bfly1_a_in_bind_read[bfly1_a_in_bind_i]));
    end
  endgenerate
  // family bfly1_b_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly1_b_in_bind_din [0:1];
  wire [63:0] bfly1_b_in_bind_dout [0:1];
  wire bfly1_b_in_bind_full_n [0:1];
  wire bfly1_b_in_bind_write [0:1];
  wire bfly1_b_in_bind_empty_n [0:1];
  wire bfly1_b_in_bind_read [0:1];
  genvar bfly1_b_in_bind_i;
  generate
    for (bfly1_b_in_bind_i = 0; bfly1_b_in_bind_i < 2; bfly1_b_in_bind_i = bfly1_b_in_bind_i + 1) begin : g_bfly1_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly1_b_in_bind_din[bfly1_b_in_bind_i]), .full_n(bfly1_b_in_bind_full_n[bfly1_b_in_bind_i]), .write(bfly1_b_in_bind_write[bfly1_b_in_bind_i]), .dout(bfly1_b_in_bind_dout[bfly1_b_in_bind_i]), .empty_n(bfly1_b_in_bind_empty_n[bfly1_b_in_bind_i]), .read(bfly1_b_in_bind_read[bfly1_b_in_bind_i]));
    end
  endgenerate
  // family perm2_u_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm2_u_in_bind_din [0:1];
  wire [63:0] perm2_u_in_bind_dout [0:1];
  wire perm2_u_in_bind_full_n [0:1];
  wire perm2_u_in_bind_write [0:1];
  wire perm2_u_in_bind_empty_n [0:1];
  wire perm2_u_in_bind_read [0:1];
  genvar perm2_u_in_bind_i;
  generate
    for (perm2_u_in_bind_i = 0; perm2_u_in_bind_i < 2; perm2_u_in_bind_i = perm2_u_in_bind_i + 1) begin : g_perm2_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm2_u_in_bind_din[perm2_u_in_bind_i]), .full_n(perm2_u_in_bind_full_n[perm2_u_in_bind_i]), .write(perm2_u_in_bind_write[perm2_u_in_bind_i]), .dout(perm2_u_in_bind_dout[perm2_u_in_bind_i]), .empty_n(perm2_u_in_bind_empty_n[perm2_u_in_bind_i]), .read(perm2_u_in_bind_read[perm2_u_in_bind_i]));
    end
  endgenerate
  // family perm2_v_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm2_v_in_bind_din [0:1];
  wire [63:0] perm2_v_in_bind_dout [0:1];
  wire perm2_v_in_bind_full_n [0:1];
  wire perm2_v_in_bind_write [0:1];
  wire perm2_v_in_bind_empty_n [0:1];
  wire perm2_v_in_bind_read [0:1];
  genvar perm2_v_in_bind_i;
  generate
    for (perm2_v_in_bind_i = 0; perm2_v_in_bind_i < 2; perm2_v_in_bind_i = perm2_v_in_bind_i + 1) begin : g_perm2_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm2_v_in_bind_din[perm2_v_in_bind_i]), .full_n(perm2_v_in_bind_full_n[perm2_v_in_bind_i]), .write(perm2_v_in_bind_write[perm2_v_in_bind_i]), .dout(perm2_v_in_bind_dout[perm2_v_in_bind_i]), .empty_n(perm2_v_in_bind_empty_n[perm2_v_in_bind_i]), .read(perm2_v_in_bind_read[perm2_v_in_bind_i]));
    end
  endgenerate
  // family bfly2_a_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly2_a_in_bind_din [0:1];
  wire [63:0] bfly2_a_in_bind_dout [0:1];
  wire bfly2_a_in_bind_full_n [0:1];
  wire bfly2_a_in_bind_write [0:1];
  wire bfly2_a_in_bind_empty_n [0:1];
  wire bfly2_a_in_bind_read [0:1];
  genvar bfly2_a_in_bind_i;
  generate
    for (bfly2_a_in_bind_i = 0; bfly2_a_in_bind_i < 2; bfly2_a_in_bind_i = bfly2_a_in_bind_i + 1) begin : g_bfly2_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly2_a_in_bind_din[bfly2_a_in_bind_i]), .full_n(bfly2_a_in_bind_full_n[bfly2_a_in_bind_i]), .write(bfly2_a_in_bind_write[bfly2_a_in_bind_i]), .dout(bfly2_a_in_bind_dout[bfly2_a_in_bind_i]), .empty_n(bfly2_a_in_bind_empty_n[bfly2_a_in_bind_i]), .read(bfly2_a_in_bind_read[bfly2_a_in_bind_i]));
    end
  endgenerate
  // family bfly2_b_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly2_b_in_bind_din [0:1];
  wire [63:0] bfly2_b_in_bind_dout [0:1];
  wire bfly2_b_in_bind_full_n [0:1];
  wire bfly2_b_in_bind_write [0:1];
  wire bfly2_b_in_bind_empty_n [0:1];
  wire bfly2_b_in_bind_read [0:1];
  genvar bfly2_b_in_bind_i;
  generate
    for (bfly2_b_in_bind_i = 0; bfly2_b_in_bind_i < 2; bfly2_b_in_bind_i = bfly2_b_in_bind_i + 1) begin : g_bfly2_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly2_b_in_bind_din[bfly2_b_in_bind_i]), .full_n(bfly2_b_in_bind_full_n[bfly2_b_in_bind_i]), .write(bfly2_b_in_bind_write[bfly2_b_in_bind_i]), .dout(bfly2_b_in_bind_dout[bfly2_b_in_bind_i]), .empty_n(bfly2_b_in_bind_empty_n[bfly2_b_in_bind_i]), .read(bfly2_b_in_bind_read[bfly2_b_in_bind_i]));
    end
  endgenerate
  // family perm3_u_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm3_u_in_bind_din [0:1];
  wire [63:0] perm3_u_in_bind_dout [0:1];
  wire perm3_u_in_bind_full_n [0:1];
  wire perm3_u_in_bind_write [0:1];
  wire perm3_u_in_bind_empty_n [0:1];
  wire perm3_u_in_bind_read [0:1];
  genvar perm3_u_in_bind_i;
  generate
    for (perm3_u_in_bind_i = 0; perm3_u_in_bind_i < 2; perm3_u_in_bind_i = perm3_u_in_bind_i + 1) begin : g_perm3_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm3_u_in_bind_din[perm3_u_in_bind_i]), .full_n(perm3_u_in_bind_full_n[perm3_u_in_bind_i]), .write(perm3_u_in_bind_write[perm3_u_in_bind_i]), .dout(perm3_u_in_bind_dout[perm3_u_in_bind_i]), .empty_n(perm3_u_in_bind_empty_n[perm3_u_in_bind_i]), .read(perm3_u_in_bind_read[perm3_u_in_bind_i]));
    end
  endgenerate
  // family perm3_v_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm3_v_in_bind_din [0:1];
  wire [63:0] perm3_v_in_bind_dout [0:1];
  wire perm3_v_in_bind_full_n [0:1];
  wire perm3_v_in_bind_write [0:1];
  wire perm3_v_in_bind_empty_n [0:1];
  wire perm3_v_in_bind_read [0:1];
  genvar perm3_v_in_bind_i;
  generate
    for (perm3_v_in_bind_i = 0; perm3_v_in_bind_i < 2; perm3_v_in_bind_i = perm3_v_in_bind_i + 1) begin : g_perm3_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm3_v_in_bind_din[perm3_v_in_bind_i]), .full_n(perm3_v_in_bind_full_n[perm3_v_in_bind_i]), .write(perm3_v_in_bind_write[perm3_v_in_bind_i]), .dout(perm3_v_in_bind_dout[perm3_v_in_bind_i]), .empty_n(perm3_v_in_bind_empty_n[perm3_v_in_bind_i]), .read(perm3_v_in_bind_read[perm3_v_in_bind_i]));
    end
  endgenerate
  // family bfly3_a_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly3_a_in_bind_din [0:1];
  wire [63:0] bfly3_a_in_bind_dout [0:1];
  wire bfly3_a_in_bind_full_n [0:1];
  wire bfly3_a_in_bind_write [0:1];
  wire bfly3_a_in_bind_empty_n [0:1];
  wire bfly3_a_in_bind_read [0:1];
  genvar bfly3_a_in_bind_i;
  generate
    for (bfly3_a_in_bind_i = 0; bfly3_a_in_bind_i < 2; bfly3_a_in_bind_i = bfly3_a_in_bind_i + 1) begin : g_bfly3_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly3_a_in_bind_din[bfly3_a_in_bind_i]), .full_n(bfly3_a_in_bind_full_n[bfly3_a_in_bind_i]), .write(bfly3_a_in_bind_write[bfly3_a_in_bind_i]), .dout(bfly3_a_in_bind_dout[bfly3_a_in_bind_i]), .empty_n(bfly3_a_in_bind_empty_n[bfly3_a_in_bind_i]), .read(bfly3_a_in_bind_read[bfly3_a_in_bind_i]));
    end
  endgenerate
  // family bfly3_b_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly3_b_in_bind_din [0:1];
  wire [63:0] bfly3_b_in_bind_dout [0:1];
  wire bfly3_b_in_bind_full_n [0:1];
  wire bfly3_b_in_bind_write [0:1];
  wire bfly3_b_in_bind_empty_n [0:1];
  wire bfly3_b_in_bind_read [0:1];
  genvar bfly3_b_in_bind_i;
  generate
    for (bfly3_b_in_bind_i = 0; bfly3_b_in_bind_i < 2; bfly3_b_in_bind_i = bfly3_b_in_bind_i + 1) begin : g_bfly3_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly3_b_in_bind_din[bfly3_b_in_bind_i]), .full_n(bfly3_b_in_bind_full_n[bfly3_b_in_bind_i]), .write(bfly3_b_in_bind_write[bfly3_b_in_bind_i]), .dout(bfly3_b_in_bind_dout[bfly3_b_in_bind_i]), .empty_n(bfly3_b_in_bind_empty_n[bfly3_b_in_bind_i]), .read(bfly3_b_in_bind_read[bfly3_b_in_bind_i]));
    end
  endgenerate
  // family perm4_u_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm4_u_in_bind_din [0:1];
  wire [63:0] perm4_u_in_bind_dout [0:1];
  wire perm4_u_in_bind_full_n [0:1];
  wire perm4_u_in_bind_write [0:1];
  wire perm4_u_in_bind_empty_n [0:1];
  wire perm4_u_in_bind_read [0:1];
  genvar perm4_u_in_bind_i;
  generate
    for (perm4_u_in_bind_i = 0; perm4_u_in_bind_i < 2; perm4_u_in_bind_i = perm4_u_in_bind_i + 1) begin : g_perm4_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm4_u_in_bind_din[perm4_u_in_bind_i]), .full_n(perm4_u_in_bind_full_n[perm4_u_in_bind_i]), .write(perm4_u_in_bind_write[perm4_u_in_bind_i]), .dout(perm4_u_in_bind_dout[perm4_u_in_bind_i]), .empty_n(perm4_u_in_bind_empty_n[perm4_u_in_bind_i]), .read(perm4_u_in_bind_read[perm4_u_in_bind_i]));
    end
  endgenerate
  // family perm4_v_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm4_v_in_bind_din [0:1];
  wire [63:0] perm4_v_in_bind_dout [0:1];
  wire perm4_v_in_bind_full_n [0:1];
  wire perm4_v_in_bind_write [0:1];
  wire perm4_v_in_bind_empty_n [0:1];
  wire perm4_v_in_bind_read [0:1];
  genvar perm4_v_in_bind_i;
  generate
    for (perm4_v_in_bind_i = 0; perm4_v_in_bind_i < 2; perm4_v_in_bind_i = perm4_v_in_bind_i + 1) begin : g_perm4_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm4_v_in_bind_din[perm4_v_in_bind_i]), .full_n(perm4_v_in_bind_full_n[perm4_v_in_bind_i]), .write(perm4_v_in_bind_write[perm4_v_in_bind_i]), .dout(perm4_v_in_bind_dout[perm4_v_in_bind_i]), .empty_n(perm4_v_in_bind_empty_n[perm4_v_in_bind_i]), .read(perm4_v_in_bind_read[perm4_v_in_bind_i]));
    end
  endgenerate
  // family bfly4_a_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly4_a_in_bind_din [0:1];
  wire [63:0] bfly4_a_in_bind_dout [0:1];
  wire bfly4_a_in_bind_full_n [0:1];
  wire bfly4_a_in_bind_write [0:1];
  wire bfly4_a_in_bind_empty_n [0:1];
  wire bfly4_a_in_bind_read [0:1];
  genvar bfly4_a_in_bind_i;
  generate
    for (bfly4_a_in_bind_i = 0; bfly4_a_in_bind_i < 2; bfly4_a_in_bind_i = bfly4_a_in_bind_i + 1) begin : g_bfly4_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly4_a_in_bind_din[bfly4_a_in_bind_i]), .full_n(bfly4_a_in_bind_full_n[bfly4_a_in_bind_i]), .write(bfly4_a_in_bind_write[bfly4_a_in_bind_i]), .dout(bfly4_a_in_bind_dout[bfly4_a_in_bind_i]), .empty_n(bfly4_a_in_bind_empty_n[bfly4_a_in_bind_i]), .read(bfly4_a_in_bind_read[bfly4_a_in_bind_i]));
    end
  endgenerate
  // family bfly4_b_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly4_b_in_bind_din [0:1];
  wire [63:0] bfly4_b_in_bind_dout [0:1];
  wire bfly4_b_in_bind_full_n [0:1];
  wire bfly4_b_in_bind_write [0:1];
  wire bfly4_b_in_bind_empty_n [0:1];
  wire bfly4_b_in_bind_read [0:1];
  genvar bfly4_b_in_bind_i;
  generate
    for (bfly4_b_in_bind_i = 0; bfly4_b_in_bind_i < 2; bfly4_b_in_bind_i = bfly4_b_in_bind_i + 1) begin : g_bfly4_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly4_b_in_bind_din[bfly4_b_in_bind_i]), .full_n(bfly4_b_in_bind_full_n[bfly4_b_in_bind_i]), .write(bfly4_b_in_bind_write[bfly4_b_in_bind_i]), .dout(bfly4_b_in_bind_dout[bfly4_b_in_bind_i]), .empty_n(bfly4_b_in_bind_empty_n[bfly4_b_in_bind_i]), .read(bfly4_b_in_bind_read[bfly4_b_in_bind_i]));
    end
  endgenerate
  // family perm5_u_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm5_u_in_bind_din [0:1];
  wire [63:0] perm5_u_in_bind_dout [0:1];
  wire perm5_u_in_bind_full_n [0:1];
  wire perm5_u_in_bind_write [0:1];
  wire perm5_u_in_bind_empty_n [0:1];
  wire perm5_u_in_bind_read [0:1];
  genvar perm5_u_in_bind_i;
  generate
    for (perm5_u_in_bind_i = 0; perm5_u_in_bind_i < 2; perm5_u_in_bind_i = perm5_u_in_bind_i + 1) begin : g_perm5_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm5_u_in_bind_din[perm5_u_in_bind_i]), .full_n(perm5_u_in_bind_full_n[perm5_u_in_bind_i]), .write(perm5_u_in_bind_write[perm5_u_in_bind_i]), .dout(perm5_u_in_bind_dout[perm5_u_in_bind_i]), .empty_n(perm5_u_in_bind_empty_n[perm5_u_in_bind_i]), .read(perm5_u_in_bind_read[perm5_u_in_bind_i]));
    end
  endgenerate
  // family perm5_v_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm5_v_in_bind_din [0:1];
  wire [63:0] perm5_v_in_bind_dout [0:1];
  wire perm5_v_in_bind_full_n [0:1];
  wire perm5_v_in_bind_write [0:1];
  wire perm5_v_in_bind_empty_n [0:1];
  wire perm5_v_in_bind_read [0:1];
  genvar perm5_v_in_bind_i;
  generate
    for (perm5_v_in_bind_i = 0; perm5_v_in_bind_i < 2; perm5_v_in_bind_i = perm5_v_in_bind_i + 1) begin : g_perm5_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm5_v_in_bind_din[perm5_v_in_bind_i]), .full_n(perm5_v_in_bind_full_n[perm5_v_in_bind_i]), .write(perm5_v_in_bind_write[perm5_v_in_bind_i]), .dout(perm5_v_in_bind_dout[perm5_v_in_bind_i]), .empty_n(perm5_v_in_bind_empty_n[perm5_v_in_bind_i]), .read(perm5_v_in_bind_read[perm5_v_in_bind_i]));
    end
  endgenerate
  // family bfly5_a_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly5_a_in_bind_din [0:1];
  wire [63:0] bfly5_a_in_bind_dout [0:1];
  wire bfly5_a_in_bind_full_n [0:1];
  wire bfly5_a_in_bind_write [0:1];
  wire bfly5_a_in_bind_empty_n [0:1];
  wire bfly5_a_in_bind_read [0:1];
  genvar bfly5_a_in_bind_i;
  generate
    for (bfly5_a_in_bind_i = 0; bfly5_a_in_bind_i < 2; bfly5_a_in_bind_i = bfly5_a_in_bind_i + 1) begin : g_bfly5_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly5_a_in_bind_din[bfly5_a_in_bind_i]), .full_n(bfly5_a_in_bind_full_n[bfly5_a_in_bind_i]), .write(bfly5_a_in_bind_write[bfly5_a_in_bind_i]), .dout(bfly5_a_in_bind_dout[bfly5_a_in_bind_i]), .empty_n(bfly5_a_in_bind_empty_n[bfly5_a_in_bind_i]), .read(bfly5_a_in_bind_read[bfly5_a_in_bind_i]));
    end
  endgenerate
  // family bfly5_b_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly5_b_in_bind_din [0:1];
  wire [63:0] bfly5_b_in_bind_dout [0:1];
  wire bfly5_b_in_bind_full_n [0:1];
  wire bfly5_b_in_bind_write [0:1];
  wire bfly5_b_in_bind_empty_n [0:1];
  wire bfly5_b_in_bind_read [0:1];
  genvar bfly5_b_in_bind_i;
  generate
    for (bfly5_b_in_bind_i = 0; bfly5_b_in_bind_i < 2; bfly5_b_in_bind_i = bfly5_b_in_bind_i + 1) begin : g_bfly5_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly5_b_in_bind_din[bfly5_b_in_bind_i]), .full_n(bfly5_b_in_bind_full_n[bfly5_b_in_bind_i]), .write(bfly5_b_in_bind_write[bfly5_b_in_bind_i]), .dout(bfly5_b_in_bind_dout[bfly5_b_in_bind_i]), .empty_n(bfly5_b_in_bind_empty_n[bfly5_b_in_bind_i]), .read(bfly5_b_in_bind_read[bfly5_b_in_bind_i]));
    end
  endgenerate
  // family perm6_u_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm6_u_in_bind_din [0:1];
  wire [63:0] perm6_u_in_bind_dout [0:1];
  wire perm6_u_in_bind_full_n [0:1];
  wire perm6_u_in_bind_write [0:1];
  wire perm6_u_in_bind_empty_n [0:1];
  wire perm6_u_in_bind_read [0:1];
  genvar perm6_u_in_bind_i;
  generate
    for (perm6_u_in_bind_i = 0; perm6_u_in_bind_i < 2; perm6_u_in_bind_i = perm6_u_in_bind_i + 1) begin : g_perm6_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm6_u_in_bind_din[perm6_u_in_bind_i]), .full_n(perm6_u_in_bind_full_n[perm6_u_in_bind_i]), .write(perm6_u_in_bind_write[perm6_u_in_bind_i]), .dout(perm6_u_in_bind_dout[perm6_u_in_bind_i]), .empty_n(perm6_u_in_bind_empty_n[perm6_u_in_bind_i]), .read(perm6_u_in_bind_read[perm6_u_in_bind_i]));
    end
  endgenerate
  // family perm6_v_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm6_v_in_bind_din [0:1];
  wire [63:0] perm6_v_in_bind_dout [0:1];
  wire perm6_v_in_bind_full_n [0:1];
  wire perm6_v_in_bind_write [0:1];
  wire perm6_v_in_bind_empty_n [0:1];
  wire perm6_v_in_bind_read [0:1];
  genvar perm6_v_in_bind_i;
  generate
    for (perm6_v_in_bind_i = 0; perm6_v_in_bind_i < 2; perm6_v_in_bind_i = perm6_v_in_bind_i + 1) begin : g_perm6_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm6_v_in_bind_din[perm6_v_in_bind_i]), .full_n(perm6_v_in_bind_full_n[perm6_v_in_bind_i]), .write(perm6_v_in_bind_write[perm6_v_in_bind_i]), .dout(perm6_v_in_bind_dout[perm6_v_in_bind_i]), .empty_n(perm6_v_in_bind_empty_n[perm6_v_in_bind_i]), .read(perm6_v_in_bind_read[perm6_v_in_bind_i]));
    end
  endgenerate
  // family bfly6_a_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly6_a_in_bind_din [0:1];
  wire [63:0] bfly6_a_in_bind_dout [0:1];
  wire bfly6_a_in_bind_full_n [0:1];
  wire bfly6_a_in_bind_write [0:1];
  wire bfly6_a_in_bind_empty_n [0:1];
  wire bfly6_a_in_bind_read [0:1];
  genvar bfly6_a_in_bind_i;
  generate
    for (bfly6_a_in_bind_i = 0; bfly6_a_in_bind_i < 2; bfly6_a_in_bind_i = bfly6_a_in_bind_i + 1) begin : g_bfly6_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly6_a_in_bind_din[bfly6_a_in_bind_i]), .full_n(bfly6_a_in_bind_full_n[bfly6_a_in_bind_i]), .write(bfly6_a_in_bind_write[bfly6_a_in_bind_i]), .dout(bfly6_a_in_bind_dout[bfly6_a_in_bind_i]), .empty_n(bfly6_a_in_bind_empty_n[bfly6_a_in_bind_i]), .read(bfly6_a_in_bind_read[bfly6_a_in_bind_i]));
    end
  endgenerate
  // family bfly6_b_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly6_b_in_bind_din [0:1];
  wire [63:0] bfly6_b_in_bind_dout [0:1];
  wire bfly6_b_in_bind_full_n [0:1];
  wire bfly6_b_in_bind_write [0:1];
  wire bfly6_b_in_bind_empty_n [0:1];
  wire bfly6_b_in_bind_read [0:1];
  genvar bfly6_b_in_bind_i;
  generate
    for (bfly6_b_in_bind_i = 0; bfly6_b_in_bind_i < 2; bfly6_b_in_bind_i = bfly6_b_in_bind_i + 1) begin : g_bfly6_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly6_b_in_bind_din[bfly6_b_in_bind_i]), .full_n(bfly6_b_in_bind_full_n[bfly6_b_in_bind_i]), .write(bfly6_b_in_bind_write[bfly6_b_in_bind_i]), .dout(bfly6_b_in_bind_dout[bfly6_b_in_bind_i]), .empty_n(bfly6_b_in_bind_empty_n[bfly6_b_in_bind_i]), .read(bfly6_b_in_bind_read[bfly6_b_in_bind_i]));
    end
  endgenerate
  // family perm7_u_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm7_u_in_bind_din [0:1];
  wire [63:0] perm7_u_in_bind_dout [0:1];
  wire perm7_u_in_bind_full_n [0:1];
  wire perm7_u_in_bind_write [0:1];
  wire perm7_u_in_bind_empty_n [0:1];
  wire perm7_u_in_bind_read [0:1];
  genvar perm7_u_in_bind_i;
  generate
    for (perm7_u_in_bind_i = 0; perm7_u_in_bind_i < 2; perm7_u_in_bind_i = perm7_u_in_bind_i + 1) begin : g_perm7_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm7_u_in_bind_din[perm7_u_in_bind_i]), .full_n(perm7_u_in_bind_full_n[perm7_u_in_bind_i]), .write(perm7_u_in_bind_write[perm7_u_in_bind_i]), .dout(perm7_u_in_bind_dout[perm7_u_in_bind_i]), .empty_n(perm7_u_in_bind_empty_n[perm7_u_in_bind_i]), .read(perm7_u_in_bind_read[perm7_u_in_bind_i]));
    end
  endgenerate
  // family perm7_v_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm7_v_in_bind_din [0:1];
  wire [63:0] perm7_v_in_bind_dout [0:1];
  wire perm7_v_in_bind_full_n [0:1];
  wire perm7_v_in_bind_write [0:1];
  wire perm7_v_in_bind_empty_n [0:1];
  wire perm7_v_in_bind_read [0:1];
  genvar perm7_v_in_bind_i;
  generate
    for (perm7_v_in_bind_i = 0; perm7_v_in_bind_i < 2; perm7_v_in_bind_i = perm7_v_in_bind_i + 1) begin : g_perm7_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm7_v_in_bind_din[perm7_v_in_bind_i]), .full_n(perm7_v_in_bind_full_n[perm7_v_in_bind_i]), .write(perm7_v_in_bind_write[perm7_v_in_bind_i]), .dout(perm7_v_in_bind_dout[perm7_v_in_bind_i]), .empty_n(perm7_v_in_bind_empty_n[perm7_v_in_bind_i]), .read(perm7_v_in_bind_read[perm7_v_in_bind_i]));
    end
  endgenerate
  // family bfly7_a_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly7_a_in_bind_din [0:1];
  wire [63:0] bfly7_a_in_bind_dout [0:1];
  wire bfly7_a_in_bind_full_n [0:1];
  wire bfly7_a_in_bind_write [0:1];
  wire bfly7_a_in_bind_empty_n [0:1];
  wire bfly7_a_in_bind_read [0:1];
  genvar bfly7_a_in_bind_i;
  generate
    for (bfly7_a_in_bind_i = 0; bfly7_a_in_bind_i < 2; bfly7_a_in_bind_i = bfly7_a_in_bind_i + 1) begin : g_bfly7_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly7_a_in_bind_din[bfly7_a_in_bind_i]), .full_n(bfly7_a_in_bind_full_n[bfly7_a_in_bind_i]), .write(bfly7_a_in_bind_write[bfly7_a_in_bind_i]), .dout(bfly7_a_in_bind_dout[bfly7_a_in_bind_i]), .empty_n(bfly7_a_in_bind_empty_n[bfly7_a_in_bind_i]), .read(bfly7_a_in_bind_read[bfly7_a_in_bind_i]));
    end
  endgenerate
  // family bfly7_b_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] bfly7_b_in_bind_din [0:1];
  wire [63:0] bfly7_b_in_bind_dout [0:1];
  wire bfly7_b_in_bind_full_n [0:1];
  wire bfly7_b_in_bind_write [0:1];
  wire bfly7_b_in_bind_empty_n [0:1];
  wire bfly7_b_in_bind_read [0:1];
  genvar bfly7_b_in_bind_i;
  generate
    for (bfly7_b_in_bind_i = 0; bfly7_b_in_bind_i < 2; bfly7_b_in_bind_i = bfly7_b_in_bind_i + 1) begin : g_bfly7_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly7_b_in_bind_din[bfly7_b_in_bind_i]), .full_n(bfly7_b_in_bind_full_n[bfly7_b_in_bind_i]), .write(bfly7_b_in_bind_write[bfly7_b_in_bind_i]), .dout(bfly7_b_in_bind_dout[bfly7_b_in_bind_i]), .empty_n(bfly7_b_in_bind_empty_n[bfly7_b_in_bind_i]), .read(bfly7_b_in_bind_read[bfly7_b_in_bind_i]));
    end
  endgenerate
  // family perm8_u_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm8_u_in_bind_din [0:1];
  wire [63:0] perm8_u_in_bind_dout [0:1];
  wire perm8_u_in_bind_full_n [0:1];
  wire perm8_u_in_bind_write [0:1];
  wire perm8_u_in_bind_empty_n [0:1];
  wire perm8_u_in_bind_read [0:1];
  genvar perm8_u_in_bind_i;
  generate
    for (perm8_u_in_bind_i = 0; perm8_u_in_bind_i < 2; perm8_u_in_bind_i = perm8_u_in_bind_i + 1) begin : g_perm8_u_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm8_u_in_bind_din[perm8_u_in_bind_i]), .full_n(perm8_u_in_bind_full_n[perm8_u_in_bind_i]), .write(perm8_u_in_bind_write[perm8_u_in_bind_i]), .dout(perm8_u_in_bind_dout[perm8_u_in_bind_i]), .empty_n(perm8_u_in_bind_empty_n[perm8_u_in_bind_i]), .read(perm8_u_in_bind_read[perm8_u_in_bind_i]));
    end
  endgenerate
  // family perm8_v_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] perm8_v_in_bind_din [0:1];
  wire [63:0] perm8_v_in_bind_dout [0:1];
  wire perm8_v_in_bind_full_n [0:1];
  wire perm8_v_in_bind_write [0:1];
  wire perm8_v_in_bind_empty_n [0:1];
  wire perm8_v_in_bind_read [0:1];
  genvar perm8_v_in_bind_i;
  generate
    for (perm8_v_in_bind_i = 0; perm8_v_in_bind_i < 2; perm8_v_in_bind_i = perm8_v_in_bind_i + 1) begin : g_perm8_v_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(perm8_v_in_bind_din[perm8_v_in_bind_i]), .full_n(perm8_v_in_bind_full_n[perm8_v_in_bind_i]), .write(perm8_v_in_bind_write[perm8_v_in_bind_i]), .dout(perm8_v_in_bind_dout[perm8_v_in_bind_i]), .empty_n(perm8_v_in_bind_empty_n[perm8_v_in_bind_i]), .read(perm8_v_in_bind_read[perm8_v_in_bind_i]));
    end
  endgenerate
  // family tail_general_a_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] tail_general_a_in_bind_din [0:1];
  wire [63:0] tail_general_a_in_bind_dout [0:1];
  wire tail_general_a_in_bind_full_n [0:1];
  wire tail_general_a_in_bind_write [0:1];
  wire tail_general_a_in_bind_empty_n [0:1];
  wire tail_general_a_in_bind_read [0:1];
  genvar tail_general_a_in_bind_i;
  generate
    for (tail_general_a_in_bind_i = 0; tail_general_a_in_bind_i < 2; tail_general_a_in_bind_i = tail_general_a_in_bind_i + 1) begin : g_tail_general_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tail_general_a_in_bind_din[tail_general_a_in_bind_i]), .full_n(tail_general_a_in_bind_full_n[tail_general_a_in_bind_i]), .write(tail_general_a_in_bind_write[tail_general_a_in_bind_i]), .dout(tail_general_a_in_bind_dout[tail_general_a_in_bind_i]), .empty_n(tail_general_a_in_bind_empty_n[tail_general_a_in_bind_i]), .read(tail_general_a_in_bind_read[tail_general_a_in_bind_i]));
    end
  endgenerate
  // family tail_general_b_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] tail_general_b_in_bind_din [0:1];
  wire [63:0] tail_general_b_in_bind_dout [0:1];
  wire tail_general_b_in_bind_full_n [0:1];
  wire tail_general_b_in_bind_write [0:1];
  wire tail_general_b_in_bind_empty_n [0:1];
  wire tail_general_b_in_bind_read [0:1];
  genvar tail_general_b_in_bind_i;
  generate
    for (tail_general_b_in_bind_i = 0; tail_general_b_in_bind_i < 2; tail_general_b_in_bind_i = tail_general_b_in_bind_i + 1) begin : g_tail_general_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tail_general_b_in_bind_din[tail_general_b_in_bind_i]), .full_n(tail_general_b_in_bind_full_n[tail_general_b_in_bind_i]), .write(tail_general_b_in_bind_write[tail_general_b_in_bind_i]), .dout(tail_general_b_in_bind_dout[tail_general_b_in_bind_i]), .empty_n(tail_general_b_in_bind_empty_n[tail_general_b_in_bind_i]), .read(tail_general_b_in_bind_read[tail_general_b_in_bind_i]));
    end
  endgenerate
  // family tail_general_a_out_a_in: 4 channel(s), 64-bit, depth 8
  wire [63:0] tail_general_a_out_a_in_din [0:3];
  wire [63:0] tail_general_a_out_a_in_dout [0:3];
  wire tail_general_a_out_a_in_full_n [0:3];
  wire tail_general_a_out_a_in_write [0:3];
  wire tail_general_a_out_a_in_empty_n [0:3];
  wire tail_general_a_out_a_in_read [0:3];
  genvar tail_general_a_out_a_in_i;
  generate
    for (tail_general_a_out_a_in_i = 0; tail_general_a_out_a_in_i < 4; tail_general_a_out_a_in_i = tail_general_a_out_a_in_i + 1) begin : g_tail_general_a_out_a_in
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tail_general_a_out_a_in_din[tail_general_a_out_a_in_i]), .full_n(tail_general_a_out_a_in_full_n[tail_general_a_out_a_in_i]), .write(tail_general_a_out_a_in_write[tail_general_a_out_a_in_i]), .dout(tail_general_a_out_a_in_dout[tail_general_a_out_a_in_i]), .empty_n(tail_general_a_out_a_in_empty_n[tail_general_a_out_a_in_i]), .read(tail_general_a_out_a_in_read[tail_general_a_out_a_in_i]));
    end
  endgenerate
  // family tail_general_b_out_a_in: 4 channel(s), 64-bit, depth 8
  wire [63:0] tail_general_b_out_a_in_din [0:3];
  wire [63:0] tail_general_b_out_a_in_dout [0:3];
  wire tail_general_b_out_a_in_full_n [0:3];
  wire tail_general_b_out_a_in_write [0:3];
  wire tail_general_b_out_a_in_empty_n [0:3];
  wire tail_general_b_out_a_in_read [0:3];
  genvar tail_general_b_out_a_in_i;
  generate
    for (tail_general_b_out_a_in_i = 0; tail_general_b_out_a_in_i < 4; tail_general_b_out_a_in_i = tail_general_b_out_a_in_i + 1) begin : g_tail_general_b_out_a_in
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tail_general_b_out_a_in_din[tail_general_b_out_a_in_i]), .full_n(tail_general_b_out_a_in_full_n[tail_general_b_out_a_in_i]), .write(tail_general_b_out_a_in_write[tail_general_b_out_a_in_i]), .dout(tail_general_b_out_a_in_dout[tail_general_b_out_a_in_i]), .empty_n(tail_general_b_out_a_in_empty_n[tail_general_b_out_a_in_i]), .read(tail_general_b_out_a_in_read[tail_general_b_out_a_in_i]));
    end
  endgenerate
  // family tail_general_a_out_b_in: 4 channel(s), 64-bit, depth 8
  wire [63:0] tail_general_a_out_b_in_din [0:3];
  wire [63:0] tail_general_a_out_b_in_dout [0:3];
  wire tail_general_a_out_b_in_full_n [0:3];
  wire tail_general_a_out_b_in_write [0:3];
  wire tail_general_a_out_b_in_empty_n [0:3];
  wire tail_general_a_out_b_in_read [0:3];
  genvar tail_general_a_out_b_in_i;
  generate
    for (tail_general_a_out_b_in_i = 0; tail_general_a_out_b_in_i < 4; tail_general_a_out_b_in_i = tail_general_a_out_b_in_i + 1) begin : g_tail_general_a_out_b_in
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tail_general_a_out_b_in_din[tail_general_a_out_b_in_i]), .full_n(tail_general_a_out_b_in_full_n[tail_general_a_out_b_in_i]), .write(tail_general_a_out_b_in_write[tail_general_a_out_b_in_i]), .dout(tail_general_a_out_b_in_dout[tail_general_a_out_b_in_i]), .empty_n(tail_general_a_out_b_in_empty_n[tail_general_a_out_b_in_i]), .read(tail_general_a_out_b_in_read[tail_general_a_out_b_in_i]));
    end
  endgenerate
  // family tail_general_b_out_b_in: 4 channel(s), 64-bit, depth 8
  wire [63:0] tail_general_b_out_b_in_din [0:3];
  wire [63:0] tail_general_b_out_b_in_dout [0:3];
  wire tail_general_b_out_b_in_full_n [0:3];
  wire tail_general_b_out_b_in_write [0:3];
  wire tail_general_b_out_b_in_empty_n [0:3];
  wire tail_general_b_out_b_in_read [0:3];
  genvar tail_general_b_out_b_in_i;
  generate
    for (tail_general_b_out_b_in_i = 0; tail_general_b_out_b_in_i < 4; tail_general_b_out_b_in_i = tail_general_b_out_b_in_i + 1) begin : g_tail_general_b_out_b_in
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tail_general_b_out_b_in_din[tail_general_b_out_b_in_i]), .full_n(tail_general_b_out_b_in_full_n[tail_general_b_out_b_in_i]), .write(tail_general_b_out_b_in_write[tail_general_b_out_b_in_i]), .dout(tail_general_b_out_b_in_dout[tail_general_b_out_b_in_i]), .empty_n(tail_general_b_out_b_in_empty_n[tail_general_b_out_b_in_i]), .read(tail_general_b_out_b_in_read[tail_general_b_out_b_in_i]));
    end
  endgenerate
  // family reorder_a_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] reorder_a_in_bind_din [0:1];
  wire [63:0] reorder_a_in_bind_dout [0:1];
  wire reorder_a_in_bind_full_n [0:1];
  wire reorder_a_in_bind_write [0:1];
  wire reorder_a_in_bind_empty_n [0:1];
  wire reorder_a_in_bind_read [0:1];
  genvar reorder_a_in_bind_i;
  generate
    for (reorder_a_in_bind_i = 0; reorder_a_in_bind_i < 2; reorder_a_in_bind_i = reorder_a_in_bind_i + 1) begin : g_reorder_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(reorder_a_in_bind_din[reorder_a_in_bind_i]), .full_n(reorder_a_in_bind_full_n[reorder_a_in_bind_i]), .write(reorder_a_in_bind_write[reorder_a_in_bind_i]), .dout(reorder_a_in_bind_dout[reorder_a_in_bind_i]), .empty_n(reorder_a_in_bind_empty_n[reorder_a_in_bind_i]), .read(reorder_a_in_bind_read[reorder_a_in_bind_i]));
    end
  endgenerate
  // family reorder_b_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] reorder_b_in_bind_din [0:1];
  wire [63:0] reorder_b_in_bind_dout [0:1];
  wire reorder_b_in_bind_full_n [0:1];
  wire reorder_b_in_bind_write [0:1];
  wire reorder_b_in_bind_empty_n [0:1];
  wire reorder_b_in_bind_read [0:1];
  genvar reorder_b_in_bind_i;
  generate
    for (reorder_b_in_bind_i = 0; reorder_b_in_bind_i < 2; reorder_b_in_bind_i = reorder_b_in_bind_i + 1) begin : g_reorder_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(reorder_b_in_bind_din[reorder_b_in_bind_i]), .full_n(reorder_b_in_bind_full_n[reorder_b_in_bind_i]), .write(reorder_b_in_bind_write[reorder_b_in_bind_i]), .dout(reorder_b_in_bind_dout[reorder_b_in_bind_i]), .empty_n(reorder_b_in_bind_empty_n[reorder_b_in_bind_i]), .read(reorder_b_in_bind_read[reorder_b_in_bind_i]));
    end
  endgenerate
  // role perm0_r0: 2 instance(s)
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
  // role perm1_r0: 2 instance(s)
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
  // role perm2_r0: 2 instance(s)
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
  // role perm3_r0: 2 instance(s)
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
  // role perm4_r0: 2 instance(s)
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
  // role perm5_r0: 2 instance(s)
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
  // role perm6_r0: 2 instance(s)
  perm6_r0 u_perm6_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm6_u_in_bind_dout[0]),
      .u_in_empty_n(perm6_u_in_bind_empty_n[0]),
      .u_in_read(perm6_u_in_bind_read[0]),
      .u_out_din(bfly6_a_in_bind_din[0]),
      .u_out_full_n(bfly6_a_in_bind_full_n[0]),
      .u_out_write(bfly6_a_in_bind_write[0]),
      .v_in_dout(perm6_v_in_bind_dout[0]),
      .v_in_empty_n(perm6_v_in_bind_empty_n[0]),
      .v_in_read(perm6_v_in_bind_read[0]),
      .v_out_din(bfly6_b_in_bind_din[0]),
      .v_out_full_n(bfly6_b_in_bind_full_n[0]),
      .v_out_write(bfly6_b_in_bind_write[0]));
  perm6_r0 u_perm6_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm6_u_in_bind_dout[1]),
      .u_in_empty_n(perm6_u_in_bind_empty_n[1]),
      .u_in_read(perm6_u_in_bind_read[1]),
      .u_out_din(bfly6_a_in_bind_din[1]),
      .u_out_full_n(bfly6_a_in_bind_full_n[1]),
      .u_out_write(bfly6_a_in_bind_write[1]),
      .v_in_dout(perm6_v_in_bind_dout[1]),
      .v_in_empty_n(perm6_v_in_bind_empty_n[1]),
      .v_in_read(perm6_v_in_bind_read[1]),
      .v_out_din(bfly6_b_in_bind_din[1]),
      .v_out_full_n(bfly6_b_in_bind_full_n[1]),
      .v_out_write(bfly6_b_in_bind_write[1]));
  // role perm7_r0: 2 instance(s)
  perm7_r0 u_perm7_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm7_u_in_bind_dout[0]),
      .u_in_empty_n(perm7_u_in_bind_empty_n[0]),
      .u_in_read(perm7_u_in_bind_read[0]),
      .u_out_din(bfly7_a_in_bind_din[0]),
      .u_out_full_n(bfly7_a_in_bind_full_n[0]),
      .u_out_write(bfly7_a_in_bind_write[0]),
      .v_in_dout(perm7_v_in_bind_dout[0]),
      .v_in_empty_n(perm7_v_in_bind_empty_n[0]),
      .v_in_read(perm7_v_in_bind_read[0]),
      .v_out_din(bfly7_b_in_bind_din[0]),
      .v_out_full_n(bfly7_b_in_bind_full_n[0]),
      .v_out_write(bfly7_b_in_bind_write[0]));
  perm7_r0 u_perm7_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm7_u_in_bind_dout[1]),
      .u_in_empty_n(perm7_u_in_bind_empty_n[1]),
      .u_in_read(perm7_u_in_bind_read[1]),
      .u_out_din(bfly7_a_in_bind_din[1]),
      .u_out_full_n(bfly7_a_in_bind_full_n[1]),
      .u_out_write(bfly7_a_in_bind_write[1]),
      .v_in_dout(perm7_v_in_bind_dout[1]),
      .v_in_empty_n(perm7_v_in_bind_empty_n[1]),
      .v_in_read(perm7_v_in_bind_read[1]),
      .v_out_din(bfly7_b_in_bind_din[1]),
      .v_out_full_n(bfly7_b_in_bind_full_n[1]),
      .v_out_write(bfly7_b_in_bind_write[1]));
  // role perm8_r0: 2 instance(s)
  perm8_r0 u_perm8_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm8_u_in_bind_dout[0]),
      .u_in_empty_n(perm8_u_in_bind_empty_n[0]),
      .u_in_read(perm8_u_in_bind_read[0]),
      .u_out_din(tail_general_a_in_bind_din[0]),
      .u_out_full_n(tail_general_a_in_bind_full_n[0]),
      .u_out_write(tail_general_a_in_bind_write[0]),
      .v_in_dout(perm8_v_in_bind_dout[0]),
      .v_in_empty_n(perm8_v_in_bind_empty_n[0]),
      .v_in_read(perm8_v_in_bind_read[0]),
      .v_out_din(tail_general_b_in_bind_din[0]),
      .v_out_full_n(tail_general_b_in_bind_full_n[0]),
      .v_out_write(tail_general_b_in_bind_write[0]));
  perm8_r0 u_perm8_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .u_in_dout(perm8_u_in_bind_dout[1]),
      .u_in_empty_n(perm8_u_in_bind_empty_n[1]),
      .u_in_read(perm8_u_in_bind_read[1]),
      .u_out_din(tail_general_a_in_bind_din[1]),
      .u_out_full_n(tail_general_a_in_bind_full_n[1]),
      .u_out_write(tail_general_a_in_bind_write[1]),
      .v_in_dout(perm8_v_in_bind_dout[1]),
      .v_in_empty_n(perm8_v_in_bind_empty_n[1]),
      .v_in_read(perm8_v_in_bind_read[1]),
      .v_out_din(tail_general_b_in_bind_din[1]),
      .v_out_full_n(tail_general_b_in_bind_full_n[1]),
      .v_out_write(tail_general_b_in_bind_write[1]));
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] bfly0_pid0_dout [0:1];
  wire bfly0_pid0_empty_n [0:1];
  wire bfly0_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_bfly0_pid0_0 (.dout(bfly0_pid0_dout[0]), .empty_n(bfly0_pid0_empty_n[0]), .read(bfly0_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly0_pid0_1 (.dout(bfly0_pid0_dout[1]), .empty_n(bfly0_pid0_empty_n[1]), .read(bfly0_pid0_read[1]));
  // role bfly0_r0: 2 instance(s)
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
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] bfly1_pid0_dout [0:1];
  wire bfly1_pid0_empty_n [0:1];
  wire bfly1_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_bfly1_pid0_0 (.dout(bfly1_pid0_dout[0]), .empty_n(bfly1_pid0_empty_n[0]), .read(bfly1_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly1_pid0_1 (.dout(bfly1_pid0_dout[1]), .empty_n(bfly1_pid0_empty_n[1]), .read(bfly1_pid0_read[1]));
  // role bfly1_r0: 2 instance(s)
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
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] bfly2_pid0_dout [0:1];
  wire bfly2_pid0_empty_n [0:1];
  wire bfly2_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_bfly2_pid0_0 (.dout(bfly2_pid0_dout[0]), .empty_n(bfly2_pid0_empty_n[0]), .read(bfly2_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly2_pid0_1 (.dout(bfly2_pid0_dout[1]), .empty_n(bfly2_pid0_empty_n[1]), .read(bfly2_pid0_read[1]));
  // role bfly2_r0: 2 instance(s)
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
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] bfly3_pid0_dout [0:1];
  wire bfly3_pid0_empty_n [0:1];
  wire bfly3_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_bfly3_pid0_0 (.dout(bfly3_pid0_dout[0]), .empty_n(bfly3_pid0_empty_n[0]), .read(bfly3_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly3_pid0_1 (.dout(bfly3_pid0_dout[1]), .empty_n(bfly3_pid0_empty_n[1]), .read(bfly3_pid0_read[1]));
  // role bfly3_r0: 2 instance(s)
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
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] bfly4_pid0_dout [0:1];
  wire bfly4_pid0_empty_n [0:1];
  wire bfly4_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_bfly4_pid0_0 (.dout(bfly4_pid0_dout[0]), .empty_n(bfly4_pid0_empty_n[0]), .read(bfly4_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly4_pid0_1 (.dout(bfly4_pid0_dout[1]), .empty_n(bfly4_pid0_empty_n[1]), .read(bfly4_pid0_read[1]));
  // role bfly4_r0: 2 instance(s)
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
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] bfly5_pid0_dout [0:1];
  wire bfly5_pid0_empty_n [0:1];
  wire bfly5_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_bfly5_pid0_0 (.dout(bfly5_pid0_dout[0]), .empty_n(bfly5_pid0_empty_n[0]), .read(bfly5_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly5_pid0_1 (.dout(bfly5_pid0_dout[1]), .empty_n(bfly5_pid0_empty_n[1]), .read(bfly5_pid0_read[1]));
  // role bfly5_r0: 2 instance(s)
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
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] bfly6_pid0_dout [0:1];
  wire bfly6_pid0_empty_n [0:1];
  wire bfly6_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_bfly6_pid0_0 (.dout(bfly6_pid0_dout[0]), .empty_n(bfly6_pid0_empty_n[0]), .read(bfly6_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly6_pid0_1 (.dout(bfly6_pid0_dout[1]), .empty_n(bfly6_pid0_empty_n[1]), .read(bfly6_pid0_read[1]));
  // role bfly6_r0: 2 instance(s)
  bfly6_r0 u_bfly6_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly6_a_in_bind_dout[0]),
      .a_in_empty_n(bfly6_a_in_bind_empty_n[0]),
      .a_in_read(bfly6_a_in_bind_read[0]),
      .a_out_din(perm7_u_in_bind_din[0]),
      .a_out_full_n(perm7_u_in_bind_full_n[0]),
      .a_out_write(perm7_u_in_bind_write[0]),
      .b_in_dout(bfly6_b_in_bind_dout[0]),
      .b_in_empty_n(bfly6_b_in_bind_empty_n[0]),
      .b_in_read(bfly6_b_in_bind_read[0]),
      .b_out_din(perm7_v_in_bind_din[0]),
      .b_out_full_n(perm7_v_in_bind_full_n[0]),
      .b_out_write(perm7_v_in_bind_write[0]),
      ._pid0_dout(bfly6_pid0_dout[0]),
      ._pid0_empty_n(bfly6_pid0_empty_n[0]),
      ._pid0_read(bfly6_pid0_read[0]));
  bfly6_r0 u_bfly6_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly6_a_in_bind_dout[1]),
      .a_in_empty_n(bfly6_a_in_bind_empty_n[1]),
      .a_in_read(bfly6_a_in_bind_read[1]),
      .a_out_din(perm7_u_in_bind_din[1]),
      .a_out_full_n(perm7_u_in_bind_full_n[1]),
      .a_out_write(perm7_u_in_bind_write[1]),
      .b_in_dout(bfly6_b_in_bind_dout[1]),
      .b_in_empty_n(bfly6_b_in_bind_empty_n[1]),
      .b_in_read(bfly6_b_in_bind_read[1]),
      .b_out_din(perm7_v_in_bind_din[1]),
      .b_out_full_n(perm7_v_in_bind_full_n[1]),
      .b_out_write(perm7_v_in_bind_write[1]),
      ._pid0_dout(bfly6_pid0_dout[1]),
      ._pid0_empty_n(bfly6_pid0_empty_n[1]),
      ._pid0_read(bfly6_pid0_read[1]));
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] bfly7_pid0_dout [0:1];
  wire bfly7_pid0_empty_n [0:1];
  wire bfly7_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_bfly7_pid0_0 (.dout(bfly7_pid0_dout[0]), .empty_n(bfly7_pid0_empty_n[0]), .read(bfly7_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly7_pid0_1 (.dout(bfly7_pid0_dout[1]), .empty_n(bfly7_pid0_empty_n[1]), .read(bfly7_pid0_read[1]));
  // role bfly7_r0: 2 instance(s)
  bfly7_r0 u_bfly7_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly7_a_in_bind_dout[0]),
      .a_in_empty_n(bfly7_a_in_bind_empty_n[0]),
      .a_in_read(bfly7_a_in_bind_read[0]),
      .a_out_din(perm8_u_in_bind_din[0]),
      .a_out_full_n(perm8_u_in_bind_full_n[0]),
      .a_out_write(perm8_u_in_bind_write[0]),
      .b_in_dout(bfly7_b_in_bind_dout[0]),
      .b_in_empty_n(bfly7_b_in_bind_empty_n[0]),
      .b_in_read(bfly7_b_in_bind_read[0]),
      .b_out_din(perm8_v_in_bind_din[0]),
      .b_out_full_n(perm8_v_in_bind_full_n[0]),
      .b_out_write(perm8_v_in_bind_write[0]),
      ._pid0_dout(bfly7_pid0_dout[0]),
      ._pid0_empty_n(bfly7_pid0_empty_n[0]),
      ._pid0_read(bfly7_pid0_read[0]));
  bfly7_r0 u_bfly7_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly7_a_in_bind_dout[1]),
      .a_in_empty_n(bfly7_a_in_bind_empty_n[1]),
      .a_in_read(bfly7_a_in_bind_read[1]),
      .a_out_din(perm8_u_in_bind_din[1]),
      .a_out_full_n(perm8_u_in_bind_full_n[1]),
      .a_out_write(perm8_u_in_bind_write[1]),
      .b_in_dout(bfly7_b_in_bind_dout[1]),
      .b_in_empty_n(bfly7_b_in_bind_empty_n[1]),
      .b_in_read(bfly7_b_in_bind_read[1]),
      .b_out_din(perm8_v_in_bind_din[1]),
      .b_out_full_n(perm8_v_in_bind_full_n[1]),
      .b_out_write(perm8_v_in_bind_write[1]),
      ._pid0_dout(bfly7_pid0_dout[1]),
      ._pid0_empty_n(bfly7_pid0_empty_n[1]),
      ._pid0_read(bfly7_pid0_read[1]));
  // role tail_general_r0: 1 instance(s)
  tail_general_r0 u_tail_general_r0_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_in_bind_dout[0]),
      .a_in_empty_n(tail_general_a_in_bind_empty_n[0]),
      .a_in_read(tail_general_a_in_bind_read[0]),
      .a_out_din(tail_general_a_out_a_in_din[2]),
      .a_out_full_n(tail_general_a_out_a_in_full_n[2]),
      .a_out_write(tail_general_a_out_a_in_write[2]),
      .b_in_dout(tail_general_b_in_bind_dout[0]),
      .b_in_empty_n(tail_general_b_in_bind_empty_n[0]),
      .b_in_read(tail_general_b_in_bind_read[0]),
      .b_out_din(tail_general_b_out_a_in_din[3]),
      .b_out_full_n(tail_general_b_out_a_in_full_n[3]),
      .b_out_write(tail_general_b_out_a_in_write[3]));
  // role tail_general_r1: 1 instance(s)
  tail_general_r1 u_tail_general_r1_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_in_bind_dout[1]),
      .a_in_empty_n(tail_general_a_in_bind_empty_n[1]),
      .a_in_read(tail_general_a_in_bind_read[1]),
      .a_out_din(tail_general_a_out_b_in_din[2]),
      .a_out_full_n(tail_general_a_out_b_in_full_n[2]),
      .a_out_write(tail_general_a_out_b_in_write[2]),
      .b_in_dout(tail_general_b_in_bind_dout[1]),
      .b_in_empty_n(tail_general_b_in_bind_empty_n[1]),
      .b_in_read(tail_general_b_in_bind_read[1]),
      .b_out_din(tail_general_b_out_b_in_din[3]),
      .b_out_full_n(tail_general_b_out_b_in_full_n[3]),
      .b_out_write(tail_general_b_out_b_in_write[3]));
  // role tail_general_r2: 1 instance(s)
  tail_general_r2 u_tail_general_r2_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_a_out_a_in_dout[2]),
      .a_in_empty_n(tail_general_a_out_a_in_empty_n[2]),
      .a_in_read(tail_general_a_out_a_in_read[2]),
      .a_out_din(reorder_a_in_bind_din[0]),
      .a_out_full_n(reorder_a_in_bind_full_n[0]),
      .a_out_write(reorder_a_in_bind_write[0]),
      .b_in_dout(tail_general_a_out_b_in_dout[2]),
      .b_in_empty_n(tail_general_a_out_b_in_empty_n[2]),
      .b_in_read(tail_general_a_out_b_in_read[2]),
      .b_out_din(reorder_b_in_bind_din[0]),
      .b_out_full_n(reorder_b_in_bind_full_n[0]),
      .b_out_write(reorder_b_in_bind_write[0]));
  // role tail_general_r3: 1 instance(s)
  tail_general_r3 u_tail_general_r3_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(tail_general_b_out_a_in_dout[3]),
      .a_in_empty_n(tail_general_b_out_a_in_empty_n[3]),
      .a_in_read(tail_general_b_out_a_in_read[3]),
      .a_out_din(reorder_a_in_bind_din[1]),
      .a_out_full_n(reorder_a_in_bind_full_n[1]),
      .a_out_write(reorder_a_in_bind_write[1]),
      .b_in_dout(tail_general_b_out_b_in_dout[3]),
      .b_in_empty_n(tail_general_b_out_b_in_empty_n[3]),
      .b_in_read(tail_general_b_out_b_in_read[3]),
      .b_out_din(reorder_b_in_bind_din[1]),
      .b_out_full_n(reorder_b_in_bind_full_n[1]),
      .b_out_write(reorder_b_in_bind_write[1]));
  // role reorder_r0: 2 instance(s)
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
endmodule
