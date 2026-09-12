`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] bfly0_a_in_bind_dout [0:0],
  input  wire bfly0_a_in_bind_empty_n [0:0],
  output wire bfly0_a_in_bind_read [0:0],
  input  wire [63:0] bfly0_b_in_bind_dout [0:0],
  input  wire bfly0_b_in_bind_empty_n [0:0],
  output wire bfly0_b_in_bind_read [0:0],
  output wire [63:0] reorder_split_a_out_bind_din [0:0],
  output wire reorder_split_a_out_bind_write [0:0],
  input  wire reorder_split_a_out_bind_full_n [0:0],
  output wire [63:0] reorder_split_b_out_bind_din [0:0],
  output wire reorder_split_b_out_bind_write [0:0],
  input  wire reorder_split_b_out_bind_full_n [0:0]
);
  // family bfly1_a_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly1_a_in_bind_din [0:0];
  wire [63:0] bfly1_a_in_bind_dout [0:0];
  wire bfly1_a_in_bind_full_n [0:0];
  wire bfly1_a_in_bind_write [0:0];
  wire bfly1_a_in_bind_empty_n [0:0];
  wire bfly1_a_in_bind_read [0:0];
  genvar bfly1_a_in_bind_i;
  generate
    for (bfly1_a_in_bind_i = 0; bfly1_a_in_bind_i < 1; bfly1_a_in_bind_i = bfly1_a_in_bind_i + 1) begin : g_bfly1_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly1_a_in_bind_din[bfly1_a_in_bind_i]), .full_n(bfly1_a_in_bind_full_n[bfly1_a_in_bind_i]), .write(bfly1_a_in_bind_write[bfly1_a_in_bind_i]), .dout(bfly1_a_in_bind_dout[bfly1_a_in_bind_i]), .empty_n(bfly1_a_in_bind_empty_n[bfly1_a_in_bind_i]), .read(bfly1_a_in_bind_read[bfly1_a_in_bind_i]));
    end
  endgenerate
  // family bfly1_b_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly1_b_in_bind_din [0:0];
  wire [63:0] bfly1_b_in_bind_dout [0:0];
  wire bfly1_b_in_bind_full_n [0:0];
  wire bfly1_b_in_bind_write [0:0];
  wire bfly1_b_in_bind_empty_n [0:0];
  wire bfly1_b_in_bind_read [0:0];
  genvar bfly1_b_in_bind_i;
  generate
    for (bfly1_b_in_bind_i = 0; bfly1_b_in_bind_i < 1; bfly1_b_in_bind_i = bfly1_b_in_bind_i + 1) begin : g_bfly1_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly1_b_in_bind_din[bfly1_b_in_bind_i]), .full_n(bfly1_b_in_bind_full_n[bfly1_b_in_bind_i]), .write(bfly1_b_in_bind_write[bfly1_b_in_bind_i]), .dout(bfly1_b_in_bind_dout[bfly1_b_in_bind_i]), .empty_n(bfly1_b_in_bind_empty_n[bfly1_b_in_bind_i]), .read(bfly1_b_in_bind_read[bfly1_b_in_bind_i]));
    end
  endgenerate
  // family bfly2_a_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly2_a_in_bind_din [0:0];
  wire [63:0] bfly2_a_in_bind_dout [0:0];
  wire bfly2_a_in_bind_full_n [0:0];
  wire bfly2_a_in_bind_write [0:0];
  wire bfly2_a_in_bind_empty_n [0:0];
  wire bfly2_a_in_bind_read [0:0];
  genvar bfly2_a_in_bind_i;
  generate
    for (bfly2_a_in_bind_i = 0; bfly2_a_in_bind_i < 1; bfly2_a_in_bind_i = bfly2_a_in_bind_i + 1) begin : g_bfly2_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly2_a_in_bind_din[bfly2_a_in_bind_i]), .full_n(bfly2_a_in_bind_full_n[bfly2_a_in_bind_i]), .write(bfly2_a_in_bind_write[bfly2_a_in_bind_i]), .dout(bfly2_a_in_bind_dout[bfly2_a_in_bind_i]), .empty_n(bfly2_a_in_bind_empty_n[bfly2_a_in_bind_i]), .read(bfly2_a_in_bind_read[bfly2_a_in_bind_i]));
    end
  endgenerate
  // family bfly2_b_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly2_b_in_bind_din [0:0];
  wire [63:0] bfly2_b_in_bind_dout [0:0];
  wire bfly2_b_in_bind_full_n [0:0];
  wire bfly2_b_in_bind_write [0:0];
  wire bfly2_b_in_bind_empty_n [0:0];
  wire bfly2_b_in_bind_read [0:0];
  genvar bfly2_b_in_bind_i;
  generate
    for (bfly2_b_in_bind_i = 0; bfly2_b_in_bind_i < 1; bfly2_b_in_bind_i = bfly2_b_in_bind_i + 1) begin : g_bfly2_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly2_b_in_bind_din[bfly2_b_in_bind_i]), .full_n(bfly2_b_in_bind_full_n[bfly2_b_in_bind_i]), .write(bfly2_b_in_bind_write[bfly2_b_in_bind_i]), .dout(bfly2_b_in_bind_dout[bfly2_b_in_bind_i]), .empty_n(bfly2_b_in_bind_empty_n[bfly2_b_in_bind_i]), .read(bfly2_b_in_bind_read[bfly2_b_in_bind_i]));
    end
  endgenerate
  // family bfly3_a_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly3_a_in_bind_din [0:0];
  wire [63:0] bfly3_a_in_bind_dout [0:0];
  wire bfly3_a_in_bind_full_n [0:0];
  wire bfly3_a_in_bind_write [0:0];
  wire bfly3_a_in_bind_empty_n [0:0];
  wire bfly3_a_in_bind_read [0:0];
  genvar bfly3_a_in_bind_i;
  generate
    for (bfly3_a_in_bind_i = 0; bfly3_a_in_bind_i < 1; bfly3_a_in_bind_i = bfly3_a_in_bind_i + 1) begin : g_bfly3_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly3_a_in_bind_din[bfly3_a_in_bind_i]), .full_n(bfly3_a_in_bind_full_n[bfly3_a_in_bind_i]), .write(bfly3_a_in_bind_write[bfly3_a_in_bind_i]), .dout(bfly3_a_in_bind_dout[bfly3_a_in_bind_i]), .empty_n(bfly3_a_in_bind_empty_n[bfly3_a_in_bind_i]), .read(bfly3_a_in_bind_read[bfly3_a_in_bind_i]));
    end
  endgenerate
  // family bfly3_b_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly3_b_in_bind_din [0:0];
  wire [63:0] bfly3_b_in_bind_dout [0:0];
  wire bfly3_b_in_bind_full_n [0:0];
  wire bfly3_b_in_bind_write [0:0];
  wire bfly3_b_in_bind_empty_n [0:0];
  wire bfly3_b_in_bind_read [0:0];
  genvar bfly3_b_in_bind_i;
  generate
    for (bfly3_b_in_bind_i = 0; bfly3_b_in_bind_i < 1; bfly3_b_in_bind_i = bfly3_b_in_bind_i + 1) begin : g_bfly3_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly3_b_in_bind_din[bfly3_b_in_bind_i]), .full_n(bfly3_b_in_bind_full_n[bfly3_b_in_bind_i]), .write(bfly3_b_in_bind_write[bfly3_b_in_bind_i]), .dout(bfly3_b_in_bind_dout[bfly3_b_in_bind_i]), .empty_n(bfly3_b_in_bind_empty_n[bfly3_b_in_bind_i]), .read(bfly3_b_in_bind_read[bfly3_b_in_bind_i]));
    end
  endgenerate
  // family bfly4_a_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly4_a_in_bind_din [0:0];
  wire [63:0] bfly4_a_in_bind_dout [0:0];
  wire bfly4_a_in_bind_full_n [0:0];
  wire bfly4_a_in_bind_write [0:0];
  wire bfly4_a_in_bind_empty_n [0:0];
  wire bfly4_a_in_bind_read [0:0];
  genvar bfly4_a_in_bind_i;
  generate
    for (bfly4_a_in_bind_i = 0; bfly4_a_in_bind_i < 1; bfly4_a_in_bind_i = bfly4_a_in_bind_i + 1) begin : g_bfly4_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly4_a_in_bind_din[bfly4_a_in_bind_i]), .full_n(bfly4_a_in_bind_full_n[bfly4_a_in_bind_i]), .write(bfly4_a_in_bind_write[bfly4_a_in_bind_i]), .dout(bfly4_a_in_bind_dout[bfly4_a_in_bind_i]), .empty_n(bfly4_a_in_bind_empty_n[bfly4_a_in_bind_i]), .read(bfly4_a_in_bind_read[bfly4_a_in_bind_i]));
    end
  endgenerate
  // family bfly4_b_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly4_b_in_bind_din [0:0];
  wire [63:0] bfly4_b_in_bind_dout [0:0];
  wire bfly4_b_in_bind_full_n [0:0];
  wire bfly4_b_in_bind_write [0:0];
  wire bfly4_b_in_bind_empty_n [0:0];
  wire bfly4_b_in_bind_read [0:0];
  genvar bfly4_b_in_bind_i;
  generate
    for (bfly4_b_in_bind_i = 0; bfly4_b_in_bind_i < 1; bfly4_b_in_bind_i = bfly4_b_in_bind_i + 1) begin : g_bfly4_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly4_b_in_bind_din[bfly4_b_in_bind_i]), .full_n(bfly4_b_in_bind_full_n[bfly4_b_in_bind_i]), .write(bfly4_b_in_bind_write[bfly4_b_in_bind_i]), .dout(bfly4_b_in_bind_dout[bfly4_b_in_bind_i]), .empty_n(bfly4_b_in_bind_empty_n[bfly4_b_in_bind_i]), .read(bfly4_b_in_bind_read[bfly4_b_in_bind_i]));
    end
  endgenerate
  // family bfly5_a_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly5_a_in_bind_din [0:0];
  wire [63:0] bfly5_a_in_bind_dout [0:0];
  wire bfly5_a_in_bind_full_n [0:0];
  wire bfly5_a_in_bind_write [0:0];
  wire bfly5_a_in_bind_empty_n [0:0];
  wire bfly5_a_in_bind_read [0:0];
  genvar bfly5_a_in_bind_i;
  generate
    for (bfly5_a_in_bind_i = 0; bfly5_a_in_bind_i < 1; bfly5_a_in_bind_i = bfly5_a_in_bind_i + 1) begin : g_bfly5_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly5_a_in_bind_din[bfly5_a_in_bind_i]), .full_n(bfly5_a_in_bind_full_n[bfly5_a_in_bind_i]), .write(bfly5_a_in_bind_write[bfly5_a_in_bind_i]), .dout(bfly5_a_in_bind_dout[bfly5_a_in_bind_i]), .empty_n(bfly5_a_in_bind_empty_n[bfly5_a_in_bind_i]), .read(bfly5_a_in_bind_read[bfly5_a_in_bind_i]));
    end
  endgenerate
  // family bfly5_b_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly5_b_in_bind_din [0:0];
  wire [63:0] bfly5_b_in_bind_dout [0:0];
  wire bfly5_b_in_bind_full_n [0:0];
  wire bfly5_b_in_bind_write [0:0];
  wire bfly5_b_in_bind_empty_n [0:0];
  wire bfly5_b_in_bind_read [0:0];
  genvar bfly5_b_in_bind_i;
  generate
    for (bfly5_b_in_bind_i = 0; bfly5_b_in_bind_i < 1; bfly5_b_in_bind_i = bfly5_b_in_bind_i + 1) begin : g_bfly5_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly5_b_in_bind_din[bfly5_b_in_bind_i]), .full_n(bfly5_b_in_bind_full_n[bfly5_b_in_bind_i]), .write(bfly5_b_in_bind_write[bfly5_b_in_bind_i]), .dout(bfly5_b_in_bind_dout[bfly5_b_in_bind_i]), .empty_n(bfly5_b_in_bind_empty_n[bfly5_b_in_bind_i]), .read(bfly5_b_in_bind_read[bfly5_b_in_bind_i]));
    end
  endgenerate
  // family bfly6_a_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly6_a_in_bind_din [0:0];
  wire [63:0] bfly6_a_in_bind_dout [0:0];
  wire bfly6_a_in_bind_full_n [0:0];
  wire bfly6_a_in_bind_write [0:0];
  wire bfly6_a_in_bind_empty_n [0:0];
  wire bfly6_a_in_bind_read [0:0];
  genvar bfly6_a_in_bind_i;
  generate
    for (bfly6_a_in_bind_i = 0; bfly6_a_in_bind_i < 1; bfly6_a_in_bind_i = bfly6_a_in_bind_i + 1) begin : g_bfly6_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly6_a_in_bind_din[bfly6_a_in_bind_i]), .full_n(bfly6_a_in_bind_full_n[bfly6_a_in_bind_i]), .write(bfly6_a_in_bind_write[bfly6_a_in_bind_i]), .dout(bfly6_a_in_bind_dout[bfly6_a_in_bind_i]), .empty_n(bfly6_a_in_bind_empty_n[bfly6_a_in_bind_i]), .read(bfly6_a_in_bind_read[bfly6_a_in_bind_i]));
    end
  endgenerate
  // family bfly6_b_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly6_b_in_bind_din [0:0];
  wire [63:0] bfly6_b_in_bind_dout [0:0];
  wire bfly6_b_in_bind_full_n [0:0];
  wire bfly6_b_in_bind_write [0:0];
  wire bfly6_b_in_bind_empty_n [0:0];
  wire bfly6_b_in_bind_read [0:0];
  genvar bfly6_b_in_bind_i;
  generate
    for (bfly6_b_in_bind_i = 0; bfly6_b_in_bind_i < 1; bfly6_b_in_bind_i = bfly6_b_in_bind_i + 1) begin : g_bfly6_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly6_b_in_bind_din[bfly6_b_in_bind_i]), .full_n(bfly6_b_in_bind_full_n[bfly6_b_in_bind_i]), .write(bfly6_b_in_bind_write[bfly6_b_in_bind_i]), .dout(bfly6_b_in_bind_dout[bfly6_b_in_bind_i]), .empty_n(bfly6_b_in_bind_empty_n[bfly6_b_in_bind_i]), .read(bfly6_b_in_bind_read[bfly6_b_in_bind_i]));
    end
  endgenerate
  // family bfly7_a_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly7_a_in_bind_din [0:0];
  wire [63:0] bfly7_a_in_bind_dout [0:0];
  wire bfly7_a_in_bind_full_n [0:0];
  wire bfly7_a_in_bind_write [0:0];
  wire bfly7_a_in_bind_empty_n [0:0];
  wire bfly7_a_in_bind_read [0:0];
  genvar bfly7_a_in_bind_i;
  generate
    for (bfly7_a_in_bind_i = 0; bfly7_a_in_bind_i < 1; bfly7_a_in_bind_i = bfly7_a_in_bind_i + 1) begin : g_bfly7_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly7_a_in_bind_din[bfly7_a_in_bind_i]), .full_n(bfly7_a_in_bind_full_n[bfly7_a_in_bind_i]), .write(bfly7_a_in_bind_write[bfly7_a_in_bind_i]), .dout(bfly7_a_in_bind_dout[bfly7_a_in_bind_i]), .empty_n(bfly7_a_in_bind_empty_n[bfly7_a_in_bind_i]), .read(bfly7_a_in_bind_read[bfly7_a_in_bind_i]));
    end
  endgenerate
  // family bfly7_b_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] bfly7_b_in_bind_din [0:0];
  wire [63:0] bfly7_b_in_bind_dout [0:0];
  wire bfly7_b_in_bind_full_n [0:0];
  wire bfly7_b_in_bind_write [0:0];
  wire bfly7_b_in_bind_empty_n [0:0];
  wire bfly7_b_in_bind_read [0:0];
  genvar bfly7_b_in_bind_i;
  generate
    for (bfly7_b_in_bind_i = 0; bfly7_b_in_bind_i < 1; bfly7_b_in_bind_i = bfly7_b_in_bind_i + 1) begin : g_bfly7_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly7_b_in_bind_din[bfly7_b_in_bind_i]), .full_n(bfly7_b_in_bind_full_n[bfly7_b_in_bind_i]), .write(bfly7_b_in_bind_write[bfly7_b_in_bind_i]), .dout(bfly7_b_in_bind_dout[bfly7_b_in_bind_i]), .empty_n(bfly7_b_in_bind_empty_n[bfly7_b_in_bind_i]), .read(bfly7_b_in_bind_read[bfly7_b_in_bind_i]));
    end
  endgenerate
  // family reorder_split_a_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] reorder_split_a_in_bind_din [0:0];
  wire [63:0] reorder_split_a_in_bind_dout [0:0];
  wire reorder_split_a_in_bind_full_n [0:0];
  wire reorder_split_a_in_bind_write [0:0];
  wire reorder_split_a_in_bind_empty_n [0:0];
  wire reorder_split_a_in_bind_read [0:0];
  genvar reorder_split_a_in_bind_i;
  generate
    for (reorder_split_a_in_bind_i = 0; reorder_split_a_in_bind_i < 1; reorder_split_a_in_bind_i = reorder_split_a_in_bind_i + 1) begin : g_reorder_split_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(reorder_split_a_in_bind_din[reorder_split_a_in_bind_i]), .full_n(reorder_split_a_in_bind_full_n[reorder_split_a_in_bind_i]), .write(reorder_split_a_in_bind_write[reorder_split_a_in_bind_i]), .dout(reorder_split_a_in_bind_dout[reorder_split_a_in_bind_i]), .empty_n(reorder_split_a_in_bind_empty_n[reorder_split_a_in_bind_i]), .read(reorder_split_a_in_bind_read[reorder_split_a_in_bind_i]));
    end
  endgenerate
  // family reorder_split_b_in_bind: 1 channel(s), 64-bit, depth 8
  wire [63:0] reorder_split_b_in_bind_din [0:0];
  wire [63:0] reorder_split_b_in_bind_dout [0:0];
  wire reorder_split_b_in_bind_full_n [0:0];
  wire reorder_split_b_in_bind_write [0:0];
  wire reorder_split_b_in_bind_empty_n [0:0];
  wire reorder_split_b_in_bind_read [0:0];
  genvar reorder_split_b_in_bind_i;
  generate
    for (reorder_split_b_in_bind_i = 0; reorder_split_b_in_bind_i < 1; reorder_split_b_in_bind_i = reorder_split_b_in_bind_i + 1) begin : g_reorder_split_b_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(reorder_split_b_in_bind_din[reorder_split_b_in_bind_i]), .full_n(reorder_split_b_in_bind_full_n[reorder_split_b_in_bind_i]), .write(reorder_split_b_in_bind_write[reorder_split_b_in_bind_i]), .dout(reorder_split_b_in_bind_dout[reorder_split_b_in_bind_i]), .empty_n(reorder_split_b_in_bind_empty_n[reorder_split_b_in_bind_i]), .read(reorder_split_b_in_bind_read[reorder_split_b_in_bind_i]));
    end
  endgenerate
  // role bfly0_r0: 1 instance(s)
  bfly0_r0 u_bfly0_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly0_a_in_bind_dout[0]),
      .a_in_empty_n(bfly0_a_in_bind_empty_n[0]),
      .a_in_read(bfly0_a_in_bind_read[0]),
      .a_out_din(bfly1_a_in_bind_din[0]),
      .a_out_full_n(bfly1_a_in_bind_full_n[0]),
      .a_out_write(bfly1_a_in_bind_write[0]),
      .b_in_dout(bfly0_b_in_bind_dout[0]),
      .b_in_empty_n(bfly0_b_in_bind_empty_n[0]),
      .b_in_read(bfly0_b_in_bind_read[0]),
      .b_out_din(bfly1_b_in_bind_din[0]),
      .b_out_full_n(bfly1_b_in_bind_full_n[0]),
      .b_out_write(bfly1_b_in_bind_write[0]));
  // role bfly1_r0: 1 instance(s)
  bfly1_r0 u_bfly1_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly1_a_in_bind_dout[0]),
      .a_in_empty_n(bfly1_a_in_bind_empty_n[0]),
      .a_in_read(bfly1_a_in_bind_read[0]),
      .a_out_din(bfly2_a_in_bind_din[0]),
      .a_out_full_n(bfly2_a_in_bind_full_n[0]),
      .a_out_write(bfly2_a_in_bind_write[0]),
      .b_in_dout(bfly1_b_in_bind_dout[0]),
      .b_in_empty_n(bfly1_b_in_bind_empty_n[0]),
      .b_in_read(bfly1_b_in_bind_read[0]),
      .b_out_din(bfly2_b_in_bind_din[0]),
      .b_out_full_n(bfly2_b_in_bind_full_n[0]),
      .b_out_write(bfly2_b_in_bind_write[0]));
  // role bfly2_r0: 1 instance(s)
  bfly2_r0 u_bfly2_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly2_a_in_bind_dout[0]),
      .a_in_empty_n(bfly2_a_in_bind_empty_n[0]),
      .a_in_read(bfly2_a_in_bind_read[0]),
      .a_out_din(bfly3_a_in_bind_din[0]),
      .a_out_full_n(bfly3_a_in_bind_full_n[0]),
      .a_out_write(bfly3_a_in_bind_write[0]),
      .b_in_dout(bfly2_b_in_bind_dout[0]),
      .b_in_empty_n(bfly2_b_in_bind_empty_n[0]),
      .b_in_read(bfly2_b_in_bind_read[0]),
      .b_out_din(bfly3_b_in_bind_din[0]),
      .b_out_full_n(bfly3_b_in_bind_full_n[0]),
      .b_out_write(bfly3_b_in_bind_write[0]));
  // role bfly3_r0: 1 instance(s)
  bfly3_r0 u_bfly3_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly3_a_in_bind_dout[0]),
      .a_in_empty_n(bfly3_a_in_bind_empty_n[0]),
      .a_in_read(bfly3_a_in_bind_read[0]),
      .a_out_din(bfly4_a_in_bind_din[0]),
      .a_out_full_n(bfly4_a_in_bind_full_n[0]),
      .a_out_write(bfly4_a_in_bind_write[0]),
      .b_in_dout(bfly3_b_in_bind_dout[0]),
      .b_in_empty_n(bfly3_b_in_bind_empty_n[0]),
      .b_in_read(bfly3_b_in_bind_read[0]),
      .b_out_din(bfly4_b_in_bind_din[0]),
      .b_out_full_n(bfly4_b_in_bind_full_n[0]),
      .b_out_write(bfly4_b_in_bind_write[0]));
  // role bfly4_r0: 1 instance(s)
  bfly4_r0 u_bfly4_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly4_a_in_bind_dout[0]),
      .a_in_empty_n(bfly4_a_in_bind_empty_n[0]),
      .a_in_read(bfly4_a_in_bind_read[0]),
      .a_out_din(bfly5_a_in_bind_din[0]),
      .a_out_full_n(bfly5_a_in_bind_full_n[0]),
      .a_out_write(bfly5_a_in_bind_write[0]),
      .b_in_dout(bfly4_b_in_bind_dout[0]),
      .b_in_empty_n(bfly4_b_in_bind_empty_n[0]),
      .b_in_read(bfly4_b_in_bind_read[0]),
      .b_out_din(bfly5_b_in_bind_din[0]),
      .b_out_full_n(bfly5_b_in_bind_full_n[0]),
      .b_out_write(bfly5_b_in_bind_write[0]));
  // role bfly5_r0: 1 instance(s)
  bfly5_r0 u_bfly5_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly5_a_in_bind_dout[0]),
      .a_in_empty_n(bfly5_a_in_bind_empty_n[0]),
      .a_in_read(bfly5_a_in_bind_read[0]),
      .a_out_din(bfly6_a_in_bind_din[0]),
      .a_out_full_n(bfly6_a_in_bind_full_n[0]),
      .a_out_write(bfly6_a_in_bind_write[0]),
      .b_in_dout(bfly5_b_in_bind_dout[0]),
      .b_in_empty_n(bfly5_b_in_bind_empty_n[0]),
      .b_in_read(bfly5_b_in_bind_read[0]),
      .b_out_din(bfly6_b_in_bind_din[0]),
      .b_out_full_n(bfly6_b_in_bind_full_n[0]),
      .b_out_write(bfly6_b_in_bind_write[0]));
  // role bfly6_r0: 1 instance(s)
  bfly6_r0 u_bfly6_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly6_a_in_bind_dout[0]),
      .a_in_empty_n(bfly6_a_in_bind_empty_n[0]),
      .a_in_read(bfly6_a_in_bind_read[0]),
      .a_out_din(bfly7_a_in_bind_din[0]),
      .a_out_full_n(bfly7_a_in_bind_full_n[0]),
      .a_out_write(bfly7_a_in_bind_write[0]),
      .b_in_dout(bfly6_b_in_bind_dout[0]),
      .b_in_empty_n(bfly6_b_in_bind_empty_n[0]),
      .b_in_read(bfly6_b_in_bind_read[0]),
      .b_out_din(bfly7_b_in_bind_din[0]),
      .b_out_full_n(bfly7_b_in_bind_full_n[0]),
      .b_out_write(bfly7_b_in_bind_write[0]));
  // role bfly7_r0: 1 instance(s)
  bfly7_r0 u_bfly7_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(bfly7_a_in_bind_dout[0]),
      .a_in_empty_n(bfly7_a_in_bind_empty_n[0]),
      .a_in_read(bfly7_a_in_bind_read[0]),
      .a_out_din(reorder_split_a_in_bind_din[0]),
      .a_out_full_n(reorder_split_a_in_bind_full_n[0]),
      .a_out_write(reorder_split_a_in_bind_write[0]),
      .b_in_dout(bfly7_b_in_bind_dout[0]),
      .b_in_empty_n(bfly7_b_in_bind_empty_n[0]),
      .b_in_read(bfly7_b_in_bind_read[0]),
      .b_out_din(reorder_split_b_in_bind_din[0]),
      .b_out_full_n(reorder_split_b_in_bind_full_n[0]),
      .b_out_write(reorder_split_b_in_bind_write[0]));
  // role reorder_split_r0: 1 instance(s)
  reorder_split_r0 u_reorder_split_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(reorder_split_a_in_bind_dout[0]),
      .a_in_empty_n(reorder_split_a_in_bind_empty_n[0]),
      .a_in_read(reorder_split_a_in_bind_read[0]),
      .a_out_din(reorder_split_a_out_bind_din[0]),
      .a_out_full_n(reorder_split_a_out_bind_full_n[0]),
      .a_out_write(reorder_split_a_out_bind_write[0]),
      .b_in_dout(reorder_split_b_in_bind_dout[0]),
      .b_in_empty_n(reorder_split_b_in_bind_empty_n[0]),
      .b_in_read(reorder_split_b_in_bind_read[0]),
      .b_out_din(reorder_split_b_out_bind_din[0]),
      .b_out_full_n(reorder_split_b_out_bind_full_n[0]),
      .b_out_write(reorder_split_b_out_bind_write[0]));
endmodule
