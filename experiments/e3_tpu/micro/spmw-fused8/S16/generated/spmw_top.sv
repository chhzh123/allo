`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [7:0] blk8_a0_in_bind_dout [0:1],
  input  wire blk8_a0_in_bind_empty_n [0:1],
  output wire blk8_a0_in_bind_read [0:1],
  input  wire [7:0] blk8_w0_in_bind_dout [0:1],
  input  wire blk8_w0_in_bind_empty_n [0:1],
  output wire blk8_w0_in_bind_read [0:1],
  input  wire [31:0] blk8_p0_in_bind_dout [0:1],
  input  wire blk8_p0_in_bind_empty_n [0:1],
  output wire blk8_p0_in_bind_read [0:1],
  input  wire [7:0] blk8_a1_in_bind_dout [0:1],
  input  wire blk8_a1_in_bind_empty_n [0:1],
  output wire blk8_a1_in_bind_read [0:1],
  input  wire [7:0] blk8_w1_in_bind_dout [0:1],
  input  wire blk8_w1_in_bind_empty_n [0:1],
  output wire blk8_w1_in_bind_read [0:1],
  input  wire [31:0] blk8_p1_in_bind_dout [0:1],
  input  wire blk8_p1_in_bind_empty_n [0:1],
  output wire blk8_p1_in_bind_read [0:1],
  input  wire [7:0] blk8_a2_in_bind_dout [0:1],
  input  wire blk8_a2_in_bind_empty_n [0:1],
  output wire blk8_a2_in_bind_read [0:1],
  input  wire [7:0] blk8_w2_in_bind_dout [0:1],
  input  wire blk8_w2_in_bind_empty_n [0:1],
  output wire blk8_w2_in_bind_read [0:1],
  input  wire [31:0] blk8_p2_in_bind_dout [0:1],
  input  wire blk8_p2_in_bind_empty_n [0:1],
  output wire blk8_p2_in_bind_read [0:1],
  input  wire [7:0] blk8_a3_in_bind_dout [0:1],
  input  wire blk8_a3_in_bind_empty_n [0:1],
  output wire blk8_a3_in_bind_read [0:1],
  input  wire [7:0] blk8_w3_in_bind_dout [0:1],
  input  wire blk8_w3_in_bind_empty_n [0:1],
  output wire blk8_w3_in_bind_read [0:1],
  input  wire [31:0] blk8_p3_in_bind_dout [0:1],
  input  wire blk8_p3_in_bind_empty_n [0:1],
  output wire blk8_p3_in_bind_read [0:1],
  input  wire [7:0] blk8_a4_in_bind_dout [0:1],
  input  wire blk8_a4_in_bind_empty_n [0:1],
  output wire blk8_a4_in_bind_read [0:1],
  input  wire [7:0] blk8_w4_in_bind_dout [0:1],
  input  wire blk8_w4_in_bind_empty_n [0:1],
  output wire blk8_w4_in_bind_read [0:1],
  input  wire [31:0] blk8_p4_in_bind_dout [0:1],
  input  wire blk8_p4_in_bind_empty_n [0:1],
  output wire blk8_p4_in_bind_read [0:1],
  input  wire [7:0] blk8_a5_in_bind_dout [0:1],
  input  wire blk8_a5_in_bind_empty_n [0:1],
  output wire blk8_a5_in_bind_read [0:1],
  input  wire [7:0] blk8_w5_in_bind_dout [0:1],
  input  wire blk8_w5_in_bind_empty_n [0:1],
  output wire blk8_w5_in_bind_read [0:1],
  input  wire [31:0] blk8_p5_in_bind_dout [0:1],
  input  wire blk8_p5_in_bind_empty_n [0:1],
  output wire blk8_p5_in_bind_read [0:1],
  input  wire [7:0] blk8_a6_in_bind_dout [0:1],
  input  wire blk8_a6_in_bind_empty_n [0:1],
  output wire blk8_a6_in_bind_read [0:1],
  input  wire [7:0] blk8_w6_in_bind_dout [0:1],
  input  wire blk8_w6_in_bind_empty_n [0:1],
  output wire blk8_w6_in_bind_read [0:1],
  input  wire [31:0] blk8_p6_in_bind_dout [0:1],
  input  wire blk8_p6_in_bind_empty_n [0:1],
  output wire blk8_p6_in_bind_read [0:1],
  input  wire [7:0] blk8_a7_in_bind_dout [0:1],
  input  wire blk8_a7_in_bind_empty_n [0:1],
  output wire blk8_a7_in_bind_read [0:1],
  input  wire [7:0] blk8_w7_in_bind_dout [0:1],
  input  wire blk8_w7_in_bind_empty_n [0:1],
  output wire blk8_w7_in_bind_read [0:1],
  input  wire [31:0] blk8_p7_in_bind_dout [0:1],
  input  wire blk8_p7_in_bind_empty_n [0:1],
  output wire blk8_p7_in_bind_read [0:1],
  output wire [31:0] lanes8_y0_out_bind_din [0:1],
  output wire lanes8_y0_out_bind_write [0:1],
  input  wire lanes8_y0_out_bind_full_n [0:1],
  output wire [31:0] lanes8_y1_out_bind_din [0:1],
  output wire lanes8_y1_out_bind_write [0:1],
  input  wire lanes8_y1_out_bind_full_n [0:1],
  output wire [31:0] lanes8_y2_out_bind_din [0:1],
  output wire lanes8_y2_out_bind_write [0:1],
  input  wire lanes8_y2_out_bind_full_n [0:1],
  output wire [31:0] lanes8_y3_out_bind_din [0:1],
  output wire lanes8_y3_out_bind_write [0:1],
  input  wire lanes8_y3_out_bind_full_n [0:1],
  output wire [31:0] lanes8_y4_out_bind_din [0:1],
  output wire lanes8_y4_out_bind_write [0:1],
  input  wire lanes8_y4_out_bind_full_n [0:1],
  output wire [31:0] lanes8_y5_out_bind_din [0:1],
  output wire lanes8_y5_out_bind_write [0:1],
  input  wire lanes8_y5_out_bind_full_n [0:1],
  output wire [31:0] lanes8_y6_out_bind_din [0:1],
  output wire lanes8_y6_out_bind_write [0:1],
  input  wire lanes8_y6_out_bind_full_n [0:1],
  output wire [31:0] lanes8_y7_out_bind_din [0:1],
  output wire lanes8_y7_out_bind_write [0:1],
  input  wire lanes8_y7_out_bind_full_n [0:1],
  input  wire [63:0] lanes8_b_mem_dout [0:1],
  input  wire lanes8_b_mem_empty_n [0:1],
  output wire lanes8_b_mem_read [0:1]
);
  // family blk8_a0_out_a0_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_a0_out_a0_in_din [0:3];
  wire [7:0] blk8_a0_out_a0_in_dout [0:3];
  wire blk8_a0_out_a0_in_full_n [0:3];
  wire blk8_a0_out_a0_in_write [0:3];
  wire blk8_a0_out_a0_in_empty_n [0:3];
  wire blk8_a0_out_a0_in_read [0:3];
  genvar blk8_a0_out_a0_in_i;
  generate
    for (blk8_a0_out_a0_in_i = 0; blk8_a0_out_a0_in_i < 4; blk8_a0_out_a0_in_i = blk8_a0_out_a0_in_i + 1) begin : g_blk8_a0_out_a0_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_a0_out_a0_in_din[blk8_a0_out_a0_in_i]), .full_n(blk8_a0_out_a0_in_full_n[blk8_a0_out_a0_in_i]), .write(blk8_a0_out_a0_in_write[blk8_a0_out_a0_in_i]), .dout(blk8_a0_out_a0_in_dout[blk8_a0_out_a0_in_i]), .empty_n(blk8_a0_out_a0_in_empty_n[blk8_a0_out_a0_in_i]), .read(blk8_a0_out_a0_in_read[blk8_a0_out_a0_in_i]));
    end
  endgenerate
  // family blk8_w0_out_w0_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_w0_out_w0_in_din [0:3];
  wire [7:0] blk8_w0_out_w0_in_dout [0:3];
  wire blk8_w0_out_w0_in_full_n [0:3];
  wire blk8_w0_out_w0_in_write [0:3];
  wire blk8_w0_out_w0_in_empty_n [0:3];
  wire blk8_w0_out_w0_in_read [0:3];
  genvar blk8_w0_out_w0_in_i;
  generate
    for (blk8_w0_out_w0_in_i = 0; blk8_w0_out_w0_in_i < 4; blk8_w0_out_w0_in_i = blk8_w0_out_w0_in_i + 1) begin : g_blk8_w0_out_w0_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_w0_out_w0_in_din[blk8_w0_out_w0_in_i]), .full_n(blk8_w0_out_w0_in_full_n[blk8_w0_out_w0_in_i]), .write(blk8_w0_out_w0_in_write[blk8_w0_out_w0_in_i]), .dout(blk8_w0_out_w0_in_dout[blk8_w0_out_w0_in_i]), .empty_n(blk8_w0_out_w0_in_empty_n[blk8_w0_out_w0_in_i]), .read(blk8_w0_out_w0_in_read[blk8_w0_out_w0_in_i]));
    end
  endgenerate
  // family blk8_p0_out_p0_in: 4 channel(s), 32-bit, depth 2
  wire [31:0] blk8_p0_out_p0_in_din [0:3];
  wire [31:0] blk8_p0_out_p0_in_dout [0:3];
  wire blk8_p0_out_p0_in_full_n [0:3];
  wire blk8_p0_out_p0_in_write [0:3];
  wire blk8_p0_out_p0_in_empty_n [0:3];
  wire blk8_p0_out_p0_in_read [0:3];
  genvar blk8_p0_out_p0_in_i;
  generate
    for (blk8_p0_out_p0_in_i = 0; blk8_p0_out_p0_in_i < 4; blk8_p0_out_p0_in_i = blk8_p0_out_p0_in_i + 1) begin : g_blk8_p0_out_p0_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_p0_out_p0_in_din[blk8_p0_out_p0_in_i]), .full_n(blk8_p0_out_p0_in_full_n[blk8_p0_out_p0_in_i]), .write(blk8_p0_out_p0_in_write[blk8_p0_out_p0_in_i]), .dout(blk8_p0_out_p0_in_dout[blk8_p0_out_p0_in_i]), .empty_n(blk8_p0_out_p0_in_empty_n[blk8_p0_out_p0_in_i]), .read(blk8_p0_out_p0_in_read[blk8_p0_out_p0_in_i]));
    end
  endgenerate
  // family blk8_a1_out_a1_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_a1_out_a1_in_din [0:3];
  wire [7:0] blk8_a1_out_a1_in_dout [0:3];
  wire blk8_a1_out_a1_in_full_n [0:3];
  wire blk8_a1_out_a1_in_write [0:3];
  wire blk8_a1_out_a1_in_empty_n [0:3];
  wire blk8_a1_out_a1_in_read [0:3];
  genvar blk8_a1_out_a1_in_i;
  generate
    for (blk8_a1_out_a1_in_i = 0; blk8_a1_out_a1_in_i < 4; blk8_a1_out_a1_in_i = blk8_a1_out_a1_in_i + 1) begin : g_blk8_a1_out_a1_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_a1_out_a1_in_din[blk8_a1_out_a1_in_i]), .full_n(blk8_a1_out_a1_in_full_n[blk8_a1_out_a1_in_i]), .write(blk8_a1_out_a1_in_write[blk8_a1_out_a1_in_i]), .dout(blk8_a1_out_a1_in_dout[blk8_a1_out_a1_in_i]), .empty_n(blk8_a1_out_a1_in_empty_n[blk8_a1_out_a1_in_i]), .read(blk8_a1_out_a1_in_read[blk8_a1_out_a1_in_i]));
    end
  endgenerate
  // family blk8_w1_out_w1_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_w1_out_w1_in_din [0:3];
  wire [7:0] blk8_w1_out_w1_in_dout [0:3];
  wire blk8_w1_out_w1_in_full_n [0:3];
  wire blk8_w1_out_w1_in_write [0:3];
  wire blk8_w1_out_w1_in_empty_n [0:3];
  wire blk8_w1_out_w1_in_read [0:3];
  genvar blk8_w1_out_w1_in_i;
  generate
    for (blk8_w1_out_w1_in_i = 0; blk8_w1_out_w1_in_i < 4; blk8_w1_out_w1_in_i = blk8_w1_out_w1_in_i + 1) begin : g_blk8_w1_out_w1_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_w1_out_w1_in_din[blk8_w1_out_w1_in_i]), .full_n(blk8_w1_out_w1_in_full_n[blk8_w1_out_w1_in_i]), .write(blk8_w1_out_w1_in_write[blk8_w1_out_w1_in_i]), .dout(blk8_w1_out_w1_in_dout[blk8_w1_out_w1_in_i]), .empty_n(blk8_w1_out_w1_in_empty_n[blk8_w1_out_w1_in_i]), .read(blk8_w1_out_w1_in_read[blk8_w1_out_w1_in_i]));
    end
  endgenerate
  // family blk8_p1_out_p1_in: 4 channel(s), 32-bit, depth 2
  wire [31:0] blk8_p1_out_p1_in_din [0:3];
  wire [31:0] blk8_p1_out_p1_in_dout [0:3];
  wire blk8_p1_out_p1_in_full_n [0:3];
  wire blk8_p1_out_p1_in_write [0:3];
  wire blk8_p1_out_p1_in_empty_n [0:3];
  wire blk8_p1_out_p1_in_read [0:3];
  genvar blk8_p1_out_p1_in_i;
  generate
    for (blk8_p1_out_p1_in_i = 0; blk8_p1_out_p1_in_i < 4; blk8_p1_out_p1_in_i = blk8_p1_out_p1_in_i + 1) begin : g_blk8_p1_out_p1_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_p1_out_p1_in_din[blk8_p1_out_p1_in_i]), .full_n(blk8_p1_out_p1_in_full_n[blk8_p1_out_p1_in_i]), .write(blk8_p1_out_p1_in_write[blk8_p1_out_p1_in_i]), .dout(blk8_p1_out_p1_in_dout[blk8_p1_out_p1_in_i]), .empty_n(blk8_p1_out_p1_in_empty_n[blk8_p1_out_p1_in_i]), .read(blk8_p1_out_p1_in_read[blk8_p1_out_p1_in_i]));
    end
  endgenerate
  // family blk8_a2_out_a2_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_a2_out_a2_in_din [0:3];
  wire [7:0] blk8_a2_out_a2_in_dout [0:3];
  wire blk8_a2_out_a2_in_full_n [0:3];
  wire blk8_a2_out_a2_in_write [0:3];
  wire blk8_a2_out_a2_in_empty_n [0:3];
  wire blk8_a2_out_a2_in_read [0:3];
  genvar blk8_a2_out_a2_in_i;
  generate
    for (blk8_a2_out_a2_in_i = 0; blk8_a2_out_a2_in_i < 4; blk8_a2_out_a2_in_i = blk8_a2_out_a2_in_i + 1) begin : g_blk8_a2_out_a2_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_a2_out_a2_in_din[blk8_a2_out_a2_in_i]), .full_n(blk8_a2_out_a2_in_full_n[blk8_a2_out_a2_in_i]), .write(blk8_a2_out_a2_in_write[blk8_a2_out_a2_in_i]), .dout(blk8_a2_out_a2_in_dout[blk8_a2_out_a2_in_i]), .empty_n(blk8_a2_out_a2_in_empty_n[blk8_a2_out_a2_in_i]), .read(blk8_a2_out_a2_in_read[blk8_a2_out_a2_in_i]));
    end
  endgenerate
  // family blk8_w2_out_w2_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_w2_out_w2_in_din [0:3];
  wire [7:0] blk8_w2_out_w2_in_dout [0:3];
  wire blk8_w2_out_w2_in_full_n [0:3];
  wire blk8_w2_out_w2_in_write [0:3];
  wire blk8_w2_out_w2_in_empty_n [0:3];
  wire blk8_w2_out_w2_in_read [0:3];
  genvar blk8_w2_out_w2_in_i;
  generate
    for (blk8_w2_out_w2_in_i = 0; blk8_w2_out_w2_in_i < 4; blk8_w2_out_w2_in_i = blk8_w2_out_w2_in_i + 1) begin : g_blk8_w2_out_w2_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_w2_out_w2_in_din[blk8_w2_out_w2_in_i]), .full_n(blk8_w2_out_w2_in_full_n[blk8_w2_out_w2_in_i]), .write(blk8_w2_out_w2_in_write[blk8_w2_out_w2_in_i]), .dout(blk8_w2_out_w2_in_dout[blk8_w2_out_w2_in_i]), .empty_n(blk8_w2_out_w2_in_empty_n[blk8_w2_out_w2_in_i]), .read(blk8_w2_out_w2_in_read[blk8_w2_out_w2_in_i]));
    end
  endgenerate
  // family blk8_p2_out_p2_in: 4 channel(s), 32-bit, depth 2
  wire [31:0] blk8_p2_out_p2_in_din [0:3];
  wire [31:0] blk8_p2_out_p2_in_dout [0:3];
  wire blk8_p2_out_p2_in_full_n [0:3];
  wire blk8_p2_out_p2_in_write [0:3];
  wire blk8_p2_out_p2_in_empty_n [0:3];
  wire blk8_p2_out_p2_in_read [0:3];
  genvar blk8_p2_out_p2_in_i;
  generate
    for (blk8_p2_out_p2_in_i = 0; blk8_p2_out_p2_in_i < 4; blk8_p2_out_p2_in_i = blk8_p2_out_p2_in_i + 1) begin : g_blk8_p2_out_p2_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_p2_out_p2_in_din[blk8_p2_out_p2_in_i]), .full_n(blk8_p2_out_p2_in_full_n[blk8_p2_out_p2_in_i]), .write(blk8_p2_out_p2_in_write[blk8_p2_out_p2_in_i]), .dout(blk8_p2_out_p2_in_dout[blk8_p2_out_p2_in_i]), .empty_n(blk8_p2_out_p2_in_empty_n[blk8_p2_out_p2_in_i]), .read(blk8_p2_out_p2_in_read[blk8_p2_out_p2_in_i]));
    end
  endgenerate
  // family blk8_a3_out_a3_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_a3_out_a3_in_din [0:3];
  wire [7:0] blk8_a3_out_a3_in_dout [0:3];
  wire blk8_a3_out_a3_in_full_n [0:3];
  wire blk8_a3_out_a3_in_write [0:3];
  wire blk8_a3_out_a3_in_empty_n [0:3];
  wire blk8_a3_out_a3_in_read [0:3];
  genvar blk8_a3_out_a3_in_i;
  generate
    for (blk8_a3_out_a3_in_i = 0; blk8_a3_out_a3_in_i < 4; blk8_a3_out_a3_in_i = blk8_a3_out_a3_in_i + 1) begin : g_blk8_a3_out_a3_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_a3_out_a3_in_din[blk8_a3_out_a3_in_i]), .full_n(blk8_a3_out_a3_in_full_n[blk8_a3_out_a3_in_i]), .write(blk8_a3_out_a3_in_write[blk8_a3_out_a3_in_i]), .dout(blk8_a3_out_a3_in_dout[blk8_a3_out_a3_in_i]), .empty_n(blk8_a3_out_a3_in_empty_n[blk8_a3_out_a3_in_i]), .read(blk8_a3_out_a3_in_read[blk8_a3_out_a3_in_i]));
    end
  endgenerate
  // family blk8_w3_out_w3_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_w3_out_w3_in_din [0:3];
  wire [7:0] blk8_w3_out_w3_in_dout [0:3];
  wire blk8_w3_out_w3_in_full_n [0:3];
  wire blk8_w3_out_w3_in_write [0:3];
  wire blk8_w3_out_w3_in_empty_n [0:3];
  wire blk8_w3_out_w3_in_read [0:3];
  genvar blk8_w3_out_w3_in_i;
  generate
    for (blk8_w3_out_w3_in_i = 0; blk8_w3_out_w3_in_i < 4; blk8_w3_out_w3_in_i = blk8_w3_out_w3_in_i + 1) begin : g_blk8_w3_out_w3_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_w3_out_w3_in_din[blk8_w3_out_w3_in_i]), .full_n(blk8_w3_out_w3_in_full_n[blk8_w3_out_w3_in_i]), .write(blk8_w3_out_w3_in_write[blk8_w3_out_w3_in_i]), .dout(blk8_w3_out_w3_in_dout[blk8_w3_out_w3_in_i]), .empty_n(blk8_w3_out_w3_in_empty_n[blk8_w3_out_w3_in_i]), .read(blk8_w3_out_w3_in_read[blk8_w3_out_w3_in_i]));
    end
  endgenerate
  // family blk8_p3_out_p3_in: 4 channel(s), 32-bit, depth 2
  wire [31:0] blk8_p3_out_p3_in_din [0:3];
  wire [31:0] blk8_p3_out_p3_in_dout [0:3];
  wire blk8_p3_out_p3_in_full_n [0:3];
  wire blk8_p3_out_p3_in_write [0:3];
  wire blk8_p3_out_p3_in_empty_n [0:3];
  wire blk8_p3_out_p3_in_read [0:3];
  genvar blk8_p3_out_p3_in_i;
  generate
    for (blk8_p3_out_p3_in_i = 0; blk8_p3_out_p3_in_i < 4; blk8_p3_out_p3_in_i = blk8_p3_out_p3_in_i + 1) begin : g_blk8_p3_out_p3_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_p3_out_p3_in_din[blk8_p3_out_p3_in_i]), .full_n(blk8_p3_out_p3_in_full_n[blk8_p3_out_p3_in_i]), .write(blk8_p3_out_p3_in_write[blk8_p3_out_p3_in_i]), .dout(blk8_p3_out_p3_in_dout[blk8_p3_out_p3_in_i]), .empty_n(blk8_p3_out_p3_in_empty_n[blk8_p3_out_p3_in_i]), .read(blk8_p3_out_p3_in_read[blk8_p3_out_p3_in_i]));
    end
  endgenerate
  // family blk8_a4_out_a4_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_a4_out_a4_in_din [0:3];
  wire [7:0] blk8_a4_out_a4_in_dout [0:3];
  wire blk8_a4_out_a4_in_full_n [0:3];
  wire blk8_a4_out_a4_in_write [0:3];
  wire blk8_a4_out_a4_in_empty_n [0:3];
  wire blk8_a4_out_a4_in_read [0:3];
  genvar blk8_a4_out_a4_in_i;
  generate
    for (blk8_a4_out_a4_in_i = 0; blk8_a4_out_a4_in_i < 4; blk8_a4_out_a4_in_i = blk8_a4_out_a4_in_i + 1) begin : g_blk8_a4_out_a4_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_a4_out_a4_in_din[blk8_a4_out_a4_in_i]), .full_n(blk8_a4_out_a4_in_full_n[blk8_a4_out_a4_in_i]), .write(blk8_a4_out_a4_in_write[blk8_a4_out_a4_in_i]), .dout(blk8_a4_out_a4_in_dout[blk8_a4_out_a4_in_i]), .empty_n(blk8_a4_out_a4_in_empty_n[blk8_a4_out_a4_in_i]), .read(blk8_a4_out_a4_in_read[blk8_a4_out_a4_in_i]));
    end
  endgenerate
  // family blk8_w4_out_w4_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_w4_out_w4_in_din [0:3];
  wire [7:0] blk8_w4_out_w4_in_dout [0:3];
  wire blk8_w4_out_w4_in_full_n [0:3];
  wire blk8_w4_out_w4_in_write [0:3];
  wire blk8_w4_out_w4_in_empty_n [0:3];
  wire blk8_w4_out_w4_in_read [0:3];
  genvar blk8_w4_out_w4_in_i;
  generate
    for (blk8_w4_out_w4_in_i = 0; blk8_w4_out_w4_in_i < 4; blk8_w4_out_w4_in_i = blk8_w4_out_w4_in_i + 1) begin : g_blk8_w4_out_w4_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_w4_out_w4_in_din[blk8_w4_out_w4_in_i]), .full_n(blk8_w4_out_w4_in_full_n[blk8_w4_out_w4_in_i]), .write(blk8_w4_out_w4_in_write[blk8_w4_out_w4_in_i]), .dout(blk8_w4_out_w4_in_dout[blk8_w4_out_w4_in_i]), .empty_n(blk8_w4_out_w4_in_empty_n[blk8_w4_out_w4_in_i]), .read(blk8_w4_out_w4_in_read[blk8_w4_out_w4_in_i]));
    end
  endgenerate
  // family blk8_p4_out_p4_in: 4 channel(s), 32-bit, depth 2
  wire [31:0] blk8_p4_out_p4_in_din [0:3];
  wire [31:0] blk8_p4_out_p4_in_dout [0:3];
  wire blk8_p4_out_p4_in_full_n [0:3];
  wire blk8_p4_out_p4_in_write [0:3];
  wire blk8_p4_out_p4_in_empty_n [0:3];
  wire blk8_p4_out_p4_in_read [0:3];
  genvar blk8_p4_out_p4_in_i;
  generate
    for (blk8_p4_out_p4_in_i = 0; blk8_p4_out_p4_in_i < 4; blk8_p4_out_p4_in_i = blk8_p4_out_p4_in_i + 1) begin : g_blk8_p4_out_p4_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_p4_out_p4_in_din[blk8_p4_out_p4_in_i]), .full_n(blk8_p4_out_p4_in_full_n[blk8_p4_out_p4_in_i]), .write(blk8_p4_out_p4_in_write[blk8_p4_out_p4_in_i]), .dout(blk8_p4_out_p4_in_dout[blk8_p4_out_p4_in_i]), .empty_n(blk8_p4_out_p4_in_empty_n[blk8_p4_out_p4_in_i]), .read(blk8_p4_out_p4_in_read[blk8_p4_out_p4_in_i]));
    end
  endgenerate
  // family blk8_a5_out_a5_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_a5_out_a5_in_din [0:3];
  wire [7:0] blk8_a5_out_a5_in_dout [0:3];
  wire blk8_a5_out_a5_in_full_n [0:3];
  wire blk8_a5_out_a5_in_write [0:3];
  wire blk8_a5_out_a5_in_empty_n [0:3];
  wire blk8_a5_out_a5_in_read [0:3];
  genvar blk8_a5_out_a5_in_i;
  generate
    for (blk8_a5_out_a5_in_i = 0; blk8_a5_out_a5_in_i < 4; blk8_a5_out_a5_in_i = blk8_a5_out_a5_in_i + 1) begin : g_blk8_a5_out_a5_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_a5_out_a5_in_din[blk8_a5_out_a5_in_i]), .full_n(blk8_a5_out_a5_in_full_n[blk8_a5_out_a5_in_i]), .write(blk8_a5_out_a5_in_write[blk8_a5_out_a5_in_i]), .dout(blk8_a5_out_a5_in_dout[blk8_a5_out_a5_in_i]), .empty_n(blk8_a5_out_a5_in_empty_n[blk8_a5_out_a5_in_i]), .read(blk8_a5_out_a5_in_read[blk8_a5_out_a5_in_i]));
    end
  endgenerate
  // family blk8_w5_out_w5_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_w5_out_w5_in_din [0:3];
  wire [7:0] blk8_w5_out_w5_in_dout [0:3];
  wire blk8_w5_out_w5_in_full_n [0:3];
  wire blk8_w5_out_w5_in_write [0:3];
  wire blk8_w5_out_w5_in_empty_n [0:3];
  wire blk8_w5_out_w5_in_read [0:3];
  genvar blk8_w5_out_w5_in_i;
  generate
    for (blk8_w5_out_w5_in_i = 0; blk8_w5_out_w5_in_i < 4; blk8_w5_out_w5_in_i = blk8_w5_out_w5_in_i + 1) begin : g_blk8_w5_out_w5_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_w5_out_w5_in_din[blk8_w5_out_w5_in_i]), .full_n(blk8_w5_out_w5_in_full_n[blk8_w5_out_w5_in_i]), .write(blk8_w5_out_w5_in_write[blk8_w5_out_w5_in_i]), .dout(blk8_w5_out_w5_in_dout[blk8_w5_out_w5_in_i]), .empty_n(blk8_w5_out_w5_in_empty_n[blk8_w5_out_w5_in_i]), .read(blk8_w5_out_w5_in_read[blk8_w5_out_w5_in_i]));
    end
  endgenerate
  // family blk8_p5_out_p5_in: 4 channel(s), 32-bit, depth 2
  wire [31:0] blk8_p5_out_p5_in_din [0:3];
  wire [31:0] blk8_p5_out_p5_in_dout [0:3];
  wire blk8_p5_out_p5_in_full_n [0:3];
  wire blk8_p5_out_p5_in_write [0:3];
  wire blk8_p5_out_p5_in_empty_n [0:3];
  wire blk8_p5_out_p5_in_read [0:3];
  genvar blk8_p5_out_p5_in_i;
  generate
    for (blk8_p5_out_p5_in_i = 0; blk8_p5_out_p5_in_i < 4; blk8_p5_out_p5_in_i = blk8_p5_out_p5_in_i + 1) begin : g_blk8_p5_out_p5_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_p5_out_p5_in_din[blk8_p5_out_p5_in_i]), .full_n(blk8_p5_out_p5_in_full_n[blk8_p5_out_p5_in_i]), .write(blk8_p5_out_p5_in_write[blk8_p5_out_p5_in_i]), .dout(blk8_p5_out_p5_in_dout[blk8_p5_out_p5_in_i]), .empty_n(blk8_p5_out_p5_in_empty_n[blk8_p5_out_p5_in_i]), .read(blk8_p5_out_p5_in_read[blk8_p5_out_p5_in_i]));
    end
  endgenerate
  // family blk8_a6_out_a6_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_a6_out_a6_in_din [0:3];
  wire [7:0] blk8_a6_out_a6_in_dout [0:3];
  wire blk8_a6_out_a6_in_full_n [0:3];
  wire blk8_a6_out_a6_in_write [0:3];
  wire blk8_a6_out_a6_in_empty_n [0:3];
  wire blk8_a6_out_a6_in_read [0:3];
  genvar blk8_a6_out_a6_in_i;
  generate
    for (blk8_a6_out_a6_in_i = 0; blk8_a6_out_a6_in_i < 4; blk8_a6_out_a6_in_i = blk8_a6_out_a6_in_i + 1) begin : g_blk8_a6_out_a6_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_a6_out_a6_in_din[blk8_a6_out_a6_in_i]), .full_n(blk8_a6_out_a6_in_full_n[blk8_a6_out_a6_in_i]), .write(blk8_a6_out_a6_in_write[blk8_a6_out_a6_in_i]), .dout(blk8_a6_out_a6_in_dout[blk8_a6_out_a6_in_i]), .empty_n(blk8_a6_out_a6_in_empty_n[blk8_a6_out_a6_in_i]), .read(blk8_a6_out_a6_in_read[blk8_a6_out_a6_in_i]));
    end
  endgenerate
  // family blk8_w6_out_w6_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_w6_out_w6_in_din [0:3];
  wire [7:0] blk8_w6_out_w6_in_dout [0:3];
  wire blk8_w6_out_w6_in_full_n [0:3];
  wire blk8_w6_out_w6_in_write [0:3];
  wire blk8_w6_out_w6_in_empty_n [0:3];
  wire blk8_w6_out_w6_in_read [0:3];
  genvar blk8_w6_out_w6_in_i;
  generate
    for (blk8_w6_out_w6_in_i = 0; blk8_w6_out_w6_in_i < 4; blk8_w6_out_w6_in_i = blk8_w6_out_w6_in_i + 1) begin : g_blk8_w6_out_w6_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_w6_out_w6_in_din[blk8_w6_out_w6_in_i]), .full_n(blk8_w6_out_w6_in_full_n[blk8_w6_out_w6_in_i]), .write(blk8_w6_out_w6_in_write[blk8_w6_out_w6_in_i]), .dout(blk8_w6_out_w6_in_dout[blk8_w6_out_w6_in_i]), .empty_n(blk8_w6_out_w6_in_empty_n[blk8_w6_out_w6_in_i]), .read(blk8_w6_out_w6_in_read[blk8_w6_out_w6_in_i]));
    end
  endgenerate
  // family blk8_p6_out_p6_in: 4 channel(s), 32-bit, depth 2
  wire [31:0] blk8_p6_out_p6_in_din [0:3];
  wire [31:0] blk8_p6_out_p6_in_dout [0:3];
  wire blk8_p6_out_p6_in_full_n [0:3];
  wire blk8_p6_out_p6_in_write [0:3];
  wire blk8_p6_out_p6_in_empty_n [0:3];
  wire blk8_p6_out_p6_in_read [0:3];
  genvar blk8_p6_out_p6_in_i;
  generate
    for (blk8_p6_out_p6_in_i = 0; blk8_p6_out_p6_in_i < 4; blk8_p6_out_p6_in_i = blk8_p6_out_p6_in_i + 1) begin : g_blk8_p6_out_p6_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_p6_out_p6_in_din[blk8_p6_out_p6_in_i]), .full_n(blk8_p6_out_p6_in_full_n[blk8_p6_out_p6_in_i]), .write(blk8_p6_out_p6_in_write[blk8_p6_out_p6_in_i]), .dout(blk8_p6_out_p6_in_dout[blk8_p6_out_p6_in_i]), .empty_n(blk8_p6_out_p6_in_empty_n[blk8_p6_out_p6_in_i]), .read(blk8_p6_out_p6_in_read[blk8_p6_out_p6_in_i]));
    end
  endgenerate
  // family blk8_a7_out_a7_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_a7_out_a7_in_din [0:3];
  wire [7:0] blk8_a7_out_a7_in_dout [0:3];
  wire blk8_a7_out_a7_in_full_n [0:3];
  wire blk8_a7_out_a7_in_write [0:3];
  wire blk8_a7_out_a7_in_empty_n [0:3];
  wire blk8_a7_out_a7_in_read [0:3];
  genvar blk8_a7_out_a7_in_i;
  generate
    for (blk8_a7_out_a7_in_i = 0; blk8_a7_out_a7_in_i < 4; blk8_a7_out_a7_in_i = blk8_a7_out_a7_in_i + 1) begin : g_blk8_a7_out_a7_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_a7_out_a7_in_din[blk8_a7_out_a7_in_i]), .full_n(blk8_a7_out_a7_in_full_n[blk8_a7_out_a7_in_i]), .write(blk8_a7_out_a7_in_write[blk8_a7_out_a7_in_i]), .dout(blk8_a7_out_a7_in_dout[blk8_a7_out_a7_in_i]), .empty_n(blk8_a7_out_a7_in_empty_n[blk8_a7_out_a7_in_i]), .read(blk8_a7_out_a7_in_read[blk8_a7_out_a7_in_i]));
    end
  endgenerate
  // family blk8_w7_out_w7_in: 4 channel(s), 8-bit, depth 2
  wire [7:0] blk8_w7_out_w7_in_din [0:3];
  wire [7:0] blk8_w7_out_w7_in_dout [0:3];
  wire blk8_w7_out_w7_in_full_n [0:3];
  wire blk8_w7_out_w7_in_write [0:3];
  wire blk8_w7_out_w7_in_empty_n [0:3];
  wire blk8_w7_out_w7_in_read [0:3];
  genvar blk8_w7_out_w7_in_i;
  generate
    for (blk8_w7_out_w7_in_i = 0; blk8_w7_out_w7_in_i < 4; blk8_w7_out_w7_in_i = blk8_w7_out_w7_in_i + 1) begin : g_blk8_w7_out_w7_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_w7_out_w7_in_din[blk8_w7_out_w7_in_i]), .full_n(blk8_w7_out_w7_in_full_n[blk8_w7_out_w7_in_i]), .write(blk8_w7_out_w7_in_write[blk8_w7_out_w7_in_i]), .dout(blk8_w7_out_w7_in_dout[blk8_w7_out_w7_in_i]), .empty_n(blk8_w7_out_w7_in_empty_n[blk8_w7_out_w7_in_i]), .read(blk8_w7_out_w7_in_read[blk8_w7_out_w7_in_i]));
    end
  endgenerate
  // family blk8_p7_out_p7_in: 4 channel(s), 32-bit, depth 2
  wire [31:0] blk8_p7_out_p7_in_din [0:3];
  wire [31:0] blk8_p7_out_p7_in_dout [0:3];
  wire blk8_p7_out_p7_in_full_n [0:3];
  wire blk8_p7_out_p7_in_write [0:3];
  wire blk8_p7_out_p7_in_empty_n [0:3];
  wire blk8_p7_out_p7_in_read [0:3];
  genvar blk8_p7_out_p7_in_i;
  generate
    for (blk8_p7_out_p7_in_i = 0; blk8_p7_out_p7_in_i < 4; blk8_p7_out_p7_in_i = blk8_p7_out_p7_in_i + 1) begin : g_blk8_p7_out_p7_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(blk8_p7_out_p7_in_din[blk8_p7_out_p7_in_i]), .full_n(blk8_p7_out_p7_in_full_n[blk8_p7_out_p7_in_i]), .write(blk8_p7_out_p7_in_write[blk8_p7_out_p7_in_i]), .dout(blk8_p7_out_p7_in_dout[blk8_p7_out_p7_in_i]), .empty_n(blk8_p7_out_p7_in_empty_n[blk8_p7_out_p7_in_i]), .read(blk8_p7_out_p7_in_read[blk8_p7_out_p7_in_i]));
    end
  endgenerate
  // family lanes8_z0_in_bind: 2 channel(s), 32-bit, depth 2
  wire [31:0] lanes8_z0_in_bind_din [0:1];
  wire [31:0] lanes8_z0_in_bind_dout [0:1];
  wire lanes8_z0_in_bind_full_n [0:1];
  wire lanes8_z0_in_bind_write [0:1];
  wire lanes8_z0_in_bind_empty_n [0:1];
  wire lanes8_z0_in_bind_read [0:1];
  genvar lanes8_z0_in_bind_i;
  generate
    for (lanes8_z0_in_bind_i = 0; lanes8_z0_in_bind_i < 2; lanes8_z0_in_bind_i = lanes8_z0_in_bind_i + 1) begin : g_lanes8_z0_in_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(lanes8_z0_in_bind_din[lanes8_z0_in_bind_i]), .full_n(lanes8_z0_in_bind_full_n[lanes8_z0_in_bind_i]), .write(lanes8_z0_in_bind_write[lanes8_z0_in_bind_i]), .dout(lanes8_z0_in_bind_dout[lanes8_z0_in_bind_i]), .empty_n(lanes8_z0_in_bind_empty_n[lanes8_z0_in_bind_i]), .read(lanes8_z0_in_bind_read[lanes8_z0_in_bind_i]));
    end
  endgenerate
  // family lanes8_z1_in_bind: 2 channel(s), 32-bit, depth 2
  wire [31:0] lanes8_z1_in_bind_din [0:1];
  wire [31:0] lanes8_z1_in_bind_dout [0:1];
  wire lanes8_z1_in_bind_full_n [0:1];
  wire lanes8_z1_in_bind_write [0:1];
  wire lanes8_z1_in_bind_empty_n [0:1];
  wire lanes8_z1_in_bind_read [0:1];
  genvar lanes8_z1_in_bind_i;
  generate
    for (lanes8_z1_in_bind_i = 0; lanes8_z1_in_bind_i < 2; lanes8_z1_in_bind_i = lanes8_z1_in_bind_i + 1) begin : g_lanes8_z1_in_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(lanes8_z1_in_bind_din[lanes8_z1_in_bind_i]), .full_n(lanes8_z1_in_bind_full_n[lanes8_z1_in_bind_i]), .write(lanes8_z1_in_bind_write[lanes8_z1_in_bind_i]), .dout(lanes8_z1_in_bind_dout[lanes8_z1_in_bind_i]), .empty_n(lanes8_z1_in_bind_empty_n[lanes8_z1_in_bind_i]), .read(lanes8_z1_in_bind_read[lanes8_z1_in_bind_i]));
    end
  endgenerate
  // family lanes8_z2_in_bind: 2 channel(s), 32-bit, depth 2
  wire [31:0] lanes8_z2_in_bind_din [0:1];
  wire [31:0] lanes8_z2_in_bind_dout [0:1];
  wire lanes8_z2_in_bind_full_n [0:1];
  wire lanes8_z2_in_bind_write [0:1];
  wire lanes8_z2_in_bind_empty_n [0:1];
  wire lanes8_z2_in_bind_read [0:1];
  genvar lanes8_z2_in_bind_i;
  generate
    for (lanes8_z2_in_bind_i = 0; lanes8_z2_in_bind_i < 2; lanes8_z2_in_bind_i = lanes8_z2_in_bind_i + 1) begin : g_lanes8_z2_in_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(lanes8_z2_in_bind_din[lanes8_z2_in_bind_i]), .full_n(lanes8_z2_in_bind_full_n[lanes8_z2_in_bind_i]), .write(lanes8_z2_in_bind_write[lanes8_z2_in_bind_i]), .dout(lanes8_z2_in_bind_dout[lanes8_z2_in_bind_i]), .empty_n(lanes8_z2_in_bind_empty_n[lanes8_z2_in_bind_i]), .read(lanes8_z2_in_bind_read[lanes8_z2_in_bind_i]));
    end
  endgenerate
  // family lanes8_z3_in_bind: 2 channel(s), 32-bit, depth 2
  wire [31:0] lanes8_z3_in_bind_din [0:1];
  wire [31:0] lanes8_z3_in_bind_dout [0:1];
  wire lanes8_z3_in_bind_full_n [0:1];
  wire lanes8_z3_in_bind_write [0:1];
  wire lanes8_z3_in_bind_empty_n [0:1];
  wire lanes8_z3_in_bind_read [0:1];
  genvar lanes8_z3_in_bind_i;
  generate
    for (lanes8_z3_in_bind_i = 0; lanes8_z3_in_bind_i < 2; lanes8_z3_in_bind_i = lanes8_z3_in_bind_i + 1) begin : g_lanes8_z3_in_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(lanes8_z3_in_bind_din[lanes8_z3_in_bind_i]), .full_n(lanes8_z3_in_bind_full_n[lanes8_z3_in_bind_i]), .write(lanes8_z3_in_bind_write[lanes8_z3_in_bind_i]), .dout(lanes8_z3_in_bind_dout[lanes8_z3_in_bind_i]), .empty_n(lanes8_z3_in_bind_empty_n[lanes8_z3_in_bind_i]), .read(lanes8_z3_in_bind_read[lanes8_z3_in_bind_i]));
    end
  endgenerate
  // family lanes8_z4_in_bind: 2 channel(s), 32-bit, depth 2
  wire [31:0] lanes8_z4_in_bind_din [0:1];
  wire [31:0] lanes8_z4_in_bind_dout [0:1];
  wire lanes8_z4_in_bind_full_n [0:1];
  wire lanes8_z4_in_bind_write [0:1];
  wire lanes8_z4_in_bind_empty_n [0:1];
  wire lanes8_z4_in_bind_read [0:1];
  genvar lanes8_z4_in_bind_i;
  generate
    for (lanes8_z4_in_bind_i = 0; lanes8_z4_in_bind_i < 2; lanes8_z4_in_bind_i = lanes8_z4_in_bind_i + 1) begin : g_lanes8_z4_in_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(lanes8_z4_in_bind_din[lanes8_z4_in_bind_i]), .full_n(lanes8_z4_in_bind_full_n[lanes8_z4_in_bind_i]), .write(lanes8_z4_in_bind_write[lanes8_z4_in_bind_i]), .dout(lanes8_z4_in_bind_dout[lanes8_z4_in_bind_i]), .empty_n(lanes8_z4_in_bind_empty_n[lanes8_z4_in_bind_i]), .read(lanes8_z4_in_bind_read[lanes8_z4_in_bind_i]));
    end
  endgenerate
  // family lanes8_z5_in_bind: 2 channel(s), 32-bit, depth 2
  wire [31:0] lanes8_z5_in_bind_din [0:1];
  wire [31:0] lanes8_z5_in_bind_dout [0:1];
  wire lanes8_z5_in_bind_full_n [0:1];
  wire lanes8_z5_in_bind_write [0:1];
  wire lanes8_z5_in_bind_empty_n [0:1];
  wire lanes8_z5_in_bind_read [0:1];
  genvar lanes8_z5_in_bind_i;
  generate
    for (lanes8_z5_in_bind_i = 0; lanes8_z5_in_bind_i < 2; lanes8_z5_in_bind_i = lanes8_z5_in_bind_i + 1) begin : g_lanes8_z5_in_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(lanes8_z5_in_bind_din[lanes8_z5_in_bind_i]), .full_n(lanes8_z5_in_bind_full_n[lanes8_z5_in_bind_i]), .write(lanes8_z5_in_bind_write[lanes8_z5_in_bind_i]), .dout(lanes8_z5_in_bind_dout[lanes8_z5_in_bind_i]), .empty_n(lanes8_z5_in_bind_empty_n[lanes8_z5_in_bind_i]), .read(lanes8_z5_in_bind_read[lanes8_z5_in_bind_i]));
    end
  endgenerate
  // family lanes8_z6_in_bind: 2 channel(s), 32-bit, depth 2
  wire [31:0] lanes8_z6_in_bind_din [0:1];
  wire [31:0] lanes8_z6_in_bind_dout [0:1];
  wire lanes8_z6_in_bind_full_n [0:1];
  wire lanes8_z6_in_bind_write [0:1];
  wire lanes8_z6_in_bind_empty_n [0:1];
  wire lanes8_z6_in_bind_read [0:1];
  genvar lanes8_z6_in_bind_i;
  generate
    for (lanes8_z6_in_bind_i = 0; lanes8_z6_in_bind_i < 2; lanes8_z6_in_bind_i = lanes8_z6_in_bind_i + 1) begin : g_lanes8_z6_in_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(lanes8_z6_in_bind_din[lanes8_z6_in_bind_i]), .full_n(lanes8_z6_in_bind_full_n[lanes8_z6_in_bind_i]), .write(lanes8_z6_in_bind_write[lanes8_z6_in_bind_i]), .dout(lanes8_z6_in_bind_dout[lanes8_z6_in_bind_i]), .empty_n(lanes8_z6_in_bind_empty_n[lanes8_z6_in_bind_i]), .read(lanes8_z6_in_bind_read[lanes8_z6_in_bind_i]));
    end
  endgenerate
  // family lanes8_z7_in_bind: 2 channel(s), 32-bit, depth 2
  wire [31:0] lanes8_z7_in_bind_din [0:1];
  wire [31:0] lanes8_z7_in_bind_dout [0:1];
  wire lanes8_z7_in_bind_full_n [0:1];
  wire lanes8_z7_in_bind_write [0:1];
  wire lanes8_z7_in_bind_empty_n [0:1];
  wire lanes8_z7_in_bind_read [0:1];
  genvar lanes8_z7_in_bind_i;
  generate
    for (lanes8_z7_in_bind_i = 0; lanes8_z7_in_bind_i < 2; lanes8_z7_in_bind_i = lanes8_z7_in_bind_i + 1) begin : g_lanes8_z7_in_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(lanes8_z7_in_bind_din[lanes8_z7_in_bind_i]), .full_n(lanes8_z7_in_bind_full_n[lanes8_z7_in_bind_i]), .write(lanes8_z7_in_bind_write[lanes8_z7_in_bind_i]), .dout(lanes8_z7_in_bind_dout[lanes8_z7_in_bind_i]), .empty_n(lanes8_z7_in_bind_empty_n[lanes8_z7_in_bind_i]), .read(lanes8_z7_in_bind_read[lanes8_z7_in_bind_i]));
    end
  endgenerate
  // role blk8_r0: 1 instance(s)
  blk8_r0 u_blk8_r0_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a0_in_dout(blk8_a0_out_a0_in_dout[3]),
      .a0_in_empty_n(blk8_a0_out_a0_in_empty_n[3]),
      .a0_in_read(blk8_a0_out_a0_in_read[3]),
      .a1_in_dout(blk8_a1_out_a1_in_dout[3]),
      .a1_in_empty_n(blk8_a1_out_a1_in_empty_n[3]),
      .a1_in_read(blk8_a1_out_a1_in_read[3]),
      .a2_in_dout(blk8_a2_out_a2_in_dout[3]),
      .a2_in_empty_n(blk8_a2_out_a2_in_empty_n[3]),
      .a2_in_read(blk8_a2_out_a2_in_read[3]),
      .a3_in_dout(blk8_a3_out_a3_in_dout[3]),
      .a3_in_empty_n(blk8_a3_out_a3_in_empty_n[3]),
      .a3_in_read(blk8_a3_out_a3_in_read[3]),
      .a4_in_dout(blk8_a4_out_a4_in_dout[3]),
      .a4_in_empty_n(blk8_a4_out_a4_in_empty_n[3]),
      .a4_in_read(blk8_a4_out_a4_in_read[3]),
      .a5_in_dout(blk8_a5_out_a5_in_dout[3]),
      .a5_in_empty_n(blk8_a5_out_a5_in_empty_n[3]),
      .a5_in_read(blk8_a5_out_a5_in_read[3]),
      .a6_in_dout(blk8_a6_out_a6_in_dout[3]),
      .a6_in_empty_n(blk8_a6_out_a6_in_empty_n[3]),
      .a6_in_read(blk8_a6_out_a6_in_read[3]),
      .a7_in_dout(blk8_a7_out_a7_in_dout[3]),
      .a7_in_empty_n(blk8_a7_out_a7_in_empty_n[3]),
      .a7_in_read(blk8_a7_out_a7_in_read[3]),
      .p0_in_dout(blk8_p0_out_p0_in_dout[3]),
      .p0_in_empty_n(blk8_p0_out_p0_in_empty_n[3]),
      .p0_in_read(blk8_p0_out_p0_in_read[3]),
      .p0_out_din(lanes8_z0_in_bind_din[1]),
      .p0_out_full_n(lanes8_z0_in_bind_full_n[1]),
      .p0_out_write(lanes8_z0_in_bind_write[1]),
      .p1_in_dout(blk8_p1_out_p1_in_dout[3]),
      .p1_in_empty_n(blk8_p1_out_p1_in_empty_n[3]),
      .p1_in_read(blk8_p1_out_p1_in_read[3]),
      .p1_out_din(lanes8_z1_in_bind_din[1]),
      .p1_out_full_n(lanes8_z1_in_bind_full_n[1]),
      .p1_out_write(lanes8_z1_in_bind_write[1]),
      .p2_in_dout(blk8_p2_out_p2_in_dout[3]),
      .p2_in_empty_n(blk8_p2_out_p2_in_empty_n[3]),
      .p2_in_read(blk8_p2_out_p2_in_read[3]),
      .p2_out_din(lanes8_z2_in_bind_din[1]),
      .p2_out_full_n(lanes8_z2_in_bind_full_n[1]),
      .p2_out_write(lanes8_z2_in_bind_write[1]),
      .p3_in_dout(blk8_p3_out_p3_in_dout[3]),
      .p3_in_empty_n(blk8_p3_out_p3_in_empty_n[3]),
      .p3_in_read(blk8_p3_out_p3_in_read[3]),
      .p3_out_din(lanes8_z3_in_bind_din[1]),
      .p3_out_full_n(lanes8_z3_in_bind_full_n[1]),
      .p3_out_write(lanes8_z3_in_bind_write[1]),
      .p4_in_dout(blk8_p4_out_p4_in_dout[3]),
      .p4_in_empty_n(blk8_p4_out_p4_in_empty_n[3]),
      .p4_in_read(blk8_p4_out_p4_in_read[3]),
      .p4_out_din(lanes8_z4_in_bind_din[1]),
      .p4_out_full_n(lanes8_z4_in_bind_full_n[1]),
      .p4_out_write(lanes8_z4_in_bind_write[1]),
      .p5_in_dout(blk8_p5_out_p5_in_dout[3]),
      .p5_in_empty_n(blk8_p5_out_p5_in_empty_n[3]),
      .p5_in_read(blk8_p5_out_p5_in_read[3]),
      .p5_out_din(lanes8_z5_in_bind_din[1]),
      .p5_out_full_n(lanes8_z5_in_bind_full_n[1]),
      .p5_out_write(lanes8_z5_in_bind_write[1]),
      .p6_in_dout(blk8_p6_out_p6_in_dout[3]),
      .p6_in_empty_n(blk8_p6_out_p6_in_empty_n[3]),
      .p6_in_read(blk8_p6_out_p6_in_read[3]),
      .p6_out_din(lanes8_z6_in_bind_din[1]),
      .p6_out_full_n(lanes8_z6_in_bind_full_n[1]),
      .p6_out_write(lanes8_z6_in_bind_write[1]),
      .p7_in_dout(blk8_p7_out_p7_in_dout[3]),
      .p7_in_empty_n(blk8_p7_out_p7_in_empty_n[3]),
      .p7_in_read(blk8_p7_out_p7_in_read[3]),
      .p7_out_din(lanes8_z7_in_bind_din[1]),
      .p7_out_full_n(lanes8_z7_in_bind_full_n[1]),
      .p7_out_write(lanes8_z7_in_bind_write[1]),
      .w0_in_dout(blk8_w0_out_w0_in_dout[3]),
      .w0_in_empty_n(blk8_w0_out_w0_in_empty_n[3]),
      .w0_in_read(blk8_w0_out_w0_in_read[3]),
      .w1_in_dout(blk8_w1_out_w1_in_dout[3]),
      .w1_in_empty_n(blk8_w1_out_w1_in_empty_n[3]),
      .w1_in_read(blk8_w1_out_w1_in_read[3]),
      .w2_in_dout(blk8_w2_out_w2_in_dout[3]),
      .w2_in_empty_n(blk8_w2_out_w2_in_empty_n[3]),
      .w2_in_read(blk8_w2_out_w2_in_read[3]),
      .w3_in_dout(blk8_w3_out_w3_in_dout[3]),
      .w3_in_empty_n(blk8_w3_out_w3_in_empty_n[3]),
      .w3_in_read(blk8_w3_out_w3_in_read[3]),
      .w4_in_dout(blk8_w4_out_w4_in_dout[3]),
      .w4_in_empty_n(blk8_w4_out_w4_in_empty_n[3]),
      .w4_in_read(blk8_w4_out_w4_in_read[3]),
      .w5_in_dout(blk8_w5_out_w5_in_dout[3]),
      .w5_in_empty_n(blk8_w5_out_w5_in_empty_n[3]),
      .w5_in_read(blk8_w5_out_w5_in_read[3]),
      .w6_in_dout(blk8_w6_out_w6_in_dout[3]),
      .w6_in_empty_n(blk8_w6_out_w6_in_empty_n[3]),
      .w6_in_read(blk8_w6_out_w6_in_read[3]),
      .w7_in_dout(blk8_w7_out_w7_in_dout[3]),
      .w7_in_empty_n(blk8_w7_out_w7_in_empty_n[3]),
      .w7_in_read(blk8_w7_out_w7_in_read[3]));
  // role blk8_r1: 1 instance(s)
  blk8_r1 u_blk8_r1_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a0_in_dout(blk8_a0_out_a0_in_dout[1]),
      .a0_in_empty_n(blk8_a0_out_a0_in_empty_n[1]),
      .a0_in_read(blk8_a0_out_a0_in_read[1]),
      .a1_in_dout(blk8_a1_out_a1_in_dout[1]),
      .a1_in_empty_n(blk8_a1_out_a1_in_empty_n[1]),
      .a1_in_read(blk8_a1_out_a1_in_read[1]),
      .a2_in_dout(blk8_a2_out_a2_in_dout[1]),
      .a2_in_empty_n(blk8_a2_out_a2_in_empty_n[1]),
      .a2_in_read(blk8_a2_out_a2_in_read[1]),
      .a3_in_dout(blk8_a3_out_a3_in_dout[1]),
      .a3_in_empty_n(blk8_a3_out_a3_in_empty_n[1]),
      .a3_in_read(blk8_a3_out_a3_in_read[1]),
      .a4_in_dout(blk8_a4_out_a4_in_dout[1]),
      .a4_in_empty_n(blk8_a4_out_a4_in_empty_n[1]),
      .a4_in_read(blk8_a4_out_a4_in_read[1]),
      .a5_in_dout(blk8_a5_out_a5_in_dout[1]),
      .a5_in_empty_n(blk8_a5_out_a5_in_empty_n[1]),
      .a5_in_read(blk8_a5_out_a5_in_read[1]),
      .a6_in_dout(blk8_a6_out_a6_in_dout[1]),
      .a6_in_empty_n(blk8_a6_out_a6_in_empty_n[1]),
      .a6_in_read(blk8_a6_out_a6_in_read[1]),
      .a7_in_dout(blk8_a7_out_a7_in_dout[1]),
      .a7_in_empty_n(blk8_a7_out_a7_in_empty_n[1]),
      .a7_in_read(blk8_a7_out_a7_in_read[1]),
      .p0_in_dout(blk8_p0_in_bind_dout[1]),
      .p0_in_empty_n(blk8_p0_in_bind_empty_n[1]),
      .p0_in_read(blk8_p0_in_bind_read[1]),
      .p0_out_din(blk8_p0_out_p0_in_din[3]),
      .p0_out_full_n(blk8_p0_out_p0_in_full_n[3]),
      .p0_out_write(blk8_p0_out_p0_in_write[3]),
      .p1_in_dout(blk8_p1_in_bind_dout[1]),
      .p1_in_empty_n(blk8_p1_in_bind_empty_n[1]),
      .p1_in_read(blk8_p1_in_bind_read[1]),
      .p1_out_din(blk8_p1_out_p1_in_din[3]),
      .p1_out_full_n(blk8_p1_out_p1_in_full_n[3]),
      .p1_out_write(blk8_p1_out_p1_in_write[3]),
      .p2_in_dout(blk8_p2_in_bind_dout[1]),
      .p2_in_empty_n(blk8_p2_in_bind_empty_n[1]),
      .p2_in_read(blk8_p2_in_bind_read[1]),
      .p2_out_din(blk8_p2_out_p2_in_din[3]),
      .p2_out_full_n(blk8_p2_out_p2_in_full_n[3]),
      .p2_out_write(blk8_p2_out_p2_in_write[3]),
      .p3_in_dout(blk8_p3_in_bind_dout[1]),
      .p3_in_empty_n(blk8_p3_in_bind_empty_n[1]),
      .p3_in_read(blk8_p3_in_bind_read[1]),
      .p3_out_din(blk8_p3_out_p3_in_din[3]),
      .p3_out_full_n(blk8_p3_out_p3_in_full_n[3]),
      .p3_out_write(blk8_p3_out_p3_in_write[3]),
      .p4_in_dout(blk8_p4_in_bind_dout[1]),
      .p4_in_empty_n(blk8_p4_in_bind_empty_n[1]),
      .p4_in_read(blk8_p4_in_bind_read[1]),
      .p4_out_din(blk8_p4_out_p4_in_din[3]),
      .p4_out_full_n(blk8_p4_out_p4_in_full_n[3]),
      .p4_out_write(blk8_p4_out_p4_in_write[3]),
      .p5_in_dout(blk8_p5_in_bind_dout[1]),
      .p5_in_empty_n(blk8_p5_in_bind_empty_n[1]),
      .p5_in_read(blk8_p5_in_bind_read[1]),
      .p5_out_din(blk8_p5_out_p5_in_din[3]),
      .p5_out_full_n(blk8_p5_out_p5_in_full_n[3]),
      .p5_out_write(blk8_p5_out_p5_in_write[3]),
      .p6_in_dout(blk8_p6_in_bind_dout[1]),
      .p6_in_empty_n(blk8_p6_in_bind_empty_n[1]),
      .p6_in_read(blk8_p6_in_bind_read[1]),
      .p6_out_din(blk8_p6_out_p6_in_din[3]),
      .p6_out_full_n(blk8_p6_out_p6_in_full_n[3]),
      .p6_out_write(blk8_p6_out_p6_in_write[3]),
      .p7_in_dout(blk8_p7_in_bind_dout[1]),
      .p7_in_empty_n(blk8_p7_in_bind_empty_n[1]),
      .p7_in_read(blk8_p7_in_bind_read[1]),
      .p7_out_din(blk8_p7_out_p7_in_din[3]),
      .p7_out_full_n(blk8_p7_out_p7_in_full_n[3]),
      .p7_out_write(blk8_p7_out_p7_in_write[3]),
      .w0_in_dout(blk8_w0_out_w0_in_dout[1]),
      .w0_in_empty_n(blk8_w0_out_w0_in_empty_n[1]),
      .w0_in_read(blk8_w0_out_w0_in_read[1]),
      .w1_in_dout(blk8_w1_out_w1_in_dout[1]),
      .w1_in_empty_n(blk8_w1_out_w1_in_empty_n[1]),
      .w1_in_read(blk8_w1_out_w1_in_read[1]),
      .w2_in_dout(blk8_w2_out_w2_in_dout[1]),
      .w2_in_empty_n(blk8_w2_out_w2_in_empty_n[1]),
      .w2_in_read(blk8_w2_out_w2_in_read[1]),
      .w3_in_dout(blk8_w3_out_w3_in_dout[1]),
      .w3_in_empty_n(blk8_w3_out_w3_in_empty_n[1]),
      .w3_in_read(blk8_w3_out_w3_in_read[1]),
      .w4_in_dout(blk8_w4_out_w4_in_dout[1]),
      .w4_in_empty_n(blk8_w4_out_w4_in_empty_n[1]),
      .w4_in_read(blk8_w4_out_w4_in_read[1]),
      .w5_in_dout(blk8_w5_out_w5_in_dout[1]),
      .w5_in_empty_n(blk8_w5_out_w5_in_empty_n[1]),
      .w5_in_read(blk8_w5_out_w5_in_read[1]),
      .w6_in_dout(blk8_w6_out_w6_in_dout[1]),
      .w6_in_empty_n(blk8_w6_out_w6_in_empty_n[1]),
      .w6_in_read(blk8_w6_out_w6_in_read[1]),
      .w7_in_dout(blk8_w7_out_w7_in_dout[1]),
      .w7_in_empty_n(blk8_w7_out_w7_in_empty_n[1]),
      .w7_in_read(blk8_w7_out_w7_in_read[1]));
  // role blk8_r2: 1 instance(s)
  blk8_r2 u_blk8_r2_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a0_in_dout(blk8_a0_in_bind_dout[1]),
      .a0_in_empty_n(blk8_a0_in_bind_empty_n[1]),
      .a0_in_read(blk8_a0_in_bind_read[1]),
      .a0_out_din(blk8_a0_out_a0_in_din[3]),
      .a0_out_full_n(blk8_a0_out_a0_in_full_n[3]),
      .a0_out_write(blk8_a0_out_a0_in_write[3]),
      .a1_in_dout(blk8_a1_in_bind_dout[1]),
      .a1_in_empty_n(blk8_a1_in_bind_empty_n[1]),
      .a1_in_read(blk8_a1_in_bind_read[1]),
      .a1_out_din(blk8_a1_out_a1_in_din[3]),
      .a1_out_full_n(blk8_a1_out_a1_in_full_n[3]),
      .a1_out_write(blk8_a1_out_a1_in_write[3]),
      .a2_in_dout(blk8_a2_in_bind_dout[1]),
      .a2_in_empty_n(blk8_a2_in_bind_empty_n[1]),
      .a2_in_read(blk8_a2_in_bind_read[1]),
      .a2_out_din(blk8_a2_out_a2_in_din[3]),
      .a2_out_full_n(blk8_a2_out_a2_in_full_n[3]),
      .a2_out_write(blk8_a2_out_a2_in_write[3]),
      .a3_in_dout(blk8_a3_in_bind_dout[1]),
      .a3_in_empty_n(blk8_a3_in_bind_empty_n[1]),
      .a3_in_read(blk8_a3_in_bind_read[1]),
      .a3_out_din(blk8_a3_out_a3_in_din[3]),
      .a3_out_full_n(blk8_a3_out_a3_in_full_n[3]),
      .a3_out_write(blk8_a3_out_a3_in_write[3]),
      .a4_in_dout(blk8_a4_in_bind_dout[1]),
      .a4_in_empty_n(blk8_a4_in_bind_empty_n[1]),
      .a4_in_read(blk8_a4_in_bind_read[1]),
      .a4_out_din(blk8_a4_out_a4_in_din[3]),
      .a4_out_full_n(blk8_a4_out_a4_in_full_n[3]),
      .a4_out_write(blk8_a4_out_a4_in_write[3]),
      .a5_in_dout(blk8_a5_in_bind_dout[1]),
      .a5_in_empty_n(blk8_a5_in_bind_empty_n[1]),
      .a5_in_read(blk8_a5_in_bind_read[1]),
      .a5_out_din(blk8_a5_out_a5_in_din[3]),
      .a5_out_full_n(blk8_a5_out_a5_in_full_n[3]),
      .a5_out_write(blk8_a5_out_a5_in_write[3]),
      .a6_in_dout(blk8_a6_in_bind_dout[1]),
      .a6_in_empty_n(blk8_a6_in_bind_empty_n[1]),
      .a6_in_read(blk8_a6_in_bind_read[1]),
      .a6_out_din(blk8_a6_out_a6_in_din[3]),
      .a6_out_full_n(blk8_a6_out_a6_in_full_n[3]),
      .a6_out_write(blk8_a6_out_a6_in_write[3]),
      .a7_in_dout(blk8_a7_in_bind_dout[1]),
      .a7_in_empty_n(blk8_a7_in_bind_empty_n[1]),
      .a7_in_read(blk8_a7_in_bind_read[1]),
      .a7_out_din(blk8_a7_out_a7_in_din[3]),
      .a7_out_full_n(blk8_a7_out_a7_in_full_n[3]),
      .a7_out_write(blk8_a7_out_a7_in_write[3]),
      .p0_in_dout(blk8_p0_out_p0_in_dout[2]),
      .p0_in_empty_n(blk8_p0_out_p0_in_empty_n[2]),
      .p0_in_read(blk8_p0_out_p0_in_read[2]),
      .p0_out_din(lanes8_z0_in_bind_din[0]),
      .p0_out_full_n(lanes8_z0_in_bind_full_n[0]),
      .p0_out_write(lanes8_z0_in_bind_write[0]),
      .p1_in_dout(blk8_p1_out_p1_in_dout[2]),
      .p1_in_empty_n(blk8_p1_out_p1_in_empty_n[2]),
      .p1_in_read(blk8_p1_out_p1_in_read[2]),
      .p1_out_din(lanes8_z1_in_bind_din[0]),
      .p1_out_full_n(lanes8_z1_in_bind_full_n[0]),
      .p1_out_write(lanes8_z1_in_bind_write[0]),
      .p2_in_dout(blk8_p2_out_p2_in_dout[2]),
      .p2_in_empty_n(blk8_p2_out_p2_in_empty_n[2]),
      .p2_in_read(blk8_p2_out_p2_in_read[2]),
      .p2_out_din(lanes8_z2_in_bind_din[0]),
      .p2_out_full_n(lanes8_z2_in_bind_full_n[0]),
      .p2_out_write(lanes8_z2_in_bind_write[0]),
      .p3_in_dout(blk8_p3_out_p3_in_dout[2]),
      .p3_in_empty_n(blk8_p3_out_p3_in_empty_n[2]),
      .p3_in_read(blk8_p3_out_p3_in_read[2]),
      .p3_out_din(lanes8_z3_in_bind_din[0]),
      .p3_out_full_n(lanes8_z3_in_bind_full_n[0]),
      .p3_out_write(lanes8_z3_in_bind_write[0]),
      .p4_in_dout(blk8_p4_out_p4_in_dout[2]),
      .p4_in_empty_n(blk8_p4_out_p4_in_empty_n[2]),
      .p4_in_read(blk8_p4_out_p4_in_read[2]),
      .p4_out_din(lanes8_z4_in_bind_din[0]),
      .p4_out_full_n(lanes8_z4_in_bind_full_n[0]),
      .p4_out_write(lanes8_z4_in_bind_write[0]),
      .p5_in_dout(blk8_p5_out_p5_in_dout[2]),
      .p5_in_empty_n(blk8_p5_out_p5_in_empty_n[2]),
      .p5_in_read(blk8_p5_out_p5_in_read[2]),
      .p5_out_din(lanes8_z5_in_bind_din[0]),
      .p5_out_full_n(lanes8_z5_in_bind_full_n[0]),
      .p5_out_write(lanes8_z5_in_bind_write[0]),
      .p6_in_dout(blk8_p6_out_p6_in_dout[2]),
      .p6_in_empty_n(blk8_p6_out_p6_in_empty_n[2]),
      .p6_in_read(blk8_p6_out_p6_in_read[2]),
      .p6_out_din(lanes8_z6_in_bind_din[0]),
      .p6_out_full_n(lanes8_z6_in_bind_full_n[0]),
      .p6_out_write(lanes8_z6_in_bind_write[0]),
      .p7_in_dout(blk8_p7_out_p7_in_dout[2]),
      .p7_in_empty_n(blk8_p7_out_p7_in_empty_n[2]),
      .p7_in_read(blk8_p7_out_p7_in_read[2]),
      .p7_out_din(lanes8_z7_in_bind_din[0]),
      .p7_out_full_n(lanes8_z7_in_bind_full_n[0]),
      .p7_out_write(lanes8_z7_in_bind_write[0]),
      .w0_in_dout(blk8_w0_in_bind_dout[1]),
      .w0_in_empty_n(blk8_w0_in_bind_empty_n[1]),
      .w0_in_read(blk8_w0_in_bind_read[1]),
      .w0_out_din(blk8_w0_out_w0_in_din[3]),
      .w0_out_full_n(blk8_w0_out_w0_in_full_n[3]),
      .w0_out_write(blk8_w0_out_w0_in_write[3]),
      .w1_in_dout(blk8_w1_in_bind_dout[1]),
      .w1_in_empty_n(blk8_w1_in_bind_empty_n[1]),
      .w1_in_read(blk8_w1_in_bind_read[1]),
      .w1_out_din(blk8_w1_out_w1_in_din[3]),
      .w1_out_full_n(blk8_w1_out_w1_in_full_n[3]),
      .w1_out_write(blk8_w1_out_w1_in_write[3]),
      .w2_in_dout(blk8_w2_in_bind_dout[1]),
      .w2_in_empty_n(blk8_w2_in_bind_empty_n[1]),
      .w2_in_read(blk8_w2_in_bind_read[1]),
      .w2_out_din(blk8_w2_out_w2_in_din[3]),
      .w2_out_full_n(blk8_w2_out_w2_in_full_n[3]),
      .w2_out_write(blk8_w2_out_w2_in_write[3]),
      .w3_in_dout(blk8_w3_in_bind_dout[1]),
      .w3_in_empty_n(blk8_w3_in_bind_empty_n[1]),
      .w3_in_read(blk8_w3_in_bind_read[1]),
      .w3_out_din(blk8_w3_out_w3_in_din[3]),
      .w3_out_full_n(blk8_w3_out_w3_in_full_n[3]),
      .w3_out_write(blk8_w3_out_w3_in_write[3]),
      .w4_in_dout(blk8_w4_in_bind_dout[1]),
      .w4_in_empty_n(blk8_w4_in_bind_empty_n[1]),
      .w4_in_read(blk8_w4_in_bind_read[1]),
      .w4_out_din(blk8_w4_out_w4_in_din[3]),
      .w4_out_full_n(blk8_w4_out_w4_in_full_n[3]),
      .w4_out_write(blk8_w4_out_w4_in_write[3]),
      .w5_in_dout(blk8_w5_in_bind_dout[1]),
      .w5_in_empty_n(blk8_w5_in_bind_empty_n[1]),
      .w5_in_read(blk8_w5_in_bind_read[1]),
      .w5_out_din(blk8_w5_out_w5_in_din[3]),
      .w5_out_full_n(blk8_w5_out_w5_in_full_n[3]),
      .w5_out_write(blk8_w5_out_w5_in_write[3]),
      .w6_in_dout(blk8_w6_in_bind_dout[1]),
      .w6_in_empty_n(blk8_w6_in_bind_empty_n[1]),
      .w6_in_read(blk8_w6_in_bind_read[1]),
      .w6_out_din(blk8_w6_out_w6_in_din[3]),
      .w6_out_full_n(blk8_w6_out_w6_in_full_n[3]),
      .w6_out_write(blk8_w6_out_w6_in_write[3]),
      .w7_in_dout(blk8_w7_in_bind_dout[1]),
      .w7_in_empty_n(blk8_w7_in_bind_empty_n[1]),
      .w7_in_read(blk8_w7_in_bind_read[1]),
      .w7_out_din(blk8_w7_out_w7_in_din[3]),
      .w7_out_full_n(blk8_w7_out_w7_in_full_n[3]),
      .w7_out_write(blk8_w7_out_w7_in_write[3]));
  // role blk8_r3: 1 instance(s)
  blk8_r3 u_blk8_r3_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a0_in_dout(blk8_a0_in_bind_dout[0]),
      .a0_in_empty_n(blk8_a0_in_bind_empty_n[0]),
      .a0_in_read(blk8_a0_in_bind_read[0]),
      .a0_out_din(blk8_a0_out_a0_in_din[1]),
      .a0_out_full_n(blk8_a0_out_a0_in_full_n[1]),
      .a0_out_write(blk8_a0_out_a0_in_write[1]),
      .a1_in_dout(blk8_a1_in_bind_dout[0]),
      .a1_in_empty_n(blk8_a1_in_bind_empty_n[0]),
      .a1_in_read(blk8_a1_in_bind_read[0]),
      .a1_out_din(blk8_a1_out_a1_in_din[1]),
      .a1_out_full_n(blk8_a1_out_a1_in_full_n[1]),
      .a1_out_write(blk8_a1_out_a1_in_write[1]),
      .a2_in_dout(blk8_a2_in_bind_dout[0]),
      .a2_in_empty_n(blk8_a2_in_bind_empty_n[0]),
      .a2_in_read(blk8_a2_in_bind_read[0]),
      .a2_out_din(blk8_a2_out_a2_in_din[1]),
      .a2_out_full_n(blk8_a2_out_a2_in_full_n[1]),
      .a2_out_write(blk8_a2_out_a2_in_write[1]),
      .a3_in_dout(blk8_a3_in_bind_dout[0]),
      .a3_in_empty_n(blk8_a3_in_bind_empty_n[0]),
      .a3_in_read(blk8_a3_in_bind_read[0]),
      .a3_out_din(blk8_a3_out_a3_in_din[1]),
      .a3_out_full_n(blk8_a3_out_a3_in_full_n[1]),
      .a3_out_write(blk8_a3_out_a3_in_write[1]),
      .a4_in_dout(blk8_a4_in_bind_dout[0]),
      .a4_in_empty_n(blk8_a4_in_bind_empty_n[0]),
      .a4_in_read(blk8_a4_in_bind_read[0]),
      .a4_out_din(blk8_a4_out_a4_in_din[1]),
      .a4_out_full_n(blk8_a4_out_a4_in_full_n[1]),
      .a4_out_write(blk8_a4_out_a4_in_write[1]),
      .a5_in_dout(blk8_a5_in_bind_dout[0]),
      .a5_in_empty_n(blk8_a5_in_bind_empty_n[0]),
      .a5_in_read(blk8_a5_in_bind_read[0]),
      .a5_out_din(blk8_a5_out_a5_in_din[1]),
      .a5_out_full_n(blk8_a5_out_a5_in_full_n[1]),
      .a5_out_write(blk8_a5_out_a5_in_write[1]),
      .a6_in_dout(blk8_a6_in_bind_dout[0]),
      .a6_in_empty_n(blk8_a6_in_bind_empty_n[0]),
      .a6_in_read(blk8_a6_in_bind_read[0]),
      .a6_out_din(blk8_a6_out_a6_in_din[1]),
      .a6_out_full_n(blk8_a6_out_a6_in_full_n[1]),
      .a6_out_write(blk8_a6_out_a6_in_write[1]),
      .a7_in_dout(blk8_a7_in_bind_dout[0]),
      .a7_in_empty_n(blk8_a7_in_bind_empty_n[0]),
      .a7_in_read(blk8_a7_in_bind_read[0]),
      .a7_out_din(blk8_a7_out_a7_in_din[1]),
      .a7_out_full_n(blk8_a7_out_a7_in_full_n[1]),
      .a7_out_write(blk8_a7_out_a7_in_write[1]),
      .p0_in_dout(blk8_p0_in_bind_dout[0]),
      .p0_in_empty_n(blk8_p0_in_bind_empty_n[0]),
      .p0_in_read(blk8_p0_in_bind_read[0]),
      .p0_out_din(blk8_p0_out_p0_in_din[2]),
      .p0_out_full_n(blk8_p0_out_p0_in_full_n[2]),
      .p0_out_write(blk8_p0_out_p0_in_write[2]),
      .p1_in_dout(blk8_p1_in_bind_dout[0]),
      .p1_in_empty_n(blk8_p1_in_bind_empty_n[0]),
      .p1_in_read(blk8_p1_in_bind_read[0]),
      .p1_out_din(blk8_p1_out_p1_in_din[2]),
      .p1_out_full_n(blk8_p1_out_p1_in_full_n[2]),
      .p1_out_write(blk8_p1_out_p1_in_write[2]),
      .p2_in_dout(blk8_p2_in_bind_dout[0]),
      .p2_in_empty_n(blk8_p2_in_bind_empty_n[0]),
      .p2_in_read(blk8_p2_in_bind_read[0]),
      .p2_out_din(blk8_p2_out_p2_in_din[2]),
      .p2_out_full_n(blk8_p2_out_p2_in_full_n[2]),
      .p2_out_write(blk8_p2_out_p2_in_write[2]),
      .p3_in_dout(blk8_p3_in_bind_dout[0]),
      .p3_in_empty_n(blk8_p3_in_bind_empty_n[0]),
      .p3_in_read(blk8_p3_in_bind_read[0]),
      .p3_out_din(blk8_p3_out_p3_in_din[2]),
      .p3_out_full_n(blk8_p3_out_p3_in_full_n[2]),
      .p3_out_write(blk8_p3_out_p3_in_write[2]),
      .p4_in_dout(blk8_p4_in_bind_dout[0]),
      .p4_in_empty_n(blk8_p4_in_bind_empty_n[0]),
      .p4_in_read(blk8_p4_in_bind_read[0]),
      .p4_out_din(blk8_p4_out_p4_in_din[2]),
      .p4_out_full_n(blk8_p4_out_p4_in_full_n[2]),
      .p4_out_write(blk8_p4_out_p4_in_write[2]),
      .p5_in_dout(blk8_p5_in_bind_dout[0]),
      .p5_in_empty_n(blk8_p5_in_bind_empty_n[0]),
      .p5_in_read(blk8_p5_in_bind_read[0]),
      .p5_out_din(blk8_p5_out_p5_in_din[2]),
      .p5_out_full_n(blk8_p5_out_p5_in_full_n[2]),
      .p5_out_write(blk8_p5_out_p5_in_write[2]),
      .p6_in_dout(blk8_p6_in_bind_dout[0]),
      .p6_in_empty_n(blk8_p6_in_bind_empty_n[0]),
      .p6_in_read(blk8_p6_in_bind_read[0]),
      .p6_out_din(blk8_p6_out_p6_in_din[2]),
      .p6_out_full_n(blk8_p6_out_p6_in_full_n[2]),
      .p6_out_write(blk8_p6_out_p6_in_write[2]),
      .p7_in_dout(blk8_p7_in_bind_dout[0]),
      .p7_in_empty_n(blk8_p7_in_bind_empty_n[0]),
      .p7_in_read(blk8_p7_in_bind_read[0]),
      .p7_out_din(blk8_p7_out_p7_in_din[2]),
      .p7_out_full_n(blk8_p7_out_p7_in_full_n[2]),
      .p7_out_write(blk8_p7_out_p7_in_write[2]),
      .w0_in_dout(blk8_w0_in_bind_dout[0]),
      .w0_in_empty_n(blk8_w0_in_bind_empty_n[0]),
      .w0_in_read(blk8_w0_in_bind_read[0]),
      .w0_out_din(blk8_w0_out_w0_in_din[1]),
      .w0_out_full_n(blk8_w0_out_w0_in_full_n[1]),
      .w0_out_write(blk8_w0_out_w0_in_write[1]),
      .w1_in_dout(blk8_w1_in_bind_dout[0]),
      .w1_in_empty_n(blk8_w1_in_bind_empty_n[0]),
      .w1_in_read(blk8_w1_in_bind_read[0]),
      .w1_out_din(blk8_w1_out_w1_in_din[1]),
      .w1_out_full_n(blk8_w1_out_w1_in_full_n[1]),
      .w1_out_write(blk8_w1_out_w1_in_write[1]),
      .w2_in_dout(blk8_w2_in_bind_dout[0]),
      .w2_in_empty_n(blk8_w2_in_bind_empty_n[0]),
      .w2_in_read(blk8_w2_in_bind_read[0]),
      .w2_out_din(blk8_w2_out_w2_in_din[1]),
      .w2_out_full_n(blk8_w2_out_w2_in_full_n[1]),
      .w2_out_write(blk8_w2_out_w2_in_write[1]),
      .w3_in_dout(blk8_w3_in_bind_dout[0]),
      .w3_in_empty_n(blk8_w3_in_bind_empty_n[0]),
      .w3_in_read(blk8_w3_in_bind_read[0]),
      .w3_out_din(blk8_w3_out_w3_in_din[1]),
      .w3_out_full_n(blk8_w3_out_w3_in_full_n[1]),
      .w3_out_write(blk8_w3_out_w3_in_write[1]),
      .w4_in_dout(blk8_w4_in_bind_dout[0]),
      .w4_in_empty_n(blk8_w4_in_bind_empty_n[0]),
      .w4_in_read(blk8_w4_in_bind_read[0]),
      .w4_out_din(blk8_w4_out_w4_in_din[1]),
      .w4_out_full_n(blk8_w4_out_w4_in_full_n[1]),
      .w4_out_write(blk8_w4_out_w4_in_write[1]),
      .w5_in_dout(blk8_w5_in_bind_dout[0]),
      .w5_in_empty_n(blk8_w5_in_bind_empty_n[0]),
      .w5_in_read(blk8_w5_in_bind_read[0]),
      .w5_out_din(blk8_w5_out_w5_in_din[1]),
      .w5_out_full_n(blk8_w5_out_w5_in_full_n[1]),
      .w5_out_write(blk8_w5_out_w5_in_write[1]),
      .w6_in_dout(blk8_w6_in_bind_dout[0]),
      .w6_in_empty_n(blk8_w6_in_bind_empty_n[0]),
      .w6_in_read(blk8_w6_in_bind_read[0]),
      .w6_out_din(blk8_w6_out_w6_in_din[1]),
      .w6_out_full_n(blk8_w6_out_w6_in_full_n[1]),
      .w6_out_write(blk8_w6_out_w6_in_write[1]),
      .w7_in_dout(blk8_w7_in_bind_dout[0]),
      .w7_in_empty_n(blk8_w7_in_bind_empty_n[0]),
      .w7_in_read(blk8_w7_in_bind_read[0]),
      .w7_out_din(blk8_w7_out_w7_in_din[1]),
      .w7_out_full_n(blk8_w7_out_w7_in_full_n[1]),
      .w7_out_write(blk8_w7_out_w7_in_write[1]));
  // role lanes8_r0: 2 instance(s)
  lanes8_r0 u_lanes8_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .b_dout(lanes8_b_mem_dout[0]),
      .b_empty_n(lanes8_b_mem_empty_n[0]),
      .b_read(lanes8_b_mem_read[0]),
      .y0_out_din(lanes8_y0_out_bind_din[0]),
      .y0_out_full_n(lanes8_y0_out_bind_full_n[0]),
      .y0_out_write(lanes8_y0_out_bind_write[0]),
      .y1_out_din(lanes8_y1_out_bind_din[0]),
      .y1_out_full_n(lanes8_y1_out_bind_full_n[0]),
      .y1_out_write(lanes8_y1_out_bind_write[0]),
      .y2_out_din(lanes8_y2_out_bind_din[0]),
      .y2_out_full_n(lanes8_y2_out_bind_full_n[0]),
      .y2_out_write(lanes8_y2_out_bind_write[0]),
      .y3_out_din(lanes8_y3_out_bind_din[0]),
      .y3_out_full_n(lanes8_y3_out_bind_full_n[0]),
      .y3_out_write(lanes8_y3_out_bind_write[0]),
      .y4_out_din(lanes8_y4_out_bind_din[0]),
      .y4_out_full_n(lanes8_y4_out_bind_full_n[0]),
      .y4_out_write(lanes8_y4_out_bind_write[0]),
      .y5_out_din(lanes8_y5_out_bind_din[0]),
      .y5_out_full_n(lanes8_y5_out_bind_full_n[0]),
      .y5_out_write(lanes8_y5_out_bind_write[0]),
      .y6_out_din(lanes8_y6_out_bind_din[0]),
      .y6_out_full_n(lanes8_y6_out_bind_full_n[0]),
      .y6_out_write(lanes8_y6_out_bind_write[0]),
      .y7_out_din(lanes8_y7_out_bind_din[0]),
      .y7_out_full_n(lanes8_y7_out_bind_full_n[0]),
      .y7_out_write(lanes8_y7_out_bind_write[0]),
      .z0_in_dout(lanes8_z0_in_bind_dout[0]),
      .z0_in_empty_n(lanes8_z0_in_bind_empty_n[0]),
      .z0_in_read(lanes8_z0_in_bind_read[0]),
      .z1_in_dout(lanes8_z1_in_bind_dout[0]),
      .z1_in_empty_n(lanes8_z1_in_bind_empty_n[0]),
      .z1_in_read(lanes8_z1_in_bind_read[0]),
      .z2_in_dout(lanes8_z2_in_bind_dout[0]),
      .z2_in_empty_n(lanes8_z2_in_bind_empty_n[0]),
      .z2_in_read(lanes8_z2_in_bind_read[0]),
      .z3_in_dout(lanes8_z3_in_bind_dout[0]),
      .z3_in_empty_n(lanes8_z3_in_bind_empty_n[0]),
      .z3_in_read(lanes8_z3_in_bind_read[0]),
      .z4_in_dout(lanes8_z4_in_bind_dout[0]),
      .z4_in_empty_n(lanes8_z4_in_bind_empty_n[0]),
      .z4_in_read(lanes8_z4_in_bind_read[0]),
      .z5_in_dout(lanes8_z5_in_bind_dout[0]),
      .z5_in_empty_n(lanes8_z5_in_bind_empty_n[0]),
      .z5_in_read(lanes8_z5_in_bind_read[0]),
      .z6_in_dout(lanes8_z6_in_bind_dout[0]),
      .z6_in_empty_n(lanes8_z6_in_bind_empty_n[0]),
      .z6_in_read(lanes8_z6_in_bind_read[0]),
      .z7_in_dout(lanes8_z7_in_bind_dout[0]),
      .z7_in_empty_n(lanes8_z7_in_bind_empty_n[0]),
      .z7_in_read(lanes8_z7_in_bind_read[0]));
  lanes8_r0 u_lanes8_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .b_dout(lanes8_b_mem_dout[1]),
      .b_empty_n(lanes8_b_mem_empty_n[1]),
      .b_read(lanes8_b_mem_read[1]),
      .y0_out_din(lanes8_y0_out_bind_din[1]),
      .y0_out_full_n(lanes8_y0_out_bind_full_n[1]),
      .y0_out_write(lanes8_y0_out_bind_write[1]),
      .y1_out_din(lanes8_y1_out_bind_din[1]),
      .y1_out_full_n(lanes8_y1_out_bind_full_n[1]),
      .y1_out_write(lanes8_y1_out_bind_write[1]),
      .y2_out_din(lanes8_y2_out_bind_din[1]),
      .y2_out_full_n(lanes8_y2_out_bind_full_n[1]),
      .y2_out_write(lanes8_y2_out_bind_write[1]),
      .y3_out_din(lanes8_y3_out_bind_din[1]),
      .y3_out_full_n(lanes8_y3_out_bind_full_n[1]),
      .y3_out_write(lanes8_y3_out_bind_write[1]),
      .y4_out_din(lanes8_y4_out_bind_din[1]),
      .y4_out_full_n(lanes8_y4_out_bind_full_n[1]),
      .y4_out_write(lanes8_y4_out_bind_write[1]),
      .y5_out_din(lanes8_y5_out_bind_din[1]),
      .y5_out_full_n(lanes8_y5_out_bind_full_n[1]),
      .y5_out_write(lanes8_y5_out_bind_write[1]),
      .y6_out_din(lanes8_y6_out_bind_din[1]),
      .y6_out_full_n(lanes8_y6_out_bind_full_n[1]),
      .y6_out_write(lanes8_y6_out_bind_write[1]),
      .y7_out_din(lanes8_y7_out_bind_din[1]),
      .y7_out_full_n(lanes8_y7_out_bind_full_n[1]),
      .y7_out_write(lanes8_y7_out_bind_write[1]),
      .z0_in_dout(lanes8_z0_in_bind_dout[1]),
      .z0_in_empty_n(lanes8_z0_in_bind_empty_n[1]),
      .z0_in_read(lanes8_z0_in_bind_read[1]),
      .z1_in_dout(lanes8_z1_in_bind_dout[1]),
      .z1_in_empty_n(lanes8_z1_in_bind_empty_n[1]),
      .z1_in_read(lanes8_z1_in_bind_read[1]),
      .z2_in_dout(lanes8_z2_in_bind_dout[1]),
      .z2_in_empty_n(lanes8_z2_in_bind_empty_n[1]),
      .z2_in_read(lanes8_z2_in_bind_read[1]),
      .z3_in_dout(lanes8_z3_in_bind_dout[1]),
      .z3_in_empty_n(lanes8_z3_in_bind_empty_n[1]),
      .z3_in_read(lanes8_z3_in_bind_read[1]),
      .z4_in_dout(lanes8_z4_in_bind_dout[1]),
      .z4_in_empty_n(lanes8_z4_in_bind_empty_n[1]),
      .z4_in_read(lanes8_z4_in_bind_read[1]),
      .z5_in_dout(lanes8_z5_in_bind_dout[1]),
      .z5_in_empty_n(lanes8_z5_in_bind_empty_n[1]),
      .z5_in_read(lanes8_z5_in_bind_read[1]),
      .z6_in_dout(lanes8_z6_in_bind_dout[1]),
      .z6_in_empty_n(lanes8_z6_in_bind_empty_n[1]),
      .z6_in_read(lanes8_z6_in_bind_read[1]),
      .z7_in_dout(lanes8_z7_in_bind_dout[1]),
      .z7_in_empty_n(lanes8_z7_in_bind_empty_n[1]),
      .z7_in_read(lanes8_z7_in_bind_read[1]));
endmodule
