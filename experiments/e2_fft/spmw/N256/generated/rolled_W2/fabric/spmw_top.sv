`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] delay0_x_in_bind_dout [0:1],
  input  wire delay0_x_in_bind_empty_n [0:1],
  output wire delay0_x_in_bind_read [0:1],
  output wire [63:0] reorder_y_out_bind_din [0:1],
  output wire reorder_y_out_bind_write [0:1],
  input  wire reorder_y_out_bind_full_n [0:1]
);
  // family delay1_x_in_bind: 2 channel(s), 64-bit, depth 2
  wire [63:0] delay1_x_in_bind_din [0:1];
  wire [63:0] delay1_x_in_bind_dout [0:1];
  wire delay1_x_in_bind_full_n [0:1];
  wire delay1_x_in_bind_write [0:1];
  wire delay1_x_in_bind_empty_n [0:1];
  wire delay1_x_in_bind_read [0:1];
  genvar delay1_x_in_bind_i;
  generate
    for (delay1_x_in_bind_i = 0; delay1_x_in_bind_i < 2; delay1_x_in_bind_i = delay1_x_in_bind_i + 1) begin : g_delay1_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(delay1_x_in_bind_din[delay1_x_in_bind_i]), .full_n(delay1_x_in_bind_full_n[delay1_x_in_bind_i]), .write(delay1_x_in_bind_write[delay1_x_in_bind_i]), .dout(delay1_x_in_bind_dout[delay1_x_in_bind_i]), .empty_n(delay1_x_in_bind_empty_n[delay1_x_in_bind_i]), .read(delay1_x_in_bind_read[delay1_x_in_bind_i]));
    end
  endgenerate
  // family delay2_x_in_bind: 2 channel(s), 64-bit, depth 2
  wire [63:0] delay2_x_in_bind_din [0:1];
  wire [63:0] delay2_x_in_bind_dout [0:1];
  wire delay2_x_in_bind_full_n [0:1];
  wire delay2_x_in_bind_write [0:1];
  wire delay2_x_in_bind_empty_n [0:1];
  wire delay2_x_in_bind_read [0:1];
  genvar delay2_x_in_bind_i;
  generate
    for (delay2_x_in_bind_i = 0; delay2_x_in_bind_i < 2; delay2_x_in_bind_i = delay2_x_in_bind_i + 1) begin : g_delay2_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(delay2_x_in_bind_din[delay2_x_in_bind_i]), .full_n(delay2_x_in_bind_full_n[delay2_x_in_bind_i]), .write(delay2_x_in_bind_write[delay2_x_in_bind_i]), .dout(delay2_x_in_bind_dout[delay2_x_in_bind_i]), .empty_n(delay2_x_in_bind_empty_n[delay2_x_in_bind_i]), .read(delay2_x_in_bind_read[delay2_x_in_bind_i]));
    end
  endgenerate
  // family delay3_x_in_bind: 2 channel(s), 64-bit, depth 2
  wire [63:0] delay3_x_in_bind_din [0:1];
  wire [63:0] delay3_x_in_bind_dout [0:1];
  wire delay3_x_in_bind_full_n [0:1];
  wire delay3_x_in_bind_write [0:1];
  wire delay3_x_in_bind_empty_n [0:1];
  wire delay3_x_in_bind_read [0:1];
  genvar delay3_x_in_bind_i;
  generate
    for (delay3_x_in_bind_i = 0; delay3_x_in_bind_i < 2; delay3_x_in_bind_i = delay3_x_in_bind_i + 1) begin : g_delay3_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(delay3_x_in_bind_din[delay3_x_in_bind_i]), .full_n(delay3_x_in_bind_full_n[delay3_x_in_bind_i]), .write(delay3_x_in_bind_write[delay3_x_in_bind_i]), .dout(delay3_x_in_bind_dout[delay3_x_in_bind_i]), .empty_n(delay3_x_in_bind_empty_n[delay3_x_in_bind_i]), .read(delay3_x_in_bind_read[delay3_x_in_bind_i]));
    end
  endgenerate
  // family delay4_x_in_bind: 2 channel(s), 64-bit, depth 2
  wire [63:0] delay4_x_in_bind_din [0:1];
  wire [63:0] delay4_x_in_bind_dout [0:1];
  wire delay4_x_in_bind_full_n [0:1];
  wire delay4_x_in_bind_write [0:1];
  wire delay4_x_in_bind_empty_n [0:1];
  wire delay4_x_in_bind_read [0:1];
  genvar delay4_x_in_bind_i;
  generate
    for (delay4_x_in_bind_i = 0; delay4_x_in_bind_i < 2; delay4_x_in_bind_i = delay4_x_in_bind_i + 1) begin : g_delay4_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(delay4_x_in_bind_din[delay4_x_in_bind_i]), .full_n(delay4_x_in_bind_full_n[delay4_x_in_bind_i]), .write(delay4_x_in_bind_write[delay4_x_in_bind_i]), .dout(delay4_x_in_bind_dout[delay4_x_in_bind_i]), .empty_n(delay4_x_in_bind_empty_n[delay4_x_in_bind_i]), .read(delay4_x_in_bind_read[delay4_x_in_bind_i]));
    end
  endgenerate
  // family delay5_x_in_bind: 2 channel(s), 64-bit, depth 2
  wire [63:0] delay5_x_in_bind_din [0:1];
  wire [63:0] delay5_x_in_bind_dout [0:1];
  wire delay5_x_in_bind_full_n [0:1];
  wire delay5_x_in_bind_write [0:1];
  wire delay5_x_in_bind_empty_n [0:1];
  wire delay5_x_in_bind_read [0:1];
  genvar delay5_x_in_bind_i;
  generate
    for (delay5_x_in_bind_i = 0; delay5_x_in_bind_i < 2; delay5_x_in_bind_i = delay5_x_in_bind_i + 1) begin : g_delay5_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(delay5_x_in_bind_din[delay5_x_in_bind_i]), .full_n(delay5_x_in_bind_full_n[delay5_x_in_bind_i]), .write(delay5_x_in_bind_write[delay5_x_in_bind_i]), .dout(delay5_x_in_bind_dout[delay5_x_in_bind_i]), .empty_n(delay5_x_in_bind_empty_n[delay5_x_in_bind_i]), .read(delay5_x_in_bind_read[delay5_x_in_bind_i]));
    end
  endgenerate
  // family delay6_x_in_bind: 2 channel(s), 64-bit, depth 2
  wire [63:0] delay6_x_in_bind_din [0:1];
  wire [63:0] delay6_x_in_bind_dout [0:1];
  wire delay6_x_in_bind_full_n [0:1];
  wire delay6_x_in_bind_write [0:1];
  wire delay6_x_in_bind_empty_n [0:1];
  wire delay6_x_in_bind_read [0:1];
  genvar delay6_x_in_bind_i;
  generate
    for (delay6_x_in_bind_i = 0; delay6_x_in_bind_i < 2; delay6_x_in_bind_i = delay6_x_in_bind_i + 1) begin : g_delay6_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(delay6_x_in_bind_din[delay6_x_in_bind_i]), .full_n(delay6_x_in_bind_full_n[delay6_x_in_bind_i]), .write(delay6_x_in_bind_write[delay6_x_in_bind_i]), .dout(delay6_x_in_bind_dout[delay6_x_in_bind_i]), .empty_n(delay6_x_in_bind_empty_n[delay6_x_in_bind_i]), .read(delay6_x_in_bind_read[delay6_x_in_bind_i]));
    end
  endgenerate
  // family cross_a_in_bind: 2 channel(s), 64-bit, depth 8
  wire [63:0] cross_a_in_bind_din [0:1];
  wire [63:0] cross_a_in_bind_dout [0:1];
  wire cross_a_in_bind_full_n [0:1];
  wire cross_a_in_bind_write [0:1];
  wire cross_a_in_bind_empty_n [0:1];
  wire cross_a_in_bind_read [0:1];
  genvar cross_a_in_bind_i;
  generate
    for (cross_a_in_bind_i = 0; cross_a_in_bind_i < 2; cross_a_in_bind_i = cross_a_in_bind_i + 1) begin : g_cross_a_in_bind
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(cross_a_in_bind_din[cross_a_in_bind_i]), .full_n(cross_a_in_bind_full_n[cross_a_in_bind_i]), .write(cross_a_in_bind_write[cross_a_in_bind_i]), .dout(cross_a_in_bind_dout[cross_a_in_bind_i]), .empty_n(cross_a_in_bind_empty_n[cross_a_in_bind_i]), .read(cross_a_in_bind_read[cross_a_in_bind_i]));
    end
  endgenerate
  // family cross_a_out_a_in: 4 channel(s), 64-bit, depth 8
  wire [63:0] cross_a_out_a_in_din [0:3];
  wire [63:0] cross_a_out_a_in_dout [0:3];
  wire cross_a_out_a_in_full_n [0:3];
  wire cross_a_out_a_in_write [0:3];
  wire cross_a_out_a_in_empty_n [0:3];
  wire cross_a_out_a_in_read [0:3];
  genvar cross_a_out_a_in_i;
  generate
    for (cross_a_out_a_in_i = 0; cross_a_out_a_in_i < 4; cross_a_out_a_in_i = cross_a_out_a_in_i + 1) begin : g_cross_a_out_a_in
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(cross_a_out_a_in_din[cross_a_out_a_in_i]), .full_n(cross_a_out_a_in_full_n[cross_a_out_a_in_i]), .write(cross_a_out_a_in_write[cross_a_out_a_in_i]), .dout(cross_a_out_a_in_dout[cross_a_out_a_in_i]), .empty_n(cross_a_out_a_in_empty_n[cross_a_out_a_in_i]), .read(cross_a_out_a_in_read[cross_a_out_a_in_i]));
    end
  endgenerate
  // family cross_b_out_b_in: 2 channel(s), 64-bit, depth 8
  wire [63:0] cross_b_out_b_in_din [0:1];
  wire [63:0] cross_b_out_b_in_dout [0:1];
  wire cross_b_out_b_in_full_n [0:1];
  wire cross_b_out_b_in_write [0:1];
  wire cross_b_out_b_in_empty_n [0:1];
  wire cross_b_out_b_in_read [0:1];
  genvar cross_b_out_b_in_i;
  generate
    for (cross_b_out_b_in_i = 0; cross_b_out_b_in_i < 2; cross_b_out_b_in_i = cross_b_out_b_in_i + 1) begin : g_cross_b_out_b_in
      spmw_fifo #(.DW(64), .DEPTH(8)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(cross_b_out_b_in_din[cross_b_out_b_in_i]), .full_n(cross_b_out_b_in_full_n[cross_b_out_b_in_i]), .write(cross_b_out_b_in_write[cross_b_out_b_in_i]), .dout(cross_b_out_b_in_dout[cross_b_out_b_in_i]), .empty_n(cross_b_out_b_in_empty_n[cross_b_out_b_in_i]), .read(cross_b_out_b_in_read[cross_b_out_b_in_i]));
    end
  endgenerate
  // family reorder_x_in_bind: 2 channel(s), 64-bit, depth 2
  wire [63:0] reorder_x_in_bind_din [0:1];
  wire [63:0] reorder_x_in_bind_dout [0:1];
  wire reorder_x_in_bind_full_n [0:1];
  wire reorder_x_in_bind_write [0:1];
  wire reorder_x_in_bind_empty_n [0:1];
  wire reorder_x_in_bind_read [0:1];
  genvar reorder_x_in_bind_i;
  generate
    for (reorder_x_in_bind_i = 0; reorder_x_in_bind_i < 2; reorder_x_in_bind_i = reorder_x_in_bind_i + 1) begin : g_reorder_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(reorder_x_in_bind_din[reorder_x_in_bind_i]), .full_n(reorder_x_in_bind_full_n[reorder_x_in_bind_i]), .write(reorder_x_in_bind_write[reorder_x_in_bind_i]), .dout(reorder_x_in_bind_dout[reorder_x_in_bind_i]), .empty_n(reorder_x_in_bind_empty_n[reorder_x_in_bind_i]), .read(reorder_x_in_bind_read[reorder_x_in_bind_i]));
    end
  endgenerate
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] delay0_pid0_dout [0:1];
  wire delay0_pid0_empty_n [0:1];
  wire delay0_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_delay0_pid0_0 (.dout(delay0_pid0_dout[0]), .empty_n(delay0_pid0_empty_n[0]), .read(delay0_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_delay0_pid0_1 (.dout(delay0_pid0_dout[1]), .empty_n(delay0_pid0_empty_n[1]), .read(delay0_pid0_read[1]));
  // role delay0_r0: 2 instance(s)
  delay0_r0 u_delay0_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay0_x_in_bind_dout[0]),
      .x_in_empty_n(delay0_x_in_bind_empty_n[0]),
      .x_in_read(delay0_x_in_bind_read[0]),
      .x_out_din(delay1_x_in_bind_din[0]),
      .x_out_full_n(delay1_x_in_bind_full_n[0]),
      .x_out_write(delay1_x_in_bind_write[0]),
      ._pid0_dout(delay0_pid0_dout[0]),
      ._pid0_empty_n(delay0_pid0_empty_n[0]),
      ._pid0_read(delay0_pid0_read[0]));
  delay0_r0 u_delay0_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay0_x_in_bind_dout[1]),
      .x_in_empty_n(delay0_x_in_bind_empty_n[1]),
      .x_in_read(delay0_x_in_bind_read[1]),
      .x_out_din(delay1_x_in_bind_din[1]),
      .x_out_full_n(delay1_x_in_bind_full_n[1]),
      .x_out_write(delay1_x_in_bind_write[1]),
      ._pid0_dout(delay0_pid0_dout[1]),
      ._pid0_empty_n(delay0_pid0_empty_n[1]),
      ._pid0_read(delay0_pid0_read[1]));
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] delay1_pid0_dout [0:1];
  wire delay1_pid0_empty_n [0:1];
  wire delay1_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_delay1_pid0_0 (.dout(delay1_pid0_dout[0]), .empty_n(delay1_pid0_empty_n[0]), .read(delay1_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_delay1_pid0_1 (.dout(delay1_pid0_dout[1]), .empty_n(delay1_pid0_empty_n[1]), .read(delay1_pid0_read[1]));
  // role delay1_r0: 2 instance(s)
  delay1_r0 u_delay1_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay1_x_in_bind_dout[0]),
      .x_in_empty_n(delay1_x_in_bind_empty_n[0]),
      .x_in_read(delay1_x_in_bind_read[0]),
      .x_out_din(delay2_x_in_bind_din[0]),
      .x_out_full_n(delay2_x_in_bind_full_n[0]),
      .x_out_write(delay2_x_in_bind_write[0]),
      ._pid0_dout(delay1_pid0_dout[0]),
      ._pid0_empty_n(delay1_pid0_empty_n[0]),
      ._pid0_read(delay1_pid0_read[0]));
  delay1_r0 u_delay1_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay1_x_in_bind_dout[1]),
      .x_in_empty_n(delay1_x_in_bind_empty_n[1]),
      .x_in_read(delay1_x_in_bind_read[1]),
      .x_out_din(delay2_x_in_bind_din[1]),
      .x_out_full_n(delay2_x_in_bind_full_n[1]),
      .x_out_write(delay2_x_in_bind_write[1]),
      ._pid0_dout(delay1_pid0_dout[1]),
      ._pid0_empty_n(delay1_pid0_empty_n[1]),
      ._pid0_read(delay1_pid0_read[1]));
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] delay2_pid0_dout [0:1];
  wire delay2_pid0_empty_n [0:1];
  wire delay2_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_delay2_pid0_0 (.dout(delay2_pid0_dout[0]), .empty_n(delay2_pid0_empty_n[0]), .read(delay2_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_delay2_pid0_1 (.dout(delay2_pid0_dout[1]), .empty_n(delay2_pid0_empty_n[1]), .read(delay2_pid0_read[1]));
  // role delay2_r0: 2 instance(s)
  delay2_r0 u_delay2_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay2_x_in_bind_dout[0]),
      .x_in_empty_n(delay2_x_in_bind_empty_n[0]),
      .x_in_read(delay2_x_in_bind_read[0]),
      .x_out_din(delay3_x_in_bind_din[0]),
      .x_out_full_n(delay3_x_in_bind_full_n[0]),
      .x_out_write(delay3_x_in_bind_write[0]),
      ._pid0_dout(delay2_pid0_dout[0]),
      ._pid0_empty_n(delay2_pid0_empty_n[0]),
      ._pid0_read(delay2_pid0_read[0]));
  delay2_r0 u_delay2_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay2_x_in_bind_dout[1]),
      .x_in_empty_n(delay2_x_in_bind_empty_n[1]),
      .x_in_read(delay2_x_in_bind_read[1]),
      .x_out_din(delay3_x_in_bind_din[1]),
      .x_out_full_n(delay3_x_in_bind_full_n[1]),
      .x_out_write(delay3_x_in_bind_write[1]),
      ._pid0_dout(delay2_pid0_dout[1]),
      ._pid0_empty_n(delay2_pid0_empty_n[1]),
      ._pid0_read(delay2_pid0_read[1]));
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] delay3_pid0_dout [0:1];
  wire delay3_pid0_empty_n [0:1];
  wire delay3_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_delay3_pid0_0 (.dout(delay3_pid0_dout[0]), .empty_n(delay3_pid0_empty_n[0]), .read(delay3_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_delay3_pid0_1 (.dout(delay3_pid0_dout[1]), .empty_n(delay3_pid0_empty_n[1]), .read(delay3_pid0_read[1]));
  // role delay3_r0: 2 instance(s)
  delay3_r0 u_delay3_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay3_x_in_bind_dout[0]),
      .x_in_empty_n(delay3_x_in_bind_empty_n[0]),
      .x_in_read(delay3_x_in_bind_read[0]),
      .x_out_din(delay4_x_in_bind_din[0]),
      .x_out_full_n(delay4_x_in_bind_full_n[0]),
      .x_out_write(delay4_x_in_bind_write[0]),
      ._pid0_dout(delay3_pid0_dout[0]),
      ._pid0_empty_n(delay3_pid0_empty_n[0]),
      ._pid0_read(delay3_pid0_read[0]));
  delay3_r0 u_delay3_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay3_x_in_bind_dout[1]),
      .x_in_empty_n(delay3_x_in_bind_empty_n[1]),
      .x_in_read(delay3_x_in_bind_read[1]),
      .x_out_din(delay4_x_in_bind_din[1]),
      .x_out_full_n(delay4_x_in_bind_full_n[1]),
      .x_out_write(delay4_x_in_bind_write[1]),
      ._pid0_dout(delay3_pid0_dout[1]),
      ._pid0_empty_n(delay3_pid0_empty_n[1]),
      ._pid0_read(delay3_pid0_read[1]));
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] delay4_pid0_dout [0:1];
  wire delay4_pid0_empty_n [0:1];
  wire delay4_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_delay4_pid0_0 (.dout(delay4_pid0_dout[0]), .empty_n(delay4_pid0_empty_n[0]), .read(delay4_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_delay4_pid0_1 (.dout(delay4_pid0_dout[1]), .empty_n(delay4_pid0_empty_n[1]), .read(delay4_pid0_read[1]));
  // role delay4_r0: 2 instance(s)
  delay4_r0 u_delay4_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay4_x_in_bind_dout[0]),
      .x_in_empty_n(delay4_x_in_bind_empty_n[0]),
      .x_in_read(delay4_x_in_bind_read[0]),
      .x_out_din(delay5_x_in_bind_din[0]),
      .x_out_full_n(delay5_x_in_bind_full_n[0]),
      .x_out_write(delay5_x_in_bind_write[0]),
      ._pid0_dout(delay4_pid0_dout[0]),
      ._pid0_empty_n(delay4_pid0_empty_n[0]),
      ._pid0_read(delay4_pid0_read[0]));
  delay4_r0 u_delay4_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay4_x_in_bind_dout[1]),
      .x_in_empty_n(delay4_x_in_bind_empty_n[1]),
      .x_in_read(delay4_x_in_bind_read[1]),
      .x_out_din(delay5_x_in_bind_din[1]),
      .x_out_full_n(delay5_x_in_bind_full_n[1]),
      .x_out_write(delay5_x_in_bind_write[1]),
      ._pid0_dout(delay4_pid0_dout[1]),
      ._pid0_empty_n(delay4_pid0_empty_n[1]),
      ._pid0_read(delay4_pid0_read[1]));
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] delay5_pid0_dout [0:1];
  wire delay5_pid0_empty_n [0:1];
  wire delay5_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_delay5_pid0_0 (.dout(delay5_pid0_dout[0]), .empty_n(delay5_pid0_empty_n[0]), .read(delay5_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_delay5_pid0_1 (.dout(delay5_pid0_dout[1]), .empty_n(delay5_pid0_empty_n[1]), .read(delay5_pid0_read[1]));
  // role delay5_r0: 2 instance(s)
  delay5_r0 u_delay5_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay5_x_in_bind_dout[0]),
      .x_in_empty_n(delay5_x_in_bind_empty_n[0]),
      .x_in_read(delay5_x_in_bind_read[0]),
      .x_out_din(delay6_x_in_bind_din[0]),
      .x_out_full_n(delay6_x_in_bind_full_n[0]),
      .x_out_write(delay6_x_in_bind_write[0]),
      ._pid0_dout(delay5_pid0_dout[0]),
      ._pid0_empty_n(delay5_pid0_empty_n[0]),
      ._pid0_read(delay5_pid0_read[0]));
  delay5_r0 u_delay5_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay5_x_in_bind_dout[1]),
      .x_in_empty_n(delay5_x_in_bind_empty_n[1]),
      .x_in_read(delay5_x_in_bind_read[1]),
      .x_out_din(delay6_x_in_bind_din[1]),
      .x_out_full_n(delay6_x_in_bind_full_n[1]),
      .x_out_write(delay6_x_in_bind_write[1]),
      ._pid0_dout(delay5_pid0_dout[1]),
      ._pid0_empty_n(delay5_pid0_empty_n[1]),
      ._pid0_read(delay5_pid0_read[1]));
  // coordinate axis 0: 2 constant source(s)
  wire [31:0] delay6_pid0_dout [0:1];
  wire delay6_pid0_empty_n [0:1];
  wire delay6_pid0_read [0:1];
  spmw_const #(.DW(32), .VAL(0)) u_delay6_pid0_0 (.dout(delay6_pid0_dout[0]), .empty_n(delay6_pid0_empty_n[0]), .read(delay6_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_delay6_pid0_1 (.dout(delay6_pid0_dout[1]), .empty_n(delay6_pid0_empty_n[1]), .read(delay6_pid0_read[1]));
  // role delay6_r0: 2 instance(s)
  delay6_r0 u_delay6_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay6_x_in_bind_dout[0]),
      .x_in_empty_n(delay6_x_in_bind_empty_n[0]),
      .x_in_read(delay6_x_in_bind_read[0]),
      .x_out_din(cross_a_in_bind_din[0]),
      .x_out_full_n(cross_a_in_bind_full_n[0]),
      .x_out_write(cross_a_in_bind_write[0]),
      ._pid0_dout(delay6_pid0_dout[0]),
      ._pid0_empty_n(delay6_pid0_empty_n[0]),
      ._pid0_read(delay6_pid0_read[0]));
  delay6_r0 u_delay6_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay6_x_in_bind_dout[1]),
      .x_in_empty_n(delay6_x_in_bind_empty_n[1]),
      .x_in_read(delay6_x_in_bind_read[1]),
      .x_out_din(cross_a_in_bind_din[1]),
      .x_out_full_n(cross_a_in_bind_full_n[1]),
      .x_out_write(cross_a_in_bind_write[1]),
      ._pid0_dout(delay6_pid0_dout[1]),
      ._pid0_empty_n(delay6_pid0_empty_n[1]),
      ._pid0_read(delay6_pid0_read[1]));
  // coordinate axis 0: 4 constant source(s)
  wire [31:0] cross_pid0_dout [0:3];
  wire cross_pid0_empty_n [0:3];
  wire cross_pid0_read [0:3];
  spmw_const #(.DW(32), .VAL(0)) u_cross_pid0_0 (.dout(cross_pid0_dout[0]), .empty_n(cross_pid0_empty_n[0]), .read(cross_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(0)) u_cross_pid0_1 (.dout(cross_pid0_dout[1]), .empty_n(cross_pid0_empty_n[1]), .read(cross_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(1)) u_cross_pid0_2 (.dout(cross_pid0_dout[2]), .empty_n(cross_pid0_empty_n[2]), .read(cross_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(1)) u_cross_pid0_3 (.dout(cross_pid0_dout[3]), .empty_n(cross_pid0_empty_n[3]), .read(cross_pid0_read[3]));
  // coordinate axis 1: 4 constant source(s)
  wire [31:0] cross_pid1_dout [0:3];
  wire cross_pid1_empty_n [0:3];
  wire cross_pid1_read [0:3];
  spmw_const #(.DW(32), .VAL(0)) u_cross_pid1_0 (.dout(cross_pid1_dout[0]), .empty_n(cross_pid1_empty_n[0]), .read(cross_pid1_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_cross_pid1_1 (.dout(cross_pid1_dout[1]), .empty_n(cross_pid1_empty_n[1]), .read(cross_pid1_read[1]));
  spmw_const #(.DW(32), .VAL(0)) u_cross_pid1_2 (.dout(cross_pid1_dout[2]), .empty_n(cross_pid1_empty_n[2]), .read(cross_pid1_read[2]));
  spmw_const #(.DW(32), .VAL(1)) u_cross_pid1_3 (.dout(cross_pid1_dout[3]), .empty_n(cross_pid1_empty_n[3]), .read(cross_pid1_read[3]));
  // role cross_r0: 2 instance(s)
  cross_r0 u_cross_r0_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(cross_a_out_a_in_dout[2]),
      .a_in_empty_n(cross_a_out_a_in_empty_n[2]),
      .a_in_read(cross_a_out_a_in_read[2]),
      .a_out_din(reorder_x_in_bind_din[0]),
      .a_out_full_n(reorder_x_in_bind_full_n[0]),
      .a_out_write(reorder_x_in_bind_write[0]),
      .b_in_dout(cross_b_out_b_in_dout[1]),
      .b_in_empty_n(cross_b_out_b_in_empty_n[1]),
      .b_in_read(cross_b_out_b_in_read[1]),
      ._pid0_dout(cross_pid0_dout[2]),
      ._pid0_empty_n(cross_pid0_empty_n[2]),
      ._pid0_read(cross_pid0_read[2]),
      ._pid1_dout(cross_pid1_dout[2]),
      ._pid1_empty_n(cross_pid1_empty_n[2]),
      ._pid1_read(cross_pid1_read[2]));
  cross_r0 u_cross_r0_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(cross_a_out_a_in_dout[3]),
      .a_in_empty_n(cross_a_out_a_in_empty_n[3]),
      .a_in_read(cross_a_out_a_in_read[3]),
      .a_out_din(reorder_x_in_bind_din[1]),
      .a_out_full_n(reorder_x_in_bind_full_n[1]),
      .a_out_write(reorder_x_in_bind_write[1]),
      .b_in_dout(cross_b_out_b_in_dout[0]),
      .b_in_empty_n(cross_b_out_b_in_empty_n[0]),
      .b_in_read(cross_b_out_b_in_read[0]),
      ._pid0_dout(cross_pid0_dout[3]),
      ._pid0_empty_n(cross_pid0_empty_n[3]),
      ._pid0_read(cross_pid0_read[3]),
      ._pid1_dout(cross_pid1_dout[3]),
      ._pid1_empty_n(cross_pid1_empty_n[3]),
      ._pid1_read(cross_pid1_read[3]));
  // role cross_r1: 2 instance(s)
  cross_r1 u_cross_r1_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(cross_a_in_bind_dout[0]),
      .a_in_empty_n(cross_a_in_bind_empty_n[0]),
      .a_in_read(cross_a_in_bind_read[0]),
      .a_out_din(cross_a_out_a_in_din[2]),
      .a_out_full_n(cross_a_out_a_in_full_n[2]),
      .a_out_write(cross_a_out_a_in_write[2]),
      .b_out_din(cross_b_out_b_in_din[0]),
      .b_out_full_n(cross_b_out_b_in_full_n[0]),
      .b_out_write(cross_b_out_b_in_write[0]));
  cross_r1 u_cross_r1_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(cross_a_in_bind_dout[1]),
      .a_in_empty_n(cross_a_in_bind_empty_n[1]),
      .a_in_read(cross_a_in_bind_read[1]),
      .a_out_din(cross_a_out_a_in_din[3]),
      .a_out_full_n(cross_a_out_a_in_full_n[3]),
      .a_out_write(cross_a_out_a_in_write[3]),
      .b_out_din(cross_b_out_b_in_din[1]),
      .b_out_full_n(cross_b_out_b_in_full_n[1]),
      .b_out_write(cross_b_out_b_in_write[1]));
  // role reorder_r0: 2 instance(s)
  reorder_r0 u_reorder_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(reorder_x_in_bind_dout[0]),
      .x_in_empty_n(reorder_x_in_bind_empty_n[0]),
      .x_in_read(reorder_x_in_bind_read[0]),
      .y_out_din(reorder_y_out_bind_din[0]),
      .y_out_full_n(reorder_y_out_bind_full_n[0]),
      .y_out_write(reorder_y_out_bind_write[0]));
  reorder_r0 u_reorder_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(reorder_x_in_bind_dout[1]),
      .x_in_empty_n(reorder_x_in_bind_empty_n[1]),
      .x_in_read(reorder_x_in_bind_read[1]),
      .y_out_din(reorder_y_out_bind_din[1]),
      .y_out_full_n(reorder_y_out_bind_full_n[1]),
      .y_out_write(reorder_y_out_bind_write[1]));
endmodule
