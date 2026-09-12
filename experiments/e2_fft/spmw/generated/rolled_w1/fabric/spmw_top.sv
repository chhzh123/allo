`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] delay0_x_in_bind_dout [0:0],
  input  wire delay0_x_in_bind_empty_n [0:0],
  output wire delay0_x_in_bind_read [0:0],
  output wire [63:0] reorder_y_out_bind_din [0:0],
  output wire reorder_y_out_bind_write [0:0],
  input  wire reorder_y_out_bind_full_n [0:0]
);
  // family delay1_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] delay1_x_in_bind_din [0:0];
  wire [63:0] delay1_x_in_bind_dout [0:0];
  wire delay1_x_in_bind_full_n [0:0];
  wire delay1_x_in_bind_write [0:0];
  wire delay1_x_in_bind_empty_n [0:0];
  wire delay1_x_in_bind_read [0:0];
  genvar delay1_x_in_bind_i;
  generate
    for (delay1_x_in_bind_i = 0; delay1_x_in_bind_i < 1; delay1_x_in_bind_i = delay1_x_in_bind_i + 1) begin : g_delay1_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(delay1_x_in_bind_din[delay1_x_in_bind_i]), .full_n(delay1_x_in_bind_full_n[delay1_x_in_bind_i]), .write(delay1_x_in_bind_write[delay1_x_in_bind_i]), .dout(delay1_x_in_bind_dout[delay1_x_in_bind_i]), .empty_n(delay1_x_in_bind_empty_n[delay1_x_in_bind_i]), .read(delay1_x_in_bind_read[delay1_x_in_bind_i]));
    end
  endgenerate
  // family delay2_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] delay2_x_in_bind_din [0:0];
  wire [63:0] delay2_x_in_bind_dout [0:0];
  wire delay2_x_in_bind_full_n [0:0];
  wire delay2_x_in_bind_write [0:0];
  wire delay2_x_in_bind_empty_n [0:0];
  wire delay2_x_in_bind_read [0:0];
  genvar delay2_x_in_bind_i;
  generate
    for (delay2_x_in_bind_i = 0; delay2_x_in_bind_i < 1; delay2_x_in_bind_i = delay2_x_in_bind_i + 1) begin : g_delay2_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(delay2_x_in_bind_din[delay2_x_in_bind_i]), .full_n(delay2_x_in_bind_full_n[delay2_x_in_bind_i]), .write(delay2_x_in_bind_write[delay2_x_in_bind_i]), .dout(delay2_x_in_bind_dout[delay2_x_in_bind_i]), .empty_n(delay2_x_in_bind_empty_n[delay2_x_in_bind_i]), .read(delay2_x_in_bind_read[delay2_x_in_bind_i]));
    end
  endgenerate
  // family delay3_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] delay3_x_in_bind_din [0:0];
  wire [63:0] delay3_x_in_bind_dout [0:0];
  wire delay3_x_in_bind_full_n [0:0];
  wire delay3_x_in_bind_write [0:0];
  wire delay3_x_in_bind_empty_n [0:0];
  wire delay3_x_in_bind_read [0:0];
  genvar delay3_x_in_bind_i;
  generate
    for (delay3_x_in_bind_i = 0; delay3_x_in_bind_i < 1; delay3_x_in_bind_i = delay3_x_in_bind_i + 1) begin : g_delay3_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(delay3_x_in_bind_din[delay3_x_in_bind_i]), .full_n(delay3_x_in_bind_full_n[delay3_x_in_bind_i]), .write(delay3_x_in_bind_write[delay3_x_in_bind_i]), .dout(delay3_x_in_bind_dout[delay3_x_in_bind_i]), .empty_n(delay3_x_in_bind_empty_n[delay3_x_in_bind_i]), .read(delay3_x_in_bind_read[delay3_x_in_bind_i]));
    end
  endgenerate
  // family delay4_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] delay4_x_in_bind_din [0:0];
  wire [63:0] delay4_x_in_bind_dout [0:0];
  wire delay4_x_in_bind_full_n [0:0];
  wire delay4_x_in_bind_write [0:0];
  wire delay4_x_in_bind_empty_n [0:0];
  wire delay4_x_in_bind_read [0:0];
  genvar delay4_x_in_bind_i;
  generate
    for (delay4_x_in_bind_i = 0; delay4_x_in_bind_i < 1; delay4_x_in_bind_i = delay4_x_in_bind_i + 1) begin : g_delay4_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(delay4_x_in_bind_din[delay4_x_in_bind_i]), .full_n(delay4_x_in_bind_full_n[delay4_x_in_bind_i]), .write(delay4_x_in_bind_write[delay4_x_in_bind_i]), .dout(delay4_x_in_bind_dout[delay4_x_in_bind_i]), .empty_n(delay4_x_in_bind_empty_n[delay4_x_in_bind_i]), .read(delay4_x_in_bind_read[delay4_x_in_bind_i]));
    end
  endgenerate
  // family delay5_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] delay5_x_in_bind_din [0:0];
  wire [63:0] delay5_x_in_bind_dout [0:0];
  wire delay5_x_in_bind_full_n [0:0];
  wire delay5_x_in_bind_write [0:0];
  wire delay5_x_in_bind_empty_n [0:0];
  wire delay5_x_in_bind_read [0:0];
  genvar delay5_x_in_bind_i;
  generate
    for (delay5_x_in_bind_i = 0; delay5_x_in_bind_i < 1; delay5_x_in_bind_i = delay5_x_in_bind_i + 1) begin : g_delay5_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(delay5_x_in_bind_din[delay5_x_in_bind_i]), .full_n(delay5_x_in_bind_full_n[delay5_x_in_bind_i]), .write(delay5_x_in_bind_write[delay5_x_in_bind_i]), .dout(delay5_x_in_bind_dout[delay5_x_in_bind_i]), .empty_n(delay5_x_in_bind_empty_n[delay5_x_in_bind_i]), .read(delay5_x_in_bind_read[delay5_x_in_bind_i]));
    end
  endgenerate
  // family delay6_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] delay6_x_in_bind_din [0:0];
  wire [63:0] delay6_x_in_bind_dout [0:0];
  wire delay6_x_in_bind_full_n [0:0];
  wire delay6_x_in_bind_write [0:0];
  wire delay6_x_in_bind_empty_n [0:0];
  wire delay6_x_in_bind_read [0:0];
  genvar delay6_x_in_bind_i;
  generate
    for (delay6_x_in_bind_i = 0; delay6_x_in_bind_i < 1; delay6_x_in_bind_i = delay6_x_in_bind_i + 1) begin : g_delay6_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(delay6_x_in_bind_din[delay6_x_in_bind_i]), .full_n(delay6_x_in_bind_full_n[delay6_x_in_bind_i]), .write(delay6_x_in_bind_write[delay6_x_in_bind_i]), .dout(delay6_x_in_bind_dout[delay6_x_in_bind_i]), .empty_n(delay6_x_in_bind_empty_n[delay6_x_in_bind_i]), .read(delay6_x_in_bind_read[delay6_x_in_bind_i]));
    end
  endgenerate
  // family delay7_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] delay7_x_in_bind_din [0:0];
  wire [63:0] delay7_x_in_bind_dout [0:0];
  wire delay7_x_in_bind_full_n [0:0];
  wire delay7_x_in_bind_write [0:0];
  wire delay7_x_in_bind_empty_n [0:0];
  wire delay7_x_in_bind_read [0:0];
  genvar delay7_x_in_bind_i;
  generate
    for (delay7_x_in_bind_i = 0; delay7_x_in_bind_i < 1; delay7_x_in_bind_i = delay7_x_in_bind_i + 1) begin : g_delay7_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(delay7_x_in_bind_din[delay7_x_in_bind_i]), .full_n(delay7_x_in_bind_full_n[delay7_x_in_bind_i]), .write(delay7_x_in_bind_write[delay7_x_in_bind_i]), .dout(delay7_x_in_bind_dout[delay7_x_in_bind_i]), .empty_n(delay7_x_in_bind_empty_n[delay7_x_in_bind_i]), .read(delay7_x_in_bind_read[delay7_x_in_bind_i]));
    end
  endgenerate
  // family reorder_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] reorder_x_in_bind_din [0:0];
  wire [63:0] reorder_x_in_bind_dout [0:0];
  wire reorder_x_in_bind_full_n [0:0];
  wire reorder_x_in_bind_write [0:0];
  wire reorder_x_in_bind_empty_n [0:0];
  wire reorder_x_in_bind_read [0:0];
  genvar reorder_x_in_bind_i;
  generate
    for (reorder_x_in_bind_i = 0; reorder_x_in_bind_i < 1; reorder_x_in_bind_i = reorder_x_in_bind_i + 1) begin : g_reorder_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(reorder_x_in_bind_din[reorder_x_in_bind_i]), .full_n(reorder_x_in_bind_full_n[reorder_x_in_bind_i]), .write(reorder_x_in_bind_write[reorder_x_in_bind_i]), .dout(reorder_x_in_bind_dout[reorder_x_in_bind_i]), .empty_n(reorder_x_in_bind_empty_n[reorder_x_in_bind_i]), .read(reorder_x_in_bind_read[reorder_x_in_bind_i]));
    end
  endgenerate
  // coordinate axis 0: 1 constant source(s)
  wire [31:0] delay0_pid0_dout [0:0];
  wire delay0_pid0_empty_n [0:0];
  wire delay0_pid0_read [0:0];
  spmw_const #(.DW(32), .VAL(0)) u_delay0_pid0_0 (.dout(delay0_pid0_dout[0]), .empty_n(delay0_pid0_empty_n[0]), .read(delay0_pid0_read[0]));
  // role delay0_r0: 1 instance(s)
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
  // coordinate axis 0: 1 constant source(s)
  wire [31:0] delay1_pid0_dout [0:0];
  wire delay1_pid0_empty_n [0:0];
  wire delay1_pid0_read [0:0];
  spmw_const #(.DW(32), .VAL(0)) u_delay1_pid0_0 (.dout(delay1_pid0_dout[0]), .empty_n(delay1_pid0_empty_n[0]), .read(delay1_pid0_read[0]));
  // role delay1_r0: 1 instance(s)
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
  // coordinate axis 0: 1 constant source(s)
  wire [31:0] delay2_pid0_dout [0:0];
  wire delay2_pid0_empty_n [0:0];
  wire delay2_pid0_read [0:0];
  spmw_const #(.DW(32), .VAL(0)) u_delay2_pid0_0 (.dout(delay2_pid0_dout[0]), .empty_n(delay2_pid0_empty_n[0]), .read(delay2_pid0_read[0]));
  // role delay2_r0: 1 instance(s)
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
  // coordinate axis 0: 1 constant source(s)
  wire [31:0] delay3_pid0_dout [0:0];
  wire delay3_pid0_empty_n [0:0];
  wire delay3_pid0_read [0:0];
  spmw_const #(.DW(32), .VAL(0)) u_delay3_pid0_0 (.dout(delay3_pid0_dout[0]), .empty_n(delay3_pid0_empty_n[0]), .read(delay3_pid0_read[0]));
  // role delay3_r0: 1 instance(s)
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
  // coordinate axis 0: 1 constant source(s)
  wire [31:0] delay4_pid0_dout [0:0];
  wire delay4_pid0_empty_n [0:0];
  wire delay4_pid0_read [0:0];
  spmw_const #(.DW(32), .VAL(0)) u_delay4_pid0_0 (.dout(delay4_pid0_dout[0]), .empty_n(delay4_pid0_empty_n[0]), .read(delay4_pid0_read[0]));
  // role delay4_r0: 1 instance(s)
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
  // coordinate axis 0: 1 constant source(s)
  wire [31:0] delay5_pid0_dout [0:0];
  wire delay5_pid0_empty_n [0:0];
  wire delay5_pid0_read [0:0];
  spmw_const #(.DW(32), .VAL(0)) u_delay5_pid0_0 (.dout(delay5_pid0_dout[0]), .empty_n(delay5_pid0_empty_n[0]), .read(delay5_pid0_read[0]));
  // role delay5_r0: 1 instance(s)
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
  // coordinate axis 0: 1 constant source(s)
  wire [31:0] delay6_pid0_dout [0:0];
  wire delay6_pid0_empty_n [0:0];
  wire delay6_pid0_read [0:0];
  spmw_const #(.DW(32), .VAL(0)) u_delay6_pid0_0 (.dout(delay6_pid0_dout[0]), .empty_n(delay6_pid0_empty_n[0]), .read(delay6_pid0_read[0]));
  // role delay6_r0: 1 instance(s)
  delay6_r0 u_delay6_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay6_x_in_bind_dout[0]),
      .x_in_empty_n(delay6_x_in_bind_empty_n[0]),
      .x_in_read(delay6_x_in_bind_read[0]),
      .x_out_din(delay7_x_in_bind_din[0]),
      .x_out_full_n(delay7_x_in_bind_full_n[0]),
      .x_out_write(delay7_x_in_bind_write[0]),
      ._pid0_dout(delay6_pid0_dout[0]),
      ._pid0_empty_n(delay6_pid0_empty_n[0]),
      ._pid0_read(delay6_pid0_read[0]));
  // coordinate axis 0: 1 constant source(s)
  wire [31:0] delay7_pid0_dout [0:0];
  wire delay7_pid0_empty_n [0:0];
  wire delay7_pid0_read [0:0];
  spmw_const #(.DW(32), .VAL(0)) u_delay7_pid0_0 (.dout(delay7_pid0_dout[0]), .empty_n(delay7_pid0_empty_n[0]), .read(delay7_pid0_read[0]));
  // role delay7_r0: 1 instance(s)
  delay7_r0 u_delay7_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(delay7_x_in_bind_dout[0]),
      .x_in_empty_n(delay7_x_in_bind_empty_n[0]),
      .x_in_read(delay7_x_in_bind_read[0]),
      .x_out_din(reorder_x_in_bind_din[0]),
      .x_out_full_n(reorder_x_in_bind_full_n[0]),
      .x_out_write(reorder_x_in_bind_write[0]),
      ._pid0_dout(delay7_pid0_dout[0]),
      ._pid0_empty_n(delay7_pid0_empty_n[0]),
      ._pid0_read(delay7_pid0_read[0]));
  // role reorder_r0: 1 instance(s)
  reorder_r0 u_reorder_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(reorder_x_in_bind_dout[0]),
      .x_in_empty_n(reorder_x_in_bind_empty_n[0]),
      .x_in_read(reorder_x_in_bind_read[0]),
      .y_out_din(reorder_y_out_bind_din[0]),
      .y_out_full_n(reorder_y_out_bind_full_n[0]),
      .y_out_write(reorder_y_out_bind_write[0]));
endmodule
