`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] stage0_x_in_bind_dout [0:0],
  input  wire stage0_x_in_bind_empty_n [0:0],
  output wire stage0_x_in_bind_read [0:0],
  output wire [63:0] reorder_y_out_bind_din [0:0],
  output wire reorder_y_out_bind_write [0:0],
  input  wire reorder_y_out_bind_full_n [0:0]
);
  // family stage1_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] stage1_x_in_bind_din [0:0];
  wire [63:0] stage1_x_in_bind_dout [0:0];
  wire stage1_x_in_bind_full_n [0:0];
  wire stage1_x_in_bind_write [0:0];
  wire stage1_x_in_bind_empty_n [0:0];
  wire stage1_x_in_bind_read [0:0];
  genvar stage1_x_in_bind_i;
  generate
    for (stage1_x_in_bind_i = 0; stage1_x_in_bind_i < 1; stage1_x_in_bind_i = stage1_x_in_bind_i + 1) begin : g_stage1_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(stage1_x_in_bind_din[stage1_x_in_bind_i]), .full_n(stage1_x_in_bind_full_n[stage1_x_in_bind_i]), .write(stage1_x_in_bind_write[stage1_x_in_bind_i]), .dout(stage1_x_in_bind_dout[stage1_x_in_bind_i]), .empty_n(stage1_x_in_bind_empty_n[stage1_x_in_bind_i]), .read(stage1_x_in_bind_read[stage1_x_in_bind_i]));
    end
  endgenerate
  // family stage2_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] stage2_x_in_bind_din [0:0];
  wire [63:0] stage2_x_in_bind_dout [0:0];
  wire stage2_x_in_bind_full_n [0:0];
  wire stage2_x_in_bind_write [0:0];
  wire stage2_x_in_bind_empty_n [0:0];
  wire stage2_x_in_bind_read [0:0];
  genvar stage2_x_in_bind_i;
  generate
    for (stage2_x_in_bind_i = 0; stage2_x_in_bind_i < 1; stage2_x_in_bind_i = stage2_x_in_bind_i + 1) begin : g_stage2_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(stage2_x_in_bind_din[stage2_x_in_bind_i]), .full_n(stage2_x_in_bind_full_n[stage2_x_in_bind_i]), .write(stage2_x_in_bind_write[stage2_x_in_bind_i]), .dout(stage2_x_in_bind_dout[stage2_x_in_bind_i]), .empty_n(stage2_x_in_bind_empty_n[stage2_x_in_bind_i]), .read(stage2_x_in_bind_read[stage2_x_in_bind_i]));
    end
  endgenerate
  // family stage3_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] stage3_x_in_bind_din [0:0];
  wire [63:0] stage3_x_in_bind_dout [0:0];
  wire stage3_x_in_bind_full_n [0:0];
  wire stage3_x_in_bind_write [0:0];
  wire stage3_x_in_bind_empty_n [0:0];
  wire stage3_x_in_bind_read [0:0];
  genvar stage3_x_in_bind_i;
  generate
    for (stage3_x_in_bind_i = 0; stage3_x_in_bind_i < 1; stage3_x_in_bind_i = stage3_x_in_bind_i + 1) begin : g_stage3_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(stage3_x_in_bind_din[stage3_x_in_bind_i]), .full_n(stage3_x_in_bind_full_n[stage3_x_in_bind_i]), .write(stage3_x_in_bind_write[stage3_x_in_bind_i]), .dout(stage3_x_in_bind_dout[stage3_x_in_bind_i]), .empty_n(stage3_x_in_bind_empty_n[stage3_x_in_bind_i]), .read(stage3_x_in_bind_read[stage3_x_in_bind_i]));
    end
  endgenerate
  // family stage4_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] stage4_x_in_bind_din [0:0];
  wire [63:0] stage4_x_in_bind_dout [0:0];
  wire stage4_x_in_bind_full_n [0:0];
  wire stage4_x_in_bind_write [0:0];
  wire stage4_x_in_bind_empty_n [0:0];
  wire stage4_x_in_bind_read [0:0];
  genvar stage4_x_in_bind_i;
  generate
    for (stage4_x_in_bind_i = 0; stage4_x_in_bind_i < 1; stage4_x_in_bind_i = stage4_x_in_bind_i + 1) begin : g_stage4_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(stage4_x_in_bind_din[stage4_x_in_bind_i]), .full_n(stage4_x_in_bind_full_n[stage4_x_in_bind_i]), .write(stage4_x_in_bind_write[stage4_x_in_bind_i]), .dout(stage4_x_in_bind_dout[stage4_x_in_bind_i]), .empty_n(stage4_x_in_bind_empty_n[stage4_x_in_bind_i]), .read(stage4_x_in_bind_read[stage4_x_in_bind_i]));
    end
  endgenerate
  // family stage5_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] stage5_x_in_bind_din [0:0];
  wire [63:0] stage5_x_in_bind_dout [0:0];
  wire stage5_x_in_bind_full_n [0:0];
  wire stage5_x_in_bind_write [0:0];
  wire stage5_x_in_bind_empty_n [0:0];
  wire stage5_x_in_bind_read [0:0];
  genvar stage5_x_in_bind_i;
  generate
    for (stage5_x_in_bind_i = 0; stage5_x_in_bind_i < 1; stage5_x_in_bind_i = stage5_x_in_bind_i + 1) begin : g_stage5_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(stage5_x_in_bind_din[stage5_x_in_bind_i]), .full_n(stage5_x_in_bind_full_n[stage5_x_in_bind_i]), .write(stage5_x_in_bind_write[stage5_x_in_bind_i]), .dout(stage5_x_in_bind_dout[stage5_x_in_bind_i]), .empty_n(stage5_x_in_bind_empty_n[stage5_x_in_bind_i]), .read(stage5_x_in_bind_read[stage5_x_in_bind_i]));
    end
  endgenerate
  // family stage6_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] stage6_x_in_bind_din [0:0];
  wire [63:0] stage6_x_in_bind_dout [0:0];
  wire stage6_x_in_bind_full_n [0:0];
  wire stage6_x_in_bind_write [0:0];
  wire stage6_x_in_bind_empty_n [0:0];
  wire stage6_x_in_bind_read [0:0];
  genvar stage6_x_in_bind_i;
  generate
    for (stage6_x_in_bind_i = 0; stage6_x_in_bind_i < 1; stage6_x_in_bind_i = stage6_x_in_bind_i + 1) begin : g_stage6_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(stage6_x_in_bind_din[stage6_x_in_bind_i]), .full_n(stage6_x_in_bind_full_n[stage6_x_in_bind_i]), .write(stage6_x_in_bind_write[stage6_x_in_bind_i]), .dout(stage6_x_in_bind_dout[stage6_x_in_bind_i]), .empty_n(stage6_x_in_bind_empty_n[stage6_x_in_bind_i]), .read(stage6_x_in_bind_read[stage6_x_in_bind_i]));
    end
  endgenerate
  // family stage7_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] stage7_x_in_bind_din [0:0];
  wire [63:0] stage7_x_in_bind_dout [0:0];
  wire stage7_x_in_bind_full_n [0:0];
  wire stage7_x_in_bind_write [0:0];
  wire stage7_x_in_bind_empty_n [0:0];
  wire stage7_x_in_bind_read [0:0];
  genvar stage7_x_in_bind_i;
  generate
    for (stage7_x_in_bind_i = 0; stage7_x_in_bind_i < 1; stage7_x_in_bind_i = stage7_x_in_bind_i + 1) begin : g_stage7_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(stage7_x_in_bind_din[stage7_x_in_bind_i]), .full_n(stage7_x_in_bind_full_n[stage7_x_in_bind_i]), .write(stage7_x_in_bind_write[stage7_x_in_bind_i]), .dout(stage7_x_in_bind_dout[stage7_x_in_bind_i]), .empty_n(stage7_x_in_bind_empty_n[stage7_x_in_bind_i]), .read(stage7_x_in_bind_read[stage7_x_in_bind_i]));
    end
  endgenerate
  // family stage8_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] stage8_x_in_bind_din [0:0];
  wire [63:0] stage8_x_in_bind_dout [0:0];
  wire stage8_x_in_bind_full_n [0:0];
  wire stage8_x_in_bind_write [0:0];
  wire stage8_x_in_bind_empty_n [0:0];
  wire stage8_x_in_bind_read [0:0];
  genvar stage8_x_in_bind_i;
  generate
    for (stage8_x_in_bind_i = 0; stage8_x_in_bind_i < 1; stage8_x_in_bind_i = stage8_x_in_bind_i + 1) begin : g_stage8_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(stage8_x_in_bind_din[stage8_x_in_bind_i]), .full_n(stage8_x_in_bind_full_n[stage8_x_in_bind_i]), .write(stage8_x_in_bind_write[stage8_x_in_bind_i]), .dout(stage8_x_in_bind_dout[stage8_x_in_bind_i]), .empty_n(stage8_x_in_bind_empty_n[stage8_x_in_bind_i]), .read(stage8_x_in_bind_read[stage8_x_in_bind_i]));
    end
  endgenerate
  // family stage9_x_in_bind: 1 channel(s), 64-bit, depth 2
  wire [63:0] stage9_x_in_bind_din [0:0];
  wire [63:0] stage9_x_in_bind_dout [0:0];
  wire stage9_x_in_bind_full_n [0:0];
  wire stage9_x_in_bind_write [0:0];
  wire stage9_x_in_bind_empty_n [0:0];
  wire stage9_x_in_bind_read [0:0];
  genvar stage9_x_in_bind_i;
  generate
    for (stage9_x_in_bind_i = 0; stage9_x_in_bind_i < 1; stage9_x_in_bind_i = stage9_x_in_bind_i + 1) begin : g_stage9_x_in_bind
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(stage9_x_in_bind_din[stage9_x_in_bind_i]), .full_n(stage9_x_in_bind_full_n[stage9_x_in_bind_i]), .write(stage9_x_in_bind_write[stage9_x_in_bind_i]), .dout(stage9_x_in_bind_dout[stage9_x_in_bind_i]), .empty_n(stage9_x_in_bind_empty_n[stage9_x_in_bind_i]), .read(stage9_x_in_bind_read[stage9_x_in_bind_i]));
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
  // role stage0_r0: 1 instance(s)
  stage0_r0 u_stage0_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(stage0_x_in_bind_dout[0]),
      .x_in_empty_n(stage0_x_in_bind_empty_n[0]),
      .x_in_read(stage0_x_in_bind_read[0]),
      .x_out_din(stage1_x_in_bind_din[0]),
      .x_out_full_n(stage1_x_in_bind_full_n[0]),
      .x_out_write(stage1_x_in_bind_write[0]));
  // role stage1_r0: 1 instance(s)
  stage1_r0 u_stage1_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(stage1_x_in_bind_dout[0]),
      .x_in_empty_n(stage1_x_in_bind_empty_n[0]),
      .x_in_read(stage1_x_in_bind_read[0]),
      .x_out_din(stage2_x_in_bind_din[0]),
      .x_out_full_n(stage2_x_in_bind_full_n[0]),
      .x_out_write(stage2_x_in_bind_write[0]));
  // role stage2_r0: 1 instance(s)
  stage2_r0 u_stage2_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(stage2_x_in_bind_dout[0]),
      .x_in_empty_n(stage2_x_in_bind_empty_n[0]),
      .x_in_read(stage2_x_in_bind_read[0]),
      .x_out_din(stage3_x_in_bind_din[0]),
      .x_out_full_n(stage3_x_in_bind_full_n[0]),
      .x_out_write(stage3_x_in_bind_write[0]));
  // role stage3_r0: 1 instance(s)
  stage3_r0 u_stage3_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(stage3_x_in_bind_dout[0]),
      .x_in_empty_n(stage3_x_in_bind_empty_n[0]),
      .x_in_read(stage3_x_in_bind_read[0]),
      .x_out_din(stage4_x_in_bind_din[0]),
      .x_out_full_n(stage4_x_in_bind_full_n[0]),
      .x_out_write(stage4_x_in_bind_write[0]));
  // role stage4_r0: 1 instance(s)
  stage4_r0 u_stage4_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(stage4_x_in_bind_dout[0]),
      .x_in_empty_n(stage4_x_in_bind_empty_n[0]),
      .x_in_read(stage4_x_in_bind_read[0]),
      .x_out_din(stage5_x_in_bind_din[0]),
      .x_out_full_n(stage5_x_in_bind_full_n[0]),
      .x_out_write(stage5_x_in_bind_write[0]));
  // role stage5_r0: 1 instance(s)
  stage5_r0 u_stage5_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(stage5_x_in_bind_dout[0]),
      .x_in_empty_n(stage5_x_in_bind_empty_n[0]),
      .x_in_read(stage5_x_in_bind_read[0]),
      .x_out_din(stage6_x_in_bind_din[0]),
      .x_out_full_n(stage6_x_in_bind_full_n[0]),
      .x_out_write(stage6_x_in_bind_write[0]));
  // role stage6_r0: 1 instance(s)
  stage6_r0 u_stage6_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(stage6_x_in_bind_dout[0]),
      .x_in_empty_n(stage6_x_in_bind_empty_n[0]),
      .x_in_read(stage6_x_in_bind_read[0]),
      .x_out_din(stage7_x_in_bind_din[0]),
      .x_out_full_n(stage7_x_in_bind_full_n[0]),
      .x_out_write(stage7_x_in_bind_write[0]));
  // role stage7_r0: 1 instance(s)
  stage7_r0 u_stage7_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(stage7_x_in_bind_dout[0]),
      .x_in_empty_n(stage7_x_in_bind_empty_n[0]),
      .x_in_read(stage7_x_in_bind_read[0]),
      .x_out_din(stage8_x_in_bind_din[0]),
      .x_out_full_n(stage8_x_in_bind_full_n[0]),
      .x_out_write(stage8_x_in_bind_write[0]));
  // role stage8_r0: 1 instance(s)
  stage8_r0 u_stage8_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(stage8_x_in_bind_dout[0]),
      .x_in_empty_n(stage8_x_in_bind_empty_n[0]),
      .x_in_read(stage8_x_in_bind_read[0]),
      .x_out_din(stage9_x_in_bind_din[0]),
      .x_out_full_n(stage9_x_in_bind_full_n[0]),
      .x_out_write(stage9_x_in_bind_write[0]));
  // role stage9_r0: 1 instance(s)
  stage9_r0 u_stage9_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .x_in_dout(stage9_x_in_bind_dout[0]),
      .x_in_empty_n(stage9_x_in_bind_empty_n[0]),
      .x_in_read(stage9_x_in_bind_read[0]),
      .x_out_din(reorder_x_in_bind_din[0]),
      .x_out_full_n(reorder_x_in_bind_full_n[0]),
      .x_out_write(reorder_x_in_bind_write[0]));
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
