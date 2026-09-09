`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [15:0] pe_x_mem_dout [0:7],
  input  wire pe_x_mem_empty_n [0:7],
  output wire pe_x_mem_read [0:7],
  input  wire [63:0] pe_w_mem_dout [0:7],
  input  wire pe_w_mem_empty_n [0:7],
  output wire pe_w_mem_read [0:7],
  output wire [31:0] switch_out_l_bind_din [0:1],
  output wire switch_out_l_bind_write [0:1],
  input  wire switch_out_l_bind_full_n [0:1],
  output wire [31:0] switch_out_r_bind_din [0:1],
  output wire switch_out_r_bind_write [0:1],
  input  wire switch_out_r_bind_full_n [0:1],
  input  wire [15:0] switch_cmd_mem_dout [0:5],
  input  wire switch_cmd_mem_empty_n [0:5],
  output wire switch_cmd_mem_read [0:5]
);
  // family pe_p0_out_p0_in: 8 channel(s), 32-bit, depth 2
  wire [31:0] pe_p0_out_p0_in_din [0:7];
  wire [31:0] pe_p0_out_p0_in_dout [0:7];
  wire pe_p0_out_p0_in_full_n [0:7];
  wire pe_p0_out_p0_in_write [0:7];
  wire pe_p0_out_p0_in_empty_n [0:7];
  wire pe_p0_out_p0_in_read [0:7];
  genvar pe_p0_out_p0_in_i;
  generate
    for (pe_p0_out_p0_in_i = 0; pe_p0_out_p0_in_i < 8; pe_p0_out_p0_in_i = pe_p0_out_p0_in_i + 1) begin : g_pe_p0_out_p0_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_p0_out_p0_in_din[pe_p0_out_p0_in_i]), .full_n(pe_p0_out_p0_in_full_n[pe_p0_out_p0_in_i]), .write(pe_p0_out_p0_in_write[pe_p0_out_p0_in_i]), .dout(pe_p0_out_p0_in_dout[pe_p0_out_p0_in_i]), .empty_n(pe_p0_out_p0_in_empty_n[pe_p0_out_p0_in_i]), .read(pe_p0_out_p0_in_read[pe_p0_out_p0_in_i]));
    end
  endgenerate
  // family pe_p1_out_p1_in: 8 channel(s), 32-bit, depth 2
  wire [31:0] pe_p1_out_p1_in_din [0:7];
  wire [31:0] pe_p1_out_p1_in_dout [0:7];
  wire pe_p1_out_p1_in_full_n [0:7];
  wire pe_p1_out_p1_in_write [0:7];
  wire pe_p1_out_p1_in_empty_n [0:7];
  wire pe_p1_out_p1_in_read [0:7];
  genvar pe_p1_out_p1_in_i;
  generate
    for (pe_p1_out_p1_in_i = 0; pe_p1_out_p1_in_i < 8; pe_p1_out_p1_in_i = pe_p1_out_p1_in_i + 1) begin : g_pe_p1_out_p1_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_p1_out_p1_in_din[pe_p1_out_p1_in_i]), .full_n(pe_p1_out_p1_in_full_n[pe_p1_out_p1_in_i]), .write(pe_p1_out_p1_in_write[pe_p1_out_p1_in_i]), .dout(pe_p1_out_p1_in_dout[pe_p1_out_p1_in_i]), .empty_n(pe_p1_out_p1_in_empty_n[pe_p1_out_p1_in_i]), .read(pe_p1_out_p1_in_read[pe_p1_out_p1_in_i]));
    end
  endgenerate
  // family switch_in_l_bind: 2 channel(s), 32-bit, depth 2
  wire [31:0] switch_in_l_bind_din [0:1];
  wire [31:0] switch_in_l_bind_dout [0:1];
  wire switch_in_l_bind_full_n [0:1];
  wire switch_in_l_bind_write [0:1];
  wire switch_in_l_bind_empty_n [0:1];
  wire switch_in_l_bind_read [0:1];
  genvar switch_in_l_bind_i;
  generate
    for (switch_in_l_bind_i = 0; switch_in_l_bind_i < 2; switch_in_l_bind_i = switch_in_l_bind_i + 1) begin : g_switch_in_l_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(switch_in_l_bind_din[switch_in_l_bind_i]), .full_n(switch_in_l_bind_full_n[switch_in_l_bind_i]), .write(switch_in_l_bind_write[switch_in_l_bind_i]), .dout(switch_in_l_bind_dout[switch_in_l_bind_i]), .empty_n(switch_in_l_bind_empty_n[switch_in_l_bind_i]), .read(switch_in_l_bind_read[switch_in_l_bind_i]));
    end
  endgenerate
  // family switch_in_r_bind: 2 channel(s), 32-bit, depth 2
  wire [31:0] switch_in_r_bind_din [0:1];
  wire [31:0] switch_in_r_bind_dout [0:1];
  wire switch_in_r_bind_full_n [0:1];
  wire switch_in_r_bind_write [0:1];
  wire switch_in_r_bind_empty_n [0:1];
  wire switch_in_r_bind_read [0:1];
  genvar switch_in_r_bind_i;
  generate
    for (switch_in_r_bind_i = 0; switch_in_r_bind_i < 2; switch_in_r_bind_i = switch_in_r_bind_i + 1) begin : g_switch_in_r_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(switch_in_r_bind_din[switch_in_r_bind_i]), .full_n(switch_in_r_bind_full_n[switch_in_r_bind_i]), .write(switch_in_r_bind_write[switch_in_r_bind_i]), .dout(switch_in_r_bind_dout[switch_in_r_bind_i]), .empty_n(switch_in_r_bind_empty_n[switch_in_r_bind_i]), .read(switch_in_r_bind_read[switch_in_r_bind_i]));
    end
  endgenerate
  // family switch_out_l_in_l: 6 channel(s), 32-bit, depth 2
  wire [31:0] switch_out_l_in_l_din [0:5];
  wire [31:0] switch_out_l_in_l_dout [0:5];
  wire switch_out_l_in_l_full_n [0:5];
  wire switch_out_l_in_l_write [0:5];
  wire switch_out_l_in_l_empty_n [0:5];
  wire switch_out_l_in_l_read [0:5];
  genvar switch_out_l_in_l_i;
  generate
    for (switch_out_l_in_l_i = 0; switch_out_l_in_l_i < 6; switch_out_l_in_l_i = switch_out_l_in_l_i + 1) begin : g_switch_out_l_in_l
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(switch_out_l_in_l_din[switch_out_l_in_l_i]), .full_n(switch_out_l_in_l_full_n[switch_out_l_in_l_i]), .write(switch_out_l_in_l_write[switch_out_l_in_l_i]), .dout(switch_out_l_in_l_dout[switch_out_l_in_l_i]), .empty_n(switch_out_l_in_l_empty_n[switch_out_l_in_l_i]), .read(switch_out_l_in_l_read[switch_out_l_in_l_i]));
    end
  endgenerate
  // family switch_out_r_in_l: 6 channel(s), 32-bit, depth 2
  wire [31:0] switch_out_r_in_l_din [0:5];
  wire [31:0] switch_out_r_in_l_dout [0:5];
  wire switch_out_r_in_l_full_n [0:5];
  wire switch_out_r_in_l_write [0:5];
  wire switch_out_r_in_l_empty_n [0:5];
  wire switch_out_r_in_l_read [0:5];
  genvar switch_out_r_in_l_i;
  generate
    for (switch_out_r_in_l_i = 0; switch_out_r_in_l_i < 6; switch_out_r_in_l_i = switch_out_r_in_l_i + 1) begin : g_switch_out_r_in_l
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(switch_out_r_in_l_din[switch_out_r_in_l_i]), .full_n(switch_out_r_in_l_full_n[switch_out_r_in_l_i]), .write(switch_out_r_in_l_write[switch_out_r_in_l_i]), .dout(switch_out_r_in_l_dout[switch_out_r_in_l_i]), .empty_n(switch_out_r_in_l_empty_n[switch_out_r_in_l_i]), .read(switch_out_r_in_l_read[switch_out_r_in_l_i]));
    end
  endgenerate
  // family switch_out_l_in_r: 6 channel(s), 32-bit, depth 2
  wire [31:0] switch_out_l_in_r_din [0:5];
  wire [31:0] switch_out_l_in_r_dout [0:5];
  wire switch_out_l_in_r_full_n [0:5];
  wire switch_out_l_in_r_write [0:5];
  wire switch_out_l_in_r_empty_n [0:5];
  wire switch_out_l_in_r_read [0:5];
  genvar switch_out_l_in_r_i;
  generate
    for (switch_out_l_in_r_i = 0; switch_out_l_in_r_i < 6; switch_out_l_in_r_i = switch_out_l_in_r_i + 1) begin : g_switch_out_l_in_r
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(switch_out_l_in_r_din[switch_out_l_in_r_i]), .full_n(switch_out_l_in_r_full_n[switch_out_l_in_r_i]), .write(switch_out_l_in_r_write[switch_out_l_in_r_i]), .dout(switch_out_l_in_r_dout[switch_out_l_in_r_i]), .empty_n(switch_out_l_in_r_empty_n[switch_out_l_in_r_i]), .read(switch_out_l_in_r_read[switch_out_l_in_r_i]));
    end
  endgenerate
  // family switch_out_r_in_r: 6 channel(s), 32-bit, depth 2
  wire [31:0] switch_out_r_in_r_din [0:5];
  wire [31:0] switch_out_r_in_r_dout [0:5];
  wire switch_out_r_in_r_full_n [0:5];
  wire switch_out_r_in_r_write [0:5];
  wire switch_out_r_in_r_empty_n [0:5];
  wire switch_out_r_in_r_read [0:5];
  genvar switch_out_r_in_r_i;
  generate
    for (switch_out_r_in_r_i = 0; switch_out_r_in_r_i < 6; switch_out_r_in_r_i = switch_out_r_in_r_i + 1) begin : g_switch_out_r_in_r
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(switch_out_r_in_r_din[switch_out_r_in_r_i]), .full_n(switch_out_r_in_r_full_n[switch_out_r_in_r_i]), .write(switch_out_r_in_r_write[switch_out_r_in_r_i]), .dout(switch_out_r_in_r_dout[switch_out_r_in_r_i]), .empty_n(switch_out_r_in_r_empty_n[switch_out_r_in_r_i]), .read(switch_out_r_in_r_read[switch_out_r_in_r_i]));
    end
  endgenerate
  // role pe_r0: 4 instance(s)
  pe_r0 u_pe_r0_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(pe_w_mem_dout[2]),
      .w_empty_n(pe_w_mem_empty_n[2]),
      .w_read(pe_w_mem_read[2]),
      .x_dout(pe_x_mem_dout[2]),
      .x_empty_n(pe_x_mem_empty_n[2]),
      .x_read(pe_x_mem_read[2]),
      .p0_in_dout(pe_p0_out_p0_in_dout[2]),
      .p0_in_empty_n(pe_p0_out_p0_in_empty_n[2]),
      .p0_in_read(pe_p0_out_p0_in_read[2]),
      .p0_out_din(pe_p0_out_p0_in_din[4]),
      .p0_out_full_n(pe_p0_out_p0_in_full_n[4]),
      .p0_out_write(pe_p0_out_p0_in_write[4]),
      .p1_in_dout(pe_p1_out_p1_in_dout[2]),
      .p1_in_empty_n(pe_p1_out_p1_in_empty_n[2]),
      .p1_in_read(pe_p1_out_p1_in_read[2]),
      .p1_out_din(pe_p1_out_p1_in_din[4]),
      .p1_out_full_n(pe_p1_out_p1_in_full_n[4]),
      .p1_out_write(pe_p1_out_p1_in_write[4]));
  pe_r0 u_pe_r0_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(pe_w_mem_dout[3]),
      .w_empty_n(pe_w_mem_empty_n[3]),
      .w_read(pe_w_mem_read[3]),
      .x_dout(pe_x_mem_dout[3]),
      .x_empty_n(pe_x_mem_empty_n[3]),
      .x_read(pe_x_mem_read[3]),
      .p0_in_dout(pe_p0_out_p0_in_dout[3]),
      .p0_in_empty_n(pe_p0_out_p0_in_empty_n[3]),
      .p0_in_read(pe_p0_out_p0_in_read[3]),
      .p0_out_din(pe_p0_out_p0_in_din[5]),
      .p0_out_full_n(pe_p0_out_p0_in_full_n[5]),
      .p0_out_write(pe_p0_out_p0_in_write[5]),
      .p1_in_dout(pe_p1_out_p1_in_dout[3]),
      .p1_in_empty_n(pe_p1_out_p1_in_empty_n[3]),
      .p1_in_read(pe_p1_out_p1_in_read[3]),
      .p1_out_din(pe_p1_out_p1_in_din[5]),
      .p1_out_full_n(pe_p1_out_p1_in_full_n[5]),
      .p1_out_write(pe_p1_out_p1_in_write[5]));
  pe_r0 u_pe_r0_2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(pe_w_mem_dout[4]),
      .w_empty_n(pe_w_mem_empty_n[4]),
      .w_read(pe_w_mem_read[4]),
      .x_dout(pe_x_mem_dout[4]),
      .x_empty_n(pe_x_mem_empty_n[4]),
      .x_read(pe_x_mem_read[4]),
      .p0_in_dout(pe_p0_out_p0_in_dout[4]),
      .p0_in_empty_n(pe_p0_out_p0_in_empty_n[4]),
      .p0_in_read(pe_p0_out_p0_in_read[4]),
      .p0_out_din(pe_p0_out_p0_in_din[6]),
      .p0_out_full_n(pe_p0_out_p0_in_full_n[6]),
      .p0_out_write(pe_p0_out_p0_in_write[6]),
      .p1_in_dout(pe_p1_out_p1_in_dout[4]),
      .p1_in_empty_n(pe_p1_out_p1_in_empty_n[4]),
      .p1_in_read(pe_p1_out_p1_in_read[4]),
      .p1_out_din(pe_p1_out_p1_in_din[6]),
      .p1_out_full_n(pe_p1_out_p1_in_full_n[6]),
      .p1_out_write(pe_p1_out_p1_in_write[6]));
  pe_r0 u_pe_r0_2_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(pe_w_mem_dout[5]),
      .w_empty_n(pe_w_mem_empty_n[5]),
      .w_read(pe_w_mem_read[5]),
      .x_dout(pe_x_mem_dout[5]),
      .x_empty_n(pe_x_mem_empty_n[5]),
      .x_read(pe_x_mem_read[5]),
      .p0_in_dout(pe_p0_out_p0_in_dout[5]),
      .p0_in_empty_n(pe_p0_out_p0_in_empty_n[5]),
      .p0_in_read(pe_p0_out_p0_in_read[5]),
      .p0_out_din(pe_p0_out_p0_in_din[7]),
      .p0_out_full_n(pe_p0_out_p0_in_full_n[7]),
      .p0_out_write(pe_p0_out_p0_in_write[7]),
      .p1_in_dout(pe_p1_out_p1_in_dout[5]),
      .p1_in_empty_n(pe_p1_out_p1_in_empty_n[5]),
      .p1_in_read(pe_p1_out_p1_in_read[5]),
      .p1_out_din(pe_p1_out_p1_in_din[7]),
      .p1_out_full_n(pe_p1_out_p1_in_full_n[7]),
      .p1_out_write(pe_p1_out_p1_in_write[7]));
  // role pe_r1: 2 instance(s)
  pe_r1 u_pe_r1_3_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(pe_w_mem_dout[6]),
      .w_empty_n(pe_w_mem_empty_n[6]),
      .w_read(pe_w_mem_read[6]),
      .x_dout(pe_x_mem_dout[6]),
      .x_empty_n(pe_x_mem_empty_n[6]),
      .x_read(pe_x_mem_read[6]),
      .p0_in_dout(pe_p0_out_p0_in_dout[6]),
      .p0_in_empty_n(pe_p0_out_p0_in_empty_n[6]),
      .p0_in_read(pe_p0_out_p0_in_read[6]),
      .p0_out_din(switch_in_l_bind_din[0]),
      .p0_out_full_n(switch_in_l_bind_full_n[0]),
      .p0_out_write(switch_in_l_bind_write[0]),
      .p1_in_dout(pe_p1_out_p1_in_dout[6]),
      .p1_in_empty_n(pe_p1_out_p1_in_empty_n[6]),
      .p1_in_read(pe_p1_out_p1_in_read[6]),
      .p1_out_din(switch_in_r_bind_din[0]),
      .p1_out_full_n(switch_in_r_bind_full_n[0]),
      .p1_out_write(switch_in_r_bind_write[0]));
  pe_r1 u_pe_r1_3_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(pe_w_mem_dout[7]),
      .w_empty_n(pe_w_mem_empty_n[7]),
      .w_read(pe_w_mem_read[7]),
      .x_dout(pe_x_mem_dout[7]),
      .x_empty_n(pe_x_mem_empty_n[7]),
      .x_read(pe_x_mem_read[7]),
      .p0_in_dout(pe_p0_out_p0_in_dout[7]),
      .p0_in_empty_n(pe_p0_out_p0_in_empty_n[7]),
      .p0_in_read(pe_p0_out_p0_in_read[7]),
      .p0_out_din(switch_in_l_bind_din[1]),
      .p0_out_full_n(switch_in_l_bind_full_n[1]),
      .p0_out_write(switch_in_l_bind_write[1]),
      .p1_in_dout(pe_p1_out_p1_in_dout[7]),
      .p1_in_empty_n(pe_p1_out_p1_in_empty_n[7]),
      .p1_in_read(pe_p1_out_p1_in_read[7]),
      .p1_out_din(switch_in_r_bind_din[1]),
      .p1_out_full_n(switch_in_r_bind_full_n[1]),
      .p1_out_write(switch_in_r_bind_write[1]));
  // role pe_r2: 2 instance(s)
  pe_r2 u_pe_r2_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(pe_w_mem_dout[0]),
      .w_empty_n(pe_w_mem_empty_n[0]),
      .w_read(pe_w_mem_read[0]),
      .x_dout(pe_x_mem_dout[0]),
      .x_empty_n(pe_x_mem_empty_n[0]),
      .x_read(pe_x_mem_read[0]),
      .p0_out_din(pe_p0_out_p0_in_din[2]),
      .p0_out_full_n(pe_p0_out_p0_in_full_n[2]),
      .p0_out_write(pe_p0_out_p0_in_write[2]),
      .p1_out_din(pe_p1_out_p1_in_din[2]),
      .p1_out_full_n(pe_p1_out_p1_in_full_n[2]),
      .p1_out_write(pe_p1_out_p1_in_write[2]));
  pe_r2 u_pe_r2_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(pe_w_mem_dout[1]),
      .w_empty_n(pe_w_mem_empty_n[1]),
      .w_read(pe_w_mem_read[1]),
      .x_dout(pe_x_mem_dout[1]),
      .x_empty_n(pe_x_mem_empty_n[1]),
      .x_read(pe_x_mem_read[1]),
      .p0_out_din(pe_p0_out_p0_in_din[3]),
      .p0_out_full_n(pe_p0_out_p0_in_full_n[3]),
      .p0_out_write(pe_p0_out_p0_in_write[3]),
      .p1_out_din(pe_p1_out_p1_in_din[3]),
      .p1_out_full_n(pe_p1_out_p1_in_full_n[3]),
      .p1_out_write(pe_p1_out_p1_in_write[3]));
  // role switch_r0: 1 instance(s)
  switch_r0 u_switch_r0_2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .cmd_dout(switch_cmd_mem_dout[4]),
      .cmd_empty_n(switch_cmd_mem_empty_n[4]),
      .cmd_read(switch_cmd_mem_read[4]),
      .in_l_dout(switch_out_l_in_l_dout[4]),
      .in_l_empty_n(switch_out_l_in_l_empty_n[4]),
      .in_l_read(switch_out_l_in_l_read[4]),
      .in_r_dout(switch_out_l_in_r_dout[4]),
      .in_r_empty_n(switch_out_l_in_r_empty_n[4]),
      .in_r_read(switch_out_l_in_r_read[4]),
      .out_l_din(switch_out_l_bind_din[0]),
      .out_l_full_n(switch_out_l_bind_full_n[0]),
      .out_l_write(switch_out_l_bind_write[0]),
      .out_r_din(switch_out_r_bind_din[0]),
      .out_r_full_n(switch_out_r_bind_full_n[0]),
      .out_r_write(switch_out_r_bind_write[0]));
  // role switch_r1: 1 instance(s)
  switch_r1 u_switch_r1_2_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .cmd_dout(switch_cmd_mem_dout[5]),
      .cmd_empty_n(switch_cmd_mem_empty_n[5]),
      .cmd_read(switch_cmd_mem_read[5]),
      .in_l_dout(switch_out_r_in_l_dout[5]),
      .in_l_empty_n(switch_out_r_in_l_empty_n[5]),
      .in_l_read(switch_out_r_in_l_read[5]),
      .in_r_dout(switch_out_r_in_r_dout[5]),
      .in_r_empty_n(switch_out_r_in_r_empty_n[5]),
      .in_r_read(switch_out_r_in_r_read[5]),
      .out_l_din(switch_out_l_bind_din[1]),
      .out_l_full_n(switch_out_l_bind_full_n[1]),
      .out_l_write(switch_out_l_bind_write[1]),
      .out_r_din(switch_out_r_bind_din[1]),
      .out_r_full_n(switch_out_r_bind_full_n[1]),
      .out_r_write(switch_out_r_bind_write[1]));
  // role switch_r2: 1 instance(s)
  switch_r2 u_switch_r2_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .cmd_dout(switch_cmd_mem_dout[2]),
      .cmd_empty_n(switch_cmd_mem_empty_n[2]),
      .cmd_read(switch_cmd_mem_read[2]),
      .in_l_dout(switch_out_l_in_l_dout[2]),
      .in_l_empty_n(switch_out_l_in_l_empty_n[2]),
      .in_l_read(switch_out_l_in_l_read[2]),
      .in_r_dout(switch_out_l_in_r_dout[2]),
      .in_r_empty_n(switch_out_l_in_r_empty_n[2]),
      .in_r_read(switch_out_l_in_r_read[2]),
      .out_l_din(switch_out_l_in_l_din[4]),
      .out_l_full_n(switch_out_l_in_l_full_n[4]),
      .out_l_write(switch_out_l_in_l_write[4]),
      .out_r_din(switch_out_r_in_l_din[5]),
      .out_r_full_n(switch_out_r_in_l_full_n[5]),
      .out_r_write(switch_out_r_in_l_write[5]));
  // role switch_r3: 1 instance(s)
  switch_r3 u_switch_r3_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .cmd_dout(switch_cmd_mem_dout[3]),
      .cmd_empty_n(switch_cmd_mem_empty_n[3]),
      .cmd_read(switch_cmd_mem_read[3]),
      .in_l_dout(switch_out_r_in_l_dout[3]),
      .in_l_empty_n(switch_out_r_in_l_empty_n[3]),
      .in_l_read(switch_out_r_in_l_read[3]),
      .in_r_dout(switch_out_r_in_r_dout[3]),
      .in_r_empty_n(switch_out_r_in_r_empty_n[3]),
      .in_r_read(switch_out_r_in_r_read[3]),
      .out_l_din(switch_out_l_in_r_din[4]),
      .out_l_full_n(switch_out_l_in_r_full_n[4]),
      .out_l_write(switch_out_l_in_r_write[4]),
      .out_r_din(switch_out_r_in_r_din[5]),
      .out_r_full_n(switch_out_r_in_r_full_n[5]),
      .out_r_write(switch_out_r_in_r_write[5]));
  // role switch_r4: 1 instance(s)
  switch_r4 u_switch_r4_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .cmd_dout(switch_cmd_mem_dout[0]),
      .cmd_empty_n(switch_cmd_mem_empty_n[0]),
      .cmd_read(switch_cmd_mem_read[0]),
      .in_l_dout(switch_in_l_bind_dout[0]),
      .in_l_empty_n(switch_in_l_bind_empty_n[0]),
      .in_l_read(switch_in_l_bind_read[0]),
      .in_r_dout(switch_in_r_bind_dout[0]),
      .in_r_empty_n(switch_in_r_bind_empty_n[0]),
      .in_r_read(switch_in_r_bind_read[0]),
      .out_l_din(switch_out_l_in_l_din[2]),
      .out_l_full_n(switch_out_l_in_l_full_n[2]),
      .out_l_write(switch_out_l_in_l_write[2]),
      .out_r_din(switch_out_r_in_l_din[3]),
      .out_r_full_n(switch_out_r_in_l_full_n[3]),
      .out_r_write(switch_out_r_in_l_write[3]));
  // role switch_r5: 1 instance(s)
  switch_r5 u_switch_r5_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .cmd_dout(switch_cmd_mem_dout[1]),
      .cmd_empty_n(switch_cmd_mem_empty_n[1]),
      .cmd_read(switch_cmd_mem_read[1]),
      .in_l_dout(switch_in_l_bind_dout[1]),
      .in_l_empty_n(switch_in_l_bind_empty_n[1]),
      .in_l_read(switch_in_l_bind_read[1]),
      .in_r_dout(switch_in_r_bind_dout[1]),
      .in_r_empty_n(switch_in_r_bind_empty_n[1]),
      .in_r_read(switch_in_r_bind_read[1]),
      .out_l_din(switch_out_l_in_r_din[2]),
      .out_l_full_n(switch_out_l_in_r_full_n[2]),
      .out_l_write(switch_out_l_in_r_write[2]),
      .out_r_din(switch_out_r_in_r_din[3]),
      .out_r_full_n(switch_out_r_in_r_full_n[3]),
      .out_r_write(switch_out_r_in_r_write[3]));
endmodule
