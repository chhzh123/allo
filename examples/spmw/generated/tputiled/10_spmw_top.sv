`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [7:0] tiled_mac_a_in_bind_dout [0:3],
  input  wire tiled_mac_a_in_bind_empty_n [0:3],
  output wire tiled_mac_a_in_bind_read [0:3],
  input  wire [15:0] tiled_mac_w_mem_dout [0:15],
  input  wire tiled_mac_w_mem_empty_n [0:15],
  output wire tiled_mac_w_mem_read [0:15],
  input  wire [31:0] tiled_vpu_op_in_bind_dout [0:0],
  input  wire tiled_vpu_op_in_bind_empty_n [0:0],
  output wire tiled_vpu_op_in_bind_read [0:0],
  output wire [31:0] tiled_vpu_y_out_bind_din [0:3],
  output wire tiled_vpu_y_out_bind_write [0:3],
  input  wire tiled_vpu_y_out_bind_full_n [0:3],
  input  wire [31:0] tiled_vpu_b_mem_dout [0:3],
  input  wire tiled_vpu_b_mem_empty_n [0:3],
  output wire tiled_vpu_b_mem_read [0:3]
);
  // family tiled_mac_a_out_a_in: 16 channel(s), 8-bit, depth 2
  wire [7:0] tiled_mac_a_out_a_in_din [0:15];
  wire [7:0] tiled_mac_a_out_a_in_dout [0:15];
  wire tiled_mac_a_out_a_in_full_n [0:15];
  wire tiled_mac_a_out_a_in_write [0:15];
  wire tiled_mac_a_out_a_in_empty_n [0:15];
  wire tiled_mac_a_out_a_in_read [0:15];
  genvar tiled_mac_a_out_a_in_i;
  generate
    for (tiled_mac_a_out_a_in_i = 0; tiled_mac_a_out_a_in_i < 16; tiled_mac_a_out_a_in_i = tiled_mac_a_out_a_in_i + 1) begin : g_tiled_mac_a_out_a_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tiled_mac_a_out_a_in_din[tiled_mac_a_out_a_in_i]), .full_n(tiled_mac_a_out_a_in_full_n[tiled_mac_a_out_a_in_i]), .write(tiled_mac_a_out_a_in_write[tiled_mac_a_out_a_in_i]), .dout(tiled_mac_a_out_a_in_dout[tiled_mac_a_out_a_in_i]), .empty_n(tiled_mac_a_out_a_in_empty_n[tiled_mac_a_out_a_in_i]), .read(tiled_mac_a_out_a_in_read[tiled_mac_a_out_a_in_i]));
    end
  endgenerate
  // family tiled_mac_p_out_p_in: 16 channel(s), 32-bit, depth 2
  wire [31:0] tiled_mac_p_out_p_in_din [0:15];
  wire [31:0] tiled_mac_p_out_p_in_dout [0:15];
  wire tiled_mac_p_out_p_in_full_n [0:15];
  wire tiled_mac_p_out_p_in_write [0:15];
  wire tiled_mac_p_out_p_in_empty_n [0:15];
  wire tiled_mac_p_out_p_in_read [0:15];
  genvar tiled_mac_p_out_p_in_i;
  generate
    for (tiled_mac_p_out_p_in_i = 0; tiled_mac_p_out_p_in_i < 16; tiled_mac_p_out_p_in_i = tiled_mac_p_out_p_in_i + 1) begin : g_tiled_mac_p_out_p_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tiled_mac_p_out_p_in_din[tiled_mac_p_out_p_in_i]), .full_n(tiled_mac_p_out_p_in_full_n[tiled_mac_p_out_p_in_i]), .write(tiled_mac_p_out_p_in_write[tiled_mac_p_out_p_in_i]), .dout(tiled_mac_p_out_p_in_dout[tiled_mac_p_out_p_in_i]), .empty_n(tiled_mac_p_out_p_in_empty_n[tiled_mac_p_out_p_in_i]), .read(tiled_mac_p_out_p_in_read[tiled_mac_p_out_p_in_i]));
    end
  endgenerate
  // family tiled_vpu_z_in_bind: 4 channel(s), 32-bit, depth 2
  wire [31:0] tiled_vpu_z_in_bind_din [0:3];
  wire [31:0] tiled_vpu_z_in_bind_dout [0:3];
  wire tiled_vpu_z_in_bind_full_n [0:3];
  wire tiled_vpu_z_in_bind_write [0:3];
  wire tiled_vpu_z_in_bind_empty_n [0:3];
  wire tiled_vpu_z_in_bind_read [0:3];
  genvar tiled_vpu_z_in_bind_i;
  generate
    for (tiled_vpu_z_in_bind_i = 0; tiled_vpu_z_in_bind_i < 4; tiled_vpu_z_in_bind_i = tiled_vpu_z_in_bind_i + 1) begin : g_tiled_vpu_z_in_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tiled_vpu_z_in_bind_din[tiled_vpu_z_in_bind_i]), .full_n(tiled_vpu_z_in_bind_full_n[tiled_vpu_z_in_bind_i]), .write(tiled_vpu_z_in_bind_write[tiled_vpu_z_in_bind_i]), .dout(tiled_vpu_z_in_bind_dout[tiled_vpu_z_in_bind_i]), .empty_n(tiled_vpu_z_in_bind_empty_n[tiled_vpu_z_in_bind_i]), .read(tiled_vpu_z_in_bind_read[tiled_vpu_z_in_bind_i]));
    end
  endgenerate
  // family tiled_vpu_op_out_op_in: 4 channel(s), 32-bit, depth 2
  wire [31:0] tiled_vpu_op_out_op_in_din [0:3];
  wire [31:0] tiled_vpu_op_out_op_in_dout [0:3];
  wire tiled_vpu_op_out_op_in_full_n [0:3];
  wire tiled_vpu_op_out_op_in_write [0:3];
  wire tiled_vpu_op_out_op_in_empty_n [0:3];
  wire tiled_vpu_op_out_op_in_read [0:3];
  genvar tiled_vpu_op_out_op_in_i;
  generate
    for (tiled_vpu_op_out_op_in_i = 0; tiled_vpu_op_out_op_in_i < 4; tiled_vpu_op_out_op_in_i = tiled_vpu_op_out_op_in_i + 1) begin : g_tiled_vpu_op_out_op_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(tiled_vpu_op_out_op_in_din[tiled_vpu_op_out_op_in_i]), .full_n(tiled_vpu_op_out_op_in_full_n[tiled_vpu_op_out_op_in_i]), .write(tiled_vpu_op_out_op_in_write[tiled_vpu_op_out_op_in_i]), .dout(tiled_vpu_op_out_op_in_dout[tiled_vpu_op_out_op_in_i]), .empty_n(tiled_vpu_op_out_op_in_empty_n[tiled_vpu_op_out_op_in_i]), .read(tiled_vpu_op_out_op_in_read[tiled_vpu_op_out_op_in_i]));
    end
  endgenerate
  // role tiled_mac_r0: 4 instance(s)
  tiled_mac_r0 u_tiled_mac_r0_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[5]),
      .w_empty_n(tiled_mac_w_mem_empty_n[5]),
      .w_read(tiled_mac_w_mem_read[5]),
      .a_in_dout(tiled_mac_a_out_a_in_dout[5]),
      .a_in_empty_n(tiled_mac_a_out_a_in_empty_n[5]),
      .a_in_read(tiled_mac_a_out_a_in_read[5]),
      .a_out_din(tiled_mac_a_out_a_in_din[6]),
      .a_out_full_n(tiled_mac_a_out_a_in_full_n[6]),
      .a_out_write(tiled_mac_a_out_a_in_write[6]),
      .p_in_dout(tiled_mac_p_out_p_in_dout[5]),
      .p_in_empty_n(tiled_mac_p_out_p_in_empty_n[5]),
      .p_in_read(tiled_mac_p_out_p_in_read[5]),
      .p_out_din(tiled_mac_p_out_p_in_din[9]),
      .p_out_full_n(tiled_mac_p_out_p_in_full_n[9]),
      .p_out_write(tiled_mac_p_out_p_in_write[9]));
  tiled_mac_r0 u_tiled_mac_r0_1_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[6]),
      .w_empty_n(tiled_mac_w_mem_empty_n[6]),
      .w_read(tiled_mac_w_mem_read[6]),
      .a_in_dout(tiled_mac_a_out_a_in_dout[6]),
      .a_in_empty_n(tiled_mac_a_out_a_in_empty_n[6]),
      .a_in_read(tiled_mac_a_out_a_in_read[6]),
      .a_out_din(tiled_mac_a_out_a_in_din[7]),
      .a_out_full_n(tiled_mac_a_out_a_in_full_n[7]),
      .a_out_write(tiled_mac_a_out_a_in_write[7]),
      .p_in_dout(tiled_mac_p_out_p_in_dout[6]),
      .p_in_empty_n(tiled_mac_p_out_p_in_empty_n[6]),
      .p_in_read(tiled_mac_p_out_p_in_read[6]),
      .p_out_din(tiled_mac_p_out_p_in_din[10]),
      .p_out_full_n(tiled_mac_p_out_p_in_full_n[10]),
      .p_out_write(tiled_mac_p_out_p_in_write[10]));
  tiled_mac_r0 u_tiled_mac_r0_2_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[9]),
      .w_empty_n(tiled_mac_w_mem_empty_n[9]),
      .w_read(tiled_mac_w_mem_read[9]),
      .a_in_dout(tiled_mac_a_out_a_in_dout[9]),
      .a_in_empty_n(tiled_mac_a_out_a_in_empty_n[9]),
      .a_in_read(tiled_mac_a_out_a_in_read[9]),
      .a_out_din(tiled_mac_a_out_a_in_din[10]),
      .a_out_full_n(tiled_mac_a_out_a_in_full_n[10]),
      .a_out_write(tiled_mac_a_out_a_in_write[10]),
      .p_in_dout(tiled_mac_p_out_p_in_dout[9]),
      .p_in_empty_n(tiled_mac_p_out_p_in_empty_n[9]),
      .p_in_read(tiled_mac_p_out_p_in_read[9]),
      .p_out_din(tiled_mac_p_out_p_in_din[13]),
      .p_out_full_n(tiled_mac_p_out_p_in_full_n[13]),
      .p_out_write(tiled_mac_p_out_p_in_write[13]));
  tiled_mac_r0 u_tiled_mac_r0_2_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[10]),
      .w_empty_n(tiled_mac_w_mem_empty_n[10]),
      .w_read(tiled_mac_w_mem_read[10]),
      .a_in_dout(tiled_mac_a_out_a_in_dout[10]),
      .a_in_empty_n(tiled_mac_a_out_a_in_empty_n[10]),
      .a_in_read(tiled_mac_a_out_a_in_read[10]),
      .a_out_din(tiled_mac_a_out_a_in_din[11]),
      .a_out_full_n(tiled_mac_a_out_a_in_full_n[11]),
      .a_out_write(tiled_mac_a_out_a_in_write[11]),
      .p_in_dout(tiled_mac_p_out_p_in_dout[10]),
      .p_in_empty_n(tiled_mac_p_out_p_in_empty_n[10]),
      .p_in_read(tiled_mac_p_out_p_in_read[10]),
      .p_out_din(tiled_mac_p_out_p_in_din[14]),
      .p_out_full_n(tiled_mac_p_out_p_in_full_n[14]),
      .p_out_write(tiled_mac_p_out_p_in_write[14]));
  // role tiled_mac_r1: 2 instance(s)
  tiled_mac_r1 u_tiled_mac_r1_3_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[13]),
      .w_empty_n(tiled_mac_w_mem_empty_n[13]),
      .w_read(tiled_mac_w_mem_read[13]),
      .a_in_dout(tiled_mac_a_out_a_in_dout[13]),
      .a_in_empty_n(tiled_mac_a_out_a_in_empty_n[13]),
      .a_in_read(tiled_mac_a_out_a_in_read[13]),
      .a_out_din(tiled_mac_a_out_a_in_din[14]),
      .a_out_full_n(tiled_mac_a_out_a_in_full_n[14]),
      .a_out_write(tiled_mac_a_out_a_in_write[14]),
      .p_in_dout(tiled_mac_p_out_p_in_dout[13]),
      .p_in_empty_n(tiled_mac_p_out_p_in_empty_n[13]),
      .p_in_read(tiled_mac_p_out_p_in_read[13]),
      .p_out_din(tiled_vpu_z_in_bind_din[1]),
      .p_out_full_n(tiled_vpu_z_in_bind_full_n[1]),
      .p_out_write(tiled_vpu_z_in_bind_write[1]));
  tiled_mac_r1 u_tiled_mac_r1_3_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[14]),
      .w_empty_n(tiled_mac_w_mem_empty_n[14]),
      .w_read(tiled_mac_w_mem_read[14]),
      .a_in_dout(tiled_mac_a_out_a_in_dout[14]),
      .a_in_empty_n(tiled_mac_a_out_a_in_empty_n[14]),
      .a_in_read(tiled_mac_a_out_a_in_read[14]),
      .a_out_din(tiled_mac_a_out_a_in_din[15]),
      .a_out_full_n(tiled_mac_a_out_a_in_full_n[15]),
      .a_out_write(tiled_mac_a_out_a_in_write[15]),
      .p_in_dout(tiled_mac_p_out_p_in_dout[14]),
      .p_in_empty_n(tiled_mac_p_out_p_in_empty_n[14]),
      .p_in_read(tiled_mac_p_out_p_in_read[14]),
      .p_out_din(tiled_vpu_z_in_bind_din[2]),
      .p_out_full_n(tiled_vpu_z_in_bind_full_n[2]),
      .p_out_write(tiled_vpu_z_in_bind_write[2]));
  // role tiled_mac_r2: 2 instance(s)
  tiled_mac_r2 u_tiled_mac_r2_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[1]),
      .w_empty_n(tiled_mac_w_mem_empty_n[1]),
      .w_read(tiled_mac_w_mem_read[1]),
      .a_in_dout(tiled_mac_a_out_a_in_dout[1]),
      .a_in_empty_n(tiled_mac_a_out_a_in_empty_n[1]),
      .a_in_read(tiled_mac_a_out_a_in_read[1]),
      .a_out_din(tiled_mac_a_out_a_in_din[2]),
      .a_out_full_n(tiled_mac_a_out_a_in_full_n[2]),
      .a_out_write(tiled_mac_a_out_a_in_write[2]),
      .p_out_din(tiled_mac_p_out_p_in_din[5]),
      .p_out_full_n(tiled_mac_p_out_p_in_full_n[5]),
      .p_out_write(tiled_mac_p_out_p_in_write[5]));
  tiled_mac_r2 u_tiled_mac_r2_0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[2]),
      .w_empty_n(tiled_mac_w_mem_empty_n[2]),
      .w_read(tiled_mac_w_mem_read[2]),
      .a_in_dout(tiled_mac_a_out_a_in_dout[2]),
      .a_in_empty_n(tiled_mac_a_out_a_in_empty_n[2]),
      .a_in_read(tiled_mac_a_out_a_in_read[2]),
      .a_out_din(tiled_mac_a_out_a_in_din[3]),
      .a_out_full_n(tiled_mac_a_out_a_in_full_n[3]),
      .a_out_write(tiled_mac_a_out_a_in_write[3]),
      .p_out_din(tiled_mac_p_out_p_in_din[6]),
      .p_out_full_n(tiled_mac_p_out_p_in_full_n[6]),
      .p_out_write(tiled_mac_p_out_p_in_write[6]));
  // role tiled_mac_r3: 2 instance(s)
  tiled_mac_r3 u_tiled_mac_r3_1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[7]),
      .w_empty_n(tiled_mac_w_mem_empty_n[7]),
      .w_read(tiled_mac_w_mem_read[7]),
      .a_in_dout(tiled_mac_a_out_a_in_dout[7]),
      .a_in_empty_n(tiled_mac_a_out_a_in_empty_n[7]),
      .a_in_read(tiled_mac_a_out_a_in_read[7]),
      .p_in_dout(tiled_mac_p_out_p_in_dout[7]),
      .p_in_empty_n(tiled_mac_p_out_p_in_empty_n[7]),
      .p_in_read(tiled_mac_p_out_p_in_read[7]),
      .p_out_din(tiled_mac_p_out_p_in_din[11]),
      .p_out_full_n(tiled_mac_p_out_p_in_full_n[11]),
      .p_out_write(tiled_mac_p_out_p_in_write[11]));
  tiled_mac_r3 u_tiled_mac_r3_2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[11]),
      .w_empty_n(tiled_mac_w_mem_empty_n[11]),
      .w_read(tiled_mac_w_mem_read[11]),
      .a_in_dout(tiled_mac_a_out_a_in_dout[11]),
      .a_in_empty_n(tiled_mac_a_out_a_in_empty_n[11]),
      .a_in_read(tiled_mac_a_out_a_in_read[11]),
      .p_in_dout(tiled_mac_p_out_p_in_dout[11]),
      .p_in_empty_n(tiled_mac_p_out_p_in_empty_n[11]),
      .p_in_read(tiled_mac_p_out_p_in_read[11]),
      .p_out_din(tiled_mac_p_out_p_in_din[15]),
      .p_out_full_n(tiled_mac_p_out_p_in_full_n[15]),
      .p_out_write(tiled_mac_p_out_p_in_write[15]));
  // role tiled_mac_r4: 2 instance(s)
  tiled_mac_r4 u_tiled_mac_r4_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[4]),
      .w_empty_n(tiled_mac_w_mem_empty_n[4]),
      .w_read(tiled_mac_w_mem_read[4]),
      .a_in_dout(tiled_mac_a_in_bind_dout[1]),
      .a_in_empty_n(tiled_mac_a_in_bind_empty_n[1]),
      .a_in_read(tiled_mac_a_in_bind_read[1]),
      .a_out_din(tiled_mac_a_out_a_in_din[5]),
      .a_out_full_n(tiled_mac_a_out_a_in_full_n[5]),
      .a_out_write(tiled_mac_a_out_a_in_write[5]),
      .p_in_dout(tiled_mac_p_out_p_in_dout[4]),
      .p_in_empty_n(tiled_mac_p_out_p_in_empty_n[4]),
      .p_in_read(tiled_mac_p_out_p_in_read[4]),
      .p_out_din(tiled_mac_p_out_p_in_din[8]),
      .p_out_full_n(tiled_mac_p_out_p_in_full_n[8]),
      .p_out_write(tiled_mac_p_out_p_in_write[8]));
  tiled_mac_r4 u_tiled_mac_r4_2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[8]),
      .w_empty_n(tiled_mac_w_mem_empty_n[8]),
      .w_read(tiled_mac_w_mem_read[8]),
      .a_in_dout(tiled_mac_a_in_bind_dout[2]),
      .a_in_empty_n(tiled_mac_a_in_bind_empty_n[2]),
      .a_in_read(tiled_mac_a_in_bind_read[2]),
      .a_out_din(tiled_mac_a_out_a_in_din[9]),
      .a_out_full_n(tiled_mac_a_out_a_in_full_n[9]),
      .a_out_write(tiled_mac_a_out_a_in_write[9]),
      .p_in_dout(tiled_mac_p_out_p_in_dout[8]),
      .p_in_empty_n(tiled_mac_p_out_p_in_empty_n[8]),
      .p_in_read(tiled_mac_p_out_p_in_read[8]),
      .p_out_din(tiled_mac_p_out_p_in_din[12]),
      .p_out_full_n(tiled_mac_p_out_p_in_full_n[12]),
      .p_out_write(tiled_mac_p_out_p_in_write[12]));
  // role tiled_mac_r5: 1 instance(s)
  tiled_mac_r5 u_tiled_mac_r5_3_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[15]),
      .w_empty_n(tiled_mac_w_mem_empty_n[15]),
      .w_read(tiled_mac_w_mem_read[15]),
      .a_in_dout(tiled_mac_a_out_a_in_dout[15]),
      .a_in_empty_n(tiled_mac_a_out_a_in_empty_n[15]),
      .a_in_read(tiled_mac_a_out_a_in_read[15]),
      .p_in_dout(tiled_mac_p_out_p_in_dout[15]),
      .p_in_empty_n(tiled_mac_p_out_p_in_empty_n[15]),
      .p_in_read(tiled_mac_p_out_p_in_read[15]),
      .p_out_din(tiled_vpu_z_in_bind_din[3]),
      .p_out_full_n(tiled_vpu_z_in_bind_full_n[3]),
      .p_out_write(tiled_vpu_z_in_bind_write[3]));
  // role tiled_mac_r6: 1 instance(s)
  tiled_mac_r6 u_tiled_mac_r6_0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[3]),
      .w_empty_n(tiled_mac_w_mem_empty_n[3]),
      .w_read(tiled_mac_w_mem_read[3]),
      .a_in_dout(tiled_mac_a_out_a_in_dout[3]),
      .a_in_empty_n(tiled_mac_a_out_a_in_empty_n[3]),
      .a_in_read(tiled_mac_a_out_a_in_read[3]),
      .p_out_din(tiled_mac_p_out_p_in_din[7]),
      .p_out_full_n(tiled_mac_p_out_p_in_full_n[7]),
      .p_out_write(tiled_mac_p_out_p_in_write[7]));
  // role tiled_mac_r7: 1 instance(s)
  tiled_mac_r7 u_tiled_mac_r7_3_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[12]),
      .w_empty_n(tiled_mac_w_mem_empty_n[12]),
      .w_read(tiled_mac_w_mem_read[12]),
      .a_in_dout(tiled_mac_a_in_bind_dout[3]),
      .a_in_empty_n(tiled_mac_a_in_bind_empty_n[3]),
      .a_in_read(tiled_mac_a_in_bind_read[3]),
      .a_out_din(tiled_mac_a_out_a_in_din[13]),
      .a_out_full_n(tiled_mac_a_out_a_in_full_n[13]),
      .a_out_write(tiled_mac_a_out_a_in_write[13]),
      .p_in_dout(tiled_mac_p_out_p_in_dout[12]),
      .p_in_empty_n(tiled_mac_p_out_p_in_empty_n[12]),
      .p_in_read(tiled_mac_p_out_p_in_read[12]),
      .p_out_din(tiled_vpu_z_in_bind_din[0]),
      .p_out_full_n(tiled_vpu_z_in_bind_full_n[0]),
      .p_out_write(tiled_vpu_z_in_bind_write[0]));
  // role tiled_mac_r8: 1 instance(s)
  tiled_mac_r8 u_tiled_mac_r8_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(tiled_mac_w_mem_dout[0]),
      .w_empty_n(tiled_mac_w_mem_empty_n[0]),
      .w_read(tiled_mac_w_mem_read[0]),
      .a_in_dout(tiled_mac_a_in_bind_dout[0]),
      .a_in_empty_n(tiled_mac_a_in_bind_empty_n[0]),
      .a_in_read(tiled_mac_a_in_bind_read[0]),
      .a_out_din(tiled_mac_a_out_a_in_din[1]),
      .a_out_full_n(tiled_mac_a_out_a_in_full_n[1]),
      .a_out_write(tiled_mac_a_out_a_in_write[1]),
      .p_out_din(tiled_mac_p_out_p_in_din[4]),
      .p_out_full_n(tiled_mac_p_out_p_in_full_n[4]),
      .p_out_write(tiled_mac_p_out_p_in_write[4]));
  // role tiled_vpu_r0: 2 instance(s)
  tiled_vpu_r0 u_tiled_vpu_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .b_dout(tiled_vpu_b_mem_dout[1]),
      .b_empty_n(tiled_vpu_b_mem_empty_n[1]),
      .b_read(tiled_vpu_b_mem_read[1]),
      .op_in_dout(tiled_vpu_op_out_op_in_dout[1]),
      .op_in_empty_n(tiled_vpu_op_out_op_in_empty_n[1]),
      .op_in_read(tiled_vpu_op_out_op_in_read[1]),
      .op_out_din(tiled_vpu_op_out_op_in_din[2]),
      .op_out_full_n(tiled_vpu_op_out_op_in_full_n[2]),
      .op_out_write(tiled_vpu_op_out_op_in_write[2]),
      .y_out_din(tiled_vpu_y_out_bind_din[1]),
      .y_out_full_n(tiled_vpu_y_out_bind_full_n[1]),
      .y_out_write(tiled_vpu_y_out_bind_write[1]),
      .z_in_dout(tiled_vpu_z_in_bind_dout[1]),
      .z_in_empty_n(tiled_vpu_z_in_bind_empty_n[1]),
      .z_in_read(tiled_vpu_z_in_bind_read[1]));
  tiled_vpu_r0 u_tiled_vpu_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .b_dout(tiled_vpu_b_mem_dout[2]),
      .b_empty_n(tiled_vpu_b_mem_empty_n[2]),
      .b_read(tiled_vpu_b_mem_read[2]),
      .op_in_dout(tiled_vpu_op_out_op_in_dout[2]),
      .op_in_empty_n(tiled_vpu_op_out_op_in_empty_n[2]),
      .op_in_read(tiled_vpu_op_out_op_in_read[2]),
      .op_out_din(tiled_vpu_op_out_op_in_din[3]),
      .op_out_full_n(tiled_vpu_op_out_op_in_full_n[3]),
      .op_out_write(tiled_vpu_op_out_op_in_write[3]),
      .y_out_din(tiled_vpu_y_out_bind_din[2]),
      .y_out_full_n(tiled_vpu_y_out_bind_full_n[2]),
      .y_out_write(tiled_vpu_y_out_bind_write[2]),
      .z_in_dout(tiled_vpu_z_in_bind_dout[2]),
      .z_in_empty_n(tiled_vpu_z_in_bind_empty_n[2]),
      .z_in_read(tiled_vpu_z_in_bind_read[2]));
  // role tiled_vpu_r1: 1 instance(s)
  tiled_vpu_r1 u_tiled_vpu_r1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .b_dout(tiled_vpu_b_mem_dout[3]),
      .b_empty_n(tiled_vpu_b_mem_empty_n[3]),
      .b_read(tiled_vpu_b_mem_read[3]),
      .op_in_dout(tiled_vpu_op_out_op_in_dout[3]),
      .op_in_empty_n(tiled_vpu_op_out_op_in_empty_n[3]),
      .op_in_read(tiled_vpu_op_out_op_in_read[3]),
      .y_out_din(tiled_vpu_y_out_bind_din[3]),
      .y_out_full_n(tiled_vpu_y_out_bind_full_n[3]),
      .y_out_write(tiled_vpu_y_out_bind_write[3]),
      .z_in_dout(tiled_vpu_z_in_bind_dout[3]),
      .z_in_empty_n(tiled_vpu_z_in_bind_empty_n[3]),
      .z_in_read(tiled_vpu_z_in_bind_read[3]));
  // role tiled_vpu_r2: 1 instance(s)
  tiled_vpu_r2 u_tiled_vpu_r2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .b_dout(tiled_vpu_b_mem_dout[0]),
      .b_empty_n(tiled_vpu_b_mem_empty_n[0]),
      .b_read(tiled_vpu_b_mem_read[0]),
      .op_in_dout(tiled_vpu_op_in_bind_dout[0]),
      .op_in_empty_n(tiled_vpu_op_in_bind_empty_n[0]),
      .op_in_read(tiled_vpu_op_in_bind_read[0]),
      .op_out_din(tiled_vpu_op_out_op_in_din[1]),
      .op_out_full_n(tiled_vpu_op_out_op_in_full_n[1]),
      .op_out_write(tiled_vpu_op_out_op_in_write[1]),
      .y_out_din(tiled_vpu_y_out_bind_din[0]),
      .y_out_full_n(tiled_vpu_y_out_bind_full_n[0]),
      .y_out_write(tiled_vpu_y_out_bind_write[0]),
      .z_in_dout(tiled_vpu_z_in_bind_dout[0]),
      .z_in_empty_n(tiled_vpu_z_in_bind_empty_n[0]),
      .z_in_read(tiled_vpu_z_in_bind_read[0]));
endmodule
