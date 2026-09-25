`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [7:0] pe_a_in_bind_dout [0:3],
  input  wire pe_a_in_bind_empty_n [0:3],
  output wire pe_a_in_bind_read [0:3],
  input  wire [7:0] pe_w_in_bind_dout [0:3],
  input  wire pe_w_in_bind_empty_n [0:3],
  output wire pe_w_in_bind_read [0:3],
  output wire [31:0] laneq_k512n32_y_out_bind_din [0:3],
  output wire laneq_k512n32_y_out_bind_write [0:3],
  input  wire laneq_k512n32_y_out_bind_full_n [0:3],
  input  wire [31:0] laneq_k512n32_b_mem_dout [0:3],
  input  wire laneq_k512n32_b_mem_empty_n [0:3],
  output wire laneq_k512n32_b_mem_read [0:3]
);
  // family pe_a_out_a_in: 16 channel(s), 8-bit, depth 0
  wire [7:0] pe_a_out_a_in_din [0:15];
  wire [7:0] pe_a_out_a_in_dout [0:15];
  wire pe_a_out_a_in_full_n [0:15];
  wire pe_a_out_a_in_write [0:15];
  wire pe_a_out_a_in_empty_n [0:15];
  wire pe_a_out_a_in_read [0:15];
  genvar pe_a_out_a_in_i;
  generate
    for (pe_a_out_a_in_i = 0; pe_a_out_a_in_i < 16; pe_a_out_a_in_i = pe_a_out_a_in_i + 1) begin : g_pe_a_out_a_in
      spmw_fifo #(.DW(8), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_a_out_a_in_din[pe_a_out_a_in_i]), .full_n(pe_a_out_a_in_full_n[pe_a_out_a_in_i]), .write(pe_a_out_a_in_write[pe_a_out_a_in_i]), .dout(pe_a_out_a_in_dout[pe_a_out_a_in_i]), .empty_n(pe_a_out_a_in_empty_n[pe_a_out_a_in_i]), .read(pe_a_out_a_in_read[pe_a_out_a_in_i]));
    end
  endgenerate
  // family pe_w_out_w_in: 16 channel(s), 8-bit, depth 0
  wire [7:0] pe_w_out_w_in_din [0:15];
  wire [7:0] pe_w_out_w_in_dout [0:15];
  wire pe_w_out_w_in_full_n [0:15];
  wire pe_w_out_w_in_write [0:15];
  wire pe_w_out_w_in_empty_n [0:15];
  wire pe_w_out_w_in_read [0:15];
  genvar pe_w_out_w_in_i;
  generate
    for (pe_w_out_w_in_i = 0; pe_w_out_w_in_i < 16; pe_w_out_w_in_i = pe_w_out_w_in_i + 1) begin : g_pe_w_out_w_in
      spmw_fifo #(.DW(8), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_w_out_w_in_din[pe_w_out_w_in_i]), .full_n(pe_w_out_w_in_full_n[pe_w_out_w_in_i]), .write(pe_w_out_w_in_write[pe_w_out_w_in_i]), .dout(pe_w_out_w_in_dout[pe_w_out_w_in_i]), .empty_n(pe_w_out_w_in_empty_n[pe_w_out_w_in_i]), .read(pe_w_out_w_in_read[pe_w_out_w_in_i]));
    end
  endgenerate
  // family pe_p_out_p_in: 16 channel(s), 32-bit, depth 0
  wire [31:0] pe_p_out_p_in_din [0:15];
  wire [31:0] pe_p_out_p_in_dout [0:15];
  wire pe_p_out_p_in_full_n [0:15];
  wire pe_p_out_p_in_write [0:15];
  wire pe_p_out_p_in_empty_n [0:15];
  wire pe_p_out_p_in_read [0:15];
  genvar pe_p_out_p_in_i;
  generate
    for (pe_p_out_p_in_i = 0; pe_p_out_p_in_i < 16; pe_p_out_p_in_i = pe_p_out_p_in_i + 1) begin : g_pe_p_out_p_in
      spmw_fifo #(.DW(32), .DEPTH(0)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_p_out_p_in_din[pe_p_out_p_in_i]), .full_n(pe_p_out_p_in_full_n[pe_p_out_p_in_i]), .write(pe_p_out_p_in_write[pe_p_out_p_in_i]), .dout(pe_p_out_p_in_dout[pe_p_out_p_in_i]), .empty_n(pe_p_out_p_in_empty_n[pe_p_out_p_in_i]), .read(pe_p_out_p_in_read[pe_p_out_p_in_i]));
    end
  endgenerate
  // family laneq_k512n32_z_in_bind: 4 channel(s), 32-bit, depth 2
  wire [31:0] laneq_k512n32_z_in_bind_din [0:3];
  wire [31:0] laneq_k512n32_z_in_bind_dout [0:3];
  wire laneq_k512n32_z_in_bind_full_n [0:3];
  wire laneq_k512n32_z_in_bind_write [0:3];
  wire laneq_k512n32_z_in_bind_empty_n [0:3];
  wire laneq_k512n32_z_in_bind_read [0:3];
  genvar laneq_k512n32_z_in_bind_i;
  generate
    for (laneq_k512n32_z_in_bind_i = 0; laneq_k512n32_z_in_bind_i < 4; laneq_k512n32_z_in_bind_i = laneq_k512n32_z_in_bind_i + 1) begin : g_laneq_k512n32_z_in_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(laneq_k512n32_z_in_bind_din[laneq_k512n32_z_in_bind_i]), .full_n(laneq_k512n32_z_in_bind_full_n[laneq_k512n32_z_in_bind_i]), .write(laneq_k512n32_z_in_bind_write[laneq_k512n32_z_in_bind_i]), .dout(laneq_k512n32_z_in_bind_dout[laneq_k512n32_z_in_bind_i]), .empty_n(laneq_k512n32_z_in_bind_empty_n[laneq_k512n32_z_in_bind_i]), .read(laneq_k512n32_z_in_bind_read[laneq_k512n32_z_in_bind_i]));
    end
  endgenerate
  // role pe_r0: 4 instance(s)
  pe_r0 u_pe_r0_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[5]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[5]),
      .a_in_read(pe_a_out_a_in_read[5]),
      .a_out_din(pe_a_out_a_in_din[6]),
      .a_out_full_n(pe_a_out_a_in_full_n[6]),
      .a_out_write(pe_a_out_a_in_write[6]),
      .p_in_dout(pe_p_out_p_in_dout[5]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[5]),
      .p_in_read(pe_p_out_p_in_read[5]),
      .p_out_din(pe_p_out_p_in_din[9]),
      .p_out_full_n(pe_p_out_p_in_full_n[9]),
      .p_out_write(pe_p_out_p_in_write[9]),
      .w_in_dout(pe_w_out_w_in_dout[5]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[5]),
      .w_in_read(pe_w_out_w_in_read[5]),
      .w_out_din(pe_w_out_w_in_din[6]),
      .w_out_full_n(pe_w_out_w_in_full_n[6]),
      .w_out_write(pe_w_out_w_in_write[6]));
  pe_r0 u_pe_r0_1_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[6]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[6]),
      .a_in_read(pe_a_out_a_in_read[6]),
      .a_out_din(pe_a_out_a_in_din[7]),
      .a_out_full_n(pe_a_out_a_in_full_n[7]),
      .a_out_write(pe_a_out_a_in_write[7]),
      .p_in_dout(pe_p_out_p_in_dout[6]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[6]),
      .p_in_read(pe_p_out_p_in_read[6]),
      .p_out_din(pe_p_out_p_in_din[10]),
      .p_out_full_n(pe_p_out_p_in_full_n[10]),
      .p_out_write(pe_p_out_p_in_write[10]),
      .w_in_dout(pe_w_out_w_in_dout[6]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[6]),
      .w_in_read(pe_w_out_w_in_read[6]),
      .w_out_din(pe_w_out_w_in_din[7]),
      .w_out_full_n(pe_w_out_w_in_full_n[7]),
      .w_out_write(pe_w_out_w_in_write[7]));
  pe_r0 u_pe_r0_2_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[9]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[9]),
      .a_in_read(pe_a_out_a_in_read[9]),
      .a_out_din(pe_a_out_a_in_din[10]),
      .a_out_full_n(pe_a_out_a_in_full_n[10]),
      .a_out_write(pe_a_out_a_in_write[10]),
      .p_in_dout(pe_p_out_p_in_dout[9]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[9]),
      .p_in_read(pe_p_out_p_in_read[9]),
      .p_out_din(pe_p_out_p_in_din[13]),
      .p_out_full_n(pe_p_out_p_in_full_n[13]),
      .p_out_write(pe_p_out_p_in_write[13]),
      .w_in_dout(pe_w_out_w_in_dout[9]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[9]),
      .w_in_read(pe_w_out_w_in_read[9]),
      .w_out_din(pe_w_out_w_in_din[10]),
      .w_out_full_n(pe_w_out_w_in_full_n[10]),
      .w_out_write(pe_w_out_w_in_write[10]));
  pe_r0 u_pe_r0_2_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[10]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[10]),
      .a_in_read(pe_a_out_a_in_read[10]),
      .a_out_din(pe_a_out_a_in_din[11]),
      .a_out_full_n(pe_a_out_a_in_full_n[11]),
      .a_out_write(pe_a_out_a_in_write[11]),
      .p_in_dout(pe_p_out_p_in_dout[10]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[10]),
      .p_in_read(pe_p_out_p_in_read[10]),
      .p_out_din(pe_p_out_p_in_din[14]),
      .p_out_full_n(pe_p_out_p_in_full_n[14]),
      .p_out_write(pe_p_out_p_in_write[14]),
      .w_in_dout(pe_w_out_w_in_dout[10]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[10]),
      .w_in_read(pe_w_out_w_in_read[10]),
      .w_out_din(pe_w_out_w_in_din[11]),
      .w_out_full_n(pe_w_out_w_in_full_n[11]),
      .w_out_write(pe_w_out_w_in_write[11]));
  // role pe_r1: 2 instance(s)
  pe_r1 u_pe_r1_3_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[13]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[13]),
      .a_in_read(pe_a_out_a_in_read[13]),
      .a_out_din(pe_a_out_a_in_din[14]),
      .a_out_full_n(pe_a_out_a_in_full_n[14]),
      .a_out_write(pe_a_out_a_in_write[14]),
      .p_in_dout(pe_p_out_p_in_dout[13]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[13]),
      .p_in_read(pe_p_out_p_in_read[13]),
      .p_out_din(laneq_k512n32_z_in_bind_din[1]),
      .p_out_full_n(laneq_k512n32_z_in_bind_full_n[1]),
      .p_out_write(laneq_k512n32_z_in_bind_write[1]),
      .w_in_dout(pe_w_out_w_in_dout[13]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[13]),
      .w_in_read(pe_w_out_w_in_read[13]),
      .w_out_din(pe_w_out_w_in_din[14]),
      .w_out_full_n(pe_w_out_w_in_full_n[14]),
      .w_out_write(pe_w_out_w_in_write[14]));
  pe_r1 u_pe_r1_3_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[14]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[14]),
      .a_in_read(pe_a_out_a_in_read[14]),
      .a_out_din(pe_a_out_a_in_din[15]),
      .a_out_full_n(pe_a_out_a_in_full_n[15]),
      .a_out_write(pe_a_out_a_in_write[15]),
      .p_in_dout(pe_p_out_p_in_dout[14]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[14]),
      .p_in_read(pe_p_out_p_in_read[14]),
      .p_out_din(laneq_k512n32_z_in_bind_din[2]),
      .p_out_full_n(laneq_k512n32_z_in_bind_full_n[2]),
      .p_out_write(laneq_k512n32_z_in_bind_write[2]),
      .w_in_dout(pe_w_out_w_in_dout[14]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[14]),
      .w_in_read(pe_w_out_w_in_read[14]),
      .w_out_din(pe_w_out_w_in_din[15]),
      .w_out_full_n(pe_w_out_w_in_full_n[15]),
      .w_out_write(pe_w_out_w_in_write[15]));
  // role pe_r2: 2 instance(s)
  pe_r2 u_pe_r2_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[1]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[1]),
      .a_in_read(pe_a_out_a_in_read[1]),
      .a_out_din(pe_a_out_a_in_din[2]),
      .a_out_full_n(pe_a_out_a_in_full_n[2]),
      .a_out_write(pe_a_out_a_in_write[2]),
      .p_out_din(pe_p_out_p_in_din[5]),
      .p_out_full_n(pe_p_out_p_in_full_n[5]),
      .p_out_write(pe_p_out_p_in_write[5]),
      .w_in_dout(pe_w_out_w_in_dout[1]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[1]),
      .w_in_read(pe_w_out_w_in_read[1]),
      .w_out_din(pe_w_out_w_in_din[2]),
      .w_out_full_n(pe_w_out_w_in_full_n[2]),
      .w_out_write(pe_w_out_w_in_write[2]));
  pe_r2 u_pe_r2_0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[2]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[2]),
      .a_in_read(pe_a_out_a_in_read[2]),
      .a_out_din(pe_a_out_a_in_din[3]),
      .a_out_full_n(pe_a_out_a_in_full_n[3]),
      .a_out_write(pe_a_out_a_in_write[3]),
      .p_out_din(pe_p_out_p_in_din[6]),
      .p_out_full_n(pe_p_out_p_in_full_n[6]),
      .p_out_write(pe_p_out_p_in_write[6]),
      .w_in_dout(pe_w_out_w_in_dout[2]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[2]),
      .w_in_read(pe_w_out_w_in_read[2]),
      .w_out_din(pe_w_out_w_in_din[3]),
      .w_out_full_n(pe_w_out_w_in_full_n[3]),
      .w_out_write(pe_w_out_w_in_write[3]));
  // role pe_r3: 2 instance(s)
  pe_r3 u_pe_r3_1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[7]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[7]),
      .a_in_read(pe_a_out_a_in_read[7]),
      .p_in_dout(pe_p_out_p_in_dout[7]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[7]),
      .p_in_read(pe_p_out_p_in_read[7]),
      .p_out_din(pe_p_out_p_in_din[11]),
      .p_out_full_n(pe_p_out_p_in_full_n[11]),
      .p_out_write(pe_p_out_p_in_write[11]),
      .w_in_dout(pe_w_out_w_in_dout[7]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[7]),
      .w_in_read(pe_w_out_w_in_read[7]));
  pe_r3 u_pe_r3_2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[11]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[11]),
      .a_in_read(pe_a_out_a_in_read[11]),
      .p_in_dout(pe_p_out_p_in_dout[11]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[11]),
      .p_in_read(pe_p_out_p_in_read[11]),
      .p_out_din(pe_p_out_p_in_din[15]),
      .p_out_full_n(pe_p_out_p_in_full_n[15]),
      .p_out_write(pe_p_out_p_in_write[15]),
      .w_in_dout(pe_w_out_w_in_dout[11]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[11]),
      .w_in_read(pe_w_out_w_in_read[11]));
  // role pe_r4: 2 instance(s)
  pe_r4 u_pe_r4_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[1]),
      .a_in_empty_n(pe_a_in_bind_empty_n[1]),
      .a_in_read(pe_a_in_bind_read[1]),
      .a_out_din(pe_a_out_a_in_din[5]),
      .a_out_full_n(pe_a_out_a_in_full_n[5]),
      .a_out_write(pe_a_out_a_in_write[5]),
      .p_in_dout(pe_p_out_p_in_dout[4]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[4]),
      .p_in_read(pe_p_out_p_in_read[4]),
      .p_out_din(pe_p_out_p_in_din[8]),
      .p_out_full_n(pe_p_out_p_in_full_n[8]),
      .p_out_write(pe_p_out_p_in_write[8]),
      .w_in_dout(pe_w_in_bind_dout[1]),
      .w_in_empty_n(pe_w_in_bind_empty_n[1]),
      .w_in_read(pe_w_in_bind_read[1]),
      .w_out_din(pe_w_out_w_in_din[5]),
      .w_out_full_n(pe_w_out_w_in_full_n[5]),
      .w_out_write(pe_w_out_w_in_write[5]));
  pe_r4 u_pe_r4_2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[2]),
      .a_in_empty_n(pe_a_in_bind_empty_n[2]),
      .a_in_read(pe_a_in_bind_read[2]),
      .a_out_din(pe_a_out_a_in_din[9]),
      .a_out_full_n(pe_a_out_a_in_full_n[9]),
      .a_out_write(pe_a_out_a_in_write[9]),
      .p_in_dout(pe_p_out_p_in_dout[8]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[8]),
      .p_in_read(pe_p_out_p_in_read[8]),
      .p_out_din(pe_p_out_p_in_din[12]),
      .p_out_full_n(pe_p_out_p_in_full_n[12]),
      .p_out_write(pe_p_out_p_in_write[12]),
      .w_in_dout(pe_w_in_bind_dout[2]),
      .w_in_empty_n(pe_w_in_bind_empty_n[2]),
      .w_in_read(pe_w_in_bind_read[2]),
      .w_out_din(pe_w_out_w_in_din[9]),
      .w_out_full_n(pe_w_out_w_in_full_n[9]),
      .w_out_write(pe_w_out_w_in_write[9]));
  // role pe_r5: 1 instance(s)
  pe_r5 u_pe_r5_3_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[15]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[15]),
      .a_in_read(pe_a_out_a_in_read[15]),
      .p_in_dout(pe_p_out_p_in_dout[15]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[15]),
      .p_in_read(pe_p_out_p_in_read[15]),
      .p_out_din(laneq_k512n32_z_in_bind_din[3]),
      .p_out_full_n(laneq_k512n32_z_in_bind_full_n[3]),
      .p_out_write(laneq_k512n32_z_in_bind_write[3]),
      .w_in_dout(pe_w_out_w_in_dout[15]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[15]),
      .w_in_read(pe_w_out_w_in_read[15]));
  // role pe_r6: 1 instance(s)
  pe_r6 u_pe_r6_0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_out_a_in_dout[3]),
      .a_in_empty_n(pe_a_out_a_in_empty_n[3]),
      .a_in_read(pe_a_out_a_in_read[3]),
      .p_out_din(pe_p_out_p_in_din[7]),
      .p_out_full_n(pe_p_out_p_in_full_n[7]),
      .p_out_write(pe_p_out_p_in_write[7]),
      .w_in_dout(pe_w_out_w_in_dout[3]),
      .w_in_empty_n(pe_w_out_w_in_empty_n[3]),
      .w_in_read(pe_w_out_w_in_read[3]));
  // role pe_r7: 1 instance(s)
  pe_r7 u_pe_r7_3_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[3]),
      .a_in_empty_n(pe_a_in_bind_empty_n[3]),
      .a_in_read(pe_a_in_bind_read[3]),
      .a_out_din(pe_a_out_a_in_din[13]),
      .a_out_full_n(pe_a_out_a_in_full_n[13]),
      .a_out_write(pe_a_out_a_in_write[13]),
      .p_in_dout(pe_p_out_p_in_dout[12]),
      .p_in_empty_n(pe_p_out_p_in_empty_n[12]),
      .p_in_read(pe_p_out_p_in_read[12]),
      .p_out_din(laneq_k512n32_z_in_bind_din[0]),
      .p_out_full_n(laneq_k512n32_z_in_bind_full_n[0]),
      .p_out_write(laneq_k512n32_z_in_bind_write[0]),
      .w_in_dout(pe_w_in_bind_dout[3]),
      .w_in_empty_n(pe_w_in_bind_empty_n[3]),
      .w_in_read(pe_w_in_bind_read[3]),
      .w_out_din(pe_w_out_w_in_din[13]),
      .w_out_full_n(pe_w_out_w_in_full_n[13]),
      .w_out_write(pe_w_out_w_in_write[13]));
  // role pe_r8: 1 instance(s)
  pe_r8 u_pe_r8_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .a_in_dout(pe_a_in_bind_dout[0]),
      .a_in_empty_n(pe_a_in_bind_empty_n[0]),
      .a_in_read(pe_a_in_bind_read[0]),
      .a_out_din(pe_a_out_a_in_din[1]),
      .a_out_full_n(pe_a_out_a_in_full_n[1]),
      .a_out_write(pe_a_out_a_in_write[1]),
      .p_out_din(pe_p_out_p_in_din[4]),
      .p_out_full_n(pe_p_out_p_in_full_n[4]),
      .p_out_write(pe_p_out_p_in_write[4]),
      .w_in_dout(pe_w_in_bind_dout[0]),
      .w_in_empty_n(pe_w_in_bind_empty_n[0]),
      .w_in_read(pe_w_in_bind_read[0]),
      .w_out_din(pe_w_out_w_in_din[1]),
      .w_out_full_n(pe_w_out_w_in_full_n[1]),
      .w_out_write(pe_w_out_w_in_write[1]));
  // role laneq_k512n32_r0: 4 instance(s)
  laneq_k512n32_r0 u_laneq_k512n32_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .b_dout(laneq_k512n32_b_mem_dout[0]),
      .b_empty_n(laneq_k512n32_b_mem_empty_n[0]),
      .b_read(laneq_k512n32_b_mem_read[0]),
      .y_out_din(laneq_k512n32_y_out_bind_din[0]),
      .y_out_full_n(laneq_k512n32_y_out_bind_full_n[0]),
      .y_out_write(laneq_k512n32_y_out_bind_write[0]),
      .z_in_dout(laneq_k512n32_z_in_bind_dout[0]),
      .z_in_empty_n(laneq_k512n32_z_in_bind_empty_n[0]),
      .z_in_read(laneq_k512n32_z_in_bind_read[0]));
  laneq_k512n32_r0 u_laneq_k512n32_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .b_dout(laneq_k512n32_b_mem_dout[1]),
      .b_empty_n(laneq_k512n32_b_mem_empty_n[1]),
      .b_read(laneq_k512n32_b_mem_read[1]),
      .y_out_din(laneq_k512n32_y_out_bind_din[1]),
      .y_out_full_n(laneq_k512n32_y_out_bind_full_n[1]),
      .y_out_write(laneq_k512n32_y_out_bind_write[1]),
      .z_in_dout(laneq_k512n32_z_in_bind_dout[1]),
      .z_in_empty_n(laneq_k512n32_z_in_bind_empty_n[1]),
      .z_in_read(laneq_k512n32_z_in_bind_read[1]));
  laneq_k512n32_r0 u_laneq_k512n32_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .b_dout(laneq_k512n32_b_mem_dout[2]),
      .b_empty_n(laneq_k512n32_b_mem_empty_n[2]),
      .b_read(laneq_k512n32_b_mem_read[2]),
      .y_out_din(laneq_k512n32_y_out_bind_din[2]),
      .y_out_full_n(laneq_k512n32_y_out_bind_full_n[2]),
      .y_out_write(laneq_k512n32_y_out_bind_write[2]),
      .z_in_dout(laneq_k512n32_z_in_bind_dout[2]),
      .z_in_empty_n(laneq_k512n32_z_in_bind_empty_n[2]),
      .z_in_read(laneq_k512n32_z_in_bind_read[2]));
  laneq_k512n32_r0 u_laneq_k512n32_r0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .b_dout(laneq_k512n32_b_mem_dout[3]),
      .b_empty_n(laneq_k512n32_b_mem_empty_n[3]),
      .b_read(laneq_k512n32_b_mem_read[3]),
      .y_out_din(laneq_k512n32_y_out_bind_din[3]),
      .y_out_full_n(laneq_k512n32_y_out_bind_full_n[3]),
      .y_out_write(laneq_k512n32_y_out_bind_write[3]),
      .z_in_dout(laneq_k512n32_z_in_bind_dout[3]),
      .z_in_empty_n(laneq_k512n32_z_in_bind_empty_n[3]),
      .z_in_read(laneq_k512n32_z_in_bind_read[3]));
endmodule
