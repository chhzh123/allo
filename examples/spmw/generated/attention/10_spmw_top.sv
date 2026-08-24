`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [7:0] mac_a_in_bind_dout [0:7],
  input  wire mac_a_in_bind_empty_n [0:7],
  output wire mac_a_in_bind_read [0:7],
  input  wire [7:0] mac_w_mem_dout [0:15],
  input  wire mac_w_mem_empty_n [0:15],
  output wire mac_w_mem_read [0:15],
  output wire [7:0] act_y_out_bind_din [0:1],
  output wire act_y_out_bind_write [0:1],
  input  wire act_y_out_bind_full_n [0:1]
);
  // family mac_a_out_a_in: 16 channel(s), 8-bit, depth 2
  wire [7:0] mac_a_out_a_in_din [0:15];
  wire [7:0] mac_a_out_a_in_dout [0:15];
  wire mac_a_out_a_in_full_n [0:15];
  wire mac_a_out_a_in_write [0:15];
  wire mac_a_out_a_in_empty_n [0:15];
  wire mac_a_out_a_in_read [0:15];
  genvar mac_a_out_a_in_i;
  generate
    for (mac_a_out_a_in_i = 0; mac_a_out_a_in_i < 16; mac_a_out_a_in_i = mac_a_out_a_in_i + 1) begin : g_mac_a_out_a_in
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(mac_a_out_a_in_din[mac_a_out_a_in_i]), .full_n(mac_a_out_a_in_full_n[mac_a_out_a_in_i]), .write(mac_a_out_a_in_write[mac_a_out_a_in_i]), .dout(mac_a_out_a_in_dout[mac_a_out_a_in_i]), .empty_n(mac_a_out_a_in_empty_n[mac_a_out_a_in_i]), .read(mac_a_out_a_in_read[mac_a_out_a_in_i]));
    end
  endgenerate
  // family mac_p_out_p_in: 14 channel(s), 32-bit, depth 2
  wire [31:0] mac_p_out_p_in_din [0:13];
  wire [31:0] mac_p_out_p_in_dout [0:13];
  wire mac_p_out_p_in_full_n [0:13];
  wire mac_p_out_p_in_write [0:13];
  wire mac_p_out_p_in_empty_n [0:13];
  wire mac_p_out_p_in_read [0:13];
  genvar mac_p_out_p_in_i;
  generate
    for (mac_p_out_p_in_i = 0; mac_p_out_p_in_i < 14; mac_p_out_p_in_i = mac_p_out_p_in_i + 1) begin : g_mac_p_out_p_in
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(mac_p_out_p_in_din[mac_p_out_p_in_i]), .full_n(mac_p_out_p_in_full_n[mac_p_out_p_in_i]), .write(mac_p_out_p_in_write[mac_p_out_p_in_i]), .dout(mac_p_out_p_in_dout[mac_p_out_p_in_i]), .empty_n(mac_p_out_p_in_empty_n[mac_p_out_p_in_i]), .read(mac_p_out_p_in_read[mac_p_out_p_in_i]));
    end
  endgenerate
  // family act_z_in_bind: 2 channel(s), 32-bit, depth 2
  wire [31:0] act_z_in_bind_din [0:1];
  wire [31:0] act_z_in_bind_dout [0:1];
  wire act_z_in_bind_full_n [0:1];
  wire act_z_in_bind_write [0:1];
  wire act_z_in_bind_empty_n [0:1];
  wire act_z_in_bind_read [0:1];
  genvar act_z_in_bind_i;
  generate
    for (act_z_in_bind_i = 0; act_z_in_bind_i < 2; act_z_in_bind_i = act_z_in_bind_i + 1) begin : g_act_z_in_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(act_z_in_bind_din[act_z_in_bind_i]), .full_n(act_z_in_bind_full_n[act_z_in_bind_i]), .write(act_z_in_bind_write[act_z_in_bind_i]), .dout(act_z_in_bind_dout[act_z_in_bind_i]), .empty_n(act_z_in_bind_empty_n[act_z_in_bind_i]), .read(act_z_in_bind_read[act_z_in_bind_i]));
    end
  endgenerate
  // role mac_r0: 6 instance(s)
  mac_r0 u_mac_r0_0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[3]),
      .w_empty_n(mac_w_mem_empty_n[3]),
      .w_read(mac_w_mem_read[3]),
      .a_in_dout(mac_a_out_a_in_dout[3]),
      .a_in_empty_n(mac_a_out_a_in_empty_n[3]),
      .a_in_read(mac_a_out_a_in_read[3]),
      .p_in_dout(mac_p_out_p_in_dout[13]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[13]),
      .p_in_read(mac_p_out_p_in_read[13]),
      .p_out_din(mac_p_out_p_in_din[3]),
      .p_out_full_n(mac_p_out_p_in_full_n[3]),
      .p_out_write(mac_p_out_p_in_write[3]));
  mac_r0 u_mac_r0_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[5]),
      .w_empty_n(mac_w_mem_empty_n[5]),
      .w_read(mac_w_mem_read[5]),
      .a_in_dout(mac_a_out_a_in_dout[5]),
      .a_in_empty_n(mac_a_out_a_in_empty_n[5]),
      .a_in_read(mac_a_out_a_in_read[5]),
      .p_in_dout(mac_p_out_p_in_dout[1]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[1]),
      .p_in_read(mac_p_out_p_in_read[1]),
      .p_out_din(mac_p_out_p_in_din[5]),
      .p_out_full_n(mac_p_out_p_in_full_n[5]),
      .p_out_write(mac_p_out_p_in_write[5]));
  mac_r0 u_mac_r0_1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[7]),
      .w_empty_n(mac_w_mem_empty_n[7]),
      .w_read(mac_w_mem_read[7]),
      .a_in_dout(mac_a_out_a_in_dout[7]),
      .a_in_empty_n(mac_a_out_a_in_empty_n[7]),
      .a_in_read(mac_a_out_a_in_read[7]),
      .p_in_dout(mac_p_out_p_in_dout[3]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[3]),
      .p_in_read(mac_p_out_p_in_read[3]),
      .p_out_din(mac_p_out_p_in_din[7]),
      .p_out_full_n(mac_p_out_p_in_full_n[7]),
      .p_out_write(mac_p_out_p_in_write[7]));
  mac_r0 u_mac_r0_2_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[9]),
      .w_empty_n(mac_w_mem_empty_n[9]),
      .w_read(mac_w_mem_read[9]),
      .a_in_dout(mac_a_out_a_in_dout[9]),
      .a_in_empty_n(mac_a_out_a_in_empty_n[9]),
      .a_in_read(mac_a_out_a_in_read[9]),
      .p_in_dout(mac_p_out_p_in_dout[5]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[5]),
      .p_in_read(mac_p_out_p_in_read[5]),
      .p_out_din(mac_p_out_p_in_din[9]),
      .p_out_full_n(mac_p_out_p_in_full_n[9]),
      .p_out_write(mac_p_out_p_in_write[9]));
  mac_r0 u_mac_r0_2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[11]),
      .w_empty_n(mac_w_mem_empty_n[11]),
      .w_read(mac_w_mem_read[11]),
      .a_in_dout(mac_a_out_a_in_dout[11]),
      .a_in_empty_n(mac_a_out_a_in_empty_n[11]),
      .a_in_read(mac_a_out_a_in_read[11]),
      .p_in_dout(mac_p_out_p_in_dout[7]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[7]),
      .p_in_read(mac_p_out_p_in_read[7]),
      .p_out_din(mac_p_out_p_in_din[11]),
      .p_out_full_n(mac_p_out_p_in_full_n[11]),
      .p_out_write(mac_p_out_p_in_write[11]));
  mac_r0 u_mac_r0_3_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[13]),
      .w_empty_n(mac_w_mem_empty_n[13]),
      .w_read(mac_w_mem_read[13]),
      .a_in_dout(mac_a_out_a_in_dout[13]),
      .a_in_empty_n(mac_a_out_a_in_empty_n[13]),
      .a_in_read(mac_a_out_a_in_read[13]),
      .p_in_dout(mac_p_out_p_in_dout[9]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[9]),
      .p_in_read(mac_p_out_p_in_read[9]),
      .p_out_din(mac_p_out_p_in_din[13]),
      .p_out_full_n(mac_p_out_p_in_full_n[13]),
      .p_out_write(mac_p_out_p_in_write[13]));
  // role mac_r1: 6 instance(s)
  mac_r1 u_mac_r1_0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[2]),
      .w_empty_n(mac_w_mem_empty_n[2]),
      .w_read(mac_w_mem_read[2]),
      .a_in_dout(mac_a_in_bind_dout[1]),
      .a_in_empty_n(mac_a_in_bind_empty_n[1]),
      .a_in_read(mac_a_in_bind_read[1]),
      .a_out_din(mac_a_out_a_in_din[3]),
      .a_out_full_n(mac_a_out_a_in_full_n[3]),
      .a_out_write(mac_a_out_a_in_write[3]),
      .p_in_dout(mac_p_out_p_in_dout[12]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[12]),
      .p_in_read(mac_p_out_p_in_read[12]),
      .p_out_din(mac_p_out_p_in_din[2]),
      .p_out_full_n(mac_p_out_p_in_full_n[2]),
      .p_out_write(mac_p_out_p_in_write[2]));
  mac_r1 u_mac_r1_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[4]),
      .w_empty_n(mac_w_mem_empty_n[4]),
      .w_read(mac_w_mem_read[4]),
      .a_in_dout(mac_a_in_bind_dout[2]),
      .a_in_empty_n(mac_a_in_bind_empty_n[2]),
      .a_in_read(mac_a_in_bind_read[2]),
      .a_out_din(mac_a_out_a_in_din[5]),
      .a_out_full_n(mac_a_out_a_in_full_n[5]),
      .a_out_write(mac_a_out_a_in_write[5]),
      .p_in_dout(mac_p_out_p_in_dout[0]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[0]),
      .p_in_read(mac_p_out_p_in_read[0]),
      .p_out_din(mac_p_out_p_in_din[4]),
      .p_out_full_n(mac_p_out_p_in_full_n[4]),
      .p_out_write(mac_p_out_p_in_write[4]));
  mac_r1 u_mac_r1_1_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[6]),
      .w_empty_n(mac_w_mem_empty_n[6]),
      .w_read(mac_w_mem_read[6]),
      .a_in_dout(mac_a_in_bind_dout[3]),
      .a_in_empty_n(mac_a_in_bind_empty_n[3]),
      .a_in_read(mac_a_in_bind_read[3]),
      .a_out_din(mac_a_out_a_in_din[7]),
      .a_out_full_n(mac_a_out_a_in_full_n[7]),
      .a_out_write(mac_a_out_a_in_write[7]),
      .p_in_dout(mac_p_out_p_in_dout[2]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[2]),
      .p_in_read(mac_p_out_p_in_read[2]),
      .p_out_din(mac_p_out_p_in_din[6]),
      .p_out_full_n(mac_p_out_p_in_full_n[6]),
      .p_out_write(mac_p_out_p_in_write[6]));
  mac_r1 u_mac_r1_2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[8]),
      .w_empty_n(mac_w_mem_empty_n[8]),
      .w_read(mac_w_mem_read[8]),
      .a_in_dout(mac_a_in_bind_dout[4]),
      .a_in_empty_n(mac_a_in_bind_empty_n[4]),
      .a_in_read(mac_a_in_bind_read[4]),
      .a_out_din(mac_a_out_a_in_din[9]),
      .a_out_full_n(mac_a_out_a_in_full_n[9]),
      .a_out_write(mac_a_out_a_in_write[9]),
      .p_in_dout(mac_p_out_p_in_dout[4]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[4]),
      .p_in_read(mac_p_out_p_in_read[4]),
      .p_out_din(mac_p_out_p_in_din[8]),
      .p_out_full_n(mac_p_out_p_in_full_n[8]),
      .p_out_write(mac_p_out_p_in_write[8]));
  mac_r1 u_mac_r1_2_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[10]),
      .w_empty_n(mac_w_mem_empty_n[10]),
      .w_read(mac_w_mem_read[10]),
      .a_in_dout(mac_a_in_bind_dout[5]),
      .a_in_empty_n(mac_a_in_bind_empty_n[5]),
      .a_in_read(mac_a_in_bind_read[5]),
      .a_out_din(mac_a_out_a_in_din[11]),
      .a_out_full_n(mac_a_out_a_in_full_n[11]),
      .a_out_write(mac_a_out_a_in_write[11]),
      .p_in_dout(mac_p_out_p_in_dout[6]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[6]),
      .p_in_read(mac_p_out_p_in_read[6]),
      .p_out_din(mac_p_out_p_in_din[10]),
      .p_out_full_n(mac_p_out_p_in_full_n[10]),
      .p_out_write(mac_p_out_p_in_write[10]));
  mac_r1 u_mac_r1_3_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[12]),
      .w_empty_n(mac_w_mem_empty_n[12]),
      .w_read(mac_w_mem_read[12]),
      .a_in_dout(mac_a_in_bind_dout[6]),
      .a_in_empty_n(mac_a_in_bind_empty_n[6]),
      .a_in_read(mac_a_in_bind_read[6]),
      .a_out_din(mac_a_out_a_in_din[13]),
      .a_out_full_n(mac_a_out_a_in_full_n[13]),
      .a_out_write(mac_a_out_a_in_write[13]),
      .p_in_dout(mac_p_out_p_in_dout[8]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[8]),
      .p_in_read(mac_p_out_p_in_read[8]),
      .p_out_din(mac_p_out_p_in_din[12]),
      .p_out_full_n(mac_p_out_p_in_full_n[12]),
      .p_out_write(mac_p_out_p_in_write[12]));
  // role mac_r2: 1 instance(s)
  mac_r2 u_mac_r2_3_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[15]),
      .w_empty_n(mac_w_mem_empty_n[15]),
      .w_read(mac_w_mem_read[15]),
      .a_in_dout(mac_a_out_a_in_dout[15]),
      .a_in_empty_n(mac_a_out_a_in_empty_n[15]),
      .a_in_read(mac_a_out_a_in_read[15]),
      .p_in_dout(mac_p_out_p_in_dout[11]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[11]),
      .p_in_read(mac_p_out_p_in_read[11]),
      .p_out_din(act_z_in_bind_din[1]),
      .p_out_full_n(act_z_in_bind_full_n[1]),
      .p_out_write(act_z_in_bind_write[1]));
  // role mac_r3: 1 instance(s)
  mac_r3 u_mac_r3_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[1]),
      .w_empty_n(mac_w_mem_empty_n[1]),
      .w_read(mac_w_mem_read[1]),
      .a_in_dout(mac_a_out_a_in_dout[1]),
      .a_in_empty_n(mac_a_out_a_in_empty_n[1]),
      .a_in_read(mac_a_out_a_in_read[1]),
      .p_out_din(mac_p_out_p_in_din[1]),
      .p_out_full_n(mac_p_out_p_in_full_n[1]),
      .p_out_write(mac_p_out_p_in_write[1]));
  // role mac_r4: 1 instance(s)
  mac_r4 u_mac_r4_3_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[14]),
      .w_empty_n(mac_w_mem_empty_n[14]),
      .w_read(mac_w_mem_read[14]),
      .a_in_dout(mac_a_in_bind_dout[7]),
      .a_in_empty_n(mac_a_in_bind_empty_n[7]),
      .a_in_read(mac_a_in_bind_read[7]),
      .a_out_din(mac_a_out_a_in_din[15]),
      .a_out_full_n(mac_a_out_a_in_full_n[15]),
      .a_out_write(mac_a_out_a_in_write[15]),
      .p_in_dout(mac_p_out_p_in_dout[10]),
      .p_in_empty_n(mac_p_out_p_in_empty_n[10]),
      .p_in_read(mac_p_out_p_in_read[10]),
      .p_out_din(act_z_in_bind_din[0]),
      .p_out_full_n(act_z_in_bind_full_n[0]),
      .p_out_write(act_z_in_bind_write[0]));
  // role mac_r5: 1 instance(s)
  mac_r5 u_mac_r5_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .w_dout(mac_w_mem_dout[0]),
      .w_empty_n(mac_w_mem_empty_n[0]),
      .w_read(mac_w_mem_read[0]),
      .a_in_dout(mac_a_in_bind_dout[0]),
      .a_in_empty_n(mac_a_in_bind_empty_n[0]),
      .a_in_read(mac_a_in_bind_read[0]),
      .a_out_din(mac_a_out_a_in_din[1]),
      .a_out_full_n(mac_a_out_a_in_full_n[1]),
      .a_out_write(mac_a_out_a_in_write[1]),
      .p_out_din(mac_p_out_p_in_din[0]),
      .p_out_full_n(mac_p_out_p_in_full_n[0]),
      .p_out_write(mac_p_out_p_in_write[0]));
  // role act_r0: 2 instance(s)
  act_r0 u_act_r0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .y_out_din(act_y_out_bind_din[0]),
      .y_out_full_n(act_y_out_bind_full_n[0]),
      .y_out_write(act_y_out_bind_write[0]),
      .z_in_dout(act_z_in_bind_dout[0]),
      .z_in_empty_n(act_z_in_bind_empty_n[0]),
      .z_in_read(act_z_in_bind_read[0]));
  act_r0 u_act_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .y_out_din(act_y_out_bind_din[1]),
      .y_out_full_n(act_y_out_bind_full_n[1]),
      .y_out_write(act_y_out_bind_write[1]),
      .z_in_dout(act_z_in_bind_dout[1]),
      .z_in_empty_n(act_z_in_bind_empty_n[1]),
      .z_in_read(act_z_in_bind_read[1]));
endmodule
