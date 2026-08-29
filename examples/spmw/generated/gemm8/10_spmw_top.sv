`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [7:0] pe_west_bind_dout [0:2],
  input  wire pe_west_bind_empty_n [0:2],
  output wire pe_west_bind_read [0:2],
  input  wire [7:0] pe_north_bind_dout [0:2],
  input  wire pe_north_bind_empty_n [0:2],
  output wire pe_north_bind_read [0:2],
  output wire [31:0] pe_c_mem_din [0:8],
  output wire pe_c_mem_write [0:8],
  input  wire pe_c_mem_full_n [0:8]
);
  // family pe_east_west: 9 channel(s), 8-bit, depth 2
  wire [7:0] pe_east_west_din [0:8];
  wire [7:0] pe_east_west_dout [0:8];
  wire pe_east_west_full_n [0:8];
  wire pe_east_west_write [0:8];
  wire pe_east_west_empty_n [0:8];
  wire pe_east_west_read [0:8];
  genvar pe_east_west_i;
  generate
    for (pe_east_west_i = 0; pe_east_west_i < 9; pe_east_west_i = pe_east_west_i + 1) begin : g_pe_east_west
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_east_west_din[pe_east_west_i]), .full_n(pe_east_west_full_n[pe_east_west_i]), .write(pe_east_west_write[pe_east_west_i]), .dout(pe_east_west_dout[pe_east_west_i]), .empty_n(pe_east_west_empty_n[pe_east_west_i]), .read(pe_east_west_read[pe_east_west_i]));
    end
  endgenerate
  // family pe_south_north: 9 channel(s), 8-bit, depth 2
  wire [7:0] pe_south_north_din [0:8];
  wire [7:0] pe_south_north_dout [0:8];
  wire pe_south_north_full_n [0:8];
  wire pe_south_north_write [0:8];
  wire pe_south_north_empty_n [0:8];
  wire pe_south_north_read [0:8];
  genvar pe_south_north_i;
  generate
    for (pe_south_north_i = 0; pe_south_north_i < 9; pe_south_north_i = pe_south_north_i + 1) begin : g_pe_south_north
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_south_north_din[pe_south_north_i]), .full_n(pe_south_north_full_n[pe_south_north_i]), .write(pe_south_north_write[pe_south_north_i]), .dout(pe_south_north_dout[pe_south_north_i]), .empty_n(pe_south_north_empty_n[pe_south_north_i]), .read(pe_south_north_read[pe_south_north_i]));
    end
  endgenerate
  // role pe_r0: 1 instance(s)
  pe_r0 u_pe_r0_2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[6]),
      .c_full_n(pe_c_mem_full_n[6]),
      .c_write(pe_c_mem_write[6]),
      .east_din(pe_east_west_din[7]),
      .east_full_n(pe_east_west_full_n[7]),
      .east_write(pe_east_west_write[7]),
      .north_dout(pe_south_north_dout[6]),
      .north_empty_n(pe_south_north_empty_n[6]),
      .north_read(pe_south_north_read[6]),
      .west_dout(pe_west_bind_dout[2]),
      .west_empty_n(pe_west_bind_empty_n[2]),
      .west_read(pe_west_bind_read[2]));
  // role pe_r1: 1 instance(s)
  pe_r1 u_pe_r1_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[3]),
      .c_full_n(pe_c_mem_full_n[3]),
      .c_write(pe_c_mem_write[3]),
      .east_din(pe_east_west_din[4]),
      .east_full_n(pe_east_west_full_n[4]),
      .east_write(pe_east_west_write[4]),
      .north_dout(pe_south_north_dout[3]),
      .north_empty_n(pe_south_north_empty_n[3]),
      .north_read(pe_south_north_read[3]),
      .south_din(pe_south_north_din[6]),
      .south_full_n(pe_south_north_full_n[6]),
      .south_write(pe_south_north_write[6]),
      .west_dout(pe_west_bind_dout[1]),
      .west_empty_n(pe_west_bind_empty_n[1]),
      .west_read(pe_west_bind_read[1]));
  // role pe_r2: 1 instance(s)
  pe_r2 u_pe_r2_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[4]),
      .c_full_n(pe_c_mem_full_n[4]),
      .c_write(pe_c_mem_write[4]),
      .east_din(pe_east_west_din[5]),
      .east_full_n(pe_east_west_full_n[5]),
      .east_write(pe_east_west_write[5]),
      .north_dout(pe_south_north_dout[4]),
      .north_empty_n(pe_south_north_empty_n[4]),
      .north_read(pe_south_north_read[4]),
      .south_din(pe_south_north_din[7]),
      .south_full_n(pe_south_north_full_n[7]),
      .south_write(pe_south_north_write[7]),
      .west_dout(pe_east_west_dout[4]),
      .west_empty_n(pe_east_west_empty_n[4]),
      .west_read(pe_east_west_read[4]));
  // role pe_r3: 1 instance(s)
  pe_r3 u_pe_r3_2_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[7]),
      .c_full_n(pe_c_mem_full_n[7]),
      .c_write(pe_c_mem_write[7]),
      .east_din(pe_east_west_din[8]),
      .east_full_n(pe_east_west_full_n[8]),
      .east_write(pe_east_west_write[8]),
      .north_dout(pe_south_north_dout[7]),
      .north_empty_n(pe_south_north_empty_n[7]),
      .north_read(pe_south_north_read[7]),
      .west_dout(pe_east_west_dout[7]),
      .west_empty_n(pe_east_west_empty_n[7]),
      .west_read(pe_east_west_read[7]));
  // role pe_r4: 1 instance(s)
  pe_r4 u_pe_r4_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[0]),
      .c_full_n(pe_c_mem_full_n[0]),
      .c_write(pe_c_mem_write[0]),
      .east_din(pe_east_west_din[1]),
      .east_full_n(pe_east_west_full_n[1]),
      .east_write(pe_east_west_write[1]),
      .north_dout(pe_north_bind_dout[0]),
      .north_empty_n(pe_north_bind_empty_n[0]),
      .north_read(pe_north_bind_read[0]),
      .south_din(pe_south_north_din[3]),
      .south_full_n(pe_south_north_full_n[3]),
      .south_write(pe_south_north_write[3]),
      .west_dout(pe_west_bind_dout[0]),
      .west_empty_n(pe_west_bind_empty_n[0]),
      .west_read(pe_west_bind_read[0]));
  // role pe_r5: 1 instance(s)
  pe_r5 u_pe_r5_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[1]),
      .c_full_n(pe_c_mem_full_n[1]),
      .c_write(pe_c_mem_write[1]),
      .east_din(pe_east_west_din[2]),
      .east_full_n(pe_east_west_full_n[2]),
      .east_write(pe_east_west_write[2]),
      .north_dout(pe_north_bind_dout[1]),
      .north_empty_n(pe_north_bind_empty_n[1]),
      .north_read(pe_north_bind_read[1]),
      .south_din(pe_south_north_din[4]),
      .south_full_n(pe_south_north_full_n[4]),
      .south_write(pe_south_north_write[4]),
      .west_dout(pe_east_west_dout[1]),
      .west_empty_n(pe_east_west_empty_n[1]),
      .west_read(pe_east_west_read[1]));
  // role pe_r6: 1 instance(s)
  pe_r6 u_pe_r6_1_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[5]),
      .c_full_n(pe_c_mem_full_n[5]),
      .c_write(pe_c_mem_write[5]),
      .north_dout(pe_south_north_dout[5]),
      .north_empty_n(pe_south_north_empty_n[5]),
      .north_read(pe_south_north_read[5]),
      .south_din(pe_south_north_din[8]),
      .south_full_n(pe_south_north_full_n[8]),
      .south_write(pe_south_north_write[8]),
      .west_dout(pe_east_west_dout[5]),
      .west_empty_n(pe_east_west_empty_n[5]),
      .west_read(pe_east_west_read[5]));
  // role pe_r7: 1 instance(s)
  pe_r7 u_pe_r7_2_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[8]),
      .c_full_n(pe_c_mem_full_n[8]),
      .c_write(pe_c_mem_write[8]),
      .north_dout(pe_south_north_dout[8]),
      .north_empty_n(pe_south_north_empty_n[8]),
      .north_read(pe_south_north_read[8]),
      .west_dout(pe_east_west_dout[8]),
      .west_empty_n(pe_east_west_empty_n[8]),
      .west_read(pe_east_west_read[8]));
  // role pe_r8: 1 instance(s)
  pe_r8 u_pe_r8_0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[2]),
      .c_full_n(pe_c_mem_full_n[2]),
      .c_write(pe_c_mem_write[2]),
      .north_dout(pe_north_bind_dout[2]),
      .north_empty_n(pe_north_bind_empty_n[2]),
      .north_read(pe_north_bind_read[2]),
      .south_din(pe_south_north_din[5]),
      .south_full_n(pe_south_north_full_n[5]),
      .south_write(pe_south_north_write[5]),
      .west_dout(pe_east_west_dout[2]),
      .west_empty_n(pe_east_west_empty_n[2]),
      .west_read(pe_east_west_read[2]));
endmodule
