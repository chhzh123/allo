`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [7:0] pe_west_bind_dout [0:3],
  input  wire pe_west_bind_empty_n [0:3],
  output wire pe_west_bind_read [0:3],
  input  wire [7:0] pe_north_bind_dout [0:3],
  input  wire pe_north_bind_empty_n [0:3],
  output wire pe_north_bind_read [0:3],
  output wire [31:0] pe_c_mem_din [0:15],
  output wire pe_c_mem_write [0:15],
  input  wire pe_c_mem_full_n [0:15]
);
  // family pe_east_west: 16 channel(s), 8-bit, depth 2
  wire [7:0] pe_east_west_din [0:15];
  wire [7:0] pe_east_west_dout [0:15];
  wire pe_east_west_full_n [0:15];
  wire pe_east_west_write [0:15];
  wire pe_east_west_empty_n [0:15];
  wire pe_east_west_read [0:15];
  genvar pe_east_west_i;
  generate
    for (pe_east_west_i = 0; pe_east_west_i < 16; pe_east_west_i = pe_east_west_i + 1) begin : g_pe_east_west
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_east_west_din[pe_east_west_i]), .full_n(pe_east_west_full_n[pe_east_west_i]), .write(pe_east_west_write[pe_east_west_i]), .dout(pe_east_west_dout[pe_east_west_i]), .empty_n(pe_east_west_empty_n[pe_east_west_i]), .read(pe_east_west_read[pe_east_west_i]));
    end
  endgenerate
  // family pe_south_north: 16 channel(s), 8-bit, depth 2
  wire [7:0] pe_south_north_din [0:15];
  wire [7:0] pe_south_north_dout [0:15];
  wire pe_south_north_full_n [0:15];
  wire pe_south_north_write [0:15];
  wire pe_south_north_empty_n [0:15];
  wire pe_south_north_read [0:15];
  genvar pe_south_north_i;
  generate
    for (pe_south_north_i = 0; pe_south_north_i < 16; pe_south_north_i = pe_south_north_i + 1) begin : g_pe_south_north
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_south_north_din[pe_south_north_i]), .full_n(pe_south_north_full_n[pe_south_north_i]), .write(pe_south_north_write[pe_south_north_i]), .dout(pe_south_north_dout[pe_south_north_i]), .empty_n(pe_south_north_empty_n[pe_south_north_i]), .read(pe_south_north_read[pe_south_north_i]));
    end
  endgenerate
  // role pe_r0: 4 instance(s)
  pe_r0 u_pe_r0_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[5]),
      .c_full_n(pe_c_mem_full_n[5]),
      .c_write(pe_c_mem_write[5]),
      .east_din(pe_east_west_din[6]),
      .east_full_n(pe_east_west_full_n[6]),
      .east_write(pe_east_west_write[6]),
      .north_dout(pe_south_north_dout[5]),
      .north_empty_n(pe_south_north_empty_n[5]),
      .north_read(pe_south_north_read[5]),
      .south_din(pe_south_north_din[9]),
      .south_full_n(pe_south_north_full_n[9]),
      .south_write(pe_south_north_write[9]),
      .west_dout(pe_east_west_dout[5]),
      .west_empty_n(pe_east_west_empty_n[5]),
      .west_read(pe_east_west_read[5]));
  pe_r0 u_pe_r0_1_2 (
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
      .south_din(pe_south_north_din[10]),
      .south_full_n(pe_south_north_full_n[10]),
      .south_write(pe_south_north_write[10]),
      .west_dout(pe_east_west_dout[6]),
      .west_empty_n(pe_east_west_empty_n[6]),
      .west_read(pe_east_west_read[6]));
  pe_r0 u_pe_r0_2_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[9]),
      .c_full_n(pe_c_mem_full_n[9]),
      .c_write(pe_c_mem_write[9]),
      .east_din(pe_east_west_din[10]),
      .east_full_n(pe_east_west_full_n[10]),
      .east_write(pe_east_west_write[10]),
      .north_dout(pe_south_north_dout[9]),
      .north_empty_n(pe_south_north_empty_n[9]),
      .north_read(pe_south_north_read[9]),
      .south_din(pe_south_north_din[13]),
      .south_full_n(pe_south_north_full_n[13]),
      .south_write(pe_south_north_write[13]),
      .west_dout(pe_east_west_dout[9]),
      .west_empty_n(pe_east_west_empty_n[9]),
      .west_read(pe_east_west_read[9]));
  pe_r0 u_pe_r0_2_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[10]),
      .c_full_n(pe_c_mem_full_n[10]),
      .c_write(pe_c_mem_write[10]),
      .east_din(pe_east_west_din[11]),
      .east_full_n(pe_east_west_full_n[11]),
      .east_write(pe_east_west_write[11]),
      .north_dout(pe_south_north_dout[10]),
      .north_empty_n(pe_south_north_empty_n[10]),
      .north_read(pe_south_north_read[10]),
      .south_din(pe_south_north_din[14]),
      .south_full_n(pe_south_north_full_n[14]),
      .south_write(pe_south_north_write[14]),
      .west_dout(pe_east_west_dout[10]),
      .west_empty_n(pe_east_west_empty_n[10]),
      .west_read(pe_east_west_read[10]));
  // role pe_r1: 2 instance(s)
  pe_r1 u_pe_r1_1_0 (
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
      .south_din(pe_south_north_din[8]),
      .south_full_n(pe_south_north_full_n[8]),
      .south_write(pe_south_north_write[8]),
      .west_dout(pe_west_bind_dout[1]),
      .west_empty_n(pe_west_bind_empty_n[1]),
      .west_read(pe_west_bind_read[1]));
  pe_r1 u_pe_r1_2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[8]),
      .c_full_n(pe_c_mem_full_n[8]),
      .c_write(pe_c_mem_write[8]),
      .east_din(pe_east_west_din[9]),
      .east_full_n(pe_east_west_full_n[9]),
      .east_write(pe_east_west_write[9]),
      .north_dout(pe_south_north_dout[8]),
      .north_empty_n(pe_south_north_empty_n[8]),
      .north_read(pe_south_north_read[8]),
      .south_din(pe_south_north_din[12]),
      .south_full_n(pe_south_north_full_n[12]),
      .south_write(pe_south_north_write[12]),
      .west_dout(pe_west_bind_dout[2]),
      .west_empty_n(pe_west_bind_empty_n[2]),
      .west_read(pe_west_bind_read[2]));
  // role pe_r2: 2 instance(s)
  pe_r2 u_pe_r2_3_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[13]),
      .c_full_n(pe_c_mem_full_n[13]),
      .c_write(pe_c_mem_write[13]),
      .east_din(pe_east_west_din[14]),
      .east_full_n(pe_east_west_full_n[14]),
      .east_write(pe_east_west_write[14]),
      .north_dout(pe_south_north_dout[13]),
      .north_empty_n(pe_south_north_empty_n[13]),
      .north_read(pe_south_north_read[13]),
      .west_dout(pe_east_west_dout[13]),
      .west_empty_n(pe_east_west_empty_n[13]),
      .west_read(pe_east_west_read[13]));
  pe_r2 u_pe_r2_3_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[14]),
      .c_full_n(pe_c_mem_full_n[14]),
      .c_write(pe_c_mem_write[14]),
      .east_din(pe_east_west_din[15]),
      .east_full_n(pe_east_west_full_n[15]),
      .east_write(pe_east_west_write[15]),
      .north_dout(pe_south_north_dout[14]),
      .north_empty_n(pe_south_north_empty_n[14]),
      .north_read(pe_south_north_read[14]),
      .west_dout(pe_east_west_dout[14]),
      .west_empty_n(pe_east_west_empty_n[14]),
      .west_read(pe_east_west_read[14]));
  // role pe_r3: 2 instance(s)
  pe_r3 u_pe_r3_0_1 (
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
      .south_din(pe_south_north_din[5]),
      .south_full_n(pe_south_north_full_n[5]),
      .south_write(pe_south_north_write[5]),
      .west_dout(pe_east_west_dout[1]),
      .west_empty_n(pe_east_west_empty_n[1]),
      .west_read(pe_east_west_read[1]));
  pe_r3 u_pe_r3_0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[2]),
      .c_full_n(pe_c_mem_full_n[2]),
      .c_write(pe_c_mem_write[2]),
      .east_din(pe_east_west_din[3]),
      .east_full_n(pe_east_west_full_n[3]),
      .east_write(pe_east_west_write[3]),
      .north_dout(pe_north_bind_dout[2]),
      .north_empty_n(pe_north_bind_empty_n[2]),
      .north_read(pe_north_bind_read[2]),
      .south_din(pe_south_north_din[6]),
      .south_full_n(pe_south_north_full_n[6]),
      .south_write(pe_south_north_write[6]),
      .west_dout(pe_east_west_dout[2]),
      .west_empty_n(pe_east_west_empty_n[2]),
      .west_read(pe_east_west_read[2]));
  // role pe_r4: 2 instance(s)
  pe_r4 u_pe_r4_1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[7]),
      .c_full_n(pe_c_mem_full_n[7]),
      .c_write(pe_c_mem_write[7]),
      .north_dout(pe_south_north_dout[7]),
      .north_empty_n(pe_south_north_empty_n[7]),
      .north_read(pe_south_north_read[7]),
      .south_din(pe_south_north_din[11]),
      .south_full_n(pe_south_north_full_n[11]),
      .south_write(pe_south_north_write[11]),
      .west_dout(pe_east_west_dout[7]),
      .west_empty_n(pe_east_west_empty_n[7]),
      .west_read(pe_east_west_read[7]));
  pe_r4 u_pe_r4_2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[11]),
      .c_full_n(pe_c_mem_full_n[11]),
      .c_write(pe_c_mem_write[11]),
      .north_dout(pe_south_north_dout[11]),
      .north_empty_n(pe_south_north_empty_n[11]),
      .north_read(pe_south_north_read[11]),
      .south_din(pe_south_north_din[15]),
      .south_full_n(pe_south_north_full_n[15]),
      .south_write(pe_south_north_write[15]),
      .west_dout(pe_east_west_dout[11]),
      .west_empty_n(pe_east_west_empty_n[11]),
      .west_read(pe_east_west_read[11]));
  // role pe_r5: 1 instance(s)
  pe_r5 u_pe_r5_3_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[12]),
      .c_full_n(pe_c_mem_full_n[12]),
      .c_write(pe_c_mem_write[12]),
      .east_din(pe_east_west_din[13]),
      .east_full_n(pe_east_west_full_n[13]),
      .east_write(pe_east_west_write[13]),
      .north_dout(pe_south_north_dout[12]),
      .north_empty_n(pe_south_north_empty_n[12]),
      .north_read(pe_south_north_read[12]),
      .west_dout(pe_west_bind_dout[3]),
      .west_empty_n(pe_west_bind_empty_n[3]),
      .west_read(pe_west_bind_read[3]));
  // role pe_r6: 1 instance(s)
  pe_r6 u_pe_r6_0_0 (
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
      .south_din(pe_south_north_din[4]),
      .south_full_n(pe_south_north_full_n[4]),
      .south_write(pe_south_north_write[4]),
      .west_dout(pe_west_bind_dout[0]),
      .west_empty_n(pe_west_bind_empty_n[0]),
      .west_read(pe_west_bind_read[0]));
  // role pe_r7: 1 instance(s)
  pe_r7 u_pe_r7_3_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[15]),
      .c_full_n(pe_c_mem_full_n[15]),
      .c_write(pe_c_mem_write[15]),
      .north_dout(pe_south_north_dout[15]),
      .north_empty_n(pe_south_north_empty_n[15]),
      .north_read(pe_south_north_read[15]),
      .west_dout(pe_east_west_dout[15]),
      .west_empty_n(pe_east_west_empty_n[15]),
      .west_read(pe_east_west_read[15]));
  // role pe_r8: 1 instance(s)
  pe_r8 u_pe_r8_0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[3]),
      .c_full_n(pe_c_mem_full_n[3]),
      .c_write(pe_c_mem_write[3]),
      .north_dout(pe_north_bind_dout[3]),
      .north_empty_n(pe_north_bind_empty_n[3]),
      .north_read(pe_north_bind_read[3]),
      .south_din(pe_south_north_din[7]),
      .south_full_n(pe_south_north_full_n[7]),
      .south_write(pe_south_north_write[7]),
      .west_dout(pe_east_west_dout[3]),
      .west_empty_n(pe_east_west_empty_n[3]),
      .west_read(pe_east_west_read[3]));
endmodule
