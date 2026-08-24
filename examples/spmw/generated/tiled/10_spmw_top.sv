`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] pe_west_bind_dout [0:1],
  input  wire pe_west_bind_empty_n [0:1],
  output wire pe_west_bind_read [0:1],
  input  wire [31:0] pe_north_bind_dout [0:1],
  input  wire pe_north_bind_empty_n [0:1],
  output wire pe_north_bind_read [0:1],
  output wire [31:0] pe_c_mem_din [0:3],
  output wire pe_c_mem_write [0:3],
  input  wire pe_c_mem_full_n [0:3],
  input  wire [31:0] pe_1_west_bind_dout [0:1],
  input  wire pe_1_west_bind_empty_n [0:1],
  output wire pe_1_west_bind_read [0:1],
  input  wire [31:0] pe_1_north_bind_dout [0:1],
  input  wire pe_1_north_bind_empty_n [0:1],
  output wire pe_1_north_bind_read [0:1],
  output wire [31:0] pe_1_c_mem_din [0:3],
  output wire pe_1_c_mem_write [0:3],
  input  wire pe_1_c_mem_full_n [0:3],
  input  wire [31:0] pe_2_west_bind_dout [0:1],
  input  wire pe_2_west_bind_empty_n [0:1],
  output wire pe_2_west_bind_read [0:1],
  input  wire [31:0] pe_2_north_bind_dout [0:1],
  input  wire pe_2_north_bind_empty_n [0:1],
  output wire pe_2_north_bind_read [0:1],
  output wire [31:0] pe_2_c_mem_din [0:3],
  output wire pe_2_c_mem_write [0:3],
  input  wire pe_2_c_mem_full_n [0:3],
  input  wire [31:0] pe_3_west_bind_dout [0:1],
  input  wire pe_3_west_bind_empty_n [0:1],
  output wire pe_3_west_bind_read [0:1],
  input  wire [31:0] pe_3_north_bind_dout [0:1],
  input  wire pe_3_north_bind_empty_n [0:1],
  output wire pe_3_north_bind_read [0:1],
  output wire [31:0] pe_3_c_mem_din [0:3],
  output wire pe_3_c_mem_write [0:3],
  input  wire pe_3_c_mem_full_n [0:3]
);
  // family pe_east_west: 4 channel(s), 32-bit, depth 2
  wire [31:0] pe_east_west_din [0:3];
  wire [31:0] pe_east_west_dout [0:3];
  wire pe_east_west_full_n [0:3];
  wire pe_east_west_write [0:3];
  wire pe_east_west_empty_n [0:3];
  wire pe_east_west_read [0:3];
  genvar pe_east_west_i;
  generate
    for (pe_east_west_i = 0; pe_east_west_i < 4; pe_east_west_i = pe_east_west_i + 1) begin : g_pe_east_west
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_east_west_din[pe_east_west_i]), .full_n(pe_east_west_full_n[pe_east_west_i]), .write(pe_east_west_write[pe_east_west_i]), .dout(pe_east_west_dout[pe_east_west_i]), .empty_n(pe_east_west_empty_n[pe_east_west_i]), .read(pe_east_west_read[pe_east_west_i]));
    end
  endgenerate
  // family pe_south_north: 4 channel(s), 32-bit, depth 2
  wire [31:0] pe_south_north_din [0:3];
  wire [31:0] pe_south_north_dout [0:3];
  wire pe_south_north_full_n [0:3];
  wire pe_south_north_write [0:3];
  wire pe_south_north_empty_n [0:3];
  wire pe_south_north_read [0:3];
  genvar pe_south_north_i;
  generate
    for (pe_south_north_i = 0; pe_south_north_i < 4; pe_south_north_i = pe_south_north_i + 1) begin : g_pe_south_north
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_south_north_din[pe_south_north_i]), .full_n(pe_south_north_full_n[pe_south_north_i]), .write(pe_south_north_write[pe_south_north_i]), .dout(pe_south_north_dout[pe_south_north_i]), .empty_n(pe_south_north_empty_n[pe_south_north_i]), .read(pe_south_north_read[pe_south_north_i]));
    end
  endgenerate
  // family pe_1_east_west: 4 channel(s), 32-bit, depth 2
  wire [31:0] pe_1_east_west_din [0:3];
  wire [31:0] pe_1_east_west_dout [0:3];
  wire pe_1_east_west_full_n [0:3];
  wire pe_1_east_west_write [0:3];
  wire pe_1_east_west_empty_n [0:3];
  wire pe_1_east_west_read [0:3];
  genvar pe_1_east_west_i;
  generate
    for (pe_1_east_west_i = 0; pe_1_east_west_i < 4; pe_1_east_west_i = pe_1_east_west_i + 1) begin : g_pe_1_east_west
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_1_east_west_din[pe_1_east_west_i]), .full_n(pe_1_east_west_full_n[pe_1_east_west_i]), .write(pe_1_east_west_write[pe_1_east_west_i]), .dout(pe_1_east_west_dout[pe_1_east_west_i]), .empty_n(pe_1_east_west_empty_n[pe_1_east_west_i]), .read(pe_1_east_west_read[pe_1_east_west_i]));
    end
  endgenerate
  // family pe_1_south_north: 4 channel(s), 32-bit, depth 2
  wire [31:0] pe_1_south_north_din [0:3];
  wire [31:0] pe_1_south_north_dout [0:3];
  wire pe_1_south_north_full_n [0:3];
  wire pe_1_south_north_write [0:3];
  wire pe_1_south_north_empty_n [0:3];
  wire pe_1_south_north_read [0:3];
  genvar pe_1_south_north_i;
  generate
    for (pe_1_south_north_i = 0; pe_1_south_north_i < 4; pe_1_south_north_i = pe_1_south_north_i + 1) begin : g_pe_1_south_north
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_1_south_north_din[pe_1_south_north_i]), .full_n(pe_1_south_north_full_n[pe_1_south_north_i]), .write(pe_1_south_north_write[pe_1_south_north_i]), .dout(pe_1_south_north_dout[pe_1_south_north_i]), .empty_n(pe_1_south_north_empty_n[pe_1_south_north_i]), .read(pe_1_south_north_read[pe_1_south_north_i]));
    end
  endgenerate
  // family pe_2_east_west: 4 channel(s), 32-bit, depth 2
  wire [31:0] pe_2_east_west_din [0:3];
  wire [31:0] pe_2_east_west_dout [0:3];
  wire pe_2_east_west_full_n [0:3];
  wire pe_2_east_west_write [0:3];
  wire pe_2_east_west_empty_n [0:3];
  wire pe_2_east_west_read [0:3];
  genvar pe_2_east_west_i;
  generate
    for (pe_2_east_west_i = 0; pe_2_east_west_i < 4; pe_2_east_west_i = pe_2_east_west_i + 1) begin : g_pe_2_east_west
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_2_east_west_din[pe_2_east_west_i]), .full_n(pe_2_east_west_full_n[pe_2_east_west_i]), .write(pe_2_east_west_write[pe_2_east_west_i]), .dout(pe_2_east_west_dout[pe_2_east_west_i]), .empty_n(pe_2_east_west_empty_n[pe_2_east_west_i]), .read(pe_2_east_west_read[pe_2_east_west_i]));
    end
  endgenerate
  // family pe_2_south_north: 4 channel(s), 32-bit, depth 2
  wire [31:0] pe_2_south_north_din [0:3];
  wire [31:0] pe_2_south_north_dout [0:3];
  wire pe_2_south_north_full_n [0:3];
  wire pe_2_south_north_write [0:3];
  wire pe_2_south_north_empty_n [0:3];
  wire pe_2_south_north_read [0:3];
  genvar pe_2_south_north_i;
  generate
    for (pe_2_south_north_i = 0; pe_2_south_north_i < 4; pe_2_south_north_i = pe_2_south_north_i + 1) begin : g_pe_2_south_north
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_2_south_north_din[pe_2_south_north_i]), .full_n(pe_2_south_north_full_n[pe_2_south_north_i]), .write(pe_2_south_north_write[pe_2_south_north_i]), .dout(pe_2_south_north_dout[pe_2_south_north_i]), .empty_n(pe_2_south_north_empty_n[pe_2_south_north_i]), .read(pe_2_south_north_read[pe_2_south_north_i]));
    end
  endgenerate
  // family pe_3_east_west: 4 channel(s), 32-bit, depth 2
  wire [31:0] pe_3_east_west_din [0:3];
  wire [31:0] pe_3_east_west_dout [0:3];
  wire pe_3_east_west_full_n [0:3];
  wire pe_3_east_west_write [0:3];
  wire pe_3_east_west_empty_n [0:3];
  wire pe_3_east_west_read [0:3];
  genvar pe_3_east_west_i;
  generate
    for (pe_3_east_west_i = 0; pe_3_east_west_i < 4; pe_3_east_west_i = pe_3_east_west_i + 1) begin : g_pe_3_east_west
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_3_east_west_din[pe_3_east_west_i]), .full_n(pe_3_east_west_full_n[pe_3_east_west_i]), .write(pe_3_east_west_write[pe_3_east_west_i]), .dout(pe_3_east_west_dout[pe_3_east_west_i]), .empty_n(pe_3_east_west_empty_n[pe_3_east_west_i]), .read(pe_3_east_west_read[pe_3_east_west_i]));
    end
  endgenerate
  // family pe_3_south_north: 4 channel(s), 32-bit, depth 2
  wire [31:0] pe_3_south_north_din [0:3];
  wire [31:0] pe_3_south_north_dout [0:3];
  wire pe_3_south_north_full_n [0:3];
  wire pe_3_south_north_write [0:3];
  wire pe_3_south_north_empty_n [0:3];
  wire pe_3_south_north_read [0:3];
  genvar pe_3_south_north_i;
  generate
    for (pe_3_south_north_i = 0; pe_3_south_north_i < 4; pe_3_south_north_i = pe_3_south_north_i + 1) begin : g_pe_3_south_north
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_3_south_north_din[pe_3_south_north_i]), .full_n(pe_3_south_north_full_n[pe_3_south_north_i]), .write(pe_3_south_north_write[pe_3_south_north_i]), .dout(pe_3_south_north_dout[pe_3_south_north_i]), .empty_n(pe_3_south_north_empty_n[pe_3_south_north_i]), .read(pe_3_south_north_read[pe_3_south_north_i]));
    end
  endgenerate
  // role pe_r0: 1 instance(s)
  pe_r0 u_pe_r0_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[2]),
      .c_full_n(pe_c_mem_full_n[2]),
      .c_write(pe_c_mem_write[2]),
      .east_din(pe_east_west_din[3]),
      .east_full_n(pe_east_west_full_n[3]),
      .east_write(pe_east_west_write[3]),
      .north_dout(pe_south_north_dout[2]),
      .north_empty_n(pe_south_north_empty_n[2]),
      .north_read(pe_south_north_read[2]),
      .west_dout(pe_west_bind_dout[1]),
      .west_empty_n(pe_west_bind_empty_n[1]),
      .west_read(pe_west_bind_read[1]));
  // role pe_r1: 1 instance(s)
  pe_r1 u_pe_r1_0_0 (
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
      .south_din(pe_south_north_din[2]),
      .south_full_n(pe_south_north_full_n[2]),
      .south_write(pe_south_north_write[2]),
      .west_dout(pe_west_bind_dout[0]),
      .west_empty_n(pe_west_bind_empty_n[0]),
      .west_read(pe_west_bind_read[0]));
  // role pe_r2: 1 instance(s)
  pe_r2 u_pe_r2_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[3]),
      .c_full_n(pe_c_mem_full_n[3]),
      .c_write(pe_c_mem_write[3]),
      .north_dout(pe_south_north_dout[3]),
      .north_empty_n(pe_south_north_empty_n[3]),
      .north_read(pe_south_north_read[3]),
      .west_dout(pe_east_west_dout[3]),
      .west_empty_n(pe_east_west_empty_n[3]),
      .west_read(pe_east_west_read[3]));
  // role pe_r3: 1 instance(s)
  pe_r3 u_pe_r3_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_c_mem_din[1]),
      .c_full_n(pe_c_mem_full_n[1]),
      .c_write(pe_c_mem_write[1]),
      .north_dout(pe_north_bind_dout[1]),
      .north_empty_n(pe_north_bind_empty_n[1]),
      .north_read(pe_north_bind_read[1]),
      .south_din(pe_south_north_din[3]),
      .south_full_n(pe_south_north_full_n[3]),
      .south_write(pe_south_north_write[3]),
      .west_dout(pe_east_west_dout[1]),
      .west_empty_n(pe_east_west_empty_n[1]),
      .west_read(pe_east_west_read[1]));
  // role pe_1_r0: 1 instance(s)
  pe_1_r0 u_pe_1_r0_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_1_c_mem_din[2]),
      .c_full_n(pe_1_c_mem_full_n[2]),
      .c_write(pe_1_c_mem_write[2]),
      .east_din(pe_1_east_west_din[3]),
      .east_full_n(pe_1_east_west_full_n[3]),
      .east_write(pe_1_east_west_write[3]),
      .north_dout(pe_1_south_north_dout[2]),
      .north_empty_n(pe_1_south_north_empty_n[2]),
      .north_read(pe_1_south_north_read[2]),
      .west_dout(pe_1_west_bind_dout[1]),
      .west_empty_n(pe_1_west_bind_empty_n[1]),
      .west_read(pe_1_west_bind_read[1]));
  // role pe_1_r1: 1 instance(s)
  pe_1_r1 u_pe_1_r1_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_1_c_mem_din[0]),
      .c_full_n(pe_1_c_mem_full_n[0]),
      .c_write(pe_1_c_mem_write[0]),
      .east_din(pe_1_east_west_din[1]),
      .east_full_n(pe_1_east_west_full_n[1]),
      .east_write(pe_1_east_west_write[1]),
      .north_dout(pe_1_north_bind_dout[0]),
      .north_empty_n(pe_1_north_bind_empty_n[0]),
      .north_read(pe_1_north_bind_read[0]),
      .south_din(pe_1_south_north_din[2]),
      .south_full_n(pe_1_south_north_full_n[2]),
      .south_write(pe_1_south_north_write[2]),
      .west_dout(pe_1_west_bind_dout[0]),
      .west_empty_n(pe_1_west_bind_empty_n[0]),
      .west_read(pe_1_west_bind_read[0]));
  // role pe_1_r2: 1 instance(s)
  pe_1_r2 u_pe_1_r2_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_1_c_mem_din[3]),
      .c_full_n(pe_1_c_mem_full_n[3]),
      .c_write(pe_1_c_mem_write[3]),
      .north_dout(pe_1_south_north_dout[3]),
      .north_empty_n(pe_1_south_north_empty_n[3]),
      .north_read(pe_1_south_north_read[3]),
      .west_dout(pe_1_east_west_dout[3]),
      .west_empty_n(pe_1_east_west_empty_n[3]),
      .west_read(pe_1_east_west_read[3]));
  // role pe_1_r3: 1 instance(s)
  pe_1_r3 u_pe_1_r3_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_1_c_mem_din[1]),
      .c_full_n(pe_1_c_mem_full_n[1]),
      .c_write(pe_1_c_mem_write[1]),
      .north_dout(pe_1_north_bind_dout[1]),
      .north_empty_n(pe_1_north_bind_empty_n[1]),
      .north_read(pe_1_north_bind_read[1]),
      .south_din(pe_1_south_north_din[3]),
      .south_full_n(pe_1_south_north_full_n[3]),
      .south_write(pe_1_south_north_write[3]),
      .west_dout(pe_1_east_west_dout[1]),
      .west_empty_n(pe_1_east_west_empty_n[1]),
      .west_read(pe_1_east_west_read[1]));
  // role pe_2_r0: 1 instance(s)
  pe_2_r0 u_pe_2_r0_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_2_c_mem_din[2]),
      .c_full_n(pe_2_c_mem_full_n[2]),
      .c_write(pe_2_c_mem_write[2]),
      .east_din(pe_2_east_west_din[3]),
      .east_full_n(pe_2_east_west_full_n[3]),
      .east_write(pe_2_east_west_write[3]),
      .north_dout(pe_2_south_north_dout[2]),
      .north_empty_n(pe_2_south_north_empty_n[2]),
      .north_read(pe_2_south_north_read[2]),
      .west_dout(pe_2_west_bind_dout[1]),
      .west_empty_n(pe_2_west_bind_empty_n[1]),
      .west_read(pe_2_west_bind_read[1]));
  // role pe_2_r1: 1 instance(s)
  pe_2_r1 u_pe_2_r1_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_2_c_mem_din[0]),
      .c_full_n(pe_2_c_mem_full_n[0]),
      .c_write(pe_2_c_mem_write[0]),
      .east_din(pe_2_east_west_din[1]),
      .east_full_n(pe_2_east_west_full_n[1]),
      .east_write(pe_2_east_west_write[1]),
      .north_dout(pe_2_north_bind_dout[0]),
      .north_empty_n(pe_2_north_bind_empty_n[0]),
      .north_read(pe_2_north_bind_read[0]),
      .south_din(pe_2_south_north_din[2]),
      .south_full_n(pe_2_south_north_full_n[2]),
      .south_write(pe_2_south_north_write[2]),
      .west_dout(pe_2_west_bind_dout[0]),
      .west_empty_n(pe_2_west_bind_empty_n[0]),
      .west_read(pe_2_west_bind_read[0]));
  // role pe_2_r2: 1 instance(s)
  pe_2_r2 u_pe_2_r2_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_2_c_mem_din[3]),
      .c_full_n(pe_2_c_mem_full_n[3]),
      .c_write(pe_2_c_mem_write[3]),
      .north_dout(pe_2_south_north_dout[3]),
      .north_empty_n(pe_2_south_north_empty_n[3]),
      .north_read(pe_2_south_north_read[3]),
      .west_dout(pe_2_east_west_dout[3]),
      .west_empty_n(pe_2_east_west_empty_n[3]),
      .west_read(pe_2_east_west_read[3]));
  // role pe_2_r3: 1 instance(s)
  pe_2_r3 u_pe_2_r3_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_2_c_mem_din[1]),
      .c_full_n(pe_2_c_mem_full_n[1]),
      .c_write(pe_2_c_mem_write[1]),
      .north_dout(pe_2_north_bind_dout[1]),
      .north_empty_n(pe_2_north_bind_empty_n[1]),
      .north_read(pe_2_north_bind_read[1]),
      .south_din(pe_2_south_north_din[3]),
      .south_full_n(pe_2_south_north_full_n[3]),
      .south_write(pe_2_south_north_write[3]),
      .west_dout(pe_2_east_west_dout[1]),
      .west_empty_n(pe_2_east_west_empty_n[1]),
      .west_read(pe_2_east_west_read[1]));
  // role pe_3_r0: 1 instance(s)
  pe_3_r0 u_pe_3_r0_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_3_c_mem_din[2]),
      .c_full_n(pe_3_c_mem_full_n[2]),
      .c_write(pe_3_c_mem_write[2]),
      .east_din(pe_3_east_west_din[3]),
      .east_full_n(pe_3_east_west_full_n[3]),
      .east_write(pe_3_east_west_write[3]),
      .north_dout(pe_3_south_north_dout[2]),
      .north_empty_n(pe_3_south_north_empty_n[2]),
      .north_read(pe_3_south_north_read[2]),
      .west_dout(pe_3_west_bind_dout[1]),
      .west_empty_n(pe_3_west_bind_empty_n[1]),
      .west_read(pe_3_west_bind_read[1]));
  // role pe_3_r1: 1 instance(s)
  pe_3_r1 u_pe_3_r1_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_3_c_mem_din[0]),
      .c_full_n(pe_3_c_mem_full_n[0]),
      .c_write(pe_3_c_mem_write[0]),
      .east_din(pe_3_east_west_din[1]),
      .east_full_n(pe_3_east_west_full_n[1]),
      .east_write(pe_3_east_west_write[1]),
      .north_dout(pe_3_north_bind_dout[0]),
      .north_empty_n(pe_3_north_bind_empty_n[0]),
      .north_read(pe_3_north_bind_read[0]),
      .south_din(pe_3_south_north_din[2]),
      .south_full_n(pe_3_south_north_full_n[2]),
      .south_write(pe_3_south_north_write[2]),
      .west_dout(pe_3_west_bind_dout[0]),
      .west_empty_n(pe_3_west_bind_empty_n[0]),
      .west_read(pe_3_west_bind_read[0]));
  // role pe_3_r2: 1 instance(s)
  pe_3_r2 u_pe_3_r2_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_3_c_mem_din[3]),
      .c_full_n(pe_3_c_mem_full_n[3]),
      .c_write(pe_3_c_mem_write[3]),
      .north_dout(pe_3_south_north_dout[3]),
      .north_empty_n(pe_3_south_north_empty_n[3]),
      .north_read(pe_3_south_north_read[3]),
      .west_dout(pe_3_east_west_dout[3]),
      .west_empty_n(pe_3_east_west_empty_n[3]),
      .west_read(pe_3_east_west_read[3]));
  // role pe_3_r3: 1 instance(s)
  pe_3_r3 u_pe_3_r3_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(pe_3_c_mem_din[1]),
      .c_full_n(pe_3_c_mem_full_n[1]),
      .c_write(pe_3_c_mem_write[1]),
      .north_dout(pe_3_north_bind_dout[1]),
      .north_empty_n(pe_3_north_bind_empty_n[1]),
      .north_read(pe_3_north_bind_read[1]),
      .south_din(pe_3_south_north_din[3]),
      .south_full_n(pe_3_south_north_full_n[3]),
      .south_write(pe_3_south_north_write[3]),
      .west_dout(pe_3_east_west_dout[1]),
      .west_empty_n(pe_3_east_west_empty_n[1]),
      .west_read(pe_3_east_west_read[1]));
endmodule
