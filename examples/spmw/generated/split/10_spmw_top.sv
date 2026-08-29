`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  output wire [31:0] drain_down_bind_din [0:3],
  output wire drain_down_bind_write [0:3],
  input  wire drain_down_bind_full_n [0:3],
  input  wire [31:0] feed_up_bind_dout [0:0],
  input  wire feed_up_bind_empty_n [0:0],
  output wire feed_up_bind_read [0:0],
  input  wire [31:0] feed_3_up_bind_dout [0:0],
  input  wire feed_3_up_bind_empty_n [0:0],
  output wire feed_3_up_bind_read [0:0]
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
  // family pe_west_bind: 4 channel(s), 8-bit, depth 2
  wire [7:0] pe_west_bind_din [0:3];
  wire [7:0] pe_west_bind_dout [0:3];
  wire pe_west_bind_full_n [0:3];
  wire pe_west_bind_write [0:3];
  wire pe_west_bind_empty_n [0:3];
  wire pe_west_bind_read [0:3];
  genvar pe_west_bind_i;
  generate
    for (pe_west_bind_i = 0; pe_west_bind_i < 4; pe_west_bind_i = pe_west_bind_i + 1) begin : g_pe_west_bind
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_west_bind_din[pe_west_bind_i]), .full_n(pe_west_bind_full_n[pe_west_bind_i]), .write(pe_west_bind_write[pe_west_bind_i]), .dout(pe_west_bind_dout[pe_west_bind_i]), .empty_n(pe_west_bind_empty_n[pe_west_bind_i]), .read(pe_west_bind_read[pe_west_bind_i]));
    end
  endgenerate
  // family pe_north_bind: 4 channel(s), 8-bit, depth 2
  wire [7:0] pe_north_bind_din [0:3];
  wire [7:0] pe_north_bind_dout [0:3];
  wire pe_north_bind_full_n [0:3];
  wire pe_north_bind_write [0:3];
  wire pe_north_bind_empty_n [0:3];
  wire pe_north_bind_read [0:3];
  genvar pe_north_bind_i;
  generate
    for (pe_north_bind_i = 0; pe_north_bind_i < 4; pe_north_bind_i = pe_north_bind_i + 1) begin : g_pe_north_bind
      spmw_fifo #(.DW(8), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(pe_north_bind_din[pe_north_bind_i]), .full_n(pe_north_bind_full_n[pe_north_bind_i]), .write(pe_north_bind_write[pe_north_bind_i]), .dout(pe_north_bind_dout[pe_north_bind_i]), .empty_n(pe_north_bind_empty_n[pe_north_bind_i]), .read(pe_north_bind_read[pe_north_bind_i]));
    end
  endgenerate
  // family drain_mine_bind: 16 channel(s), 32-bit, depth 2
  wire [31:0] drain_mine_bind_din [0:15];
  wire [31:0] drain_mine_bind_dout [0:15];
  wire drain_mine_bind_full_n [0:15];
  wire drain_mine_bind_write [0:15];
  wire drain_mine_bind_empty_n [0:15];
  wire drain_mine_bind_read [0:15];
  genvar drain_mine_bind_i;
  generate
    for (drain_mine_bind_i = 0; drain_mine_bind_i < 16; drain_mine_bind_i = drain_mine_bind_i + 1) begin : g_drain_mine_bind
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(drain_mine_bind_din[drain_mine_bind_i]), .full_n(drain_mine_bind_full_n[drain_mine_bind_i]), .write(drain_mine_bind_write[drain_mine_bind_i]), .dout(drain_mine_bind_dout[drain_mine_bind_i]), .empty_n(drain_mine_bind_empty_n[drain_mine_bind_i]), .read(drain_mine_bind_read[drain_mine_bind_i]));
    end
  endgenerate
  // family drain_down_up: 16 channel(s), 32-bit, depth 2
  wire [31:0] drain_down_up_din [0:15];
  wire [31:0] drain_down_up_dout [0:15];
  wire drain_down_up_full_n [0:15];
  wire drain_down_up_write [0:15];
  wire drain_down_up_empty_n [0:15];
  wire drain_down_up_read [0:15];
  genvar drain_down_up_i;
  generate
    for (drain_down_up_i = 0; drain_down_up_i < 16; drain_down_up_i = drain_down_up_i + 1) begin : g_drain_down_up
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(drain_down_up_din[drain_down_up_i]), .full_n(drain_down_up_full_n[drain_down_up_i]), .write(drain_down_up_write[drain_down_up_i]), .dout(drain_down_up_dout[drain_down_up_i]), .empty_n(drain_down_up_empty_n[drain_down_up_i]), .read(drain_down_up_read[drain_down_up_i]));
    end
  endgenerate
  // family feed_down_up: 4 channel(s), 32-bit, depth 2
  wire [31:0] feed_down_up_din [0:3];
  wire [31:0] feed_down_up_dout [0:3];
  wire feed_down_up_full_n [0:3];
  wire feed_down_up_write [0:3];
  wire feed_down_up_empty_n [0:3];
  wire feed_down_up_read [0:3];
  genvar feed_down_up_i;
  generate
    for (feed_down_up_i = 0; feed_down_up_i < 4; feed_down_up_i = feed_down_up_i + 1) begin : g_feed_down_up
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(feed_down_up_din[feed_down_up_i]), .full_n(feed_down_up_full_n[feed_down_up_i]), .write(feed_down_up_write[feed_down_up_i]), .dout(feed_down_up_dout[feed_down_up_i]), .empty_n(feed_down_up_empty_n[feed_down_up_i]), .read(feed_down_up_read[feed_down_up_i]));
    end
  endgenerate
  // family feed_3_down_up: 4 channel(s), 32-bit, depth 2
  wire [31:0] feed_3_down_up_din [0:3];
  wire [31:0] feed_3_down_up_dout [0:3];
  wire feed_3_down_up_full_n [0:3];
  wire feed_3_down_up_write [0:3];
  wire feed_3_down_up_empty_n [0:3];
  wire feed_3_down_up_read [0:3];
  genvar feed_3_down_up_i;
  generate
    for (feed_3_down_up_i = 0; feed_3_down_up_i < 4; feed_3_down_up_i = feed_3_down_up_i + 1) begin : g_feed_3_down_up
      spmw_fifo #(.DW(32), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(feed_3_down_up_din[feed_3_down_up_i]), .full_n(feed_3_down_up_full_n[feed_3_down_up_i]), .write(feed_3_down_up_write[feed_3_down_up_i]), .dout(feed_3_down_up_dout[feed_3_down_up_i]), .empty_n(feed_3_down_up_empty_n[feed_3_down_up_i]), .read(feed_3_down_up_read[feed_3_down_up_i]));
    end
  endgenerate
  // role pe_r0: 4 instance(s)
  pe_r0 u_pe_r0_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .c_din(drain_mine_bind_din[5]),
      .c_full_n(drain_mine_bind_full_n[5]),
      .c_write(drain_mine_bind_write[5]),
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
      .c_din(drain_mine_bind_din[6]),
      .c_full_n(drain_mine_bind_full_n[6]),
      .c_write(drain_mine_bind_write[6]),
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
      .c_din(drain_mine_bind_din[9]),
      .c_full_n(drain_mine_bind_full_n[9]),
      .c_write(drain_mine_bind_write[9]),
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
      .c_din(drain_mine_bind_din[10]),
      .c_full_n(drain_mine_bind_full_n[10]),
      .c_write(drain_mine_bind_write[10]),
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
      .c_din(drain_mine_bind_din[4]),
      .c_full_n(drain_mine_bind_full_n[4]),
      .c_write(drain_mine_bind_write[4]),
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
      .c_din(drain_mine_bind_din[8]),
      .c_full_n(drain_mine_bind_full_n[8]),
      .c_write(drain_mine_bind_write[8]),
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
      .c_din(drain_mine_bind_din[13]),
      .c_full_n(drain_mine_bind_full_n[13]),
      .c_write(drain_mine_bind_write[13]),
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
      .c_din(drain_mine_bind_din[14]),
      .c_full_n(drain_mine_bind_full_n[14]),
      .c_write(drain_mine_bind_write[14]),
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
      .c_din(drain_mine_bind_din[1]),
      .c_full_n(drain_mine_bind_full_n[1]),
      .c_write(drain_mine_bind_write[1]),
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
      .c_din(drain_mine_bind_din[2]),
      .c_full_n(drain_mine_bind_full_n[2]),
      .c_write(drain_mine_bind_write[2]),
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
      .c_din(drain_mine_bind_din[7]),
      .c_full_n(drain_mine_bind_full_n[7]),
      .c_write(drain_mine_bind_write[7]),
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
      .c_din(drain_mine_bind_din[11]),
      .c_full_n(drain_mine_bind_full_n[11]),
      .c_write(drain_mine_bind_write[11]),
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
      .c_din(drain_mine_bind_din[12]),
      .c_full_n(drain_mine_bind_full_n[12]),
      .c_write(drain_mine_bind_write[12]),
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
      .c_din(drain_mine_bind_din[0]),
      .c_full_n(drain_mine_bind_full_n[0]),
      .c_write(drain_mine_bind_write[0]),
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
      .c_din(drain_mine_bind_din[15]),
      .c_full_n(drain_mine_bind_full_n[15]),
      .c_write(drain_mine_bind_write[15]),
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
      .c_din(drain_mine_bind_din[3]),
      .c_full_n(drain_mine_bind_full_n[3]),
      .c_write(drain_mine_bind_write[3]),
      .north_dout(pe_north_bind_dout[3]),
      .north_empty_n(pe_north_bind_empty_n[3]),
      .north_read(pe_north_bind_read[3]),
      .south_din(pe_south_north_din[7]),
      .south_full_n(pe_south_north_full_n[7]),
      .south_write(pe_south_north_write[7]),
      .west_dout(pe_east_west_dout[3]),
      .west_empty_n(pe_east_west_empty_n[3]),
      .west_read(pe_east_west_read[3]));
  // coordinate axis 0: 16 constant source(s)
  wire [31:0] drain_pid0_dout [0:15];
  wire drain_pid0_empty_n [0:15];
  wire drain_pid0_read [0:15];
  spmw_const #(.DW(32), .VAL(0)) u_drain_pid0_0 (.dout(drain_pid0_dout[0]), .empty_n(drain_pid0_empty_n[0]), .read(drain_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(0)) u_drain_pid0_1 (.dout(drain_pid0_dout[1]), .empty_n(drain_pid0_empty_n[1]), .read(drain_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(0)) u_drain_pid0_2 (.dout(drain_pid0_dout[2]), .empty_n(drain_pid0_empty_n[2]), .read(drain_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(0)) u_drain_pid0_3 (.dout(drain_pid0_dout[3]), .empty_n(drain_pid0_empty_n[3]), .read(drain_pid0_read[3]));
  spmw_const #(.DW(32), .VAL(1)) u_drain_pid0_4 (.dout(drain_pid0_dout[4]), .empty_n(drain_pid0_empty_n[4]), .read(drain_pid0_read[4]));
  spmw_const #(.DW(32), .VAL(1)) u_drain_pid0_5 (.dout(drain_pid0_dout[5]), .empty_n(drain_pid0_empty_n[5]), .read(drain_pid0_read[5]));
  spmw_const #(.DW(32), .VAL(1)) u_drain_pid0_6 (.dout(drain_pid0_dout[6]), .empty_n(drain_pid0_empty_n[6]), .read(drain_pid0_read[6]));
  spmw_const #(.DW(32), .VAL(1)) u_drain_pid0_7 (.dout(drain_pid0_dout[7]), .empty_n(drain_pid0_empty_n[7]), .read(drain_pid0_read[7]));
  spmw_const #(.DW(32), .VAL(2)) u_drain_pid0_8 (.dout(drain_pid0_dout[8]), .empty_n(drain_pid0_empty_n[8]), .read(drain_pid0_read[8]));
  spmw_const #(.DW(32), .VAL(2)) u_drain_pid0_9 (.dout(drain_pid0_dout[9]), .empty_n(drain_pid0_empty_n[9]), .read(drain_pid0_read[9]));
  spmw_const #(.DW(32), .VAL(2)) u_drain_pid0_10 (.dout(drain_pid0_dout[10]), .empty_n(drain_pid0_empty_n[10]), .read(drain_pid0_read[10]));
  spmw_const #(.DW(32), .VAL(2)) u_drain_pid0_11 (.dout(drain_pid0_dout[11]), .empty_n(drain_pid0_empty_n[11]), .read(drain_pid0_read[11]));
  spmw_const #(.DW(32), .VAL(3)) u_drain_pid0_12 (.dout(drain_pid0_dout[12]), .empty_n(drain_pid0_empty_n[12]), .read(drain_pid0_read[12]));
  spmw_const #(.DW(32), .VAL(3)) u_drain_pid0_13 (.dout(drain_pid0_dout[13]), .empty_n(drain_pid0_empty_n[13]), .read(drain_pid0_read[13]));
  spmw_const #(.DW(32), .VAL(3)) u_drain_pid0_14 (.dout(drain_pid0_dout[14]), .empty_n(drain_pid0_empty_n[14]), .read(drain_pid0_read[14]));
  spmw_const #(.DW(32), .VAL(3)) u_drain_pid0_15 (.dout(drain_pid0_dout[15]), .empty_n(drain_pid0_empty_n[15]), .read(drain_pid0_read[15]));
  // coordinate axis 1: 16 constant source(s)
  wire [31:0] drain_pid1_dout [0:15];
  wire drain_pid1_empty_n [0:15];
  wire drain_pid1_read [0:15];
  spmw_const #(.DW(32), .VAL(0)) u_drain_pid1_0 (.dout(drain_pid1_dout[0]), .empty_n(drain_pid1_empty_n[0]), .read(drain_pid1_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_drain_pid1_1 (.dout(drain_pid1_dout[1]), .empty_n(drain_pid1_empty_n[1]), .read(drain_pid1_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_drain_pid1_2 (.dout(drain_pid1_dout[2]), .empty_n(drain_pid1_empty_n[2]), .read(drain_pid1_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_drain_pid1_3 (.dout(drain_pid1_dout[3]), .empty_n(drain_pid1_empty_n[3]), .read(drain_pid1_read[3]));
  spmw_const #(.DW(32), .VAL(0)) u_drain_pid1_4 (.dout(drain_pid1_dout[4]), .empty_n(drain_pid1_empty_n[4]), .read(drain_pid1_read[4]));
  spmw_const #(.DW(32), .VAL(1)) u_drain_pid1_5 (.dout(drain_pid1_dout[5]), .empty_n(drain_pid1_empty_n[5]), .read(drain_pid1_read[5]));
  spmw_const #(.DW(32), .VAL(2)) u_drain_pid1_6 (.dout(drain_pid1_dout[6]), .empty_n(drain_pid1_empty_n[6]), .read(drain_pid1_read[6]));
  spmw_const #(.DW(32), .VAL(3)) u_drain_pid1_7 (.dout(drain_pid1_dout[7]), .empty_n(drain_pid1_empty_n[7]), .read(drain_pid1_read[7]));
  spmw_const #(.DW(32), .VAL(0)) u_drain_pid1_8 (.dout(drain_pid1_dout[8]), .empty_n(drain_pid1_empty_n[8]), .read(drain_pid1_read[8]));
  spmw_const #(.DW(32), .VAL(1)) u_drain_pid1_9 (.dout(drain_pid1_dout[9]), .empty_n(drain_pid1_empty_n[9]), .read(drain_pid1_read[9]));
  spmw_const #(.DW(32), .VAL(2)) u_drain_pid1_10 (.dout(drain_pid1_dout[10]), .empty_n(drain_pid1_empty_n[10]), .read(drain_pid1_read[10]));
  spmw_const #(.DW(32), .VAL(3)) u_drain_pid1_11 (.dout(drain_pid1_dout[11]), .empty_n(drain_pid1_empty_n[11]), .read(drain_pid1_read[11]));
  spmw_const #(.DW(32), .VAL(0)) u_drain_pid1_12 (.dout(drain_pid1_dout[12]), .empty_n(drain_pid1_empty_n[12]), .read(drain_pid1_read[12]));
  spmw_const #(.DW(32), .VAL(1)) u_drain_pid1_13 (.dout(drain_pid1_dout[13]), .empty_n(drain_pid1_empty_n[13]), .read(drain_pid1_read[13]));
  spmw_const #(.DW(32), .VAL(2)) u_drain_pid1_14 (.dout(drain_pid1_dout[14]), .empty_n(drain_pid1_empty_n[14]), .read(drain_pid1_read[14]));
  spmw_const #(.DW(32), .VAL(3)) u_drain_pid1_15 (.dout(drain_pid1_dout[15]), .empty_n(drain_pid1_empty_n[15]), .read(drain_pid1_read[15]));
  // role drain_r0: 8 instance(s)
  drain_r0 u_drain_r0_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_up_din[8]),
      .down_full_n(drain_down_up_full_n[8]),
      .down_write(drain_down_up_write[8]),
      .mine_dout(drain_mine_bind_dout[4]),
      .mine_empty_n(drain_mine_bind_empty_n[4]),
      .mine_read(drain_mine_bind_read[4]),
      .up_dout(drain_down_up_dout[4]),
      .up_empty_n(drain_down_up_empty_n[4]),
      .up_read(drain_down_up_read[4]),
      ._pid0_dout(drain_pid0_dout[4]),
      ._pid0_empty_n(drain_pid0_empty_n[4]),
      ._pid0_read(drain_pid0_read[4]),
      ._pid1_dout(drain_pid1_dout[4]),
      ._pid1_empty_n(drain_pid1_empty_n[4]),
      ._pid1_read(drain_pid1_read[4]));
  drain_r0 u_drain_r0_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_up_din[9]),
      .down_full_n(drain_down_up_full_n[9]),
      .down_write(drain_down_up_write[9]),
      .mine_dout(drain_mine_bind_dout[5]),
      .mine_empty_n(drain_mine_bind_empty_n[5]),
      .mine_read(drain_mine_bind_read[5]),
      .up_dout(drain_down_up_dout[5]),
      .up_empty_n(drain_down_up_empty_n[5]),
      .up_read(drain_down_up_read[5]),
      ._pid0_dout(drain_pid0_dout[5]),
      ._pid0_empty_n(drain_pid0_empty_n[5]),
      ._pid0_read(drain_pid0_read[5]),
      ._pid1_dout(drain_pid1_dout[5]),
      ._pid1_empty_n(drain_pid1_empty_n[5]),
      ._pid1_read(drain_pid1_read[5]));
  drain_r0 u_drain_r0_1_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_up_din[10]),
      .down_full_n(drain_down_up_full_n[10]),
      .down_write(drain_down_up_write[10]),
      .mine_dout(drain_mine_bind_dout[6]),
      .mine_empty_n(drain_mine_bind_empty_n[6]),
      .mine_read(drain_mine_bind_read[6]),
      .up_dout(drain_down_up_dout[6]),
      .up_empty_n(drain_down_up_empty_n[6]),
      .up_read(drain_down_up_read[6]),
      ._pid0_dout(drain_pid0_dout[6]),
      ._pid0_empty_n(drain_pid0_empty_n[6]),
      ._pid0_read(drain_pid0_read[6]),
      ._pid1_dout(drain_pid1_dout[6]),
      ._pid1_empty_n(drain_pid1_empty_n[6]),
      ._pid1_read(drain_pid1_read[6]));
  drain_r0 u_drain_r0_1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_up_din[11]),
      .down_full_n(drain_down_up_full_n[11]),
      .down_write(drain_down_up_write[11]),
      .mine_dout(drain_mine_bind_dout[7]),
      .mine_empty_n(drain_mine_bind_empty_n[7]),
      .mine_read(drain_mine_bind_read[7]),
      .up_dout(drain_down_up_dout[7]),
      .up_empty_n(drain_down_up_empty_n[7]),
      .up_read(drain_down_up_read[7]),
      ._pid0_dout(drain_pid0_dout[7]),
      ._pid0_empty_n(drain_pid0_empty_n[7]),
      ._pid0_read(drain_pid0_read[7]),
      ._pid1_dout(drain_pid1_dout[7]),
      ._pid1_empty_n(drain_pid1_empty_n[7]),
      ._pid1_read(drain_pid1_read[7]));
  drain_r0 u_drain_r0_2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_up_din[12]),
      .down_full_n(drain_down_up_full_n[12]),
      .down_write(drain_down_up_write[12]),
      .mine_dout(drain_mine_bind_dout[8]),
      .mine_empty_n(drain_mine_bind_empty_n[8]),
      .mine_read(drain_mine_bind_read[8]),
      .up_dout(drain_down_up_dout[8]),
      .up_empty_n(drain_down_up_empty_n[8]),
      .up_read(drain_down_up_read[8]),
      ._pid0_dout(drain_pid0_dout[8]),
      ._pid0_empty_n(drain_pid0_empty_n[8]),
      ._pid0_read(drain_pid0_read[8]),
      ._pid1_dout(drain_pid1_dout[8]),
      ._pid1_empty_n(drain_pid1_empty_n[8]),
      ._pid1_read(drain_pid1_read[8]));
  drain_r0 u_drain_r0_2_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_up_din[13]),
      .down_full_n(drain_down_up_full_n[13]),
      .down_write(drain_down_up_write[13]),
      .mine_dout(drain_mine_bind_dout[9]),
      .mine_empty_n(drain_mine_bind_empty_n[9]),
      .mine_read(drain_mine_bind_read[9]),
      .up_dout(drain_down_up_dout[9]),
      .up_empty_n(drain_down_up_empty_n[9]),
      .up_read(drain_down_up_read[9]),
      ._pid0_dout(drain_pid0_dout[9]),
      ._pid0_empty_n(drain_pid0_empty_n[9]),
      ._pid0_read(drain_pid0_read[9]),
      ._pid1_dout(drain_pid1_dout[9]),
      ._pid1_empty_n(drain_pid1_empty_n[9]),
      ._pid1_read(drain_pid1_read[9]));
  drain_r0 u_drain_r0_2_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_up_din[14]),
      .down_full_n(drain_down_up_full_n[14]),
      .down_write(drain_down_up_write[14]),
      .mine_dout(drain_mine_bind_dout[10]),
      .mine_empty_n(drain_mine_bind_empty_n[10]),
      .mine_read(drain_mine_bind_read[10]),
      .up_dout(drain_down_up_dout[10]),
      .up_empty_n(drain_down_up_empty_n[10]),
      .up_read(drain_down_up_read[10]),
      ._pid0_dout(drain_pid0_dout[10]),
      ._pid0_empty_n(drain_pid0_empty_n[10]),
      ._pid0_read(drain_pid0_read[10]),
      ._pid1_dout(drain_pid1_dout[10]),
      ._pid1_empty_n(drain_pid1_empty_n[10]),
      ._pid1_read(drain_pid1_read[10]));
  drain_r0 u_drain_r0_2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_up_din[15]),
      .down_full_n(drain_down_up_full_n[15]),
      .down_write(drain_down_up_write[15]),
      .mine_dout(drain_mine_bind_dout[11]),
      .mine_empty_n(drain_mine_bind_empty_n[11]),
      .mine_read(drain_mine_bind_read[11]),
      .up_dout(drain_down_up_dout[11]),
      .up_empty_n(drain_down_up_empty_n[11]),
      .up_read(drain_down_up_read[11]),
      ._pid0_dout(drain_pid0_dout[11]),
      ._pid0_empty_n(drain_pid0_empty_n[11]),
      ._pid0_read(drain_pid0_read[11]),
      ._pid1_dout(drain_pid1_dout[11]),
      ._pid1_empty_n(drain_pid1_empty_n[11]),
      ._pid1_read(drain_pid1_read[11]));
  // role drain_r1: 4 instance(s)
  drain_r1 u_drain_r1_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_up_din[4]),
      .down_full_n(drain_down_up_full_n[4]),
      .down_write(drain_down_up_write[4]),
      .mine_dout(drain_mine_bind_dout[0]),
      .mine_empty_n(drain_mine_bind_empty_n[0]),
      .mine_read(drain_mine_bind_read[0]),
      ._pid0_dout(drain_pid0_dout[0]),
      ._pid0_empty_n(drain_pid0_empty_n[0]),
      ._pid0_read(drain_pid0_read[0]),
      ._pid1_dout(drain_pid1_dout[0]),
      ._pid1_empty_n(drain_pid1_empty_n[0]),
      ._pid1_read(drain_pid1_read[0]));
  drain_r1 u_drain_r1_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_up_din[5]),
      .down_full_n(drain_down_up_full_n[5]),
      .down_write(drain_down_up_write[5]),
      .mine_dout(drain_mine_bind_dout[1]),
      .mine_empty_n(drain_mine_bind_empty_n[1]),
      .mine_read(drain_mine_bind_read[1]),
      ._pid0_dout(drain_pid0_dout[1]),
      ._pid0_empty_n(drain_pid0_empty_n[1]),
      ._pid0_read(drain_pid0_read[1]),
      ._pid1_dout(drain_pid1_dout[1]),
      ._pid1_empty_n(drain_pid1_empty_n[1]),
      ._pid1_read(drain_pid1_read[1]));
  drain_r1 u_drain_r1_0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_up_din[6]),
      .down_full_n(drain_down_up_full_n[6]),
      .down_write(drain_down_up_write[6]),
      .mine_dout(drain_mine_bind_dout[2]),
      .mine_empty_n(drain_mine_bind_empty_n[2]),
      .mine_read(drain_mine_bind_read[2]),
      ._pid0_dout(drain_pid0_dout[2]),
      ._pid0_empty_n(drain_pid0_empty_n[2]),
      ._pid0_read(drain_pid0_read[2]),
      ._pid1_dout(drain_pid1_dout[2]),
      ._pid1_empty_n(drain_pid1_empty_n[2]),
      ._pid1_read(drain_pid1_read[2]));
  drain_r1 u_drain_r1_0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_up_din[7]),
      .down_full_n(drain_down_up_full_n[7]),
      .down_write(drain_down_up_write[7]),
      .mine_dout(drain_mine_bind_dout[3]),
      .mine_empty_n(drain_mine_bind_empty_n[3]),
      .mine_read(drain_mine_bind_read[3]),
      ._pid0_dout(drain_pid0_dout[3]),
      ._pid0_empty_n(drain_pid0_empty_n[3]),
      ._pid0_read(drain_pid0_read[3]),
      ._pid1_dout(drain_pid1_dout[3]),
      ._pid1_empty_n(drain_pid1_empty_n[3]),
      ._pid1_read(drain_pid1_read[3]));
  // role drain_r2: 4 instance(s)
  drain_r2 u_drain_r2_3_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_bind_din[0]),
      .down_full_n(drain_down_bind_full_n[0]),
      .down_write(drain_down_bind_write[0]),
      .mine_dout(drain_mine_bind_dout[12]),
      .mine_empty_n(drain_mine_bind_empty_n[12]),
      .mine_read(drain_mine_bind_read[12]),
      .up_dout(drain_down_up_dout[12]),
      .up_empty_n(drain_down_up_empty_n[12]),
      .up_read(drain_down_up_read[12]),
      ._pid0_dout(drain_pid0_dout[12]),
      ._pid0_empty_n(drain_pid0_empty_n[12]),
      ._pid0_read(drain_pid0_read[12]),
      ._pid1_dout(drain_pid1_dout[12]),
      ._pid1_empty_n(drain_pid1_empty_n[12]),
      ._pid1_read(drain_pid1_read[12]));
  drain_r2 u_drain_r2_3_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_bind_din[1]),
      .down_full_n(drain_down_bind_full_n[1]),
      .down_write(drain_down_bind_write[1]),
      .mine_dout(drain_mine_bind_dout[13]),
      .mine_empty_n(drain_mine_bind_empty_n[13]),
      .mine_read(drain_mine_bind_read[13]),
      .up_dout(drain_down_up_dout[13]),
      .up_empty_n(drain_down_up_empty_n[13]),
      .up_read(drain_down_up_read[13]),
      ._pid0_dout(drain_pid0_dout[13]),
      ._pid0_empty_n(drain_pid0_empty_n[13]),
      ._pid0_read(drain_pid0_read[13]),
      ._pid1_dout(drain_pid1_dout[13]),
      ._pid1_empty_n(drain_pid1_empty_n[13]),
      ._pid1_read(drain_pid1_read[13]));
  drain_r2 u_drain_r2_3_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_bind_din[2]),
      .down_full_n(drain_down_bind_full_n[2]),
      .down_write(drain_down_bind_write[2]),
      .mine_dout(drain_mine_bind_dout[14]),
      .mine_empty_n(drain_mine_bind_empty_n[14]),
      .mine_read(drain_mine_bind_read[14]),
      .up_dout(drain_down_up_dout[14]),
      .up_empty_n(drain_down_up_empty_n[14]),
      .up_read(drain_down_up_read[14]),
      ._pid0_dout(drain_pid0_dout[14]),
      ._pid0_empty_n(drain_pid0_empty_n[14]),
      ._pid0_read(drain_pid0_read[14]),
      ._pid1_dout(drain_pid1_dout[14]),
      ._pid1_empty_n(drain_pid1_empty_n[14]),
      ._pid1_read(drain_pid1_read[14]));
  drain_r2 u_drain_r2_3_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(drain_down_bind_din[3]),
      .down_full_n(drain_down_bind_full_n[3]),
      .down_write(drain_down_bind_write[3]),
      .mine_dout(drain_mine_bind_dout[15]),
      .mine_empty_n(drain_mine_bind_empty_n[15]),
      .mine_read(drain_mine_bind_read[15]),
      .up_dout(drain_down_up_dout[15]),
      .up_empty_n(drain_down_up_empty_n[15]),
      .up_read(drain_down_up_read[15]),
      ._pid0_dout(drain_pid0_dout[15]),
      ._pid0_empty_n(drain_pid0_empty_n[15]),
      ._pid0_read(drain_pid0_read[15]),
      ._pid1_dout(drain_pid1_dout[15]),
      ._pid1_empty_n(drain_pid1_empty_n[15]),
      ._pid1_read(drain_pid1_read[15]));
  // coordinate axis 0: 4 constant source(s)
  wire [31:0] feed_pid0_dout [0:3];
  wire feed_pid0_empty_n [0:3];
  wire feed_pid0_read [0:3];
  spmw_const #(.DW(32), .VAL(0)) u_feed_pid0_0 (.dout(feed_pid0_dout[0]), .empty_n(feed_pid0_empty_n[0]), .read(feed_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_feed_pid0_1 (.dout(feed_pid0_dout[1]), .empty_n(feed_pid0_empty_n[1]), .read(feed_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_feed_pid0_2 (.dout(feed_pid0_dout[2]), .empty_n(feed_pid0_empty_n[2]), .read(feed_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_feed_pid0_3 (.dout(feed_pid0_dout[3]), .empty_n(feed_pid0_empty_n[3]), .read(feed_pid0_read[3]));
  // role feed_r0: 2 instance(s)
  feed_r0 u_feed_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(feed_down_up_din[2]),
      .down_full_n(feed_down_up_full_n[2]),
      .down_write(feed_down_up_write[2]),
      .lane_din(pe_west_bind_din[1]),
      .lane_full_n(pe_west_bind_full_n[1]),
      .lane_write(pe_west_bind_write[1]),
      .up_dout(feed_down_up_dout[1]),
      .up_empty_n(feed_down_up_empty_n[1]),
      .up_read(feed_down_up_read[1]),
      ._pid0_dout(feed_pid0_dout[1]),
      ._pid0_empty_n(feed_pid0_empty_n[1]),
      ._pid0_read(feed_pid0_read[1]));
  feed_r0 u_feed_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(feed_down_up_din[3]),
      .down_full_n(feed_down_up_full_n[3]),
      .down_write(feed_down_up_write[3]),
      .lane_din(pe_west_bind_din[2]),
      .lane_full_n(pe_west_bind_full_n[2]),
      .lane_write(pe_west_bind_write[2]),
      .up_dout(feed_down_up_dout[2]),
      .up_empty_n(feed_down_up_empty_n[2]),
      .up_read(feed_down_up_read[2]),
      ._pid0_dout(feed_pid0_dout[2]),
      ._pid0_empty_n(feed_pid0_empty_n[2]),
      ._pid0_read(feed_pid0_read[2]));
  // role feed_r1: 1 instance(s)
  feed_r1 u_feed_r1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(feed_down_up_din[1]),
      .down_full_n(feed_down_up_full_n[1]),
      .down_write(feed_down_up_write[1]),
      .lane_din(pe_west_bind_din[0]),
      .lane_full_n(pe_west_bind_full_n[0]),
      .lane_write(pe_west_bind_write[0]),
      .up_dout(feed_up_bind_dout[0]),
      .up_empty_n(feed_up_bind_empty_n[0]),
      .up_read(feed_up_bind_read[0]),
      ._pid0_dout(feed_pid0_dout[0]),
      ._pid0_empty_n(feed_pid0_empty_n[0]),
      ._pid0_read(feed_pid0_read[0]));
  // role feed_r2: 1 instance(s)
  feed_r2 u_feed_r2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lane_din(pe_west_bind_din[3]),
      .lane_full_n(pe_west_bind_full_n[3]),
      .lane_write(pe_west_bind_write[3]),
      .up_dout(feed_down_up_dout[3]),
      .up_empty_n(feed_down_up_empty_n[3]),
      .up_read(feed_down_up_read[3]),
      ._pid0_dout(feed_pid0_dout[3]),
      ._pid0_empty_n(feed_pid0_empty_n[3]),
      ._pid0_read(feed_pid0_read[3]));
  // coordinate axis 0: 4 constant source(s)
  wire [31:0] feed_3_pid0_dout [0:3];
  wire feed_3_pid0_empty_n [0:3];
  wire feed_3_pid0_read [0:3];
  spmw_const #(.DW(32), .VAL(0)) u_feed_3_pid0_0 (.dout(feed_3_pid0_dout[0]), .empty_n(feed_3_pid0_empty_n[0]), .read(feed_3_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_feed_3_pid0_1 (.dout(feed_3_pid0_dout[1]), .empty_n(feed_3_pid0_empty_n[1]), .read(feed_3_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_feed_3_pid0_2 (.dout(feed_3_pid0_dout[2]), .empty_n(feed_3_pid0_empty_n[2]), .read(feed_3_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_feed_3_pid0_3 (.dout(feed_3_pid0_dout[3]), .empty_n(feed_3_pid0_empty_n[3]), .read(feed_3_pid0_read[3]));
  // role feed_3_r0: 2 instance(s)
  feed_3_r0 u_feed_3_r0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(feed_3_down_up_din[2]),
      .down_full_n(feed_3_down_up_full_n[2]),
      .down_write(feed_3_down_up_write[2]),
      .lane_din(pe_north_bind_din[1]),
      .lane_full_n(pe_north_bind_full_n[1]),
      .lane_write(pe_north_bind_write[1]),
      .up_dout(feed_3_down_up_dout[1]),
      .up_empty_n(feed_3_down_up_empty_n[1]),
      .up_read(feed_3_down_up_read[1]),
      ._pid0_dout(feed_3_pid0_dout[1]),
      ._pid0_empty_n(feed_3_pid0_empty_n[1]),
      ._pid0_read(feed_3_pid0_read[1]));
  feed_3_r0 u_feed_3_r0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(feed_3_down_up_din[3]),
      .down_full_n(feed_3_down_up_full_n[3]),
      .down_write(feed_3_down_up_write[3]),
      .lane_din(pe_north_bind_din[2]),
      .lane_full_n(pe_north_bind_full_n[2]),
      .lane_write(pe_north_bind_write[2]),
      .up_dout(feed_3_down_up_dout[2]),
      .up_empty_n(feed_3_down_up_empty_n[2]),
      .up_read(feed_3_down_up_read[2]),
      ._pid0_dout(feed_3_pid0_dout[2]),
      ._pid0_empty_n(feed_3_pid0_empty_n[2]),
      ._pid0_read(feed_3_pid0_read[2]));
  // role feed_3_r1: 1 instance(s)
  feed_3_r1 u_feed_3_r1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .down_din(feed_3_down_up_din[1]),
      .down_full_n(feed_3_down_up_full_n[1]),
      .down_write(feed_3_down_up_write[1]),
      .lane_din(pe_north_bind_din[0]),
      .lane_full_n(pe_north_bind_full_n[0]),
      .lane_write(pe_north_bind_write[0]),
      .up_dout(feed_3_up_bind_dout[0]),
      .up_empty_n(feed_3_up_bind_empty_n[0]),
      .up_read(feed_3_up_bind_read[0]),
      ._pid0_dout(feed_3_pid0_dout[0]),
      ._pid0_empty_n(feed_3_pid0_empty_n[0]),
      ._pid0_read(feed_3_pid0_read[0]));
  // role feed_3_r2: 1 instance(s)
  feed_3_r2 u_feed_3_r2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lane_din(pe_north_bind_din[3]),
      .lane_full_n(pe_north_bind_full_n[3]),
      .lane_write(pe_north_bind_write[3]),
      .up_dout(feed_3_down_up_dout[3]),
      .up_empty_n(feed_3_down_up_empty_n[3]),
      .up_read(feed_3_down_up_read[3]),
      ._pid0_dout(feed_3_pid0_dout[3]),
      ._pid0_empty_n(feed_3_pid0_empty_n[3]),
      ._pid0_read(feed_3_pid0_read[3]));
endmodule
