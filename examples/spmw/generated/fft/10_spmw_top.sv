`timescale 1ns/1ps

module spmw_top (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] bfly_up_in_bind_dout [0:3],
  input  wire bfly_up_in_bind_empty_n [0:3],
  output wire bfly_up_in_bind_read [0:3],
  input  wire [63:0] bfly_lo_in_bind_dout [0:3],
  input  wire bfly_lo_in_bind_empty_n [0:3],
  output wire bfly_lo_in_bind_read [0:3],
  output wire [63:0] bfly_up_out_bind_din [0:3],
  output wire bfly_up_out_bind_write [0:3],
  input  wire bfly_up_out_bind_full_n [0:3],
  output wire [63:0] bfly_lo_out_bind_din [0:3],
  output wire bfly_lo_out_bind_write [0:3],
  input  wire bfly_lo_out_bind_full_n [0:3]
);
  // family bfly_key: 16 channel(s), 64-bit, depth 2
  wire [63:0] bfly_key_din [0:15];
  wire [63:0] bfly_key_dout [0:15];
  wire bfly_key_full_n [0:15];
  wire bfly_key_write [0:15];
  wire bfly_key_empty_n [0:15];
  wire bfly_key_read [0:15];
  genvar bfly_key_i;
  generate
    for (bfly_key_i = 0; bfly_key_i < 16; bfly_key_i = bfly_key_i + 1) begin : g_bfly_key
      spmw_fifo #(.DW(64), .DEPTH(2)) u (.clk(ap_clk), .rst_n(ap_rst_n), .din(bfly_key_din[bfly_key_i]), .full_n(bfly_key_full_n[bfly_key_i]), .write(bfly_key_write[bfly_key_i]), .dout(bfly_key_dout[bfly_key_i]), .empty_n(bfly_key_empty_n[bfly_key_i]), .read(bfly_key_read[bfly_key_i]));
    end
  endgenerate
  // coordinate axis 0: 12 constant source(s)
  wire [31:0] bfly_pid0_dout [0:11];
  wire bfly_pid0_empty_n [0:11];
  wire bfly_pid0_read [0:11];
  spmw_const #(.DW(32), .VAL(0)) u_bfly_pid0_0 (.dout(bfly_pid0_dout[0]), .empty_n(bfly_pid0_empty_n[0]), .read(bfly_pid0_read[0]));
  spmw_const #(.DW(32), .VAL(0)) u_bfly_pid0_1 (.dout(bfly_pid0_dout[1]), .empty_n(bfly_pid0_empty_n[1]), .read(bfly_pid0_read[1]));
  spmw_const #(.DW(32), .VAL(0)) u_bfly_pid0_2 (.dout(bfly_pid0_dout[2]), .empty_n(bfly_pid0_empty_n[2]), .read(bfly_pid0_read[2]));
  spmw_const #(.DW(32), .VAL(0)) u_bfly_pid0_3 (.dout(bfly_pid0_dout[3]), .empty_n(bfly_pid0_empty_n[3]), .read(bfly_pid0_read[3]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly_pid0_4 (.dout(bfly_pid0_dout[4]), .empty_n(bfly_pid0_empty_n[4]), .read(bfly_pid0_read[4]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly_pid0_5 (.dout(bfly_pid0_dout[5]), .empty_n(bfly_pid0_empty_n[5]), .read(bfly_pid0_read[5]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly_pid0_6 (.dout(bfly_pid0_dout[6]), .empty_n(bfly_pid0_empty_n[6]), .read(bfly_pid0_read[6]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly_pid0_7 (.dout(bfly_pid0_dout[7]), .empty_n(bfly_pid0_empty_n[7]), .read(bfly_pid0_read[7]));
  spmw_const #(.DW(32), .VAL(2)) u_bfly_pid0_8 (.dout(bfly_pid0_dout[8]), .empty_n(bfly_pid0_empty_n[8]), .read(bfly_pid0_read[8]));
  spmw_const #(.DW(32), .VAL(2)) u_bfly_pid0_9 (.dout(bfly_pid0_dout[9]), .empty_n(bfly_pid0_empty_n[9]), .read(bfly_pid0_read[9]));
  spmw_const #(.DW(32), .VAL(2)) u_bfly_pid0_10 (.dout(bfly_pid0_dout[10]), .empty_n(bfly_pid0_empty_n[10]), .read(bfly_pid0_read[10]));
  spmw_const #(.DW(32), .VAL(2)) u_bfly_pid0_11 (.dout(bfly_pid0_dout[11]), .empty_n(bfly_pid0_empty_n[11]), .read(bfly_pid0_read[11]));
  // coordinate axis 1: 12 constant source(s)
  wire [31:0] bfly_pid1_dout [0:11];
  wire bfly_pid1_empty_n [0:11];
  wire bfly_pid1_read [0:11];
  spmw_const #(.DW(32), .VAL(0)) u_bfly_pid1_0 (.dout(bfly_pid1_dout[0]), .empty_n(bfly_pid1_empty_n[0]), .read(bfly_pid1_read[0]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly_pid1_1 (.dout(bfly_pid1_dout[1]), .empty_n(bfly_pid1_empty_n[1]), .read(bfly_pid1_read[1]));
  spmw_const #(.DW(32), .VAL(2)) u_bfly_pid1_2 (.dout(bfly_pid1_dout[2]), .empty_n(bfly_pid1_empty_n[2]), .read(bfly_pid1_read[2]));
  spmw_const #(.DW(32), .VAL(3)) u_bfly_pid1_3 (.dout(bfly_pid1_dout[3]), .empty_n(bfly_pid1_empty_n[3]), .read(bfly_pid1_read[3]));
  spmw_const #(.DW(32), .VAL(0)) u_bfly_pid1_4 (.dout(bfly_pid1_dout[4]), .empty_n(bfly_pid1_empty_n[4]), .read(bfly_pid1_read[4]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly_pid1_5 (.dout(bfly_pid1_dout[5]), .empty_n(bfly_pid1_empty_n[5]), .read(bfly_pid1_read[5]));
  spmw_const #(.DW(32), .VAL(2)) u_bfly_pid1_6 (.dout(bfly_pid1_dout[6]), .empty_n(bfly_pid1_empty_n[6]), .read(bfly_pid1_read[6]));
  spmw_const #(.DW(32), .VAL(3)) u_bfly_pid1_7 (.dout(bfly_pid1_dout[7]), .empty_n(bfly_pid1_empty_n[7]), .read(bfly_pid1_read[7]));
  spmw_const #(.DW(32), .VAL(0)) u_bfly_pid1_8 (.dout(bfly_pid1_dout[8]), .empty_n(bfly_pid1_empty_n[8]), .read(bfly_pid1_read[8]));
  spmw_const #(.DW(32), .VAL(1)) u_bfly_pid1_9 (.dout(bfly_pid1_dout[9]), .empty_n(bfly_pid1_empty_n[9]), .read(bfly_pid1_read[9]));
  spmw_const #(.DW(32), .VAL(2)) u_bfly_pid1_10 (.dout(bfly_pid1_dout[10]), .empty_n(bfly_pid1_empty_n[10]), .read(bfly_pid1_read[10]));
  spmw_const #(.DW(32), .VAL(3)) u_bfly_pid1_11 (.dout(bfly_pid1_dout[11]), .empty_n(bfly_pid1_empty_n[11]), .read(bfly_pid1_read[11]));
  // role bfly_r0: 4 instance(s)
  bfly_r0 u_bfly_r0_1_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lo_in_dout(bfly_key_dout[2]),
      .lo_in_empty_n(bfly_key_empty_n[2]),
      .lo_in_read(bfly_key_read[2]),
      .lo_out_din(bfly_key_din[10]),
      .lo_out_full_n(bfly_key_full_n[10]),
      .lo_out_write(bfly_key_write[10]),
      .up_in_dout(bfly_key_dout[0]),
      .up_in_empty_n(bfly_key_empty_n[0]),
      .up_in_read(bfly_key_read[0]),
      .up_out_din(bfly_key_din[8]),
      .up_out_full_n(bfly_key_full_n[8]),
      .up_out_write(bfly_key_write[8]),
      ._pid0_dout(bfly_pid0_dout[4]),
      ._pid0_empty_n(bfly_pid0_empty_n[4]),
      ._pid0_read(bfly_pid0_read[4]),
      ._pid1_dout(bfly_pid1_dout[4]),
      ._pid1_empty_n(bfly_pid1_empty_n[4]),
      ._pid1_read(bfly_pid1_read[4]));
  bfly_r0 u_bfly_r0_1_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lo_in_dout(bfly_key_dout[3]),
      .lo_in_empty_n(bfly_key_empty_n[3]),
      .lo_in_read(bfly_key_read[3]),
      .lo_out_din(bfly_key_din[11]),
      .lo_out_full_n(bfly_key_full_n[11]),
      .lo_out_write(bfly_key_write[11]),
      .up_in_dout(bfly_key_dout[1]),
      .up_in_empty_n(bfly_key_empty_n[1]),
      .up_in_read(bfly_key_read[1]),
      .up_out_din(bfly_key_din[9]),
      .up_out_full_n(bfly_key_full_n[9]),
      .up_out_write(bfly_key_write[9]),
      ._pid0_dout(bfly_pid0_dout[5]),
      ._pid0_empty_n(bfly_pid0_empty_n[5]),
      ._pid0_read(bfly_pid0_read[5]),
      ._pid1_dout(bfly_pid1_dout[5]),
      ._pid1_empty_n(bfly_pid1_empty_n[5]),
      ._pid1_read(bfly_pid1_read[5]));
  bfly_r0 u_bfly_r0_1_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lo_in_dout(bfly_key_dout[6]),
      .lo_in_empty_n(bfly_key_empty_n[6]),
      .lo_in_read(bfly_key_read[6]),
      .lo_out_din(bfly_key_din[14]),
      .lo_out_full_n(bfly_key_full_n[14]),
      .lo_out_write(bfly_key_write[14]),
      .up_in_dout(bfly_key_dout[4]),
      .up_in_empty_n(bfly_key_empty_n[4]),
      .up_in_read(bfly_key_read[4]),
      .up_out_din(bfly_key_din[12]),
      .up_out_full_n(bfly_key_full_n[12]),
      .up_out_write(bfly_key_write[12]),
      ._pid0_dout(bfly_pid0_dout[6]),
      ._pid0_empty_n(bfly_pid0_empty_n[6]),
      ._pid0_read(bfly_pid0_read[6]),
      ._pid1_dout(bfly_pid1_dout[6]),
      ._pid1_empty_n(bfly_pid1_empty_n[6]),
      ._pid1_read(bfly_pid1_read[6]));
  bfly_r0 u_bfly_r0_1_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lo_in_dout(bfly_key_dout[7]),
      .lo_in_empty_n(bfly_key_empty_n[7]),
      .lo_in_read(bfly_key_read[7]),
      .lo_out_din(bfly_key_din[15]),
      .lo_out_full_n(bfly_key_full_n[15]),
      .lo_out_write(bfly_key_write[15]),
      .up_in_dout(bfly_key_dout[5]),
      .up_in_empty_n(bfly_key_empty_n[5]),
      .up_in_read(bfly_key_read[5]),
      .up_out_din(bfly_key_din[13]),
      .up_out_full_n(bfly_key_full_n[13]),
      .up_out_write(bfly_key_write[13]),
      ._pid0_dout(bfly_pid0_dout[7]),
      ._pid0_empty_n(bfly_pid0_empty_n[7]),
      ._pid0_read(bfly_pid0_read[7]),
      ._pid1_dout(bfly_pid1_dout[7]),
      ._pid1_empty_n(bfly_pid1_empty_n[7]),
      ._pid1_read(bfly_pid1_read[7]));
  // role bfly_r1: 4 instance(s)
  bfly_r1 u_bfly_r1_2_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lo_in_dout(bfly_key_dout[12]),
      .lo_in_empty_n(bfly_key_empty_n[12]),
      .lo_in_read(bfly_key_read[12]),
      .lo_out_din(bfly_lo_out_bind_din[0]),
      .lo_out_full_n(bfly_lo_out_bind_full_n[0]),
      .lo_out_write(bfly_lo_out_bind_write[0]),
      .up_in_dout(bfly_key_dout[8]),
      .up_in_empty_n(bfly_key_empty_n[8]),
      .up_in_read(bfly_key_read[8]),
      .up_out_din(bfly_up_out_bind_din[0]),
      .up_out_full_n(bfly_up_out_bind_full_n[0]),
      .up_out_write(bfly_up_out_bind_write[0]),
      ._pid0_dout(bfly_pid0_dout[8]),
      ._pid0_empty_n(bfly_pid0_empty_n[8]),
      ._pid0_read(bfly_pid0_read[8]),
      ._pid1_dout(bfly_pid1_dout[8]),
      ._pid1_empty_n(bfly_pid1_empty_n[8]),
      ._pid1_read(bfly_pid1_read[8]));
  bfly_r1 u_bfly_r1_2_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lo_in_dout(bfly_key_dout[13]),
      .lo_in_empty_n(bfly_key_empty_n[13]),
      .lo_in_read(bfly_key_read[13]),
      .lo_out_din(bfly_lo_out_bind_din[1]),
      .lo_out_full_n(bfly_lo_out_bind_full_n[1]),
      .lo_out_write(bfly_lo_out_bind_write[1]),
      .up_in_dout(bfly_key_dout[9]),
      .up_in_empty_n(bfly_key_empty_n[9]),
      .up_in_read(bfly_key_read[9]),
      .up_out_din(bfly_up_out_bind_din[1]),
      .up_out_full_n(bfly_up_out_bind_full_n[1]),
      .up_out_write(bfly_up_out_bind_write[1]),
      ._pid0_dout(bfly_pid0_dout[9]),
      ._pid0_empty_n(bfly_pid0_empty_n[9]),
      ._pid0_read(bfly_pid0_read[9]),
      ._pid1_dout(bfly_pid1_dout[9]),
      ._pid1_empty_n(bfly_pid1_empty_n[9]),
      ._pid1_read(bfly_pid1_read[9]));
  bfly_r1 u_bfly_r1_2_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lo_in_dout(bfly_key_dout[14]),
      .lo_in_empty_n(bfly_key_empty_n[14]),
      .lo_in_read(bfly_key_read[14]),
      .lo_out_din(bfly_lo_out_bind_din[2]),
      .lo_out_full_n(bfly_lo_out_bind_full_n[2]),
      .lo_out_write(bfly_lo_out_bind_write[2]),
      .up_in_dout(bfly_key_dout[10]),
      .up_in_empty_n(bfly_key_empty_n[10]),
      .up_in_read(bfly_key_read[10]),
      .up_out_din(bfly_up_out_bind_din[2]),
      .up_out_full_n(bfly_up_out_bind_full_n[2]),
      .up_out_write(bfly_up_out_bind_write[2]),
      ._pid0_dout(bfly_pid0_dout[10]),
      ._pid0_empty_n(bfly_pid0_empty_n[10]),
      ._pid0_read(bfly_pid0_read[10]),
      ._pid1_dout(bfly_pid1_dout[10]),
      ._pid1_empty_n(bfly_pid1_empty_n[10]),
      ._pid1_read(bfly_pid1_read[10]));
  bfly_r1 u_bfly_r1_2_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lo_in_dout(bfly_key_dout[15]),
      .lo_in_empty_n(bfly_key_empty_n[15]),
      .lo_in_read(bfly_key_read[15]),
      .lo_out_din(bfly_lo_out_bind_din[3]),
      .lo_out_full_n(bfly_lo_out_bind_full_n[3]),
      .lo_out_write(bfly_lo_out_bind_write[3]),
      .up_in_dout(bfly_key_dout[11]),
      .up_in_empty_n(bfly_key_empty_n[11]),
      .up_in_read(bfly_key_read[11]),
      .up_out_din(bfly_up_out_bind_din[3]),
      .up_out_full_n(bfly_up_out_bind_full_n[3]),
      .up_out_write(bfly_up_out_bind_write[3]),
      ._pid0_dout(bfly_pid0_dout[11]),
      ._pid0_empty_n(bfly_pid0_empty_n[11]),
      ._pid0_read(bfly_pid0_read[11]),
      ._pid1_dout(bfly_pid1_dout[11]),
      ._pid1_empty_n(bfly_pid1_empty_n[11]),
      ._pid1_read(bfly_pid1_read[11]));
  // role bfly_r2: 4 instance(s)
  bfly_r2 u_bfly_r2_0_0 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lo_in_dout(bfly_lo_in_bind_dout[0]),
      .lo_in_empty_n(bfly_lo_in_bind_empty_n[0]),
      .lo_in_read(bfly_lo_in_bind_read[0]),
      .lo_out_din(bfly_key_din[1]),
      .lo_out_full_n(bfly_key_full_n[1]),
      .lo_out_write(bfly_key_write[1]),
      .up_in_dout(bfly_up_in_bind_dout[0]),
      .up_in_empty_n(bfly_up_in_bind_empty_n[0]),
      .up_in_read(bfly_up_in_bind_read[0]),
      .up_out_din(bfly_key_din[0]),
      .up_out_full_n(bfly_key_full_n[0]),
      .up_out_write(bfly_key_write[0]),
      ._pid0_dout(bfly_pid0_dout[0]),
      ._pid0_empty_n(bfly_pid0_empty_n[0]),
      ._pid0_read(bfly_pid0_read[0]),
      ._pid1_dout(bfly_pid1_dout[0]),
      ._pid1_empty_n(bfly_pid1_empty_n[0]),
      ._pid1_read(bfly_pid1_read[0]));
  bfly_r2 u_bfly_r2_0_1 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lo_in_dout(bfly_lo_in_bind_dout[1]),
      .lo_in_empty_n(bfly_lo_in_bind_empty_n[1]),
      .lo_in_read(bfly_lo_in_bind_read[1]),
      .lo_out_din(bfly_key_din[3]),
      .lo_out_full_n(bfly_key_full_n[3]),
      .lo_out_write(bfly_key_write[3]),
      .up_in_dout(bfly_up_in_bind_dout[1]),
      .up_in_empty_n(bfly_up_in_bind_empty_n[1]),
      .up_in_read(bfly_up_in_bind_read[1]),
      .up_out_din(bfly_key_din[2]),
      .up_out_full_n(bfly_key_full_n[2]),
      .up_out_write(bfly_key_write[2]),
      ._pid0_dout(bfly_pid0_dout[1]),
      ._pid0_empty_n(bfly_pid0_empty_n[1]),
      ._pid0_read(bfly_pid0_read[1]),
      ._pid1_dout(bfly_pid1_dout[1]),
      ._pid1_empty_n(bfly_pid1_empty_n[1]),
      ._pid1_read(bfly_pid1_read[1]));
  bfly_r2 u_bfly_r2_0_2 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lo_in_dout(bfly_lo_in_bind_dout[2]),
      .lo_in_empty_n(bfly_lo_in_bind_empty_n[2]),
      .lo_in_read(bfly_lo_in_bind_read[2]),
      .lo_out_din(bfly_key_din[5]),
      .lo_out_full_n(bfly_key_full_n[5]),
      .lo_out_write(bfly_key_write[5]),
      .up_in_dout(bfly_up_in_bind_dout[2]),
      .up_in_empty_n(bfly_up_in_bind_empty_n[2]),
      .up_in_read(bfly_up_in_bind_read[2]),
      .up_out_din(bfly_key_din[4]),
      .up_out_full_n(bfly_key_full_n[4]),
      .up_out_write(bfly_key_write[4]),
      ._pid0_dout(bfly_pid0_dout[2]),
      ._pid0_empty_n(bfly_pid0_empty_n[2]),
      ._pid0_read(bfly_pid0_read[2]),
      ._pid1_dout(bfly_pid1_dout[2]),
      ._pid1_empty_n(bfly_pid1_empty_n[2]),
      ._pid1_read(bfly_pid1_read[2]));
  bfly_r2 u_bfly_r2_0_3 (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .lo_in_dout(bfly_lo_in_bind_dout[3]),
      .lo_in_empty_n(bfly_lo_in_bind_empty_n[3]),
      .lo_in_read(bfly_lo_in_bind_read[3]),
      .lo_out_din(bfly_key_din[7]),
      .lo_out_full_n(bfly_key_full_n[7]),
      .lo_out_write(bfly_key_write[7]),
      .up_in_dout(bfly_up_in_bind_dout[3]),
      .up_in_empty_n(bfly_up_in_bind_empty_n[3]),
      .up_in_read(bfly_up_in_bind_read[3]),
      .up_out_din(bfly_key_din[6]),
      .up_out_full_n(bfly_key_full_n[6]),
      .up_out_write(bfly_key_write[6]),
      ._pid0_dout(bfly_pid0_dout[3]),
      ._pid0_empty_n(bfly_pid0_empty_n[3]),
      ._pid0_read(bfly_pid0_read[3]),
      ._pid1_dout(bfly_pid1_dout[3]),
      ._pid1_empty_n(bfly_pid1_empty_n[3]),
      ._pid1_read(bfly_pid1_read[3]));
endmodule
