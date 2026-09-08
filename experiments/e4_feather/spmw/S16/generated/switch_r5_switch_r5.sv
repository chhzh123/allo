`timescale 1ns/1ps

module switch_r5 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [15:0] cmd_dout,
  input  wire cmd_empty_n,
  output wire cmd_read,
  input  wire [31:0] in_l_dout,
  input  wire in_l_empty_n,
  output wire in_l_read,
  input  wire [31:0] in_r_dout,
  input  wire in_r_empty_n,
  output wire in_r_read,
  output wire [31:0] out_l_din,
  input  wire out_l_full_n,
  output wire out_l_write,
  output wire [31:0] out_r_din,
  input  wire out_r_full_n,
  output wire out_r_write
);
  switch_r5_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(cmd_dout),
      .v0_empty_n(cmd_empty_n),
      .v0_read(cmd_read),
      .v1_dout(in_l_dout),
      .v1_empty_n(in_l_empty_n),
      .v1_read(in_l_read),
      .v2_dout(in_r_dout),
      .v2_empty_n(in_r_empty_n),
      .v2_read(in_r_read),
      .v3_din(out_l_din),
      .v3_full_n(out_l_full_n),
      .v3_write(out_l_write),
      .v4_din(out_r_din),
      .v4_full_n(out_r_full_n),
      .v4_write(out_r_write));
endmodule
