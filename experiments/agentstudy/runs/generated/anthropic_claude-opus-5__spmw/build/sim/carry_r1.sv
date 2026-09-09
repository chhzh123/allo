`timescale 1ns/1ps

module carry_r1 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] _pid1_dout,
  input  wire _pid1_empty_n,
  output wire _pid1_read,
  input  wire [31:0] c_in_dout,
  input  wire c_in_empty_n,
  output wire c_in_read,
  output wire [31:0] c_out_din,
  input  wire c_out_full_n,
  output wire c_out_write,
  input  wire [31:0] acc_in_dout,
  input  wire acc_in_empty_n,
  output wire acc_in_read
);
  carry_r1_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(_pid1_dout),
      .v0_empty_n(_pid1_empty_n),
      .v0_read(_pid1_read),
      .v1_dout(c_in_dout),
      .v1_empty_n(c_in_empty_n),
      .v1_read(c_in_read),
      .v2_din(c_out_din),
      .v2_full_n(c_out_full_n),
      .v2_write(c_out_write),
      .v3_dout(acc_in_dout),
      .v3_empty_n(acc_in_empty_n),
      .v3_read(acc_in_read));
endmodule
