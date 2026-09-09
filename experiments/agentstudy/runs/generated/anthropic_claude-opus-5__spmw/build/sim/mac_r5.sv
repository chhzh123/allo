`timescale 1ns/1ps

module mac_r5 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [7:0] a_in_dout,
  input  wire a_in_empty_n,
  output wire a_in_read,
  input  wire [7:0] b_in_dout,
  input  wire b_in_empty_n,
  output wire b_in_read,
  output wire [31:0] acc_out_din,
  input  wire acc_out_full_n,
  output wire acc_out_write
);
  mac_r5_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(a_in_dout),
      .v0_empty_n(a_in_empty_n),
      .v0_read(a_in_read),
      .v1_dout(b_in_dout),
      .v1_empty_n(b_in_empty_n),
      .v1_read(b_in_read),
      .v2_din(acc_out_din),
      .v2_full_n(acc_out_full_n),
      .v2_write(acc_out_write));
endmodule
