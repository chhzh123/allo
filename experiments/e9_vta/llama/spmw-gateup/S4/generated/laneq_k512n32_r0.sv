`timescale 1ns/1ps

module laneq_k512n32_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] b_dout,
  input  wire b_empty_n,
  output wire b_read,
  output wire [31:0] y_out_din,
  input  wire y_out_full_n,
  output wire y_out_write,
  input  wire [31:0] z_in_dout,
  input  wire z_in_empty_n,
  output wire z_in_read
);
  laneq_k512n32_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(b_dout),
      .v0_empty_n(b_empty_n),
      .v0_read(b_read),
      .v1_din(y_out_din),
      .v1_full_n(y_out_full_n),
      .v1_write(y_out_write),
      .v2_dout(z_in_dout),
      .v2_empty_n(z_in_empty_n),
      .v2_read(z_in_read));
endmodule
