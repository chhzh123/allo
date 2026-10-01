`timescale 1ns/1ps

module plane16_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] b_in_dout,
  input  wire b_in_empty_n,
  output wire b_in_read,
  input  wire [15:0] c_in_dout,
  input  wire c_in_empty_n,
  output wire c_in_read,
  output wire [31:0] y_out_din,
  input  wire y_out_full_n,
  output wire y_out_write,
  input  wire [31:0] z_in_dout,
  input  wire z_in_empty_n,
  output wire z_in_read
);
  plane16_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(b_in_dout),
      .v0_empty_n(b_in_empty_n),
      .v0_read(b_in_read),
      .v1_dout(c_in_dout),
      .v1_empty_n(c_in_empty_n),
      .v1_read(c_in_read),
      .v2_din(y_out_din),
      .v2_full_n(y_out_full_n),
      .v2_write(y_out_write),
      .v3_dout(z_in_dout),
      .v3_empty_n(z_in_empty_n),
      .v3_read(z_in_read));
endmodule
