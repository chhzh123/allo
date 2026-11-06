`timescale 1ns/1ps

module tap_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  output wire [15:0] c_out_din,
  input  wire c_out_full_n,
  output wire c_out_write,
  input  wire [15:0] u_in_dout,
  input  wire u_in_empty_n,
  output wire u_in_read,
  output wire [15:0] u_out_din,
  input  wire u_out_full_n,
  output wire u_out_write
);
  tap_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_din(c_out_din),
      .v0_full_n(c_out_full_n),
      .v0_write(c_out_write),
      .v1_dout(u_in_dout),
      .v1_empty_n(u_in_empty_n),
      .v1_read(u_in_read),
      .v2_din(u_out_din),
      .v2_full_n(u_out_full_n),
      .v2_write(u_out_write));
endmodule
