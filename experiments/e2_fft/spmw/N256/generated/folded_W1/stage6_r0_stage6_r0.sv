`timescale 1ns/1ps

module stage6_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] x_in_dout,
  input  wire x_in_empty_n,
  output wire x_in_read,
  output wire [63:0] x_out_din,
  input  wire x_out_full_n,
  output wire x_out_write
);
  stage6_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(x_in_dout),
      .v0_empty_n(x_in_empty_n),
      .v0_read(x_in_read),
      .v1_din(x_out_din),
      .v1_full_n(x_out_full_n),
      .v1_write(x_out_write));
endmodule
