`timescale 1ns/1ps

module pe_1_r2 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  output wire [31:0] c_din,
  input  wire c_full_n,
  output wire c_write,
  input  wire [31:0] north_dout,
  input  wire north_empty_n,
  output wire north_read,
  input  wire [31:0] west_dout,
  input  wire west_empty_n,
  output wire west_read
);
  pe_1_r2_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_din(c_din),
      .v0_full_n(c_full_n),
      .v0_write(c_write),
      .v1_dout(north_dout),
      .v1_empty_n(north_empty_n),
      .v1_read(north_read),
      .v2_dout(west_dout),
      .v2_empty_n(west_empty_n),
      .v2_read(west_read));
endmodule
