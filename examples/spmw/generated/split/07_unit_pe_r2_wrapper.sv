`timescale 1ns/1ps

module pe_r2 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  output wire [31:0] c_din,
  input  wire c_full_n,
  output wire c_write,
  output wire [7:0] east_din,
  input  wire east_full_n,
  output wire east_write,
  input  wire [7:0] north_dout,
  input  wire north_empty_n,
  output wire north_read,
  input  wire [7:0] west_dout,
  input  wire west_empty_n,
  output wire west_read
);
  pe_r2_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_din(c_din),
      .v0_full_n(c_full_n),
      .v0_write(c_write),
      .v1_din(east_din),
      .v1_full_n(east_full_n),
      .v1_write(east_write),
      .v2_dout(north_dout),
      .v2_empty_n(north_empty_n),
      .v2_read(north_read),
      .v3_dout(west_dout),
      .v3_empty_n(west_empty_n),
      .v3_read(west_read));
endmodule
