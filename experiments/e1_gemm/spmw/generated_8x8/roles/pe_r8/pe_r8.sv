`timescale 1ns/1ps

module pe_r8 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [7:0] west_dout,
  input  wire west_empty_n,
  output wire west_read,
  input  wire [7:0] north_dout,
  input  wire north_empty_n,
  output wire north_read,
  output wire [7:0] south_din,
  input  wire south_full_n,
  output wire south_write,
  output wire [31:0] c_din,
  input  wire c_full_n,
  output wire c_write
);
  pe_r8_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(west_dout),
      .v0_empty_n(west_empty_n),
      .v0_read(west_read),
      .v1_dout(north_dout),
      .v1_empty_n(north_empty_n),
      .v1_read(north_read),
      .v2_din(south_din),
      .v2_full_n(south_full_n),
      .v2_write(south_write),
      .v3_din(c_din),
      .v3_full_n(c_full_n),
      .v3_write(c_write));
endmodule
