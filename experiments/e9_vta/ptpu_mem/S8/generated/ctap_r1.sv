`timescale 1ns/1ps

module ctap_r1 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [71:0] r_in_dout,
  input  wire r_in_empty_n,
  output wire r_in_read,
  output wire [71:0] r_out_din,
  input  wire r_out_full_n,
  output wire r_out_write,
  input  wire [15:0] y_in_dout,
  input  wire y_in_empty_n,
  output wire y_in_read,
  input  wire [31:0] _pid0_dout,
  input  wire _pid0_empty_n,
  output wire _pid0_read
);
  ctap_r1_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(r_in_dout),
      .v0_empty_n(r_in_empty_n),
      .v0_read(r_in_read),
      .v1_din(r_out_din),
      .v1_full_n(r_out_full_n),
      .v1_write(r_out_write),
      .v2_dout(y_in_dout),
      .v2_empty_n(y_in_empty_n),
      .v2_read(y_in_read),
      .v3_dout(_pid0_dout),
      .v3_empty_n(_pid0_empty_n),
      .v3_read(_pid0_read));
endmodule
