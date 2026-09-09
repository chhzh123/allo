`timescale 1ns/1ps

module pe_r2 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [15:0] x_dout,
  input  wire x_empty_n,
  output wire x_read,
  input  wire [127:0] w_dout,
  input  wire w_empty_n,
  output wire w_read,
  output wire [31:0] p0_out_din,
  input  wire p0_out_full_n,
  output wire p0_out_write,
  output wire [31:0] p1_out_din,
  input  wire p1_out_full_n,
  output wire p1_out_write
);
  pe_r2_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(x_dout),
      .v0_empty_n(x_empty_n),
      .v0_read(x_read),
      .v1_dout(w_dout),
      .v1_empty_n(w_empty_n),
      .v1_read(w_read),
      .v2_din(p0_out_din),
      .v2_full_n(p0_out_full_n),
      .v2_write(p0_out_write),
      .v3_din(p1_out_din),
      .v3_full_n(p1_out_full_n),
      .v3_write(p1_out_write));
endmodule
