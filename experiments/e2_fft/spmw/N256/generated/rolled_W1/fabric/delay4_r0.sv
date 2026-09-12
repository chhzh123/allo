`timescale 1ns/1ps

module delay4_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] _pid0_dout,
  input  wire _pid0_empty_n,
  output wire _pid0_read,
  input  wire [63:0] x_in_dout,
  input  wire x_in_empty_n,
  output wire x_in_read,
  output wire [63:0] x_out_din,
  input  wire x_out_full_n,
  output wire x_out_write
);
  delay4_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(_pid0_dout),
      .v0_empty_n(_pid0_empty_n),
      .v0_read(_pid0_read),
      .v1_dout(x_in_dout),
      .v1_empty_n(x_in_empty_n),
      .v1_read(x_in_read),
      .v2_din(x_out_din),
      .v2_full_n(x_out_full_n),
      .v2_write(x_out_write));
endmodule
