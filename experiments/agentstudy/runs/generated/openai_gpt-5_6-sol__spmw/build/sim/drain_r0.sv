`timescale 1ns/1ps

module drain_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] _pid1_dout,
  input  wire _pid1_empty_n,
  output wire _pid1_read,
  input  wire [31:0] own_dout,
  input  wire own_empty_n,
  output wire own_read,
  output wire [31:0] west_out_din,
  input  wire west_out_full_n,
  output wire west_out_write,
  input  wire [31:0] east_in_dout,
  input  wire east_in_empty_n,
  output wire east_in_read
);
  drain_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(_pid1_dout),
      .v0_empty_n(_pid1_empty_n),
      .v0_read(_pid1_read),
      .v1_dout(own_dout),
      .v1_empty_n(own_empty_n),
      .v1_read(own_read),
      .v2_din(west_out_din),
      .v2_full_n(west_out_full_n),
      .v2_write(west_out_write),
      .v3_dout(east_in_dout),
      .v3_empty_n(east_in_empty_n),
      .v3_read(east_in_read));
endmodule
