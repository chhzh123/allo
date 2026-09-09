`timescale 1ns/1ps

module drain_r2 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [31:0] own_dout,
  input  wire own_empty_n,
  output wire own_read,
  output wire [31:0] west_out_din,
  input  wire west_out_full_n,
  output wire west_out_write
);
  drain_r2_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(own_dout),
      .v0_empty_n(own_empty_n),
      .v0_read(own_read),
      .v1_din(west_out_din),
      .v1_full_n(west_out_full_n),
      .v1_write(west_out_write));
endmodule
