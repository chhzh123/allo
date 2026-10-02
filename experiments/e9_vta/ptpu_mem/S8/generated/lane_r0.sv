`timescale 1ns/1ps

module lane_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire [63:0] c_in_dout,
  input  wire c_in_empty_n,
  output wire c_in_read,
  output wire [15:0] y_out_din,
  input  wire y_out_full_n,
  output wire y_out_write,
  input  wire [31:0] z_in_dout,
  input  wire z_in_empty_n,
  output wire z_in_read,
  input  wire [31:0] _pid0_dout,
  input  wire _pid0_empty_n,
  output wire _pid0_read
);
  lane_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_dout(c_in_dout),
      .v0_empty_n(c_in_empty_n),
      .v0_read(c_in_read),
      .v1_din(y_out_din),
      .v1_full_n(y_out_full_n),
      .v1_write(y_out_write),
      .v2_dout(z_in_dout),
      .v2_empty_n(z_in_empty_n),
      .v2_read(z_in_read),
      .v3_dout(_pid0_dout),
      .v3_empty_n(_pid0_empty_n),
      .v3_read(_pid0_read));
endmodule
