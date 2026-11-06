`timescale 1ns/1ps

module wreq16_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  output wire [63:0] done_din,
  input  wire done_full_n,
  output wire done_write,
  input  wire [63:0] wr_ack_dout,
  input  wire wr_ack_empty_n,
  output wire wr_ack_read,
  output wire [63:0] wr_cmd_din,
  input  wire wr_cmd_full_n,
  output wire wr_cmd_write,
  input  wire [63:0] y_cmd_dout,
  input  wire y_cmd_empty_n,
  output wire y_cmd_read
);
  wreq16_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_din(done_din),
      .v0_full_n(done_full_n),
      .v0_write(done_write),
      .v1_dout(wr_ack_dout),
      .v1_empty_n(wr_ack_empty_n),
      .v1_read(wr_ack_read),
      .v2_din(wr_cmd_din),
      .v2_full_n(wr_cmd_full_n),
      .v2_write(wr_cmd_write),
      .v3_dout(y_cmd_dout),
      .v3_empty_n(y_cmd_empty_n),
      .v3_read(y_cmd_read));
endmodule
