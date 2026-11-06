`timescale 1ns/1ps

module pack8_r0 (
  input  wire ap_clk,
  input  wire ap_rst_n,
  output wire [7:0] credit_din,
  input  wire credit_full_n,
  output wire credit_write,
  input  wire [71:0] row_in_dout,
  input  wire row_in_empty_n,
  output wire row_in_read,
  output wire [63:0] wr_data_din,
  input  wire wr_data_full_n,
  output wire wr_data_write
);
  pack8_r0_0 u (
      .ap_clk(ap_clk),
      .ap_rst(~ap_rst_n),
      .v0_din(credit_din),
      .v0_full_n(credit_full_n),
      .v0_write(credit_write),
      .v1_dout(row_in_dout),
      .v1_empty_n(row_in_empty_n),
      .v1_read(row_in_read),
      .v2_din(wr_data_din),
      .v2_full_n(wr_data_full_n),
      .v2_write(wr_data_write));
endmodule
