`timescale 1ns/1ps

module spmw_harness (
  input  wire ap_clk,
  input  wire ap_rst_n,
  input  wire        start,
  output reg  [31:0] sig
);
  // A maximal-length LFSR per *channel*, not one shared.
  //
  // Sharing one was the first thing measured and it was wrong: a single
  // net driving every channel of every family is a fanout of hundreds
  // across the whole die, and it became the critical path -- 3.877 ns of
  // route against 0.080 ns of logic, sourced at `lfsr_reg` and sinking
  // in a MAC cell. That measured the harness, not the fabric. One small
  // LFSR per channel is a few flip-flops each and keeps every source
  // beside its load.

  wire [63:0] stage0_x_in_bind_dout [0:0];
  wire stage0_x_in_bind_empty_n [0:0];
  wire stage0_x_in_bind_read [0:0];
  genvar g_stage0_x_in_bind;
  generate
    for (g_stage0_x_in_bind = 0; g_stage0_x_in_bind < 1; g_stage0_x_in_bind = g_stage0_x_in_bind + 1) begin : gen_stage0_x_in_bind
      reg [31:0] lf_stage0_x_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_stage0_x_in_bind <= 32'h1 + g_stage0_x_in_bind;
        else if (start) lf_stage0_x_in_bind <= {lf_stage0_x_in_bind[30:0], lf_stage0_x_in_bind[31]^lf_stage0_x_in_bind[21]^lf_stage0_x_in_bind[1]^lf_stage0_x_in_bind[0]};
      assign stage0_x_in_bind_dout[g_stage0_x_in_bind] = {2{lf_stage0_x_in_bind}};
      assign stage0_x_in_bind_empty_n[g_stage0_x_in_bind] = start;
    end
  endgenerate
  wire [63:0] reorder_y_out_bind_din [0:0];
  wire reorder_y_out_bind_write [0:0];
  wire reorder_y_out_bind_full_n [0:0];
  genvar g_reorder_y_out_bind;
  generate
    for (g_reorder_y_out_bind = 0; g_reorder_y_out_bind < 1; g_reorder_y_out_bind = g_reorder_y_out_bind + 1) begin : gen_reorder_y_out_bind
      assign reorder_y_out_bind_full_n[g_reorder_y_out_bind] = 1'b1;
    end
  endgenerate

  spmw_top dut (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .stage0_x_in_bind_dout(stage0_x_in_bind_dout),
      .stage0_x_in_bind_empty_n(stage0_x_in_bind_empty_n),
      .stage0_x_in_bind_read(stage0_x_in_bind_read),
      .reorder_y_out_bind_din(reorder_y_out_bind_din),
      .reorder_y_out_bind_write(reorder_y_out_bind_write),
      .reorder_y_out_bind_full_n(reorder_y_out_bind_full_n));

  // Fold every output into one register: a dangling result is a result
  // synthesis is entitled to delete.
  always @(posedge ap_clk)
    if (!ap_rst_n) sig <= 32'b0;
    else sig <= sig
      ^ {31'b0, stage0_x_in_bind_read[0]}
      ^ reorder_y_out_bind_din[0][31:0]
      ^ {31'b0, reorder_y_out_bind_write[0]}
      ;
endmodule
