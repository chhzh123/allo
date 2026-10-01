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

  wire [15:0] pe_a_in_bind_dout [0:7];
  wire pe_a_in_bind_empty_n [0:7];
  wire pe_a_in_bind_read [0:7];
  genvar g_pe_a_in_bind;
  generate
    for (g_pe_a_in_bind = 0; g_pe_a_in_bind < 8; g_pe_a_in_bind = g_pe_a_in_bind + 1) begin : gen_pe_a_in_bind
      reg [31:0] lf_pe_a_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_pe_a_in_bind <= 32'h1 + g_pe_a_in_bind;
        else if (start) lf_pe_a_in_bind <= {lf_pe_a_in_bind[30:0], lf_pe_a_in_bind[31]^lf_pe_a_in_bind[21]^lf_pe_a_in_bind[1]^lf_pe_a_in_bind[0]};
      assign pe_a_in_bind_dout[g_pe_a_in_bind] = lf_pe_a_in_bind[15:0];
      assign pe_a_in_bind_empty_n[g_pe_a_in_bind] = start;
    end
  endgenerate
  wire [7:0] pe_w_in_bind_dout [0:7];
  wire pe_w_in_bind_empty_n [0:7];
  wire pe_w_in_bind_read [0:7];
  genvar g_pe_w_in_bind;
  generate
    for (g_pe_w_in_bind = 0; g_pe_w_in_bind < 8; g_pe_w_in_bind = g_pe_w_in_bind + 1) begin : gen_pe_w_in_bind
      reg [31:0] lf_pe_w_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_pe_w_in_bind <= 32'h1 + g_pe_w_in_bind;
        else if (start) lf_pe_w_in_bind <= {lf_pe_w_in_bind[30:0], lf_pe_w_in_bind[31]^lf_pe_w_in_bind[21]^lf_pe_w_in_bind[1]^lf_pe_w_in_bind[0]};
      assign pe_w_in_bind_dout[g_pe_w_in_bind] = lf_pe_w_in_bind[7:0];
      assign pe_w_in_bind_empty_n[g_pe_w_in_bind] = start;
    end
  endgenerate
  wire [63:0] seq_op_in_bind_dout [0:0];
  wire seq_op_in_bind_empty_n [0:0];
  wire seq_op_in_bind_read [0:0];
  genvar g_seq_op_in_bind;
  generate
    for (g_seq_op_in_bind = 0; g_seq_op_in_bind < 1; g_seq_op_in_bind = g_seq_op_in_bind + 1) begin : gen_seq_op_in_bind
      reg [31:0] lf_seq_op_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_seq_op_in_bind <= 32'h1 + g_seq_op_in_bind;
        else if (start) lf_seq_op_in_bind <= {lf_seq_op_in_bind[30:0], lf_seq_op_in_bind[31]^lf_seq_op_in_bind[21]^lf_seq_op_in_bind[1]^lf_seq_op_in_bind[0]};
      assign seq_op_in_bind_dout[g_seq_op_in_bind] = {2{lf_seq_op_in_bind}};
      assign seq_op_in_bind_empty_n[g_seq_op_in_bind] = start;
    end
  endgenerate
  wire [31:0] plane8_b_in_bind_dout [0:7];
  wire plane8_b_in_bind_empty_n [0:7];
  wire plane8_b_in_bind_read [0:7];
  genvar g_plane8_b_in_bind;
  generate
    for (g_plane8_b_in_bind = 0; g_plane8_b_in_bind < 8; g_plane8_b_in_bind = g_plane8_b_in_bind + 1) begin : gen_plane8_b_in_bind
      reg [31:0] lf_plane8_b_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_plane8_b_in_bind <= 32'h1 + g_plane8_b_in_bind;
        else if (start) lf_plane8_b_in_bind <= {lf_plane8_b_in_bind[30:0], lf_plane8_b_in_bind[31]^lf_plane8_b_in_bind[21]^lf_plane8_b_in_bind[1]^lf_plane8_b_in_bind[0]};
      assign plane8_b_in_bind_dout[g_plane8_b_in_bind] = lf_plane8_b_in_bind[31:0];
      assign plane8_b_in_bind_empty_n[g_plane8_b_in_bind] = start;
    end
  endgenerate
  wire [31:0] plane8_y_out_bind_din [0:7];
  wire plane8_y_out_bind_write [0:7];
  wire plane8_y_out_bind_full_n [0:7];
  genvar g_plane8_y_out_bind;
  generate
    for (g_plane8_y_out_bind = 0; g_plane8_y_out_bind < 8; g_plane8_y_out_bind = g_plane8_y_out_bind + 1) begin : gen_plane8_y_out_bind
      assign plane8_y_out_bind_full_n[g_plane8_y_out_bind] = 1'b1;
    end
  endgenerate

  spmw_top dut (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .pe_a_in_bind_dout(pe_a_in_bind_dout),
      .pe_a_in_bind_empty_n(pe_a_in_bind_empty_n),
      .pe_a_in_bind_read(pe_a_in_bind_read),
      .pe_w_in_bind_dout(pe_w_in_bind_dout),
      .pe_w_in_bind_empty_n(pe_w_in_bind_empty_n),
      .pe_w_in_bind_read(pe_w_in_bind_read),
      .seq_op_in_bind_dout(seq_op_in_bind_dout),
      .seq_op_in_bind_empty_n(seq_op_in_bind_empty_n),
      .seq_op_in_bind_read(seq_op_in_bind_read),
      .plane8_b_in_bind_dout(plane8_b_in_bind_dout),
      .plane8_b_in_bind_empty_n(plane8_b_in_bind_empty_n),
      .plane8_b_in_bind_read(plane8_b_in_bind_read),
      .plane8_y_out_bind_din(plane8_y_out_bind_din),
      .plane8_y_out_bind_write(plane8_y_out_bind_write),
      .plane8_y_out_bind_full_n(plane8_y_out_bind_full_n));

  // Fold every output into one register: a dangling result is a result
  // synthesis is entitled to delete.
  always @(posedge ap_clk)
    if (!ap_rst_n) sig <= 32'b0;
    else sig <= sig
      ^ {31'b0, pe_a_in_bind_read[0]}
      ^ {31'b0, pe_w_in_bind_read[0]}
      ^ {31'b0, seq_op_in_bind_read[0]}
      ^ {31'b0, plane8_b_in_bind_read[0]}
      ^ plane8_y_out_bind_din[0][31:0]
      ^ {31'b0, plane8_y_out_bind_write[0]}
      ^ plane8_y_out_bind_din[1][31:0]
      ^ {31'b0, plane8_y_out_bind_write[1]}
      ^ plane8_y_out_bind_din[2][31:0]
      ^ {31'b0, plane8_y_out_bind_write[2]}
      ^ plane8_y_out_bind_din[3][31:0]
      ^ {31'b0, plane8_y_out_bind_write[3]}
      ^ plane8_y_out_bind_din[4][31:0]
      ^ {31'b0, plane8_y_out_bind_write[4]}
      ^ plane8_y_out_bind_din[5][31:0]
      ^ {31'b0, plane8_y_out_bind_write[5]}
      ^ plane8_y_out_bind_din[6][31:0]
      ^ {31'b0, plane8_y_out_bind_write[6]}
      ^ plane8_y_out_bind_din[7][31:0]
      ^ {31'b0, plane8_y_out_bind_write[7]}
      ;
endmodule
