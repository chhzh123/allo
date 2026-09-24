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

  wire [7:0] blk16_a0_in_bind_dout [0:0];
  wire blk16_a0_in_bind_empty_n [0:0];
  wire blk16_a0_in_bind_read [0:0];
  genvar g_blk16_a0_in_bind;
  generate
    for (g_blk16_a0_in_bind = 0; g_blk16_a0_in_bind < 1; g_blk16_a0_in_bind = g_blk16_a0_in_bind + 1) begin : gen_blk16_a0_in_bind
      reg [31:0] lf_blk16_a0_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a0_in_bind <= 32'h1 + g_blk16_a0_in_bind;
        else if (start) lf_blk16_a0_in_bind <= {lf_blk16_a0_in_bind[30:0], lf_blk16_a0_in_bind[31]^lf_blk16_a0_in_bind[21]^lf_blk16_a0_in_bind[1]^lf_blk16_a0_in_bind[0]};
      assign blk16_a0_in_bind_dout[g_blk16_a0_in_bind] = lf_blk16_a0_in_bind[7:0];
      assign blk16_a0_in_bind_empty_n[g_blk16_a0_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w0_in_bind_dout [0:0];
  wire blk16_w0_in_bind_empty_n [0:0];
  wire blk16_w0_in_bind_read [0:0];
  genvar g_blk16_w0_in_bind;
  generate
    for (g_blk16_w0_in_bind = 0; g_blk16_w0_in_bind < 1; g_blk16_w0_in_bind = g_blk16_w0_in_bind + 1) begin : gen_blk16_w0_in_bind
      reg [31:0] lf_blk16_w0_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w0_in_bind <= 32'h1 + g_blk16_w0_in_bind;
        else if (start) lf_blk16_w0_in_bind <= {lf_blk16_w0_in_bind[30:0], lf_blk16_w0_in_bind[31]^lf_blk16_w0_in_bind[21]^lf_blk16_w0_in_bind[1]^lf_blk16_w0_in_bind[0]};
      assign blk16_w0_in_bind_dout[g_blk16_w0_in_bind] = lf_blk16_w0_in_bind[7:0];
      assign blk16_w0_in_bind_empty_n[g_blk16_w0_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p0_in_bind_dout [0:0];
  wire blk16_p0_in_bind_empty_n [0:0];
  wire blk16_p0_in_bind_read [0:0];
  genvar g_blk16_p0_in_bind;
  generate
    for (g_blk16_p0_in_bind = 0; g_blk16_p0_in_bind < 1; g_blk16_p0_in_bind = g_blk16_p0_in_bind + 1) begin : gen_blk16_p0_in_bind
      reg [31:0] lf_blk16_p0_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p0_in_bind <= 32'h1 + g_blk16_p0_in_bind;
        else if (start) lf_blk16_p0_in_bind <= {lf_blk16_p0_in_bind[30:0], lf_blk16_p0_in_bind[31]^lf_blk16_p0_in_bind[21]^lf_blk16_p0_in_bind[1]^lf_blk16_p0_in_bind[0]};
      assign blk16_p0_in_bind_dout[g_blk16_p0_in_bind] = lf_blk16_p0_in_bind[31:0];
      assign blk16_p0_in_bind_empty_n[g_blk16_p0_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a1_in_bind_dout [0:0];
  wire blk16_a1_in_bind_empty_n [0:0];
  wire blk16_a1_in_bind_read [0:0];
  genvar g_blk16_a1_in_bind;
  generate
    for (g_blk16_a1_in_bind = 0; g_blk16_a1_in_bind < 1; g_blk16_a1_in_bind = g_blk16_a1_in_bind + 1) begin : gen_blk16_a1_in_bind
      reg [31:0] lf_blk16_a1_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a1_in_bind <= 32'h1 + g_blk16_a1_in_bind;
        else if (start) lf_blk16_a1_in_bind <= {lf_blk16_a1_in_bind[30:0], lf_blk16_a1_in_bind[31]^lf_blk16_a1_in_bind[21]^lf_blk16_a1_in_bind[1]^lf_blk16_a1_in_bind[0]};
      assign blk16_a1_in_bind_dout[g_blk16_a1_in_bind] = lf_blk16_a1_in_bind[7:0];
      assign blk16_a1_in_bind_empty_n[g_blk16_a1_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w1_in_bind_dout [0:0];
  wire blk16_w1_in_bind_empty_n [0:0];
  wire blk16_w1_in_bind_read [0:0];
  genvar g_blk16_w1_in_bind;
  generate
    for (g_blk16_w1_in_bind = 0; g_blk16_w1_in_bind < 1; g_blk16_w1_in_bind = g_blk16_w1_in_bind + 1) begin : gen_blk16_w1_in_bind
      reg [31:0] lf_blk16_w1_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w1_in_bind <= 32'h1 + g_blk16_w1_in_bind;
        else if (start) lf_blk16_w1_in_bind <= {lf_blk16_w1_in_bind[30:0], lf_blk16_w1_in_bind[31]^lf_blk16_w1_in_bind[21]^lf_blk16_w1_in_bind[1]^lf_blk16_w1_in_bind[0]};
      assign blk16_w1_in_bind_dout[g_blk16_w1_in_bind] = lf_blk16_w1_in_bind[7:0];
      assign blk16_w1_in_bind_empty_n[g_blk16_w1_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p1_in_bind_dout [0:0];
  wire blk16_p1_in_bind_empty_n [0:0];
  wire blk16_p1_in_bind_read [0:0];
  genvar g_blk16_p1_in_bind;
  generate
    for (g_blk16_p1_in_bind = 0; g_blk16_p1_in_bind < 1; g_blk16_p1_in_bind = g_blk16_p1_in_bind + 1) begin : gen_blk16_p1_in_bind
      reg [31:0] lf_blk16_p1_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p1_in_bind <= 32'h1 + g_blk16_p1_in_bind;
        else if (start) lf_blk16_p1_in_bind <= {lf_blk16_p1_in_bind[30:0], lf_blk16_p1_in_bind[31]^lf_blk16_p1_in_bind[21]^lf_blk16_p1_in_bind[1]^lf_blk16_p1_in_bind[0]};
      assign blk16_p1_in_bind_dout[g_blk16_p1_in_bind] = lf_blk16_p1_in_bind[31:0];
      assign blk16_p1_in_bind_empty_n[g_blk16_p1_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a2_in_bind_dout [0:0];
  wire blk16_a2_in_bind_empty_n [0:0];
  wire blk16_a2_in_bind_read [0:0];
  genvar g_blk16_a2_in_bind;
  generate
    for (g_blk16_a2_in_bind = 0; g_blk16_a2_in_bind < 1; g_blk16_a2_in_bind = g_blk16_a2_in_bind + 1) begin : gen_blk16_a2_in_bind
      reg [31:0] lf_blk16_a2_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a2_in_bind <= 32'h1 + g_blk16_a2_in_bind;
        else if (start) lf_blk16_a2_in_bind <= {lf_blk16_a2_in_bind[30:0], lf_blk16_a2_in_bind[31]^lf_blk16_a2_in_bind[21]^lf_blk16_a2_in_bind[1]^lf_blk16_a2_in_bind[0]};
      assign blk16_a2_in_bind_dout[g_blk16_a2_in_bind] = lf_blk16_a2_in_bind[7:0];
      assign blk16_a2_in_bind_empty_n[g_blk16_a2_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w2_in_bind_dout [0:0];
  wire blk16_w2_in_bind_empty_n [0:0];
  wire blk16_w2_in_bind_read [0:0];
  genvar g_blk16_w2_in_bind;
  generate
    for (g_blk16_w2_in_bind = 0; g_blk16_w2_in_bind < 1; g_blk16_w2_in_bind = g_blk16_w2_in_bind + 1) begin : gen_blk16_w2_in_bind
      reg [31:0] lf_blk16_w2_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w2_in_bind <= 32'h1 + g_blk16_w2_in_bind;
        else if (start) lf_blk16_w2_in_bind <= {lf_blk16_w2_in_bind[30:0], lf_blk16_w2_in_bind[31]^lf_blk16_w2_in_bind[21]^lf_blk16_w2_in_bind[1]^lf_blk16_w2_in_bind[0]};
      assign blk16_w2_in_bind_dout[g_blk16_w2_in_bind] = lf_blk16_w2_in_bind[7:0];
      assign blk16_w2_in_bind_empty_n[g_blk16_w2_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p2_in_bind_dout [0:0];
  wire blk16_p2_in_bind_empty_n [0:0];
  wire blk16_p2_in_bind_read [0:0];
  genvar g_blk16_p2_in_bind;
  generate
    for (g_blk16_p2_in_bind = 0; g_blk16_p2_in_bind < 1; g_blk16_p2_in_bind = g_blk16_p2_in_bind + 1) begin : gen_blk16_p2_in_bind
      reg [31:0] lf_blk16_p2_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p2_in_bind <= 32'h1 + g_blk16_p2_in_bind;
        else if (start) lf_blk16_p2_in_bind <= {lf_blk16_p2_in_bind[30:0], lf_blk16_p2_in_bind[31]^lf_blk16_p2_in_bind[21]^lf_blk16_p2_in_bind[1]^lf_blk16_p2_in_bind[0]};
      assign blk16_p2_in_bind_dout[g_blk16_p2_in_bind] = lf_blk16_p2_in_bind[31:0];
      assign blk16_p2_in_bind_empty_n[g_blk16_p2_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a3_in_bind_dout [0:0];
  wire blk16_a3_in_bind_empty_n [0:0];
  wire blk16_a3_in_bind_read [0:0];
  genvar g_blk16_a3_in_bind;
  generate
    for (g_blk16_a3_in_bind = 0; g_blk16_a3_in_bind < 1; g_blk16_a3_in_bind = g_blk16_a3_in_bind + 1) begin : gen_blk16_a3_in_bind
      reg [31:0] lf_blk16_a3_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a3_in_bind <= 32'h1 + g_blk16_a3_in_bind;
        else if (start) lf_blk16_a3_in_bind <= {lf_blk16_a3_in_bind[30:0], lf_blk16_a3_in_bind[31]^lf_blk16_a3_in_bind[21]^lf_blk16_a3_in_bind[1]^lf_blk16_a3_in_bind[0]};
      assign blk16_a3_in_bind_dout[g_blk16_a3_in_bind] = lf_blk16_a3_in_bind[7:0];
      assign blk16_a3_in_bind_empty_n[g_blk16_a3_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w3_in_bind_dout [0:0];
  wire blk16_w3_in_bind_empty_n [0:0];
  wire blk16_w3_in_bind_read [0:0];
  genvar g_blk16_w3_in_bind;
  generate
    for (g_blk16_w3_in_bind = 0; g_blk16_w3_in_bind < 1; g_blk16_w3_in_bind = g_blk16_w3_in_bind + 1) begin : gen_blk16_w3_in_bind
      reg [31:0] lf_blk16_w3_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w3_in_bind <= 32'h1 + g_blk16_w3_in_bind;
        else if (start) lf_blk16_w3_in_bind <= {lf_blk16_w3_in_bind[30:0], lf_blk16_w3_in_bind[31]^lf_blk16_w3_in_bind[21]^lf_blk16_w3_in_bind[1]^lf_blk16_w3_in_bind[0]};
      assign blk16_w3_in_bind_dout[g_blk16_w3_in_bind] = lf_blk16_w3_in_bind[7:0];
      assign blk16_w3_in_bind_empty_n[g_blk16_w3_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p3_in_bind_dout [0:0];
  wire blk16_p3_in_bind_empty_n [0:0];
  wire blk16_p3_in_bind_read [0:0];
  genvar g_blk16_p3_in_bind;
  generate
    for (g_blk16_p3_in_bind = 0; g_blk16_p3_in_bind < 1; g_blk16_p3_in_bind = g_blk16_p3_in_bind + 1) begin : gen_blk16_p3_in_bind
      reg [31:0] lf_blk16_p3_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p3_in_bind <= 32'h1 + g_blk16_p3_in_bind;
        else if (start) lf_blk16_p3_in_bind <= {lf_blk16_p3_in_bind[30:0], lf_blk16_p3_in_bind[31]^lf_blk16_p3_in_bind[21]^lf_blk16_p3_in_bind[1]^lf_blk16_p3_in_bind[0]};
      assign blk16_p3_in_bind_dout[g_blk16_p3_in_bind] = lf_blk16_p3_in_bind[31:0];
      assign blk16_p3_in_bind_empty_n[g_blk16_p3_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a4_in_bind_dout [0:0];
  wire blk16_a4_in_bind_empty_n [0:0];
  wire blk16_a4_in_bind_read [0:0];
  genvar g_blk16_a4_in_bind;
  generate
    for (g_blk16_a4_in_bind = 0; g_blk16_a4_in_bind < 1; g_blk16_a4_in_bind = g_blk16_a4_in_bind + 1) begin : gen_blk16_a4_in_bind
      reg [31:0] lf_blk16_a4_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a4_in_bind <= 32'h1 + g_blk16_a4_in_bind;
        else if (start) lf_blk16_a4_in_bind <= {lf_blk16_a4_in_bind[30:0], lf_blk16_a4_in_bind[31]^lf_blk16_a4_in_bind[21]^lf_blk16_a4_in_bind[1]^lf_blk16_a4_in_bind[0]};
      assign blk16_a4_in_bind_dout[g_blk16_a4_in_bind] = lf_blk16_a4_in_bind[7:0];
      assign blk16_a4_in_bind_empty_n[g_blk16_a4_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w4_in_bind_dout [0:0];
  wire blk16_w4_in_bind_empty_n [0:0];
  wire blk16_w4_in_bind_read [0:0];
  genvar g_blk16_w4_in_bind;
  generate
    for (g_blk16_w4_in_bind = 0; g_blk16_w4_in_bind < 1; g_blk16_w4_in_bind = g_blk16_w4_in_bind + 1) begin : gen_blk16_w4_in_bind
      reg [31:0] lf_blk16_w4_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w4_in_bind <= 32'h1 + g_blk16_w4_in_bind;
        else if (start) lf_blk16_w4_in_bind <= {lf_blk16_w4_in_bind[30:0], lf_blk16_w4_in_bind[31]^lf_blk16_w4_in_bind[21]^lf_blk16_w4_in_bind[1]^lf_blk16_w4_in_bind[0]};
      assign blk16_w4_in_bind_dout[g_blk16_w4_in_bind] = lf_blk16_w4_in_bind[7:0];
      assign blk16_w4_in_bind_empty_n[g_blk16_w4_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p4_in_bind_dout [0:0];
  wire blk16_p4_in_bind_empty_n [0:0];
  wire blk16_p4_in_bind_read [0:0];
  genvar g_blk16_p4_in_bind;
  generate
    for (g_blk16_p4_in_bind = 0; g_blk16_p4_in_bind < 1; g_blk16_p4_in_bind = g_blk16_p4_in_bind + 1) begin : gen_blk16_p4_in_bind
      reg [31:0] lf_blk16_p4_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p4_in_bind <= 32'h1 + g_blk16_p4_in_bind;
        else if (start) lf_blk16_p4_in_bind <= {lf_blk16_p4_in_bind[30:0], lf_blk16_p4_in_bind[31]^lf_blk16_p4_in_bind[21]^lf_blk16_p4_in_bind[1]^lf_blk16_p4_in_bind[0]};
      assign blk16_p4_in_bind_dout[g_blk16_p4_in_bind] = lf_blk16_p4_in_bind[31:0];
      assign blk16_p4_in_bind_empty_n[g_blk16_p4_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a5_in_bind_dout [0:0];
  wire blk16_a5_in_bind_empty_n [0:0];
  wire blk16_a5_in_bind_read [0:0];
  genvar g_blk16_a5_in_bind;
  generate
    for (g_blk16_a5_in_bind = 0; g_blk16_a5_in_bind < 1; g_blk16_a5_in_bind = g_blk16_a5_in_bind + 1) begin : gen_blk16_a5_in_bind
      reg [31:0] lf_blk16_a5_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a5_in_bind <= 32'h1 + g_blk16_a5_in_bind;
        else if (start) lf_blk16_a5_in_bind <= {lf_blk16_a5_in_bind[30:0], lf_blk16_a5_in_bind[31]^lf_blk16_a5_in_bind[21]^lf_blk16_a5_in_bind[1]^lf_blk16_a5_in_bind[0]};
      assign blk16_a5_in_bind_dout[g_blk16_a5_in_bind] = lf_blk16_a5_in_bind[7:0];
      assign blk16_a5_in_bind_empty_n[g_blk16_a5_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w5_in_bind_dout [0:0];
  wire blk16_w5_in_bind_empty_n [0:0];
  wire blk16_w5_in_bind_read [0:0];
  genvar g_blk16_w5_in_bind;
  generate
    for (g_blk16_w5_in_bind = 0; g_blk16_w5_in_bind < 1; g_blk16_w5_in_bind = g_blk16_w5_in_bind + 1) begin : gen_blk16_w5_in_bind
      reg [31:0] lf_blk16_w5_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w5_in_bind <= 32'h1 + g_blk16_w5_in_bind;
        else if (start) lf_blk16_w5_in_bind <= {lf_blk16_w5_in_bind[30:0], lf_blk16_w5_in_bind[31]^lf_blk16_w5_in_bind[21]^lf_blk16_w5_in_bind[1]^lf_blk16_w5_in_bind[0]};
      assign blk16_w5_in_bind_dout[g_blk16_w5_in_bind] = lf_blk16_w5_in_bind[7:0];
      assign blk16_w5_in_bind_empty_n[g_blk16_w5_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p5_in_bind_dout [0:0];
  wire blk16_p5_in_bind_empty_n [0:0];
  wire blk16_p5_in_bind_read [0:0];
  genvar g_blk16_p5_in_bind;
  generate
    for (g_blk16_p5_in_bind = 0; g_blk16_p5_in_bind < 1; g_blk16_p5_in_bind = g_blk16_p5_in_bind + 1) begin : gen_blk16_p5_in_bind
      reg [31:0] lf_blk16_p5_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p5_in_bind <= 32'h1 + g_blk16_p5_in_bind;
        else if (start) lf_blk16_p5_in_bind <= {lf_blk16_p5_in_bind[30:0], lf_blk16_p5_in_bind[31]^lf_blk16_p5_in_bind[21]^lf_blk16_p5_in_bind[1]^lf_blk16_p5_in_bind[0]};
      assign blk16_p5_in_bind_dout[g_blk16_p5_in_bind] = lf_blk16_p5_in_bind[31:0];
      assign blk16_p5_in_bind_empty_n[g_blk16_p5_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a6_in_bind_dout [0:0];
  wire blk16_a6_in_bind_empty_n [0:0];
  wire blk16_a6_in_bind_read [0:0];
  genvar g_blk16_a6_in_bind;
  generate
    for (g_blk16_a6_in_bind = 0; g_blk16_a6_in_bind < 1; g_blk16_a6_in_bind = g_blk16_a6_in_bind + 1) begin : gen_blk16_a6_in_bind
      reg [31:0] lf_blk16_a6_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a6_in_bind <= 32'h1 + g_blk16_a6_in_bind;
        else if (start) lf_blk16_a6_in_bind <= {lf_blk16_a6_in_bind[30:0], lf_blk16_a6_in_bind[31]^lf_blk16_a6_in_bind[21]^lf_blk16_a6_in_bind[1]^lf_blk16_a6_in_bind[0]};
      assign blk16_a6_in_bind_dout[g_blk16_a6_in_bind] = lf_blk16_a6_in_bind[7:0];
      assign blk16_a6_in_bind_empty_n[g_blk16_a6_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w6_in_bind_dout [0:0];
  wire blk16_w6_in_bind_empty_n [0:0];
  wire blk16_w6_in_bind_read [0:0];
  genvar g_blk16_w6_in_bind;
  generate
    for (g_blk16_w6_in_bind = 0; g_blk16_w6_in_bind < 1; g_blk16_w6_in_bind = g_blk16_w6_in_bind + 1) begin : gen_blk16_w6_in_bind
      reg [31:0] lf_blk16_w6_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w6_in_bind <= 32'h1 + g_blk16_w6_in_bind;
        else if (start) lf_blk16_w6_in_bind <= {lf_blk16_w6_in_bind[30:0], lf_blk16_w6_in_bind[31]^lf_blk16_w6_in_bind[21]^lf_blk16_w6_in_bind[1]^lf_blk16_w6_in_bind[0]};
      assign blk16_w6_in_bind_dout[g_blk16_w6_in_bind] = lf_blk16_w6_in_bind[7:0];
      assign blk16_w6_in_bind_empty_n[g_blk16_w6_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p6_in_bind_dout [0:0];
  wire blk16_p6_in_bind_empty_n [0:0];
  wire blk16_p6_in_bind_read [0:0];
  genvar g_blk16_p6_in_bind;
  generate
    for (g_blk16_p6_in_bind = 0; g_blk16_p6_in_bind < 1; g_blk16_p6_in_bind = g_blk16_p6_in_bind + 1) begin : gen_blk16_p6_in_bind
      reg [31:0] lf_blk16_p6_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p6_in_bind <= 32'h1 + g_blk16_p6_in_bind;
        else if (start) lf_blk16_p6_in_bind <= {lf_blk16_p6_in_bind[30:0], lf_blk16_p6_in_bind[31]^lf_blk16_p6_in_bind[21]^lf_blk16_p6_in_bind[1]^lf_blk16_p6_in_bind[0]};
      assign blk16_p6_in_bind_dout[g_blk16_p6_in_bind] = lf_blk16_p6_in_bind[31:0];
      assign blk16_p6_in_bind_empty_n[g_blk16_p6_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a7_in_bind_dout [0:0];
  wire blk16_a7_in_bind_empty_n [0:0];
  wire blk16_a7_in_bind_read [0:0];
  genvar g_blk16_a7_in_bind;
  generate
    for (g_blk16_a7_in_bind = 0; g_blk16_a7_in_bind < 1; g_blk16_a7_in_bind = g_blk16_a7_in_bind + 1) begin : gen_blk16_a7_in_bind
      reg [31:0] lf_blk16_a7_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a7_in_bind <= 32'h1 + g_blk16_a7_in_bind;
        else if (start) lf_blk16_a7_in_bind <= {lf_blk16_a7_in_bind[30:0], lf_blk16_a7_in_bind[31]^lf_blk16_a7_in_bind[21]^lf_blk16_a7_in_bind[1]^lf_blk16_a7_in_bind[0]};
      assign blk16_a7_in_bind_dout[g_blk16_a7_in_bind] = lf_blk16_a7_in_bind[7:0];
      assign blk16_a7_in_bind_empty_n[g_blk16_a7_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w7_in_bind_dout [0:0];
  wire blk16_w7_in_bind_empty_n [0:0];
  wire blk16_w7_in_bind_read [0:0];
  genvar g_blk16_w7_in_bind;
  generate
    for (g_blk16_w7_in_bind = 0; g_blk16_w7_in_bind < 1; g_blk16_w7_in_bind = g_blk16_w7_in_bind + 1) begin : gen_blk16_w7_in_bind
      reg [31:0] lf_blk16_w7_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w7_in_bind <= 32'h1 + g_blk16_w7_in_bind;
        else if (start) lf_blk16_w7_in_bind <= {lf_blk16_w7_in_bind[30:0], lf_blk16_w7_in_bind[31]^lf_blk16_w7_in_bind[21]^lf_blk16_w7_in_bind[1]^lf_blk16_w7_in_bind[0]};
      assign blk16_w7_in_bind_dout[g_blk16_w7_in_bind] = lf_blk16_w7_in_bind[7:0];
      assign blk16_w7_in_bind_empty_n[g_blk16_w7_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p7_in_bind_dout [0:0];
  wire blk16_p7_in_bind_empty_n [0:0];
  wire blk16_p7_in_bind_read [0:0];
  genvar g_blk16_p7_in_bind;
  generate
    for (g_blk16_p7_in_bind = 0; g_blk16_p7_in_bind < 1; g_blk16_p7_in_bind = g_blk16_p7_in_bind + 1) begin : gen_blk16_p7_in_bind
      reg [31:0] lf_blk16_p7_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p7_in_bind <= 32'h1 + g_blk16_p7_in_bind;
        else if (start) lf_blk16_p7_in_bind <= {lf_blk16_p7_in_bind[30:0], lf_blk16_p7_in_bind[31]^lf_blk16_p7_in_bind[21]^lf_blk16_p7_in_bind[1]^lf_blk16_p7_in_bind[0]};
      assign blk16_p7_in_bind_dout[g_blk16_p7_in_bind] = lf_blk16_p7_in_bind[31:0];
      assign blk16_p7_in_bind_empty_n[g_blk16_p7_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a8_in_bind_dout [0:0];
  wire blk16_a8_in_bind_empty_n [0:0];
  wire blk16_a8_in_bind_read [0:0];
  genvar g_blk16_a8_in_bind;
  generate
    for (g_blk16_a8_in_bind = 0; g_blk16_a8_in_bind < 1; g_blk16_a8_in_bind = g_blk16_a8_in_bind + 1) begin : gen_blk16_a8_in_bind
      reg [31:0] lf_blk16_a8_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a8_in_bind <= 32'h1 + g_blk16_a8_in_bind;
        else if (start) lf_blk16_a8_in_bind <= {lf_blk16_a8_in_bind[30:0], lf_blk16_a8_in_bind[31]^lf_blk16_a8_in_bind[21]^lf_blk16_a8_in_bind[1]^lf_blk16_a8_in_bind[0]};
      assign blk16_a8_in_bind_dout[g_blk16_a8_in_bind] = lf_blk16_a8_in_bind[7:0];
      assign blk16_a8_in_bind_empty_n[g_blk16_a8_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w8_in_bind_dout [0:0];
  wire blk16_w8_in_bind_empty_n [0:0];
  wire blk16_w8_in_bind_read [0:0];
  genvar g_blk16_w8_in_bind;
  generate
    for (g_blk16_w8_in_bind = 0; g_blk16_w8_in_bind < 1; g_blk16_w8_in_bind = g_blk16_w8_in_bind + 1) begin : gen_blk16_w8_in_bind
      reg [31:0] lf_blk16_w8_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w8_in_bind <= 32'h1 + g_blk16_w8_in_bind;
        else if (start) lf_blk16_w8_in_bind <= {lf_blk16_w8_in_bind[30:0], lf_blk16_w8_in_bind[31]^lf_blk16_w8_in_bind[21]^lf_blk16_w8_in_bind[1]^lf_blk16_w8_in_bind[0]};
      assign blk16_w8_in_bind_dout[g_blk16_w8_in_bind] = lf_blk16_w8_in_bind[7:0];
      assign blk16_w8_in_bind_empty_n[g_blk16_w8_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p8_in_bind_dout [0:0];
  wire blk16_p8_in_bind_empty_n [0:0];
  wire blk16_p8_in_bind_read [0:0];
  genvar g_blk16_p8_in_bind;
  generate
    for (g_blk16_p8_in_bind = 0; g_blk16_p8_in_bind < 1; g_blk16_p8_in_bind = g_blk16_p8_in_bind + 1) begin : gen_blk16_p8_in_bind
      reg [31:0] lf_blk16_p8_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p8_in_bind <= 32'h1 + g_blk16_p8_in_bind;
        else if (start) lf_blk16_p8_in_bind <= {lf_blk16_p8_in_bind[30:0], lf_blk16_p8_in_bind[31]^lf_blk16_p8_in_bind[21]^lf_blk16_p8_in_bind[1]^lf_blk16_p8_in_bind[0]};
      assign blk16_p8_in_bind_dout[g_blk16_p8_in_bind] = lf_blk16_p8_in_bind[31:0];
      assign blk16_p8_in_bind_empty_n[g_blk16_p8_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a9_in_bind_dout [0:0];
  wire blk16_a9_in_bind_empty_n [0:0];
  wire blk16_a9_in_bind_read [0:0];
  genvar g_blk16_a9_in_bind;
  generate
    for (g_blk16_a9_in_bind = 0; g_blk16_a9_in_bind < 1; g_blk16_a9_in_bind = g_blk16_a9_in_bind + 1) begin : gen_blk16_a9_in_bind
      reg [31:0] lf_blk16_a9_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a9_in_bind <= 32'h1 + g_blk16_a9_in_bind;
        else if (start) lf_blk16_a9_in_bind <= {lf_blk16_a9_in_bind[30:0], lf_blk16_a9_in_bind[31]^lf_blk16_a9_in_bind[21]^lf_blk16_a9_in_bind[1]^lf_blk16_a9_in_bind[0]};
      assign blk16_a9_in_bind_dout[g_blk16_a9_in_bind] = lf_blk16_a9_in_bind[7:0];
      assign blk16_a9_in_bind_empty_n[g_blk16_a9_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w9_in_bind_dout [0:0];
  wire blk16_w9_in_bind_empty_n [0:0];
  wire blk16_w9_in_bind_read [0:0];
  genvar g_blk16_w9_in_bind;
  generate
    for (g_blk16_w9_in_bind = 0; g_blk16_w9_in_bind < 1; g_blk16_w9_in_bind = g_blk16_w9_in_bind + 1) begin : gen_blk16_w9_in_bind
      reg [31:0] lf_blk16_w9_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w9_in_bind <= 32'h1 + g_blk16_w9_in_bind;
        else if (start) lf_blk16_w9_in_bind <= {lf_blk16_w9_in_bind[30:0], lf_blk16_w9_in_bind[31]^lf_blk16_w9_in_bind[21]^lf_blk16_w9_in_bind[1]^lf_blk16_w9_in_bind[0]};
      assign blk16_w9_in_bind_dout[g_blk16_w9_in_bind] = lf_blk16_w9_in_bind[7:0];
      assign blk16_w9_in_bind_empty_n[g_blk16_w9_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p9_in_bind_dout [0:0];
  wire blk16_p9_in_bind_empty_n [0:0];
  wire blk16_p9_in_bind_read [0:0];
  genvar g_blk16_p9_in_bind;
  generate
    for (g_blk16_p9_in_bind = 0; g_blk16_p9_in_bind < 1; g_blk16_p9_in_bind = g_blk16_p9_in_bind + 1) begin : gen_blk16_p9_in_bind
      reg [31:0] lf_blk16_p9_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p9_in_bind <= 32'h1 + g_blk16_p9_in_bind;
        else if (start) lf_blk16_p9_in_bind <= {lf_blk16_p9_in_bind[30:0], lf_blk16_p9_in_bind[31]^lf_blk16_p9_in_bind[21]^lf_blk16_p9_in_bind[1]^lf_blk16_p9_in_bind[0]};
      assign blk16_p9_in_bind_dout[g_blk16_p9_in_bind] = lf_blk16_p9_in_bind[31:0];
      assign blk16_p9_in_bind_empty_n[g_blk16_p9_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a10_in_bind_dout [0:0];
  wire blk16_a10_in_bind_empty_n [0:0];
  wire blk16_a10_in_bind_read [0:0];
  genvar g_blk16_a10_in_bind;
  generate
    for (g_blk16_a10_in_bind = 0; g_blk16_a10_in_bind < 1; g_blk16_a10_in_bind = g_blk16_a10_in_bind + 1) begin : gen_blk16_a10_in_bind
      reg [31:0] lf_blk16_a10_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a10_in_bind <= 32'h1 + g_blk16_a10_in_bind;
        else if (start) lf_blk16_a10_in_bind <= {lf_blk16_a10_in_bind[30:0], lf_blk16_a10_in_bind[31]^lf_blk16_a10_in_bind[21]^lf_blk16_a10_in_bind[1]^lf_blk16_a10_in_bind[0]};
      assign blk16_a10_in_bind_dout[g_blk16_a10_in_bind] = lf_blk16_a10_in_bind[7:0];
      assign blk16_a10_in_bind_empty_n[g_blk16_a10_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w10_in_bind_dout [0:0];
  wire blk16_w10_in_bind_empty_n [0:0];
  wire blk16_w10_in_bind_read [0:0];
  genvar g_blk16_w10_in_bind;
  generate
    for (g_blk16_w10_in_bind = 0; g_blk16_w10_in_bind < 1; g_blk16_w10_in_bind = g_blk16_w10_in_bind + 1) begin : gen_blk16_w10_in_bind
      reg [31:0] lf_blk16_w10_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w10_in_bind <= 32'h1 + g_blk16_w10_in_bind;
        else if (start) lf_blk16_w10_in_bind <= {lf_blk16_w10_in_bind[30:0], lf_blk16_w10_in_bind[31]^lf_blk16_w10_in_bind[21]^lf_blk16_w10_in_bind[1]^lf_blk16_w10_in_bind[0]};
      assign blk16_w10_in_bind_dout[g_blk16_w10_in_bind] = lf_blk16_w10_in_bind[7:0];
      assign blk16_w10_in_bind_empty_n[g_blk16_w10_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p10_in_bind_dout [0:0];
  wire blk16_p10_in_bind_empty_n [0:0];
  wire blk16_p10_in_bind_read [0:0];
  genvar g_blk16_p10_in_bind;
  generate
    for (g_blk16_p10_in_bind = 0; g_blk16_p10_in_bind < 1; g_blk16_p10_in_bind = g_blk16_p10_in_bind + 1) begin : gen_blk16_p10_in_bind
      reg [31:0] lf_blk16_p10_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p10_in_bind <= 32'h1 + g_blk16_p10_in_bind;
        else if (start) lf_blk16_p10_in_bind <= {lf_blk16_p10_in_bind[30:0], lf_blk16_p10_in_bind[31]^lf_blk16_p10_in_bind[21]^lf_blk16_p10_in_bind[1]^lf_blk16_p10_in_bind[0]};
      assign blk16_p10_in_bind_dout[g_blk16_p10_in_bind] = lf_blk16_p10_in_bind[31:0];
      assign blk16_p10_in_bind_empty_n[g_blk16_p10_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a11_in_bind_dout [0:0];
  wire blk16_a11_in_bind_empty_n [0:0];
  wire blk16_a11_in_bind_read [0:0];
  genvar g_blk16_a11_in_bind;
  generate
    for (g_blk16_a11_in_bind = 0; g_blk16_a11_in_bind < 1; g_blk16_a11_in_bind = g_blk16_a11_in_bind + 1) begin : gen_blk16_a11_in_bind
      reg [31:0] lf_blk16_a11_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a11_in_bind <= 32'h1 + g_blk16_a11_in_bind;
        else if (start) lf_blk16_a11_in_bind <= {lf_blk16_a11_in_bind[30:0], lf_blk16_a11_in_bind[31]^lf_blk16_a11_in_bind[21]^lf_blk16_a11_in_bind[1]^lf_blk16_a11_in_bind[0]};
      assign blk16_a11_in_bind_dout[g_blk16_a11_in_bind] = lf_blk16_a11_in_bind[7:0];
      assign blk16_a11_in_bind_empty_n[g_blk16_a11_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w11_in_bind_dout [0:0];
  wire blk16_w11_in_bind_empty_n [0:0];
  wire blk16_w11_in_bind_read [0:0];
  genvar g_blk16_w11_in_bind;
  generate
    for (g_blk16_w11_in_bind = 0; g_blk16_w11_in_bind < 1; g_blk16_w11_in_bind = g_blk16_w11_in_bind + 1) begin : gen_blk16_w11_in_bind
      reg [31:0] lf_blk16_w11_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w11_in_bind <= 32'h1 + g_blk16_w11_in_bind;
        else if (start) lf_blk16_w11_in_bind <= {lf_blk16_w11_in_bind[30:0], lf_blk16_w11_in_bind[31]^lf_blk16_w11_in_bind[21]^lf_blk16_w11_in_bind[1]^lf_blk16_w11_in_bind[0]};
      assign blk16_w11_in_bind_dout[g_blk16_w11_in_bind] = lf_blk16_w11_in_bind[7:0];
      assign blk16_w11_in_bind_empty_n[g_blk16_w11_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p11_in_bind_dout [0:0];
  wire blk16_p11_in_bind_empty_n [0:0];
  wire blk16_p11_in_bind_read [0:0];
  genvar g_blk16_p11_in_bind;
  generate
    for (g_blk16_p11_in_bind = 0; g_blk16_p11_in_bind < 1; g_blk16_p11_in_bind = g_blk16_p11_in_bind + 1) begin : gen_blk16_p11_in_bind
      reg [31:0] lf_blk16_p11_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p11_in_bind <= 32'h1 + g_blk16_p11_in_bind;
        else if (start) lf_blk16_p11_in_bind <= {lf_blk16_p11_in_bind[30:0], lf_blk16_p11_in_bind[31]^lf_blk16_p11_in_bind[21]^lf_blk16_p11_in_bind[1]^lf_blk16_p11_in_bind[0]};
      assign blk16_p11_in_bind_dout[g_blk16_p11_in_bind] = lf_blk16_p11_in_bind[31:0];
      assign blk16_p11_in_bind_empty_n[g_blk16_p11_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a12_in_bind_dout [0:0];
  wire blk16_a12_in_bind_empty_n [0:0];
  wire blk16_a12_in_bind_read [0:0];
  genvar g_blk16_a12_in_bind;
  generate
    for (g_blk16_a12_in_bind = 0; g_blk16_a12_in_bind < 1; g_blk16_a12_in_bind = g_blk16_a12_in_bind + 1) begin : gen_blk16_a12_in_bind
      reg [31:0] lf_blk16_a12_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a12_in_bind <= 32'h1 + g_blk16_a12_in_bind;
        else if (start) lf_blk16_a12_in_bind <= {lf_blk16_a12_in_bind[30:0], lf_blk16_a12_in_bind[31]^lf_blk16_a12_in_bind[21]^lf_blk16_a12_in_bind[1]^lf_blk16_a12_in_bind[0]};
      assign blk16_a12_in_bind_dout[g_blk16_a12_in_bind] = lf_blk16_a12_in_bind[7:0];
      assign blk16_a12_in_bind_empty_n[g_blk16_a12_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w12_in_bind_dout [0:0];
  wire blk16_w12_in_bind_empty_n [0:0];
  wire blk16_w12_in_bind_read [0:0];
  genvar g_blk16_w12_in_bind;
  generate
    for (g_blk16_w12_in_bind = 0; g_blk16_w12_in_bind < 1; g_blk16_w12_in_bind = g_blk16_w12_in_bind + 1) begin : gen_blk16_w12_in_bind
      reg [31:0] lf_blk16_w12_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w12_in_bind <= 32'h1 + g_blk16_w12_in_bind;
        else if (start) lf_blk16_w12_in_bind <= {lf_blk16_w12_in_bind[30:0], lf_blk16_w12_in_bind[31]^lf_blk16_w12_in_bind[21]^lf_blk16_w12_in_bind[1]^lf_blk16_w12_in_bind[0]};
      assign blk16_w12_in_bind_dout[g_blk16_w12_in_bind] = lf_blk16_w12_in_bind[7:0];
      assign blk16_w12_in_bind_empty_n[g_blk16_w12_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p12_in_bind_dout [0:0];
  wire blk16_p12_in_bind_empty_n [0:0];
  wire blk16_p12_in_bind_read [0:0];
  genvar g_blk16_p12_in_bind;
  generate
    for (g_blk16_p12_in_bind = 0; g_blk16_p12_in_bind < 1; g_blk16_p12_in_bind = g_blk16_p12_in_bind + 1) begin : gen_blk16_p12_in_bind
      reg [31:0] lf_blk16_p12_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p12_in_bind <= 32'h1 + g_blk16_p12_in_bind;
        else if (start) lf_blk16_p12_in_bind <= {lf_blk16_p12_in_bind[30:0], lf_blk16_p12_in_bind[31]^lf_blk16_p12_in_bind[21]^lf_blk16_p12_in_bind[1]^lf_blk16_p12_in_bind[0]};
      assign blk16_p12_in_bind_dout[g_blk16_p12_in_bind] = lf_blk16_p12_in_bind[31:0];
      assign blk16_p12_in_bind_empty_n[g_blk16_p12_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a13_in_bind_dout [0:0];
  wire blk16_a13_in_bind_empty_n [0:0];
  wire blk16_a13_in_bind_read [0:0];
  genvar g_blk16_a13_in_bind;
  generate
    for (g_blk16_a13_in_bind = 0; g_blk16_a13_in_bind < 1; g_blk16_a13_in_bind = g_blk16_a13_in_bind + 1) begin : gen_blk16_a13_in_bind
      reg [31:0] lf_blk16_a13_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a13_in_bind <= 32'h1 + g_blk16_a13_in_bind;
        else if (start) lf_blk16_a13_in_bind <= {lf_blk16_a13_in_bind[30:0], lf_blk16_a13_in_bind[31]^lf_blk16_a13_in_bind[21]^lf_blk16_a13_in_bind[1]^lf_blk16_a13_in_bind[0]};
      assign blk16_a13_in_bind_dout[g_blk16_a13_in_bind] = lf_blk16_a13_in_bind[7:0];
      assign blk16_a13_in_bind_empty_n[g_blk16_a13_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w13_in_bind_dout [0:0];
  wire blk16_w13_in_bind_empty_n [0:0];
  wire blk16_w13_in_bind_read [0:0];
  genvar g_blk16_w13_in_bind;
  generate
    for (g_blk16_w13_in_bind = 0; g_blk16_w13_in_bind < 1; g_blk16_w13_in_bind = g_blk16_w13_in_bind + 1) begin : gen_blk16_w13_in_bind
      reg [31:0] lf_blk16_w13_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w13_in_bind <= 32'h1 + g_blk16_w13_in_bind;
        else if (start) lf_blk16_w13_in_bind <= {lf_blk16_w13_in_bind[30:0], lf_blk16_w13_in_bind[31]^lf_blk16_w13_in_bind[21]^lf_blk16_w13_in_bind[1]^lf_blk16_w13_in_bind[0]};
      assign blk16_w13_in_bind_dout[g_blk16_w13_in_bind] = lf_blk16_w13_in_bind[7:0];
      assign blk16_w13_in_bind_empty_n[g_blk16_w13_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p13_in_bind_dout [0:0];
  wire blk16_p13_in_bind_empty_n [0:0];
  wire blk16_p13_in_bind_read [0:0];
  genvar g_blk16_p13_in_bind;
  generate
    for (g_blk16_p13_in_bind = 0; g_blk16_p13_in_bind < 1; g_blk16_p13_in_bind = g_blk16_p13_in_bind + 1) begin : gen_blk16_p13_in_bind
      reg [31:0] lf_blk16_p13_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p13_in_bind <= 32'h1 + g_blk16_p13_in_bind;
        else if (start) lf_blk16_p13_in_bind <= {lf_blk16_p13_in_bind[30:0], lf_blk16_p13_in_bind[31]^lf_blk16_p13_in_bind[21]^lf_blk16_p13_in_bind[1]^lf_blk16_p13_in_bind[0]};
      assign blk16_p13_in_bind_dout[g_blk16_p13_in_bind] = lf_blk16_p13_in_bind[31:0];
      assign blk16_p13_in_bind_empty_n[g_blk16_p13_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a14_in_bind_dout [0:0];
  wire blk16_a14_in_bind_empty_n [0:0];
  wire blk16_a14_in_bind_read [0:0];
  genvar g_blk16_a14_in_bind;
  generate
    for (g_blk16_a14_in_bind = 0; g_blk16_a14_in_bind < 1; g_blk16_a14_in_bind = g_blk16_a14_in_bind + 1) begin : gen_blk16_a14_in_bind
      reg [31:0] lf_blk16_a14_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a14_in_bind <= 32'h1 + g_blk16_a14_in_bind;
        else if (start) lf_blk16_a14_in_bind <= {lf_blk16_a14_in_bind[30:0], lf_blk16_a14_in_bind[31]^lf_blk16_a14_in_bind[21]^lf_blk16_a14_in_bind[1]^lf_blk16_a14_in_bind[0]};
      assign blk16_a14_in_bind_dout[g_blk16_a14_in_bind] = lf_blk16_a14_in_bind[7:0];
      assign blk16_a14_in_bind_empty_n[g_blk16_a14_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w14_in_bind_dout [0:0];
  wire blk16_w14_in_bind_empty_n [0:0];
  wire blk16_w14_in_bind_read [0:0];
  genvar g_blk16_w14_in_bind;
  generate
    for (g_blk16_w14_in_bind = 0; g_blk16_w14_in_bind < 1; g_blk16_w14_in_bind = g_blk16_w14_in_bind + 1) begin : gen_blk16_w14_in_bind
      reg [31:0] lf_blk16_w14_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w14_in_bind <= 32'h1 + g_blk16_w14_in_bind;
        else if (start) lf_blk16_w14_in_bind <= {lf_blk16_w14_in_bind[30:0], lf_blk16_w14_in_bind[31]^lf_blk16_w14_in_bind[21]^lf_blk16_w14_in_bind[1]^lf_blk16_w14_in_bind[0]};
      assign blk16_w14_in_bind_dout[g_blk16_w14_in_bind] = lf_blk16_w14_in_bind[7:0];
      assign blk16_w14_in_bind_empty_n[g_blk16_w14_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p14_in_bind_dout [0:0];
  wire blk16_p14_in_bind_empty_n [0:0];
  wire blk16_p14_in_bind_read [0:0];
  genvar g_blk16_p14_in_bind;
  generate
    for (g_blk16_p14_in_bind = 0; g_blk16_p14_in_bind < 1; g_blk16_p14_in_bind = g_blk16_p14_in_bind + 1) begin : gen_blk16_p14_in_bind
      reg [31:0] lf_blk16_p14_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p14_in_bind <= 32'h1 + g_blk16_p14_in_bind;
        else if (start) lf_blk16_p14_in_bind <= {lf_blk16_p14_in_bind[30:0], lf_blk16_p14_in_bind[31]^lf_blk16_p14_in_bind[21]^lf_blk16_p14_in_bind[1]^lf_blk16_p14_in_bind[0]};
      assign blk16_p14_in_bind_dout[g_blk16_p14_in_bind] = lf_blk16_p14_in_bind[31:0];
      assign blk16_p14_in_bind_empty_n[g_blk16_p14_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_a15_in_bind_dout [0:0];
  wire blk16_a15_in_bind_empty_n [0:0];
  wire blk16_a15_in_bind_read [0:0];
  genvar g_blk16_a15_in_bind;
  generate
    for (g_blk16_a15_in_bind = 0; g_blk16_a15_in_bind < 1; g_blk16_a15_in_bind = g_blk16_a15_in_bind + 1) begin : gen_blk16_a15_in_bind
      reg [31:0] lf_blk16_a15_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_a15_in_bind <= 32'h1 + g_blk16_a15_in_bind;
        else if (start) lf_blk16_a15_in_bind <= {lf_blk16_a15_in_bind[30:0], lf_blk16_a15_in_bind[31]^lf_blk16_a15_in_bind[21]^lf_blk16_a15_in_bind[1]^lf_blk16_a15_in_bind[0]};
      assign blk16_a15_in_bind_dout[g_blk16_a15_in_bind] = lf_blk16_a15_in_bind[7:0];
      assign blk16_a15_in_bind_empty_n[g_blk16_a15_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk16_w15_in_bind_dout [0:0];
  wire blk16_w15_in_bind_empty_n [0:0];
  wire blk16_w15_in_bind_read [0:0];
  genvar g_blk16_w15_in_bind;
  generate
    for (g_blk16_w15_in_bind = 0; g_blk16_w15_in_bind < 1; g_blk16_w15_in_bind = g_blk16_w15_in_bind + 1) begin : gen_blk16_w15_in_bind
      reg [31:0] lf_blk16_w15_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_w15_in_bind <= 32'h1 + g_blk16_w15_in_bind;
        else if (start) lf_blk16_w15_in_bind <= {lf_blk16_w15_in_bind[30:0], lf_blk16_w15_in_bind[31]^lf_blk16_w15_in_bind[21]^lf_blk16_w15_in_bind[1]^lf_blk16_w15_in_bind[0]};
      assign blk16_w15_in_bind_dout[g_blk16_w15_in_bind] = lf_blk16_w15_in_bind[7:0];
      assign blk16_w15_in_bind_empty_n[g_blk16_w15_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk16_p15_in_bind_dout [0:0];
  wire blk16_p15_in_bind_empty_n [0:0];
  wire blk16_p15_in_bind_read [0:0];
  genvar g_blk16_p15_in_bind;
  generate
    for (g_blk16_p15_in_bind = 0; g_blk16_p15_in_bind < 1; g_blk16_p15_in_bind = g_blk16_p15_in_bind + 1) begin : gen_blk16_p15_in_bind
      reg [31:0] lf_blk16_p15_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk16_p15_in_bind <= 32'h1 + g_blk16_p15_in_bind;
        else if (start) lf_blk16_p15_in_bind <= {lf_blk16_p15_in_bind[30:0], lf_blk16_p15_in_bind[31]^lf_blk16_p15_in_bind[21]^lf_blk16_p15_in_bind[1]^lf_blk16_p15_in_bind[0]};
      assign blk16_p15_in_bind_dout[g_blk16_p15_in_bind] = lf_blk16_p15_in_bind[31:0];
      assign blk16_p15_in_bind_empty_n[g_blk16_p15_in_bind] = start;
    end
  endgenerate
  wire [31:0] lanes16_y0_out_bind_din [0:0];
  wire lanes16_y0_out_bind_write [0:0];
  wire lanes16_y0_out_bind_full_n [0:0];
  genvar g_lanes16_y0_out_bind;
  generate
    for (g_lanes16_y0_out_bind = 0; g_lanes16_y0_out_bind < 1; g_lanes16_y0_out_bind = g_lanes16_y0_out_bind + 1) begin : gen_lanes16_y0_out_bind
      assign lanes16_y0_out_bind_full_n[g_lanes16_y0_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y1_out_bind_din [0:0];
  wire lanes16_y1_out_bind_write [0:0];
  wire lanes16_y1_out_bind_full_n [0:0];
  genvar g_lanes16_y1_out_bind;
  generate
    for (g_lanes16_y1_out_bind = 0; g_lanes16_y1_out_bind < 1; g_lanes16_y1_out_bind = g_lanes16_y1_out_bind + 1) begin : gen_lanes16_y1_out_bind
      assign lanes16_y1_out_bind_full_n[g_lanes16_y1_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y2_out_bind_din [0:0];
  wire lanes16_y2_out_bind_write [0:0];
  wire lanes16_y2_out_bind_full_n [0:0];
  genvar g_lanes16_y2_out_bind;
  generate
    for (g_lanes16_y2_out_bind = 0; g_lanes16_y2_out_bind < 1; g_lanes16_y2_out_bind = g_lanes16_y2_out_bind + 1) begin : gen_lanes16_y2_out_bind
      assign lanes16_y2_out_bind_full_n[g_lanes16_y2_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y3_out_bind_din [0:0];
  wire lanes16_y3_out_bind_write [0:0];
  wire lanes16_y3_out_bind_full_n [0:0];
  genvar g_lanes16_y3_out_bind;
  generate
    for (g_lanes16_y3_out_bind = 0; g_lanes16_y3_out_bind < 1; g_lanes16_y3_out_bind = g_lanes16_y3_out_bind + 1) begin : gen_lanes16_y3_out_bind
      assign lanes16_y3_out_bind_full_n[g_lanes16_y3_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y4_out_bind_din [0:0];
  wire lanes16_y4_out_bind_write [0:0];
  wire lanes16_y4_out_bind_full_n [0:0];
  genvar g_lanes16_y4_out_bind;
  generate
    for (g_lanes16_y4_out_bind = 0; g_lanes16_y4_out_bind < 1; g_lanes16_y4_out_bind = g_lanes16_y4_out_bind + 1) begin : gen_lanes16_y4_out_bind
      assign lanes16_y4_out_bind_full_n[g_lanes16_y4_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y5_out_bind_din [0:0];
  wire lanes16_y5_out_bind_write [0:0];
  wire lanes16_y5_out_bind_full_n [0:0];
  genvar g_lanes16_y5_out_bind;
  generate
    for (g_lanes16_y5_out_bind = 0; g_lanes16_y5_out_bind < 1; g_lanes16_y5_out_bind = g_lanes16_y5_out_bind + 1) begin : gen_lanes16_y5_out_bind
      assign lanes16_y5_out_bind_full_n[g_lanes16_y5_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y6_out_bind_din [0:0];
  wire lanes16_y6_out_bind_write [0:0];
  wire lanes16_y6_out_bind_full_n [0:0];
  genvar g_lanes16_y6_out_bind;
  generate
    for (g_lanes16_y6_out_bind = 0; g_lanes16_y6_out_bind < 1; g_lanes16_y6_out_bind = g_lanes16_y6_out_bind + 1) begin : gen_lanes16_y6_out_bind
      assign lanes16_y6_out_bind_full_n[g_lanes16_y6_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y7_out_bind_din [0:0];
  wire lanes16_y7_out_bind_write [0:0];
  wire lanes16_y7_out_bind_full_n [0:0];
  genvar g_lanes16_y7_out_bind;
  generate
    for (g_lanes16_y7_out_bind = 0; g_lanes16_y7_out_bind < 1; g_lanes16_y7_out_bind = g_lanes16_y7_out_bind + 1) begin : gen_lanes16_y7_out_bind
      assign lanes16_y7_out_bind_full_n[g_lanes16_y7_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y8_out_bind_din [0:0];
  wire lanes16_y8_out_bind_write [0:0];
  wire lanes16_y8_out_bind_full_n [0:0];
  genvar g_lanes16_y8_out_bind;
  generate
    for (g_lanes16_y8_out_bind = 0; g_lanes16_y8_out_bind < 1; g_lanes16_y8_out_bind = g_lanes16_y8_out_bind + 1) begin : gen_lanes16_y8_out_bind
      assign lanes16_y8_out_bind_full_n[g_lanes16_y8_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y9_out_bind_din [0:0];
  wire lanes16_y9_out_bind_write [0:0];
  wire lanes16_y9_out_bind_full_n [0:0];
  genvar g_lanes16_y9_out_bind;
  generate
    for (g_lanes16_y9_out_bind = 0; g_lanes16_y9_out_bind < 1; g_lanes16_y9_out_bind = g_lanes16_y9_out_bind + 1) begin : gen_lanes16_y9_out_bind
      assign lanes16_y9_out_bind_full_n[g_lanes16_y9_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y10_out_bind_din [0:0];
  wire lanes16_y10_out_bind_write [0:0];
  wire lanes16_y10_out_bind_full_n [0:0];
  genvar g_lanes16_y10_out_bind;
  generate
    for (g_lanes16_y10_out_bind = 0; g_lanes16_y10_out_bind < 1; g_lanes16_y10_out_bind = g_lanes16_y10_out_bind + 1) begin : gen_lanes16_y10_out_bind
      assign lanes16_y10_out_bind_full_n[g_lanes16_y10_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y11_out_bind_din [0:0];
  wire lanes16_y11_out_bind_write [0:0];
  wire lanes16_y11_out_bind_full_n [0:0];
  genvar g_lanes16_y11_out_bind;
  generate
    for (g_lanes16_y11_out_bind = 0; g_lanes16_y11_out_bind < 1; g_lanes16_y11_out_bind = g_lanes16_y11_out_bind + 1) begin : gen_lanes16_y11_out_bind
      assign lanes16_y11_out_bind_full_n[g_lanes16_y11_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y12_out_bind_din [0:0];
  wire lanes16_y12_out_bind_write [0:0];
  wire lanes16_y12_out_bind_full_n [0:0];
  genvar g_lanes16_y12_out_bind;
  generate
    for (g_lanes16_y12_out_bind = 0; g_lanes16_y12_out_bind < 1; g_lanes16_y12_out_bind = g_lanes16_y12_out_bind + 1) begin : gen_lanes16_y12_out_bind
      assign lanes16_y12_out_bind_full_n[g_lanes16_y12_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y13_out_bind_din [0:0];
  wire lanes16_y13_out_bind_write [0:0];
  wire lanes16_y13_out_bind_full_n [0:0];
  genvar g_lanes16_y13_out_bind;
  generate
    for (g_lanes16_y13_out_bind = 0; g_lanes16_y13_out_bind < 1; g_lanes16_y13_out_bind = g_lanes16_y13_out_bind + 1) begin : gen_lanes16_y13_out_bind
      assign lanes16_y13_out_bind_full_n[g_lanes16_y13_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y14_out_bind_din [0:0];
  wire lanes16_y14_out_bind_write [0:0];
  wire lanes16_y14_out_bind_full_n [0:0];
  genvar g_lanes16_y14_out_bind;
  generate
    for (g_lanes16_y14_out_bind = 0; g_lanes16_y14_out_bind < 1; g_lanes16_y14_out_bind = g_lanes16_y14_out_bind + 1) begin : gen_lanes16_y14_out_bind
      assign lanes16_y14_out_bind_full_n[g_lanes16_y14_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes16_y15_out_bind_din [0:0];
  wire lanes16_y15_out_bind_write [0:0];
  wire lanes16_y15_out_bind_full_n [0:0];
  genvar g_lanes16_y15_out_bind;
  generate
    for (g_lanes16_y15_out_bind = 0; g_lanes16_y15_out_bind < 1; g_lanes16_y15_out_bind = g_lanes16_y15_out_bind + 1) begin : gen_lanes16_y15_out_bind
      assign lanes16_y15_out_bind_full_n[g_lanes16_y15_out_bind] = 1'b1;
    end
  endgenerate
  wire [63:0] lanes16_b_mem_dout [0:0];
  wire lanes16_b_mem_empty_n [0:0];
  wire lanes16_b_mem_read [0:0];
  genvar g_lanes16_b_mem;
  generate
    for (g_lanes16_b_mem = 0; g_lanes16_b_mem < 1; g_lanes16_b_mem = g_lanes16_b_mem + 1) begin : gen_lanes16_b_mem
      reg [31:0] lf_lanes16_b_mem;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_lanes16_b_mem <= 32'h1 + g_lanes16_b_mem;
        else if (start) lf_lanes16_b_mem <= {lf_lanes16_b_mem[30:0], lf_lanes16_b_mem[31]^lf_lanes16_b_mem[21]^lf_lanes16_b_mem[1]^lf_lanes16_b_mem[0]};
      assign lanes16_b_mem_dout[g_lanes16_b_mem] = {2{lf_lanes16_b_mem}};
      assign lanes16_b_mem_empty_n[g_lanes16_b_mem] = start;
    end
  endgenerate

  spmw_top dut (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .blk16_a0_in_bind_dout(blk16_a0_in_bind_dout),
      .blk16_a0_in_bind_empty_n(blk16_a0_in_bind_empty_n),
      .blk16_a0_in_bind_read(blk16_a0_in_bind_read),
      .blk16_w0_in_bind_dout(blk16_w0_in_bind_dout),
      .blk16_w0_in_bind_empty_n(blk16_w0_in_bind_empty_n),
      .blk16_w0_in_bind_read(blk16_w0_in_bind_read),
      .blk16_p0_in_bind_dout(blk16_p0_in_bind_dout),
      .blk16_p0_in_bind_empty_n(blk16_p0_in_bind_empty_n),
      .blk16_p0_in_bind_read(blk16_p0_in_bind_read),
      .blk16_a1_in_bind_dout(blk16_a1_in_bind_dout),
      .blk16_a1_in_bind_empty_n(blk16_a1_in_bind_empty_n),
      .blk16_a1_in_bind_read(blk16_a1_in_bind_read),
      .blk16_w1_in_bind_dout(blk16_w1_in_bind_dout),
      .blk16_w1_in_bind_empty_n(blk16_w1_in_bind_empty_n),
      .blk16_w1_in_bind_read(blk16_w1_in_bind_read),
      .blk16_p1_in_bind_dout(blk16_p1_in_bind_dout),
      .blk16_p1_in_bind_empty_n(blk16_p1_in_bind_empty_n),
      .blk16_p1_in_bind_read(blk16_p1_in_bind_read),
      .blk16_a2_in_bind_dout(blk16_a2_in_bind_dout),
      .blk16_a2_in_bind_empty_n(blk16_a2_in_bind_empty_n),
      .blk16_a2_in_bind_read(blk16_a2_in_bind_read),
      .blk16_w2_in_bind_dout(blk16_w2_in_bind_dout),
      .blk16_w2_in_bind_empty_n(blk16_w2_in_bind_empty_n),
      .blk16_w2_in_bind_read(blk16_w2_in_bind_read),
      .blk16_p2_in_bind_dout(blk16_p2_in_bind_dout),
      .blk16_p2_in_bind_empty_n(blk16_p2_in_bind_empty_n),
      .blk16_p2_in_bind_read(blk16_p2_in_bind_read),
      .blk16_a3_in_bind_dout(blk16_a3_in_bind_dout),
      .blk16_a3_in_bind_empty_n(blk16_a3_in_bind_empty_n),
      .blk16_a3_in_bind_read(blk16_a3_in_bind_read),
      .blk16_w3_in_bind_dout(blk16_w3_in_bind_dout),
      .blk16_w3_in_bind_empty_n(blk16_w3_in_bind_empty_n),
      .blk16_w3_in_bind_read(blk16_w3_in_bind_read),
      .blk16_p3_in_bind_dout(blk16_p3_in_bind_dout),
      .blk16_p3_in_bind_empty_n(blk16_p3_in_bind_empty_n),
      .blk16_p3_in_bind_read(blk16_p3_in_bind_read),
      .blk16_a4_in_bind_dout(blk16_a4_in_bind_dout),
      .blk16_a4_in_bind_empty_n(blk16_a4_in_bind_empty_n),
      .blk16_a4_in_bind_read(blk16_a4_in_bind_read),
      .blk16_w4_in_bind_dout(blk16_w4_in_bind_dout),
      .blk16_w4_in_bind_empty_n(blk16_w4_in_bind_empty_n),
      .blk16_w4_in_bind_read(blk16_w4_in_bind_read),
      .blk16_p4_in_bind_dout(blk16_p4_in_bind_dout),
      .blk16_p4_in_bind_empty_n(blk16_p4_in_bind_empty_n),
      .blk16_p4_in_bind_read(blk16_p4_in_bind_read),
      .blk16_a5_in_bind_dout(blk16_a5_in_bind_dout),
      .blk16_a5_in_bind_empty_n(blk16_a5_in_bind_empty_n),
      .blk16_a5_in_bind_read(blk16_a5_in_bind_read),
      .blk16_w5_in_bind_dout(blk16_w5_in_bind_dout),
      .blk16_w5_in_bind_empty_n(blk16_w5_in_bind_empty_n),
      .blk16_w5_in_bind_read(blk16_w5_in_bind_read),
      .blk16_p5_in_bind_dout(blk16_p5_in_bind_dout),
      .blk16_p5_in_bind_empty_n(blk16_p5_in_bind_empty_n),
      .blk16_p5_in_bind_read(blk16_p5_in_bind_read),
      .blk16_a6_in_bind_dout(blk16_a6_in_bind_dout),
      .blk16_a6_in_bind_empty_n(blk16_a6_in_bind_empty_n),
      .blk16_a6_in_bind_read(blk16_a6_in_bind_read),
      .blk16_w6_in_bind_dout(blk16_w6_in_bind_dout),
      .blk16_w6_in_bind_empty_n(blk16_w6_in_bind_empty_n),
      .blk16_w6_in_bind_read(blk16_w6_in_bind_read),
      .blk16_p6_in_bind_dout(blk16_p6_in_bind_dout),
      .blk16_p6_in_bind_empty_n(blk16_p6_in_bind_empty_n),
      .blk16_p6_in_bind_read(blk16_p6_in_bind_read),
      .blk16_a7_in_bind_dout(blk16_a7_in_bind_dout),
      .blk16_a7_in_bind_empty_n(blk16_a7_in_bind_empty_n),
      .blk16_a7_in_bind_read(blk16_a7_in_bind_read),
      .blk16_w7_in_bind_dout(blk16_w7_in_bind_dout),
      .blk16_w7_in_bind_empty_n(blk16_w7_in_bind_empty_n),
      .blk16_w7_in_bind_read(blk16_w7_in_bind_read),
      .blk16_p7_in_bind_dout(blk16_p7_in_bind_dout),
      .blk16_p7_in_bind_empty_n(blk16_p7_in_bind_empty_n),
      .blk16_p7_in_bind_read(blk16_p7_in_bind_read),
      .blk16_a8_in_bind_dout(blk16_a8_in_bind_dout),
      .blk16_a8_in_bind_empty_n(blk16_a8_in_bind_empty_n),
      .blk16_a8_in_bind_read(blk16_a8_in_bind_read),
      .blk16_w8_in_bind_dout(blk16_w8_in_bind_dout),
      .blk16_w8_in_bind_empty_n(blk16_w8_in_bind_empty_n),
      .blk16_w8_in_bind_read(blk16_w8_in_bind_read),
      .blk16_p8_in_bind_dout(blk16_p8_in_bind_dout),
      .blk16_p8_in_bind_empty_n(blk16_p8_in_bind_empty_n),
      .blk16_p8_in_bind_read(blk16_p8_in_bind_read),
      .blk16_a9_in_bind_dout(blk16_a9_in_bind_dout),
      .blk16_a9_in_bind_empty_n(blk16_a9_in_bind_empty_n),
      .blk16_a9_in_bind_read(blk16_a9_in_bind_read),
      .blk16_w9_in_bind_dout(blk16_w9_in_bind_dout),
      .blk16_w9_in_bind_empty_n(blk16_w9_in_bind_empty_n),
      .blk16_w9_in_bind_read(blk16_w9_in_bind_read),
      .blk16_p9_in_bind_dout(blk16_p9_in_bind_dout),
      .blk16_p9_in_bind_empty_n(blk16_p9_in_bind_empty_n),
      .blk16_p9_in_bind_read(blk16_p9_in_bind_read),
      .blk16_a10_in_bind_dout(blk16_a10_in_bind_dout),
      .blk16_a10_in_bind_empty_n(blk16_a10_in_bind_empty_n),
      .blk16_a10_in_bind_read(blk16_a10_in_bind_read),
      .blk16_w10_in_bind_dout(blk16_w10_in_bind_dout),
      .blk16_w10_in_bind_empty_n(blk16_w10_in_bind_empty_n),
      .blk16_w10_in_bind_read(blk16_w10_in_bind_read),
      .blk16_p10_in_bind_dout(blk16_p10_in_bind_dout),
      .blk16_p10_in_bind_empty_n(blk16_p10_in_bind_empty_n),
      .blk16_p10_in_bind_read(blk16_p10_in_bind_read),
      .blk16_a11_in_bind_dout(blk16_a11_in_bind_dout),
      .blk16_a11_in_bind_empty_n(blk16_a11_in_bind_empty_n),
      .blk16_a11_in_bind_read(blk16_a11_in_bind_read),
      .blk16_w11_in_bind_dout(blk16_w11_in_bind_dout),
      .blk16_w11_in_bind_empty_n(blk16_w11_in_bind_empty_n),
      .blk16_w11_in_bind_read(blk16_w11_in_bind_read),
      .blk16_p11_in_bind_dout(blk16_p11_in_bind_dout),
      .blk16_p11_in_bind_empty_n(blk16_p11_in_bind_empty_n),
      .blk16_p11_in_bind_read(blk16_p11_in_bind_read),
      .blk16_a12_in_bind_dout(blk16_a12_in_bind_dout),
      .blk16_a12_in_bind_empty_n(blk16_a12_in_bind_empty_n),
      .blk16_a12_in_bind_read(blk16_a12_in_bind_read),
      .blk16_w12_in_bind_dout(blk16_w12_in_bind_dout),
      .blk16_w12_in_bind_empty_n(blk16_w12_in_bind_empty_n),
      .blk16_w12_in_bind_read(blk16_w12_in_bind_read),
      .blk16_p12_in_bind_dout(blk16_p12_in_bind_dout),
      .blk16_p12_in_bind_empty_n(blk16_p12_in_bind_empty_n),
      .blk16_p12_in_bind_read(blk16_p12_in_bind_read),
      .blk16_a13_in_bind_dout(blk16_a13_in_bind_dout),
      .blk16_a13_in_bind_empty_n(blk16_a13_in_bind_empty_n),
      .blk16_a13_in_bind_read(blk16_a13_in_bind_read),
      .blk16_w13_in_bind_dout(blk16_w13_in_bind_dout),
      .blk16_w13_in_bind_empty_n(blk16_w13_in_bind_empty_n),
      .blk16_w13_in_bind_read(blk16_w13_in_bind_read),
      .blk16_p13_in_bind_dout(blk16_p13_in_bind_dout),
      .blk16_p13_in_bind_empty_n(blk16_p13_in_bind_empty_n),
      .blk16_p13_in_bind_read(blk16_p13_in_bind_read),
      .blk16_a14_in_bind_dout(blk16_a14_in_bind_dout),
      .blk16_a14_in_bind_empty_n(blk16_a14_in_bind_empty_n),
      .blk16_a14_in_bind_read(blk16_a14_in_bind_read),
      .blk16_w14_in_bind_dout(blk16_w14_in_bind_dout),
      .blk16_w14_in_bind_empty_n(blk16_w14_in_bind_empty_n),
      .blk16_w14_in_bind_read(blk16_w14_in_bind_read),
      .blk16_p14_in_bind_dout(blk16_p14_in_bind_dout),
      .blk16_p14_in_bind_empty_n(blk16_p14_in_bind_empty_n),
      .blk16_p14_in_bind_read(blk16_p14_in_bind_read),
      .blk16_a15_in_bind_dout(blk16_a15_in_bind_dout),
      .blk16_a15_in_bind_empty_n(blk16_a15_in_bind_empty_n),
      .blk16_a15_in_bind_read(blk16_a15_in_bind_read),
      .blk16_w15_in_bind_dout(blk16_w15_in_bind_dout),
      .blk16_w15_in_bind_empty_n(blk16_w15_in_bind_empty_n),
      .blk16_w15_in_bind_read(blk16_w15_in_bind_read),
      .blk16_p15_in_bind_dout(blk16_p15_in_bind_dout),
      .blk16_p15_in_bind_empty_n(blk16_p15_in_bind_empty_n),
      .blk16_p15_in_bind_read(blk16_p15_in_bind_read),
      .lanes16_y0_out_bind_din(lanes16_y0_out_bind_din),
      .lanes16_y0_out_bind_write(lanes16_y0_out_bind_write),
      .lanes16_y0_out_bind_full_n(lanes16_y0_out_bind_full_n),
      .lanes16_y1_out_bind_din(lanes16_y1_out_bind_din),
      .lanes16_y1_out_bind_write(lanes16_y1_out_bind_write),
      .lanes16_y1_out_bind_full_n(lanes16_y1_out_bind_full_n),
      .lanes16_y2_out_bind_din(lanes16_y2_out_bind_din),
      .lanes16_y2_out_bind_write(lanes16_y2_out_bind_write),
      .lanes16_y2_out_bind_full_n(lanes16_y2_out_bind_full_n),
      .lanes16_y3_out_bind_din(lanes16_y3_out_bind_din),
      .lanes16_y3_out_bind_write(lanes16_y3_out_bind_write),
      .lanes16_y3_out_bind_full_n(lanes16_y3_out_bind_full_n),
      .lanes16_y4_out_bind_din(lanes16_y4_out_bind_din),
      .lanes16_y4_out_bind_write(lanes16_y4_out_bind_write),
      .lanes16_y4_out_bind_full_n(lanes16_y4_out_bind_full_n),
      .lanes16_y5_out_bind_din(lanes16_y5_out_bind_din),
      .lanes16_y5_out_bind_write(lanes16_y5_out_bind_write),
      .lanes16_y5_out_bind_full_n(lanes16_y5_out_bind_full_n),
      .lanes16_y6_out_bind_din(lanes16_y6_out_bind_din),
      .lanes16_y6_out_bind_write(lanes16_y6_out_bind_write),
      .lanes16_y6_out_bind_full_n(lanes16_y6_out_bind_full_n),
      .lanes16_y7_out_bind_din(lanes16_y7_out_bind_din),
      .lanes16_y7_out_bind_write(lanes16_y7_out_bind_write),
      .lanes16_y7_out_bind_full_n(lanes16_y7_out_bind_full_n),
      .lanes16_y8_out_bind_din(lanes16_y8_out_bind_din),
      .lanes16_y8_out_bind_write(lanes16_y8_out_bind_write),
      .lanes16_y8_out_bind_full_n(lanes16_y8_out_bind_full_n),
      .lanes16_y9_out_bind_din(lanes16_y9_out_bind_din),
      .lanes16_y9_out_bind_write(lanes16_y9_out_bind_write),
      .lanes16_y9_out_bind_full_n(lanes16_y9_out_bind_full_n),
      .lanes16_y10_out_bind_din(lanes16_y10_out_bind_din),
      .lanes16_y10_out_bind_write(lanes16_y10_out_bind_write),
      .lanes16_y10_out_bind_full_n(lanes16_y10_out_bind_full_n),
      .lanes16_y11_out_bind_din(lanes16_y11_out_bind_din),
      .lanes16_y11_out_bind_write(lanes16_y11_out_bind_write),
      .lanes16_y11_out_bind_full_n(lanes16_y11_out_bind_full_n),
      .lanes16_y12_out_bind_din(lanes16_y12_out_bind_din),
      .lanes16_y12_out_bind_write(lanes16_y12_out_bind_write),
      .lanes16_y12_out_bind_full_n(lanes16_y12_out_bind_full_n),
      .lanes16_y13_out_bind_din(lanes16_y13_out_bind_din),
      .lanes16_y13_out_bind_write(lanes16_y13_out_bind_write),
      .lanes16_y13_out_bind_full_n(lanes16_y13_out_bind_full_n),
      .lanes16_y14_out_bind_din(lanes16_y14_out_bind_din),
      .lanes16_y14_out_bind_write(lanes16_y14_out_bind_write),
      .lanes16_y14_out_bind_full_n(lanes16_y14_out_bind_full_n),
      .lanes16_y15_out_bind_din(lanes16_y15_out_bind_din),
      .lanes16_y15_out_bind_write(lanes16_y15_out_bind_write),
      .lanes16_y15_out_bind_full_n(lanes16_y15_out_bind_full_n),
      .lanes16_b_mem_dout(lanes16_b_mem_dout),
      .lanes16_b_mem_empty_n(lanes16_b_mem_empty_n),
      .lanes16_b_mem_read(lanes16_b_mem_read));

  // Fold every output into one register: a dangling result is a result
  // synthesis is entitled to delete.
  always @(posedge ap_clk)
    if (!ap_rst_n) sig <= 32'b0;
    else sig <= sig
      ^ {31'b0, blk16_a0_in_bind_read[0]}
      ^ {31'b0, blk16_w0_in_bind_read[0]}
      ^ {31'b0, blk16_p0_in_bind_read[0]}
      ^ {31'b0, blk16_a1_in_bind_read[0]}
      ^ {31'b0, blk16_w1_in_bind_read[0]}
      ^ {31'b0, blk16_p1_in_bind_read[0]}
      ^ {31'b0, blk16_a2_in_bind_read[0]}
      ^ {31'b0, blk16_w2_in_bind_read[0]}
      ^ {31'b0, blk16_p2_in_bind_read[0]}
      ^ {31'b0, blk16_a3_in_bind_read[0]}
      ^ {31'b0, blk16_w3_in_bind_read[0]}
      ^ {31'b0, blk16_p3_in_bind_read[0]}
      ^ {31'b0, blk16_a4_in_bind_read[0]}
      ^ {31'b0, blk16_w4_in_bind_read[0]}
      ^ {31'b0, blk16_p4_in_bind_read[0]}
      ^ {31'b0, blk16_a5_in_bind_read[0]}
      ^ {31'b0, blk16_w5_in_bind_read[0]}
      ^ {31'b0, blk16_p5_in_bind_read[0]}
      ^ {31'b0, blk16_a6_in_bind_read[0]}
      ^ {31'b0, blk16_w6_in_bind_read[0]}
      ^ {31'b0, blk16_p6_in_bind_read[0]}
      ^ {31'b0, blk16_a7_in_bind_read[0]}
      ^ {31'b0, blk16_w7_in_bind_read[0]}
      ^ {31'b0, blk16_p7_in_bind_read[0]}
      ^ {31'b0, blk16_a8_in_bind_read[0]}
      ^ {31'b0, blk16_w8_in_bind_read[0]}
      ^ {31'b0, blk16_p8_in_bind_read[0]}
      ^ {31'b0, blk16_a9_in_bind_read[0]}
      ^ {31'b0, blk16_w9_in_bind_read[0]}
      ^ {31'b0, blk16_p9_in_bind_read[0]}
      ^ {31'b0, blk16_a10_in_bind_read[0]}
      ^ {31'b0, blk16_w10_in_bind_read[0]}
      ^ {31'b0, blk16_p10_in_bind_read[0]}
      ^ {31'b0, blk16_a11_in_bind_read[0]}
      ^ {31'b0, blk16_w11_in_bind_read[0]}
      ^ {31'b0, blk16_p11_in_bind_read[0]}
      ^ {31'b0, blk16_a12_in_bind_read[0]}
      ^ {31'b0, blk16_w12_in_bind_read[0]}
      ^ {31'b0, blk16_p12_in_bind_read[0]}
      ^ {31'b0, blk16_a13_in_bind_read[0]}
      ^ {31'b0, blk16_w13_in_bind_read[0]}
      ^ {31'b0, blk16_p13_in_bind_read[0]}
      ^ {31'b0, blk16_a14_in_bind_read[0]}
      ^ {31'b0, blk16_w14_in_bind_read[0]}
      ^ {31'b0, blk16_p14_in_bind_read[0]}
      ^ {31'b0, blk16_a15_in_bind_read[0]}
      ^ {31'b0, blk16_w15_in_bind_read[0]}
      ^ {31'b0, blk16_p15_in_bind_read[0]}
      ^ lanes16_y0_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y0_out_bind_write[0]}
      ^ lanes16_y1_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y1_out_bind_write[0]}
      ^ lanes16_y2_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y2_out_bind_write[0]}
      ^ lanes16_y3_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y3_out_bind_write[0]}
      ^ lanes16_y4_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y4_out_bind_write[0]}
      ^ lanes16_y5_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y5_out_bind_write[0]}
      ^ lanes16_y6_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y6_out_bind_write[0]}
      ^ lanes16_y7_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y7_out_bind_write[0]}
      ^ lanes16_y8_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y8_out_bind_write[0]}
      ^ lanes16_y9_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y9_out_bind_write[0]}
      ^ lanes16_y10_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y10_out_bind_write[0]}
      ^ lanes16_y11_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y11_out_bind_write[0]}
      ^ lanes16_y12_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y12_out_bind_write[0]}
      ^ lanes16_y13_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y13_out_bind_write[0]}
      ^ lanes16_y14_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y14_out_bind_write[0]}
      ^ lanes16_y15_out_bind_din[0][31:0]
      ^ {31'b0, lanes16_y15_out_bind_write[0]}
      ^ {31'b0, lanes16_b_mem_read[0]}
      ;
endmodule
