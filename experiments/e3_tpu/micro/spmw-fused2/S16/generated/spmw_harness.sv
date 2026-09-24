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

  wire [7:0] blk2_a0_in_bind_dout [0:7];
  wire blk2_a0_in_bind_empty_n [0:7];
  wire blk2_a0_in_bind_read [0:7];
  genvar g_blk2_a0_in_bind;
  generate
    for (g_blk2_a0_in_bind = 0; g_blk2_a0_in_bind < 8; g_blk2_a0_in_bind = g_blk2_a0_in_bind + 1) begin : gen_blk2_a0_in_bind
      reg [31:0] lf_blk2_a0_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk2_a0_in_bind <= 32'h1 + g_blk2_a0_in_bind;
        else if (start) lf_blk2_a0_in_bind <= {lf_blk2_a0_in_bind[30:0], lf_blk2_a0_in_bind[31]^lf_blk2_a0_in_bind[21]^lf_blk2_a0_in_bind[1]^lf_blk2_a0_in_bind[0]};
      assign blk2_a0_in_bind_dout[g_blk2_a0_in_bind] = lf_blk2_a0_in_bind[7:0];
      assign blk2_a0_in_bind_empty_n[g_blk2_a0_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk2_w0_in_bind_dout [0:7];
  wire blk2_w0_in_bind_empty_n [0:7];
  wire blk2_w0_in_bind_read [0:7];
  genvar g_blk2_w0_in_bind;
  generate
    for (g_blk2_w0_in_bind = 0; g_blk2_w0_in_bind < 8; g_blk2_w0_in_bind = g_blk2_w0_in_bind + 1) begin : gen_blk2_w0_in_bind
      reg [31:0] lf_blk2_w0_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk2_w0_in_bind <= 32'h1 + g_blk2_w0_in_bind;
        else if (start) lf_blk2_w0_in_bind <= {lf_blk2_w0_in_bind[30:0], lf_blk2_w0_in_bind[31]^lf_blk2_w0_in_bind[21]^lf_blk2_w0_in_bind[1]^lf_blk2_w0_in_bind[0]};
      assign blk2_w0_in_bind_dout[g_blk2_w0_in_bind] = lf_blk2_w0_in_bind[7:0];
      assign blk2_w0_in_bind_empty_n[g_blk2_w0_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk2_p0_in_bind_dout [0:7];
  wire blk2_p0_in_bind_empty_n [0:7];
  wire blk2_p0_in_bind_read [0:7];
  genvar g_blk2_p0_in_bind;
  generate
    for (g_blk2_p0_in_bind = 0; g_blk2_p0_in_bind < 8; g_blk2_p0_in_bind = g_blk2_p0_in_bind + 1) begin : gen_blk2_p0_in_bind
      reg [31:0] lf_blk2_p0_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk2_p0_in_bind <= 32'h1 + g_blk2_p0_in_bind;
        else if (start) lf_blk2_p0_in_bind <= {lf_blk2_p0_in_bind[30:0], lf_blk2_p0_in_bind[31]^lf_blk2_p0_in_bind[21]^lf_blk2_p0_in_bind[1]^lf_blk2_p0_in_bind[0]};
      assign blk2_p0_in_bind_dout[g_blk2_p0_in_bind] = lf_blk2_p0_in_bind[31:0];
      assign blk2_p0_in_bind_empty_n[g_blk2_p0_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk2_a1_in_bind_dout [0:7];
  wire blk2_a1_in_bind_empty_n [0:7];
  wire blk2_a1_in_bind_read [0:7];
  genvar g_blk2_a1_in_bind;
  generate
    for (g_blk2_a1_in_bind = 0; g_blk2_a1_in_bind < 8; g_blk2_a1_in_bind = g_blk2_a1_in_bind + 1) begin : gen_blk2_a1_in_bind
      reg [31:0] lf_blk2_a1_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk2_a1_in_bind <= 32'h1 + g_blk2_a1_in_bind;
        else if (start) lf_blk2_a1_in_bind <= {lf_blk2_a1_in_bind[30:0], lf_blk2_a1_in_bind[31]^lf_blk2_a1_in_bind[21]^lf_blk2_a1_in_bind[1]^lf_blk2_a1_in_bind[0]};
      assign blk2_a1_in_bind_dout[g_blk2_a1_in_bind] = lf_blk2_a1_in_bind[7:0];
      assign blk2_a1_in_bind_empty_n[g_blk2_a1_in_bind] = start;
    end
  endgenerate
  wire [7:0] blk2_w1_in_bind_dout [0:7];
  wire blk2_w1_in_bind_empty_n [0:7];
  wire blk2_w1_in_bind_read [0:7];
  genvar g_blk2_w1_in_bind;
  generate
    for (g_blk2_w1_in_bind = 0; g_blk2_w1_in_bind < 8; g_blk2_w1_in_bind = g_blk2_w1_in_bind + 1) begin : gen_blk2_w1_in_bind
      reg [31:0] lf_blk2_w1_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk2_w1_in_bind <= 32'h1 + g_blk2_w1_in_bind;
        else if (start) lf_blk2_w1_in_bind <= {lf_blk2_w1_in_bind[30:0], lf_blk2_w1_in_bind[31]^lf_blk2_w1_in_bind[21]^lf_blk2_w1_in_bind[1]^lf_blk2_w1_in_bind[0]};
      assign blk2_w1_in_bind_dout[g_blk2_w1_in_bind] = lf_blk2_w1_in_bind[7:0];
      assign blk2_w1_in_bind_empty_n[g_blk2_w1_in_bind] = start;
    end
  endgenerate
  wire [31:0] blk2_p1_in_bind_dout [0:7];
  wire blk2_p1_in_bind_empty_n [0:7];
  wire blk2_p1_in_bind_read [0:7];
  genvar g_blk2_p1_in_bind;
  generate
    for (g_blk2_p1_in_bind = 0; g_blk2_p1_in_bind < 8; g_blk2_p1_in_bind = g_blk2_p1_in_bind + 1) begin : gen_blk2_p1_in_bind
      reg [31:0] lf_blk2_p1_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_blk2_p1_in_bind <= 32'h1 + g_blk2_p1_in_bind;
        else if (start) lf_blk2_p1_in_bind <= {lf_blk2_p1_in_bind[30:0], lf_blk2_p1_in_bind[31]^lf_blk2_p1_in_bind[21]^lf_blk2_p1_in_bind[1]^lf_blk2_p1_in_bind[0]};
      assign blk2_p1_in_bind_dout[g_blk2_p1_in_bind] = lf_blk2_p1_in_bind[31:0];
      assign blk2_p1_in_bind_empty_n[g_blk2_p1_in_bind] = start;
    end
  endgenerate
  wire [31:0] lanes2_y0_out_bind_din [0:7];
  wire lanes2_y0_out_bind_write [0:7];
  wire lanes2_y0_out_bind_full_n [0:7];
  genvar g_lanes2_y0_out_bind;
  generate
    for (g_lanes2_y0_out_bind = 0; g_lanes2_y0_out_bind < 8; g_lanes2_y0_out_bind = g_lanes2_y0_out_bind + 1) begin : gen_lanes2_y0_out_bind
      assign lanes2_y0_out_bind_full_n[g_lanes2_y0_out_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] lanes2_y1_out_bind_din [0:7];
  wire lanes2_y1_out_bind_write [0:7];
  wire lanes2_y1_out_bind_full_n [0:7];
  genvar g_lanes2_y1_out_bind;
  generate
    for (g_lanes2_y1_out_bind = 0; g_lanes2_y1_out_bind < 8; g_lanes2_y1_out_bind = g_lanes2_y1_out_bind + 1) begin : gen_lanes2_y1_out_bind
      assign lanes2_y1_out_bind_full_n[g_lanes2_y1_out_bind] = 1'b1;
    end
  endgenerate
  wire [63:0] lanes2_b_mem_dout [0:7];
  wire lanes2_b_mem_empty_n [0:7];
  wire lanes2_b_mem_read [0:7];
  genvar g_lanes2_b_mem;
  generate
    for (g_lanes2_b_mem = 0; g_lanes2_b_mem < 8; g_lanes2_b_mem = g_lanes2_b_mem + 1) begin : gen_lanes2_b_mem
      reg [31:0] lf_lanes2_b_mem;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_lanes2_b_mem <= 32'h1 + g_lanes2_b_mem;
        else if (start) lf_lanes2_b_mem <= {lf_lanes2_b_mem[30:0], lf_lanes2_b_mem[31]^lf_lanes2_b_mem[21]^lf_lanes2_b_mem[1]^lf_lanes2_b_mem[0]};
      assign lanes2_b_mem_dout[g_lanes2_b_mem] = {2{lf_lanes2_b_mem}};
      assign lanes2_b_mem_empty_n[g_lanes2_b_mem] = start;
    end
  endgenerate

  spmw_top dut (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .blk2_a0_in_bind_dout(blk2_a0_in_bind_dout),
      .blk2_a0_in_bind_empty_n(blk2_a0_in_bind_empty_n),
      .blk2_a0_in_bind_read(blk2_a0_in_bind_read),
      .blk2_w0_in_bind_dout(blk2_w0_in_bind_dout),
      .blk2_w0_in_bind_empty_n(blk2_w0_in_bind_empty_n),
      .blk2_w0_in_bind_read(blk2_w0_in_bind_read),
      .blk2_p0_in_bind_dout(blk2_p0_in_bind_dout),
      .blk2_p0_in_bind_empty_n(blk2_p0_in_bind_empty_n),
      .blk2_p0_in_bind_read(blk2_p0_in_bind_read),
      .blk2_a1_in_bind_dout(blk2_a1_in_bind_dout),
      .blk2_a1_in_bind_empty_n(blk2_a1_in_bind_empty_n),
      .blk2_a1_in_bind_read(blk2_a1_in_bind_read),
      .blk2_w1_in_bind_dout(blk2_w1_in_bind_dout),
      .blk2_w1_in_bind_empty_n(blk2_w1_in_bind_empty_n),
      .blk2_w1_in_bind_read(blk2_w1_in_bind_read),
      .blk2_p1_in_bind_dout(blk2_p1_in_bind_dout),
      .blk2_p1_in_bind_empty_n(blk2_p1_in_bind_empty_n),
      .blk2_p1_in_bind_read(blk2_p1_in_bind_read),
      .lanes2_y0_out_bind_din(lanes2_y0_out_bind_din),
      .lanes2_y0_out_bind_write(lanes2_y0_out_bind_write),
      .lanes2_y0_out_bind_full_n(lanes2_y0_out_bind_full_n),
      .lanes2_y1_out_bind_din(lanes2_y1_out_bind_din),
      .lanes2_y1_out_bind_write(lanes2_y1_out_bind_write),
      .lanes2_y1_out_bind_full_n(lanes2_y1_out_bind_full_n),
      .lanes2_b_mem_dout(lanes2_b_mem_dout),
      .lanes2_b_mem_empty_n(lanes2_b_mem_empty_n),
      .lanes2_b_mem_read(lanes2_b_mem_read));

  // Fold every output into one register: a dangling result is a result
  // synthesis is entitled to delete.
  always @(posedge ap_clk)
    if (!ap_rst_n) sig <= 32'b0;
    else sig <= sig
      ^ {31'b0, blk2_a0_in_bind_read[0]}
      ^ {31'b0, blk2_w0_in_bind_read[0]}
      ^ {31'b0, blk2_p0_in_bind_read[0]}
      ^ {31'b0, blk2_a1_in_bind_read[0]}
      ^ {31'b0, blk2_w1_in_bind_read[0]}
      ^ {31'b0, blk2_p1_in_bind_read[0]}
      ^ lanes2_y0_out_bind_din[0][31:0]
      ^ {31'b0, lanes2_y0_out_bind_write[0]}
      ^ lanes2_y0_out_bind_din[1][31:0]
      ^ {31'b0, lanes2_y0_out_bind_write[1]}
      ^ lanes2_y0_out_bind_din[2][31:0]
      ^ {31'b0, lanes2_y0_out_bind_write[2]}
      ^ lanes2_y0_out_bind_din[3][31:0]
      ^ {31'b0, lanes2_y0_out_bind_write[3]}
      ^ lanes2_y0_out_bind_din[4][31:0]
      ^ {31'b0, lanes2_y0_out_bind_write[4]}
      ^ lanes2_y0_out_bind_din[5][31:0]
      ^ {31'b0, lanes2_y0_out_bind_write[5]}
      ^ lanes2_y0_out_bind_din[6][31:0]
      ^ {31'b0, lanes2_y0_out_bind_write[6]}
      ^ lanes2_y0_out_bind_din[7][31:0]
      ^ {31'b0, lanes2_y0_out_bind_write[7]}
      ^ lanes2_y1_out_bind_din[0][31:0]
      ^ {31'b0, lanes2_y1_out_bind_write[0]}
      ^ lanes2_y1_out_bind_din[1][31:0]
      ^ {31'b0, lanes2_y1_out_bind_write[1]}
      ^ lanes2_y1_out_bind_din[2][31:0]
      ^ {31'b0, lanes2_y1_out_bind_write[2]}
      ^ lanes2_y1_out_bind_din[3][31:0]
      ^ {31'b0, lanes2_y1_out_bind_write[3]}
      ^ lanes2_y1_out_bind_din[4][31:0]
      ^ {31'b0, lanes2_y1_out_bind_write[4]}
      ^ lanes2_y1_out_bind_din[5][31:0]
      ^ {31'b0, lanes2_y1_out_bind_write[5]}
      ^ lanes2_y1_out_bind_din[6][31:0]
      ^ {31'b0, lanes2_y1_out_bind_write[6]}
      ^ lanes2_y1_out_bind_din[7][31:0]
      ^ {31'b0, lanes2_y1_out_bind_write[7]}
      ^ {31'b0, lanes2_b_mem_read[0]}
      ;
endmodule
