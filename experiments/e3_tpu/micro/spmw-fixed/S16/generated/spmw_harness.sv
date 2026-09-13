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

  wire [7:0] mac_a_in_bind_dout [0:15];
  wire mac_a_in_bind_empty_n [0:15];
  wire mac_a_in_bind_read [0:15];
  genvar g_mac_a_in_bind;
  generate
    for (g_mac_a_in_bind = 0; g_mac_a_in_bind < 16; g_mac_a_in_bind = g_mac_a_in_bind + 1) begin : gen_mac_a_in_bind
      reg [31:0] lf_mac_a_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_mac_a_in_bind <= 32'h1 + g_mac_a_in_bind;
        else if (start) lf_mac_a_in_bind <= {lf_mac_a_in_bind[30:0], lf_mac_a_in_bind[31]^lf_mac_a_in_bind[21]^lf_mac_a_in_bind[1]^lf_mac_a_in_bind[0]};
      assign mac_a_in_bind_dout[g_mac_a_in_bind] = lf_mac_a_in_bind[7:0];
      assign mac_a_in_bind_empty_n[g_mac_a_in_bind] = start;
    end
  endgenerate
  wire [31:0] mac_w_in_bind_dout [0:15];
  wire mac_w_in_bind_empty_n [0:15];
  wire mac_w_in_bind_read [0:15];
  genvar g_mac_w_in_bind;
  generate
    for (g_mac_w_in_bind = 0; g_mac_w_in_bind < 16; g_mac_w_in_bind = g_mac_w_in_bind + 1) begin : gen_mac_w_in_bind
      reg [31:0] lf_mac_w_in_bind;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_mac_w_in_bind <= 32'h1 + g_mac_w_in_bind;
        else if (start) lf_mac_w_in_bind <= {lf_mac_w_in_bind[30:0], lf_mac_w_in_bind[31]^lf_mac_w_in_bind[21]^lf_mac_w_in_bind[1]^lf_mac_w_in_bind[0]};
      assign mac_w_in_bind_dout[g_mac_w_in_bind] = lf_mac_w_in_bind[31:0];
      assign mac_w_in_bind_empty_n[g_mac_w_in_bind] = start;
    end
  endgenerate
  wire [31:0] vpu_y_out_bind_din [0:15];
  wire vpu_y_out_bind_write [0:15];
  wire vpu_y_out_bind_full_n [0:15];
  genvar g_vpu_y_out_bind;
  generate
    for (g_vpu_y_out_bind = 0; g_vpu_y_out_bind < 16; g_vpu_y_out_bind = g_vpu_y_out_bind + 1) begin : gen_vpu_y_out_bind
      assign vpu_y_out_bind_full_n[g_vpu_y_out_bind] = 1'b1;
    end
  endgenerate
  wire [63:0] vpu_b_mem_dout [0:15];
  wire vpu_b_mem_empty_n [0:15];
  wire vpu_b_mem_read [0:15];
  genvar g_vpu_b_mem;
  generate
    for (g_vpu_b_mem = 0; g_vpu_b_mem < 16; g_vpu_b_mem = g_vpu_b_mem + 1) begin : gen_vpu_b_mem
      reg [31:0] lf_vpu_b_mem;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_vpu_b_mem <= 32'h1 + g_vpu_b_mem;
        else if (start) lf_vpu_b_mem <= {lf_vpu_b_mem[30:0], lf_vpu_b_mem[31]^lf_vpu_b_mem[21]^lf_vpu_b_mem[1]^lf_vpu_b_mem[0]};
      assign vpu_b_mem_dout[g_vpu_b_mem] = {2{lf_vpu_b_mem}};
      assign vpu_b_mem_empty_n[g_vpu_b_mem] = start;
    end
  endgenerate

  spmw_top dut (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .mac_a_in_bind_dout(mac_a_in_bind_dout),
      .mac_a_in_bind_empty_n(mac_a_in_bind_empty_n),
      .mac_a_in_bind_read(mac_a_in_bind_read),
      .mac_w_in_bind_dout(mac_w_in_bind_dout),
      .mac_w_in_bind_empty_n(mac_w_in_bind_empty_n),
      .mac_w_in_bind_read(mac_w_in_bind_read),
      .vpu_y_out_bind_din(vpu_y_out_bind_din),
      .vpu_y_out_bind_write(vpu_y_out_bind_write),
      .vpu_y_out_bind_full_n(vpu_y_out_bind_full_n),
      .vpu_b_mem_dout(vpu_b_mem_dout),
      .vpu_b_mem_empty_n(vpu_b_mem_empty_n),
      .vpu_b_mem_read(vpu_b_mem_read));

  // Fold every output into one register: a dangling result is a result
  // synthesis is entitled to delete.
  always @(posedge ap_clk)
    if (!ap_rst_n) sig <= 32'b0;
    else sig <= sig
      ^ {31'b0, mac_a_in_bind_read[0]}
      ^ {31'b0, mac_w_in_bind_read[0]}
      ^ vpu_y_out_bind_din[0][31:0]
      ^ {31'b0, vpu_y_out_bind_write[0]}
      ^ vpu_y_out_bind_din[1][31:0]
      ^ {31'b0, vpu_y_out_bind_write[1]}
      ^ vpu_y_out_bind_din[2][31:0]
      ^ {31'b0, vpu_y_out_bind_write[2]}
      ^ vpu_y_out_bind_din[3][31:0]
      ^ {31'b0, vpu_y_out_bind_write[3]}
      ^ vpu_y_out_bind_din[4][31:0]
      ^ {31'b0, vpu_y_out_bind_write[4]}
      ^ vpu_y_out_bind_din[5][31:0]
      ^ {31'b0, vpu_y_out_bind_write[5]}
      ^ vpu_y_out_bind_din[6][31:0]
      ^ {31'b0, vpu_y_out_bind_write[6]}
      ^ vpu_y_out_bind_din[7][31:0]
      ^ {31'b0, vpu_y_out_bind_write[7]}
      ^ vpu_y_out_bind_din[8][31:0]
      ^ {31'b0, vpu_y_out_bind_write[8]}
      ^ vpu_y_out_bind_din[9][31:0]
      ^ {31'b0, vpu_y_out_bind_write[9]}
      ^ vpu_y_out_bind_din[10][31:0]
      ^ {31'b0, vpu_y_out_bind_write[10]}
      ^ vpu_y_out_bind_din[11][31:0]
      ^ {31'b0, vpu_y_out_bind_write[11]}
      ^ vpu_y_out_bind_din[12][31:0]
      ^ {31'b0, vpu_y_out_bind_write[12]}
      ^ vpu_y_out_bind_din[13][31:0]
      ^ {31'b0, vpu_y_out_bind_write[13]}
      ^ vpu_y_out_bind_din[14][31:0]
      ^ {31'b0, vpu_y_out_bind_write[14]}
      ^ vpu_y_out_bind_din[15][31:0]
      ^ {31'b0, vpu_y_out_bind_write[15]}
      ^ {31'b0, vpu_b_mem_read[0]}
      ;
endmodule
