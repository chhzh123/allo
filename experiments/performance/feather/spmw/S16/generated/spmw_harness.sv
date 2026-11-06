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

  wire [15:0] pe_x_mem_dout [0:127];
  wire pe_x_mem_empty_n [0:127];
  wire pe_x_mem_read [0:127];
  genvar g_pe_x_mem;
  generate
    for (g_pe_x_mem = 0; g_pe_x_mem < 128; g_pe_x_mem = g_pe_x_mem + 1) begin : gen_pe_x_mem
      reg [31:0] lf_pe_x_mem;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_pe_x_mem <= 32'h1 + g_pe_x_mem;
        else if (start) lf_pe_x_mem <= {lf_pe_x_mem[30:0], lf_pe_x_mem[31]^lf_pe_x_mem[21]^lf_pe_x_mem[1]^lf_pe_x_mem[0]};
      assign pe_x_mem_dout[g_pe_x_mem] = lf_pe_x_mem[15:0];
      assign pe_x_mem_empty_n[g_pe_x_mem] = start;
    end
  endgenerate
  wire [255:0] pe_w_mem_dout [0:127];
  wire pe_w_mem_empty_n [0:127];
  wire pe_w_mem_read [0:127];
  genvar g_pe_w_mem;
  generate
    for (g_pe_w_mem = 0; g_pe_w_mem < 128; g_pe_w_mem = g_pe_w_mem + 1) begin : gen_pe_w_mem
      reg [31:0] lf_pe_w_mem;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_pe_w_mem <= 32'h1 + g_pe_w_mem;
        else if (start) lf_pe_w_mem <= {lf_pe_w_mem[30:0], lf_pe_w_mem[31]^lf_pe_w_mem[21]^lf_pe_w_mem[1]^lf_pe_w_mem[0]};
      assign pe_w_mem_dout[g_pe_w_mem] = {8{lf_pe_w_mem}};
      assign pe_w_mem_empty_n[g_pe_w_mem] = start;
    end
  endgenerate
  wire [31:0] switch_out_l_bind_din [0:7];
  wire switch_out_l_bind_write [0:7];
  wire switch_out_l_bind_full_n [0:7];
  genvar g_switch_out_l_bind;
  generate
    for (g_switch_out_l_bind = 0; g_switch_out_l_bind < 8; g_switch_out_l_bind = g_switch_out_l_bind + 1) begin : gen_switch_out_l_bind
      assign switch_out_l_bind_full_n[g_switch_out_l_bind] = 1'b1;
    end
  endgenerate
  wire [31:0] switch_out_r_bind_din [0:7];
  wire switch_out_r_bind_write [0:7];
  wire switch_out_r_bind_full_n [0:7];
  genvar g_switch_out_r_bind;
  generate
    for (g_switch_out_r_bind = 0; g_switch_out_r_bind < 8; g_switch_out_r_bind = g_switch_out_r_bind + 1) begin : gen_switch_out_r_bind
      assign switch_out_r_bind_full_n[g_switch_out_r_bind] = 1'b1;
    end
  endgenerate
  wire [15:0] switch_cmd_mem_dout [0:63];
  wire switch_cmd_mem_empty_n [0:63];
  wire switch_cmd_mem_read [0:63];
  genvar g_switch_cmd_mem;
  generate
    for (g_switch_cmd_mem = 0; g_switch_cmd_mem < 64; g_switch_cmd_mem = g_switch_cmd_mem + 1) begin : gen_switch_cmd_mem
      reg [31:0] lf_switch_cmd_mem;
      always @(posedge ap_clk)
        if (!ap_rst_n) lf_switch_cmd_mem <= 32'h1 + g_switch_cmd_mem;
        else if (start) lf_switch_cmd_mem <= {lf_switch_cmd_mem[30:0], lf_switch_cmd_mem[31]^lf_switch_cmd_mem[21]^lf_switch_cmd_mem[1]^lf_switch_cmd_mem[0]};
      assign switch_cmd_mem_dout[g_switch_cmd_mem] = lf_switch_cmd_mem[15:0];
      assign switch_cmd_mem_empty_n[g_switch_cmd_mem] = start;
    end
  endgenerate

  spmw_top dut (
      .ap_clk(ap_clk),
      .ap_rst_n(ap_rst_n),
      .pe_x_mem_dout(pe_x_mem_dout),
      .pe_x_mem_empty_n(pe_x_mem_empty_n),
      .pe_x_mem_read(pe_x_mem_read),
      .pe_w_mem_dout(pe_w_mem_dout),
      .pe_w_mem_empty_n(pe_w_mem_empty_n),
      .pe_w_mem_read(pe_w_mem_read),
      .switch_out_l_bind_din(switch_out_l_bind_din),
      .switch_out_l_bind_write(switch_out_l_bind_write),
      .switch_out_l_bind_full_n(switch_out_l_bind_full_n),
      .switch_out_r_bind_din(switch_out_r_bind_din),
      .switch_out_r_bind_write(switch_out_r_bind_write),
      .switch_out_r_bind_full_n(switch_out_r_bind_full_n),
      .switch_cmd_mem_dout(switch_cmd_mem_dout),
      .switch_cmd_mem_empty_n(switch_cmd_mem_empty_n),
      .switch_cmd_mem_read(switch_cmd_mem_read));

  // Fold every output into one register: a dangling result is a result
  // synthesis is entitled to delete.
  always @(posedge ap_clk)
    if (!ap_rst_n) sig <= 32'b0;
    else sig <= sig
      ^ {31'b0, pe_x_mem_read[0]}
      ^ {31'b0, pe_w_mem_read[0]}
      ^ switch_out_l_bind_din[0][31:0]
      ^ {31'b0, switch_out_l_bind_write[0]}
      ^ switch_out_l_bind_din[1][31:0]
      ^ {31'b0, switch_out_l_bind_write[1]}
      ^ switch_out_l_bind_din[2][31:0]
      ^ {31'b0, switch_out_l_bind_write[2]}
      ^ switch_out_l_bind_din[3][31:0]
      ^ {31'b0, switch_out_l_bind_write[3]}
      ^ switch_out_l_bind_din[4][31:0]
      ^ {31'b0, switch_out_l_bind_write[4]}
      ^ switch_out_l_bind_din[5][31:0]
      ^ {31'b0, switch_out_l_bind_write[5]}
      ^ switch_out_l_bind_din[6][31:0]
      ^ {31'b0, switch_out_l_bind_write[6]}
      ^ switch_out_l_bind_din[7][31:0]
      ^ {31'b0, switch_out_l_bind_write[7]}
      ^ switch_out_r_bind_din[0][31:0]
      ^ {31'b0, switch_out_r_bind_write[0]}
      ^ switch_out_r_bind_din[1][31:0]
      ^ {31'b0, switch_out_r_bind_write[1]}
      ^ switch_out_r_bind_din[2][31:0]
      ^ {31'b0, switch_out_r_bind_write[2]}
      ^ switch_out_r_bind_din[3][31:0]
      ^ {31'b0, switch_out_r_bind_write[3]}
      ^ switch_out_r_bind_din[4][31:0]
      ^ {31'b0, switch_out_r_bind_write[4]}
      ^ switch_out_r_bind_din[5][31:0]
      ^ {31'b0, switch_out_r_bind_write[5]}
      ^ switch_out_r_bind_din[6][31:0]
      ^ {31'b0, switch_out_r_bind_write[6]}
      ^ switch_out_r_bind_din[7][31:0]
      ^ {31'b0, switch_out_r_bind_write[7]}
      ^ {31'b0, switch_cmd_mem_read[0]}
      ;
endmodule
