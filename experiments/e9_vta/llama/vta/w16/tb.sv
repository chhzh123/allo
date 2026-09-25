
`timescale 1ns/1ps
module tb;
  localparam DIM = 16, L = 64, SHIFT = 12;
  localparam KB = 128, NB = 8, KBC = 32, NBC = 8;
  localparam KCH = KB / KBC, NCH = NB / NBC;
  localparam NINP = KB * L, NWGT = KB * NB, NOUT = NB * L, NACC = NBC * L;
  reg clk = 1'b0, rst = 1'b1;
  always #1 clk = ~clk;

  reg  [7:0]  inpm [0:NINP-1][0:DIM-1];
  reg  [7:0]  wgtm [0:NWGT-1][0:DIM-1][0:DIM-1];
  reg  [31:0] uopm [0:L-1];
  reg  [31:0] accm [0:NACC-1][0:DIM-1];
  reg  [31:0] outm [0:NOUT-1][0:DIM-1];
  reg  [31:0] want [0:NOUT-1][0:DIM-1];
  integer inp_base = 0, wgt_base = 0;  // the pages the load unit has swapped in

  reg  signed [7:0]  inp_d [0:DIM-1];
  reg  signed [7:0]  wgt_d [0:DIM-1][0:DIM-1];
  reg  signed [31:0] acc_d [0:DIM-1];
  // Each unit gets its own read-valid, as in vta_micro_bench.
  reg  inp_dv = 0, wgt_dv = 0;
  reg  g_acc_dv = 0, a_acc_dv = 0, g_uop_dv = 0, a_uop_dv = 0;
  reg  [10:0] u0 = 0, u1 = 0; reg [9:0] u2 = 0;

  reg sel = 0;                      // 0 = GEMM, 1 = ALU: never both
  reg g_start = 0, a_start = 0;
  reg [2:0] alu_op = 3'd0; reg [15:0] alu_imm = 16'd0;
  // The GEMM's decoded instruction, held while it runs.
  reg g_reset = 1'b0; reg [13:0] g_lp1 = 14'd1;

  wire g_uop_iv, g_inp_iv, g_wgt_iv, g_acc_iv, g_acc_wv, g_done;
  wire a_uop_iv, a_acc_iv, a_acc_wv, a_done;
  wire [10:0] g_uop_ib, inp_ib, g_acc_ib, g_acc_wi;
  wire [10:0] a_uop_ib, a_acc_ib, a_acc_wi;
  wire [9:0]  wgt_ib;
  wire signed [31:0] g_accwd [0:DIM-1];
  wire signed [31:0] a_accwd [0:DIM-1];
  wire signed [7:0]  a_outwd [0:DIM-1];

  wire        uop_iv  = sel ? a_uop_iv : g_uop_iv;
  wire [10:0] uop_ib  = sel ? a_uop_ib : g_uop_ib;
  wire        acc_iv  = sel ? a_acc_iv : g_acc_iv;
  wire [10:0] acc_ib  = sel ? a_acc_ib : g_acc_ib;
  wire        acc_wv  = sel ? a_acc_wv : g_acc_wv;
  wire [10:0] acc_wi  = sel ? a_acc_wi : g_acc_wi;
  wire signed [31:0] acc_wd [0:DIM-1];
  genvar gi;
  generate for (gi = 0; gi < DIM; gi = gi + 1) begin : wsel
    assign acc_wd[gi] = sel ? a_accwd[gi] : g_accwd[gi];
  end endgenerate

  integer cyc = 0, t0 = -1, t1 = -1, i, j, k, cn, ck, errs = 0, n_instr = 0;
  integer c_gemm = 0, c_reset = 0, c_alu = 0, ts;

  TensorGemm gemm (
    .clock(clk), .reset(rst), .io_start(g_start), .io_done(g_done),
    .io_dec_wgt_1(10'd8), .io_dec_wgt_0(10'd1),
    .io_dec_inp_1(11'd64), .io_dec_inp_0(11'd0),
    .io_dec_acc_1(11'd0), .io_dec_acc_0(11'd64),
    .io_dec_empty_0(1'b0),
    .io_dec_lp_1(g_lp1), .io_dec_lp_0(14'd8),
    .io_dec_uop_end(14'd64), .io_dec_uop_begin(13'd0),
    .io_dec_reset(g_reset), .io_dec_push_next(1'b0), .io_dec_push_prev(1'b0),
    .io_dec_pop_next(1'b0), .io_dec_pop_prev(1'b0), .io_dec_op(3'd0),
    .io_uop_idx_valid(g_uop_iv), .io_uop_idx_bits(g_uop_ib),
    .io_uop_data_valid(g_uop_dv),
    .io_uop_data_bits_u2(u2), .io_uop_data_bits_u1(u1), .io_uop_data_bits_u0(u0),
    .io_inp_rd_0_idx_valid(g_inp_iv), .io_inp_rd_0_idx_bits(inp_ib),
    .io_inp_rd_0_data_valid(inp_dv),
    .io_inp_rd_0_data_bits_0_0(inp_d[0]),
    .io_inp_rd_0_data_bits_0_1(inp_d[1]),
    .io_inp_rd_0_data_bits_0_2(inp_d[2]),
    .io_inp_rd_0_data_bits_0_3(inp_d[3]),
    .io_inp_rd_0_data_bits_0_4(inp_d[4]),
    .io_inp_rd_0_data_bits_0_5(inp_d[5]),
    .io_inp_rd_0_data_bits_0_6(inp_d[6]),
    .io_inp_rd_0_data_bits_0_7(inp_d[7]),
    .io_inp_rd_0_data_bits_0_8(inp_d[8]),
    .io_inp_rd_0_data_bits_0_9(inp_d[9]),
    .io_inp_rd_0_data_bits_0_10(inp_d[10]),
    .io_inp_rd_0_data_bits_0_11(inp_d[11]),
    .io_inp_rd_0_data_bits_0_12(inp_d[12]),
    .io_inp_rd_0_data_bits_0_13(inp_d[13]),
    .io_inp_rd_0_data_bits_0_14(inp_d[14]),
    .io_inp_rd_0_data_bits_0_15(inp_d[15]),
    .io_inp_wr_0_valid(), .io_inp_wr_0_bits_idx(),
    .io_inp_wr_0_bits_data_0_0(),
    .io_inp_wr_0_bits_data_0_1(),
    .io_inp_wr_0_bits_data_0_2(),
    .io_inp_wr_0_bits_data_0_3(),
    .io_inp_wr_0_bits_data_0_4(),
    .io_inp_wr_0_bits_data_0_5(),
    .io_inp_wr_0_bits_data_0_6(),
    .io_inp_wr_0_bits_data_0_7(),
    .io_inp_wr_0_bits_data_0_8(),
    .io_inp_wr_0_bits_data_0_9(),
    .io_inp_wr_0_bits_data_0_10(),
    .io_inp_wr_0_bits_data_0_11(),
    .io_inp_wr_0_bits_data_0_12(),
    .io_inp_wr_0_bits_data_0_13(),
    .io_inp_wr_0_bits_data_0_14(),
    .io_inp_wr_0_bits_data_0_15(),
    .io_wgt_rd_0_idx_valid(g_wgt_iv), .io_wgt_rd_0_idx_bits(wgt_ib),
    .io_wgt_rd_0_data_valid(wgt_dv),
    .io_wgt_rd_0_data_bits_0_0(wgt_d[0][0]),
    .io_wgt_rd_0_data_bits_0_1(wgt_d[0][1]),
    .io_wgt_rd_0_data_bits_0_2(wgt_d[0][2]),
    .io_wgt_rd_0_data_bits_0_3(wgt_d[0][3]),
    .io_wgt_rd_0_data_bits_0_4(wgt_d[0][4]),
    .io_wgt_rd_0_data_bits_0_5(wgt_d[0][5]),
    .io_wgt_rd_0_data_bits_0_6(wgt_d[0][6]),
    .io_wgt_rd_0_data_bits_0_7(wgt_d[0][7]),
    .io_wgt_rd_0_data_bits_0_8(wgt_d[0][8]),
    .io_wgt_rd_0_data_bits_0_9(wgt_d[0][9]),
    .io_wgt_rd_0_data_bits_0_10(wgt_d[0][10]),
    .io_wgt_rd_0_data_bits_0_11(wgt_d[0][11]),
    .io_wgt_rd_0_data_bits_0_12(wgt_d[0][12]),
    .io_wgt_rd_0_data_bits_0_13(wgt_d[0][13]),
    .io_wgt_rd_0_data_bits_0_14(wgt_d[0][14]),
    .io_wgt_rd_0_data_bits_0_15(wgt_d[0][15]),
    .io_wgt_rd_0_data_bits_1_0(wgt_d[1][0]),
    .io_wgt_rd_0_data_bits_1_1(wgt_d[1][1]),
    .io_wgt_rd_0_data_bits_1_2(wgt_d[1][2]),
    .io_wgt_rd_0_data_bits_1_3(wgt_d[1][3]),
    .io_wgt_rd_0_data_bits_1_4(wgt_d[1][4]),
    .io_wgt_rd_0_data_bits_1_5(wgt_d[1][5]),
    .io_wgt_rd_0_data_bits_1_6(wgt_d[1][6]),
    .io_wgt_rd_0_data_bits_1_7(wgt_d[1][7]),
    .io_wgt_rd_0_data_bits_1_8(wgt_d[1][8]),
    .io_wgt_rd_0_data_bits_1_9(wgt_d[1][9]),
    .io_wgt_rd_0_data_bits_1_10(wgt_d[1][10]),
    .io_wgt_rd_0_data_bits_1_11(wgt_d[1][11]),
    .io_wgt_rd_0_data_bits_1_12(wgt_d[1][12]),
    .io_wgt_rd_0_data_bits_1_13(wgt_d[1][13]),
    .io_wgt_rd_0_data_bits_1_14(wgt_d[1][14]),
    .io_wgt_rd_0_data_bits_1_15(wgt_d[1][15]),
    .io_wgt_rd_0_data_bits_2_0(wgt_d[2][0]),
    .io_wgt_rd_0_data_bits_2_1(wgt_d[2][1]),
    .io_wgt_rd_0_data_bits_2_2(wgt_d[2][2]),
    .io_wgt_rd_0_data_bits_2_3(wgt_d[2][3]),
    .io_wgt_rd_0_data_bits_2_4(wgt_d[2][4]),
    .io_wgt_rd_0_data_bits_2_5(wgt_d[2][5]),
    .io_wgt_rd_0_data_bits_2_6(wgt_d[2][6]),
    .io_wgt_rd_0_data_bits_2_7(wgt_d[2][7]),
    .io_wgt_rd_0_data_bits_2_8(wgt_d[2][8]),
    .io_wgt_rd_0_data_bits_2_9(wgt_d[2][9]),
    .io_wgt_rd_0_data_bits_2_10(wgt_d[2][10]),
    .io_wgt_rd_0_data_bits_2_11(wgt_d[2][11]),
    .io_wgt_rd_0_data_bits_2_12(wgt_d[2][12]),
    .io_wgt_rd_0_data_bits_2_13(wgt_d[2][13]),
    .io_wgt_rd_0_data_bits_2_14(wgt_d[2][14]),
    .io_wgt_rd_0_data_bits_2_15(wgt_d[2][15]),
    .io_wgt_rd_0_data_bits_3_0(wgt_d[3][0]),
    .io_wgt_rd_0_data_bits_3_1(wgt_d[3][1]),
    .io_wgt_rd_0_data_bits_3_2(wgt_d[3][2]),
    .io_wgt_rd_0_data_bits_3_3(wgt_d[3][3]),
    .io_wgt_rd_0_data_bits_3_4(wgt_d[3][4]),
    .io_wgt_rd_0_data_bits_3_5(wgt_d[3][5]),
    .io_wgt_rd_0_data_bits_3_6(wgt_d[3][6]),
    .io_wgt_rd_0_data_bits_3_7(wgt_d[3][7]),
    .io_wgt_rd_0_data_bits_3_8(wgt_d[3][8]),
    .io_wgt_rd_0_data_bits_3_9(wgt_d[3][9]),
    .io_wgt_rd_0_data_bits_3_10(wgt_d[3][10]),
    .io_wgt_rd_0_data_bits_3_11(wgt_d[3][11]),
    .io_wgt_rd_0_data_bits_3_12(wgt_d[3][12]),
    .io_wgt_rd_0_data_bits_3_13(wgt_d[3][13]),
    .io_wgt_rd_0_data_bits_3_14(wgt_d[3][14]),
    .io_wgt_rd_0_data_bits_3_15(wgt_d[3][15]),
    .io_wgt_rd_0_data_bits_4_0(wgt_d[4][0]),
    .io_wgt_rd_0_data_bits_4_1(wgt_d[4][1]),
    .io_wgt_rd_0_data_bits_4_2(wgt_d[4][2]),
    .io_wgt_rd_0_data_bits_4_3(wgt_d[4][3]),
    .io_wgt_rd_0_data_bits_4_4(wgt_d[4][4]),
    .io_wgt_rd_0_data_bits_4_5(wgt_d[4][5]),
    .io_wgt_rd_0_data_bits_4_6(wgt_d[4][6]),
    .io_wgt_rd_0_data_bits_4_7(wgt_d[4][7]),
    .io_wgt_rd_0_data_bits_4_8(wgt_d[4][8]),
    .io_wgt_rd_0_data_bits_4_9(wgt_d[4][9]),
    .io_wgt_rd_0_data_bits_4_10(wgt_d[4][10]),
    .io_wgt_rd_0_data_bits_4_11(wgt_d[4][11]),
    .io_wgt_rd_0_data_bits_4_12(wgt_d[4][12]),
    .io_wgt_rd_0_data_bits_4_13(wgt_d[4][13]),
    .io_wgt_rd_0_data_bits_4_14(wgt_d[4][14]),
    .io_wgt_rd_0_data_bits_4_15(wgt_d[4][15]),
    .io_wgt_rd_0_data_bits_5_0(wgt_d[5][0]),
    .io_wgt_rd_0_data_bits_5_1(wgt_d[5][1]),
    .io_wgt_rd_0_data_bits_5_2(wgt_d[5][2]),
    .io_wgt_rd_0_data_bits_5_3(wgt_d[5][3]),
    .io_wgt_rd_0_data_bits_5_4(wgt_d[5][4]),
    .io_wgt_rd_0_data_bits_5_5(wgt_d[5][5]),
    .io_wgt_rd_0_data_bits_5_6(wgt_d[5][6]),
    .io_wgt_rd_0_data_bits_5_7(wgt_d[5][7]),
    .io_wgt_rd_0_data_bits_5_8(wgt_d[5][8]),
    .io_wgt_rd_0_data_bits_5_9(wgt_d[5][9]),
    .io_wgt_rd_0_data_bits_5_10(wgt_d[5][10]),
    .io_wgt_rd_0_data_bits_5_11(wgt_d[5][11]),
    .io_wgt_rd_0_data_bits_5_12(wgt_d[5][12]),
    .io_wgt_rd_0_data_bits_5_13(wgt_d[5][13]),
    .io_wgt_rd_0_data_bits_5_14(wgt_d[5][14]),
    .io_wgt_rd_0_data_bits_5_15(wgt_d[5][15]),
    .io_wgt_rd_0_data_bits_6_0(wgt_d[6][0]),
    .io_wgt_rd_0_data_bits_6_1(wgt_d[6][1]),
    .io_wgt_rd_0_data_bits_6_2(wgt_d[6][2]),
    .io_wgt_rd_0_data_bits_6_3(wgt_d[6][3]),
    .io_wgt_rd_0_data_bits_6_4(wgt_d[6][4]),
    .io_wgt_rd_0_data_bits_6_5(wgt_d[6][5]),
    .io_wgt_rd_0_data_bits_6_6(wgt_d[6][6]),
    .io_wgt_rd_0_data_bits_6_7(wgt_d[6][7]),
    .io_wgt_rd_0_data_bits_6_8(wgt_d[6][8]),
    .io_wgt_rd_0_data_bits_6_9(wgt_d[6][9]),
    .io_wgt_rd_0_data_bits_6_10(wgt_d[6][10]),
    .io_wgt_rd_0_data_bits_6_11(wgt_d[6][11]),
    .io_wgt_rd_0_data_bits_6_12(wgt_d[6][12]),
    .io_wgt_rd_0_data_bits_6_13(wgt_d[6][13]),
    .io_wgt_rd_0_data_bits_6_14(wgt_d[6][14]),
    .io_wgt_rd_0_data_bits_6_15(wgt_d[6][15]),
    .io_wgt_rd_0_data_bits_7_0(wgt_d[7][0]),
    .io_wgt_rd_0_data_bits_7_1(wgt_d[7][1]),
    .io_wgt_rd_0_data_bits_7_2(wgt_d[7][2]),
    .io_wgt_rd_0_data_bits_7_3(wgt_d[7][3]),
    .io_wgt_rd_0_data_bits_7_4(wgt_d[7][4]),
    .io_wgt_rd_0_data_bits_7_5(wgt_d[7][5]),
    .io_wgt_rd_0_data_bits_7_6(wgt_d[7][6]),
    .io_wgt_rd_0_data_bits_7_7(wgt_d[7][7]),
    .io_wgt_rd_0_data_bits_7_8(wgt_d[7][8]),
    .io_wgt_rd_0_data_bits_7_9(wgt_d[7][9]),
    .io_wgt_rd_0_data_bits_7_10(wgt_d[7][10]),
    .io_wgt_rd_0_data_bits_7_11(wgt_d[7][11]),
    .io_wgt_rd_0_data_bits_7_12(wgt_d[7][12]),
    .io_wgt_rd_0_data_bits_7_13(wgt_d[7][13]),
    .io_wgt_rd_0_data_bits_7_14(wgt_d[7][14]),
    .io_wgt_rd_0_data_bits_7_15(wgt_d[7][15]),
    .io_wgt_rd_0_data_bits_8_0(wgt_d[8][0]),
    .io_wgt_rd_0_data_bits_8_1(wgt_d[8][1]),
    .io_wgt_rd_0_data_bits_8_2(wgt_d[8][2]),
    .io_wgt_rd_0_data_bits_8_3(wgt_d[8][3]),
    .io_wgt_rd_0_data_bits_8_4(wgt_d[8][4]),
    .io_wgt_rd_0_data_bits_8_5(wgt_d[8][5]),
    .io_wgt_rd_0_data_bits_8_6(wgt_d[8][6]),
    .io_wgt_rd_0_data_bits_8_7(wgt_d[8][7]),
    .io_wgt_rd_0_data_bits_8_8(wgt_d[8][8]),
    .io_wgt_rd_0_data_bits_8_9(wgt_d[8][9]),
    .io_wgt_rd_0_data_bits_8_10(wgt_d[8][10]),
    .io_wgt_rd_0_data_bits_8_11(wgt_d[8][11]),
    .io_wgt_rd_0_data_bits_8_12(wgt_d[8][12]),
    .io_wgt_rd_0_data_bits_8_13(wgt_d[8][13]),
    .io_wgt_rd_0_data_bits_8_14(wgt_d[8][14]),
    .io_wgt_rd_0_data_bits_8_15(wgt_d[8][15]),
    .io_wgt_rd_0_data_bits_9_0(wgt_d[9][0]),
    .io_wgt_rd_0_data_bits_9_1(wgt_d[9][1]),
    .io_wgt_rd_0_data_bits_9_2(wgt_d[9][2]),
    .io_wgt_rd_0_data_bits_9_3(wgt_d[9][3]),
    .io_wgt_rd_0_data_bits_9_4(wgt_d[9][4]),
    .io_wgt_rd_0_data_bits_9_5(wgt_d[9][5]),
    .io_wgt_rd_0_data_bits_9_6(wgt_d[9][6]),
    .io_wgt_rd_0_data_bits_9_7(wgt_d[9][7]),
    .io_wgt_rd_0_data_bits_9_8(wgt_d[9][8]),
    .io_wgt_rd_0_data_bits_9_9(wgt_d[9][9]),
    .io_wgt_rd_0_data_bits_9_10(wgt_d[9][10]),
    .io_wgt_rd_0_data_bits_9_11(wgt_d[9][11]),
    .io_wgt_rd_0_data_bits_9_12(wgt_d[9][12]),
    .io_wgt_rd_0_data_bits_9_13(wgt_d[9][13]),
    .io_wgt_rd_0_data_bits_9_14(wgt_d[9][14]),
    .io_wgt_rd_0_data_bits_9_15(wgt_d[9][15]),
    .io_wgt_rd_0_data_bits_10_0(wgt_d[10][0]),
    .io_wgt_rd_0_data_bits_10_1(wgt_d[10][1]),
    .io_wgt_rd_0_data_bits_10_2(wgt_d[10][2]),
    .io_wgt_rd_0_data_bits_10_3(wgt_d[10][3]),
    .io_wgt_rd_0_data_bits_10_4(wgt_d[10][4]),
    .io_wgt_rd_0_data_bits_10_5(wgt_d[10][5]),
    .io_wgt_rd_0_data_bits_10_6(wgt_d[10][6]),
    .io_wgt_rd_0_data_bits_10_7(wgt_d[10][7]),
    .io_wgt_rd_0_data_bits_10_8(wgt_d[10][8]),
    .io_wgt_rd_0_data_bits_10_9(wgt_d[10][9]),
    .io_wgt_rd_0_data_bits_10_10(wgt_d[10][10]),
    .io_wgt_rd_0_data_bits_10_11(wgt_d[10][11]),
    .io_wgt_rd_0_data_bits_10_12(wgt_d[10][12]),
    .io_wgt_rd_0_data_bits_10_13(wgt_d[10][13]),
    .io_wgt_rd_0_data_bits_10_14(wgt_d[10][14]),
    .io_wgt_rd_0_data_bits_10_15(wgt_d[10][15]),
    .io_wgt_rd_0_data_bits_11_0(wgt_d[11][0]),
    .io_wgt_rd_0_data_bits_11_1(wgt_d[11][1]),
    .io_wgt_rd_0_data_bits_11_2(wgt_d[11][2]),
    .io_wgt_rd_0_data_bits_11_3(wgt_d[11][3]),
    .io_wgt_rd_0_data_bits_11_4(wgt_d[11][4]),
    .io_wgt_rd_0_data_bits_11_5(wgt_d[11][5]),
    .io_wgt_rd_0_data_bits_11_6(wgt_d[11][6]),
    .io_wgt_rd_0_data_bits_11_7(wgt_d[11][7]),
    .io_wgt_rd_0_data_bits_11_8(wgt_d[11][8]),
    .io_wgt_rd_0_data_bits_11_9(wgt_d[11][9]),
    .io_wgt_rd_0_data_bits_11_10(wgt_d[11][10]),
    .io_wgt_rd_0_data_bits_11_11(wgt_d[11][11]),
    .io_wgt_rd_0_data_bits_11_12(wgt_d[11][12]),
    .io_wgt_rd_0_data_bits_11_13(wgt_d[11][13]),
    .io_wgt_rd_0_data_bits_11_14(wgt_d[11][14]),
    .io_wgt_rd_0_data_bits_11_15(wgt_d[11][15]),
    .io_wgt_rd_0_data_bits_12_0(wgt_d[12][0]),
    .io_wgt_rd_0_data_bits_12_1(wgt_d[12][1]),
    .io_wgt_rd_0_data_bits_12_2(wgt_d[12][2]),
    .io_wgt_rd_0_data_bits_12_3(wgt_d[12][3]),
    .io_wgt_rd_0_data_bits_12_4(wgt_d[12][4]),
    .io_wgt_rd_0_data_bits_12_5(wgt_d[12][5]),
    .io_wgt_rd_0_data_bits_12_6(wgt_d[12][6]),
    .io_wgt_rd_0_data_bits_12_7(wgt_d[12][7]),
    .io_wgt_rd_0_data_bits_12_8(wgt_d[12][8]),
    .io_wgt_rd_0_data_bits_12_9(wgt_d[12][9]),
    .io_wgt_rd_0_data_bits_12_10(wgt_d[12][10]),
    .io_wgt_rd_0_data_bits_12_11(wgt_d[12][11]),
    .io_wgt_rd_0_data_bits_12_12(wgt_d[12][12]),
    .io_wgt_rd_0_data_bits_12_13(wgt_d[12][13]),
    .io_wgt_rd_0_data_bits_12_14(wgt_d[12][14]),
    .io_wgt_rd_0_data_bits_12_15(wgt_d[12][15]),
    .io_wgt_rd_0_data_bits_13_0(wgt_d[13][0]),
    .io_wgt_rd_0_data_bits_13_1(wgt_d[13][1]),
    .io_wgt_rd_0_data_bits_13_2(wgt_d[13][2]),
    .io_wgt_rd_0_data_bits_13_3(wgt_d[13][3]),
    .io_wgt_rd_0_data_bits_13_4(wgt_d[13][4]),
    .io_wgt_rd_0_data_bits_13_5(wgt_d[13][5]),
    .io_wgt_rd_0_data_bits_13_6(wgt_d[13][6]),
    .io_wgt_rd_0_data_bits_13_7(wgt_d[13][7]),
    .io_wgt_rd_0_data_bits_13_8(wgt_d[13][8]),
    .io_wgt_rd_0_data_bits_13_9(wgt_d[13][9]),
    .io_wgt_rd_0_data_bits_13_10(wgt_d[13][10]),
    .io_wgt_rd_0_data_bits_13_11(wgt_d[13][11]),
    .io_wgt_rd_0_data_bits_13_12(wgt_d[13][12]),
    .io_wgt_rd_0_data_bits_13_13(wgt_d[13][13]),
    .io_wgt_rd_0_data_bits_13_14(wgt_d[13][14]),
    .io_wgt_rd_0_data_bits_13_15(wgt_d[13][15]),
    .io_wgt_rd_0_data_bits_14_0(wgt_d[14][0]),
    .io_wgt_rd_0_data_bits_14_1(wgt_d[14][1]),
    .io_wgt_rd_0_data_bits_14_2(wgt_d[14][2]),
    .io_wgt_rd_0_data_bits_14_3(wgt_d[14][3]),
    .io_wgt_rd_0_data_bits_14_4(wgt_d[14][4]),
    .io_wgt_rd_0_data_bits_14_5(wgt_d[14][5]),
    .io_wgt_rd_0_data_bits_14_6(wgt_d[14][6]),
    .io_wgt_rd_0_data_bits_14_7(wgt_d[14][7]),
    .io_wgt_rd_0_data_bits_14_8(wgt_d[14][8]),
    .io_wgt_rd_0_data_bits_14_9(wgt_d[14][9]),
    .io_wgt_rd_0_data_bits_14_10(wgt_d[14][10]),
    .io_wgt_rd_0_data_bits_14_11(wgt_d[14][11]),
    .io_wgt_rd_0_data_bits_14_12(wgt_d[14][12]),
    .io_wgt_rd_0_data_bits_14_13(wgt_d[14][13]),
    .io_wgt_rd_0_data_bits_14_14(wgt_d[14][14]),
    .io_wgt_rd_0_data_bits_14_15(wgt_d[14][15]),
    .io_wgt_rd_0_data_bits_15_0(wgt_d[15][0]),
    .io_wgt_rd_0_data_bits_15_1(wgt_d[15][1]),
    .io_wgt_rd_0_data_bits_15_2(wgt_d[15][2]),
    .io_wgt_rd_0_data_bits_15_3(wgt_d[15][3]),
    .io_wgt_rd_0_data_bits_15_4(wgt_d[15][4]),
    .io_wgt_rd_0_data_bits_15_5(wgt_d[15][5]),
    .io_wgt_rd_0_data_bits_15_6(wgt_d[15][6]),
    .io_wgt_rd_0_data_bits_15_7(wgt_d[15][7]),
    .io_wgt_rd_0_data_bits_15_8(wgt_d[15][8]),
    .io_wgt_rd_0_data_bits_15_9(wgt_d[15][9]),
    .io_wgt_rd_0_data_bits_15_10(wgt_d[15][10]),
    .io_wgt_rd_0_data_bits_15_11(wgt_d[15][11]),
    .io_wgt_rd_0_data_bits_15_12(wgt_d[15][12]),
    .io_wgt_rd_0_data_bits_15_13(wgt_d[15][13]),
    .io_wgt_rd_0_data_bits_15_14(wgt_d[15][14]),
    .io_wgt_rd_0_data_bits_15_15(wgt_d[15][15]),
    .io_wgt_wr_0_valid(), .io_wgt_wr_0_bits_idx(),
    .io_wgt_wr_0_bits_data_0_0(),
    .io_wgt_wr_0_bits_data_0_1(),
    .io_wgt_wr_0_bits_data_0_2(),
    .io_wgt_wr_0_bits_data_0_3(),
    .io_wgt_wr_0_bits_data_0_4(),
    .io_wgt_wr_0_bits_data_0_5(),
    .io_wgt_wr_0_bits_data_0_6(),
    .io_wgt_wr_0_bits_data_0_7(),
    .io_wgt_wr_0_bits_data_0_8(),
    .io_wgt_wr_0_bits_data_0_9(),
    .io_wgt_wr_0_bits_data_0_10(),
    .io_wgt_wr_0_bits_data_0_11(),
    .io_wgt_wr_0_bits_data_0_12(),
    .io_wgt_wr_0_bits_data_0_13(),
    .io_wgt_wr_0_bits_data_0_14(),
    .io_wgt_wr_0_bits_data_0_15(),
    .io_wgt_wr_0_bits_data_1_0(),
    .io_wgt_wr_0_bits_data_1_1(),
    .io_wgt_wr_0_bits_data_1_2(),
    .io_wgt_wr_0_bits_data_1_3(),
    .io_wgt_wr_0_bits_data_1_4(),
    .io_wgt_wr_0_bits_data_1_5(),
    .io_wgt_wr_0_bits_data_1_6(),
    .io_wgt_wr_0_bits_data_1_7(),
    .io_wgt_wr_0_bits_data_1_8(),
    .io_wgt_wr_0_bits_data_1_9(),
    .io_wgt_wr_0_bits_data_1_10(),
    .io_wgt_wr_0_bits_data_1_11(),
    .io_wgt_wr_0_bits_data_1_12(),
    .io_wgt_wr_0_bits_data_1_13(),
    .io_wgt_wr_0_bits_data_1_14(),
    .io_wgt_wr_0_bits_data_1_15(),
    .io_wgt_wr_0_bits_data_2_0(),
    .io_wgt_wr_0_bits_data_2_1(),
    .io_wgt_wr_0_bits_data_2_2(),
    .io_wgt_wr_0_bits_data_2_3(),
    .io_wgt_wr_0_bits_data_2_4(),
    .io_wgt_wr_0_bits_data_2_5(),
    .io_wgt_wr_0_bits_data_2_6(),
    .io_wgt_wr_0_bits_data_2_7(),
    .io_wgt_wr_0_bits_data_2_8(),
    .io_wgt_wr_0_bits_data_2_9(),
    .io_wgt_wr_0_bits_data_2_10(),
    .io_wgt_wr_0_bits_data_2_11(),
    .io_wgt_wr_0_bits_data_2_12(),
    .io_wgt_wr_0_bits_data_2_13(),
    .io_wgt_wr_0_bits_data_2_14(),
    .io_wgt_wr_0_bits_data_2_15(),
    .io_wgt_wr_0_bits_data_3_0(),
    .io_wgt_wr_0_bits_data_3_1(),
    .io_wgt_wr_0_bits_data_3_2(),
    .io_wgt_wr_0_bits_data_3_3(),
    .io_wgt_wr_0_bits_data_3_4(),
    .io_wgt_wr_0_bits_data_3_5(),
    .io_wgt_wr_0_bits_data_3_6(),
    .io_wgt_wr_0_bits_data_3_7(),
    .io_wgt_wr_0_bits_data_3_8(),
    .io_wgt_wr_0_bits_data_3_9(),
    .io_wgt_wr_0_bits_data_3_10(),
    .io_wgt_wr_0_bits_data_3_11(),
    .io_wgt_wr_0_bits_data_3_12(),
    .io_wgt_wr_0_bits_data_3_13(),
    .io_wgt_wr_0_bits_data_3_14(),
    .io_wgt_wr_0_bits_data_3_15(),
    .io_wgt_wr_0_bits_data_4_0(),
    .io_wgt_wr_0_bits_data_4_1(),
    .io_wgt_wr_0_bits_data_4_2(),
    .io_wgt_wr_0_bits_data_4_3(),
    .io_wgt_wr_0_bits_data_4_4(),
    .io_wgt_wr_0_bits_data_4_5(),
    .io_wgt_wr_0_bits_data_4_6(),
    .io_wgt_wr_0_bits_data_4_7(),
    .io_wgt_wr_0_bits_data_4_8(),
    .io_wgt_wr_0_bits_data_4_9(),
    .io_wgt_wr_0_bits_data_4_10(),
    .io_wgt_wr_0_bits_data_4_11(),
    .io_wgt_wr_0_bits_data_4_12(),
    .io_wgt_wr_0_bits_data_4_13(),
    .io_wgt_wr_0_bits_data_4_14(),
    .io_wgt_wr_0_bits_data_4_15(),
    .io_wgt_wr_0_bits_data_5_0(),
    .io_wgt_wr_0_bits_data_5_1(),
    .io_wgt_wr_0_bits_data_5_2(),
    .io_wgt_wr_0_bits_data_5_3(),
    .io_wgt_wr_0_bits_data_5_4(),
    .io_wgt_wr_0_bits_data_5_5(),
    .io_wgt_wr_0_bits_data_5_6(),
    .io_wgt_wr_0_bits_data_5_7(),
    .io_wgt_wr_0_bits_data_5_8(),
    .io_wgt_wr_0_bits_data_5_9(),
    .io_wgt_wr_0_bits_data_5_10(),
    .io_wgt_wr_0_bits_data_5_11(),
    .io_wgt_wr_0_bits_data_5_12(),
    .io_wgt_wr_0_bits_data_5_13(),
    .io_wgt_wr_0_bits_data_5_14(),
    .io_wgt_wr_0_bits_data_5_15(),
    .io_wgt_wr_0_bits_data_6_0(),
    .io_wgt_wr_0_bits_data_6_1(),
    .io_wgt_wr_0_bits_data_6_2(),
    .io_wgt_wr_0_bits_data_6_3(),
    .io_wgt_wr_0_bits_data_6_4(),
    .io_wgt_wr_0_bits_data_6_5(),
    .io_wgt_wr_0_bits_data_6_6(),
    .io_wgt_wr_0_bits_data_6_7(),
    .io_wgt_wr_0_bits_data_6_8(),
    .io_wgt_wr_0_bits_data_6_9(),
    .io_wgt_wr_0_bits_data_6_10(),
    .io_wgt_wr_0_bits_data_6_11(),
    .io_wgt_wr_0_bits_data_6_12(),
    .io_wgt_wr_0_bits_data_6_13(),
    .io_wgt_wr_0_bits_data_6_14(),
    .io_wgt_wr_0_bits_data_6_15(),
    .io_wgt_wr_0_bits_data_7_0(),
    .io_wgt_wr_0_bits_data_7_1(),
    .io_wgt_wr_0_bits_data_7_2(),
    .io_wgt_wr_0_bits_data_7_3(),
    .io_wgt_wr_0_bits_data_7_4(),
    .io_wgt_wr_0_bits_data_7_5(),
    .io_wgt_wr_0_bits_data_7_6(),
    .io_wgt_wr_0_bits_data_7_7(),
    .io_wgt_wr_0_bits_data_7_8(),
    .io_wgt_wr_0_bits_data_7_9(),
    .io_wgt_wr_0_bits_data_7_10(),
    .io_wgt_wr_0_bits_data_7_11(),
    .io_wgt_wr_0_bits_data_7_12(),
    .io_wgt_wr_0_bits_data_7_13(),
    .io_wgt_wr_0_bits_data_7_14(),
    .io_wgt_wr_0_bits_data_7_15(),
    .io_wgt_wr_0_bits_data_8_0(),
    .io_wgt_wr_0_bits_data_8_1(),
    .io_wgt_wr_0_bits_data_8_2(),
    .io_wgt_wr_0_bits_data_8_3(),
    .io_wgt_wr_0_bits_data_8_4(),
    .io_wgt_wr_0_bits_data_8_5(),
    .io_wgt_wr_0_bits_data_8_6(),
    .io_wgt_wr_0_bits_data_8_7(),
    .io_wgt_wr_0_bits_data_8_8(),
    .io_wgt_wr_0_bits_data_8_9(),
    .io_wgt_wr_0_bits_data_8_10(),
    .io_wgt_wr_0_bits_data_8_11(),
    .io_wgt_wr_0_bits_data_8_12(),
    .io_wgt_wr_0_bits_data_8_13(),
    .io_wgt_wr_0_bits_data_8_14(),
    .io_wgt_wr_0_bits_data_8_15(),
    .io_wgt_wr_0_bits_data_9_0(),
    .io_wgt_wr_0_bits_data_9_1(),
    .io_wgt_wr_0_bits_data_9_2(),
    .io_wgt_wr_0_bits_data_9_3(),
    .io_wgt_wr_0_bits_data_9_4(),
    .io_wgt_wr_0_bits_data_9_5(),
    .io_wgt_wr_0_bits_data_9_6(),
    .io_wgt_wr_0_bits_data_9_7(),
    .io_wgt_wr_0_bits_data_9_8(),
    .io_wgt_wr_0_bits_data_9_9(),
    .io_wgt_wr_0_bits_data_9_10(),
    .io_wgt_wr_0_bits_data_9_11(),
    .io_wgt_wr_0_bits_data_9_12(),
    .io_wgt_wr_0_bits_data_9_13(),
    .io_wgt_wr_0_bits_data_9_14(),
    .io_wgt_wr_0_bits_data_9_15(),
    .io_wgt_wr_0_bits_data_10_0(),
    .io_wgt_wr_0_bits_data_10_1(),
    .io_wgt_wr_0_bits_data_10_2(),
    .io_wgt_wr_0_bits_data_10_3(),
    .io_wgt_wr_0_bits_data_10_4(),
    .io_wgt_wr_0_bits_data_10_5(),
    .io_wgt_wr_0_bits_data_10_6(),
    .io_wgt_wr_0_bits_data_10_7(),
    .io_wgt_wr_0_bits_data_10_8(),
    .io_wgt_wr_0_bits_data_10_9(),
    .io_wgt_wr_0_bits_data_10_10(),
    .io_wgt_wr_0_bits_data_10_11(),
    .io_wgt_wr_0_bits_data_10_12(),
    .io_wgt_wr_0_bits_data_10_13(),
    .io_wgt_wr_0_bits_data_10_14(),
    .io_wgt_wr_0_bits_data_10_15(),
    .io_wgt_wr_0_bits_data_11_0(),
    .io_wgt_wr_0_bits_data_11_1(),
    .io_wgt_wr_0_bits_data_11_2(),
    .io_wgt_wr_0_bits_data_11_3(),
    .io_wgt_wr_0_bits_data_11_4(),
    .io_wgt_wr_0_bits_data_11_5(),
    .io_wgt_wr_0_bits_data_11_6(),
    .io_wgt_wr_0_bits_data_11_7(),
    .io_wgt_wr_0_bits_data_11_8(),
    .io_wgt_wr_0_bits_data_11_9(),
    .io_wgt_wr_0_bits_data_11_10(),
    .io_wgt_wr_0_bits_data_11_11(),
    .io_wgt_wr_0_bits_data_11_12(),
    .io_wgt_wr_0_bits_data_11_13(),
    .io_wgt_wr_0_bits_data_11_14(),
    .io_wgt_wr_0_bits_data_11_15(),
    .io_wgt_wr_0_bits_data_12_0(),
    .io_wgt_wr_0_bits_data_12_1(),
    .io_wgt_wr_0_bits_data_12_2(),
    .io_wgt_wr_0_bits_data_12_3(),
    .io_wgt_wr_0_bits_data_12_4(),
    .io_wgt_wr_0_bits_data_12_5(),
    .io_wgt_wr_0_bits_data_12_6(),
    .io_wgt_wr_0_bits_data_12_7(),
    .io_wgt_wr_0_bits_data_12_8(),
    .io_wgt_wr_0_bits_data_12_9(),
    .io_wgt_wr_0_bits_data_12_10(),
    .io_wgt_wr_0_bits_data_12_11(),
    .io_wgt_wr_0_bits_data_12_12(),
    .io_wgt_wr_0_bits_data_12_13(),
    .io_wgt_wr_0_bits_data_12_14(),
    .io_wgt_wr_0_bits_data_12_15(),
    .io_wgt_wr_0_bits_data_13_0(),
    .io_wgt_wr_0_bits_data_13_1(),
    .io_wgt_wr_0_bits_data_13_2(),
    .io_wgt_wr_0_bits_data_13_3(),
    .io_wgt_wr_0_bits_data_13_4(),
    .io_wgt_wr_0_bits_data_13_5(),
    .io_wgt_wr_0_bits_data_13_6(),
    .io_wgt_wr_0_bits_data_13_7(),
    .io_wgt_wr_0_bits_data_13_8(),
    .io_wgt_wr_0_bits_data_13_9(),
    .io_wgt_wr_0_bits_data_13_10(),
    .io_wgt_wr_0_bits_data_13_11(),
    .io_wgt_wr_0_bits_data_13_12(),
    .io_wgt_wr_0_bits_data_13_13(),
    .io_wgt_wr_0_bits_data_13_14(),
    .io_wgt_wr_0_bits_data_13_15(),
    .io_wgt_wr_0_bits_data_14_0(),
    .io_wgt_wr_0_bits_data_14_1(),
    .io_wgt_wr_0_bits_data_14_2(),
    .io_wgt_wr_0_bits_data_14_3(),
    .io_wgt_wr_0_bits_data_14_4(),
    .io_wgt_wr_0_bits_data_14_5(),
    .io_wgt_wr_0_bits_data_14_6(),
    .io_wgt_wr_0_bits_data_14_7(),
    .io_wgt_wr_0_bits_data_14_8(),
    .io_wgt_wr_0_bits_data_14_9(),
    .io_wgt_wr_0_bits_data_14_10(),
    .io_wgt_wr_0_bits_data_14_11(),
    .io_wgt_wr_0_bits_data_14_12(),
    .io_wgt_wr_0_bits_data_14_13(),
    .io_wgt_wr_0_bits_data_14_14(),
    .io_wgt_wr_0_bits_data_14_15(),
    .io_wgt_wr_0_bits_data_15_0(),
    .io_wgt_wr_0_bits_data_15_1(),
    .io_wgt_wr_0_bits_data_15_2(),
    .io_wgt_wr_0_bits_data_15_3(),
    .io_wgt_wr_0_bits_data_15_4(),
    .io_wgt_wr_0_bits_data_15_5(),
    .io_wgt_wr_0_bits_data_15_6(),
    .io_wgt_wr_0_bits_data_15_7(),
    .io_wgt_wr_0_bits_data_15_8(),
    .io_wgt_wr_0_bits_data_15_9(),
    .io_wgt_wr_0_bits_data_15_10(),
    .io_wgt_wr_0_bits_data_15_11(),
    .io_wgt_wr_0_bits_data_15_12(),
    .io_wgt_wr_0_bits_data_15_13(),
    .io_wgt_wr_0_bits_data_15_14(),
    .io_wgt_wr_0_bits_data_15_15(),
    .io_acc_rd_0_idx_valid(g_acc_iv), .io_acc_rd_0_idx_bits(g_acc_ib),
    .io_acc_rd_0_data_valid(g_acc_dv),
    .io_acc_rd_0_data_bits_0_0(acc_d[0]),
    .io_acc_rd_0_data_bits_0_1(acc_d[1]),
    .io_acc_rd_0_data_bits_0_2(acc_d[2]),
    .io_acc_rd_0_data_bits_0_3(acc_d[3]),
    .io_acc_rd_0_data_bits_0_4(acc_d[4]),
    .io_acc_rd_0_data_bits_0_5(acc_d[5]),
    .io_acc_rd_0_data_bits_0_6(acc_d[6]),
    .io_acc_rd_0_data_bits_0_7(acc_d[7]),
    .io_acc_rd_0_data_bits_0_8(acc_d[8]),
    .io_acc_rd_0_data_bits_0_9(acc_d[9]),
    .io_acc_rd_0_data_bits_0_10(acc_d[10]),
    .io_acc_rd_0_data_bits_0_11(acc_d[11]),
    .io_acc_rd_0_data_bits_0_12(acc_d[12]),
    .io_acc_rd_0_data_bits_0_13(acc_d[13]),
    .io_acc_rd_0_data_bits_0_14(acc_d[14]),
    .io_acc_rd_0_data_bits_0_15(acc_d[15]),
    .io_acc_wr_0_valid(g_acc_wv), .io_acc_wr_0_bits_idx(g_acc_wi),
    .io_acc_wr_0_bits_data_0_0(g_accwd[0]),
    .io_acc_wr_0_bits_data_0_1(g_accwd[1]),
    .io_acc_wr_0_bits_data_0_2(g_accwd[2]),
    .io_acc_wr_0_bits_data_0_3(g_accwd[3]),
    .io_acc_wr_0_bits_data_0_4(g_accwd[4]),
    .io_acc_wr_0_bits_data_0_5(g_accwd[5]),
    .io_acc_wr_0_bits_data_0_6(g_accwd[6]),
    .io_acc_wr_0_bits_data_0_7(g_accwd[7]),
    .io_acc_wr_0_bits_data_0_8(g_accwd[8]),
    .io_acc_wr_0_bits_data_0_9(g_accwd[9]),
    .io_acc_wr_0_bits_data_0_10(g_accwd[10]),
    .io_acc_wr_0_bits_data_0_11(g_accwd[11]),
    .io_acc_wr_0_bits_data_0_12(g_accwd[12]),
    .io_acc_wr_0_bits_data_0_13(g_accwd[13]),
    .io_acc_wr_0_bits_data_0_14(g_accwd[14]),
    .io_acc_wr_0_bits_data_0_15(g_accwd[15]),
    .io_out_rd_0_idx_valid(), .io_out_rd_0_idx_bits(),
    .io_out_rd_0_data_valid(1'b0),
    .io_out_rd_0_data_bits_0_0(8'sd0),
    .io_out_rd_0_data_bits_0_1(8'sd0),
    .io_out_rd_0_data_bits_0_2(8'sd0),
    .io_out_rd_0_data_bits_0_3(8'sd0),
    .io_out_rd_0_data_bits_0_4(8'sd0),
    .io_out_rd_0_data_bits_0_5(8'sd0),
    .io_out_rd_0_data_bits_0_6(8'sd0),
    .io_out_rd_0_data_bits_0_7(8'sd0),
    .io_out_rd_0_data_bits_0_8(8'sd0),
    .io_out_rd_0_data_bits_0_9(8'sd0),
    .io_out_rd_0_data_bits_0_10(8'sd0),
    .io_out_rd_0_data_bits_0_11(8'sd0),
    .io_out_rd_0_data_bits_0_12(8'sd0),
    .io_out_rd_0_data_bits_0_13(8'sd0),
    .io_out_rd_0_data_bits_0_14(8'sd0),
    .io_out_rd_0_data_bits_0_15(8'sd0),
    .io_out_wr_0_valid(), .io_out_wr_0_bits_idx(),
    .io_out_wr_0_bits_data_0_0(),
    .io_out_wr_0_bits_data_0_1(),
    .io_out_wr_0_bits_data_0_2(),
    .io_out_wr_0_bits_data_0_3(),
    .io_out_wr_0_bits_data_0_4(),
    .io_out_wr_0_bits_data_0_5(),
    .io_out_wr_0_bits_data_0_6(),
    .io_out_wr_0_bits_data_0_7(),
    .io_out_wr_0_bits_data_0_8(),
    .io_out_wr_0_bits_data_0_9(),
    .io_out_wr_0_bits_data_0_10(),
    .io_out_wr_0_bits_data_0_11(),
    .io_out_wr_0_bits_data_0_12(),
    .io_out_wr_0_bits_data_0_13(),
    .io_out_wr_0_bits_data_0_14(),
    .io_out_wr_0_bits_data_0_15(),
    .io_state(), .io_inflight()
  );

  TensorAlu alu (
    .clock(clk), .reset(rst), .io_start(a_start), .io_done(a_done),
    .io_dec_alu_imm(alu_imm), .io_dec_alu_use_imm(1'b1), .io_dec_alu_op(alu_op),
    .io_dec_src_1(11'd0), .io_dec_src_0(11'd64),
    .io_dec_dst_1(11'd0), .io_dec_dst_0(11'd64),
    .io_dec_empty_0(1'b0),
    .io_dec_lp_1(14'd1), .io_dec_lp_0(14'd8),
    .io_dec_uop_end(14'd64), .io_dec_uop_begin(13'd0),
    .io_dec_reset(1'b0), .io_dec_push_next(1'b0), .io_dec_push_prev(1'b0),
    .io_dec_pop_next(1'b0), .io_dec_pop_prev(1'b0), .io_dec_op(3'd0),
    .io_uop_idx_valid(a_uop_iv), .io_uop_idx_bits(a_uop_ib),
    .io_uop_data_valid(a_uop_dv),
    .io_uop_data_bits_u2(u2), .io_uop_data_bits_u1(u1), .io_uop_data_bits_u0(u0),
    .io_acc_rd_0_idx_valid(a_acc_iv), .io_acc_rd_0_idx_bits(a_acc_ib),
    .io_acc_rd_0_data_valid(a_acc_dv),
    .io_acc_rd_0_data_bits_0_0(acc_d[0]),
    .io_acc_rd_0_data_bits_0_1(acc_d[1]),
    .io_acc_rd_0_data_bits_0_2(acc_d[2]),
    .io_acc_rd_0_data_bits_0_3(acc_d[3]),
    .io_acc_rd_0_data_bits_0_4(acc_d[4]),
    .io_acc_rd_0_data_bits_0_5(acc_d[5]),
    .io_acc_rd_0_data_bits_0_6(acc_d[6]),
    .io_acc_rd_0_data_bits_0_7(acc_d[7]),
    .io_acc_rd_0_data_bits_0_8(acc_d[8]),
    .io_acc_rd_0_data_bits_0_9(acc_d[9]),
    .io_acc_rd_0_data_bits_0_10(acc_d[10]),
    .io_acc_rd_0_data_bits_0_11(acc_d[11]),
    .io_acc_rd_0_data_bits_0_12(acc_d[12]),
    .io_acc_rd_0_data_bits_0_13(acc_d[13]),
    .io_acc_rd_0_data_bits_0_14(acc_d[14]),
    .io_acc_rd_0_data_bits_0_15(acc_d[15]),
    .io_acc_wr_0_valid(a_acc_wv), .io_acc_wr_0_bits_idx(a_acc_wi),
    .io_acc_wr_0_bits_data_0_0(a_accwd[0]),
    .io_acc_wr_0_bits_data_0_1(a_accwd[1]),
    .io_acc_wr_0_bits_data_0_2(a_accwd[2]),
    .io_acc_wr_0_bits_data_0_3(a_accwd[3]),
    .io_acc_wr_0_bits_data_0_4(a_accwd[4]),
    .io_acc_wr_0_bits_data_0_5(a_accwd[5]),
    .io_acc_wr_0_bits_data_0_6(a_accwd[6]),
    .io_acc_wr_0_bits_data_0_7(a_accwd[7]),
    .io_acc_wr_0_bits_data_0_8(a_accwd[8]),
    .io_acc_wr_0_bits_data_0_9(a_accwd[9]),
    .io_acc_wr_0_bits_data_0_10(a_accwd[10]),
    .io_acc_wr_0_bits_data_0_11(a_accwd[11]),
    .io_acc_wr_0_bits_data_0_12(a_accwd[12]),
    .io_acc_wr_0_bits_data_0_13(a_accwd[13]),
    .io_acc_wr_0_bits_data_0_14(a_accwd[14]),
    .io_acc_wr_0_bits_data_0_15(a_accwd[15]),
    .io_out_rd_0_idx_valid(), .io_out_rd_0_idx_bits(),
    .io_out_rd_0_data_valid(1'b0),
    .io_out_rd_0_data_bits_0_0(8'sd0),
    .io_out_rd_0_data_bits_0_1(8'sd0),
    .io_out_rd_0_data_bits_0_2(8'sd0),
    .io_out_rd_0_data_bits_0_3(8'sd0),
    .io_out_rd_0_data_bits_0_4(8'sd0),
    .io_out_rd_0_data_bits_0_5(8'sd0),
    .io_out_rd_0_data_bits_0_6(8'sd0),
    .io_out_rd_0_data_bits_0_7(8'sd0),
    .io_out_rd_0_data_bits_0_8(8'sd0),
    .io_out_rd_0_data_bits_0_9(8'sd0),
    .io_out_rd_0_data_bits_0_10(8'sd0),
    .io_out_rd_0_data_bits_0_11(8'sd0),
    .io_out_rd_0_data_bits_0_12(8'sd0),
    .io_out_rd_0_data_bits_0_13(8'sd0),
    .io_out_rd_0_data_bits_0_14(8'sd0),
    .io_out_rd_0_data_bits_0_15(8'sd0),
    .io_out_wr_0_valid(), .io_out_wr_0_bits_idx(),
    .io_out_wr_0_bits_data_0_0(a_outwd[0]),
    .io_out_wr_0_bits_data_0_1(a_outwd[1]),
    .io_out_wr_0_bits_data_0_2(a_outwd[2]),
    .io_out_wr_0_bits_data_0_3(a_outwd[3]),
    .io_out_wr_0_bits_data_0_4(a_outwd[4]),
    .io_out_wr_0_bits_data_0_5(a_outwd[5]),
    .io_out_wr_0_bits_data_0_6(a_outwd[6]),
    .io_out_wr_0_bits_data_0_7(a_outwd[7]),
    .io_out_wr_0_bits_data_0_8(a_outwd[8]),
    .io_out_wr_0_bits_data_0_9(a_outwd[9]),
    .io_out_wr_0_bits_data_0_10(a_outwd[10]),
    .io_out_wr_0_bits_data_0_11(a_outwd[11]),
    .io_out_wr_0_bits_data_0_12(a_outwd[12]),
    .io_out_wr_0_bits_data_0_13(a_outwd[13]),
    .io_out_wr_0_bits_data_0_14(a_outwd[14]),
    .io_out_wr_0_bits_data_0_15(a_outwd[15])
  );

  always @(posedge clk) begin
    g_uop_dv <= g_uop_iv;  a_uop_dv <= a_uop_iv;
    if (uop_iv) begin
      if (sel) begin
        u0 <= uop_ib[10:0]; u1 <= uop_ib[10:0]; u2 <= 10'd0;
      end else begin
        u0 <= uopm[uop_ib][10:0]; u1 <= uopm[uop_ib][21:11]; u2 <= uopm[uop_ib][31:22];
      end
    end
    inp_dv <= g_inp_iv;
    if (g_inp_iv) begin
        inp_d[0] <= inpm[inp_base + inp_ib][0];
        inp_d[1] <= inpm[inp_base + inp_ib][1];
        inp_d[2] <= inpm[inp_base + inp_ib][2];
        inp_d[3] <= inpm[inp_base + inp_ib][3];
        inp_d[4] <= inpm[inp_base + inp_ib][4];
        inp_d[5] <= inpm[inp_base + inp_ib][5];
        inp_d[6] <= inpm[inp_base + inp_ib][6];
        inp_d[7] <= inpm[inp_base + inp_ib][7];
        inp_d[8] <= inpm[inp_base + inp_ib][8];
        inp_d[9] <= inpm[inp_base + inp_ib][9];
        inp_d[10] <= inpm[inp_base + inp_ib][10];
        inp_d[11] <= inpm[inp_base + inp_ib][11];
        inp_d[12] <= inpm[inp_base + inp_ib][12];
        inp_d[13] <= inpm[inp_base + inp_ib][13];
        inp_d[14] <= inpm[inp_base + inp_ib][14];
        inp_d[15] <= inpm[inp_base + inp_ib][15];
    end
    wgt_dv <= g_wgt_iv;
    if (g_wgt_iv) begin
        wgt_d[0][0] <= wgtm[wgt_base + wgt_ib][0][0];
        wgt_d[0][1] <= wgtm[wgt_base + wgt_ib][0][1];
        wgt_d[0][2] <= wgtm[wgt_base + wgt_ib][0][2];
        wgt_d[0][3] <= wgtm[wgt_base + wgt_ib][0][3];
        wgt_d[0][4] <= wgtm[wgt_base + wgt_ib][0][4];
        wgt_d[0][5] <= wgtm[wgt_base + wgt_ib][0][5];
        wgt_d[0][6] <= wgtm[wgt_base + wgt_ib][0][6];
        wgt_d[0][7] <= wgtm[wgt_base + wgt_ib][0][7];
        wgt_d[0][8] <= wgtm[wgt_base + wgt_ib][0][8];
        wgt_d[0][9] <= wgtm[wgt_base + wgt_ib][0][9];
        wgt_d[0][10] <= wgtm[wgt_base + wgt_ib][0][10];
        wgt_d[0][11] <= wgtm[wgt_base + wgt_ib][0][11];
        wgt_d[0][12] <= wgtm[wgt_base + wgt_ib][0][12];
        wgt_d[0][13] <= wgtm[wgt_base + wgt_ib][0][13];
        wgt_d[0][14] <= wgtm[wgt_base + wgt_ib][0][14];
        wgt_d[0][15] <= wgtm[wgt_base + wgt_ib][0][15];
        wgt_d[1][0] <= wgtm[wgt_base + wgt_ib][1][0];
        wgt_d[1][1] <= wgtm[wgt_base + wgt_ib][1][1];
        wgt_d[1][2] <= wgtm[wgt_base + wgt_ib][1][2];
        wgt_d[1][3] <= wgtm[wgt_base + wgt_ib][1][3];
        wgt_d[1][4] <= wgtm[wgt_base + wgt_ib][1][4];
        wgt_d[1][5] <= wgtm[wgt_base + wgt_ib][1][5];
        wgt_d[1][6] <= wgtm[wgt_base + wgt_ib][1][6];
        wgt_d[1][7] <= wgtm[wgt_base + wgt_ib][1][7];
        wgt_d[1][8] <= wgtm[wgt_base + wgt_ib][1][8];
        wgt_d[1][9] <= wgtm[wgt_base + wgt_ib][1][9];
        wgt_d[1][10] <= wgtm[wgt_base + wgt_ib][1][10];
        wgt_d[1][11] <= wgtm[wgt_base + wgt_ib][1][11];
        wgt_d[1][12] <= wgtm[wgt_base + wgt_ib][1][12];
        wgt_d[1][13] <= wgtm[wgt_base + wgt_ib][1][13];
        wgt_d[1][14] <= wgtm[wgt_base + wgt_ib][1][14];
        wgt_d[1][15] <= wgtm[wgt_base + wgt_ib][1][15];
        wgt_d[2][0] <= wgtm[wgt_base + wgt_ib][2][0];
        wgt_d[2][1] <= wgtm[wgt_base + wgt_ib][2][1];
        wgt_d[2][2] <= wgtm[wgt_base + wgt_ib][2][2];
        wgt_d[2][3] <= wgtm[wgt_base + wgt_ib][2][3];
        wgt_d[2][4] <= wgtm[wgt_base + wgt_ib][2][4];
        wgt_d[2][5] <= wgtm[wgt_base + wgt_ib][2][5];
        wgt_d[2][6] <= wgtm[wgt_base + wgt_ib][2][6];
        wgt_d[2][7] <= wgtm[wgt_base + wgt_ib][2][7];
        wgt_d[2][8] <= wgtm[wgt_base + wgt_ib][2][8];
        wgt_d[2][9] <= wgtm[wgt_base + wgt_ib][2][9];
        wgt_d[2][10] <= wgtm[wgt_base + wgt_ib][2][10];
        wgt_d[2][11] <= wgtm[wgt_base + wgt_ib][2][11];
        wgt_d[2][12] <= wgtm[wgt_base + wgt_ib][2][12];
        wgt_d[2][13] <= wgtm[wgt_base + wgt_ib][2][13];
        wgt_d[2][14] <= wgtm[wgt_base + wgt_ib][2][14];
        wgt_d[2][15] <= wgtm[wgt_base + wgt_ib][2][15];
        wgt_d[3][0] <= wgtm[wgt_base + wgt_ib][3][0];
        wgt_d[3][1] <= wgtm[wgt_base + wgt_ib][3][1];
        wgt_d[3][2] <= wgtm[wgt_base + wgt_ib][3][2];
        wgt_d[3][3] <= wgtm[wgt_base + wgt_ib][3][3];
        wgt_d[3][4] <= wgtm[wgt_base + wgt_ib][3][4];
        wgt_d[3][5] <= wgtm[wgt_base + wgt_ib][3][5];
        wgt_d[3][6] <= wgtm[wgt_base + wgt_ib][3][6];
        wgt_d[3][7] <= wgtm[wgt_base + wgt_ib][3][7];
        wgt_d[3][8] <= wgtm[wgt_base + wgt_ib][3][8];
        wgt_d[3][9] <= wgtm[wgt_base + wgt_ib][3][9];
        wgt_d[3][10] <= wgtm[wgt_base + wgt_ib][3][10];
        wgt_d[3][11] <= wgtm[wgt_base + wgt_ib][3][11];
        wgt_d[3][12] <= wgtm[wgt_base + wgt_ib][3][12];
        wgt_d[3][13] <= wgtm[wgt_base + wgt_ib][3][13];
        wgt_d[3][14] <= wgtm[wgt_base + wgt_ib][3][14];
        wgt_d[3][15] <= wgtm[wgt_base + wgt_ib][3][15];
        wgt_d[4][0] <= wgtm[wgt_base + wgt_ib][4][0];
        wgt_d[4][1] <= wgtm[wgt_base + wgt_ib][4][1];
        wgt_d[4][2] <= wgtm[wgt_base + wgt_ib][4][2];
        wgt_d[4][3] <= wgtm[wgt_base + wgt_ib][4][3];
        wgt_d[4][4] <= wgtm[wgt_base + wgt_ib][4][4];
        wgt_d[4][5] <= wgtm[wgt_base + wgt_ib][4][5];
        wgt_d[4][6] <= wgtm[wgt_base + wgt_ib][4][6];
        wgt_d[4][7] <= wgtm[wgt_base + wgt_ib][4][7];
        wgt_d[4][8] <= wgtm[wgt_base + wgt_ib][4][8];
        wgt_d[4][9] <= wgtm[wgt_base + wgt_ib][4][9];
        wgt_d[4][10] <= wgtm[wgt_base + wgt_ib][4][10];
        wgt_d[4][11] <= wgtm[wgt_base + wgt_ib][4][11];
        wgt_d[4][12] <= wgtm[wgt_base + wgt_ib][4][12];
        wgt_d[4][13] <= wgtm[wgt_base + wgt_ib][4][13];
        wgt_d[4][14] <= wgtm[wgt_base + wgt_ib][4][14];
        wgt_d[4][15] <= wgtm[wgt_base + wgt_ib][4][15];
        wgt_d[5][0] <= wgtm[wgt_base + wgt_ib][5][0];
        wgt_d[5][1] <= wgtm[wgt_base + wgt_ib][5][1];
        wgt_d[5][2] <= wgtm[wgt_base + wgt_ib][5][2];
        wgt_d[5][3] <= wgtm[wgt_base + wgt_ib][5][3];
        wgt_d[5][4] <= wgtm[wgt_base + wgt_ib][5][4];
        wgt_d[5][5] <= wgtm[wgt_base + wgt_ib][5][5];
        wgt_d[5][6] <= wgtm[wgt_base + wgt_ib][5][6];
        wgt_d[5][7] <= wgtm[wgt_base + wgt_ib][5][7];
        wgt_d[5][8] <= wgtm[wgt_base + wgt_ib][5][8];
        wgt_d[5][9] <= wgtm[wgt_base + wgt_ib][5][9];
        wgt_d[5][10] <= wgtm[wgt_base + wgt_ib][5][10];
        wgt_d[5][11] <= wgtm[wgt_base + wgt_ib][5][11];
        wgt_d[5][12] <= wgtm[wgt_base + wgt_ib][5][12];
        wgt_d[5][13] <= wgtm[wgt_base + wgt_ib][5][13];
        wgt_d[5][14] <= wgtm[wgt_base + wgt_ib][5][14];
        wgt_d[5][15] <= wgtm[wgt_base + wgt_ib][5][15];
        wgt_d[6][0] <= wgtm[wgt_base + wgt_ib][6][0];
        wgt_d[6][1] <= wgtm[wgt_base + wgt_ib][6][1];
        wgt_d[6][2] <= wgtm[wgt_base + wgt_ib][6][2];
        wgt_d[6][3] <= wgtm[wgt_base + wgt_ib][6][3];
        wgt_d[6][4] <= wgtm[wgt_base + wgt_ib][6][4];
        wgt_d[6][5] <= wgtm[wgt_base + wgt_ib][6][5];
        wgt_d[6][6] <= wgtm[wgt_base + wgt_ib][6][6];
        wgt_d[6][7] <= wgtm[wgt_base + wgt_ib][6][7];
        wgt_d[6][8] <= wgtm[wgt_base + wgt_ib][6][8];
        wgt_d[6][9] <= wgtm[wgt_base + wgt_ib][6][9];
        wgt_d[6][10] <= wgtm[wgt_base + wgt_ib][6][10];
        wgt_d[6][11] <= wgtm[wgt_base + wgt_ib][6][11];
        wgt_d[6][12] <= wgtm[wgt_base + wgt_ib][6][12];
        wgt_d[6][13] <= wgtm[wgt_base + wgt_ib][6][13];
        wgt_d[6][14] <= wgtm[wgt_base + wgt_ib][6][14];
        wgt_d[6][15] <= wgtm[wgt_base + wgt_ib][6][15];
        wgt_d[7][0] <= wgtm[wgt_base + wgt_ib][7][0];
        wgt_d[7][1] <= wgtm[wgt_base + wgt_ib][7][1];
        wgt_d[7][2] <= wgtm[wgt_base + wgt_ib][7][2];
        wgt_d[7][3] <= wgtm[wgt_base + wgt_ib][7][3];
        wgt_d[7][4] <= wgtm[wgt_base + wgt_ib][7][4];
        wgt_d[7][5] <= wgtm[wgt_base + wgt_ib][7][5];
        wgt_d[7][6] <= wgtm[wgt_base + wgt_ib][7][6];
        wgt_d[7][7] <= wgtm[wgt_base + wgt_ib][7][7];
        wgt_d[7][8] <= wgtm[wgt_base + wgt_ib][7][8];
        wgt_d[7][9] <= wgtm[wgt_base + wgt_ib][7][9];
        wgt_d[7][10] <= wgtm[wgt_base + wgt_ib][7][10];
        wgt_d[7][11] <= wgtm[wgt_base + wgt_ib][7][11];
        wgt_d[7][12] <= wgtm[wgt_base + wgt_ib][7][12];
        wgt_d[7][13] <= wgtm[wgt_base + wgt_ib][7][13];
        wgt_d[7][14] <= wgtm[wgt_base + wgt_ib][7][14];
        wgt_d[7][15] <= wgtm[wgt_base + wgt_ib][7][15];
        wgt_d[8][0] <= wgtm[wgt_base + wgt_ib][8][0];
        wgt_d[8][1] <= wgtm[wgt_base + wgt_ib][8][1];
        wgt_d[8][2] <= wgtm[wgt_base + wgt_ib][8][2];
        wgt_d[8][3] <= wgtm[wgt_base + wgt_ib][8][3];
        wgt_d[8][4] <= wgtm[wgt_base + wgt_ib][8][4];
        wgt_d[8][5] <= wgtm[wgt_base + wgt_ib][8][5];
        wgt_d[8][6] <= wgtm[wgt_base + wgt_ib][8][6];
        wgt_d[8][7] <= wgtm[wgt_base + wgt_ib][8][7];
        wgt_d[8][8] <= wgtm[wgt_base + wgt_ib][8][8];
        wgt_d[8][9] <= wgtm[wgt_base + wgt_ib][8][9];
        wgt_d[8][10] <= wgtm[wgt_base + wgt_ib][8][10];
        wgt_d[8][11] <= wgtm[wgt_base + wgt_ib][8][11];
        wgt_d[8][12] <= wgtm[wgt_base + wgt_ib][8][12];
        wgt_d[8][13] <= wgtm[wgt_base + wgt_ib][8][13];
        wgt_d[8][14] <= wgtm[wgt_base + wgt_ib][8][14];
        wgt_d[8][15] <= wgtm[wgt_base + wgt_ib][8][15];
        wgt_d[9][0] <= wgtm[wgt_base + wgt_ib][9][0];
        wgt_d[9][1] <= wgtm[wgt_base + wgt_ib][9][1];
        wgt_d[9][2] <= wgtm[wgt_base + wgt_ib][9][2];
        wgt_d[9][3] <= wgtm[wgt_base + wgt_ib][9][3];
        wgt_d[9][4] <= wgtm[wgt_base + wgt_ib][9][4];
        wgt_d[9][5] <= wgtm[wgt_base + wgt_ib][9][5];
        wgt_d[9][6] <= wgtm[wgt_base + wgt_ib][9][6];
        wgt_d[9][7] <= wgtm[wgt_base + wgt_ib][9][7];
        wgt_d[9][8] <= wgtm[wgt_base + wgt_ib][9][8];
        wgt_d[9][9] <= wgtm[wgt_base + wgt_ib][9][9];
        wgt_d[9][10] <= wgtm[wgt_base + wgt_ib][9][10];
        wgt_d[9][11] <= wgtm[wgt_base + wgt_ib][9][11];
        wgt_d[9][12] <= wgtm[wgt_base + wgt_ib][9][12];
        wgt_d[9][13] <= wgtm[wgt_base + wgt_ib][9][13];
        wgt_d[9][14] <= wgtm[wgt_base + wgt_ib][9][14];
        wgt_d[9][15] <= wgtm[wgt_base + wgt_ib][9][15];
        wgt_d[10][0] <= wgtm[wgt_base + wgt_ib][10][0];
        wgt_d[10][1] <= wgtm[wgt_base + wgt_ib][10][1];
        wgt_d[10][2] <= wgtm[wgt_base + wgt_ib][10][2];
        wgt_d[10][3] <= wgtm[wgt_base + wgt_ib][10][3];
        wgt_d[10][4] <= wgtm[wgt_base + wgt_ib][10][4];
        wgt_d[10][5] <= wgtm[wgt_base + wgt_ib][10][5];
        wgt_d[10][6] <= wgtm[wgt_base + wgt_ib][10][6];
        wgt_d[10][7] <= wgtm[wgt_base + wgt_ib][10][7];
        wgt_d[10][8] <= wgtm[wgt_base + wgt_ib][10][8];
        wgt_d[10][9] <= wgtm[wgt_base + wgt_ib][10][9];
        wgt_d[10][10] <= wgtm[wgt_base + wgt_ib][10][10];
        wgt_d[10][11] <= wgtm[wgt_base + wgt_ib][10][11];
        wgt_d[10][12] <= wgtm[wgt_base + wgt_ib][10][12];
        wgt_d[10][13] <= wgtm[wgt_base + wgt_ib][10][13];
        wgt_d[10][14] <= wgtm[wgt_base + wgt_ib][10][14];
        wgt_d[10][15] <= wgtm[wgt_base + wgt_ib][10][15];
        wgt_d[11][0] <= wgtm[wgt_base + wgt_ib][11][0];
        wgt_d[11][1] <= wgtm[wgt_base + wgt_ib][11][1];
        wgt_d[11][2] <= wgtm[wgt_base + wgt_ib][11][2];
        wgt_d[11][3] <= wgtm[wgt_base + wgt_ib][11][3];
        wgt_d[11][4] <= wgtm[wgt_base + wgt_ib][11][4];
        wgt_d[11][5] <= wgtm[wgt_base + wgt_ib][11][5];
        wgt_d[11][6] <= wgtm[wgt_base + wgt_ib][11][6];
        wgt_d[11][7] <= wgtm[wgt_base + wgt_ib][11][7];
        wgt_d[11][8] <= wgtm[wgt_base + wgt_ib][11][8];
        wgt_d[11][9] <= wgtm[wgt_base + wgt_ib][11][9];
        wgt_d[11][10] <= wgtm[wgt_base + wgt_ib][11][10];
        wgt_d[11][11] <= wgtm[wgt_base + wgt_ib][11][11];
        wgt_d[11][12] <= wgtm[wgt_base + wgt_ib][11][12];
        wgt_d[11][13] <= wgtm[wgt_base + wgt_ib][11][13];
        wgt_d[11][14] <= wgtm[wgt_base + wgt_ib][11][14];
        wgt_d[11][15] <= wgtm[wgt_base + wgt_ib][11][15];
        wgt_d[12][0] <= wgtm[wgt_base + wgt_ib][12][0];
        wgt_d[12][1] <= wgtm[wgt_base + wgt_ib][12][1];
        wgt_d[12][2] <= wgtm[wgt_base + wgt_ib][12][2];
        wgt_d[12][3] <= wgtm[wgt_base + wgt_ib][12][3];
        wgt_d[12][4] <= wgtm[wgt_base + wgt_ib][12][4];
        wgt_d[12][5] <= wgtm[wgt_base + wgt_ib][12][5];
        wgt_d[12][6] <= wgtm[wgt_base + wgt_ib][12][6];
        wgt_d[12][7] <= wgtm[wgt_base + wgt_ib][12][7];
        wgt_d[12][8] <= wgtm[wgt_base + wgt_ib][12][8];
        wgt_d[12][9] <= wgtm[wgt_base + wgt_ib][12][9];
        wgt_d[12][10] <= wgtm[wgt_base + wgt_ib][12][10];
        wgt_d[12][11] <= wgtm[wgt_base + wgt_ib][12][11];
        wgt_d[12][12] <= wgtm[wgt_base + wgt_ib][12][12];
        wgt_d[12][13] <= wgtm[wgt_base + wgt_ib][12][13];
        wgt_d[12][14] <= wgtm[wgt_base + wgt_ib][12][14];
        wgt_d[12][15] <= wgtm[wgt_base + wgt_ib][12][15];
        wgt_d[13][0] <= wgtm[wgt_base + wgt_ib][13][0];
        wgt_d[13][1] <= wgtm[wgt_base + wgt_ib][13][1];
        wgt_d[13][2] <= wgtm[wgt_base + wgt_ib][13][2];
        wgt_d[13][3] <= wgtm[wgt_base + wgt_ib][13][3];
        wgt_d[13][4] <= wgtm[wgt_base + wgt_ib][13][4];
        wgt_d[13][5] <= wgtm[wgt_base + wgt_ib][13][5];
        wgt_d[13][6] <= wgtm[wgt_base + wgt_ib][13][6];
        wgt_d[13][7] <= wgtm[wgt_base + wgt_ib][13][7];
        wgt_d[13][8] <= wgtm[wgt_base + wgt_ib][13][8];
        wgt_d[13][9] <= wgtm[wgt_base + wgt_ib][13][9];
        wgt_d[13][10] <= wgtm[wgt_base + wgt_ib][13][10];
        wgt_d[13][11] <= wgtm[wgt_base + wgt_ib][13][11];
        wgt_d[13][12] <= wgtm[wgt_base + wgt_ib][13][12];
        wgt_d[13][13] <= wgtm[wgt_base + wgt_ib][13][13];
        wgt_d[13][14] <= wgtm[wgt_base + wgt_ib][13][14];
        wgt_d[13][15] <= wgtm[wgt_base + wgt_ib][13][15];
        wgt_d[14][0] <= wgtm[wgt_base + wgt_ib][14][0];
        wgt_d[14][1] <= wgtm[wgt_base + wgt_ib][14][1];
        wgt_d[14][2] <= wgtm[wgt_base + wgt_ib][14][2];
        wgt_d[14][3] <= wgtm[wgt_base + wgt_ib][14][3];
        wgt_d[14][4] <= wgtm[wgt_base + wgt_ib][14][4];
        wgt_d[14][5] <= wgtm[wgt_base + wgt_ib][14][5];
        wgt_d[14][6] <= wgtm[wgt_base + wgt_ib][14][6];
        wgt_d[14][7] <= wgtm[wgt_base + wgt_ib][14][7];
        wgt_d[14][8] <= wgtm[wgt_base + wgt_ib][14][8];
        wgt_d[14][9] <= wgtm[wgt_base + wgt_ib][14][9];
        wgt_d[14][10] <= wgtm[wgt_base + wgt_ib][14][10];
        wgt_d[14][11] <= wgtm[wgt_base + wgt_ib][14][11];
        wgt_d[14][12] <= wgtm[wgt_base + wgt_ib][14][12];
        wgt_d[14][13] <= wgtm[wgt_base + wgt_ib][14][13];
        wgt_d[14][14] <= wgtm[wgt_base + wgt_ib][14][14];
        wgt_d[14][15] <= wgtm[wgt_base + wgt_ib][14][15];
        wgt_d[15][0] <= wgtm[wgt_base + wgt_ib][15][0];
        wgt_d[15][1] <= wgtm[wgt_base + wgt_ib][15][1];
        wgt_d[15][2] <= wgtm[wgt_base + wgt_ib][15][2];
        wgt_d[15][3] <= wgtm[wgt_base + wgt_ib][15][3];
        wgt_d[15][4] <= wgtm[wgt_base + wgt_ib][15][4];
        wgt_d[15][5] <= wgtm[wgt_base + wgt_ib][15][5];
        wgt_d[15][6] <= wgtm[wgt_base + wgt_ib][15][6];
        wgt_d[15][7] <= wgtm[wgt_base + wgt_ib][15][7];
        wgt_d[15][8] <= wgtm[wgt_base + wgt_ib][15][8];
        wgt_d[15][9] <= wgtm[wgt_base + wgt_ib][15][9];
        wgt_d[15][10] <= wgtm[wgt_base + wgt_ib][15][10];
        wgt_d[15][11] <= wgtm[wgt_base + wgt_ib][15][11];
        wgt_d[15][12] <= wgtm[wgt_base + wgt_ib][15][12];
        wgt_d[15][13] <= wgtm[wgt_base + wgt_ib][15][13];
        wgt_d[15][14] <= wgtm[wgt_base + wgt_ib][15][14];
        wgt_d[15][15] <= wgtm[wgt_base + wgt_ib][15][15];
    end
    g_acc_dv <= g_acc_iv;  a_acc_dv <= a_acc_iv;
    if (acc_iv) begin
        acc_d[0] <= accm[acc_ib][0];
        acc_d[1] <= accm[acc_ib][1];
        acc_d[2] <= accm[acc_ib][2];
        acc_d[3] <= accm[acc_ib][3];
        acc_d[4] <= accm[acc_ib][4];
        acc_d[5] <= accm[acc_ib][5];
        acc_d[6] <= accm[acc_ib][6];
        acc_d[7] <= accm[acc_ib][7];
        acc_d[8] <= accm[acc_ib][8];
        acc_d[9] <= accm[acc_ib][9];
        acc_d[10] <= accm[acc_ib][10];
        acc_d[11] <= accm[acc_ib][11];
        acc_d[12] <= accm[acc_ib][12];
        acc_d[13] <= accm[acc_ib][13];
        acc_d[14] <= accm[acc_ib][14];
        acc_d[15] <= accm[acc_ib][15];
    end
    if (acc_wv) begin
        accm[acc_wi][0] <= acc_wd[0];
        accm[acc_wi][1] <= acc_wd[1];
        accm[acc_wi][2] <= acc_wd[2];
        accm[acc_wi][3] <= acc_wd[3];
        accm[acc_wi][4] <= acc_wd[4];
        accm[acc_wi][5] <= acc_wd[5];
        accm[acc_wi][6] <= acc_wd[6];
        accm[acc_wi][7] <= acc_wd[7];
        accm[acc_wi][8] <= acc_wd[8];
        accm[acc_wi][9] <= acc_wd[9];
        accm[acc_wi][10] <= acc_wd[10];
        accm[acc_wi][11] <= acc_wd[11];
        accm[acc_wi][12] <= acc_wd[12];
        accm[acc_wi][13] <= acc_wd[13];
        accm[acc_wi][14] <= acc_wd[14];
        accm[acc_wi][15] <= acc_wd[15];
      t1 = cyc;
    end
    if (!rst) cyc = cyc + 1;
    if (!rst && uop_iv && t0 < 0) t0 = cyc;
  end

  task automatic run_gemm(input rs, input [13:0] lp1);
    begin
      sel = 0; g_reset = rs; g_lp1 = lp1; ts = cyc; n_instr = n_instr + 1;
      @(posedge clk); g_start = 1; @(posedge clk); g_start = 0;
      for (k = 0; k < 100000000; k = k + 1) begin
        @(posedge clk);
        if (g_done) k = 100000000;
      end
      @(posedge clk);
      if (rs) c_reset = c_reset + (cyc - ts); else c_gemm = c_gemm + (cyc - ts);
    end
  endtask

  task automatic run_alu(input [2:0] op, input [15:0] imm);
    begin
      sel = 1; alu_op = op; alu_imm = imm; ts = cyc; n_instr = n_instr + 1;
      @(posedge clk); a_start = 1; @(posedge clk); a_start = 0;
      for (k = 0; k < 100000000; k = k + 1) begin
        @(posedge clk);
        if (a_done) k = 100000000;
      end
      @(posedge clk);
      c_alu = c_alu + (cyc - ts);
    end
  endtask

  initial begin
    $readmemh("inp.dat", inpm);
    $readmemh("wgt.dat", wgtm);
    $readmemh("want.dat", want);
    $readmemh("uop.dat", uopm);
    repeat (8) @(posedge clk);
    rst = 0; @(posedge clk);

    for (cn = 0; cn < NCH; cn = cn + 1) begin
      run_gemm(1'b1, 14'd1);                   // zero this chunk's accumulator
      for (ck = 0; ck < KCH; ck = ck + 1) begin
        inp_base = ck * KBC * L;
        wgt_base = (cn * KCH + ck) * KBC * NBC;
        run_gemm(1'b0, 14'd32);
      end
      run_alu(3'd3, SHIFT);           // requantise
      run_alu(3'd1, 16'hff80);        // clip at -128
      run_alu(3'd0, 16'd127);         // and at 127
      // The store unit's work, outside the measured units.
      for (i = 0; i < NACC; i = i + 1)
        for (j = 0; j < DIM; j = j + 1) outm[cn * NACC + i][j] = accm[i][j];
    end

    repeat (8) @(posedge clk);
    for (i = 0; i < NOUT; i = i + 1)
      for (j = 0; j < DIM; j = j + 1)
        if (outm[i][j] !== want[i][j]) begin
          if (errs < 8)
            $display("VTALLAMA MISMATCH out[%0d][%0d] got=%0d want=%0d",
                     i, j, $signed(outm[i][j]), $signed(want[i][j]));
          errs = errs + 1;
        end
    $display("VTALLAMA RESULT errs=%0d cycles=%0d gemm=%0d reset=%0d alu=%0d instructions=%0d pages=%0dx%0d",
             errs, (t1 - t0), c_gemm, c_reset, c_alu, n_instr, KCH, NCH);
    $display("VTALLAMA %s", (errs == 0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
