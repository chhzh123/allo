
`timescale 1ns/1ps
module tb;
  localparam DIM = 8, NT = 16, NACC = 128, SHIFT = 7;
  localparam NINP = 128, NWGT = 16, NUOP = 8, TILED = 0;
  localparam NBM = 8, KBM = 8, KBNB = 1;  // one tile's rows
  reg clk = 1'b0, rst = 1'b1;
  always #1 clk = ~clk;

  reg  [7:0]  inpm [0:NINP-1][0:DIM-1];
  reg  [7:0]  wgtm [0:NWGT-1][0:DIM-1][0:DIM-1];
  reg  [31:0] uopm [0:NUOP-1];
  reg  [10:0] tile = 0;  // the tile a tiled program is on; 0 when batched
  reg  [31:0] accm [0:NACC-1][0:DIM-1];
  reg  [31:0] bias [0:NACC-1][0:DIM-1];
  reg  [31:0] want [0:NACC-1][0:DIM-1];

  reg  signed [7:0]  inp_d [0:DIM-1];
  reg  signed [7:0]  wgt_d [0:DIM-1][0:DIM-1];
  reg  signed [31:0] acc_d [0:DIM-1];
  // Each unit gets its *own* read-valid, derived from its own request.
  // A shared one fires TensorAlu.scala:287 --
  // `assert(acc.rd.data.valid === (valid_r3 || src_valid_r3))` -- the moment
  // the GEMM's accumulator reads are visible to an idle ALU.
  reg  inp_dv = 0, wgt_dv = 0, uop_dv = 0;
  reg  g_acc_dv = 0, a_acc_dv = 0, g_uop_dv = 0, a_uop_dv = 0;
  reg  [10:0] u0 = 0, u1 = 0; reg [9:0] u2 = 0;

  // `sel` picks which unit owns the shared micro-op and accumulator ports.
  // They never run together -- Compute.scala asserts exactly that -- so a
  // mux is the whole arbitration.
  reg sel = 0;                      // 0 = GEMM, 1 = ALU
  reg g_start = 0, a_start = 0;
  reg [2:0] alu_op = 3'd0; reg [15:0] alu_imm = 16'd0;

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

  integer cyc = 0, t0 = -1, t1 = -1, tfirst = -1, tg = -1, i, j, k, t, errs = 0;

  TensorGemm gemm (
    .clock(clk), .reset(rst), .io_start(g_start), .io_done(g_done),
    .io_dec_wgt_1(10'd0), .io_dec_wgt_0(10'd1),
    .io_dec_inp_1(11'd0), .io_dec_inp_0(11'd8),
    .io_dec_acc_1(11'd0), .io_dec_acc_0(11'd8),
    .io_dec_empty_0(1'b0),
    .io_dec_lp_1(14'd1), .io_dec_lp_0(TILED ? 14'd1 : 14'd16),
    .io_dec_uop_end(14'd8), .io_dec_uop_begin(13'd0),
    .io_dec_reset(1'b0), .io_dec_push_next(1'b0), .io_dec_push_prev(1'b0),
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
    .io_inp_wr_0_valid(), .io_inp_wr_0_bits_idx(),
    .io_inp_wr_0_bits_data_0_0(),
    .io_inp_wr_0_bits_data_0_1(),
    .io_inp_wr_0_bits_data_0_2(),
    .io_inp_wr_0_bits_data_0_3(),
    .io_inp_wr_0_bits_data_0_4(),
    .io_inp_wr_0_bits_data_0_5(),
    .io_inp_wr_0_bits_data_0_6(),
    .io_inp_wr_0_bits_data_0_7(),
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
    .io_wgt_rd_0_data_bits_1_0(wgt_d[1][0]),
    .io_wgt_rd_0_data_bits_1_1(wgt_d[1][1]),
    .io_wgt_rd_0_data_bits_1_2(wgt_d[1][2]),
    .io_wgt_rd_0_data_bits_1_3(wgt_d[1][3]),
    .io_wgt_rd_0_data_bits_1_4(wgt_d[1][4]),
    .io_wgt_rd_0_data_bits_1_5(wgt_d[1][5]),
    .io_wgt_rd_0_data_bits_1_6(wgt_d[1][6]),
    .io_wgt_rd_0_data_bits_1_7(wgt_d[1][7]),
    .io_wgt_rd_0_data_bits_2_0(wgt_d[2][0]),
    .io_wgt_rd_0_data_bits_2_1(wgt_d[2][1]),
    .io_wgt_rd_0_data_bits_2_2(wgt_d[2][2]),
    .io_wgt_rd_0_data_bits_2_3(wgt_d[2][3]),
    .io_wgt_rd_0_data_bits_2_4(wgt_d[2][4]),
    .io_wgt_rd_0_data_bits_2_5(wgt_d[2][5]),
    .io_wgt_rd_0_data_bits_2_6(wgt_d[2][6]),
    .io_wgt_rd_0_data_bits_2_7(wgt_d[2][7]),
    .io_wgt_rd_0_data_bits_3_0(wgt_d[3][0]),
    .io_wgt_rd_0_data_bits_3_1(wgt_d[3][1]),
    .io_wgt_rd_0_data_bits_3_2(wgt_d[3][2]),
    .io_wgt_rd_0_data_bits_3_3(wgt_d[3][3]),
    .io_wgt_rd_0_data_bits_3_4(wgt_d[3][4]),
    .io_wgt_rd_0_data_bits_3_5(wgt_d[3][5]),
    .io_wgt_rd_0_data_bits_3_6(wgt_d[3][6]),
    .io_wgt_rd_0_data_bits_3_7(wgt_d[3][7]),
    .io_wgt_rd_0_data_bits_4_0(wgt_d[4][0]),
    .io_wgt_rd_0_data_bits_4_1(wgt_d[4][1]),
    .io_wgt_rd_0_data_bits_4_2(wgt_d[4][2]),
    .io_wgt_rd_0_data_bits_4_3(wgt_d[4][3]),
    .io_wgt_rd_0_data_bits_4_4(wgt_d[4][4]),
    .io_wgt_rd_0_data_bits_4_5(wgt_d[4][5]),
    .io_wgt_rd_0_data_bits_4_6(wgt_d[4][6]),
    .io_wgt_rd_0_data_bits_4_7(wgt_d[4][7]),
    .io_wgt_rd_0_data_bits_5_0(wgt_d[5][0]),
    .io_wgt_rd_0_data_bits_5_1(wgt_d[5][1]),
    .io_wgt_rd_0_data_bits_5_2(wgt_d[5][2]),
    .io_wgt_rd_0_data_bits_5_3(wgt_d[5][3]),
    .io_wgt_rd_0_data_bits_5_4(wgt_d[5][4]),
    .io_wgt_rd_0_data_bits_5_5(wgt_d[5][5]),
    .io_wgt_rd_0_data_bits_5_6(wgt_d[5][6]),
    .io_wgt_rd_0_data_bits_5_7(wgt_d[5][7]),
    .io_wgt_rd_0_data_bits_6_0(wgt_d[6][0]),
    .io_wgt_rd_0_data_bits_6_1(wgt_d[6][1]),
    .io_wgt_rd_0_data_bits_6_2(wgt_d[6][2]),
    .io_wgt_rd_0_data_bits_6_3(wgt_d[6][3]),
    .io_wgt_rd_0_data_bits_6_4(wgt_d[6][4]),
    .io_wgt_rd_0_data_bits_6_5(wgt_d[6][5]),
    .io_wgt_rd_0_data_bits_6_6(wgt_d[6][6]),
    .io_wgt_rd_0_data_bits_6_7(wgt_d[6][7]),
    .io_wgt_rd_0_data_bits_7_0(wgt_d[7][0]),
    .io_wgt_rd_0_data_bits_7_1(wgt_d[7][1]),
    .io_wgt_rd_0_data_bits_7_2(wgt_d[7][2]),
    .io_wgt_rd_0_data_bits_7_3(wgt_d[7][3]),
    .io_wgt_rd_0_data_bits_7_4(wgt_d[7][4]),
    .io_wgt_rd_0_data_bits_7_5(wgt_d[7][5]),
    .io_wgt_rd_0_data_bits_7_6(wgt_d[7][6]),
    .io_wgt_rd_0_data_bits_7_7(wgt_d[7][7]),
    .io_wgt_wr_0_valid(), .io_wgt_wr_0_bits_idx(),
    .io_wgt_wr_0_bits_data_0_0(),
    .io_wgt_wr_0_bits_data_0_1(),
    .io_wgt_wr_0_bits_data_0_2(),
    .io_wgt_wr_0_bits_data_0_3(),
    .io_wgt_wr_0_bits_data_0_4(),
    .io_wgt_wr_0_bits_data_0_5(),
    .io_wgt_wr_0_bits_data_0_6(),
    .io_wgt_wr_0_bits_data_0_7(),
    .io_wgt_wr_0_bits_data_1_0(),
    .io_wgt_wr_0_bits_data_1_1(),
    .io_wgt_wr_0_bits_data_1_2(),
    .io_wgt_wr_0_bits_data_1_3(),
    .io_wgt_wr_0_bits_data_1_4(),
    .io_wgt_wr_0_bits_data_1_5(),
    .io_wgt_wr_0_bits_data_1_6(),
    .io_wgt_wr_0_bits_data_1_7(),
    .io_wgt_wr_0_bits_data_2_0(),
    .io_wgt_wr_0_bits_data_2_1(),
    .io_wgt_wr_0_bits_data_2_2(),
    .io_wgt_wr_0_bits_data_2_3(),
    .io_wgt_wr_0_bits_data_2_4(),
    .io_wgt_wr_0_bits_data_2_5(),
    .io_wgt_wr_0_bits_data_2_6(),
    .io_wgt_wr_0_bits_data_2_7(),
    .io_wgt_wr_0_bits_data_3_0(),
    .io_wgt_wr_0_bits_data_3_1(),
    .io_wgt_wr_0_bits_data_3_2(),
    .io_wgt_wr_0_bits_data_3_3(),
    .io_wgt_wr_0_bits_data_3_4(),
    .io_wgt_wr_0_bits_data_3_5(),
    .io_wgt_wr_0_bits_data_3_6(),
    .io_wgt_wr_0_bits_data_3_7(),
    .io_wgt_wr_0_bits_data_4_0(),
    .io_wgt_wr_0_bits_data_4_1(),
    .io_wgt_wr_0_bits_data_4_2(),
    .io_wgt_wr_0_bits_data_4_3(),
    .io_wgt_wr_0_bits_data_4_4(),
    .io_wgt_wr_0_bits_data_4_5(),
    .io_wgt_wr_0_bits_data_4_6(),
    .io_wgt_wr_0_bits_data_4_7(),
    .io_wgt_wr_0_bits_data_5_0(),
    .io_wgt_wr_0_bits_data_5_1(),
    .io_wgt_wr_0_bits_data_5_2(),
    .io_wgt_wr_0_bits_data_5_3(),
    .io_wgt_wr_0_bits_data_5_4(),
    .io_wgt_wr_0_bits_data_5_5(),
    .io_wgt_wr_0_bits_data_5_6(),
    .io_wgt_wr_0_bits_data_5_7(),
    .io_wgt_wr_0_bits_data_6_0(),
    .io_wgt_wr_0_bits_data_6_1(),
    .io_wgt_wr_0_bits_data_6_2(),
    .io_wgt_wr_0_bits_data_6_3(),
    .io_wgt_wr_0_bits_data_6_4(),
    .io_wgt_wr_0_bits_data_6_5(),
    .io_wgt_wr_0_bits_data_6_6(),
    .io_wgt_wr_0_bits_data_6_7(),
    .io_wgt_wr_0_bits_data_7_0(),
    .io_wgt_wr_0_bits_data_7_1(),
    .io_wgt_wr_0_bits_data_7_2(),
    .io_wgt_wr_0_bits_data_7_3(),
    .io_wgt_wr_0_bits_data_7_4(),
    .io_wgt_wr_0_bits_data_7_5(),
    .io_wgt_wr_0_bits_data_7_6(),
    .io_wgt_wr_0_bits_data_7_7(),
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
    .io_acc_wr_0_valid(g_acc_wv), .io_acc_wr_0_bits_idx(g_acc_wi),
    .io_acc_wr_0_bits_data_0_0(g_accwd[0]),
    .io_acc_wr_0_bits_data_0_1(g_accwd[1]),
    .io_acc_wr_0_bits_data_0_2(g_accwd[2]),
    .io_acc_wr_0_bits_data_0_3(g_accwd[3]),
    .io_acc_wr_0_bits_data_0_4(g_accwd[4]),
    .io_acc_wr_0_bits_data_0_5(g_accwd[5]),
    .io_acc_wr_0_bits_data_0_6(g_accwd[6]),
    .io_acc_wr_0_bits_data_0_7(g_accwd[7]),
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
    .io_out_wr_0_valid(), .io_out_wr_0_bits_idx(),
    .io_out_wr_0_bits_data_0_0(),
    .io_out_wr_0_bits_data_0_1(),
    .io_out_wr_0_bits_data_0_2(),
    .io_out_wr_0_bits_data_0_3(),
    .io_out_wr_0_bits_data_0_4(),
    .io_out_wr_0_bits_data_0_5(),
    .io_out_wr_0_bits_data_0_6(),
    .io_out_wr_0_bits_data_0_7(),
    .io_state(), .io_inflight()
  );

  TensorAlu alu (
    .clock(clk), .reset(rst), .io_start(a_start), .io_done(a_done),
    .io_dec_alu_imm(alu_imm), .io_dec_alu_use_imm(1'b1), .io_dec_alu_op(alu_op),
    .io_dec_src_1(11'd0), .io_dec_src_0(11'd8),
    .io_dec_dst_1(11'd0), .io_dec_dst_0(11'd8),
    .io_dec_empty_0(1'b0),
    .io_dec_lp_1(14'd1), .io_dec_lp_0(TILED ? 14'd1 : 14'd16),
    .io_dec_uop_end(14'd8), .io_dec_uop_begin(13'd0),
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
    .io_acc_wr_0_valid(a_acc_wv), .io_acc_wr_0_bits_idx(a_acc_wi),
    .io_acc_wr_0_bits_data_0_0(a_accwd[0]),
    .io_acc_wr_0_bits_data_0_1(a_accwd[1]),
    .io_acc_wr_0_bits_data_0_2(a_accwd[2]),
    .io_acc_wr_0_bits_data_0_3(a_accwd[3]),
    .io_acc_wr_0_bits_data_0_4(a_accwd[4]),
    .io_acc_wr_0_bits_data_0_5(a_accwd[5]),
    .io_acc_wr_0_bits_data_0_6(a_accwd[6]),
    .io_acc_wr_0_bits_data_0_7(a_accwd[7]),
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
    .io_out_wr_0_valid(), .io_out_wr_0_bits_idx(),
    .io_out_wr_0_bits_data_0_0(a_outwd[0]),
    .io_out_wr_0_bits_data_0_1(a_outwd[1]),
    .io_out_wr_0_bits_data_0_2(a_outwd[2]),
    .io_out_wr_0_bits_data_0_3(a_outwd[3]),
    .io_out_wr_0_bits_data_0_4(a_outwd[4]),
    .io_out_wr_0_bits_data_0_5(a_outwd[5]),
    .io_out_wr_0_bits_data_0_6(a_outwd[6]),
    .io_out_wr_0_bits_data_0_7(a_outwd[7])
  );

  always @(posedge clk) begin
    g_uop_dv <= g_uop_iv;  a_uop_dv <= a_uop_iv;
    if (uop_iv) begin
      if (sel) begin
        u0 <= uop_ib[10:0] + tile * NBM; u1 <= uop_ib[10:0] + tile * NBM; u2 <= 10'd0;
      end else begin
        u0 <= uopm[uop_ib][10:0] + tile * NBM;
        u1 <= uopm[uop_ib][21:11] + tile * KBM;
        u2 <= uopm[uop_ib][31:22] + tile * KBNB;
      end
    end
    inp_dv <= g_inp_iv;
    if (g_inp_iv) begin
        inp_d[0] <= inpm[inp_ib][0];
        inp_d[1] <= inpm[inp_ib][1];
        inp_d[2] <= inpm[inp_ib][2];
        inp_d[3] <= inpm[inp_ib][3];
        inp_d[4] <= inpm[inp_ib][4];
        inp_d[5] <= inpm[inp_ib][5];
        inp_d[6] <= inpm[inp_ib][6];
        inp_d[7] <= inpm[inp_ib][7];
    end
    wgt_dv <= g_wgt_iv;
    if (g_wgt_iv) begin
        wgt_d[0][0] <= wgtm[wgt_ib][0][0];
        wgt_d[0][1] <= wgtm[wgt_ib][0][1];
        wgt_d[0][2] <= wgtm[wgt_ib][0][2];
        wgt_d[0][3] <= wgtm[wgt_ib][0][3];
        wgt_d[0][4] <= wgtm[wgt_ib][0][4];
        wgt_d[0][5] <= wgtm[wgt_ib][0][5];
        wgt_d[0][6] <= wgtm[wgt_ib][0][6];
        wgt_d[0][7] <= wgtm[wgt_ib][0][7];
        wgt_d[1][0] <= wgtm[wgt_ib][1][0];
        wgt_d[1][1] <= wgtm[wgt_ib][1][1];
        wgt_d[1][2] <= wgtm[wgt_ib][1][2];
        wgt_d[1][3] <= wgtm[wgt_ib][1][3];
        wgt_d[1][4] <= wgtm[wgt_ib][1][4];
        wgt_d[1][5] <= wgtm[wgt_ib][1][5];
        wgt_d[1][6] <= wgtm[wgt_ib][1][6];
        wgt_d[1][7] <= wgtm[wgt_ib][1][7];
        wgt_d[2][0] <= wgtm[wgt_ib][2][0];
        wgt_d[2][1] <= wgtm[wgt_ib][2][1];
        wgt_d[2][2] <= wgtm[wgt_ib][2][2];
        wgt_d[2][3] <= wgtm[wgt_ib][2][3];
        wgt_d[2][4] <= wgtm[wgt_ib][2][4];
        wgt_d[2][5] <= wgtm[wgt_ib][2][5];
        wgt_d[2][6] <= wgtm[wgt_ib][2][6];
        wgt_d[2][7] <= wgtm[wgt_ib][2][7];
        wgt_d[3][0] <= wgtm[wgt_ib][3][0];
        wgt_d[3][1] <= wgtm[wgt_ib][3][1];
        wgt_d[3][2] <= wgtm[wgt_ib][3][2];
        wgt_d[3][3] <= wgtm[wgt_ib][3][3];
        wgt_d[3][4] <= wgtm[wgt_ib][3][4];
        wgt_d[3][5] <= wgtm[wgt_ib][3][5];
        wgt_d[3][6] <= wgtm[wgt_ib][3][6];
        wgt_d[3][7] <= wgtm[wgt_ib][3][7];
        wgt_d[4][0] <= wgtm[wgt_ib][4][0];
        wgt_d[4][1] <= wgtm[wgt_ib][4][1];
        wgt_d[4][2] <= wgtm[wgt_ib][4][2];
        wgt_d[4][3] <= wgtm[wgt_ib][4][3];
        wgt_d[4][4] <= wgtm[wgt_ib][4][4];
        wgt_d[4][5] <= wgtm[wgt_ib][4][5];
        wgt_d[4][6] <= wgtm[wgt_ib][4][6];
        wgt_d[4][7] <= wgtm[wgt_ib][4][7];
        wgt_d[5][0] <= wgtm[wgt_ib][5][0];
        wgt_d[5][1] <= wgtm[wgt_ib][5][1];
        wgt_d[5][2] <= wgtm[wgt_ib][5][2];
        wgt_d[5][3] <= wgtm[wgt_ib][5][3];
        wgt_d[5][4] <= wgtm[wgt_ib][5][4];
        wgt_d[5][5] <= wgtm[wgt_ib][5][5];
        wgt_d[5][6] <= wgtm[wgt_ib][5][6];
        wgt_d[5][7] <= wgtm[wgt_ib][5][7];
        wgt_d[6][0] <= wgtm[wgt_ib][6][0];
        wgt_d[6][1] <= wgtm[wgt_ib][6][1];
        wgt_d[6][2] <= wgtm[wgt_ib][6][2];
        wgt_d[6][3] <= wgtm[wgt_ib][6][3];
        wgt_d[6][4] <= wgtm[wgt_ib][6][4];
        wgt_d[6][5] <= wgtm[wgt_ib][6][5];
        wgt_d[6][6] <= wgtm[wgt_ib][6][6];
        wgt_d[6][7] <= wgtm[wgt_ib][6][7];
        wgt_d[7][0] <= wgtm[wgt_ib][7][0];
        wgt_d[7][1] <= wgtm[wgt_ib][7][1];
        wgt_d[7][2] <= wgtm[wgt_ib][7][2];
        wgt_d[7][3] <= wgtm[wgt_ib][7][3];
        wgt_d[7][4] <= wgtm[wgt_ib][7][4];
        wgt_d[7][5] <= wgtm[wgt_ib][7][5];
        wgt_d[7][6] <= wgtm[wgt_ib][7][6];
        wgt_d[7][7] <= wgtm[wgt_ib][7][7];
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
      t1 = cyc;
      // The first tile is done when the clip has written its last row.
      if (sel && alu_op == 3'd0 && acc_wi < NBM) tfirst = cyc;
    end
    if (!rst) cyc = cyc + 1;
    if (!rst && uop_iv && t0 < 0) t0 = cyc;
  end

  task automatic run_alu(input [2:0] op, input [15:0] imm);
    begin
      sel = 1; alu_op = op; alu_imm = imm;
      @(posedge clk); a_start = 1; @(posedge clk); a_start = 0;
      for (k = 0; k < 200000; k = k + 1) begin
        @(posedge clk);
        if (a_done) k = 200000;
      end
      @(posedge clk);
    end
  endtask

  initial begin
    $readmemh("inp.dat", inpm);
    $readmemh("wgt.dat", wgtm);
    $readmemh("bias.dat", bias);
    $readmemh("want.dat", want);
    $readmemh("uop.dat", uopm);
    // The bias is the accumulator's initial value, which is how all three
    // engines get it -- Gemmini and SPMW fold it into the epilogue, VTA
    // preloads the accumulator. Nobody is charged a pass for it.
    for (i = 0; i < NACC; i = i + 1)
      for (j = 0; j < DIM; j = j + 1) accm[i][j] = bias[i][j];
    repeat (8) @(posedge clk);
    rst = 0; @(posedge clk);

    for (t = 0; t < (TILED ? NT : 1); t = t + 1) begin
      tile = t;
      sel = 0;
      g_start = 1; @(posedge clk); g_start = 0;
      for (k = 0; k < 200000; k = k + 1) begin
        @(posedge clk);
        if (g_done) k = 200000;
      end
      @(posedge clk);
      if (tg < 0) tg = cyc;

      run_alu(3'd1, 16'd0);        // ReLU
      run_alu(3'd3, 16'd7); // requantise
      run_alu(3'd0, 16'd127);      // clip to int8
    end

    repeat (8) @(posedge clk);
    for (i = 0; i < NACC; i = i + 1)
      for (j = 0; j < DIM; j = j + 1)
        if (accm[i][j] !== want[i][j]) begin
          if (errs < 8)
            $display("VTAMICRO MISMATCH acc[%0d][%0d] got=%0d want=%0d",
                     i, j, $signed(accm[i][j]), $signed(want[i][j]));
          errs = errs + 1;
        end
    $display("VTAMICRO RESULT tiles=%0d errs=%0d cycles=%0d per_tile=%0d first_tile=%0d first_gemm=%0d",
             NT, errs, (t1 - t0), (t1 - t0) / NT, tfirst - t0 + 1, tg - t0 + 1);
    $display("VTAMICRO %s", (errs == 0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
