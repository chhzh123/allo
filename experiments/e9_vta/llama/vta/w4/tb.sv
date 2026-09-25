
`timescale 1ns/1ps
module tb;
  localparam DIM = 4, L = 64, SHIFT = 12;
  localparam KB = 512, NB = 32, KBC = 32, NBC = 32;
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
    .io_dec_wgt_1(10'd32), .io_dec_wgt_0(10'd1),
    .io_dec_inp_1(11'd64), .io_dec_inp_0(11'd0),
    .io_dec_acc_1(11'd0), .io_dec_acc_0(11'd64),
    .io_dec_empty_0(1'b0),
    .io_dec_lp_1(g_lp1), .io_dec_lp_0(14'd32),
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
    .io_inp_wr_0_valid(), .io_inp_wr_0_bits_idx(),
    .io_inp_wr_0_bits_data_0_0(),
    .io_inp_wr_0_bits_data_0_1(),
    .io_inp_wr_0_bits_data_0_2(),
    .io_inp_wr_0_bits_data_0_3(),
    .io_wgt_rd_0_idx_valid(g_wgt_iv), .io_wgt_rd_0_idx_bits(wgt_ib),
    .io_wgt_rd_0_data_valid(wgt_dv),
    .io_wgt_rd_0_data_bits_0_0(wgt_d[0][0]),
    .io_wgt_rd_0_data_bits_0_1(wgt_d[0][1]),
    .io_wgt_rd_0_data_bits_0_2(wgt_d[0][2]),
    .io_wgt_rd_0_data_bits_0_3(wgt_d[0][3]),
    .io_wgt_rd_0_data_bits_1_0(wgt_d[1][0]),
    .io_wgt_rd_0_data_bits_1_1(wgt_d[1][1]),
    .io_wgt_rd_0_data_bits_1_2(wgt_d[1][2]),
    .io_wgt_rd_0_data_bits_1_3(wgt_d[1][3]),
    .io_wgt_rd_0_data_bits_2_0(wgt_d[2][0]),
    .io_wgt_rd_0_data_bits_2_1(wgt_d[2][1]),
    .io_wgt_rd_0_data_bits_2_2(wgt_d[2][2]),
    .io_wgt_rd_0_data_bits_2_3(wgt_d[2][3]),
    .io_wgt_rd_0_data_bits_3_0(wgt_d[3][0]),
    .io_wgt_rd_0_data_bits_3_1(wgt_d[3][1]),
    .io_wgt_rd_0_data_bits_3_2(wgt_d[3][2]),
    .io_wgt_rd_0_data_bits_3_3(wgt_d[3][3]),
    .io_wgt_wr_0_valid(), .io_wgt_wr_0_bits_idx(),
    .io_wgt_wr_0_bits_data_0_0(),
    .io_wgt_wr_0_bits_data_0_1(),
    .io_wgt_wr_0_bits_data_0_2(),
    .io_wgt_wr_0_bits_data_0_3(),
    .io_wgt_wr_0_bits_data_1_0(),
    .io_wgt_wr_0_bits_data_1_1(),
    .io_wgt_wr_0_bits_data_1_2(),
    .io_wgt_wr_0_bits_data_1_3(),
    .io_wgt_wr_0_bits_data_2_0(),
    .io_wgt_wr_0_bits_data_2_1(),
    .io_wgt_wr_0_bits_data_2_2(),
    .io_wgt_wr_0_bits_data_2_3(),
    .io_wgt_wr_0_bits_data_3_0(),
    .io_wgt_wr_0_bits_data_3_1(),
    .io_wgt_wr_0_bits_data_3_2(),
    .io_wgt_wr_0_bits_data_3_3(),
    .io_acc_rd_0_idx_valid(g_acc_iv), .io_acc_rd_0_idx_bits(g_acc_ib),
    .io_acc_rd_0_data_valid(g_acc_dv),
    .io_acc_rd_0_data_bits_0_0(acc_d[0]),
    .io_acc_rd_0_data_bits_0_1(acc_d[1]),
    .io_acc_rd_0_data_bits_0_2(acc_d[2]),
    .io_acc_rd_0_data_bits_0_3(acc_d[3]),
    .io_acc_wr_0_valid(g_acc_wv), .io_acc_wr_0_bits_idx(g_acc_wi),
    .io_acc_wr_0_bits_data_0_0(g_accwd[0]),
    .io_acc_wr_0_bits_data_0_1(g_accwd[1]),
    .io_acc_wr_0_bits_data_0_2(g_accwd[2]),
    .io_acc_wr_0_bits_data_0_3(g_accwd[3]),
    .io_out_rd_0_idx_valid(), .io_out_rd_0_idx_bits(),
    .io_out_rd_0_data_valid(1'b0),
    .io_out_rd_0_data_bits_0_0(8'sd0),
    .io_out_rd_0_data_bits_0_1(8'sd0),
    .io_out_rd_0_data_bits_0_2(8'sd0),
    .io_out_rd_0_data_bits_0_3(8'sd0),
    .io_out_wr_0_valid(), .io_out_wr_0_bits_idx(),
    .io_out_wr_0_bits_data_0_0(),
    .io_out_wr_0_bits_data_0_1(),
    .io_out_wr_0_bits_data_0_2(),
    .io_out_wr_0_bits_data_0_3(),
    .io_state(), .io_inflight()
  );

  TensorAlu alu (
    .clock(clk), .reset(rst), .io_start(a_start), .io_done(a_done),
    .io_dec_alu_imm(alu_imm), .io_dec_alu_use_imm(1'b1), .io_dec_alu_op(alu_op),
    .io_dec_src_1(11'd0), .io_dec_src_0(11'd64),
    .io_dec_dst_1(11'd0), .io_dec_dst_0(11'd64),
    .io_dec_empty_0(1'b0),
    .io_dec_lp_1(14'd1), .io_dec_lp_0(14'd32),
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
    .io_acc_wr_0_valid(a_acc_wv), .io_acc_wr_0_bits_idx(a_acc_wi),
    .io_acc_wr_0_bits_data_0_0(a_accwd[0]),
    .io_acc_wr_0_bits_data_0_1(a_accwd[1]),
    .io_acc_wr_0_bits_data_0_2(a_accwd[2]),
    .io_acc_wr_0_bits_data_0_3(a_accwd[3]),
    .io_out_rd_0_idx_valid(), .io_out_rd_0_idx_bits(),
    .io_out_rd_0_data_valid(1'b0),
    .io_out_rd_0_data_bits_0_0(8'sd0),
    .io_out_rd_0_data_bits_0_1(8'sd0),
    .io_out_rd_0_data_bits_0_2(8'sd0),
    .io_out_rd_0_data_bits_0_3(8'sd0),
    .io_out_wr_0_valid(), .io_out_wr_0_bits_idx(),
    .io_out_wr_0_bits_data_0_0(a_outwd[0]),
    .io_out_wr_0_bits_data_0_1(a_outwd[1]),
    .io_out_wr_0_bits_data_0_2(a_outwd[2]),
    .io_out_wr_0_bits_data_0_3(a_outwd[3])
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
    end
    wgt_dv <= g_wgt_iv;
    if (g_wgt_iv) begin
        wgt_d[0][0] <= wgtm[wgt_base + wgt_ib][0][0];
        wgt_d[0][1] <= wgtm[wgt_base + wgt_ib][0][1];
        wgt_d[0][2] <= wgtm[wgt_base + wgt_ib][0][2];
        wgt_d[0][3] <= wgtm[wgt_base + wgt_ib][0][3];
        wgt_d[1][0] <= wgtm[wgt_base + wgt_ib][1][0];
        wgt_d[1][1] <= wgtm[wgt_base + wgt_ib][1][1];
        wgt_d[1][2] <= wgtm[wgt_base + wgt_ib][1][2];
        wgt_d[1][3] <= wgtm[wgt_base + wgt_ib][1][3];
        wgt_d[2][0] <= wgtm[wgt_base + wgt_ib][2][0];
        wgt_d[2][1] <= wgtm[wgt_base + wgt_ib][2][1];
        wgt_d[2][2] <= wgtm[wgt_base + wgt_ib][2][2];
        wgt_d[2][3] <= wgtm[wgt_base + wgt_ib][2][3];
        wgt_d[3][0] <= wgtm[wgt_base + wgt_ib][3][0];
        wgt_d[3][1] <= wgtm[wgt_base + wgt_ib][3][1];
        wgt_d[3][2] <= wgtm[wgt_base + wgt_ib][3][2];
        wgt_d[3][3] <= wgtm[wgt_base + wgt_ib][3][3];
    end
    g_acc_dv <= g_acc_iv;  a_acc_dv <= a_acc_iv;
    if (acc_iv) begin
        acc_d[0] <= accm[acc_ib][0];
        acc_d[1] <= accm[acc_ib][1];
        acc_d[2] <= accm[acc_ib][2];
        acc_d[3] <= accm[acc_ib][3];
    end
    if (acc_wv) begin
        accm[acc_wi][0] <= acc_wd[0];
        accm[acc_wi][1] <= acc_wd[1];
        accm[acc_wi][2] <= acc_wd[2];
        accm[acc_wi][3] <= acc_wd[3];
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
