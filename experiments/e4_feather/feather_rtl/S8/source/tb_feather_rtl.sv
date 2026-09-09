// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

`timescale 1ns/1ps
// One tile through the original FEATHER RTL (maeri-project/FEATHER, ISCA'24,
// FEATHER_RTL/RTL -- shipped for synthesis, without a testbench): an N x N
// NEST with N weights a PE, a pass-through BIRRD, and the cycle at every
// phase boundary. The bench drives the controller the way the shipped RTL
// behaves (feather_controller.v, feather_pe.v), rather than fixing it:
//
//  - Weight feed: one SRAM row a cycle with pe_sel counting one PE a cycle;
//    the SRAM's read register puts the data a cycle behind pe_sel, so row a
//    is stored by PE (a+1) mod N^2 (row N^3-1 wraps to PE 0 of the next
//    sweep). A PE stores one weight a sweep of N^2 cycles, at index
//    (sweep mod N): N^3 cycles for the array.
//  - The ping/pong select toggles every sweep and a PE computes from one
//    buffer, so its even-index weights sit in one buffer and its odd-index
//    ones in the other. The bench zeroes the odd-index weights, so the dot
//    product the RTL can form is the whole one, and feeds N+1 sweeps so the
//    select ends on the buffer holding the even indices.
//  - A feed state never ends on its own: a write pulse while the read
//    address equals addr_end ends it, the config word saying where to go.
//  - Activation pass: the read register again puts the data a cycle behind
//    the valid, so a pass is [stale, row 0 .. row N-1]. A PE emits the sum of
//    the products between two "last weight index" cycles, provided the input
//    valid is still up three cycles after the second; so the valid runs four
//    cycles past row N-1 (TA = N + 5). The stale cycle is itself a "last
//    index" cycle, so every PE first emits an empty sum: the bus carries N
//    bubble rows, then the N rows of the tile.
//  - The last activation row is written on the same edge that ends the
//    weight feed, so the pass starts the cycle after the feed's last cycle
//    (the chain delays of weights and activations match row by row).
//  - NP passes (tiles) of activations go through back to back with the
//    weights resident: a PE's sum window then spans the last row of one
//    pass and rows 0..N-2 of the next, which is the whole dot product
//    because the weight of index N-1 is odd, hence zero here. The cycles
//    between the last rows of consecutive tiles are the RTL's throughput.
module tb;
  parameter N = 8;
  parameter NP = 1;                  // activation passes (tiles) back to back
  localparam LOG2N = $clog2(N);
  localparam NW = N * N * N;
  localparam WA = $clog2(NW);
  localparam NA = N * NP;            // activation rows in the SRAM
  localparam IA = ($clog2(NA) < 2) ? 2 : $clog2(NA);
  localparam NSTAGE = 2 * LOG2N;
  localparam CMDW = (2 * NSTAGE) * (N / 2);
  localparam OA = 4;
  localparam TW = (N + 1) * N * N;   // weight feed cycles driven: N sweeps by design, one more for the select parity
  localparam TA = NA + 5;            // activation feed cycles: stale, the rows, four for the last emission
  localparam NROWS = N + NA;         // rows on the bus: N bubbles then the tiles

  reg clk = 0; always #2 clk = ~clk;
  reg rst_n = 0, en = 0;
  reg w_valid = 0; reg [8*N-1:0] w_data = 0; reg [WA-1:0] w_addr = 0, w_end = NW - 1;
  reg a_valid = 0; reg [8*N-1:0] a_data = 0; reg [IA-1:0] a_addr = 0, a_end = NA - 1;
  reg [7:0] cfg = {4'd2, 4'd4};  // weights: feed the array after the fill; activations: drain after the pass
  reg [OA*N + 2*N - 1:0] outbuf_instr = 0;
  wire [N-1:0] wr_rdy;
  wire [8*N-1:0] oacts_rd;

  feather_top #(
    .DPE_COL_NUM(N), .DPE_ROW_NUM(N),
    .WEIGHTS_SRAM_BANK_ADDR_WIDTH(WA),
    .IACTS_SRAM_BANK_ADDR_WIDTH(IA),
    .OUTBUF_SRAM_BANK_ADDR_WIDTH(OA),
    .INSTR_SRAM_BANK_ADDR_WIDTH(IA)
  ) dut (
    .clk(clk), .rst_n(rst_n), .i_feather_top_en(en),
    .i_iacts_zp(8'd0), .i_iacts_zp_valid(1'b1), .i_weights_zp(8'd0), .i_weights_zp_valid(1'b1),
    .i_weights_write_valid(w_valid), .i_weights_write_data(w_data), .i_weights_write_addr(w_addr), .i_weights_write_addr_end(w_end),
    .i_iacts_write_valid(a_valid), .i_iacts_write_data(a_data), .i_iacts_write_addr(a_addr), .i_iacts_write_addr_end(a_end),
    .i_instr_write_valid(1'b0), .i_instr_write_data({CMDW{1'b0}}), .i_instr_write_addr({IA{1'b0}}),
    .i_all_buf_pingpong_config(cfg), .i_outbuf_wr_instr(outbuf_instr), .o_outbuf_data_wr_rdy(wr_rdy),
    .i_scale_val(32'd1), .i_oacts_read_valid(1'b0), .o_oacts_read_data(oacts_rd),
    .i_oacts_read_addr({IA*N{1'b0}}), .i_oacts_read_addr_end({IA*N{1'b0}})
  );

  // the tile: unsigned bytes (the PE's arithmetic is unsigned with a zero point, 0 here)
  integer iacts [0:NA-1][0:N-1];        // [t*N + k][j]: row k of pass t
  integer wts   [0:N-1][0:N-1][0:N-1];  // [i][j][k]; odd k zero
  integer expect_out [0:NA-1][0:N-1];   // [t*N + i][j]: sum_k iacts[t*N + k][j] * wts[i][j][k]
  integer got [0:NROWS-1][0:N-1];
  integer t_row [0:NROWS-1];
  integer perm [0:N-1];
  integer i, j, k, s, c, e, t, found, bad;

  wire [3:0] wstate = dut.feather_CONTROLLER_INST.r_weights_buf_ping_pong_state;
  wire [3:0] astate = dut.feather_CONTROLLER_INST.r_acts_buf_ping_pong_state;
  wire [LOG2N-1:0] last_pe_cntr = dut.feather_GENVAR_DPE_INST_COL_ITER[N-1].feather_GENVAR_DPE_INST_ROW_ITER[N-1].feather_PE_OTHER_ROWS.r_weights_wr_cntr;

  integer cyc = 0; always @(posedge clk) cyc <= cyc + 1;
  integer fc_w = 0, fc_a = 0, row_cnt = 0;
  integer t_en = -1, t_wfeed = -1, t_wstore = -1, t_afeed = -1;
  reg [LOG2N-1:0] prev_cntr = 0;
  reg [32*N-1:0] bus;

  always @(posedge clk) begin
    if (wstate == 2) begin if (fc_w == 0) t_wfeed = cyc; fc_w = fc_w + 1; end
    if (astate == 2) begin if (fc_a == 0) t_afeed = cyc; fc_a = fc_a + 1; end
    if (last_pe_cntr !== prev_cntr) begin t_wstore = cyc; prev_cntr = last_pe_cntr; end
    if (|dut.w_o_birrd_data_bus_valid && row_cnt < NROWS) begin
      bus = dut.w_o_birrd_data_bus;
      t_row[row_cnt] = cyc;
      for (j = 0; j < N; j = j + 1) got[row_cnt][j] = bus[32*j +: 32];
      row_cnt = row_cnt + 1;
    end
  end

  // every PE's ping file against the tile's weights, on demand
  event check_w;
  reg [N*N-1:0] pe_bad = 0;
  genvar gc, gr;
  generate
    for (gc = 0; gc < N; gc = gc + 1) begin : CC
      for (gr = 0; gr < N; gr = gr + 1) begin : RR
        always @(check_w) begin : chk
          integer kk;
          for (kk = 0; kk < N; kk = kk + 1)
            if ($unsigned(dut.feather_GENVAR_DPE_INST_COL_ITER[gc].feather_GENVAR_DPE_INST_ROW_ITER[gr].feather_PE_OTHER_ROWS.r_local_weights_buffer_ping[kk]) !== wts[gr][gc][kk])
              pe_bad[gc * N + gr] = 1'b1;
        end
      end
    end
  endgenerate

  task write_weights;  // row a is stored by PE q = (a+1) mod N^2 in sweep s = (a+1) / N^2 mod N
    integer a, q, col, row;
    begin
      for (a = 0; a < NW; a = a + 1) begin
        q = (a + 1) % (N * N); s = ((a + 1) / (N * N)) % N;
        col = q / N; row = q % N;
        @(negedge clk);
        w_valid = 1; w_addr = a; w_data = 0;
        for (j = 0; j < N; j = j + 1) w_data[8*j +: 8] = wts[row][j][s];  // byte col is what PE q takes
      end
      @(negedge clk); w_valid = 0;
    end
  endtask

  task write_iact_row(input integer kk);
    begin
      a_valid = 1; a_addr = kk; a_data = 0;
      for (j = 0; j < N; j = j + 1) a_data[8*j +: 8] = iacts[kk][j];
    end
  endtask

  integer nbad_w, bubbles, seed = 1;
  initial begin
    // pseudo-random bytes, so no two columns of the tile are alike and the column map is unambiguous
    for (k = 0; k < NA; k = k + 1) for (j = 0; j < N; j = j + 1) iacts[k][j] = $unsigned($random(seed)) % 9;
    for (i = 0; i < N; i = i + 1) for (j = 0; j < N; j = j + 1) for (k = 0; k < N; k = k + 1)
      wts[i][j][k] = (k % 2) ? 0 : $unsigned($random(seed)) % 7;
    for (t = 0; t < NP; t = t + 1) for (i = 0; i < N; i = i + 1) for (j = 0; j < N; j = j + 1) begin
      expect_out[t*N + i][j] = 0;
      for (k = 0; k < N; k = k + 1) expect_out[t*N + i][j] = expect_out[t*N + i][j] + iacts[t*N + k][j] * wts[i][j][k];
    end
    for (j = 0; j < N; j = j + 1) outbuf_instr[OA*N + j] = 1'b1;  // bypass the scaler on every bank
    repeat (5) @(negedge clk); rst_n = 1;
    repeat (2) @(negedge clk); en = 1; t_en = cyc;
    @(negedge clk);
    write_weights();                       // the last row (addr_end) starts the feed
    // all but the last activation row go in during the weight feed; the last one on the edge that ends it
    for (k = 0; k < NA - 1; k = k + 1) begin @(negedge clk); write_iact_row(k); end
    @(negedge clk); a_valid = 0;
    w_end = N * N - 1; cfg[7:4] = 4'd3;    // the feed ends on a pulse at read address N^2-1 in sweep N, going to FILL_PONG
    wait (fc_w == TW - 1);
    @(negedge clk); w_valid = 1; w_addr = 0; write_iact_row(NA - 1);
    @(negedge clk); w_valid = 0; a_valid = 0;
    a_end = (NA + 4) % (1 << IA);          // the read address in activation feed cycle NA+4
    wait (fc_a == TA - 1);
    @(negedge clk); a_valid = 1; a_addr = 0;
    @(negedge clk); a_valid = 0;
    // outputs
    fork
      wait (row_cnt == NROWS);
      begin repeat (4 * N + 4 * NSTAGE + 64) @(negedge clk); end
    join_any
    repeat (8) @(negedge clk);
    -> check_w; #1;
    nbad_w = 0; for (c = 0; c < N * N; c = c + 1) if (pe_bad[c]) nbad_w = nbad_w + 1;
    // the tiles are the last NA rows; the rows before them are the bubbles
    bubbles = row_cnt - NA;
    bad = 0;
    for (c = 0; c < N; c = c + 1) begin  // output column c carries expected column perm[c] (BIRRD's pass-through wiring)
      perm[c] = -1;
      for (e = 0; e < N && perm[c] < 0; e = e + 1) begin
        found = 1;
        for (i = 0; i < NA; i = i + 1) if (got[bubbles + i][c] !== expect_out[i][e]) found = 0;
        if (found) perm[c] = e;
      end
      if (perm[c] < 0) bad = bad + 1;
    end
    for (c = 0; c < N; c = c + 1) for (e = c + 1; e < N; e = e + 1) if (perm[c] >= 0 && perm[c] == perm[e]) bad = bad + 1;
    $display("FEATHER RTL N=%0d: enable@%0d weight-feed@%0d last-store@%0d act-feed@%0d first-row@%0d first-tile-row@%0d last-row@%0d rows=%0d",
             N, t_en, t_wfeed, t_wstore, t_afeed, t_row[0], (bubbles >= 0) ? t_row[bubbles] : -1, t_row[row_cnt-1], row_cnt);
    $display("FEATHER RTL N=%0d: weight feed %0d cycles driven (N^3=%0d by design), last PE store %0d cycles after feed start; activation pass %0d cycles (feed start to last row of tile 1, %0d bubble rows first, first tile row at +%0d); feed start to last row of tile 1 %0d",
             N, TW, N*N*N, t_wstore - t_wfeed + 1, t_row[bubbles + N - 1] - t_afeed + 1, bubbles, t_row[bubbles] - t_afeed + 1, t_row[bubbles + N - 1] - t_wfeed + 1);
    if (NP > 1)
      $display("FEATHER RTL N=%0d: %0d tiles back to back with the weights resident: last rows at +%0d and +%0d after the activation feed start, %0d cycles a tile",
               N, NP, t_row[bubbles + N - 1] - t_afeed + 1, t_row[bubbles + NA - 1] - t_afeed + 1, (t_row[bubbles + NA - 1] - t_row[bubbles + N - 1]) / (NP - 1));
    $write("FEATHER RTL N=%0d: weight files %0d PE(s) wrong of %0d; tile columns %0d wrong of %0d; column map:", N, nbad_w, N*N, bad, N);
    for (c = 0; c < N; c = c + 1) $write(" %0d", perm[c]);
    $display("");
    if (bad > 0) begin
      $write("  tile row 0 got:"); for (c = 0; c < N; c = c + 1) $write(" %0d", got[(bubbles > 0) ? bubbles : 0][c]); $display("");
      $write("  expected row 0:"); for (c = 0; c < N; c = c + 1) $write(" %0d", expect_out[0][c]); $display("");
      $write("  bubble row 0:  "); for (c = 0; c < N; c = c + 1) $write(" %0d", got[0][c]); $display("");
    end
    $display("FEATHER RTL N=%0d NP=%0d: %s", N, NP, (bad == 0 && nbad_w == 0 && bubbles == N) ? "PASS" : "FAIL");
    $finish;
  end
  initial begin #(3000000); $display("FEATHER RTL N=%0d NP=%0d: TIMEOUT wstate=%0d astate=%0d fc_w=%0d fc_a=%0d rows=%0d", N, NP, wstate, astate, fc_w, fc_a, row_cnt); $finish; end
endmodule
