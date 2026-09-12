// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

`timescale 1ns/1ps
// General-weight bench for FEATHER's RTL (maeri-project/FEATHER, FEATHER_RTL/RTL),
// the shipped controller or the corrected one: T tiles of an N x N NEST through
// the real BIRRD, arbitrary weights in every PE file (odd indices included),
// unsigned operands with zero points, one BIRRD program per tile. Every SRAM is
// pre-loaded from hex files written by e4_feather_gen.py ($readmemh through the
// hierarchy; the fill is outside the measured cycles, as in tb_feather_rtl.sv),
// the controller is then driven through its ports alone, and the bus rows in a
// window around each tile's expected output are logged with their cycle for the
// Python checker (the numpy model of the RTL's own arithmetic and network).
//
// Timeline (controller time; each PE row sees it a cycle later than the row above):
//  - The activation feed starts first (A0: the first cycle of its feed state) and
//    runs for the whole simulation; its read address is the cycle count since A0.
//    Tile t's rows sit at addresses A_BASE + t*A_STRIDE + k, its BIRRD program at
//    A_BASE + t*A_STRIDE + N + 3 + r for output row r (the instruction SRAM is
//    read at the activation address, two cycles ahead of the data it steers).
//  - The weight feed starts G cycles later (F0). MODE 0: the feeds of T tiles are
//    back to back, WLEN cycles each, addr_end advanced a feed at a time; the
//    activation rows of tile t are placed during feed t+1, when the corrected
//    controller's PEs read the buffer feed t filled. MODE 1: one feed, ended with
//    the write pulse the shipped controller wants, then T passes back to back at
//    N rows a tile.
//  - Tile t's first bus row is expected at A0 + A_BASE + t*A_STRIDE + N + 5 + 2 log2 N.
module tb;
  parameter N = 8;
  parameter T = 1;
  parameter MODE = 1;
  parameter WA = 9;           // weight SRAM address width
  parameter IA = 10;          // activation / instruction SRAM address width
  parameter A_BASE = 519;     // activation address of tile 0's row 0
  parameter A_STRIDE = 8;     // and the stride between tiles
  parameter G = 8;            // weight feed starts G cycles after the activation feed
  parameter WLEN = N * N * N; // cycles in one weight feed: N^3 over the published
                              // one-PE-a-cycle select, N^2 over the row-wise one
                              // (scripts/loader/apply_row_loader.py)
  parameter WIN = 4;          // bus rows logged from WIN cycles before a tile's first row to WIN after its last
  localparam LOG2N = $clog2(N);
  localparam NSTAGE = 2 * LOG2N;
  localparam CMDW = 2 * NSTAGE * (N / 2);
  localparam OA = 4;
  localparam N3 = N * N * N;  // the PE files: N^2 PEs, N bytes each -- the load's
                              // size, which WLEN does not change, only its cost
  localparam LAT = N + 5 + NSTAGE;

  reg clk = 0; always #2 clk = ~clk;
  reg rst_n = 0, en = 0;
  reg w_valid = 0; reg [8*N-1:0] w_data = 0; reg [WA-1:0] w_addr = 0, w_end = 0;
  reg a_valid = 0; reg [8*N-1:0] a_data = 0; reg [IA-1:0] a_addr = 0, a_end = {IA{1'b1}};
  reg [7:0] cfg = {4'd2, 4'd4};  // weights: feed after the fill; activations: (never taken)
  reg [7:0] zpa = 0, zpw = 0;
  reg [OA*N + 2*N - 1:0] outbuf_instr = 0;
  wire [N-1:0] wr_rdy;
  wire [8*N-1:0] oacts_rd;
  string dir = ".";
  integer zpa_i = 0, zpw_i = 0;
  reg [8*N-1:0] wrow0 [0:0];

  feather_top #(
    .DPE_COL_NUM(N), .DPE_ROW_NUM(N),
    .WEIGHTS_SRAM_BANK_ADDR_WIDTH(WA),
    .IACTS_SRAM_BANK_ADDR_WIDTH(IA),
    .OUTBUF_SRAM_BANK_ADDR_WIDTH(OA),
    .INSTR_SRAM_BANK_ADDR_WIDTH(IA)
  ) dut (
    .clk(clk), .rst_n(rst_n), .i_feather_top_en(en),
    .i_iacts_zp(zpa), .i_iacts_zp_valid(1'b1), .i_weights_zp(zpw), .i_weights_zp_valid(1'b1),
    .i_weights_write_valid(w_valid), .i_weights_write_data(w_data), .i_weights_write_addr(w_addr), .i_weights_write_addr_end(w_end),
    .i_iacts_write_valid(a_valid), .i_iacts_write_data(a_data), .i_iacts_write_addr(a_addr), .i_iacts_write_addr_end(a_end),
    .i_instr_write_valid(1'b0), .i_instr_write_data({CMDW{1'b0}}), .i_instr_write_addr({IA{1'b0}}),
    .i_all_buf_pingpong_config(cfg), .i_outbuf_wr_instr(outbuf_instr), .o_outbuf_data_wr_rdy(wr_rdy),
    .i_scale_val(32'd1), .i_oacts_read_valid(1'b0), .o_oacts_read_data(oacts_rd),
    .i_oacts_read_addr({IA*N{1'b0}}), .i_oacts_read_addr_end({IA*N{1'b0}})
  );

  wire [3:0] wstate = dut.feather_CONTROLLER_INST.r_weights_buf_ping_pong_state;
  wire [3:0] astate = dut.feather_CONTROLLER_INST.r_acts_buf_ping_pong_state;
  wire sel = dut.feather_CONTROLLER_INST.r_weights_ping_pong_sel;

  integer cyc = 0; always @(posedge clk) cyc <= cyc + 1;
  integer A0 = -1, F0 = -1, ntoggle = 0, tcur = 0, nlogged = 0, fd, j, t;
  reg prev_sel = 0;
  reg [32*N-1:0] bus;

  function integer win_lo(input integer tt); win_lo = A0 + A_BASE + tt * A_STRIDE + LAT - WIN; endfunction
  function integer win_hi(input integer tt); win_hi = A0 + A_BASE + tt * A_STRIDE + LAT + N + WIN; endfunction

  always @(posedge clk) begin
    if (astate == 2 && A0 < 0) A0 = cyc;
    if (wstate == 2 && F0 < 0) F0 = cyc;
    if (sel !== prev_sel) begin
      ntoggle = ntoggle + 1;
      if (ntoggle <= 6 || (ntoggle % 512) == 0) $display("E4 TOGGLE n=%0d cycle=%0d sel=%0d", ntoggle, cyc, sel);
      prev_sel = sel;
    end
    if (A0 >= 0) begin
      while (tcur < T && cyc >= win_hi(tcur)) tcur = tcur + 1;
      if (tcur < T && cyc >= win_lo(tcur) && (|dut.w_o_birrd_data_bus_valid)) begin
        bus = dut.w_o_birrd_data_bus;
        $fwrite(fd, "%0d", cyc);
        for (j = 0; j < N; j = j + 1) $fwrite(fd, " %0d", bus[32*j +: 32]);
        $fwrite(fd, "\n");
        nlogged = nlogged + 1;
      end
    end
  end

  // the SRAM images: one file per activation bank (column), one for the
  // instruction words, one for the weight rows (row 0 = what the read register
  // must hold at the feed's first cycle; the start pulse writes the real row 0)
  event load_ev, dump_ev;
  reg [7:0] pef [0:N3-1];
  integer ping_ok [0:N-1];
  integer pong_ok [0:N-1];
  genvar gj, gc, gr;
  generate
    for (gj = 0; gj < N; gj = gj + 1) begin : LD
      always @(load_ev)
        $readmemh($sformatf("%s/iacts_bank%0d.hex", dir, gj), dut.IACTS_PING_SRAM.SP_SRAM_BANKS[gj].sram_bank_sp_inst.r_sram_bank);
    end
    for (gc = 0; gc < N; gc = gc + 1) begin : CC
      for (gr = 0; gr < N; gr = gr + 1) begin : RR
        always @(dump_ev) begin : chk
          integer kk;
          for (kk = 0; kk < N; kk = kk + 1) begin
            if (dut.feather_GENVAR_DPE_INST_COL_ITER[gc].feather_GENVAR_DPE_INST_ROW_ITER[gr].feather_PE_OTHER_ROWS.r_local_weights_buffer_ping[kk] === pef[(gc * N + gr) * N + kk])
              ping_ok[kk] = ping_ok[kk] + 1;
            if (dut.feather_GENVAR_DPE_INST_COL_ITER[gc].feather_GENVAR_DPE_INST_ROW_ITER[gr].feather_PE_OTHER_ROWS.r_local_weights_buffer_pong[kk] === pef[(gc * N + gr) * N + kk])
              pong_ok[kk] = pong_ok[kk] + 1;
          end
        end
      end
    end
  endgenerate

  integer F0x = -1, maxcyc, dbg = 0;
  // +dbg=1: PE (0,0)'s index counter, emission and the column / bus valids around the first pass
  always @(posedge clk)
    if (dbg && F0x > 0 && cyc >= F0x + WLEN - 4 && cyc <= F0x + WLEN + 3 * N + 12)
      $display("E4D cyc=%0d wv=%0d wtu=%0d iv=%0d ia=%0d cnt=%0d w=%0d ready=%0d nsip=%0d sum=%0d out=%0d ov=%0d colv=%0d col=%0d busv=%b sel=%0d",
        cyc,
        dut.feather_GENVAR_DPE_INST_COL_ITER[0].feather_GENVAR_DPE_INST_ROW_ITER[0].feather_PE_OTHER_ROWS.i_weights_valid,
        dut.feather_GENVAR_DPE_INST_COL_ITER[0].feather_GENVAR_DPE_INST_ROW_ITER[0].feather_PE_OTHER_ROWS.i_weights_to_use,
        dut.feather_GENVAR_DPE_INST_COL_ITER[0].feather_GENVAR_DPE_INST_ROW_ITER[0].feather_PE_OTHER_ROWS.i_iacts_valid,
        dut.feather_GENVAR_DPE_INST_COL_ITER[0].feather_GENVAR_DPE_INST_ROW_ITER[0].feather_PE_OTHER_ROWS.i_iacts,
        dut.feather_GENVAR_DPE_INST_COL_ITER[0].feather_GENVAR_DPE_INST_ROW_ITER[0].feather_PE_OTHER_ROWS.r_weights_sel_for_iacts_use,
        dut.feather_GENVAR_DPE_INST_COL_ITER[0].feather_GENVAR_DPE_INST_ROW_ITER[0].feather_PE_OTHER_ROWS.w_selected_weight,
        dut.feather_GENVAR_DPE_INST_COL_ITER[0].feather_GENVAR_DPE_INST_ROW_ITER[0].feather_PE_OTHER_ROWS.r_output_ready,
        dut.feather_GENVAR_DPE_INST_COL_ITER[0].feather_GENVAR_DPE_INST_ROW_ITER[0].feather_PE_OTHER_ROWS.r_next_sum_in_prog,
        dut.feather_GENVAR_DPE_INST_COL_ITER[0].feather_GENVAR_DPE_INST_ROW_ITER[0].feather_PE_OTHER_ROWS.r_sum,
        dut.feather_GENVAR_DPE_INST_COL_ITER[0].feather_GENVAR_DPE_INST_ROW_ITER[0].feather_PE_OTHER_ROWS.o_out_data,
        dut.feather_GENVAR_DPE_INST_COL_ITER[0].feather_GENVAR_DPE_INST_ROW_ITER[0].feather_PE_OTHER_ROWS.o_out_data_valid,
        dut.w_o_col_mux_data_valid[0], dut.w_o_col_mux_data[0], dut.w_o_birrd_data_bus_valid, sel);
  initial begin
    if (!$value$plusargs("dir=%s", dir)) dir = ".";
    if (!$value$plusargs("dbg=%d", dbg)) dbg = 0;
    if ($value$plusargs("zpa=%d", zpa_i)) zpa = zpa_i;
    if ($value$plusargs("zpw=%d", zpw_i)) zpw = zpw_i;
    for (j = 0; j < N; j = j + 1) begin ping_ok[j] = 0; pong_ok[j] = 0; end
    fd = $fopen({dir, "/bus.log"}, "w");
    for (j = 0; j < N; j = j + 1) outbuf_instr[OA*N + j] = 1'b1;  // bypass the scaler on every bank
    repeat (5) @(negedge clk); rst_n = 1;
    @(negedge clk);
    $readmemh({dir, "/weights.hex"}, dut.WEIGHTS_PING_BUFFER.SP_SRAM_BANKS[0].sram_bank_sp_inst.r_sram_bank);
    $readmemh({dir, "/instr.hex"}, dut.INSTR_SRAM.SP_SRAM_BANKS[0].sram_bank_sp_inst.r_sram_bank);
    $readmemh({dir, "/pefiles.hex"}, pef);
    $readmemh({dir, "/wrow0.hex"}, wrow0);
    -> load_ev; #1;
    @(negedge clk); en = 1;
    repeat (4) @(negedge clk);
    // the activation fill 'ends' with a write at addr_end (the last row, unused): the feed runs from the next cycle
    a_valid = 1; a_addr = a_end; a_data = 0;
    @(negedge clk); a_valid = 0;
    repeat (G - 1) @(negedge clk);
    // the weight fill ends with the write of row 0 at addr_end = 0: the read register keeps the row it
    // loaded from address 0 before (PE 0's first weight), the SRAM now holds the real row 0
    w_valid = 1; w_addr = 0; w_end = 0; w_data = wrow0[0];
    @(negedge clk); w_valid = 0; w_end = WLEN - 1; F0x = cyc;
    $display("E4 START A0=%0d F0=%0d G=%0d", F0x - G, F0x, G);
    if (MODE == 0) begin
      for (t = 1; t < T; t = t + 1) begin
        wait (cyc == F0x + t * WLEN); @(negedge clk); w_end = (t + 1) * WLEN - 1;
      end
    end else begin
      // end the feed the shipped way: a write pulse while the read address equals addr_end, going to FILL_PONG
      wait (cyc == F0x + WLEN - 1); @(negedge clk); w_valid = 1; w_addr = 0; cfg[7:4] = 4'd3;
      @(negedge clk); w_valid = 0;
      wait (cyc == F0x + WLEN + N + 6); @(negedge clk);
      -> dump_ev; #1;
      for (j = 0; j < N; j = j + 1) $display("E4 BUF k=%0d ping=%0d pong=%0d of=%0d", j, ping_ok[j], pong_ok[j], N * N);
    end
    wait (cyc == win_hi(T - 1) + 4);
    $display("E4 END cycle=%0d logged=%0d toggles=%0d A0obs=%0d F0obs=%0d", cyc, nlogged, ntoggle, A0, F0);
    $fclose(fd);
    $finish;
  end
  initial begin
    maxcyc = ((MODE == 0) ? (T + 2) * WLEN : 3 * WLEN + T * N) + 4000;
    wait (cyc == maxcyc);
    $display("E4 TIMEOUT cycle=%0d wstate=%0d astate=%0d logged=%0d tcur=%0d", cyc, wstate, astate, nlogged, tcur);
    $fclose(fd);
    $finish;
  end
endmodule
