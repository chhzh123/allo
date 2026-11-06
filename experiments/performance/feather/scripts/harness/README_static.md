# E4: FEATHER's RTL (original and corrected) against the SPMW port

FEATHER (Tong et al., ISCA 2024; github.com/maeri-project/FEATHER,
`FEATHER_RTL/RTL`, checkout at `/scratch/hc676/feather_ref`, commit 225951b)
versus its SPMW port (`tests/dataflow/spmw/test_spmw_feather.py` in
`/scratch/hc676/allo`; the checkout's HEAD reads f436658a with local edits, but
the six FEATHER-relevant files -- test_spmw_feather.py, tb_feather_rtl.sv,
spmw_feather_rtl.py, spmw_build_array.py, allo/spmw/cosim.py, allo/spmw/rtl.py
-- are byte-identical to the worktree at 2b810b7, which is the tree the task
calls d7ce0da). Tools: Vitis HLS / Vivado 2023.2, xsim, part
xcu280-fsvh2892-2L-e, 3.333 ns. Everything in this directory was produced by
the scripts in `/scratch/hc676/e4_work` (copied here under `scripts/`).

Directory layout:

- `results.csv` -- one row per system x workload x size x mode.
- `README.md` -- this file: commands, the RTL diff, the command-word mapping,
  layouts, caveats, tables.
- `reports/<run_id>/` -- xsim logs, Vivado utilisation / timing / route reports.
- `validation/<run_id>/` -- seeds and settings (`meta.json`), operands
  (`operands.npz`), expected outputs (`expect.npy` / `Y` in the npz), error
  counts (`check.json` / `result.json`), the bench's E4 lines, the logged bus
  rows (`bus.log`, gzipped when large) or the per-tile completion log.
- `scripts/` -- the generator, the benches, the drivers and the collector.

## 1. What the half-file restriction is: the RTL, not the harness

The shipped controller (`feather_controller.v`) and PE (`feather_pe.v`) do the
following, read from the code and confirmed in simulation:

- In the weight feed state the controller reads one SRAM row a cycle and
  counts `r_pe_sel` one PE a cycle (`r_pe_sel <= r_pe_sel + 1`, PE_SEL_WIDTH =
  log2(COL) + log2(ROW) bits, so it wraps every N^2 cycles). A PE stores a
  weight only when `i_pe_sel == THIS_PE_ID` (THIS_PE_ID = N*col + row), at
  index `r_weights_wr_cntr`, which counts *its own* stores. So every PE stores
  exactly one weight per sweep of N^2 cycles, at index s in sweep s.
- The PE writes the buffer `i_weights_ping_pong_sel` names (`0` -> ping, `1`
  -> pong) and the compute reads the *other* one
  (`w_selected_weight = (sel == 1) ? ping[idx] : pong[idx]`).
- The controller toggles the select whenever `r_pe_sel == N^2 - 1`, i.e. once
  per sweep, in every state (`feather_controller.v` line 580, the block the
  diff below replaces).

Hence in sweep s every PE's index-s weight goes to ping when s is even and to
pong when s is odd, for every PE at once, whatever the SRAM holds: the buffer
chosen for a store depends only on the sweep parity, each PE stores once per
sweep, and the valid to the PEs is unconditionally high in the feed state
(`w_weights_valid_from_ctrl_to_dpe = ~0`), so no PE can skip a sweep. No SRAM
layout can put a PE's N weights into one buffer for even N (every N here is a
power of two). The compute reads one buffer, so it sees the even-index half of
every file (or the odd half, depending on the select's final value). The
select is internal (the controller drives it into the PE chain); a testbench
cannot override it without editing the RTL. That is why
`tb_feather_rtl.sv` had to zero the odd-index weights.

The direct evidence, from the general bench (`tb_feather_rtl_general.sv`) on
the *unmodified* RTL after one complete feed of a general tile (nonzero odd
indices, zero points 7/5 so that a coincidental match of small values cannot
happen), dumping every PE's ping and pong arrays and counting, per weight
index k, the PEs whose ping[k] / pong[k] holds the expected weight
(`validation/rtl_orig_N*_tile_general_orig/*/check.json`, key `buffers`):

    N = 8, 64 PEs:   k=0 ping 64 pong 0 | k=1 ping 0 pong 64 | k=2 64/0 | k=3 0/64 | ... | k=7 0/64
    N = 16, 256 PEs: k even: ping 256 pong 0;  k odd: ping 0 pong 256
    N = 4, 16 PEs:   k=0 16/0, k=1 0/16, k=2 16/0, k=3 0/16

and the tile's outputs are then wrong in every element (every one of the
original RTL's general-weight tiles fails: rows `rtl_orig_N*_tile_general_orig`
in results.csv). The bench also logs every toggle: the shipped controller
toggles N times per feed (once per N^2 cycles); the corrected one once.

Three further defects of the shipped controller were found on the way and are
*not* patched (they are outside the restriction and the measurement does not
need them; they are listed so nobody else trips over them):

1. In `WEIGHTS_PINGPONG_PING_FEED_DPE` the end-of-feed reset of
   `r_weights_pingpong_rd_addr` and `r_pe_sel` is overridden by the
   unconditional increment that follows it in the same always block (the last
   nonblocking assignment wins), so a feed's addresses wrap naturally and the
   state only changes when a write pulse arrives while the read address equals
   `addr_end` -- "a feed never ends on its own".
2. `WEIGHTS_PINGPONG_PONG_FEED_DPE` never advances `r_pe_sel` or sets
   `r_weights_to_use`, and the pong SRAM's read address / enable are driven
   only when the state is `PING_FEED_DPE` (a copy-paste slip), so a feed from
   the pong SRAM cannot deliver weights to more than one PE. The shipped design
   can therefore feed weights only from the ping SRAM, and the ping SRAM can be
   written only in `FILL_PING`, i.e. never while a feed runs: the SRAM-level
   ping/pong cannot overlap a fill with a feed. All whole-workload runs here
   pre-load the ping SRAM with every tile's rows and keep the controller in
   `PING_FEED_DPE` for the whole run, advancing `addr_end` a tile at a time,
   which is the same sequence of feed events the corrected select needs.
3. The PE's zero-point arithmetic is unsigned: `{1'b0, w} - {1'b0, zp}` is a
   9-bit unsigned difference and the product of two such is an unsigned 18-bit
   value, so an operand *below* its zero point is not negated but wrapped
   (a - zp + 512). The RTL's result equals the signed dot product only when
   every difference is non-negative. The reference model here computes exactly
   what the RTL computes (`e4_feather_gen.py: rtl_cols`), reports per run
   whether that equals the signed mathematics (`signed_equals_rtl` in
   `meta.json`), and every workload uses legal operands (u = zp + d, d >= 0,
   so the SPMW side computes the same d values as int8).

## 2. The corrected variant

`/scratch/hc676/e4_feather_fixed/RTL` is a copy of the shipped RTL with one
change in `feather_controller.v` (`/scratch/hc676/e4_feather_fixed/feather_controller.diff`,
copy in `scripts/`): the select toggles once per complete feed -- when the
feed's read address reaches `i_weights_write_addr_end` in a feed state --
instead of once per sweep.

```
--- feather_ref/FEATHER_RTL/RTL/feather_controller.v
+++ e4_feather_fixed/RTL/feather_controller.v
@@ -577,7 +577,19 @@
             endcase
             //________________________________________________________________________________________________________________________
             //************************************************************************************************//
-            if(r_pe_sel == $unsigned(DPE_COL_NUM*WEIGHTS_DEPTH)-1)
+            // [E4 corrected variant] The PEs' ping/pong select toggles once per
+            // complete feed -- the read address reaching addr_end in a feed
+            // state -- instead of once per sweep of DPE_COL_NUM*WEIGHTS_DEPTH
+            // cycles. pe_sel visits every PE once a sweep and a PE stores one
+            // weight per visit, so the shipped toggle put a PE's even-index
+            // weights in one buffer and its odd-index ones in the other, and the
+            // compute, which reads one buffer, saw half of every file. A feed of
+            // WEIGHTS_DEPTH sweeps now fills one buffer completely, the select
+            // flips as it ends, and the next feed fills the other while the
+            // compute reads this one.
+            if(((r_weights_buf_ping_pong_state == WEIGHTS_PINGPONG_PING_FEED_DPE) ||
+                (r_weights_buf_ping_pong_state == WEIGHTS_PINGPONG_PONG_FEED_DPE)) &&
+               (r_weights_pingpong_rd_addr == i_weights_write_addr_end))
             begin
                 r_weights_ping_pong_sel <=  ~r_weights_ping_pong_sel;
             end
```

With it a feed of N sweeps (N^3 cycles) fills one buffer with the complete
N-weight file of every PE, the select flips at the feed's last cycle, the
compute reads that buffer from the next cycle on, and the next feed fills the
other buffer meanwhile. The same bench on the corrected RTL shows every index
of every PE in one buffer (`k = 0..N-1: ping N^2, pong 0`), one toggle per
feed, and bit-exact tiles.

## 3. The general bench and its model

`tb_feather_rtl_general.sv` + `e4_feather_gen.py` + `e4_rtl_run.py` (all in
`scripts/`). The generator writes, per run: the weight SRAM image
(`weights.hex`), one activation image per bank/column (`iacts_bank<j>.hex`,
sparse `@addr` format), the instruction image (`instr.hex`), tile 0's PE files
(`pefiles.hex`, for the buffer dump), the expected bus rows (`expect.npy`),
the operands (`operands.npz`) and `meta.json`. The bench pre-loads every SRAM
through the hierarchy (`$readmemh` into
`dut.WEIGHTS_PING_BUFFER.SP_SRAM_BANKS[0].sram_bank_sp_inst.r_sram_bank`
etc.), then drives the controller through its ports only, and logs the bus rows
around each tile's expected output with their cycle; the Python checker
matches them against the model and measures the completion events.

Model of the RTL (all in `e4_feather_gen.py`, self-tested against the drivers'
`feather_ref` at 4/8/16 for the GEMM and conv programs and random ones):

- NEST: `cols[i, j] = sum_k ((a[k, j] - zpa) mod 512) * ((w[i, j, k] - zpw) mod 512) mod 2^32`
  (the PE at row i, column j holds `weights[i, j, :]`, the same tile tensors
  as the drivers' `tile_operands`).
- BIRRD: 2 log2 N stages; switch i of a stage takes ports 2i (low) and 2i+1
  (high); after stage s port q goes to `reverse_bits(q, min(2+s, log2 N, 2 log2 N - s))`
  (`birrd_simple_cmd_flow_seq.v`), the same wiring as the SPMW port's.
- Timing (controller time; PE row r sees everything r cycles later): the
  activation row at SRAM address a is at the PEs' input during cycle
  A0 + a + 1 (A0 = the feed state's first cycle; the read register lags the
  address a cycle). The PEs' index counter starts at F0 + 1 (F0 = the weight
  feed's first cycle; `r_weights_to_use` becomes N-1 then) and advances every
  cycle from then on (valid is high in every feed state), so a tile's row 0
  must be at a cycle equal to F0 + 1 mod N. A PE emits, four cycles after the
  last-index cycle that closes a window, the sum of the N products of the
  rows in that window; the column mux adds a cycle, BIRRD 2 log2 N; so tile
  t's first bus row is at A0 + a(t, 0) + N + 5 + 2 log2 N, measured as exactly
  that at every size (`offsets: [0]` in every check.json).
- Weight feed: feed cycle c (from F0) reads row c; the row is at the PEs'
  input during c + 1 while pe_sel = (c + 1) mod N^2, so row c is stored by PE
  (c+1) mod N^2 at index floor((c+1)/N^2) of the feed. PE 0's index-0 weight
  of tile t >= 1 is the last row of tile t-1's region; for tile 0 it is the row
  the read register holds when the feed starts, which the bench arranges by
  loading address 0 with that row and letting the feed's start pulse (the
  shipped protocol: a write at `addr_end`, here `addr_end = 0`) rewrite the
  real row 0. Each SRAM row therefore carries one useful byte (the shipped
  controller stores one PE a cycle), which is the bandwidth assumption stated
  in results.csv.
- Instruction: the controller reads the instruction SRAM at the activation
  read address and registers it once more, so the word steering output row r
  of tile t sits at address a(t, 0) + N + 3 + r; one program per tile is
  therefore natural (the conv workload changes programs tile by tile).

Modes: MODE 0 (`_feed` rows) = one feed per tile back to back, `addr_end`
advanced per feed, tile t's activations placed in feed t+1's window (the PEs
then read the buffer feed t filled); MODE 1 (`_resident` rows) = one feed,
ended the shipped way (a write pulse at `addr_end` -> FILL_PONG), then the
tiles' activations back to back at N rows a tile through tile 0's weights.

## 4. The BIRRD command word

`birrd_2x2_simple_cmd_flow_seq.v`: an egg's own command is the most
significant two bits of its incoming per-row word and it forwards the rest,
registered, to the next stage (the command pipelines with the data); the codes
are `00` pass, `01` sum on the low output (port 2i), `10` sum on the high
output (port 2i+1), `11` swap. `birrd_simple_cmd_flow_seq.v`: the controller's
word is `{row N/2-1, ..., row 1, row 0}`, row i at bits `[i*2S +: 2S]` with S =
2 log2 N stages, stage 0 in the row's MS two bits. The drivers' `feather_ref`
semantics are `PS = 0`, `AR = 1` (sum on the *right* output, 2j+1), `AL = 2`
(sum on the left, 2j), `SW = 3`, so the mapping is

    numpy PS -> 00,  AR -> 10,  AL -> 01,  SW -> 11

(the drivers' "right" is the RTL's "high"; the RTL's own diagram calls `01`
"Add-Right" because it draws the high input on the left). Confirmed against
`feather_ref` on tiles (self-test) and by the RTL passing the drivers' GEMM
programs at 8 and 16 and the four conv programs at 4; the other mapping
(`AR -> 01`) gives different rows on every program with an add, so the test
tells them apart. At N = 4 the RTL's BIRRD has 2 log2 4 = 4 stages while the
drivers' programs (and the SPMW port, following `examples/feather`) have
three: the program fills the RTL's first three stages, the fourth passes, and
the RTL's wiring after its third stage (reverse two bits) leaves RTL column p
carrying the drivers' column `reverse_bits(p, 2)` -- columns 1 and 2 swapped;
the host reads accordingly.

## 5. Workloads and layouts

GEMM M = N = K = 128, batch one, the drivers' tiling (`examples/feather/gemm.py`
as reimplemented in `test_spmw_feather.py`): Mt = AW/2, Kt = 2 AH, Nt = AH,
loop order n, m, k (k innermost), tiles = (M/Mt)(K/Kt)(N/Nt) = 4096 at
AW = AH = 8 and 512 at 16. A tile's iActs [AH, AW] holds A[m-block, k-block]
split in two K halves side by side (column j < Mt: A[j, k], column j + Mt:
A[j, AH + k]); weights [AH, AW, AH] hold B[k-block, n-block] transposed and
replicated across the Mt columns of each half; the BIRRD program adds column j
to column j + Mt and reorders, and the drivers' `gemm_extract` columns
([6,5,2,1] at 8, [8,10,11,9,5,6,7,4] at 16) carry the Mt results per output
row. The host accumulates the K/Kt partial tiles (int64) and the result is
checked against A @ B exactly (`host_check` in check.json). The RTL run stores
uint8 u = zp + d (d in [0, 8) at 8 with zero points 7/5; d in [0, 32) at 16
with zero points 0/0); the SPMW run gets the same d as int8 (same seed, same
generator).

Convolution, batch one, input 16x16x64 NHWC, 64 output channels, 3x3, stride
1, padding 1 (same), AW = AH = 4 (the drivers' conv programs exist for 4): the
drivers' tiling from `test_conv_4x4_matches_numpy`: for each output position
(p, q) the 3x3xC window in NHWC order is a [RS, C] matrix; the reduction axis
RS = 9 is padded to 12 (zero-difference operands, i.e. u = zp) so it splits
into AH = 4 rows; a tile is iActs = window[vn:vn+4, ct:ct+4] ([AH = 4 kernel
positions, AW = 4 channels]) and weights[i, j, k] = w[mt+i, ct+j, vn+k]
(OIHW weights flattened over RS); the program `conv_insts()[(p*Q+q) % 4]`
sums the four channel columns into output column pos = (p*Q+q) % 4 (RTL column
`reverse_bits(pos, 2)`); the host accumulates over ct (16 channel blocks) and
vn (3 blocks) per (p, q, mt). Tiles = P*Q * M/AH * C/AW * RS_pad/AH = 256 *
16 * 16 * 3 = 196,608, of which a third of the reduction rows are padding
(9 -> 12). Output layout: `out[m, p, q]`, checked against a direct numpy
convolution. Zero points 128/128, d in [0, 8). Host packing (im2col windows,
the zero-point offset, the RS padding) is done in Python before the run and is
outside the measured cycles on both sides.

Weights-resident runs: the same tiles' activations and programs through tile
0's weights, for the datapath's own throughput -- the RTL's MODE 1 and SPMW's
`feather_stream_x` -- so those rows do not compute the workload's result, and
say so.

## 6. Commands

RTL, original, P&R (one per size; `scripts/pnr_ooc.sh`):

    /scratch/hc676/e4_work/pnr_ooc.sh /scratch/hc676/feather_ref/FEATHER_RTL/RTL <N> /scratch/hc676/e4_pnr/orig_<N>
    /scratch/hc676/e4_work/pnr_ooc.sh /scratch/hc676/e4_feather_fixed/RTL <N> /scratch/hc676/e4_pnr/fixed_<N>

(`synth_design -top feather_top -mode out_of_context -generic DPE_COL_NUM=N
-generic DPE_ROW_NUM=N`, every other parameter the shipped default -- the
authors' own Figure-14 reports use the same depth-4 SRAMs -- then opt, place,
phys_opt, route at `create_clock -period 3.333`.)

RTL, general bench, single tiles (the validation matrices):

    cd /scratch/hc676/e4_work
    python3 e4_matrix.py --rtl /scratch/hc676/e4_feather_fixed/RTL --N 8 --tag fixed --set validate --out /scratch/hc676/e4_rtl_runs/matrix
    python3 e4_matrix.py --rtl /scratch/hc676/feather_ref/FEATHER_RTL/RTL --N 8 --tag orig --set orig --out /scratch/hc676/e4_rtl_runs/matrix

RTL, whole workloads (corrected RTL):

    python3 e4_rtl_run.py --rtl /scratch/hc676/e4_feather_fixed/RTL --N 8  --out /scratch/hc676/e4_rtl_runs/wl_gemm_N8_m0  --workload gemm --gemm 128,128,128 --mode 0 --zpa 7 --zpw 5 --pattern small --seed 0
    python3 e4_rtl_run.py --rtl /scratch/hc676/e4_feather_fixed/RTL --N 8  --out /scratch/hc676/e4_rtl_runs/wl_gemm_N8_m1  --workload gemm --gemm 128,128,128 --mode 1 --zpa 7 --zpw 5 --pattern small --seed 0
    python3 e4_rtl_run.py --rtl /scratch/hc676/e4_feather_fixed/RTL --N 16 --out /scratch/hc676/e4_rtl_runs/wl_gemm_N16_m0 --workload gemm --gemm 128,128,128 --mode 0 --zpa 0 --zpw 0 --pattern mid --seed 0
    python3 e4_rtl_run.py --rtl /scratch/hc676/e4_feather_fixed/RTL --N 16 --out /scratch/hc676/e4_rtl_runs/wl_gemm_N16_m1 --workload gemm --gemm 128,128,128 --mode 1 --zpa 0 --zpw 0 --pattern mid --seed 0
    python3 e4_rtl_run.py --rtl /scratch/hc676/e4_feather_fixed/RTL --N 4  --out /scratch/hc676/e4_rtl_runs/wl_conv_N4_m0  --workload conv --conv 64,16,16,64 --mode 0 --zpa 128 --zpw 128 --pattern small --seed 0
    python3 e4_rtl_run.py --rtl /scratch/hc676/e4_feather_fixed/RTL --N 4  --out /scratch/hc676/e4_rtl_runs/wl_conv_N4_m1  --workload conv --conv 64,16,16,64 --mode 1 --zpa 128 --zpw 128 --pattern small --seed 0

RTL, the record's restricted bench (for the original's baseline rows):

    cd /scratch/hc676/allo && python3 scripts/spmw_feather_rtl.py --feather /scratch/hc676/feather_ref --sizes 4,8,16,32 --passes 2 --out /scratch/hc676/e4_rtl_runs/restricted

SPMW, whole workloads (registry-free driver; HLS of the two roles at 300 MHz,
Vivado elaboration of the array, xsim with a $readmemh bench of the same shape
as `allo.spmw.cosim`'s plus a per-tile completion log):

    python3 e4_spmw_run.py --N 8  --workload gemm --gemm 128,128,128 --pattern small --seed 0 --out /scratch/hc676/e4_spmw/wl_gemm_N8_stream
    python3 e4_spmw_run.py --N 8  --workload gemm --gemm 128,128,128 --pattern small --seed 0 --resident --out /scratch/hc676/e4_spmw/wl_gemm_N8_x
    python3 e4_spmw_run.py --N 16 --workload gemm --gemm 128,128,128 --pattern mid   --seed 0 --out /scratch/hc676/e4_spmw/wl_gemm_N16_stream
    python3 e4_spmw_run.py --N 16 --workload gemm --gemm 128,128,128 --pattern mid   --seed 0 --resident --out /scratch/hc676/e4_spmw/wl_gemm_N16_x
    python3 e4_spmw_run.py --N 4  --workload conv --conv 64,16,16,64 --pattern small --seed 0 --out /scratch/hc676/e4_spmw/wl_conv_N4_stream
    python3 e4_spmw_run.py --N 4  --workload conv --conv 64,16,16,64 --pattern small --seed 0 --resident --out /scratch/hc676/e4_spmw/wl_conv_N4_x

SPMW, P&R (the registry's `feather-stream`, NT = 16, and `feather`):

    cd /scratch/hc676/allo && python3 scripts/spmw_build_array.py --design feather-stream --size <N> --frequency 300 --jobs 8 --pnr --out /scratch/hc676/e4_pnr/spmw_feather-stream_<N>
    cd /scratch/hc676/allo && python3 scripts/spmw_build_array.py --design feather        --size <N> --frequency 300 --jobs 8 --pnr --out /scratch/hc676/e4_pnr/spmw_feather_<N>

Collecting: `python3 e4_collect.py` writes results.csv and copies the reports
and validation files.

## 7. Bandwidth assumptions, stated

- RTL (both variants): the weight feed reads one SRAM row of N bytes per
  cycle and stores one byte of it (one PE a cycle): N^3 cycles per tile,
  N bytes/cycle read, 1 byte/cycle used. The activation pass reads one row of
  N bytes per cycle and one instruction word per row. Every SRAM is
  pre-loaded; the SRAM fill (T * N^3 rows at one row per cycle for the
  weights, which the shipped controller cannot overlap with a feed, see
  section 1) is outside the measured cycles, exactly as in the record's
  single-tile measurement.
- SPMW `feather_stream`: every edge port of the array is driven at one token
  per cycle by the bench (what the fabric accepts, not what a memory system
  supplies): per tile each pair-PE takes 2 activation tokens and 2N weight
  tokens on its own ports and each switch one command token; the array's
  weight intake is N^2/2 tokens (int32 words carrying an int8 each) per cycle,
  i.e. N^2/2 useful bytes per cycle against the RTL's one. `feather_stream_x`
  loads the weight files and commands once (inside its first-tile latency)
  and streams 2 activation tokens per pair-PE per tile.

## 8. Caveats

- The RTL's P&R uses the shipped SRAM depths (4 rows), as the authors'
  Figure-14 synthesis did; the workload runs use behavioural SRAMs sized for
  the whole workload (2^21 to 2^24 rows) which are simulation-only.
- The SPMW `feather_stream` engine keeps its weight file as int32 words
  (a fabric-simulator limitation noted in the design's docstring), four times
  the register bits of the RTL's 8-bit files; `feather` (single tile) uses
  int8 files. Both are placed.
- The SPMW P&R top is `spmw_harness` (LFSR drivers and sinks around
  `spmw_top`, the SPMW flow's standard way of placing an array whose edge
  ports exceed the pins); the RTL is placed out of context with no harness.
- The RTL's zero-point arithmetic is unsigned (section 1, item 3); all
  workloads use legal operands so the RTL, the model and the SPMW side agree
  with the signed mathematics.
- `feather_stream`/`feather_stream_x` for the P&R rows are the registry's
  NT = 16 builds; NT only changes loop bounds.
