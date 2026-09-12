# E4: FEATHER's RTL against its SPMW port, general weights, whole workloads, P&R

FEATHER (Tong et al., ISCA 2024; github.com/maeri-project/FEATHER,
`FEATHER_RTL/RTL`, checkout `/scratch/hc676/feather_ref`, commit 225951b)
versus its SPMW port (`tests/dataflow/spmw/test_spmw_feather.py`; the Allo
checkout `/scratch/hc676/allo`, whose six FEATHER-relevant files --
test_spmw_feather.py, tb_feather_rtl.sv, spmw_feather_rtl.py,
spmw_build_array.py, allo/spmw/cosim.py, allo/spmw/rtl.py -- are byte-identical
to the worktree at 2b810b7). Tools: Vitis HLS / Vivado 2023.2, xsim, part
xcu280-fsvh2892-2L-e, 3.333 ns (300 MHz) everywhere.

Two agents produced this directory. The first (its workspace `/scratch/hc676/e4_work`,
runs under `e4_rtl_runs`, `e4_spmw`, `e4_pnr`) found and patched the RTL's
weight-buffer defect, wrote the general bench and its model, validated the
corrected RTL on single tiles with general weights, ran the whole workloads with
*constrained* operands (small non-negative ranges), and started the P&R chains.
The second (workspace `/scratch/hc676/e4b_work`, runs under `e4b_rtl_runs`,
`e4b_spmw`) re-ran every whole workload with *general* operands on both sides,
added GEMM at 4x4 and convolution at 8x8 and 16x16 (with BIRRD reduce programs
for those widths), added the host-level reduction check to the SPMW runs and a
demonstration of what the RTL does with mixed-sign differences, and collected
everything. `results.csv` says in its `source` column which agent's run each
row is.

Directory layout:

- `results.csv`, `results.json` -- one row per run (both agents' runs).
- `README.md` -- this file. `README_tables.md` -- the tables below, as the
  collector wrote them.
- `reports/<run_id>/` -- per run: `check.json`/`result.json`, the bench's
  `E4`/`SPMW` lines from the xsim log, Vivado utilisation / timing / route
  reports for the P&R rows, `tiles.log.gz` (SPMW per-tile completion cycles).
- `validation/<run_id>/` -- the `check.json` (RTL) / `result.json` (SPMW) of
  every run, plus `meta.json` (seed, zero points, pattern, layout constants);
  for the single-tile matrices one subdirectory per run.
- `scripts/harness/`, `scripts/reporting/` -- both workspaces' scripts;
  `scripts/feather_controller.diff` -- the RTL patch.

## 1. Columns of results.csv

`run_id`; `experiment_id` (E4); `system` (`SPMW port` | `FEATHER RTL`);
`variant` (SPMW: `feather_stream` = every operand streamed per tile,
`feather_stream_x` = weights and commands resident, `feather` = the registry's
single-tile design used for P&R; RTL: `corrected controller` | `shipped
controller`); `workload` (the tiling in words); `array_size`; `weight_mode`
(`general` | `constrained` | the mixed-sign demonstration); `reorder_program`
(the BIRRD program(s) the run used); `implementation_mode` (`cosim` = the SPMW
port's assembled array in xsim, `rtl_sim` = FEATHER's RTL in xsim, `pnr_ooc`);
`target_mhz` (300); `status`; `validation_pass` (what was checked and the
outcome); `tiles`; `first_output_cycles`; `completion_cycles`;
`cycles_per_tile` (the steady-state interval between consecutive tiles' last
output rows, median); `lut`, `ff`, `dsp`, `bram_18k_equiv`, `uram`, `wns_ns`,
`tns_ns`, `unrouted` (P&R rows); `hls_wall_s`, `total_wall_s`; `report_paths`;
`failure_reason`; `source` (which agent's run).

## 2. What the RTL patch changes and why (the first agent's finding, verified)

The shipped controller (`feather_controller.v`) and PE (`feather_pe.v`) do the
following, read from the code and confirmed in simulation:

- In the weight feed state the controller reads one SRAM row a cycle and
  counts `r_pe_sel` one PE a cycle (PE_SEL_WIDTH = log2(COL) + log2(ROW) bits,
  so it wraps every N^2 cycles). A PE stores a weight only when
  `i_pe_sel == THIS_PE_ID`, at index `r_weights_wr_cntr`, which counts *its
  own* stores: every PE stores exactly one weight per sweep of N^2 cycles, at
  index s in sweep s.
- The PE writes the buffer `i_weights_ping_pong_sel` names (`0` ping, `1`
  pong) and the compute reads the *other* one
  (`w_selected_weight = (sel == 1) ? ping[idx] : pong[idx]`).
- The controller toggles the select whenever `r_pe_sel == N^2 - 1`, i.e. once
  per sweep, in every state (`feather_controller.v` line 580).

So in sweep s every PE's index-s weight goes to ping for even s and to pong for
odd s, whatever the SRAM holds: the buffer is chosen by the sweep's parity,
each PE stores once per sweep, and the valid to the PEs is unconditionally
high in the feed state. No SRAM layout can put a PE's N weights into one
buffer. The compute reads one buffer, so it sees the even-index half of every
file (or the odd half). The select is internal; a testbench cannot override it
without editing the RTL. That is why the record's restricted bench
(`tests/dataflow/spmw/tb_feather_rtl.sv`) had to zero the odd-index weights.

Direct evidence (`validation/rtl_orig_N*_single_tiles_orig/*/check.json`, key
`buffers`: per weight index k, how many of the N^2 PEs hold the expected weight
in ping / pong after one complete feed of a general tile on the unmodified RTL):

    N = 8, 64 PEs:   k even: ping 64 pong 0;  k odd: ping 0 pong 64
    N = 16, 256 PEs: k even: ping 256 pong 0; k odd: ping 0 pong 256
    N = 4, 16 PEs:   k even: ping 16 pong 0;  k odd: ping 0 pong 16

and every general-weight tile on the shipped RTL is wrong in every element
(rows `rtl_orig_N*_single_tiles_orig`, and this agent's whole GEMM at 4x4 on
the shipped controller, `rtl_orig_gemm128_N4_feed_general`). The bench logs
every toggle: the shipped controller toggles N times per feed, the corrected
one once.

The corrected variant `/scratch/hc676/e4_feather_fixed/RTL` is a copy of the
shipped RTL with one change in `feather_controller.v`
(`scripts/feather_controller.diff`): the select toggles once per complete feed
-- when the feed's read address reaches `i_weights_write_addr_end` in a feed
state -- instead of once per sweep:

```
-            if(r_pe_sel == $unsigned(DPE_COL_NUM*WEIGHTS_DEPTH)-1)
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
feed, and bit-exact tiles. The patch does not change the datapath, the feed's
bandwidth (still one PE a cycle) or the P&R result beyond noise (compare the
`pnr_rtl_orig_N*` and `pnr_rtl_fixed_N*` rows).

Three further defects of the shipped controller were found and are *not*
patched (outside the restriction; the measurement does not need them):

1. In `WEIGHTS_PINGPONG_PING_FEED_DPE` the end-of-feed reset of the read
   address and `r_pe_sel` is overridden by the unconditional increment in the
   same always block, so a feed's addresses wrap and the state only changes
   when a write pulse arrives while the read address equals `addr_end`: a feed
   never ends on its own.
2. `WEIGHTS_PINGPONG_PONG_FEED_DPE` never advances `r_pe_sel` or sets
   `r_weights_to_use`, and the pong SRAM's read address / enable are driven
   only in `PING_FEED_DPE`, so a feed from the pong SRAM cannot deliver weights
   to more than one PE; the ping SRAM can be written only in `FILL_PING`, never
   while a feed runs: the SRAM-level ping/pong cannot overlap a fill with a
   feed. Every whole-workload run pre-loads the ping SRAM with every tile's
   rows and keeps the controller in `PING_FEED_DPE`, advancing `addr_end` a
   tile at a time.
3. **The PE's zero-point arithmetic is unsigned** (`feather_pe.v`):
   `w_iacts_sub_zp = {1'b0, w_iacts} - {1'b0, r_i_iacts_zp}` is a 9-bit
   unsigned wire, `w_mul_iacts_weights = r_iacts_sub_zp * r_weights_sub_zp` an
   unsigned 9 x 9 -> 18-bit product, zero-extended into the 32-bit
   accumulator. An operand *below* its zero point is not negated but wrapped
   ((u - zp) mod 512), so the RTL equals the signed dot product only when
   every difference is non-negative. See section 4 for what this means for
   "general weights" and for the demonstration rows.

## 3. The benches and how cycles are counted on each side

### 3.1 FEATHER RTL: `tb_feather_rtl_general.sv` + `e4_feather_gen.py` + `e4_rtl_run.py`

The generator writes, per run, the weight SRAM image, one activation image per
bank (column), the instruction image, tile 0's PE files (for the buffer dump),
the expected bus rows (`expect.npy`), the operands and `meta.json`. The bench
pre-loads every SRAM through the hierarchy (`$readmemh` into the SRAM banks'
arrays; the fill is outside the measured cycles, as in the record's bench),
then drives the controller through its ports only, and logs the bus rows
around each tile's expected output with their cycle; the Python checker matches
them against the model of the RTL and measures the completion events.

Model of the RTL (`e4_feather_gen.py`, self-tested against the drivers'
`feather_ref` at 4/8/16 for the GEMM programs, every conv program and random
programs): NEST `cols[i, j] = sum_k ((a[k, j] - zpa) mod 512) * ((w[i, j, k] - zpw) mod 512) mod 2^32`;
BIRRD: 2 log2 N stages, switch i of a stage takes ports 2i (low) and 2i+1
(high), codes `00` pass, `01` sum on the low output, `10` sum on the high
output, `11` swap, after stage s port q goes to
`reverse_bits(q, min(2+s, log2 N, 2 log2 N - s))`. The drivers' PS/AR/AL/SW
map to `00/10/01/11`; at N = 4 the RTL's network has four stages against the
drivers' three (the program fills the first three, the fourth passes) and RTL
column p carries the drivers' column `reverse_bits(p, 2)`.

Timeline (controller time; PE row r sees everything r cycles later): the
activation feed starts at A0 and its read address is the cycle count since A0;
the weight feed starts G = 8 cycles later at F0. **Cycles on the RTL side are
counted from F0, the first cycle of the weight feed**, with every SRAM already
loaded:

- MODE 0 ("a weight feed per tile", rows `*_feed_*`): T feeds back to back,
  N^3 cycles each, `addr_end` advanced a feed at a time; tile t's activation
  rows are placed during feed t+1, when the corrected controller's PEs read the
  buffer feed t filled. `first_output_cycles` = the cycle of tile 0's last bus
  row + 1 - F0 = N^3 + 2N + 5 + 2 log2 N (81 / 539 / 4141 at N = 4 / 8 / 16);
  `completion_cycles` = the last tile's last bus row + 1 - F0;
  `cycles_per_tile` = the spacing of consecutive tiles' last rows = N^3 (the
  feed: one SRAM row of N bytes read per cycle, one byte stored -- the shipped
  controller stores one PE a cycle).
- MODE 1 ("weights resident", rows `*_resident_*`): one feed of tile 0's
  weights, ended the shipped way (a write pulse at `addr_end`), then every
  tile's activations (and its program) back to back at N rows a tile through
  the resident weights; `cycles_per_tile` = N. These rows measure the
  datapath's throughput once the weights are in place; they do not compute
  the workload's result (every tile uses tile 0's weights), and say so.

The check: every tile's N bus rows must equal the model's rows bit for bit
(`tiles_ok/tiles`, `bad_elements`), at the predicted cycle (`offsets: [0]`).
For MODE 0 workload runs the checker additionally reduces the logged rows the
way the workload's host does (`host_check`): GEMM partial tiles accumulated
over K into C and compared with `A @ B` exactly; conv column sums accumulated
over channel and tap blocks and compared with a direct numpy convolution.

### 3.2 SPMW port: `e4_spmw_run.py`

Builds `feather_stream(N, N, NT)` (every operand streamed: per tile each
pair-PE takes 2 activation tokens and 2N weight tokens, each switch one command
token) or `feather_stream_x(N, N, NT)` (weights and commands resident,
activations streamed) through `spmw_build_array.py`'s own stage / synthesise
(HLS of the roles at 300 MHz) / assemble (Vivado elaboration of the array),
then drives the assembled array in xsim with a `$readmemh` bench of the same
shape as `allo.spmw.cosim`'s: every edge input stream offers a token every
cycle (what the fabric accepts, not what a memory system supplies), every
output token is compared with the drivers' reference `feather_ref` (the
`SPMW COSIM PASS (tokens, errors)` line), and the bench logs the cycle at which
every output channel completes each tile. **Cycles on the SPMW side are counted
from the release of reset** with every input stream ready: `first_output_cycles`
= the cycle at which tile 0's last output token is written (`first_tile_done`),
`completion_cycles` = the last tile's, `cycles_per_tile` = the spacing of
consecutive tiles' completions. The per-tile throughput of the streamed engine
is 2N cycles (the weight file: 2N tokens per pair-PE at one a cycle) and of the
resident engine N cycles (the AH rows of the tile).

The host check (this agent's addition): the rows the cosim verified token by
token, one [N, N] block per tile in the drivers' column order, reduced by the
same `gemm_reduce` / `conv_reduce` as the RTL side against numpy on the same
int8 tensors (`host_check`, `host_bad` in `result.json`).

### 3.3 Reading the two counts against each other

Both counts start at "everything is loaded / every stream is ready" and end at
the last output row of the last tile; neither includes the SRAM fills (RTL) or
a memory system (SPMW). The RTL's count starts with the weight feed (its
activation SRAM is pre-loaded and its activation feed had started 8 cycles
earlier); the SPMW count starts with the first tokens of every stream. The
steady-state `cycles_per_tile` is the comparable number: N^3 vs 2N with a
weight change per tile, N vs N with the weights resident. The headline ratio
with a weight change per tile is the weight-load bandwidth (the shipped
controller stores one PE a cycle; the SPMW port loads every PE's file in
parallel at one token a cycle), not the datapath.

### The feed-mode total does not move with the array

`completion_cycles` in the GEMM feed rows is 2,097,169 / 2,097,179 / 2,097,197
at 4x4 / 8x8 / 16x16: the same number three times, while `tiles` falls 32,768 ->
4,096 -> 512 and the resident rows scale properly (131,149 -> 33,299 -> 12,317).
Each is `T * N^3` plus the pipeline's tail, and `weight_rows` in the runs'
`meta.json` -- the SRAM rows the weight port must deliver -- is `T * N^3` =
2,097,152 at all three sizes. This is FEATHER's behaviour, not an artefact of
the stimulus, and two facts of its design are both needed to put it there.

**The loader admits one PE a cycle.** `r_pe_sel` is a free-running counter over
the array's N^2 PEs (`feather_controller.v`: `r_pe_sel + 1` every feed cycle,
wrapping at `DPE_COL_NUM*WEIGHTS_DEPTH - 1`), and a PE writes its file only on
`i_pe_sel == THIS_PE_ID` (`feather_pe.v`), where `THIS_PE_ID` is the global
`DPE_ROW_NUM*col + row` (`feather_top.v`). A PE's file is `WEIGHTS_DEPTH =
DPE_ROW_NUM = N` deep, so a full reload is N^2 PEs x N entries = N^3 cycles.
The port is N bytes wide and one row is read per cycle (`WEIGHTS_NUM_BANKS = 1`,
`WEIGHTS_DATA_WIDTH = 8*DPE_COL_NUM`), so N^3 rows must cross it per tile with
one of every N bytes used. Probed in xsim over a two-tile MODE 0 run -- counting
the PE-file write condition on every PE every cycle -- one steady feed performs
64 writes in 64 cycles at 4x4, 512 in 512 at 8x8 and 4,096 in 4,096 at 16x16,
never more than 2 in any one cycle and idle in half of them: an average of
exactly one weight a cycle at every size, which is the array's whole weight
bandwidth.

**The drivers' GEMM layout replicates each weight across N/2 PE rows**
(`examples/feather/gemm.py`: `[B_left.transpose()] * (AW // 2)`), so a tile
holds N^3 weight bytes for 2N^2 distinct ones. A tile is also `Mt*Kt*Nt =
(N/2)(2N)(N) = N^3` multiply-accumulates. Per tile the array therefore loads
exactly as many weights as it performs MACs -- weight reuse one -- so re-feeding
every tile costs the whole GEMM one cycle per MAC, `M*K*N` = 128^3 = 2,097,152,
whatever the array's size.

Doubling the array quadruples the MACs in a tile and quadruples the weights it
must be given to do them, through a port that does not widen. So with a weight
change per tile a bigger FEATHER array finishes this GEMM in the same time as a
smaller one; only `first_output_cycles` (81 / 539 / 4,141, itself N^3 + 2N + 5 +
2 log2 N) and the resident rows scale. Widening the loader to use the whole
N-byte row would make the feed `T * N^2` -- 524,288 / 262,144 / 131,072, which is
arithmetic and not a run, no such loader exists -- and even that still falls only
2x per doubling, not 4x. Half the invariance is the loader's
one-PE-a-cycle rate; half is the mapping's N/2-fold weight replication, and that
half is architectural.

`weight_rows` is computed from the stimulus, not assumed: the convolution rows
come out at 12,582,912 at 4x4 against 16,777,216 at 8x8 and 16x16, because the 9
taps pad up to a multiple of AH -- 12 reduction rows at AH = 4, 16 at AH = 8 and
16 -- and the count follows the padded reduction exactly (`completion_cycles`
12,582,929 / 16,777,243 / 16,777,261).

## 4. Stimuli: what "general weights" means on each side

- **SPMW port, general** (`weight_mode` = general, this agent's runs):
  `--pattern full`: every stored byte is uniform on [0, 256) and the port reads
  it as int8, i.e. activations and weights uniform on [-128, 127], mixed sign,
  full range, seed 0. The reference is the signed int32 arithmetic of
  `feather_ref`; the host check compares with numpy on the same int8 tensors.
- **FEATHER RTL, general** (this agent's runs): the *same bytes* (same
  generator, same seed, zero points 0/0), so d = u - zp = u is uniform on
  [0, 255]: the full range of the RTL's own (unsigned, zero-point) number
  system. The RTL's unsigned datapath (section 2, item 3) cannot form a signed
  product for u < zp, so "mixed sign" on the RTL is not a stimulus its
  arithmetic supports; the rows `rtl_fixed_N*_tiles_mixed_sign_demo` show what
  it does instead: stored bytes uniform on [0, 256) with zero points 128/128
  (differences in [-128, 127]); the RTL still matches the unsigned model bit for
  bit (`tiles_ok = tiles`) and `signed_equals_rtl = false` says the result is
  not the signed dot product. Cycle counts do not depend on the operand
  values on either side.
- **Constrained** (the first agent's runs, kept for reference): d in [0, 8)
  with zero points 7/5 (GEMM 8x8), d in [0, 32) with zero points 0 (GEMM
  16x16), d in [0, 8) with zero points 128 (conv 4x4); the SPMW side got the
  same d as int8.
- **Single tiles, general** (the first agent's validation matrices,
  `rtl_fixed_N*_single_tiles_validate`): programs x patterns (small,
  full-range, sparse) x zero points (0/0, 7/5, 128/128) x seeds 0/1/2, weights
  resident, one tile each; 162 runs at N = 4, 54 at 8 and 16, all bit-exact.

BIRRD programs ("real reorder programs"): the drivers' GEMM layout programs
for AW = 4 / 8 / 16 (the output-column order `gemm_extract`: [2, 0], [6, 5, 2, 1],
[8, 10, 11, 9, 5, 6, 7, 4]) and, for convolution, one reduce program per output
position in a line -- the drivers' four at AW = 4, and at AW = 8 and 16 programs
found by `e4_feather_gen.find_reduce_program` (a tree of adds over the first
log2 AW stages, then the total routed through the remaining butterfly stages to
column (p*Q + q) mod AW; verified by the selftest against `feather_ref` and, under
the command mapping, against the RTL model). The conv workload changes its
program tile by tile.

## 5. Workloads and layouts

GEMM M = N = K = 128, batch one, the drivers' tiling (`examples/feather/gemm.py`
as reimplemented in `test_spmw_feather.py`): Mt = AW/2, Kt = 2 AH, Nt = AH,
loop order n, m, k (k innermost), tiles = (M/Mt)(K/Kt)(N/Nt) = 32768 at 4x4,
4096 at 8x8, 512 at 16x16. A tile's iActs [AH, AW] holds A[m-block, k-block]
split in two K halves side by side; weights [AH, AW, AH] hold B[k-block,
n-block] transposed and replicated across the Mt columns of each half; the
BIRRD program adds column j to column j + Mt and reorders; the host accumulates
the K/Kt partial tiles and the result is checked against A @ B exactly.

Convolution, batch one, input 16x16x64 NHWC, 64 output channels, 3x3, stride
1, padding 1 (same), the drivers' tiling from `test_conv_4x4_matches_numpy`
generalised to AW = AH = N: for each output position (p, q) the 3x3xC window
in NHWC order is a [RS, C] matrix; the reduction axis RS = 9 is padded to a
multiple of AH with zero-difference operands (12 at N = 4, 16 at N = 8 and 16);
a tile is iActs = window[vn:vn+AH, ct:ct+AW] and weights[i, j, k] = w[mt+i,
ct+j, vn+k]; the program sums the AW channel columns into output column
(p*Q+q) mod AW; the host accumulates over ct and vn per (p, q, mt). Tiles =
P*Q * M/AH * C/AW * RS_pad/AH = 196608 (N = 4), 32768 (N = 8), 4096 (N = 16).
Host packing (im2col windows, the zero-point offset, the RS padding) is done
in Python before the run and is outside the measured cycles on both sides.

## 6. Place and route

RTL: `feather_top` out of context (`synth_design -mode out_of_context -generic
DPE_COL_NUM=N -generic DPE_ROW_NUM=N`, every other parameter the shipped
default -- depth-4 SRAM register arrays, as the authors' own Figure-14 reports
-- then opt, place, phys_opt, route at `create_clock -period 3.333`), shipped
and corrected controller, N = 4 / 8 / 16 / 32 (`scripts/harness/pnr_ooc.sh`).

SPMW port: the registry's `feather-stream` (NT = 16; NT only changes loop
bounds) and `feather` (single tile, resident int8 files) through
`spmw_build_array.py --pnr --frequency 300` (`spmw_harness` top: LFSR drivers
and sinks around `spmw_top`, the SPMW flow's standard way of placing an array
whose edge ports exceed the pins; opt, place, phys_opt, route at 3.333 ns),
queued by the first agent's `spmw_pnr_chain.sh` behind the RTL chains so that
at most two Vivado jobs of this package run at once.

## 7. Commands

RTL, general bench, whole workloads (this agent; `scripts/reporting/chains.sh`
runs them all):

    cd /scratch/hc676/e4b_work
    python3 e4_rtl_run.py --rtl /scratch/hc676/e4_feather_fixed/RTL --N <N> --out /scratch/hc676/e4b_rtl_runs/wl_gemm_N<N>_m<0|1> --workload gemm --gemm 128,128,128 --pattern full --seed 0 --mode <0|1> --zpa 0 --zpw 0
    python3 e4_rtl_run.py --rtl /scratch/hc676/e4_feather_fixed/RTL --N <N> --out /scratch/hc676/e4b_rtl_runs/wl_conv_N<N>_m<0|1> --workload conv --conv 64,16,16,64 --pattern full --seed 0 --mode <0|1> --zpa 0 --zpw 0
    python3 e4_rtl_run.py --rtl /scratch/hc676/e4_feather_fixed/RTL --N <N> --out /scratch/hc676/e4b_rtl_runs/demo_mixed_N<N> --workload tile --tiles 4 --program gemm --pattern mixed --zpa 128 --zpw 128 --seed 0 --mode 1
    python3 e4_rtl_run.py --rtl /scratch/hc676/feather_ref/FEATHER_RTL/RTL --N 4 --out /scratch/hc676/e4b_rtl_runs/wl_gemm_N4_m0_orig --workload gemm --gemm 128,128,128 --pattern full --seed 0 --mode 0 --zpa 0 --zpw 0

SPMW port, whole workloads (this agent):

    python3 e4_spmw_run.py --N <N> --workload gemm --gemm 128,128,128 --pattern full --seed 0 [--resident] --out /scratch/hc676/e4b_spmw/wl_gemm_N<N>_<stream|x>
    python3 e4_spmw_run.py --N <N> --workload conv --conv 64,16,16,64 --pattern full --seed 0 [--resident] --out /scratch/hc676/e4b_spmw/wl_conv_N<N>_<stream|x>

(`--reuse-build` re-used the first agent's HLS'd roles and assembled arrays for
GEMM 8/16 and conv 4, which are NT-specific; GEMM 4 and conv 8/16 were built
afresh.)

Single-tile matrices and the constrained workloads (first agent):

    cd /scratch/hc676/e4_work
    python3 e4_matrix.py --rtl /scratch/hc676/e4_feather_fixed/RTL --N <N> --tag fixed --set validate --out /scratch/hc676/e4_rtl_runs/matrix
    python3 e4_matrix.py --rtl /scratch/hc676/feather_ref/FEATHER_RTL/RTL --N <N> --tag orig --set orig --out /scratch/hc676/e4_rtl_runs/matrix
    python3 e4_rtl_run.py --rtl /scratch/hc676/e4_feather_fixed/RTL --N 8 --out /scratch/hc676/e4_rtl_runs/wl_gemm_N8_m0 --workload gemm --gemm 128,128,128 --mode 0 --zpa 7 --zpw 5 --pattern small --seed 0   (and the N16 mid / conv N4 small variants, see scripts/harness)
    python3 e4_spmw_run.py --N 8 --workload gemm --gemm 128,128,128 --pattern small --seed 0 --out /scratch/hc676/e4_spmw/wl_gemm_N8_stream   (etc.)
    cd /scratch/hc676/allo && python3 scripts/spmw_feather_rtl.py --feather /scratch/hc676/feather_ref --sizes 4,8,16,32 --passes 2 --out /scratch/hc676/e4_rtl_runs/restricted

P&R:

    /scratch/hc676/e4_work/pnr_ooc.sh /scratch/hc676/feather_ref/FEATHER_RTL/RTL <N> /scratch/hc676/e4_pnr/orig_<N>
    /scratch/hc676/e4_work/pnr_ooc.sh /scratch/hc676/e4_feather_fixed/RTL <N> /scratch/hc676/e4_pnr/fixed_<N>
    cd /scratch/hc676/allo && python3 scripts/spmw_build_array.py --design feather-stream --size <N> --frequency 300 --jobs 8 --pnr --out /scratch/hc676/e4_pnr/spmw_feather-stream_<N>
    cd /scratch/hc676/allo && python3 scripts/spmw_build_array.py --design feather --size <N> --frequency 300 --jobs 8 --pnr --out /scratch/hc676/e4_pnr/spmw_feather_<N>

Collecting: `python3 /scratch/hc676/e4b_work/e4b_collect.py`.

The second agent's changes to the first agent's scripts (`scripts/reporting/patch_gen.py`,
`patch_spmw.py` apply them; the Allo checkout was not modified):

- `e4_feather_gen.py`: the `mixed` pattern; `conv_insts(AW)` with
  `find_reduce_program` for AW = 8 / 16; the conv tiling for any AW; the
  selftest checks every conv program sums all columns into its position.
- `e4_spmw_run.py`: imports the generator from its own directory; records the
  operands' semantics; the host reduction check (`host_check`, `host_bad`).

## 8. Caveats

- The RTL's number system is unsigned with zero points (section 2, item 3);
  its general-weight rows use the full range of that system, not signed
  operands. The SPMW port's general rows are mixed-sign int8. Cycle counts are
  independent of the values on both sides.
- The RTL's P&R uses the shipped SRAM depths (4 rows); the workload runs use
  behavioural SRAMs sized for the whole workload (2^21 to 2^25 rows), which are
  simulation-only.
- The RTL's weight feed stores one PE a cycle by design (N^3 cycles a tile,
  N bytes read per cycle, one used); the SPMW `feather_stream` engine takes
  N^2/2 weight tokens a cycle across the array's edge ports (int32 words
  carrying an int8 each). The streamed engine keeps its weight file as int32
  words (a fabric-simulator limitation noted in the design's docstring), four
  times the register bits of the RTL's 8-bit files; `feather` (the single-tile
  design) uses int8 files. Both are placed.
- The SPMW P&R top is `spmw_harness`; the RTL is placed out of context with no
  harness.
- Resident rows (RTL MODE 1, SPMW `feather_stream_x`) run every tile through
  tile 0's weights: datapath throughput, not the workload's result.
- Conv at 8x8 and 16x16 pads the 9 taps to 16 (7/16 of the reduction rows are
  zero-difference padding on both sides alike); at 4x4 the padding is 9 -> 12.
