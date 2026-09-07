# E6 -- grouped attention-PV against the conventional array on the same hardware and the same workload

Package `e6_attention` of `spmw_eval_remaining_2026-09-06`.  Platform: AMD Alveo U280
(`xcu280-fsvh2892-2L-e`), Vitis HLS / Vivado 2023.2, 300 MHz (3.333 ns), out-of-context
place and route.  All runs on `brg-zhang-xcel`; the build tree is `/scratch/hc676/e6_work`
(scripts, per-run directories, `logs/`), the Allo checkout `/scratch/hc676/allo`.

The paper compares a grouped 4x4 attention-PV array with an ungrouped 4x2 one, i.e. with
unequal shapes and unequal reductions (`paper_grouped_4x4` / `paper_ungrouped_4x2` below,
kept for reference).  This package makes the fair comparison: **the same sixteen PEs, the
same sixteen DSPs, the same product** `Y = relu(P[M,8] @ V[8,2]) >> 2` at M = 6, 64, 4096,
cycles from RTL co-simulation of the assembled array, resources from routed designs.

## Contents

| file | what |
|---|---|
| `results.csv` | one row per (variant, M, seed) co-simulation and per routed design, the requested columns |
| `results.json` | the same rows with per-pass / per-lane detail, the shapes, a summary table, provenance (which agent ran what) |
| `results_detail.csv` | per-pass and result-lane accounting of every co-simulation (the first collector's format) |
| `reports/<run_id>/` | the tool reports behind each row (`run.json`, `util.rpt`, `timing.rpt`, `route.rpt`, `util_hier.rpt`, `vivado_pnr.log`, Vitis `*_csynth.rpt` / `*_cosim.rpt` / `vitis_hls.log`) |
| `validation/<run_id>/` | co-simulation logs, the traced token stream (`tokens.txt`: tensor, index, cycle, value), the launch tensors (`launch.npz`: inputs, RTL outputs, arrival cycle per element), `row_done.txt` (completion cycle of every query row) |
| `reports/source_change/` | the two SPMW fabrics and their diff, the three HLS kernels and their diffs, every build/collect script and queue file used |
| `reports/diagnostics/` | the handshake probes behind the 3-of-4 finding (`probe_early.log`, raw `probe_*_cycles0-127.txt`, `probe_rows.py`) |
| `reports/hls_failed_before_depth_fix/` | the first agent's HLS co-simulations that deadlocked (RTL deadlock reports) |
| `reports/job_logs/` | every job's stdout and exit marker |

## The designs

Both SPMW fabrics live in `tests/dataflow/spmw/test_spmw_attention_passes.py` (added by the
first agent; identical copies in the server checkout and the local worktree).  They share
the paper's PE (`mac`: `p_out = p_in + a_in * w; a_out = a_in`, int8 x int8 into an int32
psum, II = 1, one DSP48E2 each) and the paper's `act` (`relu` then `>> 2` to int8), taken
token-for-token from `test_spmw_attention.py` (`test_the_units_are_the_papers`).  The
array is `grouped_mxu(WsIO, (4, 4), G)` in both -- the difference is `G`, the boundary
bindings and one extra output port on the boundary unit (`adaptation_diff()`; every
changed line is a signature, placement, binding or axis, `test_the_adaptation_is_topology_and_bindings`).

| variant | array | reduction | launches per product | notes |
|---|---|---|---|---|
| `grouped_constseed` (`grouped`) | 4x4, G = 2 column slabs of d = 2 | 8 = 2 slabs x 4 rows, psum chain serpentines from slab 0's bottom into slab 1's top | 1 | the test file's `pv_grouped`: slab 0's top row seeded with `spmw.stream_in(0, ...)`, which the backend folds into the PE |
| `grouped_zeroseed` (`grouped_zs`) | same | same | 1 | identical, except the seed is a streamed zero tensor `Zin[M,2]` (`e6_build.py: pv_grouped_zs`); see the 3-of-4 finding |
| `conventional_2pass` (`conventional`) | 4x4, G = 1 | 4 per launch: two launches over 4-row weight tiles; the int32 partial leaves un-shifted through `drain.z_out` and re-enters the top row as `Zin` | 2 | d = 2 of the 4 columns carry weights, 2 are zero padding; `drain` = `act` plus one tap line |
| `paper_grouped_4x4` (`capacity_4x4`) | 4x4, G = 2 | 8 | 1 | `scripts/spmw_sweep_table3.attention_pv(4, 4, 2, M)` -- the paper's grouped design; the same fabric as `grouped_constseed` (same cycles) |
| `paper_ungrouped_4x2` (`capacity_4x2`) | 4x2, G = 1, **8 PEs** | **4** (half the product) | 1 | `attention_pv(4, 2, 1, M)` -- the paper's ungrouped design; not the fixed workload |
| `conventional_foldedseed_pass1_only` (`convz0`) | 4x4, G = 1 | 4 (pass 1 only) | 1 | DIAGNOSTIC: the conventional mesh with the top-row seed folded to the constant 0 |
| `hls_grouped` | 4x4, G = 2 | 8 | 1 | `examples/spmw/baselines/hls/attention_pv.cpp` (the paper's hand-written baseline) at `DIM 4, GROUPS 2`; `reports/source_change/attn_grouped.cpp` |
| `hls_conventional` | 4x4, G = 1 | 4 per launch | 2 | the same kernel with the slab machinery removed and `Zin`/`Zout` ports for the hand-over; `attn_conventional.cpp` |
| `hls_conventional_packedlanes` | same | same | 2 | as `hls_conventional`, with a row's four int32 lanes crossing `m_axi` as one 128-bit word; `attn_conventional_packed.cpp` |

Why the budget is equal: every 16-PE design instantiates 16 copies of the same `mac`
body and routes to exactly 16 DSPs (the `dsp` column of every routed row); the links
between PEs are the same depth-2 slice FIFOs (`spmw_fifo.sv`); the boundary is the same
harness; the operands are the same numpy arrays (`default_rng(seed).integers(-4, 4)`,
seeds 0-2, `operands()` in the test file, written out for the HLS testbench by
`e6_hls_operands.py`); the numeric semantics are the same (int8 in, int32 accumulate,
ReLU, `>> 2`, int8 out), and every design's output is checked bit for bit against numpy
and -- for the SPMW fabrics -- against the reference simulator's run of the same
fabric (`test_both_designs_agree_bit_for_bit`, `test_the_passes_hand_over_the_raw_sum`).
The conventional design's second launch is seeded with what the first launch's **RTL**
actually produced (`e6_build.py`, `Z = got["Zout"]`), not with the reference's partial.

## Results

| design | PEs | M=6 | M=64 | M=4096 | cycles / row at M=4096 |
|---|---|---|---|---|---|
| SPMW grouped, constant seed (`grouped`) | 16 | 45 (36) | 123 (42) | 5499 (42) | 1.343 |
| SPMW grouped, streamed zero seed (`grouped_zs`) | 16 | 46 (37) | 104 (37) | 4136 (37) | 1.010 |
| SPMW conventional, two launches (`conventional`) | 16 | 76 (59) | 192 (117) | 8256 (4149) | 2.016 |
| paper's grouped 4x4 (`capacity_4x4`) | 16 | 45 (36) | 123 (42) | 5499 (42) | 1.343 |
| paper's ungrouped 4x2, 8 PEs, half the reduction (`capacity_4x2`) | 8 | 29 (20) | 107 (22) | 5483 (22) | 1.339 |
| DIAGNOSTIC conventional, folded seed, launch 1 only (`convz0`) | 16 | not run | 131 (23) | not run | - |
| HLS baseline grouped (`attn_grouped.cpp`) | 16 | 360 (-) | 430 (-) | 4470 (-) | 1.091 |
| HLS conventional, int32 lanes unwidened, II=4 ports (`attn_conventional.cpp`) | 16 | 794 (-) | 1418 (-) | 37706 (-) | 9.206 |
| HLS conventional, lanes packed to 128 bit (`attn_conventional_packed.cpp`) | 16 | not run | not run | not run | - |

Cells are `completion_cycles (first_output_cycles)`; SPMW cells are seeds 0-2 (identical counts), HLS cells seed 0 through the RTL.  The conventional designs' cycles are the sum of their two launches; their first output is the first final `Y`, in the second launch.

Speedup on equal hardware (completion cycles, conventional / grouped):

| M | grouped (constant seed) vs conventional | grouped (streamed seed) vs conventional | HLS grouped vs HLS conventional | HLS grouped vs HLS conventional (packed lanes) |
|---|---|---|---|---|
| 6 | 1.69x | 1.65x | 2.21x | - |
| 64 | 1.56x | 1.85x | 3.30x | - |
| 4096 | 1.50x | 2.00x | 8.44x | - |

Across systems at M=4096 (same hardware budget, but the HLS cycles include `load_w` and the modelled `m_axi` latency): spmw_grouped_constseed / hls_grouped = 0.81x, spmw_conventional / hls_conventional = 4.57x (SPMW cycles in the denominator: > 1 means the SPMW build finishes sooner).

Routed resources (300 MHz, out of context; the `dsp` column is the PE count):

| run_id | design | M | LUT | FF | DSP | BRAM18 | URAM | WNS ns | TNS ns | unrouted | array-only LUT / FF | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spmw_cap42_4096_pnr | paper_ungrouped_4x2 | 4096 | 529 | 1065 | 8 | 0 | 0 | 1.437 | 0.0 | 0 | 481 / 900 | ok |
| spmw_cap44_4096_pnr | paper_grouped_4x4 | 4096 | 1037 | 2321 | 16 | 0 | 0 | 1.336 | 0.0 | 0 | 941 / 2012 | ok |
| spmw_conv_4096_pnr | conventional_2pass | 4096 | 1207 | 2997 | 16 | 0 | 0 | 1.333 | 0.0 | 0 | 1123 / 2584 | ok |
| spmw_conv_4096_pnr_trim | conventional_2pass_paddingtiedoff | 4096 | 908 | 1729 | 8 | 0 | 0 | 1.358 | 0.0 | 0 | 858 / 1476 | ok |
| spmw_conv_64_pnr | conventional_2pass | 64 | 1027 | 2873 | 16 | 0 | 0 | 1.119 | 0.0 | 0 | 943 / 2460 | ok |
| spmw_grp_4096_pnr | grouped_constseed | 4096 | 1037 | 2321 | 16 | 0 | 0 | 1.279 | 0.0 | 0 | 941 / 2012 | ok |
| spmw_grpzs_4096_pnr | grouped_zeroseed | 4096 | 1072 | 2553 | 16 | 0 | 0 | 1.146 | 0.0 | 0 | 974 / 2180 | ok |
| hls_grouped_pnr_ooc_kernel | hls_grouped | all |  |  |  |  |  |  |  |  |  | missing -- kernel-level P&R did not reach IMPLEMENTATION OK (not run) |
| hls_grouped_pnr_ooc_array | hls_grouped | all |  |  |  |  |  |  |  |  | (this row) | missing -- array-level P&R did not reach IMPLEMENTATION OK (not run) |
| hls_conventional_pnr_ooc_kernel | hls_conventional | all |  |  |  |  |  |  |  |  |  | missing -- kernel-level P&R did not reach IMPLEMENTATION OK (not run) |
| hls_conventional_pnr_ooc_array | hls_conventional | all |  |  |  |  |  |  |  |  | (this row) | missing -- array-level P&R did not reach IMPLEMENTATION OK (not run) |
| hls_conventional_packed_pnr_ooc_kernel | hls_conventional_packedlanes | all |  |  |  |  |  |  |  |  |  | missing -- kernel-level P&R did not reach IMPLEMENTATION OK (not run) |
| hls_conventional_packed_pnr_ooc_array | hls_conventional_packedlanes | all |  |  |  |  |  |  |  |  | (this row) | missing -- array-level P&R did not reach IMPLEMENTATION OK (not run) |

SPMW rows are the routed `spmw_harness` (array + LFSR loaders + output fold); the array-only column is the `dut` line of `util_hier.rpt`.  HLS `_kernel` rows include the m_axi / s_axilite adapters; HLS `_array` rows are `attention_pv_array_pass` alone.  All at 300 MHz, out of context, timing met wherever WNS >= 0.

## How the cycles are measured

SPMW (`e6_build.py`, the stock `allo.spmw.cosim.Testbench` plus a per-token trace): the
harness offers every boundary input channel's tokens from the first cycle after reset,
independently per channel, and every output channel is always ready; `cyc` counts
posedges after reset release.  `completion_cycles` is the harness's `total = c + 1` where
`c` is the cycle at which the **last expected output token of any lane** arrived (for the
conventional design this includes the padding lanes, which trail the result lanes by
8-16 cycles; `results_detail.csv` has the result-lane number too).  `first_output_cycles`
is `first + 1` for the first output token.  A product that takes two launches
(`conventional_2pass`) is the sum of the two launches' totals -- they cannot overlap, the
second launch's `Zin` is the first launch's `Zout` -- and its `first_output_cycles` is the
first launch's total plus the second launch's first output, because only the second launch's
`Y` is the product.  `cycles_per_query_row` = completion / M.  Each cosim row is one seed;
the three seeds' cycle counts are identical in every variant (the datapath is
data-independent).

HLS (`e6_hls_run.sh`): Vitis HLS `csim_design` with seeds 0-2 (the testbench compares `Y`
against numpy's `O`), `csynth_design`, then `cosim_design` with seed 0; `completion_cycles`
is the co-simulation's transaction latency (`ap_start` to `ap_done`), summed over the two
launches of the conventional kernels (the cosim report's min and max are the two
transactions).  It includes what the SPMW harness does not have: `load_w` (16 `m_axi`
reads of V, 87 cycles) and the `m_axi` adapters with Vitis' modelled 64-cycle read
latency, so the HLS-vs-SPMW cycle ratio is not an array-vs-array number; the
grouped-vs-conventional ratio within each system is.  Vitis reports no first-output time.

## The 3-of-4 rhythm (the first agent's open question), settled

Every SPMW design whose top row is seeded by the **folded constant** -- `grouped_constseed`,
the paper's `capacity_4x2` / `capacity_4x4`, and the diagnostic `conventional_foldedseed`
-- streams 3 query rows per 4 cycles once the pipeline fills (4096 rows in 5499 cycles);
the conventional fabric, whose top row is seeded from a stream, runs at 1 row per cycle
(4128 per launch).  The handshake probe (`e6_probe.py`, every channel's write / full_n /
read / empty_n per cycle, `reports/diagnostics/`) shows where it starts:

* `spmw.stream_in(0, into=P.p_in)` is folded into the consuming PEs (`allo/spmw/bindings.py`:
  a rank-0 source "folds into the consuming sites, zero hardware").  The folded roles
  (`mac_r3`, `mac_r5` in the grouped build: `a_in`, `a_out`/`p_out`, no `p_in`) synthesize
  with an iteration latency of 4 instead of 5 -- one cycle less from `a_in` to `p_out` than
  a normal PE takes from `p_in` to `a_out`.
* In the top-left diamond PE(0,0) -> {PE(0,1) by `a`, PE(1,0) by `p`} -> PE(1,1), the upper
  path is therefore one cycle shorter.  PE(0,1)'s psum link to PE(1,1) receives two tokens
  (cycles 8 and 9 in `probe_rows.py`'s dump) before PE(1,1) can start at cycle 10, so the
  depth-2 slice FIFO is full; its `full_n` is a flop, so PE(0,1) stalls at cycle 10 even
  though PE(1,1) pops that very cycle.
* That stall circulates: PE(0,1) not reading -> `a` link from PE(0,0) full at 11 -> PE(0,0)
  stalls -> its `p` link to PE(1,0) empty at 12 -> PE(1,0) stalls -> its `a` link to PE(1,1)
  empty at 13 -> PE(1,1) stalls -> PE(0,1)'s `p` link full again at 14.  Four hops, one cycle
  each, nothing in the loop absorbs a bubble: period 4, one lost cycle per period, forever.
  (`probe_early.log`: every channel of the array shows the `1110` pattern from its first
  token on; `mac_a_out_a_in.full_n[1]` -- PE(0,0)->PE(0,1) -- is the only FIFO that ever
  fills.)
* Two diagnostics confirm the cause and rule out the grouping itself:
  `grouped_zeroseed` (the same grouped fabric, the seed a streamed zero tensor) runs at
  1 row/cycle (M=64: 104 cycles vs 123; M=4096: 4136 vs 5499), and
  `conventional_foldedseed_pass1_only` (the conventional mesh with the constant folded in)
  drops to the rhythm (M=64, one launch: 131 cycles vs 96).

Consequence for the comparison: with the paper's own binding the grouped design is
**1.50x** faster than the conventional one at M=4096 (8256 / 5499); with the seed streamed
instead of folded it is **2.00x** (8256 / 4136), which is the ideal of 16 useful PEs in one
launch against 8 useful PEs in each of two.  The 3-of-4 rhythm is a backend artifact of
constant folding (a one-cycle latency mismatch meeting a depth-2 link with a registered
`full_n`), not a property of grouping; it costs the paper's grouped design and its
ungrouped 4x2 design (`paper_ungrouped_4x2`, 5483 cycles at M=4096) the same 25%.  Fixes
on the SPMW side would be any of: do not fold rank-0 sources into PEs, pad the folded role
to the unfolded latency, or give links written by a folded role one more slot.

## Commands

    # environment (never `set -u` before sourcing)
    export PATH=/scratch/hc676/allo-agent/bin:$PATH LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
    source /work/shared/common/allo/vitis_2023.2_u280.sh
    cd /scratch/hc676/allo

    # one SPMW variant at one M: HLS the roles, assemble, cosim seeds 0-2 (and/or P&R)
    python3 /scratch/hc676/e6_work/e6_build.py --design grouped      --seq 4096 --out /scratch/hc676/e6_work/grp_4096  --cosim --jobs 8
    python3 /scratch/hc676/e6_work/e6_build.py --design conventional --seq 4096 --out /scratch/hc676/e6_work/conv_4096 --cosim --jobs 8
    python3 /scratch/hc676/e6_work/e6_build.py --design grouped      --seq 4096 --out /scratch/hc676/e6_work/grp_4096_pnr --pnr --jobs 8
    #   --design in {conventional, grouped, grouped_zs, conventional_z0, capacity_4x2, capacity_4x4}; --trim-padding ties the
    #   conventional design's padding-column channels off in the harness so synthesis may remove them
    # the handshake probe on a built design
    python3 /scratch/hc676/e6_work/e6_probe.py --out /scratch/hc676/e6_work/grp_4096 --design grouped --seq 4096 --window 0:128
    # HLS side: csim (3 seeds) + csynth + cosim (seed 0); P&R of the array / of the whole kernel
    /scratch/hc676/e6_work/hls/e6_hls_run.sh grouped 4096          # also conventional, conventional_packed
    /scratch/hc676/e6_work/hls/e6_hls_pnr_array.sh grouped 4096    # attention_pv_array_pass, out of context
    /scratch/hc676/e6_work/hls/e6_hls_pnr_array.sh grouped 4096 attention_pv
    # the queues actually used (queue1/queue2: first agent; queue3/queue_hls: this agent), 1-2 jobs at a time
    bash /scratch/hc676/e6_work/e6_queue.sh /scratch/hc676/e6_work/queue3.txt 1
    # the tests behind the fabrics
    python3 -m pytest tests/dataflow/spmw/test_spmw_attention_passes.py -q
    # this bundle
    python3 /scratch/hc676/e6_work/e6_collect2.py

## Provenance

`results.json` marks every run `previous agent` or `this agent` (by start time; the first
agent's last file writes are 21:46 on 2026-09-06; its process was dead by 22:44 when this agent started).  The first agent's runs, reused as they
are: all SPMW co-simulations of `grouped`, `conventional`, `capacity_4x2`, `capacity_4x4` at
M = 6, 64, 4096 (seeds 0-2), the M=4096 P&R of those four, the steady-state probes
(`probe_grp.txt`, `probe_conv.txt`), `pytest_full`.  This agent's runs: the `grouped_zs` and
`conventional_z0` diagnostics, the early-window probes, the M=6 and M=64 P&R of `grouped` and
`conventional`, the padding-tied-off P&R, the `grouped_zs` P&R, every HLS row (the first
agent's HLS attempts all failed -- below), the `pytest_hls2` / `pytest_existing` runs.

## Failures and caveats

* **The first agent's HLS co-simulations all failed** (`reports/hls_failed_before_depth_fix/`):
  M=6 for a missing `m_axi` depth (fixed in its sources before it died); M=64 with a
  SIGSEGV / heap corruption in the co-simulation wrapper, because the wrapper copies
  `depth` elements from every pointer and the testbench's buffers were sized for `seq`
  (fixed in `tb.cpp`: buffers are allocated for the depth's 4096 rows whatever `seq` is);
  M=4096 with an **RTL deadlock** in both kernels: `feed_a` / `seed_p` write every row's /
  lane's stream of a step in one unrolled iteration and `drain` reads every lane in one, the
  rows' wavefronts drift apart by up to 4 x 5 cycles, and with `hls::stream`'s default depth
  of 2 the feeder blocks on a full downstream FIFO while the PE that would drain it starves
  (Vitis' dependence-cycle reports are in `xsim_temp.log` / `vitis_hls.log` there).  Fixed by
  `#pragma HLS STREAM depth = 32` on the `a`, `p`, `d` stream arrays of both kernels -- streams
  only, the PE bodies, count and DSPs are unchanged.  The archived paper baseline
  (`DIM 16`) has the same feeder structure and default depths, so it would need the same
  pragma to co-simulate; it was not co-simulated here.  The hung xsim of that deadlocked run
  (`hls_conventional_4096`, 66 minutes at 100% CPU after the deadlock report) was the one
  process this agent stopped.
* **`hls_conventional`'s throughput is bounded by its ports, not its array**: `seed_p_loop`
  and `drain_loop` synthesize at II = 4 because the four int32 `Zin` / `Zout` accesses per
  step are not widened (Vitis: "burst ... bit width 32" on gmem2 / gmem3; the int8 ports of
  the same kernel and of `hls_grouped` were widened and run at II = 1), so a launch takes
  ~4.6 cycles per row.  `hls_conventional_packedlanes` packs the four lanes into one 128-bit
  word by hand, which is what an HLS designer does about it; both are reported.
* `export_design -flow impl` (the project-mode Vivado route) failed to launch its `synth_1`
  run on this host for every kernel (`reports/hls_<design>_export_attempt/impl.log`); the HLS
  P&R numbers come instead from the same non-project Vivado flow the SPMW builds use
  (`e6_hls_pnr_array.sh`: `synth_design -mode out_of_context`, `opt`, `place`, `phys_opt`,
  `route`, default directives), once with the dataflow array as top (`..._pnr_ooc_array`,
  the counterpart of the SPMW `dut`) and once with the whole kernel (`..._pnr_ooc_kernel`,
  AXI adapters included).
* The SPMW routed numbers are of `spmw_harness` = the array (`dut`) plus the LFSR loaders
  and the output fold that keep synthesis honest; `results.json` carries the `dut`-only and
  harness-only LUT/FF split from `util_hier.rpt` (`lut_array_only`, `lut_harness_only`, ...).
  The conventional design's two padding columns are real hardware in that build (the
  harness drives their weights and reads their outputs); `conventional_2pass_paddingtiedoff`
  is the same design with those channels tied off so synthesis may remove them -- what a
  designer who does not care about the idle columns would build.
* SPMW P&R is per M because the trip count is compiled into the roles; the HLS kernels'
  `steps` is a runtime argument, so one RTL (the M=4096 project's) serves every M
  (`M = all` in those rows).
* `first_output_cycles` is empty for the HLS rows (Vitis reports per-launch latency only)
  and for every P&R row; the fabric-simulator rows (`implementation_mode = simulator`) have
  no cycles at all -- `allo/backend/simulator.py` is a functional dataflow simulator without
  a clock -- and exist to record the bit-exact validation of both fabrics at every M and seed.
* `paper_ungrouped_4x2` computes half the reduction (L = 4) on 8 PEs; it is in the tables
  only to reproduce the paper's unequal comparison, not as a fixed-workload point.
* One seed per HLS co-simulation (seed 0 through the RTL; seeds 0-2 through the C model);
  three seeds per SPMW co-simulation.
