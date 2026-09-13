# E3 micro: SPMW and Gemmini on one workload

E3 measured SPMW as a TPU. `../gemmini/` measured Gemmini as one. Neither
measured the other's workload, so there was no comparison -- two columns of
numbers about different things. This is the workload they share, run on three
designs: SPMW's programmable stage engine, the same workload on a
fixed-function SPMW datapath, and Gemmini.

## The result

**Programmability is the whole of SPMW's loss, and it is worth 41x exactly.**

| S | | interval / tile | cycles / output row | array busy | first tile | LUT | FF | DSP | slack |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | Gemmini | 6 | 1.500 | 66.7% | **22** | **2,132** | **1,378** | 0 | **+0.591** |
| | SPMW, fixed | **4** | **1.000** | **100%** | 63 | 6,775 | 8,664 | 0 | +0.544 |
| | SPMW, programmable | 164 | 41.0 | 2.4% | 219 | 11,098 | 13,585 | 32 | +0.371 |
| 8 | Gemmini | 10 | 1.250 | 80.0% | **38** | **8,042** | **4,818** | 0 | +0.456 |
| | SPMW, fixed | **8** | **1.000** | **100%** | 112 | 27,037 | 34,488 | 0 | **+0.616** |
| | SPMW, programmable | 328 | 41.0 | 2.4% | 428 | 33,678 | 46,201 | 96 | +0.425 |
| 16 | Gemmini | 18 | 1.125 | 88.9% | **70** | **31,932** | **18,675** | 0 | +0.245 |
| | SPMW, fixed | **16** | **1.000** | **100%** | 213 | 107,930 | 137,728 | 0 | **+0.289** |
| | SPMW, programmable | 656 | 41.0 | 2.4% | 852 | 112,860 | 168,377 | 320 | +0.190 |

Cycles from xsim on the assembled array and from the Gemmini driver, bit-exact
against one shared golden result on both sides. Area and timing from routing on
the U280 at 3.333 ns; the SPMW figures are the `dut` instance's, not the
harness total. BRAM is zero in every row.

**Both sides spend the same kind of resource.** Gemmini routes with zero DSP
blocks -- its PE's multiply is Chisel arithmetic Vivado maps to fabric -- and
the fixed datapath's identical `a * wt` was being inferred into one DSP per
element, so the lookup-table columns were not measuring the same thing. It is
now bound to fabric as well and both are DSP-free. That cost the port 26%, 29%
and 30% of its lookup tables and 12%, 13% and 13% of its registers, and it is
why the numbers here are larger than an earlier version of this table.
The programmable row is left DSP-inferred: it is SPMW-against-SPMW context
rather than half of the comparison, and its `S² + 4S` is a fact about the ISA.

Four things to read off it.

**Take the programmability out and SPMW wins the throughput column.** The
interval falls from `41S` to exactly `S` -- 41.0x at every size, because the
programmable cost never depended on the array and the fixed one is entirely the
array. Gemmini's is `S + 2`, so the fixed datapath is 1.50x, 1.25x and 1.125x
faster per tile, and it is the one design here that keeps its multiply array
busy every cycle.

**It loses the latency column by about 3x**, and that is a different mechanism
from the interval, described under "the first tile" below. Gemmini's first
tile is `4S + 6` exactly -- 22, 38, 70, and 134 at 32x32, which was measured
separately and lands on the same law. SPMW's is `14.2S + 8.5` before the weight
link was deepened and `12.5S + 13` after.

**It does not win the area column, and the gap widens once both multipliers
are in the same place:** 3.2x, 3.4x and 3.4x the lookup tables, 6.3x, 7.2x and
7.4x the registers. That is *after* removing everything the workload does not
use. What remains is not the instruction set -- it is the composition model,
and the registers are where it shows.

**Against its own programmable self the fixed datapath is a straight win**:
41x the throughput for 61%, 80% and 96% of the lookup tables, and no DSPs at
all against `S² + 4S`. Nothing here is a trade between the two SPMW rows; the
programmable engine is paying for a generality this workload never asks for.

## The workload, identical on all three

One `S x S x S` tile per launch, sixteen launches back to back:

    int8 A x int8 B -> int32 accumulate -> + bias -> ReLU -> arithmetic shift
    -> clip to int8

ReLU rather than GELU: Gemmini's `AccumulatorScale` with `has_normalizations =
false` does ReLU and nothing else, so a transformer-shaped epilogue is not
something all three could be asked for. Everything runs on device -- E3's board
design requantised on the host, and this does not.

**The order is Gemmini's.** `AccumulatorScale` activates, *then* scales, then
clips to the narrow type, so the reference is `clip8(relu(A·B + bias) >>
shift)`. The other order also produces plausible numbers, which is why it is
written down rather than assumed.

**One stimulus, one golden result, one seed.**
`tests/dataflow/spmw/test_spmw_tpu_micro.py` generates the operands and the
expected output and writes them to `stimulus/stim_S<n>.txt`;
`MxuVpuStream.scala` reads that file, and
`test_spmw_tpu_micro_fixed.py` imports the same generator rather than writing
its own. `test_it_is_the_same_workload_as_the_programmable_engine` asserts that
the two SPMW designs are handed the same activations, the same weight bytes,
the same bias and the same shift, so the pair is an ablation and not two
designs.

Two properties of the stimulus are asserted rather than hoped for. The shift is
**derived**, not chosen -- the smallest that saturates between 1% and 20% of
outputs -- so `clip8` is exercised instead of being a bound neither reaches;
and the ReLU fires on about half the outputs at every size.

| S | shift | outputs saturating the clip | outputs zeroed by ReLU |
|---:|---:|---:|---:|
| 4 | 7 | 4.7% (12 of 256) | 53.1% (136) |
| 8 | 7 | 14.6% (150 of 1,024) | 48.7% (499) |
| 16 | 8 | 6.7% (274 of 4,096) | 50.5% (2,067) |

The bias is int8-valued because Gemmini's `MeshWithDelays` declares its
top-of-array partial-sum port at *weight* width -- `B_TYPE` is
`Vec(meshColumns, Vec(tileColumns, weightType))`, eight bits here -- so a wider
bias is not something both systems can be handed. That is the one narrowing in
the workload and it narrows toward Gemmini.

## What "fixed-function" removed, and what it kept

`tests/dataflow/spmw/test_spmw_tpu_micro_fixed.py` is the same arithmetic on
the same operands with the control gone:

| | programmable (`tpumicro`) | fixed (`tpumicro-fixed`) |
|---|---|---|
| cell control | fetch, decode, `MSWEEP`/`MPASS`/`MLOAD` | one flat loop, trip count known |
| lane control | 16-word program, 14-opcode decode | straight-line code |
| the clip | five instructions out of `MAX` and `SUB` | one comparison |
| `MUL` opcode | four DSPs a lane, never issued | not present |
| instruction chain | an `op` link across the whole mesh and lane row | not present |

Held constant, deliberately: the same 32-bit packed weight file resident for
all sixteen tiles, the same one-psum-per-output dataflow, the same stimulus and
golden result.

**The shift stays a runtime input**, arriving in the lane's constant memory
next to the bias rather than folded in as a Python constant. Gemmini's
`AccumulatorScale` takes its scale per command, and folding it would turn
SPMW's barrel shifter into wiring and win the area column by answering a
different question. `test_the_shift_is_an_input_and_not_a_constant` pins it.

## Where the 41x went: the dispatch does not pipeline

The programmable engine's *inner* loops were always fine, and the reports say
so rather than the cycle counts implying it. What could not pipeline was
everything above them:

    spmw/S8/report/mac_r0_0_csynth.rpt
      VITIS_LOOP_32_2                     latency 7 ~ 65546   Pipelined no
    spmw/S8/report/vpu_r0_0_csynth.rpt
      VITIS_LOOP_87_6_VITIS_LOOP_90_7     latency 3 ~ 65544   Pipelined no

Neither can: each holds a nested loop whose trip count is an instruction field.
So an instruction cost about 3.6 cycles rather than one, and the ten-instruction
epilogue cost 36 of the 41.

**In the fixed design there is no such loop.** Every loop in every role
reports `Pipelined yes` at II=1 -- nine matrix-cell roles and one lane role at
S=8, thirty-eight loops, no exceptions:

    spmw-fixed/S8/report/mac_r0_0_Pipeline_l_S_r_2_r_csynth.rpt
      l_S_r_2_r     achieved 1   target 1   trip 128   Pipelined yes
    spmw-fixed/S8/report/vpu_r0_0_Pipeline_l_S_m_0_m_csynth.rpt
      l_S__m_0__m   achieved 1   target 1   trip 128   Pipelined yes

`results.json` carries the full list for both designs under each row's `ii`
key; the programmable rows have a `not pipelined` entry in every one of their
twelve roles and the fixed rows have none in any of their ten.

The old ablation still stands and still says what it said: `tpumicro-noclip`
runs the *same* programmable netlist with the five instructions that spell the
clip removed, and the interval falls from 41 to 23 cycles per output row --
**3.6 cycles an instruction, 44% of the programmable interval, spent because
the VPU ISA has `MAX` and no `MIN`.** Adding `MIN` would have taken 41 to 23
and left SPMW 15x to 20x behind Gemmini. Removing the dispatch took it to 1.

## The second overhead, which only the fixed datapath could expose

With every loop at II=1 the assembled array still ran at **half rate** --
`2S - 2` cycles a tile rather than `S`. That is not the units and not the
workload: it is SPMW's default link.

At depth two `spmw_fifo` is a bare register slice whose `full_n` is a flop
(`~v1`), and a producer's pipelined loop cannot re-offer inside that
turnaround. At three and above the same module becomes a LUT-RAM behind that
slice, and `full_n` comes from a count instead. Depths 3, 4, 6 and 8 were all
measured at S=8 and produced **byte-identical cycle traces**, so anything past
the slice suffices; four is the smallest power of two that clears it, it fits
the same LUT-RAM primitives as eight (416 either way at S=4), and it routed
smaller and with more slack than eight.

**And the weight link was the same mistake a second time.** It kept the default
slice on the reasoning that it "moves only during the load" -- but the load is
*serial down each row*, every cell forwarding the words for the cells beyond
it, so a half-rate weight link is paid `S x kw` times inside the first-tile
latency. Three rows, differing in one parameter each, all routed DSP-free under
the same binding:

| S | a / p link | w link | interval / tile | cycles / row | first tile | LUT | FF | slack |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 2 | 6 | 1.500 | 68 | **5,818** | **7,568** | **+0.834** |
| | 4 | 2 | **4** | **1.000** | 65 | 6,676 | 8,580 | +0.568 |
| | 4 | 4 | **4** | **1.000** | **63** | 6,775 | 8,664 | +0.544 |
| 8 | 2 | 2 | 14 | 1.750 | 137 | **22,922** | **32,116** | +0.328 |
| | 4 | 2 | **8** | **1.000** | 122 | 26,399 | 34,096 | +0.514 |
| | 4 | 4 | **8** | **1.000** | **112** | 27,037 | 34,488 | **+0.616** |
| 16 | 2 | 2 | 30 | 1.875 | 278 | **92,880** | **130,776** | **+0.468** |
| | 4 | 2 | **16** | **1.000** | 235 | 104,430 | 136,048 | +0.210 |
| | 4 | 4 | **16** | **1.000** | **213** | 107,930 | 137,728 | +0.289 |

Deepening the **data** links buys 1.50x, 1.75x and 1.88x the throughput for
15%, 15% and 12% more lookup tables. Deepening the **weight** link on top of
that buys nothing in interval -- it cannot, the links it fixes are idle in the
steady state -- and takes 3%, 8% and 9% off the first tile for 1%, 2% and 3%
more lookup tables. Both are worth taking; the first is the larger effect by
far.

Three things worth saying plainly. The half-rate default was invisible for as
long as the epilogue was a program -- at 41 cycles an output row nothing about
the links could matter -- so it is a defect the *fast* design found. "II=1 in
every report" turned out not to imply "II=1 on the array": the loops are a
per-unit property and the rate is a property of the assembled fabric. And the
reasoning that kept the weight link shallow -- "it only moves during the load"
-- was true and irrelevant, because a serial load pays a per-beat cost `S x kw`
times.

## What is left, and why it is registers

After the instruction set is gone the remaining gap is the composition model.
SPMW's split backend synthesises each role as an independent IP and joins them
with a handshaked FIFO on every link; Gemmini's mesh is a synchronous systolic
array whose PE-to-PE connection is a bare register with no handshake at all.

At S=16 that is 256 activation links, 256 partial-sum links, 256 weight links
and 16 lane links, each a LUT-RAM plus a two-entry output slice, and the cell
itself carries a five-stage HLS pipeline. The measured cost is **538
flip-flops per cell against Gemmini's 73**, and it is the whole of the 7.4x
register ratio.

That five-stage pipeline is the same thing the latency section blames, and it
is *not* the weight fetch. Unpacking the weights at load time so the step loop
multiplies by an array element rather than by a shift-and-mask left the
iteration latency at 5 and made the latency worse; the five stages are HLS's
pipelining of a streaming multiply-add, and they were 5 across three different
cell bodies. Fusing several cells into one HLS unit would remove hops rather
than shorten them, which is what `spmw.place`'s `fold` and `unroll` are
declared for -- and `driver.py` raises `SPMWPlacementError` for both, so it is
not expressible today.

The multiply is no longer the other half. Both designs now route with zero DSP
blocks: Gemmini's PE multiply is Chisel arithmetic Vivado maps to fabric, and
the cell's is bound to fabric with `#pragma HLS bind_op ... op=mul
impl=fabric`. **The pragma alone was not enough.** Every role reported 0 DSPs
in csynth at every size and Vivado still inferred 256 when it re-synthesised
the assembled 16x16 array, while leaving 4x4 and 8x8 in fabric from the same
role code -- so `synth_design` is given `-max_dsp 0` as well, which
`spmw_build_array.py` defaults on for any design that asks for fabric
multiplies. E4 hit the identical asymmetry against FEATHER's RTL and is fixed
the same way. The programmable engine's extra `4S` -- four DSPs a lane for a
`MUL` opcode the program never issued -- is a fact about the ISA and its row is
left DSP-inferred to show it.

## What this comparison does not say

**The fixed datapath is not a TPU.** It computes one thing. The netlist the
programmable rows measure is `gpt_stage_v1.stage_engine` **unmodified** -- the
one that ran a GPT-2 block on the U280, softmax and all -- reprogrammed by
changing sixteen words in a stream. What this table prices is what that
flexibility costs when it is not used: 41x of throughput, 1.4-2.1x of LUTs, and
four DSPs a lane for a multiply the program never issues.

**Sixteen tiles is a burst, not a stream, and the two systems load weights
differently.** SPMW's interval excludes any weight reload because all sixteen
tiles are resident; Gemmini's `S + 2` includes a reload it overlaps with the
previous tile's compute. SPMW pays for that once, up front, and it shows in the
latency and in the whole-launch span:

| S | | first tile (cycles) | 16-tile span | span / tile |
|---:|---|---:|---:|---:|
| 4 | Gemmini | **22** | **111** | **6.94** |
| | SPMW, fixed | 63 | 130 | 8.12 |
| 8 | Gemmini | **38** | **187** | **11.69** |
| | SPMW, fixed | 112 | 251 | 15.69 |
| 16 | Gemmini | **70** | **339** | **21.19** |
| | SPMW, fixed | 213 | 494 | 30.88 |

**Over exactly this sixteen-tile burst Gemmini finishes first at every size**,
by 1.17x, 1.34x and 1.46x, even though its steady interval is the slower of the
two. Whether the steady interval or the burst span is the number that matters
depends entirely on how long the stream is. Both are here, measured, and
fitting each side's own span as `overhead + interval x tiles` puts the crossover
at **26 tiles at 4x4, 48 at 8x8 and 94 at 16x16** -- SPMW's steady rate wins
any run longer than that and loses every run shorter.

### The first tile, and what is in it

Both first-tile latencies are straight lines in the array's side, and the
slopes are what separate them:

| | law | 4 | 8 | 16 | 32 |
|---|---|---:|---:|---:|---:|
| Gemmini | `4S + 6` | 22 | 38 | 70 | 134 |
| SPMW, fixed | `12.5S + 13` | 63 | 112 | 213 | -- |

Gemmini's law was checked at a fourth point: 32x32 was elaborated and run
separately and came out at 134, which `4S + 6` predicts exactly. Its latency
already contains a weight load -- the driver's warm-up pass shifts tile 0's
weights in through `d` before any activation, and `first_in` is taken from that
pass -- so this is not SPMW paying for a load that Gemmini avoids.

Two measurements split SPMW's slope. Shrinking the resident weight file moves
it directly, about 2 cycles of slope per word a cell holds -- 14.16 with four
words, 11.68 with two, 9.82 with one -- so at sixteen tiles roughly a third of
the first-tile latency is the file. Extrapolate the file away and the slope is
still about 8 against Gemmini's 4, and that residue is the per-hop cost: a
partial sum crosses `dim` cells and each crossing is an independently
synthesised HLS pipeline plus a handshaked FIFO where Gemmini's is one
flip-flop.

E4 measures the same slope from the other side. FEATHER's port has a
first-output law of `9.4N + 16` -- the same ~10 cycles an array dimension --
against the RTL's `1.11N² + 17`, and there SPMW *wins* above N≈8. The per-hop
cost is a fixed handicap against a tightly pipelined systolic array and an
asset against a baseline whose load is quadratic.

**The programmable engine's row is near its worst case.** A tile with `K = S`
gives `ACCN` exactly one partial sum to fold, so the dispatch has nothing to
amortise over. E3's GPT stage swept 64 tiles per instruction; putting
`plen = 4` and `sweep = 64` through the same fit gives roughly 1.3 cycles per
partial sum. That figure is arithmetic on the fit, not a measurement of that
design.

**Neither side carries DMA, a scratchpad, or a controller.** These are
datapaths routed out of context.

## How it was run

    # SPMW: cycles, then area and timing (same design, two runs)
    run_spmw_micro.sh <S> cosim tpumicro-fixed     # -> SPMW COSIM PASS, SPMW XFORM ...
    run_spmw_micro.sh <S> pnr   tpumicro-fixed     # -> util.rpt, timing.rpt, routed.dcp
    run_spmw_micro.sh <S> cosim tpumicro           # the programmable engine
    run_spmw_micro.sh <S> cosim tpumicro-noclip    # the missing-MIN ablation
    run_spmw_micro.sh <S> cosim tpumicro-slice     # every link at depth 2
    run_spmw_micro.sh <S> cosim tpumicro-wslice    # only the weight link at depth 2
    run_spmw_micro.sh 8   cosim tpumicro-fixed3    # and -fixed4, -fixed6: the depth sweep
    spmw_hier.sh <S> [fixed|slice|wslice]          # -> util_hier.rpt, the dut/harness split

Every SPMW route above binds the integer multiply to fabric, because Gemmini
spends no DSP blocks and two lookup-table counts only compare if the
multiplier is in the same place. That takes two directives, not one:
`#pragma HLS bind_op ... op=mul impl=fabric` from the design's
`spmw_bind_mul_fabric`, and `synth_design -max_dsp 0`, which
`spmw_build_array.py` defaults on for such a design -- HLS reported 0 DSPs at
every size and Vivado still inferred 256 for the assembled 16x16 array without
it. `SPMW_BIND_MUL=0` measures the DSP-inferred form.

    # Gemmini: cycles against the shared stimulus (area was already measured)
    elab_mxuvpu.sh                           # -> mxuvpu_out_<S>_shift/MxuVpu.v
    run_gem_xsim.sh <S>                      # the table's numbers, 16s a size
    run_gem_stream.sh <S>                    # the chiseltest cross-check

    # the table
    collect_micro.py --root /scratch/hc676/e3_micro \
        --area-csv ../gemmini/results.csv --out results.json --csv results.csv
    stage_micro.sh                           # the tree this directory holds

Two things about the Gemmini runs that cost time to learn. Each size needs its
own copy of the project: two `sbt -batch` runs in one directory share `target/`
and corrupt each other's compilation, which is how the first attempt at 8x8 and
16x16 was spent. And chiseltest's interpreter does not scale, so the xsim path
above exists -- but it was calibrated against chiseltest before being used, not
instead of it. Where they differ, at 4x4 and 8x8, it is by a constant one cycle
of interval and the table quotes xsim, the reading that flatters SPMW.

## Files

- `stimulus/` -- the shared operands and golden results, one file per size.
- `spmw/S<n>/` -- the programmable stage engine. `source/` imports
  `gpt_stage_v1.stage_engine` from `tests/dataflow/spmw/`, unchanged; only the
  program is new. `report/cosim_cycles_noclip.txt` is the missing-`MIN`
  ablation on the same netlist.
- `spmw-fixed/S<n>/` -- the fixed-function datapath. `report/slice_*` is every
  link on SPMW's default depth-2 slice and `report/wslice_*` only the weight
  link; both are different netlists and so bring their own area and timing.
- `*/S<n>/generated/` -- what the split backend emitted: one `.cpp` and one
  `.sv` per role, plus the fabric. 12 roles for the programmable engine at
  every size, 10 for the fixed one.
- `*/S<n>/report/` -- routed utilisation, timing and route status, the
  hierarchical split, the C synthesis reports the II claims are quoted from,
  and the cosimulation's cycle lines. `roles_reported.txt` names which matrix
  cell and which lane the committed `_csynth.rpt` files belong to.
- `gemmini/S<n>/source/` -- the streaming driver and the top it drives.
- `gemmini/S<n>/generated/` -- the xsim testbench, and how to re-emit the
  Verilog rather than commit 2.4 MB of it.
