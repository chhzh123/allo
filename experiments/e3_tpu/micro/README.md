# E3 micro: SPMW and Gemmini on one workload

E3 measured SPMW as a TPU. `../gemmini/` measured Gemmini as one. Neither
measured the other's workload, so there was no comparison -- two columns of
numbers about different things. This is the workload they share.

## The result

**SPMW loses every measured column.** It is 27 to 36 times slower per tile, 4
to 5 times larger in lookup tables, about 10 times larger in registers, it
spends DSP blocks where Gemmini spends none, and it closes timing with less
slack. Nothing here is close.

The interesting part is that the loss is one mechanism, and it is measurable:
SPMW's epilogue is a *program*, and its instruction dispatch does not pipeline.
SPMW keeps its multiply array busy one cycle in 41 at every size; Gemmini's is
busy two cycles in three at 4x4, rising to eight in nine at 16x16.

## The workload, identical on both sides

One `S x S x S` tile per launch, sixteen launches back to back:

    int8 A x int8 B -> int32 accumulate -> + bias -> ReLU -> arithmetic shift
    -> clip to int8

ReLU rather than GELU: Gemmini's `AccumulatorScale` with `has_normalizations =
false` does ReLU and nothing else, so a transformer-shaped epilogue is not
something both systems could be asked for. Everything runs on device on both
sides -- E3's board design requantised on the host, and this does not.

**The order is Gemmini's.** `AccumulatorScale` activates, *then* scales, then
clips to the narrow type, so the reference is `clip8(relu(A·B + bias) >>
shift)`. The other order also produces plausible numbers, which is why it is
written down rather than assumed.

**One stimulus, one golden result, one seed.**
`tests/dataflow/spmw/test_spmw_tpu_micro.py` generates the operands and the
expected output and writes them to `stimulus/stim_S<n>.txt`;
`MxuVpuStream.scala` reads that file rather than generating its own. The two
halves are checked against the same bytes.

Two properties of the stimulus are asserted rather than hoped for. The shift is
**derived**, not chosen -- the smallest that saturates between 1% and 20% of
outputs -- so `clip8` is exercised on both sides instead of being a bound
neither reaches; and the ReLU fires on about half the outputs at every size. At
S=16, 280 of 4,096 outputs clip to 127 and 2,067 are zeroed.

| S | shift | outputs clipped | outputs zeroed by ReLU |
|---:|---:|---:|---:|
| 4 | 7 | 4.7% | 53.1% |
| 8 | 7 | 14.6% | 48.1% |
| 16 | 8 | 6.7% | 50.1% |

**The bias is int8-valued.** Not a simplification we chose: `MeshWithDelays`
declares its partial-sum input `B_TYPE = Vec(meshColumns, Vec(tileColumns,
weightType))`, eight bits wide here, so the top-of-array bias port cannot carry
more. SPMW holds it in a 32-bit register and would take any width. This is the
one place the shared workload was narrowed to what both machines can do, and it
narrows toward Gemmini.

## The table

`xcu280-fsvh2892-2L-e`, out of context, 3.333 ns target, nothing unrouted
anywhere. Cycles from simulation of the assembled design, not from a model.
BRAM is RAMB18 equivalents, `2 x Block RAM Tile`.

| S | system | latency | interval | cyc/out row | LUT | FF | DSP | BRAM18 | WNS (ns) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | Gemmini MXU+VPU | 22 | **6** | 1.5 | **2,132** | **1,378** | **0** | 0 | **+0.591** |
| 4 | SPMW stage engine | 219 | 164 | 41 | 11,098 | 13,585 | 32 | 0 | +0.371 |
| 8 | Gemmini MXU+VPU | 38 | **10** | 1.25 | **8,042** | **4,818** | **0** | 0 | **+0.456** |
| 8 | SPMW stage engine | 428 | 328 | 41 | 33,678 | 46,201 | 96 | 0 | +0.425 |
| 16 | Gemmini MXU+VPU | 70 | **18** | 1.125 | **31,932** | **18,675** | **0** | 0 | **+0.245** |
| 16 | SPMW stage engine | 852 | 656 | 41 | not routed | not routed | not routed | not routed | not routed |

**SPMW's 16x16 row is cycles only.** Its place-and-route was still running when
this was written, and the cells are marked rather than filled from the trend --
an earlier version of this table carried literal `S16LUT` placeholders, which
is worse than an empty cell because it looks like data.

Latency is the first input beat to the last output beat of the first tile.
Interval is the median gap between tile completions over the last three
quarters, the definition E2 uses for its FFTs. Both systems are perfectly
steady: at every size Gemmini's minimum and maximum inter-tile gap are equal,
and SPMW's interval is exactly `41 x S`.

**Gemmini's cycles are xsim's, and the number that flatters SPMW.** There are
two harnesses here. `MxuVpuStream.scala` drives the Chisel design under
chiseltest and is the driver of record; with no verilator on the machine
chiseltest falls back to a Scala interpreter, which took 63 seconds at 4x4 and
had not finished 8x8 in forty minutes -- E1 measured six hours for a 16x16
mesh the same way. So the table quotes an xsim testbench on the emitted
Verilog, which does all three sizes in 43 seconds.

Both harnesses report `correct=true` with every output row checked, and they
disagree by a **constant one cycle per tile** at both sizes chiseltest could
finish:

| | chiseltest | xsim | rows checked |
|---|---:|---:|---|
| S=4 | latency 20, interval 5 | latency 22, interval 6 | 68/68, 0 wrong |
| S=8 | latency 36, interval 9 | latency 38, interval 10 | 136/136, 0 wrong |

The cause is visible in the waveform: the xsim driver's request handshake takes
two cycles where chiseltest's takes one, because `io_req_ready` is low for the
first cycle after a pass ends. A driver artefact either way, and reproducible
rather than noise -- +1 interval and +2 latency at both sizes.

The table takes the **larger** Gemmini interval of the two, which is the one
that makes SPMW's deficit look smaller. Quoting chiseltest instead would widen
it from 27x to 33x at S=4 and from 33x to 36x at S=8.

SPMW's LUT and FF are the `dut` instance out of `report_utilization
-hierarchical`, not the routed top. `--pnr` routes `spmw_harness`, which wraps
the array in one LFSR per channel so its stream ports stop being pins, and
Gemmini's `MxuVpu` is routed bare -- so the array-only figure is the
like-for-like one. The harness turns out not to matter: 29 LUTs and 509 FFs at
S=4, 57 and 941 at S=8.

## Where SPMW loses, and by how much

**Throughput, by a factor of 27 to 36.** Per output row SPMW spends 41 cycles
where Gemmini spends 1.5 falling to 1.125. Expressed as how busy the multiply
array is kept -- `S³` MACs per tile against `interval x S²` MAC-cycles
available:

| S | Gemmini | SPMW |
|---:|---:|---:|
| 4 | 66.7% | 2.4% |
| 8 | 80.0% | 2.4% |
| 16 | 88.9% | 2.4% |

SPMW's is `1/41` at every size, because its cost per output row does not depend
on the array at all. Gemmini's is `S/(S+2)` and improves as the array grows,
which is the more damaging half: the gap widens with size, from 27x to 36x.

**Area, by 4-5x in LUTs and about 10x in registers**, and SPMW spends DSP
blocks that Gemmini does not spend at all. The DSP count is worth reading
closely: 32 at S=4 and 96 at S=8, which is `S²` for the matrix cells' `a * wt`
plus **four per vector lane** for the `MUL` opcode -- an int32 by int32
multiply that this program never executes and still pays for in silicon. That
is the cost of a general instruction set stated in hardware rather than in
prose.

**Clock**, narrowly: +0.371 against +0.591 ns at S=4 and +0.425 against +0.456
at S=8. Both close.

BRAM is the one column that ties, and it ties at zero: the weight file here is
four 32-bit words per cell and the instruction buffer sixteen, small enough
that HLS builds both out of registers. E3's board design, whose file was 256
tiles deep, used 187 RAMB18 -- so this zero is a property of the tile size, not
of the engine.

## Why: the dispatch does not pipeline, and there is no MIN

Two loops in SPMW's design **do** reach II=1, and the reports say so rather
than the cycle counts implying it:

- `spmw/S8/report/mac_r4_0_Pipeline_VITIS_LOOP_78_4_csynth.rpt` -- the matrix
  cell's `MSWEEP` step loop: `achieved 1`, `target 1`, `Pipelined yes`. One
  activation and one partial sum per cycle.
- `spmw/S8/report/vpu_r1_0_Pipeline_VITIS_LOOP_116_8_csynth.rpt` -- the vector
  lane's `ACCN` accumulate loop: `achieved 1`, `target 1`, `Pipelined yes`. One
  partial sum folded per cycle.

The loops **above** them are the problem, and the same reports say that too:

- `vpu_r1_0_csynth.rpt`: `VITIS_LOOP_80_6_VITIS_LOOP_83_7`, the lane's
  instruction dispatch, `Latency 3 ~ 65544`, **`Pipelined no`**.
- `mac_r4_0_csynth.rpt`: `VITIS_LOOP_32_2`, the cell's instruction loop,
  `Latency 7 ~ 65546`, **`Pipelined no`**.

Neither can pipeline: each contains a nested loop whose trip count is an
instruction field. So an instruction costs about 3.6 cycles rather than one,
and the ten-instruction epilogue costs 36 of the 41.

That number is measured, not derived. `tpumicro-noclip` runs the **same
netlist** with the five instructions that spell the clip removed:

| variant | instructions | interval | cyc/out row |
|---|---:|---|---:|
| micro | 10 | 164 / 328 / 656 | 41 |
| no-clip | 5 | 92 / 184 / 368 | 23 |

Five instructions cost 18 cycles per output row -- **3.6 cycles each, and 44%
of SPMW's interval**. Those five exist because the VPU ISA has `MAX` and no
`MIN`, so `min(x, 127)` has to be spelled `127 - max(127 - x, 0)`: two
`LOADI`s, two `SUB`s and a `MAX` for something Gemmini does in a wire. Fitting
the two points gives `cycles per output row = 3.6 x instructions + 5`.

**Adding a MIN opcode would not close the gap.** Even at five instructions SPMW
would be at 23 cycles per output row against Gemmini's one. The clip is the
largest single line item and it is still less than half.

Nor would the other obvious savings. Two of the ten instructions are `LOADI
127`, and they are re-executed on every output row only because the lane's
program body has no prologue: the register file is zeroed once per launch and
then the same instructions run per row, so a constant cannot be hoisted. Giving
one register an initial value from the lane's constants would remove both -- 41
cycles becomes about 34, and 27x becomes 22x. And `--ii` does not help either,
because the dispatch loops are not pipelined at any target: they contain a
nested loop whose trip count is an instruction field, and that is a structural
refusal rather than a scheduling one.

## What this comparison does not say

The workload is the worst case for SPMW's design, and deliberately so -- it is
the workload Gemmini can do at this scope, not the one SPMW was built for. A
tile with `K = S` gives `ACCN` exactly one partial sum to fold, so the
instruction dispatch has nothing to amortise over -- 41 cycles buy one partial
sum. E3's GPT stage swept 64 tiles per instruction, and putting `plen = 4` and
`sweep = 64` through the same fit gives roughly 1.3 cycles per partial sum, a
thirtieth of the cost. That figure is arithmetic on the fit, not a measurement
of that design; what is measured is that this benchmark gives the engine
nothing to amortise over, and it was built for deep reductions.

It also measures one program on a machine that runs many. The netlist here is
`gpt_stage_v1.stage_engine` **unmodified** -- the same one that ran a GPT-2
block on the U280, softmax and all -- reprogrammed by changing sixteen words in
a stream. Gemmini's `MxuVpu` at this scope does requantise-and-ReLU and cannot
be told to do anything else. That flexibility is real and this table cannot
price it; what the table does price is what the flexibility costs when it is
not used, and the answer is 27-36x of throughput, 4-5x of LUTs, and four DSPs
per lane for a multiply the program never issues.

Finally, neither side carries DMA, a scratchpad, or a controller. These are
datapaths routed out of context.

## How it was run

    # SPMW: cycles, then area and timing (same design, two runs)
    run_spmw_micro.sh <S> cosim              # -> SPMW COSIM PASS, SPMW XFORM ...
    run_spmw_micro.sh <S> pnr                # -> util.rpt, timing.rpt, routed.dcp
    run_spmw_micro.sh <S> cosim tpumicro-noclip 4
    spmw_hier.sh <S>                         # -> util_hier.rpt, the dut/harness split

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
instead of it.

## Files

- `stimulus/` -- the shared operands and golden results, one file per size.
- `spmw/S<n>/source/` -- the design. It imports `gpt_stage_v1.stage_engine`
  from `tests/dataflow/spmw/`, unchanged; only the program is new.
- `spmw/S<n>/generated/` -- what the split backend emitted: one `.cpp` and one
  `.sv` per role, plus the fabric. 12 roles at every size.
- `spmw/S<n>/report/` -- routed utilisation, timing and route status, the
  hierarchical split, the C synthesis reports the II claims are quoted from,
  and the cosimulation's cycle lines for both variants.
- `gemmini/S<n>/source/` -- the streaming driver and the top it drives.
- `gemmini/S<n>/generated/` -- the xsim testbench, and how to re-emit the
  Verilog rather than commit 2.4 MB of it.
- `gemmini/S<n>/report/` -- per-tile completion cycles from both harnesses,
  kept in separate files so it is always clear which produced which. Area and
  timing are not repeated here: `../gemmini/report/S<n>/` already holds them
  for this exact top, and re-routing would only add placer variance between the
  two halves of one table.
- `scripts/` -- everything above, as it ran.
