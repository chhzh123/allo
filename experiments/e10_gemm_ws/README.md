# E10: an S x S weight-stationary array computing S x S x S, in three systems

E1 with the other dataflow. Every design computes `C = A B` for 8-bit signed
inputs and 32-bit signed outputs with `M = N = K = S`, on S x S cells holding
one multiplier each, as in E1 -- but here a cell holds a **weight**. Cell
(k, n) keeps `B[k][n]`, the activations cross the array along its rows, and the
partial sums run down its columns, so a row of C leaves the last row of cells
every step. In E1 a cell holds a result and both operands pass it.

`results.csv` is the table, with E1's columns; `scripts/make_results.py`
writes it from the reports in this folder.

## Layout

E1's, one directory per array size:

    <framework>/S<size>/source/      what the framework was given
    <framework>/S<size>/generated/   what it emitted
    <framework>/S<size>/report/      the reports each number was read from

| Folder | Design |
|---|---|
| `spmw_mesh/` | the SPMW array with its boundary buffers, no memory interface |
| `spmw_kernel/` | the same array as a kernel: three memory movers and a control register file |
| `autosa/` | AutoSA's generated design, its own memory interface included |
| `autosa_wide/` | the same with `--host-serialize`, which packs its ports to 512 bits |
| `allo/` | a weight-stationary array written in the style of Allo's library one, its own memory interface included |

`spmw_mesh` is the only **array-scope** row. The others are **kernel scope**:
each is routed with its own memory masters and control. That includes the SPMW
kernel, which E1's was not; the last section says what that changes.

## Results

Cycles are E1's: **first memory beat to last memory beat** of one launch, so
the control writes, the start bit and each testbench's polling are outside it.
SPMW's kernel testbench reports it; AutoSA's and Allo's come from a waveform
of their own cosimulation (`scripts/beats.py`). The figure each flow reports
itself is kept beside it. `spmw_mesh` moves no memory beats, so its cycles run
from the first input token to the last output token.

| Array | Design | Port in/out | Cycles | (flow reports) | LUT | FF | DSP | BRAM18 | Slack | Clock |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4x4 | SPMW mesh | streams | 39 |  | 1,004 | 2,157 | 16 | 0 | +1.191 ns | 467 MHz |
|  | SPMW kernel | 512/512 | 62 | 80 | 5,583 | 10,929 | 16 | 46 | +0.415 ns | 343 MHz |
|  | AutoSA | 32/32 | 108 | 184 | 6,349 | 8,112 | 16 | 7 | +0.703 ns | 380 MHz |
|  | AutoSA, wide | 128/512 | 90 | 166 | 6,812 | 9,292 | 16 | 28 | +0.402 ns | 341 MHz |
|  | Allo | 32/32 | 139 | 188 | 3,764 | 4,925 | 16 | 3 | +0.719 ns | 383 MHz |
| 8x8 | SPMW mesh | streams | 90 |  | 4,564 | 10,037 | 64 | 0 | +0.971 ns | 423 MHz |
|  | SPMW kernel | 512/512 | 123 | 142 | 9,217 | 19,602 | 64 | 46 | +0.397 ns | 341 MHz |
|  | AutoSA | 32/32 | 336 | 409 | 17,992 | 20,730 | 64 | 11 | +0.684 ns | 378 MHz |
|  | AutoSA, wide | 512/512 | 216 | 292 | 19,759 | 23,491 | 64 | 54 | +0.566 ns | 361 MHz |
|  | Allo | 32/32 | 333 | 382 | 9,182 | 10,003 | 64 | 3 | +0.435 ns | 345 MHz |
| 16x16 | SPMW mesh | streams | 195 |  | 18,794 | 43,205 | 256 | 0 | +0.570 ns | 362 MHz |
|  | SPMW kernel | 512/512 | 227 | 246 | 24,346 | 53,156 | 256 | 46 | +0.494 ns | 352 MHz |
|  | AutoSA | 32/32 | 1,239 | 1,318 | 50,462 | 59,327 | 256 | 60 | +0.617 ns | 368 MHz |
|  | AutoSA, wide | 512/512 | 691 | 769 | 52,179 | 62,190 | 256 | 103 | +0.518 ns | 355 MHz |
|  | Allo | 32/32 | 1,021 | 1,070 | 36,022 | 30,633 | 256 | 3 | +0.158 ns | 315 MHz |
| 32x32 | SPMW mesh | streams | 423 |  | 79,305 | 179,557 | 1,024 | 0 | +0.500 ns | 353 MHz |
|  | SPMW kernel | 512/512 | 495 | 514 | 87,050 | 192,131 | 1,024 | 46 | +0.276 ns | 327 MHz |
|  | AutoSA | 32/32 | 4,743 | 4,819 | 181,372 | 219,076 | 1,024 | 149 | +0.031 ns | 303 MHz |
|  | AutoSA, wide | 512/512 | 2,371 | 2,443 | 183,237 | 222,028 | 1,024 | 193 | +0.168 ns | 316 MHz |
|  | Allo |  |  |  |  |  |  |  |  |  |

Every row is routed out of context at 3.333 ns on `xcu280-fsvh2892-2L-e` with
nothing unrouted; slack is against that target and the clock is the period it
implies. Port width is read off each design's synthesised RTL
(`report/port_widths.txt`). BRAM18 is twice the block RAM tiles. The blank
row is Allo's 32x32 array, which did not get through synthesis (below);
nothing in the table is an estimate.

AutoSA's two 32x32 arrays took 9.6 hours to synthesise and about five more to
cosimulate. In both, the simulation's deadlock monitor reports one of the 32
modules that collect C part-way through the run, at 9.4 of 16.2 microseconds
as generated and 7.4 of 8.3 with wide ports. The run is not stopped: every
result is written, the testbench's check of them passes, and the tool's
verdict is a pass with `HLS 200-742` logged beside it
(`autosa*/S32/report/deadlock_monitor.txt`, `cosim_verdict.txt`). The monitor
reports nothing at the smaller sizes.

Three things to take from it.

**SPMW's kernel finishes first at every size, and the gap widens.** 62 cycles
at 4x4 against 90 to 139; 123 at 8x8 against AutoSA's 216 with matched 512-bit
ports, 336 as generated, and Allo's 333; 227 at 16x16 against 691, 1,239 and
1,021; 495 at 32x32 against 2,371 and 4,743. At the clocks each routed to,
the 16x16 launch takes 0.64 microseconds in SPMW and 1.9, 3.4 and 3.2 in the
other three, and the 32x32 launch 1.5 against AutoSA's 7.5 and 15.7.

**In area it depends on the size and on the column.** Three 512-bit memory
masters and the control interface are a fixed 3,300 lookup tables, 5,400
registers and 46 block RAMs whatever the array, which at 4x4 is more than the
array they serve; Allo's kernel is the smallest there. By 8x8 the SPMW kernel
has Allo's lookup tables (9,217 against 9,182) and half of AutoSA's; at 16x16
it has two thirds of Allo's (24,346 against 36,022) and under half of AutoSA's
(50,462, or 52,179 wide); at 32x32 it again has under half of AutoSA's (87,050
against 181,372). In registers it is close to AutoSA from 8x8 up (19,602
against 20,730, 53,156 against 59,327, 192,131 against 219,076) and 1.7 to 2.2
times Allo, because it keeps every link between two cells in a register slice
where Allo keeps a shift-register FIFO (`gemm_fifo_w8_d24_S_ShiftReg` and its
like). Its 46 block RAMs are the masters and do not grow. AutoSA's do -- 7,
11, 60 and 149 as generated -- and at 16x16 they are 16 in its cells, each of
which declares one, and 41 in the modules that distribute A and collect C.

**It does not have the fastest clock until the array is large.** SPMW's mesh
alone closes at 467, 423, 362 and 353 MHz, on a link into a cell's multiplier
at the first three sizes and in the harness's own stimulus generator at the
last. But every kernel's worst path is control, and SPMW's -- in its control
wrapper or a mover, and at 32x32 in a feeder's check that all 32 edge buffers
have room -- holds it to 343, 341, 352 and 327 MHz. That is below AutoSA as
generated up to 16x16 (380, 378, 368) and above it at 32x32 (303, or 316
wide), and below Allo at the two smaller sizes (383, 345, 315). AutoSA's worst
path is the collection of its modules' done signals, Allo's the handshakes of
its tile's FIFOs.

## The three designs

### SPMW

`spmw_mesh/S*/source/test_spmw_gemm_ws.py`; the suite runs it against numpy on
the reference and the simulator, and both RTL simulations here are fed
operands that span int8, with a row and a column of each extreme
(`spmw_operands`; the `RESIM` lines in the build logs are the reruns that
switched to them, on unchanged hardware). A cell, and the array:

```python
@spmw.unit
def mac(io: WsIO, site: spmw.Site):
    row, _col = site.rank
    below: uint8 = n - 1 - row  # sized: a bare `row` counts in 32 bits
    w: int8 = io.w_in.get()  # mine
    for _r in range(below):  # the rest are for the rows below me
        io.w_out.put(io.w_in.get())
    for m in range(n):
        a = io.a_in.get()
        p = io.p_in.get()
        io.p_out.put(p + a * w)
        io.a_out.put(a)

@spmw.fabric
def g(A: int8[n, n], B: int8[n, n], C: int32[n, n]):
    P = spmw.place(mac, on=mesh)
    spmw.stream_in(B, into=P.w_in, index=(..., P.cols))  # column c: B[:, c]
    spmw.stream_in(A, into=P.a_in, index=(..., P.rows))  # row k: A[:, k]
    spmw.stream_in(0, into=P.p_in)  # every column's sum starts at 0
    spmw.gather(C, from_=P.p_out, index=(..., P.cols))  # column c: C[:, c]
```

The weights reach their cells down their own column, once: a cell keeps the
first it sees and hands the rest on. Nine roles at any size. Nothing has to be
laid out for the array -- a row of A is a step of activations, a row of B a
step of weights, a step of results a row of C -- so its edge is three bindings,
and packaged as a kernel it is three movers, each carrying a row a step on a
512-bit port.

A weight-stationary array's wavefront is a diagonal: row k wants its
activations k cells later than row 0, and column c's results leave c cells
after column 0's. Something has to hold that skew. Here it is the edge
streams, which the packaging already makes a launch deep (`spmw_fifo
#(.DEPTH(8))` on each of the 24 edge channels at 8x8, against `DEPTH(2)` on
the links between cells; `spmw_kernel/S8/generated/`). The first version
of this kernel copied E1's shape instead, a chain of feeder units in front of
the array on two-deep links, and ran at one row every S cycles or so: 357
cycles at 16x16 where this one takes 227.

### AutoSA

`autosa/S*/source/autosa_command.txt` is the exact command. It is E1's with
the two changes AutoSA's own documentation makes for this array
(`docs/examples/mm.rst`, "Array 6: [j,k]"):

    --sa-sizes="{kernel[]->space_time[5];kernel[]->array_part[S,S,S];kernel[]->latency[1,1];kernel[]->simd[1]}"
    --local-reduce --reduce-op="+" --array-contraction

`space_time[5]` takes loops j and k as the space loops, so `B[k][j]` never
leaves its cell, A crosses the array and C is summed along it; `--local-reduce
--reduce-op="+"` tells the compiler the design reduces along its cells, which
it cannot infer. Two things about those flags are worth knowing before
repeating this:

- **Without them the generated design does not synthesise here.** Every cell
  gets the same three result streams and decides inside, from its row number,
  which it uses, so S(S+1) streams are left with one end. Vitis HLS 2023.2
  stops with `HLS 200-970`.
- **With the documentation's third flag, `--simd-touch-space`, the generator
  writes no design**: at `simd[1]`, E1's setting, it exits 0 after its
  vectorisation analysis and leaves stub files. The flag is for vectorising
  along a space loop and is left out.

What comes out at 8x8 is 64 cells, 64 modules that each hold one cell's
weight and send it to the cell again every step (`B_IO_L1_in`), 8 + 8 modules
distributing A and B, 8 collecting C, three memory readers and writers, and 16
boundary stubs. Each cell also declares `local_C[S][S]` with a block RAM
pragma (`kernel_kernel.cpp`, `void PE`); 8 block RAMs remain in the 64 cells
after routing, 16 in the 256 at 16x16.

As generated, AutoSA packs nothing here: its ports are `A_t1`, `B_t1`, `C_t1`,
one element a beat. `--host-serialize` (`autosa_wide/`) packs them to 512
bits, the same six beats SPMW moves at 8x8, and unlike E1's array it also
generates at 4x4, where a whole 16-byte operand is one 128-bit beat.

### Allo

Allo's library has one systolic array (`allo/library/systolic.py`) and it is
output-stationary. `scripts/ws_allo_gen.py` is the weight-stationary one,
written against the same API the same way, and its docstring sets the two
side by side: the cell reads its weight once and adds into a sum that arrives
and leaves; the buffers between cells carry A along one axis and the partial
sum along the other; the tile's loader writes A and a zero sum and its drain
stores C; the schedule partitions, pipelines, unfolds the cell grid and turns
the buffers into FIFOs with the library's own calls. Before it emits anything
the generator runs the design on Allo's LLVM backend against numpy, and after
it checks the emitted code has S x S cell functions that each read their
weight once, outside the loop. C simulation runs three seeds; the
cosimulation that is timed runs one launch.

Vitis HLS gives each cell's weight a channel of its own: at 16x16 the tile
holds 256 `gemm_fifo_w8_d*_S` FIFOs for the weights beside the 272 that carry
A and the 272 that carry the partial sums.

**At 32x32 the design did not get through `csynth_design` in four hours**
(`allo/S32/report/build_4h.log`, `hls rc=124`). It stopped in the tool's "Loop,
function and other optimizations" phase, which takes 0.9, 6.4 and 134 seconds
at the three smaller sizes (`allo/S*/report/hls_phases.txt`) and took 870 for
E1's 32x32 output-stationary array. It passes the LLVM check and C simulation
at that size. A second run, given a day, got through that phase in 42,884
seconds -- 11.9 hours, 320 times what 16x16 takes -- and had scheduled 952 of
its modules when it was stopped in error 25 minutes later
(`allo/S32/report/hls_phases_stopped.txt`). A third was started at once; the
row's cells are blank until it ends.

How a cell gets its weight is not what holds the tool there.
`scripts/ws_probe2.py` is the same array with the tile copying B into a grid
of FIFOs and each cell reading its own, in place of every cell being handed B
and its indices. That phase takes 114 seconds in it at 16x16
(`allo/S16/report/variant_fifo_phases.txt`), and at 32x32 it was stopped
after three and a half hours without leaving it
(`allo/S32/report/variant_fifo.txt`, `rc=124`). The variant was synthesised
and nothing else; no row of the table is from it.

## The element is identical in all three

The loop that has to pipeline is the same loop, and the tool pipelines it the
same way. At 8x8:

| System | Loop | Latency | Iteration latency | Interval | Trip |
|---|---|---:|---:|---:|---:|
| SPMW | `l_S_m_1_m` | 11 | 5 | 1 | 8 |
| AutoSA | `VITIS_LOOP_630_1` | 11 | 5 | 1 | 8 |
| Allo | `l_step_m9` | 11 | 5 | 1 | 8 |

SPMW's and Allo's are the same hardware to the register: 165 flip-flops, 136
lookup tables and one DSP block each (`report/mac_r4_0_Pipeline_l_S_m_1_m_csynth.rpt`,
`report/PE_kernel_ws_gemm_1_1_Pipeline_l_step_m9_csynth.rpt`). So nothing in
the table comes from the arithmetic.

## Where the differences come from

### The memory port

Beats moved for A, B and C, counted off each cosimulation:

| Array | SPMW kernel | AutoSA | AutoSA, wide | Allo |
|---|---:|---:|---:|---:|
| 4x4 | 1 + 1 + 1 = 3 | 16 + 16 + 16 = 48 | 1 + 1 + 1 = 3 | 4 + 4 + 16 = 24 |
| 8x8 | 1 + 1 + 4 = 6 | 64 + 64 + 64 = 192 | 1 + 1 + 4 = 6 | 16 + 16 + 64 = 96 |
| 16x16 | 4 + 4 + 16 = 24 | 256 + 256 + 256 = 768 | 4 + 4 + 16 = 24 | 64 + 64 + 256 = 384 |
| 32x32 | 16 + 16 + 64 = 96 | 1024 + 1024 + 1024 = 3,072 | 16 + 16 + 64 = 96 |  |

SPMW's counts are its port's arithmetic, rows of bytes or of 32-bit sums at
512 bits a beat. AutoSA as generated moves one element a beat on every port,
which is narrower than it was in E1, where the same generator packed a row.
Allo moves four bytes a beat in and one result a beat out, as in E1, and as
in E1 it loads its operands one after the other: at 16x16 A's beats land at
cycles 79 to 142 and B's at 349 to 412.

### What sits between the port and the array

With the ports matched, wide AutoSA and SPMW move the same six beats at 8x8
and differ by 93 cycles. AutoSA's two operand beats arrive at cycle 93 and its
first result beat leaves at 305; SPMW's whole launch, first operand beat to
last result beat, is 123. At 16x16 the same 24 beats take 691 cycles and 227,
and at 32x32 the same 96 take 2,371 and 495. The array is the same grid of
cells with the same loop, so the difference is the network: AutoSA's operands
cross a reader, a chain of distributors and, for B, a chain of weight holders,
each of which keeps its own share of the tile in a buffer before sending it
on, and its results cross a chain of collectors that do the same. SPMW's
movers write straight into the array's edge streams.

### Where the area is

`scripts/breakdown.py` splits each routed kernel three ways from its
hierarchical utilisation: the **array** (cells, the links between them, and
for AutoSA the module that holds each cell's weight), the **network** between
the memory masters and the array, and the **memory** masters with the control
interface.

| Array | Design | part | LUT | FF | BRAM18 | DSP |
|---|---|---|---:|---:|---:|---:|
| 8x8 | SPMW kernel | array | 4,427 | 9,800 | 0 | 64 |
|  |  | network | 1,591 | 4,480 | 0 | 0 |
|  |  | memory | 3,202 | 5,322 | 46 | 0 |
|  | AutoSA | array | 10,641 | 12,386 | 8 | 64 |
|  |  | network | 5,808 | 5,557 | 0 | 0 |
|  |  | memory | 1,893 | 2,787 | 3 | 0 |
|  | AutoSA, wide | array | 10,736 | 12,388 | 8 | 64 |
|  |  | network | 5,953 | 5,928 | 0 | 0 |
|  |  | memory | 3,408 | 5,175 | 46 | 0 |
|  | Allo | array | 6,690 | 5,585 | 0 | 64 |
|  |  | network | 533 | 1,631 | 0 | 0 |
|  |  | memory | 1,959 | 2,787 | 3 | 0 |
| 16x16 | SPMW kernel | array | 18,564 | 42,776 | 0 | 256 |
|  |  | network | 2,472 | 4,953 | 0 | 0 |
|  |  | memory | 3,336 | 5,427 | 46 | 0 |
|  | AutoSA | array | 42,951 | 50,269 | 16 | 256 |
|  |  | network | 6,794 | 6,271 | 41 | 0 |
|  |  | memory | 1,926 | 2,787 | 3 | 0 |
|  | AutoSA, wide | array | 42,885 | 50,264 | 16 | 256 |
|  |  | network | 7,098 | 6,752 | 41 | 0 |
|  |  | memory | 3,403 | 5,174 | 46 | 0 |
|  | Allo | array | 33,058 | 23,940 | 0 | 256 |
|  |  | network | 993 | 3,906 | 0 | 0 |
|  |  | memory | 1,971 | 2,787 | 3 | 0 |

- **A 512-bit master costs the same whoever instantiates it.** SPMW's three
  and wide AutoSA's three are within 6 per cent of each other -- 3,202 and
  3,408 lookup tables at 8x8, 5,322 and 5,175 registers, 46 block RAMs each --
  and neither grows with the array. The 32-bit ones are 1,900 lookup tables,
  2,800 registers and 3 block RAMs in AutoSA and in Allo alike.
- **Allo's network is the smallest by far**, a loader and a store loop. It is
  also why Allo is slow: those loops move one tile element at a time.
- **SPMW's array has the fewest lookup tables and AutoSA's the most**: 4,427,
  6,690 and 10,641 at 8x8; 18,564, 33,058 and 42,951 at 16x16. In registers
  Allo's is the smallest and the other two keep their order: 23,940, 42,776
  and 50,269 at 16x16.

## Weight-stationary against output-stationary

The same systems on the same workload in E1, cycles first beat to last:

| Array | SPMW mesh | SPMW kernel | AutoSA | AutoSA, wide | Allo |
|---|---:|---:|---:|---:|---:|
| 4x4 | 16 -> 39 | 47 -> 62 | 78 -> 108 | -- -> 90 | 118 -> 139 |
| 8x8 | 28 -> 90 | 82 -> 123 | 222 -> 336 | 130 -> 216 | 288 -> 333 |
| 16x16 | 52 -> 195 | 133 -> 227 | 782 -> 1,239 | 374 -> 691 | 928 -> 1,021 |
| 32x32 | 100 -> 423 | 280 -> 495 | 2,990 -> 4,743 | 1,222 -> 2,371 | 3,408 -> -- |

Every system is slower here, and SPMW's array the most: 39 cycles against 16
at 4x4, 423 against 100 at 32x32. That is the dataflow and the tool together.
In the output-stationary array an operand hops from cell to cell in one cycle
and a result is finished after K steps wherever its cell is. In this one a
result passes through all S cells of its column, and each is the five-stage
multiply-add above. An 8x8 run with a monitor on every link shows the
schedule: the top row reads its first activation at cycle 17, once it has
passed the last weight down; the wavefront then advances between four and five
cycles a row and the same a column; and the last of the 8 rows of results
leaves at cycle 90.

The column hop is the arithmetic. The row hop need not be -- the activation
is only forwarded -- but Vitis HLS schedules every stream write of a loop
iteration in one state, the last one, so the activation waits for the
multiply-add it took no part in. Writing the forward first in the unit does
not change the schedule. E1's cell forwards in one cycle because nothing later
is written in its loop.

Allo moves least, 333 against 288 at 8x8, because its time is its memory
interface in both.

The workload is one S-cubed tile, which is the case that suits a
weight-stationary array least: the weights are loaded for S rows of
activations and then thrown away. Nothing here measures the case it is built
for, many rows through one set of weights; E3's engine does.

## What "kernel" includes here, and what E1's did not

The SPMW kernel row above is the packaged kernel -- the nine roles, the three
movers with their AXI masters, the control register file -- routed out of
context by the recipe the AutoSA and Allo rows use (`report/pnr.tcl`).

E1's `spmw_kernel` row was not. Its cycles came from the packaged kernel's
simulation, but its area, block RAM and clock came from
`spmw_build_array.py --pnr` of the array and its distribution chains, with no
mover and no control in the route. Routed the way this folder's kernel is,
E1's kernel is:

| Array | E1's table | routed with its movers and control |
|---|---|---|
| 4x4 | 1,867 LUT, 2,516 FF, 0 BRAM18, 492 MHz | 6,548 LUT, 11,104 FF, 46 BRAM18, 331 MHz |
| 8x8 | 8,015 LUT, 10,220 FF, 0 BRAM18, 476 MHz | 12,833 LUT, 19,944 FF, 46 BRAM18, 350 MHz |
| 16x16 | 32,756 LUT, 40,732 FF, 0 BRAM18, 358 MHz | 38,507 LUT, 53,682 FF, 46 BRAM18, 353 MHz |
| 32x32 | 137,741 LUT, 163,708 FF, 0 BRAM18, 357 MHz | 147,551 LUT, 193,953 FF, 46 BRAM18, 330 MHz |

So E1's SPMW kernel row is an array-scope figure in three of its columns. Its
"0" block RAM against wide AutoSA's 46 was the movers missing, not staging
AutoSA needs and SPMW does not: the two have the same 46. Its 476 MHz against
wide AutoSA's 385 at 8x8 was the array's clock against a kernel's: with its
masters the SPMW kernel closes at 350 there, and at 353 and 330 against wide
AutoSA's 339 and 309 at the two larger sizes. The lookup-table and register
gaps to wide AutoSA shrink from 1.9x and 2.5x to 1.2x and 1.3x at 8x8, and
from 1.4x and 2.0x to 1.3x and 1.7x at 32x32. The cycle counts in E1 are
unaffected. `experiments/e1_gemm/README.md` carries the correction, and the
reports are in `experiments/e1_gemm/spmw_kernel/S*/report/full_kernel/`.

## Reproducing

Everything was built on brg-zhang-xcel with Vitis HLS and Vivado 2023.2.

    scripts/spmw_ws.sh <S>          # SPMW mesh: cosimulation, then route
    scripts/spmw_ws_kernel.sh <S>   # SPMW kernel: package, simulate, route
    scripts/spmw_ws_kernel.sh <S> 4 autosa e1kernel   # E1's kernel, routed whole
    scripts/autosa_ws.sh <S> narrow|wide
    scripts/autosa_ws_tail.sh 32 narrow|wide <pid>   # 32x32 only: routes beside the cosimulation
    scripts/allo_ws.sh <S>          # runs ws_allo_gen.py first
    scripts/probe32.sh              # the FIFO-weight variant of Allo's array, csynth at 32x32
    scripts/collect_ws.sh           # stages this folder's shape from the builds
    python3 scripts/make_results.py # results.csv and the table above
    python3 scripts/breakdown.py    # the area split
