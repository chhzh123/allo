# E3: a mini-TPU block on the board

Same layout as E1 and E2 -- `<framework>/<design>/{source,generated,report}` --
with one difference this experiment forces: **E3 has one bitstream, not one per
row.** `gptkern_302` runs both workloads, and the two rows of `results.csv`
carry the same LUT, FF, DSP and slack because they are the same
place-and-route. So there is one design folder rather than one per row; a
per-workload split would duplicate one set of hardware numbers and imply two
builds that do not exist.

This is also the only experiment whose numbers come from **hardware**, not
cosimulation: an Alveo U280, through XRT.

## Results

| Workload | Launches/block | Device kernel (s) | best | Transfer (s) | Pack (s) | Host math (s) | Wall (s) | reps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2-medium, one block, seq 128 | 624 | 0.0727 | 0.0723 | 0.372 | 4.64 | 0.053 | 13.1 | 4 |
| llama-7b, one block, seq 128 | 4,064 | 0.7152 | 0.7046 | 2.382 | 28.85 | 0.039 | 138.1 | 2 |

One bitstream, routed at 3.333 ns: **132,922 LUT, 116,913 FF, 304 DSP, 187
BRAM18, 0 URAM, WNS +0.016 ns.** That slack is 0.5% of the period -- this
design only just closes timing, unlike the E1 and E2 arrays.

## Read the split before the totals

The kernel time is the device doing the matrix work. Everything else in the row
is not:

- **`pack_s` dominates the wall clock** -- 4.6 s of 13.1 s for GPT-2 and 28.9 s
  of 138.1 s for Llama, against 0.07 s and 0.72 s of actual kernel. The host
  spends 60 to 200 times longer preparing operands than the device spends
  computing on them.
- The design is **hybrid**: LayerNorm/RMSNorm, RoPE, the causal mask,
  GELU/SiLU-gate, the residual adds and every requantisation run on the host.
  The device does the matrix multiplies. Any per-block time here is a
  measurement of that division of labour, not of a whole transformer block in
  hardware.

Quoting `wall_s` as a block latency would be wrong in both directions, and
quoting `device_kernel_s` as one would be wrong in the other. The row carries
all five components for that reason.

## The comparison against Gemmini

The rows above are SPMW on its own. `gemmini/` measures Gemmini's MXU+VPU as a
TPU at the same scope -- 4x4 up to 32x32, cycles and area, with every
configuration parameter written down -- and `micro/` is the workload both
systems run: a tiled int8 GEMM with bias, requantisation, ReLU and the clip to
int8, entirely on device, one shared stimulus, bit-exact on both sides.

`micro/` runs it on **three** designs, and the answer depends on which one:

| | interval / tile (S = 4, 8, 16) | against Gemmini's `S + 2` |
|---|---|---|
| SPMW, the stage engine above | 164, 328, 656 | 27x to 36x slower |
| SPMW, fixed-function | **4, 8, 16** | **1.50x, 1.25x, 1.125x faster** |
| Gemmini MXU+VPU | 6, 10, 18 | -- |

**All of the loss is the programmability, and it is worth exactly 41x.** The
engine above spends 41 cycles an output row whatever the array size, because
its epilogue is a program and its instruction dispatch cannot pipeline. Strip
the dispatch and the interval becomes exactly `S`, which is one output row a
cycle and the best of the three. What that costs is the netlist's generality:
the fixed datapath computes one thing, where the engine above ran a whole GPT-2
block, softmax included, by changing sixteen words in a stream.

The area column does not flip. Even fixed-function, SPMW is 2.5-2.6x Gemmini's
lookup tables and 5.6-6.5x its registers, and that residue is the composition
model rather than the instruction set: a handshaked FIFO on every link where
Gemmini's systolic mesh has a bare register. `micro/README.md` has the full
account, including the sixteen-tile burst, which Gemmini still wins.

Note that `micro/` also moves the requantisation onto the device. The rows
above do it on the host, which is one of the reasons `pack_s` dominates them.

## Files

- `source/` -- the host driver (`spmw_gpt_block.py`), the stage engine and the
  XRT runner, plus how the bitstream was linked (`build.sh`, `link.cfg`,
  `kernel.xml`, `package.tcl`) and what the host was told about it
  (`args.json`).
- `generated/` -- what the split backend emitted: one `.cpp` and one `.sv` per
  role and per feeder, plus the fabric (`spmw_top.sv`, `spmw_fifo.sv`,
  `spmw_const.sv`, `spmw_harness.sv`). 16 roles.
- `report/` -- the routed timing and utilisation for the kernel, the link and
  package logs, and both board runs with their scripts and result JSON.
- `gemmini/` -- Gemmini's MXU+VPU as the baseline: the exact configuration,
  area and timing at 4x4 to 32x32 in two scale forms, and cycles.
- `micro/` -- the workload both systems run, in the
  `<framework>/<size>/{source,generated,report}` shape E1, E2 and E4 use:
  `spmw/` is the stage engine, `spmw-fixed/` the fixed-function datapath,
  `gemmini/` the baseline, `stimulus/` the operands all three are checked
  against.

The full build tree is 3.2 GB on the machine; what is here is the subset that
reproduces the table.
