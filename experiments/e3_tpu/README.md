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
TPU at the same scope, and `micro/` is the workload both systems run -- a tiled
int8 GEMM with bias, requantisation, ReLU and the clip to int8, entirely on
device, at 4x4, 8x8 and 16x16. That is where the two are set side by side; the
short version is that SPMW loses every measured column, by 27-36x on
throughput, and `micro/README.md` says why.

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

The full build tree is 3.2 GB on the machine; what is here is the subset that
reproduces the table.
