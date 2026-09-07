# E3: complete transformer blocks on the mini-TPU stage engine (board)

Status: **pass** for both blocks (every device launch checked against the
integer reference; 0 mismatched values in every repetition). The blocks are
**hybrid**: the GEMMs and the three softmax passes run on the device, the
rest on the host (see "What runs where").

## Hardware

- Board: Alveo U280 (xcu280-fsvh2892-2L-e) on brg-zhang-xcel, XRT via pyxrt.
- Bitstream: `gptkern_302` = `spmw_build_array.py --design gptstage_v1 --size 16`
  packaged by `spmw_package_kernel.py`, linked by v++ (Vitis/Vivado 2023.2)
  at 300 MHz with the ExtraTimingOpt placement. The design of record from
  SPMW_EXPERIMENTS.md ("302 on the board"); its source is
  `tests/dataflow/spmw/gpt_stage_v1.py` (16x16 int8 array, 256-entry weight
  file per cell, one ISA lane per column, three softmax passes as lane
  programs).
- Kernel resources, routed, kernel hierarchy only (`reports/gptkern_302_util_kernel.rpt`,
  `reports/gptkern_302_timing_summary_head.rpt`):

  | LUT | FF | DSP | BRAM36 | BRAM18 | BRAM 18k-equiv | URAM | WNS | TNS | WHS |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 132,922 | 116,913 | 304 | 75 | 37 | 187 | 0 | +0.016 ns | 0 | +0.006 ns |

  Timing met at 3.333 ns (all clocks). The Vitis shell's own logic is excluded.

## Blocks

Synthetic weights (seed 0); one block = one decoder layer with prefill of
a 128-token sequence. `scripts/spmw_gpt_block.py` drives the device through
`scripts/spmw_xrt_runner.py` (a persistent pyxrt process on the system
Python; one run handle per launch shape, restarted, as the walker's timed loop
does).

| model | hidden | heads x head | FFN | norm | extras | launches / block |
|---|---:|---:|---:|---|---|---:|
| GPT-2 medium | 1024 | 16 x 64 | 4096 (GELU) | LayerNorm | causal mask | 624 |
| LLaMA-7B | 4096 | 32 x 128 | 11008 (SiLU gate) | RMSNorm | RoPE, causal mask | 4064 |

Launch shapes (all on the 302 bitstream, unchanged): `proj` (K=1024, 4 slabs
x 128 rows = 512 outputs of 16 columns), `ffn2` (K=4096, 1 slab), `score`
(K=64 per head, 64 queries a launch), `ctx` (K=128), and the three softmax
lane passes per (head, 16-query group). LLaMA's wider shapes are decomposed
on the host: K > 1024 projections as K/1024 launches whose partial sums
are added and shifted once (frozen integer semantics), head width 128 as
two 64-column launches, K=11008 FFN2 as three `ffn2` launches (4096, 4096,
2816).

## What runs where

Device: Q/K/V/O projections, scores, the three softmax passes, context,
FFN1, FFN3 (gate), FFN2. Host (numpy, float32 then requantised to int8):
LayerNorm / RMSNorm, RoPE, causal mask, GELU (tanh form) / SiLU gate,
residual adds, all requantisation (`>> 4` after projections, the softmax
probability scale PROB_BITS=6), and the packing of every launch's operands
into the kernel's memory layout (`pack`). The integer semantics are those
of `gpt_stage_v1.attention_head_ref` (QUANT_SCORE=6, EXP_SHIFT=5,
EXP_BASE=8, PROB_BITS=6, RCP_BITS=14).

## Latency breakdown (means over repetitions; `stage_tables.md` has every stage)

| block | reps | device kernel | host<->device transfers | operand packing (host) | host math | wall |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 medium, one layer | 4 | 72.7 ms (best 72.3) | 371.8 ms | 4.64 s | 53.0 ms | 13.08 s |
| LLaMA-7B, one layer | 2 | 715.2 ms (best 704.6) | 2381.5 ms | 28.85 s | 38.8 ms | 138.1 s |

Definitions: *device kernel* = sum over launches of the time from
`run.start()` to `wait()` returning (dispatch, execution and completion
notification of one launch at a time; nothing is pipelined across launches);
*transfers* = the host->device writes and syncs of each launch's operand
buffers plus the drain sync back; *packing* = numpy assembly of each launch's
operand buffers; *host math* = the normalisations, activations, residuals and
requantisation; *wall* = the whole block including the reference computation
and the mismatch check of every launch, file I/O to the runner process, and
the Python driver -- it is a validation harness's wall time, not a serving
number.

Per launch, the device cost is 216 us for a `proj` launch (109 us of array
work at 300 MHz, 64 cycles a row for 512 rows), 190 us for `ffn2`, 100 us for
`score`, 107 us for `ctx` and 67-80 us for a softmax pass (0.4 us of array
work: launch overhead almost entirely). The walker's back-to-back timing of
the same shapes (SPMW_EXPERIMENTS.md: 55.4 ms a GPT-2 medium layer) issues a
hundred launches of one shape with no buffer syncs in between; the chained
block interleaves a write, a launch and a drain read, which is where the
72.7 vs 55.4 ms sits (the softmax passes at 67-80 us against 42-56 us).

The GPT-2 medium layer's device time is 30% of the block's transfer time and
1.6% of its packing time: on this bitstream the complete block is bound by
the host-side data movement of a design whose launches are small (a launch
carries at most 512 output rows), not by the array.

## Validation

Every launch's device output is compared with the integer reference for that
launch (`stage_operands(...)[-1]`) before it is consumed by the next stage;
`mismatched_values` is the total over the block. The device-free reference
run (`--no-device`, `reports/e3_llama_ref.log`) exercises the same
decomposition with the reference in place of the device. GPT-2 medium: 4
repetitions on the device (`reports/e3_board2.log`, plus the first two runs
`e3_block_dev.log`, `e3_block_timed.log` made before the runner reused its
run handle: same 0 mismatches, 141-151 ms of device time). LLaMA-7B: 2
repetitions (`reports/e3_board2.log`).

## Caveats

- Hybrid: the host does the non-GEMM math and all layout work. No number
  here is an end-to-end serving latency.
- One layer, prefill of 128 tokens, synthetic int8 weights and activations;
  no KV cache, no decode phase.
- LLaMA-7B's block is run at the 302 bitstream's shapes; the head width and
  K decomposition are host-side loops, so the launch count (4064) is what a
  16x16 engine with this launch set needs, not a design tuned for LLaMA.
- Softmax launch overhead dominates the device time of both blocks (the
  three passes are 39% of GPT-2's device time at 0.4 us of array work each);
  the batched lane (`gptkern_v8b`) is the fix on the design side and was not
  used here.

## Files

- `results.csv`, `results.json`: one row per block run; `stage_tables.md`:
  per-stage means.
- `reports/*_results.json`: the raw per-repetition, per-stage timings;
  `reports/*.log` and `reports/*.sh`: the runs and the scripts that made them.
- Sources: `scripts/spmw_gpt_block.py`, `scripts/spmw_xrt_runner.py`,
  `tests/dataflow/spmw/gpt_stage_v1.py` at the commit in `../sources/commit.txt`
  (the two scripts are new in this evaluation and are committed with it).
