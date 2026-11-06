# Language-aware source recount (E7)

Counter: `/Users/chhzh123/Desktop/projects/allo/.claude/worktrees/spmw-allo-implementation-99c949/scripts/spmw_loc2.py` (sha256 `68a47ad0a17aa8a52a9c4711b798bbfb6331bddee3f4ddb94f1164aa932c4134`), spmw_loc2 2026-09-06. Manifest sha256 `2365f0cee0daf0a41d7c12114927e3ec993776ecf7a2cee98ff893077070a26c`.

Rule: cloc's -- a physical line counts when it holds code; comment-only lines, docstrings and standalone string statements do not; code with a trailing comment does. Python via `tokenize`+`ast`, C/C++ and SystemVerilog via a comment- and string-aware stripper, TeX listings extracted first. `design` = PE/unit bodies, interfaces, topology/link rules, loaders/drains/boundary bindings and design-specific interface code; `config` = imports/includes, constants, type aliases and sizes; `workload` = programs and their encoders; `test` = tests, references, operand generators and host code (reported, not counted). The headline columns are `design` only. Source size is not development time.

## Provenance

- Allo worktree at commit d7ce0da49d0da574a36b51507a1253ce11dd6bec (branch hc/spmw-allo-implementation-99c949); none of the counted files is modified in the working tree (git status at count time). Paper workspace: /Users/chhzh123/Desktop/projects/spmw/spmw_asplos27.
- Archived provisional numbers: data/evidence/table5.json and tables/loc.tex (HLS 99/117/113/-/117/107, SPMW 32/57/33/127/40/55), produced by data/evidence/source/scripts/spmw_loc.py (identical to the worktree's scripts/spmw_loc.py) on the HLS baselines at commit 4f5f283 (3 Sep 2026) and the 27 Aug version (99b7ddd) of test_spmw_daisy.py; re-running that counter on today's files gives HLS 99/117/160/-/120/122 and SPMW 32/64/33/127/40/55.
- Why the archived SPMW numbers were too high: spmw_loc.py::region() writes the selected lines to <file>.py.region and loc() dispatches on the .py suffix, so every SPMW region was counted with the C/C++ rule -- docstrings and #-comment-only lines counted as code, and the regions ran up to the first test, taking instantiations and operand generators with them. (Its Python counter had a second latent bug, unused for the table: the interior lines of multi-line docstrings counted because the drop check only recognised lines that start with quotes or '#'.)
- Why some archived HLS numbers are lower than today's: three baselines were rewritten on 3 Sep after the archive (8ccf7fa: gemm_tiled.cpp float32 -> int8 with fan-out loaders, attention_pv.cpp split PEs and an array_pass function; 5e1a84b: a silently ignored pragma; 2fcf5bd: gemm_systolic.cpp seed folded into pe_top to fix a hardware deadlock). gemm_output_stationary.cpp and gemm_multicache.cpp are unchanged.
- Categories on the HLS side: includes, #defines and typedefs are config (4-10 lines per file); everything else in a baseline file is design (PEs, loaders, sinks, drains, the dataflow region and the extern C top with its AXI pragmas). On the SPMW side the two allo imports, size constants and module-level instantiations are config. The archived HLS numbers were whole files, i.e. design + config.
- Local run on the Mac only (Python 3.14.7; the counter and its pytest use the standard library and pytest only); no hardware tools, no remote host, nothing outside scripts/spmw_loc2.py, tests/dataflow/spmw/loc_fixtures/, tests/dataflow/spmw/test_spmw_loc2.py and this directory was written.

## Design-only counts, with the archived provisional numbers

| Design | HLS | SPMW | HLS/SPMW | archived HLS | archived SPMW | archived ratio | note |
|---|---:|---:|---:|---:|---:|---:|---|
| Systolic GEMM (output-stationary, int8 into int32) | 94 | 24 | 3.92x | 99 | 32 | 3.09x | HLS file unchanged: 99 = 94 design + 5 config (includes, DIM, typedefs). SPMW 32 was the region gemm_int8_of..test_ under the C rule: 2 docstring lines, the `gemm_int8 = gemm_int8_of(N)` instantiation (config here) and the 5-line `_operands` generator; 32 - 2 - 1 - 5 = 24. |
| Multi-cache GEMM (daisy-chained column drain, int16) | 113 | 40 | 2.83x | 117 | 57 | 2.05x | HLS unchanged: 117 = 113 + 4 config. SPMW 57 was the 27 Aug file (99b7ddd) under the C rule: 1 docstring + 9 comment-only lines, the `g.spmw_parts` test hook, the instantiation, the parts-unpacking line and the 5-line `_operands` generator; 57 - 17 = 39 for daisy_of then, and the 4 Sep commit 2474283 (operand/accumulator type parameters) makes it 40 today. |
| Tiled GEMM (two-level: tile engines of PE meshes) | 153 | 32 | 4.78x | 113 | 33 | 3.42x | HLS rewritten after the archive (8ccf7fa, 3 Sep: float32 -> int8/int32, fan-out loaders and a storer because one AXI bundle may be read by one dataflow process): 113 (108 design + 5 config) became 160 (153 + 7). SPMW 33 - 1 docstring line = 32. The sides now differ in datatype and in how tiles are fed (loaders vs memory-fed children). |
| FFT butterfly network (spatial one-shot + streaming batch) | -- | 103 | -- | -- | 127 | -- | Archived SPMW 127 (region bfly_pair.._operands under the C rule) included 3 docstring lines, 18 comment-only lines (the streaming-form banner and binding comments), BATCH (config) and bitrev_perm (the host-side input permutation): 127 - 24 = 103 = 58 spatial + 45 streaming. Archived HLS 407 was an external HP-FFT-HLS checkout, absent locally and a different (folded) design: unfilled. |
| Mini-TPU matrix unit (weight-stationary MXU + activation row) | 113 | 40 | 2.83x | 117 | 40 | 2.92x | HLS changed after the archive (2fcf5bd, 3 Sep: the seed process folded into a pe_top row and deeper streams, fixing a deadlock on hardware): 117 (110 design + 7 config) became 120 (113 + 7). SPMW 40 unchanged: the archived region had no docstring or comment-only lines. Scope caveat: the HLS kernel also tiles over `tiles` with a per-tile weight reload and packs 16 int8 per 128-bit beat; tpu_matmul is one tile pass with scalar ports. |
| Mini-TPU, complete stage engine (instruction-driven MXU cells with weight files, fold units, vector lanes) | -- | 267 | -- | -- | -- | -- |  |
| FEATHER (NEST weight-file mesh + BIRRD butterfly reduction/reorder) | -- | 97 | -- | -- | -- | -- |  |
| Grouped attention-PV (the matrix unit adapted: column slabs, serpentine psum chain) | 112 | 53 | 2.11x | 107 | 55 | 1.95x | HLS changed after the archive (8ccf7fa: pe split into pe_forward/pe_edge and the array moved into array_pass because a DATAFLOW pragma in a bare block was silently ignored; 5e1a84b): 107 (97 design + 10 config) became 122 (112 + 10). SPMW 55 - 2 docstring lines = 53. |

## Breakdown by category (HLS / SPMW)

| Design | design | config | workload | generated | test+host (excluded) | design+config | ratio (design+config) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Systolic GEMM (output-stationary, int8 into int32) | 94 / 24 | 5 / 4 | 0 / 0 | 0 / 0 | 0 / 53 | 99 / 28 | 3.54x |
| Multi-cache GEMM (daisy-chained column drain, int16) | 113 / 40 | 4 / 4 | 0 / 0 | 0 / 0 | 0 / 34 | 117 / 44 | 2.66x |
| Tiled GEMM (two-level: tile engines of PE meshes) | 153 / 32 | 7 / 5 | 0 / 0 | 0 / 0 | 0 / 47 | 160 / 37 | 4.32x |
| FFT butterfly network (spatial one-shot + streaming batch) | -- / 103 | -- / 7 | -- / 0 | -- / 0 | -- / 150 | -- / 110 | -- |
| Mini-TPU matrix unit (weight-stationary MXU + activation row) | 113 / 40 | 7 / 4 | 0 / 0 | 0 / 0 | 0 / 92 | 120 / 44 | 2.73x |
| Mini-TPU, complete stage engine (instruction-driven MXU cells with weight files, fold units, vector lanes) | -- / 267 | -- / 61 | -- / 135 | -- / 0 | -- / 265 | -- / 328 | -- |
| FEATHER (NEST weight-file mesh + BIRRD butterfly reduction/reorder) | -- / 97 | -- / 4 | -- / 38 | -- / 0 | -- / 403 | -- / 101 | -- |
| Grouped attention-PV (the matrix unit adapted: column slabs, serpentine psum chain) | 112 / 53 | 10 / 5 | 0 / 0 | 0 / 0 | 0 / 50 | 122 / 58 | 2.10x |
| Variant: matrix unit with a staged, double-buffered weight tile (tpu_staged) | 113 / 54 | 7 / 4 | 0 / 0 | 0 / 0 | 0 / 78 | 120 / 58 | 2.07x |
| Variant: FFT streaming network at any power-of-two size (fft_stream_of) | -- / 67 | -- / 7 | -- / 0 | -- / 0 | -- / 186 | -- / 74 | -- |
| Variant: FEATHER with operands streamed per tile (feather_stream) | -- / 104 | -- / 4 | -- / 38 | -- / 0 | -- / 396 | -- / 108 | -- |
| Variant: FEATHER with resident weights, activations streamed (feather_stream_x) | -- / 100 | -- / 4 | -- / 38 | -- / 0 | -- / 400 | -- / 104 | -- |
| Variant: the earlier instruction-driven engine (test_spmw_tpu_isa.py _engine: MXU with NW-deep weight file + VPU lanes) | -- / 161 | -- / 26 | -- / 27 | -- / 0 | -- / 106 | -- / 187 | -- |
| Variant: the stage engine as measured on the U280 (gpt_stage_v1.py, design of record) | -- / 163 | -- / 51 | -- / 117 | -- / 0 | -- / 123 | -- / 214 | -- |

## Listing fragments and generated code (not comparable to the rows above)

| Item | HLS side | SPMW side | what it is |
|---|---:|---:|---|
| Paper listing, Fig. comparison (code/hls-spmw-comparison.tex): systolic GEMM fragments | 30 design, 0 generated, 0 test/host | 26 design, 0 generated, 0 test/host | Both panels are fragments with constants and helper bodies omitted (the HLS one is adapted from gemm_output_stationary.cpp in float), so they are not comparable with the full-design rows. |
| Paper listing, Fig. tpu (code/tpu.tex): weight-stationary matrix unit with activation row | -- | 26 design, 0 generated, 0 test/host | The act unit and constants are omitted in the listing. |
| Paper listing, Fig. attn (code/attn.tex): grouped attention topology and fabric | -- | 28 design, 0 generated, 0 test/host | Reuses mac and act from Fig. tpu; imports elided. This is the adaptation-only view of attention_pv (grouped_mxu + attention_pv). |
| Paper listing, Fig. fft (code/fft.tex): FFT topology fragment | -- | 19 design, 0 generated, 0 test/host | Uses spmw.to() with owner/port_of where the source file uses key form; the boundary helper fft_io is elided. |
| AutoSA int8 output-stationary GEMM (generated HLS C++; the systolic row's generated counterpart) | 0 design, 8560 generated, 79 test/host | -- | examples/spmw/autosa/int8: input_kernel.c is the hand-written loop nest given to AutoSA (workload); kernel.h, kernel_kernel.h and kernel_kernel.cpp are AutoSA's output (generated); kernel_host.cpp is AutoSA's testbench (test). The generated array includes the C-drain and DRAM I/O networks and AXI control that the SPMW systolic design does not generate; see examples/spmw/autosa/README.md. |

## Adaptation: design lines that differ from the original

Computed with `difflib` over comment-stripped, whitespace-collapsed design lines of the two selections; `added`/`removed` are line counts, not a measure of effort.

| Design | side | original | adapted | unchanged | added | removed |
|---|---|---:|---:|---:|---:|---:|
| Grouped attention-PV (the matrix unit adapted: column slabs, serpentine psum chain) | hls | 113 | 112 | 51 | 61 | 62 |
| Grouped attention-PV (the matrix unit adapted: column slabs, serpentine psum chain) | spmw | 40 | 53 | 26 | 27 | 14 |

## Notes per design

### Systolic GEMM (output-stationary, int8 into int32) (`systolic_gemm`)

- **Vitis HLS C++**: design 94, config 5, workload 0, generated 0, test/host 0
  - design 94: `gemm_output_stationary.cpp` pe, feed_west, feed_north, sink, drain, extern "C" -- PE, west/north loaders, edge sinks, drain, and the extern C top with its AXI pragmas and stream array
  - config 5: `gemm_output_stationary.cpp` ~^#include~, ~^#define~, ~^typedef~
- **SPMW (Python)**: design 24, config 4, workload 0, generated 0, test/host 53
  - design 24: `test_spmw_gemm_int8.py` gemm_int8_of -- MacIO, pe, fabric g, and the size-parameterised wrapper
  - config 4: `test_spmw_gemm_int8.py` N, gemm_int8, ~^import allo\.spmw~, ~^from allo\.ir\.types~
  - test (remainder, excluded) 53: `test_spmw_gemm_int8.py` everything not selected above
- HLS: examples/spmw/baselines/hls/gemm_output_stationary.cpp, a fixed 16x16 mesh; unchanged since the archived count (commit 4f5f283).
- SPMW: gemm_int8_of(size) in tests/dataflow/spmw/test_spmw_gemm_int8.py -- MacIO, pe, the fabric g and the enclosing size-parameterised function; the module instantiates it at N=4 (config).

### Multi-cache GEMM (daisy-chained column drain, int16) (`multicache_gemm`)

- **Vitis HLS C++**: design 113, config 4, workload 0, generated 0, test/host 0
  - design 113: `gemm_multicache.cpp` column_t, pe, feed_west, feed_north, sink, seed_chain, drain_chain, extern "C"
  - config 4: `gemm_multicache.cpp` ~^#include~, ~^#define~, ~^typedef~
- **SPMW (Python)**: design 40, config 4, workload 0, generated 0, test/host 34
  - design 40: `test_spmw_daisy.py` daisy_of minus L89-89 -- line 89 `g.spmw_parts = (MacIO, topo, pe)` is a test hook
  - config 4: `test_spmw_daisy.py` M, daisy_gemm, ~^import allo\.spmw~, ~^from allo\.ir\.types~
  - test (remainder, excluded) 34: `test_spmw_daisy.py` everything not selected above
- HLS: examples/spmw/baselines/hls/gemm_multicache.cpp; unchanged since the archived count.
- SPMW: daisy_of(size, operand, accum) in tests/dataflow/spmw/test_spmw_daisy.py -- MacIO with the c_in/c_out chain, the explicit Topology, pe reading its site, fabric g. The line `g.spmw_parts = ...` inside daisy_of is a test hook and is excluded (reported under test).

### Tiled GEMM (two-level: tile engines of PE meshes) (`tiled_gemm`)

- **Vitis HLS C++**: design 153, config 7, workload 0, generated 0, test/host 0
  - design 153: `gemm_tiled.cpp` pe, load_a, load_b, sink, store_c, tile_engine, extern "C"
  - config 7: `gemm_tiled.cpp` ~^#include~, ~^#define~, ~^typedef~
- **SPMW (Python)**: design 32, config 5, workload 0, generated 0, test/host 47
  - design 32: `test_spmw_tiled.py` MacIO, pe, TileIO, tile_gemm, tiled_gemm
  - config 5: `test_spmw_tiled.py` M, Rt, TM, ~^import allo\.spmw~, ~^from allo\.ir\.types~
  - test (remainder, excluded) 47: `test_spmw_tiled.py` everything not selected above
- HLS: examples/spmw/baselines/hls/gemm_tiled.cpp -- a 2x2 grid of 8x8 engines fed by two fan-out loaders and one storer, int8 into int32. Rewritten after the archived count (commit 8ccf7fa, 2026-09-03: float32 -> int8, per-operand loaders because one AXI bundle may be read by one dataflow process).
- SPMW: tests/dataflow/spmw/test_spmw_tiled.py -- MacIO, pe, TileIO, the placeable tile_gemm fabric and the parent tiled_gemm; float32, 2x2 tiles of 2x2 PEs, memory-fed children (spmw.shard), as the paper's caption says. The two sides now differ in datatype and in how children are fed; the paper's Fig. 4 connected-tile construction has no executable source on either side.

### FFT butterfly network (spatial one-shot + streaming batch) (`fft`)

- **Vitis HLS C++**: no counterpart counted. No hand-written HLS FFT source exists in the Allo tree or the paper workspace; the archived 407 was an external HP-FFT-HLS (n1024/UF32) checkout, absent locally and a different (folded) design; examples/spmw/baselines/sycl/fft.cpp is SYCL, not HLS.
- **SPMW (Python)**: design 103, config 7, workload 0, generated 0, test/host 150
  - design 58: `test_spmw_fft.py` bfly_pair, twiddles, bitrev, BflyIO, bfly_links, topo, bfly, fft_spatial -- the one-shot spatial network and its helpers
  - design 45: `test_spmw_fft.py` StreamIO, stream_links, stream_topo, bfly_stream, fft_stream -- the streaming (batched) form of the same network
  - config 7: `test_spmw_fft.py` FFT_N, S, HALF, csample, BATCH, ~^import allo\.spmw~, ~^from allo\.ir\.types~
  - test (remainder, excluded) 150: `test_spmw_fft.py` everything not selected above
- SPMW: tests/dataflow/spmw/test_spmw_fft.py. Two designs share the file: fft_spatial (BflyIO, key-form bfly_links, topo, the bfly unit, bit-reversed boundary bindings) and fft_stream (StreamIO, stream_links, stream_topo, bfly_stream looping over BATCH transforms, affine bindings). bfly_pair, bitrev and twiddles are design-specific helpers (butterfly geometry, bit reversal, the twiddle ROM contents) and are counted as design; bitrev_perm is the host-side input permutation and stays in test/host.
- No hand-written Vitis HLS FFT exists in either tree. The archived 407 came from an external HP-FFT-HLS checkout (n1024/UF32) through spmw_loc.py --hpfft; it is not present locally and is a different, folded, expert-tuned design. examples/spmw/baselines/sycl/fft.cpp is SYCL/oneAPI, not HLS, and its header says it never built. The FFT row therefore stays unfilled on the HLS side.

### Mini-TPU matrix unit (weight-stationary MXU + activation row) (`tpu_matrix_unit`)

- **Vitis HLS C++**: design 113, config 7, workload 0, generated 0, test/host 0
  - design 113: `gemm_systolic.cpp` pe, feed_a, pe_top, sink_a, drain, tile_pass, extern "C"
  - config 7: `gemm_systolic.cpp` ~^#include~, ~^#define~, ~^typedef~
- **SPMW (Python)**: design 40, config 4, workload 0, generated 0, test/host 92
  - design 40: `test_spmw_tpu.py` WsIO, ActIO, mac, act, mxu_links, mxu, tpu_matmul
  - config 4: `test_spmw_tpu.py` KT, SHIFT, ~^import allo\.spmw~, ~^from allo\.ir\.types~
  - test (remainder, excluded) 92: `test_spmw_tpu.py` everything not selected above
- HLS: examples/spmw/baselines/hls/gemm_systolic.cpp -- 16x16 weight-stationary int8 array, ReLU-and-shift drain, 128-bit packed A/Y buses and a host-tiled `tiles` loop that reloads the weights per tile. Changed after the archived count (commit 2fcf5bd, 2026-09-03: the seed process folded into a pe_top row and deeper streams, to fix a hardware deadlock).
- SPMW: tests/dataflow/spmw/test_spmw_tpu.py -- WsIO, ActIO, mac, act, mxu_links, mxu, tpu_matmul: one tile pass (MT rows) with scalar ports; no bus packing and no tile loop. Scope difference: the HLS kernel also carries the per-tile weight reload and 128-bit beat packing; the SPMW variant tpu_staged (double-buffered weight tile) is counted separately as a variant row.

### Mini-TPU, complete stage engine (instruction-driven MXU cells with weight files, fold units, vector lanes) (`tpu_full`)

- **Vitis HLS C++**: no counterpart counted. No hand-written C++ HLS mini-TPU exists in either tree (examples/transformer_hls.py is an Allo/Python description).
- **SPMW (Python)**: design 267, config 61, workload 135, generated 0, test/host 265
  - design 258: `test_spmw_gpt_stage.py` LaneIO, FoldIO, stage_engine minus L501-502 -- lines 501-502 `engine.spmw_parts = ...`, `engine.spmw_fold = fold` are test hooks
  - design 9: `test_spmw_gpt_stage.py` MacIO -- module-level MacIO is no longer referenced (stage_engine declares CellIO)
  - config 44: `test_spmw_gpt_stage.py` MSWEEP, MPASS, MLOAD, MREP, GRP, SMXB, NRM, GMAX, NBG, ACCN, QUANT_SCORE@70, EXP_SHIFT@71, EXP_BASE@72, PROB_BITS@73, QUANT_SCORE@654, EXP_SHIFT@655, EXP_BASE@656, PROB_BITS@657, gpt_stage_of, L27-28, L30-51 -- imports (incl. the 22-line ISA import block), opcodes, fixed-point constants, and the board-shape instantiation
  - workload 131: `test_spmw_gpt_stage.py` mxu_sweep, mxu_pass, mxu_load, mxu_rep, mxu_program, stage_vprog, running_max_vprog, _EXP, _vprog, _EXP_FUSED, row_sum_fused, normalise_fused, row_max_vprog, row_sum_vprog, normalise_vprog, grouped_vprog, attention_head -- instruction encoders, the softmax/GEMM programs and the attention-head launch sequence
  - config 17: `test_spmw_tpu_isa.py` ACCZ, ADD, EXP2, LOADB, LOADI, LOADR, LOADZ, MAX, MUL, NB, NOP, NPROG, RCP_BITS, REGS, SHR, STORE, SUB -- the base ISA constants the stage engine imports
  - workload 4: `test_spmw_tpu_isa.py` vpu_word, vpu_header -- the lane instruction encoders the stage engine imports
  - test (remainder, excluded) 265: `test_spmw_gpt_stage.py` everything not selected above
- SPMW: stage_engine(dim, kfile, outs, sweep, ...) in tests/dataflow/spmw/test_spmw_gpt_stage.py -- CellIO, the mxu/chain/folds topologies, the mac cell (MLOAD/MREP/MSWEEP/MPASS), the fold unit, the vpu lane (LaneIO, GRP/SMXB/NRM/ACCN dispatch) and the fabric; LaneIO and FoldIO at module level. The two trailing test hooks in stage_engine (`engine.spmw_parts`, `engine.spmw_fold`) are excluded. The module-level MacIO (9 lines) is no longer referenced -- stage_engine declares CellIO -- and is listed as its own design part so it can be subtracted.
- Design 267 = 258 (LaneIO 7, FoldIO 4, stage_engine 247) + 9 for the unreferenced module-level MacIO; without that declaration the engine is 258 design lines. Config 61 = 44 in the file (of which 24 are import lines) + 17 base-ISA constants imported from test_spmw_tpu_isa.py.
- The engine imports its base ISA (17 opcode/size constants) and the vpu_word/vpu_header encoders from test_spmw_tpu_isa.py; those are counted here as config and workload (shared parts). Programs (mxu_* and *_vprog encoders, the attention-head launch sequence) are workload; operand layout (stage_operands, file_to_stream, lane_bias, pass_operands, identity_file), references and tests are test/host.
- No C++ HLS counterpart exists as source: examples/transformer_hls.py and allo/library/systolic.py are Allo (Python) descriptions, not hand-written C++ HLS, and no HLS mini-TPU with an instruction stream was written. The row is SPMW-only.
- The frozen board-measured version (gpt_stage_v1.py, the design of record for the U280 numbers) is counted as a variant row.

### FEATHER (NEST weight-file mesh + BIRRD butterfly reduction/reorder) (`feather`)

- **Vitis HLS C++**: no counterpart counted. FEATHER's reference is SystemVerilog RTL (maeri-project/FEATHER), not present locally; examples/feather/*.py is Allo, not C++ HLS.
- **SPMW (Python)**: design 97, config 4, workload 38, generated 0, test/host 403
  - design 97: `test_spmw_feather.py` reverse_bits, birrd_shape, stage_bits, feather minus L163-168 -- lines 163-168 are the cosim hooks (spmw_parts, default BIRRD program as spmw_operands)
  - config 4: `test_spmw_feather.py` PS, ~^from math import~, ~^import allo\.spmw~, ~^from allo\.ir\.types~
  - workload 38: `test_spmw_feather.py` gemm_insts, conv_insts -- the BIRRD command programs from the Allo drivers
  - test (remainder, excluded) 403: `test_spmw_feather.py` everything not selected above
- SPMW: feather(AW, AH) in tests/dataflow/spmw/test_spmw_feather.py -- PEIO, SwIO, the nest and birrd topologies (wiring = the paper's bit-reversal butterfly), the pe and switch units and the engine fabric; reverse_bits, birrd_shape and stage_bits are design-specific topology helpers. The cosim glue at the end of feather() (lines 163-168: spmw_parts, a default BIRRD program as spmw_operands) is excluded. gemm_insts/conv_insts are BIRRD programs (workload); tile layout, references and GEMM/conv drivers are test/host.
- Design 97 = reverse_bits 7 + birrd_shape 3 + stage_bits 3 + feather() 84 (PEIO 7, SwIO 6, nest 8, wiring 10, birrd 1, pe 11, switch 17, engine fabric 20, glue 4).
- No HLS counterpart: FEATHER's original implementation is SystemVerilog RTL (maeri-project/FEATHER, FEATHER_RTL/RTL), which is not in either tree -- only the testbench tests/dataflow/spmw/tb_feather_rtl.sv (162 SV lines, test code) is local. examples/feather/*.py is an Allo dataflow description, not C++ HLS. The row is SPMW-only; the streamed-operand variants feather_stream and feather_stream_x are counted as variant rows.

### Grouped attention-PV (the matrix unit adapted: column slabs, serpentine psum chain) (`attention_pv`)

- **Vitis HLS C++**: design 112, config 10, workload 0, generated 0, test/host 0
  - design 112: `attention_pv.cpp` pe_forward, pe_edge, feed_a, seed_p, drain, array_pass, extern "C"
  - config 10: `attention_pv.cpp` ~^#include~, ~^#define~, ~^#ifndef~, ~^#endif~, ~^typedef~
- **SPMW (Python)**: design 53, config 5, workload 0, generated 0, test/host 50
  - design 53: `test_spmw_attention.py` WsIO, ActIO, mac, act, grouped_mxu, attention_pv
  - config 5: `test_spmw_attention.py` R, MT, SHIFT, ~^import allo\.spmw~, ~^from allo\.ir\.types~
  - test (remainder, excluded) 50: `test_spmw_attention.py` everything not selected above
- HLS: examples/spmw/baselines/hls/attention_pv.cpp -- pe_forward/pe_edge, the GROUPS-dependent loader, seed, drain and the array_pass dataflow function. Changed after the archived count (commits 8ccf7fa and 5e1a84b, 2026-09-03: the forwarding flag became two PE functions and the array moved into its own function because a DATAFLOW pragma in a bare block was silently ignored).
- SPMW: tests/dataflow/spmw/test_spmw_attention.py -- WsIO, ActIO, mac and act (copies of the matrix unit's), grouped_mxu (the link rule) and the attention_pv(groups) fabric. Counted self-contained, like the HLS file; the adaptation table below gives the design lines that differ from tpu_matrix_unit on each side (difflib over comment-stripped code lines). On the HLS side that diff also absorbs unrelated differences between the two files (bus packing, the tiles loop), so it overstates the adaptation.

### Variant: matrix unit with a staged, double-buffered weight tile (tpu_staged) (`tpu_matrix_unit_staged`)

- **Vitis HLS C++**: design 113, config 7, workload 0, generated 0, test/host 0
  - design 113: `gemm_systolic.cpp` pe, feed_a, pe_top, sink_a, drain, tile_pass, extern "C"
  - config 7: `gemm_systolic.cpp` ~^#include~, ~^#define~, ~^typedef~
- **SPMW (Python)**: design 54, config 4, workload 0, generated 0, test/host 78
  - design 54: `test_spmw_tpu.py` WsIO, ActIO, mac, act, mxu_links, mxu, tpu_matmul, tpu_staged
  - config 4: `test_spmw_tpu.py` KT, SHIFT, ~^import allo\.spmw~, ~^from allo\.ir\.types~
  - test (remainder, excluded) 78: `test_spmw_tpu.py` everything not selected above
- Same HLS file as tpu_matrix_unit. SPMW adds the tpu_staged fabric (phases, a banked double-buffered weight memory) on top of the same units and topology.

### Variant: FFT streaming network at any power-of-two size (fft_stream_of) (`fft_stream_of`)

- **SPMW (Python)**: design 67, config 7, workload 0, generated 0, test/host 186
  - design 67: `test_spmw_fft.py` twiddles, bitrev, fft_stream_of
  - config 7: `test_spmw_fft.py` FFT_N, S, HALF, csample, BATCH, ~^import allo\.spmw~, ~^from allo\.ir\.types~
  - test (remainder, excluded) 186: `test_spmw_fft.py` everything not selected above
- The size-parameterised streaming FFT used for the N=256 measurements; a self-contained function (its own IO class, links, butterfly and fabric) plus the shared twiddles/bitrev helpers.

### Variant: FEATHER with operands streamed per tile (feather_stream) (`feather_stream`)

- **SPMW (Python)**: design 104, config 4, workload 38, generated 0, test/host 396
  - design 104: `test_spmw_feather.py` reverse_bits, birrd_shape, stage_bits, feather_stream minus L286-289
  - config 4: `test_spmw_feather.py` PS, ~^from math import~, ~^import allo\.spmw~, ~^from allo\.ir\.types~
  - workload 38: `test_spmw_feather.py` gemm_insts, conv_insts
  - test (remainder, excluded) 396: `test_spmw_feather.py` everything not selected above
- The same engine taking NT tiles per launch with activations, weight files and commands streamed; cosim hooks (lines 286-289) excluded.

### Variant: FEATHER with resident weights, activations streamed (feather_stream_x) (`feather_stream_x`)

- **SPMW (Python)**: design 100, config 4, workload 38, generated 0, test/host 400
  - design 100: `test_spmw_feather.py` reverse_bits, birrd_shape, stage_bits, feather_stream_x minus L410-413
  - config 4: `test_spmw_feather.py` PS, ~^from math import~, ~^import allo\.spmw~, ~^from allo\.ir\.types~
  - workload 38: `test_spmw_feather.py` gemm_insts, conv_insts
  - test (remainder, excluded) 400: `test_spmw_feather.py` everything not selected above
- The counterpart of the RTL's activation pass: weight files and commands resident, NT tiles of activations streamed; cosim hooks (lines 410-413) excluded.

### Variant: the earlier instruction-driven engine (test_spmw_tpu_isa.py _engine: MXU with NW-deep weight file + VPU lanes) (`tpu_isa_engine`)

- **SPMW (Python)**: design 161, config 26, workload 27, generated 0, test/host 106
  - design 161: `test_spmw_tpu_isa.py` MacIO, VpuIO, _engine minus L357-357
  - config 26: `test_spmw_tpu_isa.py` D, SEQ, NW, NB, REGS, NPROG, MACC, MZERO, MSKIP, NOP, LOADZ, LOADB, LOADI, ADD, MUL, MAX, SHR, STORE, ACCZ, SUB, EXP2, LOADR, RCP_BITS, ONE_PASS, ~^import allo\.spmw~, ~^from allo\.ir\.types~
  - workload 27: `test_spmw_tpu_isa.py` mxu_word, vpu_word, mxu_program, READS_Z, vpu_header, vpu_program, PASSTHROUGH
  - test (remainder, excluded) 106: `test_spmw_tpu_isa.py` everything not selected above
- The predecessor of the stage engine: MacIO/VpuIO, the _engine fabric with the mac and vpu units (no fold unit, one instruction per step). Its trailing `engine.spmw_parts` hook (line 357) is excluded; ONE_PASS is the instantiation (config), PASSTHROUGH a program (workload), the cosim operands, reference and tests are test/host.

### Variant: the stage engine as measured on the U280 (gpt_stage_v1.py, design of record) (`gpt_stage_v1`)

- **SPMW (Python)**: design 163, config 51, workload 117, generated 0, test/host 123
  - design 148: `gpt_stage_v1.py` stage_engine minus L294-294
  - design 9: `gpt_stage_v1.py` MacIO -- module-level MacIO is not referenced (stage_engine declares CellIO)
  - config 34: `gpt_stage_v1.py` MSWEEP, MPASS, MLOAD, ACCN, QUANT_SCORE, EXP_SHIFT, EXP_BASE, PROB_BITS, gpt_stage_of, L15-16, L18-39
  - workload 113: `gpt_stage_v1.py` mxu_sweep, mxu_pass, mxu_load, mxu_program, stage_vprog, running_max_vprog, _EXP, _vprog, row_max_vprog, row_sum_vprog, normalise_vprog, attention_head
  - design 6: `test_spmw_tpu_isa.py` VpuIO -- the lane interface v1 imports
  - config 17: `test_spmw_tpu_isa.py` ACCZ, ADD, EXP2, LOADB, LOADI, LOADR, LOADZ, MAX, MUL, NB, NOP, NPROG, RCP_BITS, REGS, SHR, STORE, SUB
  - workload 4: `test_spmw_tpu_isa.py` vpu_word, vpu_header
  - test (remainder, excluded) 123: `gpt_stage_v1.py` everything not selected above
- A frozen copy of the stage engine at the brick transport (no MREP, no lane groups, no fold unit; the lane is test_spmw_tpu_isa's VpuIO). Its `engine.spmw_parts` hook (line 294) is excluded; the module-level MacIO (unused, superseded by CellIO) is listed separately as in tpu_full.

### Paper listing, Fig. comparison (code/hls-spmw-comparison.tex): systolic GEMM fragments (`fragment_systolic`)

- **HLS fragment**: design 30, config 0, workload 0, generated 0, test/host 0
  - design 30: `hls-spmw-comparison.tex#listing0` (whole file)
- **SPMW fragment**: design 26, config 0, workload 0, generated 0, test/host 0
  - design 26: `hls-spmw-comparison.tex#listing1` (whole file)
- Both panels are fragments with constants and helper bodies omitted (the HLS one is adapted from gemm_output_stationary.cpp in float), so they are not comparable with the full-design rows.

### Paper listing, Fig. tpu (code/tpu.tex): weight-stationary matrix unit with activation row (`fragment_tpu`)

- **SPMW fragment**: design 26, config 0, workload 0, generated 0, test/host 0
  - design 26: `tpu.tex#listing0` (whole file)
- The act unit and constants are omitted in the listing.

### Paper listing, Fig. attn (code/attn.tex): grouped attention topology and fabric (`fragment_attn`)

- **SPMW fragment**: design 28, config 0, workload 0, generated 0, test/host 0
  - design 28: `attn.tex#listing0` (whole file)
- Reuses mac and act from Fig. tpu; imports elided. This is the adaptation-only view of attention_pv (grouped_mxu + attention_pv).

### Paper listing, Fig. fft (code/fft.tex): FFT topology fragment (`fragment_fft`)

- **SPMW fragment**: design 19, config 0, workload 0, generated 0, test/host 0
  - design 19: `fft.tex#listing0` (whole file)
- Uses spmw.to() with owner/port_of where the source file uses key form; the boundary helper fft_io is elided.

### AutoSA int8 output-stationary GEMM (generated HLS C++; the systolic row's generated counterpart) (`autosa_gemm_int8`)

- **AutoSA (input + generated)**: design 0, config 0, workload 33, generated 8560, test/host 79
  - workload 33: `input_kernel.c` (whole file) -- the polyhedral C input
  - generated 8: `kernel.h` (whole file)
  - generated 22: `kernel_kernel.h` (whole file)
  - generated 8530: `kernel_kernel.cpp` (whole file)
  - test 79: `kernel_host.cpp` (whole file) -- AutoSA's testbench
- examples/spmw/autosa/int8: input_kernel.c is the hand-written loop nest given to AutoSA (workload); kernel.h, kernel_kernel.h and kernel_kernel.cpp are AutoSA's output (generated); kernel_host.cpp is AutoSA's testbench (test). The generated array includes the C-drain and DRAM I/O networks and AXI control that the SPMW systolic design does not generate; see examples/spmw/autosa/README.md.

