# Language-aware source recount (E7)

Counter: `/scratch/hc676/allo/scripts/spmw_loc2.py` (sha256 `fea0ba570ece6b7694b43e3edb747e5761cc0daa83088f7227b48fea18568c56`), spmw_loc2 2026-09-06. Manifest sha256 `69e5322bdc40da120f4cb7ae5a6fc63e7f28d35145db21d85c7fae4352634010`.

Rule: cloc's -- a physical line counts when it holds code; comment-only lines, docstrings and standalone string statements do not; code with a trailing comment does. Python via `tokenize`+`ast`, C/C++ and SystemVerilog via a comment- and string-aware stripper, TeX listings extracted first. `design` = PE/unit bodies, interfaces, topology/link rules, loaders/drains/boundary bindings and design-specific interface code; `config` = imports/includes, constants, type aliases and sizes; `workload` = programs and their encoders; `test` = tests, references, operand generators and host code (reported, not counted). The headline columns are `design` only. Source size is not development time.

## Provenance

- Counted on brg-zhang-xcel with scripts/spmw_loc2.py as black-formatted there (sha256 fea0ba570ece6b7694b43e3edb747e5761cc0daa83088f7227b48fea18568c56; the E7 archive's 68a47ad0... is the same program before black), over /scratch/hc676/allo (the worktree's rsync) and /scratch/hc676/HP-FFT-HLS at c4611b8.

## Design-only counts, with the archived provisional numbers

| Design | HLS | SPMW | HLS/SPMW | archived HLS | archived SPMW | archived ratio | note |
|---|---:|---:|---:|---:|---:|---:|---|
| Folded radix-2 FFT (HP-FFT n256/UF1, one sample per cycle, vs the SPMW single-path delay-feedback pipeline) | 299 | 106 | 2.82x | 407 | -- | -- | The archived 407 was HP-FFT n1024/UF32 (a 32-samples-per-cycle unrolled variant) counted with the C rule on the whole file; the architecture comparable to the SPMW SDF pipeline is UF1 (one sample per cycle), n256 and n1024 differ by 6 lines. |

## Breakdown by category (HLS / SPMW)

| Design | design | config | workload | generated | test+host (excluded) | design+config | ratio (design+config) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Folded radix-2 FFT (HP-FFT n256/UF1, one sample per cycle, vs the SPMW single-path delay-feedback pipeline) | 299 / 106 | 26 / 5 | 0 / 0 | 0 / 0 | 57 / 41 | 325 / 111 | 2.93x |

## Notes per design

### Folded radix-2 FFT (HP-FFT n256/UF1, one sample per cycle, vs the SPMW single-path delay-feedback pipeline) (`fft_folded`)

- **Vitis HLS C++ (HP-FFT UF1)**: design 299, config 26, workload 0, generated 0, test/host 57
  - design 299: `FFT.cpp` revIdxTab, twiddles, RADIX2_BFLY_double_buffer_quarter_CY, output_result_array_to_stream, bit_reverse, reverse_input_stream_UF1, FFT_stage_spatial_unroll, RADIX2_BFLY_double_buffer_quarter_onlycompute, FFT_Stage1_vectorstream_parameterize, FFT_DIT_spatial_unroll_CY_stream_vector, FFT_TOP -- every function and the two global tables of the design file
  - config 1: `FFT.cpp` ~^#include~
  - config 25: `FFT.h` (whole file) -- includes, sizes, the top's declaration
  - test 57: `testbench.cpp` (whole file)
- **SPMW (Python)**: design 106, config 5, workload 0, generated 0, test/host 41
  - design 106: `test_spmw_fft_sdf.py` twiddles, bitrev, fft_sdf_of
  - config 5: `test_spmw_fft_sdf.py` csample, ~^import~, ~^from~
  - test 39: `test_spmw_fft_sdf.py` operands, check, test_sdf_matches_numpy, test_sdf_impulses_and_tone
  - test (remainder, excluded) 2: `test_spmw_fft_sdf.py` everything not selected above
- HLS: HP-FFT-HLS commit c4611b8, n256/UF1/FFT.cpp: the double-buffered butterfly, the input bit-reversal stream, the stage functions (spatially unrolled per stage, pipelined), the vector-stream stage-1 form, the DIT driver and FFT_TOP; FFT.h holds includes, sizes (FFT_NUM, EXP2_FFT, UF) and the top's declaration (3 lines, counted as config with the header).
- SPMW: tests/dataflow/spmw/test_spmw_fft_sdf.py fft_sdf_of(n, batch): StageIO/ReorderIO, the chain topology, the stage unit (delay line + butterfly + twiddle), the reorder unit (double buffer, bit-reversal ROM) and the engine fabric; twiddles and bitrev are the ROM contents (design). csample is a type alias (config).

