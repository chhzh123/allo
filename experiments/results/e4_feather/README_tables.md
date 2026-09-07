### Whole workloads, general weights (this agent's runs): cycles measured in xsim on both sides

| workload | N | tiles | system | mode | first tile done | last tile done (whole workload) | cycles per tile (steady state, median) | validation |
|---|---|---|---|---|---|---|---|---|
| GEMM 128^3 | 4 | 32768 | SPMW port feather_stream | every operand streamed per tile | 50 | 262182 | 8 | pass: SPMW COSIM PASS (524288/524288 tokens, 0 errors); host reduction vs numpy: pass (0 bad ele |
| GEMM 128^3 | 4 | 32768 | FEATHER RTL (corrected) | a weight feed per tile | 81 | 2097169 | 64 | pass: 32768/32768 tiles bit-exact vs the RTL-arithmetic model; host reduction vs numpy: pass (0  |
| GEMM 128^3 | 4 | 32768 | SPMW port feather_stream_x | weights resident | 50 | 131118 | 4 | pass: SPMW COSIM PASS (524288/524288 tokens, 0 errors) |
| GEMM 128^3 | 4 | 32768 | FEATHER RTL (corrected) | weights resident | 81 | 131149 | 4 | pass: 32768/32768 tiles bit-exact vs the RTL-arithmetic model; signed_equals_rtl=True |
| GEMM 128^3 | 8 | 4096 | SPMW port feather_stream | every operand streamed per tile | 132 | 65644 | 16 | pass: SPMW COSIM PASS (262144/262144 tokens, 0 errors); host reduction vs numpy: pass (0 bad ele |
| GEMM 128^3 | 8 | 4096 | FEATHER RTL (corrected) | a weight feed per tile | 539 | 2097179 | 512 | pass: 4096/4096 tiles bit-exact vs the RTL-arithmetic model; host reduction vs numpy: pass (0 ba |
| GEMM 128^3 | 8 | 4096 | SPMW port feather_stream_x | weights resident | 96 | 32856 | 8 | pass: SPMW COSIM PASS (262144/262144 tokens, 0 errors) |
| GEMM 128^3 | 8 | 4096 | FEATHER RTL (corrected) | weights resident | 539 | 33299 | 8 | pass: 4096/4096 tiles bit-exact vs the RTL-arithmetic model; signed_equals_rtl=True |
| GEMM 128^3 | 16 | 512 | SPMW port feather_stream | every operand streamed per tile | 384 | 16720 | 32 | pass: SPMW COSIM PASS (131072/131072 tokens, 0 errors); host reduction vs numpy: pass (0 bad ele |
| GEMM 128^3 | 16 | 512 | FEATHER RTL (corrected) | a weight feed per tile | 4141 | 2097197 | 4096 | pass: 512/512 tiles bit-exact vs the RTL-arithmetic model; host reduction vs numpy: pass (0 bad  |
| GEMM 128^3 | 16 | 512 | SPMW port feather_stream_x | weights resident | 164 | 8340 | 16 | pass: SPMW COSIM PASS (131072/131072 tokens, 0 errors) |
| GEMM 128^3 | 16 | 512 | FEATHER RTL (corrected) | weights resident | 4141 | 12317 | 16 | pass: 512/512 tiles bit-exact vs the RTL-arithmetic model; signed_equals_rtl=True |
| conv 16x16x64->64 3x3 | 4 | 196608 | SPMW port feather_stream | every operand streamed per tile | 50 | 1572902 | 8 | pass: SPMW COSIM PASS (3145728/3145728 tokens, 0 errors); host reduction vs numpy: pass (0 bad e |
| conv 16x16x64->64 3x3 | 4 | 196608 | FEATHER RTL (corrected) | a weight feed per tile | 81 | 12582929 | 64 | pass: 196608/196608 tiles bit-exact vs the RTL-arithmetic model; host reduction vs numpy: pass ( |
| conv 16x16x64->64 3x3 | 4 | 196608 | SPMW port feather_stream_x | weights resident | 50 | 786478 | 4 | pass: SPMW COSIM PASS (3145728/3145728 tokens, 0 errors) |
| conv 16x16x64->64 3x3 | 4 | 196608 | FEATHER RTL (corrected) | weights resident | 81 | 786509 | 4 | pass: 196608/196608 tiles bit-exact vs the RTL-arithmetic model; signed_equals_rtl=True |
| conv 16x16x64->64 3x3 | 8 | 32768 | SPMW port feather_stream | every operand streamed per tile | 132 | 524396 | 16 | pass: SPMW COSIM PASS (2097152/2097152 tokens, 0 errors); host reduction vs numpy: pass (0 bad e |
| conv 16x16x64->64 3x3 | 8 | 32768 | FEATHER RTL (corrected) | a weight feed per tile | 539 | 16777243 | 512 | pass: 32768/32768 tiles bit-exact vs the RTL-arithmetic model; host reduction vs numpy: pass (0  |
| conv 16x16x64->64 3x3 | 8 | 32768 | SPMW port feather_stream_x | weights resident | 96 | 262232 | 8 | pass: SPMW COSIM PASS (2097152/2097152 tokens, 0 errors) |
| conv 16x16x64->64 3x3 | 8 | 32768 | FEATHER RTL (corrected) | weights resident | 539 | 262675 | 8 | pass: 32768/32768 tiles bit-exact vs the RTL-arithmetic model; signed_equals_rtl=True |
| conv 16x16x64->64 3x3 | 16 | 4096 | SPMW port feather_stream | every operand streamed per tile | 384 | 131408 | 32 | pass: SPMW COSIM PASS (1048576/1048576 tokens, 0 errors); host reduction vs numpy: pass (0 bad e |
| conv 16x16x64->64 3x3 | 16 | 4096 | FEATHER RTL (corrected) | a weight feed per tile | 4141 | 16777261 | 4096 | pass: 4096/4096 tiles bit-exact vs the RTL-arithmetic model; host reduction vs numpy: pass (0 ba |
| conv 16x16x64->64 3x3 | 16 | 4096 | SPMW port feather_stream_x | weights resident | 164 | 65684 | 16 | pass: SPMW COSIM PASS (1048576/1048576 tokens, 0 errors) |
| conv 16x16x64->64 3x3 | 16 | 4096 | FEATHER RTL (corrected) | weights resident | 4141 | 69661 | 16 | pass: 4096/4096 tiles bit-exact vs the RTL-arithmetic model; signed_equals_rtl=True |

### The same workloads with the previous agent's constrained operands (its runs)

| workload | N | system | mode | first tile done | last tile done | cycles per tile | validation |
|---|---|---|---|---|---|---|---|
| GEMM 128^3 | 8 | SPMW port feather_stream | streamed | 132 | 65644 | 16 | pass: SPMW COSIM PASS (262144/262144 tokens, 0 errors); (no host reduction r |
| GEMM 128^3 | 8 | FEATHER RTL (corrected) | a feed per tile | 539 | 2097179 | 512 | pass: 4096/4096 tiles bit-exact vs the RTL-arithmetic model; host reduction  |
| GEMM 128^3 | 8 | SPMW port feather_stream_x | resident | 96 | 32856 | 8 | pass: SPMW COSIM PASS (262144/262144 tokens, 0 errors) |
| GEMM 128^3 | 8 | FEATHER RTL (corrected) | resident | 539 | 33299 | 8 | pass: 4096/4096 tiles bit-exact vs the RTL-arithmetic model; signed_equals_r |
| GEMM 128^3 | 16 | SPMW port feather_stream | streamed | 384 | 16720 | 32 | pass: SPMW COSIM PASS (131072/131072 tokens, 0 errors); (no host reduction r |
| GEMM 128^3 | 16 | FEATHER RTL (corrected) | a feed per tile | 4141 | 2097197 | 4096 | pass: 512/512 tiles bit-exact vs the RTL-arithmetic model; host reduction vs |
| GEMM 128^3 | 16 | SPMW port feather_stream_x | resident | 164 | 8340 | 16 | pass: SPMW COSIM PASS (131072/131072 tokens, 0 errors) |
| GEMM 128^3 | 16 | FEATHER RTL (corrected) | resident | 4141 | 12317 | 16 | pass: 512/512 tiles bit-exact vs the RTL-arithmetic model; signed_equals_rtl |
| conv 16x16x64->64 3x3 | 4 | SPMW port feather_stream | streamed | 50 | 1572902 | 8 | pass: SPMW COSIM PASS (3145728/3145728 tokens, 0 errors); (no host reduction |
| conv 16x16x64->64 3x3 | 4 | FEATHER RTL (corrected) | a feed per tile | 81 | 12582929 | 64 | pass: 196608/196608 tiles bit-exact vs the RTL-arithmetic model; host reduct |
| conv 16x16x64->64 3x3 | 4 | SPMW port feather_stream_x | resident | 50 | 786478 | 4 | pass: SPMW COSIM PASS (3145728/3145728 tokens, 0 errors) |
| conv 16x16x64->64 3x3 | 4 | FEATHER RTL (corrected) | resident | 81 | 786509 | 4 | pass: 196608/196608 tiles bit-exact vs the RTL-arithmetic model; signed_equa |

### Place-and-route, xcu280-fsvh2892-2L-e, Vivado 2023.2, 3.333 ns

| design | N | status | LUT | FF | DSP | BRAM (18k eq.) | URAM | WNS ns | TNS ns | unrouted | total s | HLS s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPMW port feather_stream (spmw_harness) | 4 | pass | 7225 | 10646 | 24 | 0 | 0 | 0.571 | 0.000 | 0 | 526 | 87.0 |
| SPMW port feather, single tile (spmw_harness) | 4 | pass | 2305 | 3639 | 16 | 0 | 0 | 1.200 | 0.000 | 0 | 434 | 88.7 |
| FEATHER RTL corrected, feather_top OOC | 4 | pass | 2309 | 3378 | 0 | 0 | 0 | 1.174 | 0.000 | 0 | 530 | - |
| FEATHER RTL shipped, feather_top OOC | 4 | pass | 2308 | 3378 | 0 | 0 | 0 | 1.100 | 0.000 | 0 | 371 | - |
| SPMW port feather_stream (spmw_harness) | 8 | pass | 53060 | 59624 | 96 | 0 | 0 | 0.083 | 0.000 | 0 | 2011 | 88.3 |
| SPMW port feather, single tile (spmw_harness) | 8 | pass | 9974 | 14989 | 64 | 0 | 0 | 0.866 | 0.000 | 0 | 684 | 98.3 |
| FEATHER RTL corrected, feather_top OOC | 8 | pass | 9694 | 15332 | 0 | 0 | 0 | 0.355 | 0.000 | 0 | 1011 | - |
| FEATHER RTL shipped, feather_top OOC | 8 | pass | 9755 | 15334 | 0 | 0 | 0 | 0.379 | 0.000 | 0 | 529 | - |
| SPMW port feather_stream (spmw_harness) | 16 | pass | 314670 | 349009 | 384 | 0 | 0 | -0.699 | -3060.044 | 0 | 11782 | 92.5 |
| SPMW port feather, single tile (spmw_harness) | 16 | pass | 31019 | 49673 | 256 | 0 | 0 | 0.338 | 0.000 | 0 | 1608 | 94.6 |
| FEATHER RTL corrected, feather_top OOC | 16 | pass | 57499 | 91251 | 0 | 0 | 0 | 0.120 | 0.000 | 0 | 1901 | - |
| FEATHER RTL shipped, feather_top OOC | 16 | pass | 57595 | 91259 | 0 | 0 | 0 | 0.062 | 0.000 | 0 | 2506 | - |
| SPMW port feather_stream (spmw_harness) | 32 | not_run | - | - | - | - | - | - | - | - | - | - |
| SPMW port feather, single tile (spmw_harness) | 32 | not_run | - | - | - | - | - | - | - | - | - | - |
| FEATHER RTL corrected, feather_top OOC | 32 | pass | 338694 | 624270 | 0 | 0 | 0 | -0.227 | -344.040 | 0 | 12145 | - |
| FEATHER RTL shipped, feather_top OOC | 32 | pass | 338438 | 624470 | 0 | 0 | 0 | -0.206 | -76.175 | 0 | 11396 | - |
