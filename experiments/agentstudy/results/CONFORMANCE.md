# Architecture conformance: all fifteen designs

Every submitted design was read against the five questions in
`task/CONFORMANCE.md`. Questions 1 to 4 are pass criteria; question 5 is
recorded but does not decide anything.

**Result: fifteen of fifteen conform. No design is demoted, and the graded
table in `INTERIM.md` stands as it is.** The bars that designs failed are the
measured ones -- interval, latency, multipliers, correctness -- not the
architecture.

## Verdicts

`acc` = 64 accumulators, one per grid position, each holding one output
element across all eight steps. `nbr` = every operand arrives from a
neighbour or an edge port. `A/B` = A east and B south, one element per step.
`chain` = results reach the port through a chain of elements. `grid` =
recorded only: how the 64 positions are written.

| Model | Arm | acc | nbr | A/B | chain | Verdict | grid (recorded) |
|---|---|:-:|:-:|:-:|:-:|---|---|
| Opus 5 | SPMW | yes | yes | yes | yes | **conforms** | placed on a topology |
| GPT-5.6 Sol | SPMW | yes | yes | yes | yes | **conforms** | placed on a topology |
| DeepSeek | SPMW | yes | yes | yes | yes | **conforms** | placed on a topology |
| Kimi K3 | SPMW | yes | yes | yes | yes | **conforms** | placed on a topology |
| GLM 5.3 | SPMW | yes | yes | yes | yes | **conforms** | placed on a topology |
| Opus 5 | SystemVerilog | yes | yes | yes | yes | **conforms** | `generate` loop |
| GPT-5.6 Sol | SystemVerilog | yes | yes | yes | yes | **conforms** | `generate` loop |
| DeepSeek | SystemVerilog | yes | yes | yes | yes | **conforms** | `generate` loop |
| Kimi K3 | SystemVerilog | yes | yes | yes | yes | **conforms** | `generate` loop |
| GLM 5.3 | SystemVerilog | yes | yes | yes | yes | **conforms** | `generate` loop |
| Opus 5 | HLS | yes | yes | yes | yes | **conforms** | 128 calls written out |
| DeepSeek | HLS | yes | yes | yes | yes | **conforms** | macro expansion |
| Kimi K3 | HLS | yes | yes | yes | yes | **conforms** | macro expansion |
| GLM 5.3 | HLS | yes | yes | yes | yes | **conforms** | macro expansion |
| GPT-5.6 Sol | HLS | yes | yes | yes | yes | **conforms**, does not compile | macro expansion |

## Evidence, by arm

**SPMW.** All five express the grid once and place it, and all five wire only
nearest neighbours: `spmw.to((i, j + 1), ...)` for A, `spmw.to((i + 1, j), ...)`
for B, and `spmw.stream_in` / `spmw.gather` only at the edges. Nothing is
broadcast and no site reads another site's state.

The structural split that produced the interval result is visible at the
topology level rather than inside any loop body. Opus and GPT-5.6 Sol declare
**two** meshes and place **two** units -- a `mac` mesh and a separate
`carry`/`drain` mesh -- so the drain is its own unit and runs concurrently with
the next product. DeepSeek, Kimi and GLM declare **one** mesh carrying
`PEIO.c_out` alongside `a_out` and `b_out`, so the drain shares the element's
single thread of control. Both shapes are conforming systolic arrays; only the
first pipelines.

**SystemVerilog.** All five instantiate one `pe` module from a `generate`
loop over an 8x8 wire mesh, with `a_d[i][j+1]` driven by PE(i,j) and
`a_d[i][0]` by the edge port, and the same for B southward. Kimi and GLM, the
two reviewed here for the first time, both reach the port through the row's
own chain: Kimi's results flow west to `d_d[i][0]` (`design.sv:218-225`), GLM's
flow east to `o_dat[i][7]` (`design.sv:279`). Direction is not constrained by
the specification, only that a chain of elements carries the value. The shared
`cyc` frame counter in both is schedule control, not an operand, so it does not
engage question 2.

**HLS.** All five build the grid from one element process instantiated per
position inside a `#pragma HLS dataflow` region, with point-to-point
`hls::stream` links. Opus splits each element into `pe_mac_*` and `pe_res_*`,
the same drain-in-its-own-unit shape as its SPMW entry; its instantiation is
the only one in the study written out fully by hand, 128 calls with no macro
or template.

## GPT-5.6 Sol in the HLS arm: the failure, classified

The interim report recorded this trial as "never correct" without separating
the causes. They are two different failures, and both are the model's.

**Builds 1 to 7 compiled, elaborated, simulated, and deadlocked.** Each
returned `STUDY RESULT TIMEOUT after 20001 cycles` having emitted 0, 0, 36, 64,
64, 17 and 64 of the 128 expected outputs. The design never drained a full
product set; the model made partial progress and lost it again.

**Build 8, its last, does not compile.** It fails at four lines that open a
loop body and then put a pragma on the same line:

    while(1) { #pragma HLS pipeline II=1

A preprocessing directive must be the first token on its line, so this is a
plain C++ syntax error, not an HLS one. Reproduced with
`clang++ -fsyntax-only -std=c++14` against stub headers, the submitted file
gives four `error: expected expression` at lines 71, 77, 83 and 89 -- the
`first`, `sink8`, `sink1` and `output` functions. The trial's other functions
place the same pragma correctly, so this was introduced in the final edit.

That last build consumed 5.1 seconds of tool time against 140 to 330 for the
builds that ran, exactly as the fail-fast staging intends, and it exhausted the
build budget at that moment. The file left on disk is therefore the broken one,
which is what the twelve-product re-measurement recorded as `BUILD_FAIL`.

The trial was stopped by builds, not tokens: 296,563 of its 500,000 tokens were
spent, and roughly 200,000 remained unused.

**What this does and does not excuse.** The syntax error decided which artifact
was graded, but not the outcome: no build in the trial ever produced 128
correct outputs, so the design would have failed on correctness regardless.
Architecturally the design is sound -- 64 accumulators, neighbour-only links, a
per-row drain chain built from `result_stage<N>` forwarding processes -- which
is why it is marked conforming above. It is counted as a failure on
correctness, as it was before.
