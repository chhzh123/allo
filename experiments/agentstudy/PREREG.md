# Pre-registration: agentic design across three hardware representations

Written before the first graded trial. Anything decided after this point is
recorded as an amendment with its date and reason.

## Question

Given the same fixed architecture, the same specification and the same budget,
what does it cost a coding agent to reach a correct, fast-enough hardware
design in SPMW, in C++ for Vitis HLS, and in SystemVerilog?

This is a fixed-architecture implementation study, not an architecture search.
The specification mandates the structure; the measurement is the cost of
expressing it and getting it right.

## Design

One task. Three arms, differing only in language. Five models, one trial per
model per arm, so fifteen graded trials. **The model is the replication unit**:
with one trial per cell there is no within-cell variance estimate, and the
analysis is five paired comparisons, reported with a sign test. A success rate
per arm is not claimed.

| Frozen before the first trial | Value |
|---|---|
| Models | `anthropic/claude-opus-5`, `openai/gpt-5.6-sol`, `moonshotai/kimi-k3`, `z-ai/glm-5.3`, `deepseek/deepseek-v4-pro` |
| Temperature | 0.0 |
| Budget | 200,000 billed tokens, or 25 builds, whichever comes first |
| Task | 8x8 output-stationary int8 systolic array, one 8x8 by 8x8 product |
| Device | xcu280-fsvh2892-2L-e, 3.333 ns |

## Pass bars, and where they come from

| Bar | Value | Derivation |
|---|---|---|
| Correctness | every value exact on held-out vectors | |
| Latency | 22 to 64 cycles, first product after reset | floor: a value cannot cross the grid faster than one element a cycle. ceiling: twice the port-bandwidth bound |
| Throughput | at most 16 cycles between product starts | twice the 8-cycle port-bandwidth floor |
| Multipliers | exactly 64 | one per element, mandated |
| Logic | at most 25,000 lookup tables | stops unrolling from buying latency |
| Timing | routes out of context with non-negative slack | |

Both upper bars follow one rule, twice the floor the ports impose, fixed before
any design was written. A calibration build confirmed the mandated architecture
reaches latency 35 and interval 14 in **both** Vitis HLS and SPMW, so the bars
are reachable in more than one arm. Those calibration designs are thrown away
and are not reported.

## Fairness controls

- **Documentation parity.** Each arm gets its own complete API reference,
  trimmed to what the task needs, capped within four percent of the others by
  size (7,446 / 7,490 / 7,796 characters), plus one worked example of the same
  unrelated design, a four-stage scaling chain, in its own language. No arm
  gets an example resembling a systolic array.
- **A decision recorded rather than hidden.** The SPMW reference documents
  `spmw.mesh`, the helper that wires a two-dimensional grid, because the other
  two references likewise document their most powerful constructs
  (`#pragma HLS dataflow`, `generate`). Omitting an API from its own reference
  would be the larger thumb on the scale. This is the most attackable line in
  the setup and is flagged as such.
- **One harness, one testbench.** Every arm is graded by the same SystemVerilog
  testbench on the same vectors, so a cycle and a wrong value mean the same
  thing in all three.
- **Interface stubs.** Each arm starts from a file with the ports declared and
  the body empty, so no arm spends tokens transcribing a port list.
- **Held-out vectors.** The agent sees two products; grading uses four it has
  never seen, with the int8 extremes pinned.
- **Sealed working directory.** Two tools, no network beyond the model API, no
  reference implementation on disk, no access outside the trial directory.

## What is measured

Per trial: whether it submitted, whether it passed each bar, billed tokens,
builds used, model seconds and tool seconds. Per segment, meaning the work
between two consecutive builds: tokens, model seconds, tool seconds, and the
time inside each tool stage separately, so a trial reads as a sequence of
attempts rather than one total.

For every failed build, where the error surfaced: at compile, at elaboration,
at synthesis, or in simulation. That distribution is the mechanism the study is
really about, and it is reported whatever the headline numbers do.

## Conformance

A harness cannot check that a design is systolic. All fifteen artifacts are
read against five written questions (`task/CONFORMANCE.md`), and a design that
fails any of the first four is reported non-conforming, counted separately, and
never counted as a success however fast it is.

## What this study cannot say

With one trial per cell, nothing about within-cell variance. With one task,
nothing about other kinds of design. And SPMW is absent from every model's
training data while the other two are abundant in it, which no protocol
removes; the documentation cap bounds it but does not eliminate it.
