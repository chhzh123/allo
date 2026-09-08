# Agentic design across three hardware representations: interim results

**Status: incomplete.** Five of fifteen trials have finished, ten are running.
Nothing here has been graded on the held-out vectors, routed, or reviewed for
architecture conformance, so no trial below is yet a pass. Read this as a
progress record, not a result.

## What is being measured

Five models write the same 8x8 output-stationary int8 systolic array, once each,
in SPMW, in C++ for Vitis HLS, and in SystemVerilog. The architecture is
mandated by the specification rather than chosen. All three arms are graded by
one testbench on one set of vectors, so a cycle and a wrong value mean the same
thing in each. Protocol, bars and their derivation are in `PREREG.md`, written
before the first graded trial, with five amendments each recording what changed
and why.

## Where the trials stand

Cycles are the best a trial has reached so far, from its own builds. The bars
are latency between 22 and 64 cycles and interval at most 16.

| Model | Arm | State | Builds | Tokens | First build meeting both bars | Latency | Interval |
|---|---|---|---:|---:|---:|---:|---:|
| Kimi K3 | SPMW | submitted | 1 | 62,073 | **1** | 35 | 14 |
| Kimi K3 | SystemVerilog | submitted | 2 | 122,374 | **1** | 32 | 16 |
| Kimi K3 | HLS | out of builds | 8 | 317,380 | never | 72 | 14 |
| DeepSeek V4 Pro | SPMW | submitted | 2 | 67,083 | 2 | 35 | 14 |
| DeepSeek V4 Pro | HLS | submitted | 2 | 187,897 | 2 | 39 | 8 |
| DeepSeek V4 Pro | SystemVerilog | out of tokens | 4 | 503,813 | 1 | 31 | 8 |
| Opus 5 | SPMW | running | 0 | 82,588 | — | — | — |
| Opus 5 | HLS | running | 1 | 55,668 | — | — | — |
| Opus 5 | SystemVerilog | running | 2 | 174,521 | 2 | 34 | 8 |
| GPT-5.6 Sol | SPMW | running | 2 | 35,967 | 2 | 34 | 14 |
| GPT-5.6 Sol | HLS | running | 2 | 74,041 | — | — | — |
| GPT-5.6 Sol | SystemVerilog | running | 4 | 201,026 | 4 | 30 | 8 |
| GLM 5.3 | SPMW | running | 3 | 135,733 | — | 33 | 30 |
| GLM 5.3 | HLS | running | 2 | 57,330 | — | 35 | 24 |
| GLM 5.3 | SystemVerilog | running | 1 | 60,695 | — | — | — |

## What can already be said, and what cannot

**A model that reaches the bars in SPMW does so on its first or second build.**
Every SPMW trial that has met both bars did it at build 1 or 2, at 36,000 to
67,000 tokens. No arm has been cheaper in tokens on any model that finished it.

**Three models independently produced the same SPMW design point**: latency 35,
interval 14, which is also what a hand-written reference gets. That is a
convergence worth reporting whichever way the rest lands.

**SystemVerilog reaches the best cycle counts.** Latencies of 30 to 34 with an
interval of 8, at the port-bandwidth floor, appear in four SystemVerilog trials.
Whether those designs conform to the mandated architecture is exactly what the
review will decide; an interval of 8 is achievable systolically, but it is also
what a globally connected multiplier array would give.

**HLS is where trials go wrong.** Kimi spent all eight builds and 317,000 tokens
to reach a correct design at latency 72 and interval 32, missing both bars.

**Stopping early is a real difference between models.** Three trials submitted
after one or two builds. DeepSeek's SystemVerilog trial reached a design meeting
both bars at build 1, replaced it, recovered, broke it again, and exhausted its
budget without submitting.

None of this is a result yet. A trial that meets the bars in its own simulation
may still fail on held-out vectors, fail to route at 300 MHz, or be found
non-conforming on reading. Those three checks are what turn the table above into
findings.

## Honesty about the harness

The study took four launches to get running, and the first three were killed by
faults of mine, not by the models: a login shell that could block, the
transcript sitting inside the model's own working directory where one model read
it back for 112,000 tokens, a patch that accumulated tool time before computing
it and so killed every agent at its first build, a missing timescale flag that
failed every SystemVerilog design regardless of quality, error text swallowed so
models debugged blind, and an unset output limit that let one model spend all
131,072 of its output tokens on reasoning and return nothing.

All are fixed and each is recorded. Three trials were repeated because the
harness, not the model, ended them; the rule and every repeat are named in
`PREREG.md`. The discarded launches are kept.

## Files

- `PREREG.md` protocol and amendments
- `task/` the specification, the conformance criteria, the grading testbench, and the vectors
- `docs/` the three documentation packs, matched by size and by worked example
- `arms/` build-and-grade for each language
- `harness/` the agent loop, the prompt template, the interface stubs, the runners
- `designs/` what each trial actually wrote
- `results/` per-trial and per-build rows
