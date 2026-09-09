# Agentic design across three hardware representations: interim results

**Status: all fifteen trials have run, been re-measured over twelve unseen
products, and been routed on the device. The architecture review covers nine of
fifteen.** The six unreviewed designs could still be demoted, and nothing else
is outstanding.

## The question

Five models write the same 8x8 output-stationary int8 systolic array, once each,
in SPMW, in C++ for Vitis HLS, and in SystemVerilog. The architecture is
mandated by the specification rather than chosen, so this measures the cost of
expressing a fixed design and getting it right, not the cost of inventing one.

## How a trial is set up

**One prompt template, five slots.** The arms differ only in the language name,
the file to write, the top-level symbol, the reference pack and the build
command. Everything else is byte-identical. This is what a SPMW trial received:

> You are designing digital hardware. Your task is in `TASK.md` in your working
> directory. Read it first, in full.
>
> You write **SPMW, a Python-embedded language for spatial hardware**. Put your
> design in the single file **`design.py`**, which must define **a fabric named
> `engine`**. A starting file is already there with the interface declared and
> the body empty; fill it in. Write no other file. The test harness and its
> vectors already exist, and you must not modify them.
>
> Reference documentation for the language is in `REFERENCE.md`. It is the only
> documentation you have.
>
> You have exactly two tools: `write_file(path, content)` and `run(command)`.
> The commands that run are `build`, `route`, `cat <path>` and `ls`. Nothing
> else runs. You have no network access, and no access to any file outside your
> working directory.
>
> Your budget is 500,000 tokens, 8 builds and 3 routing runs. All three are
> counted for you and reported after every build.
>
> When you believe your design meets every requirement in `TASK.md`, write
> SUBMIT on a line by itself. Your design is then checked against vectors you
> have not seen and routed on the target device. You will not see those results,
> so satisfy yourself with the tools you have before submitting.

The bolded phrases are the only slots. The other arms read "C++ for AMD Vitis
HLS", `design.cpp`, "a function named `gemm_tile`", and "SystemVerilog",
`design.sv`, "a module named `dut_norm`".

**The tools.** `build` compiles, elaborates and simulates against the visible
vectors, reporting wrong values, latency, interval and per-stage timings.
`route` places and routes the last build on the real device and reports lookup
tables, registers, multipliers and worst slack; about 400 seconds, three per
trial. `cat` and `ls` serve only the task, the reference and the design file.

**The API.** OpenRouter chat completions, native tool calling, with a text-block
fallback recorded whenever a model answers in prose instead of a call.
Temperature 0. Each request bounded at 65,536 output tokens of which at most
32,768 may be reasoning, applied identically to all five and inside the smallest
completion ceiling among them. Tokens are as OpenRouter bills them, reasoning
included. Slugs: `anthropic/claude-opus-5`, `openai/gpt-5.6-sol`,
`moonshotai/kimi-k3`, `z-ai/glm-5.3`, `deepseek/deepseek-v4-pro`.

**Documentation parity.** Each arm gets its own complete API reference, within
four percent of the others by size, plus one worked example of the same
unrelated design, a four-stage scaling chain, in its own language. No arm gets
an example resembling a systolic array.

## What the numbers can be

There is no upper bound: a correct design can be arbitrarily slow, which is why
the task states thresholds. The floors follow from the setting.

| Quantity | Floor | Why |
|---|---:|---|
| Interval | 8 cycles | each of the 16 ports carries 8 values per product, one per cycle |
| Latency | about 29 cycles | `PE(7,7)` sees its last operand no earlier than cycle 22, then leaves eighth on its row's chain |

Thresholds are twice each floor: interval at most 16, latency at most 64. The
best trial reaches latency 30 and interval 8, which is the floor exactly.

## A measurement fault, and what it changed

The visible vectors carry **two** products. That is not a steady state: a design
with deep input buffering accepts the second product quickly and only backs up
later. Every design was therefore re-measured over **twelve** products, and the
correction is large.

| Arm | Met the bars on two products | Met them on twelve | Median tokens |
|---|---|---|---:|
| SPMW | 5 of 5 | **2 of 5** | 67,083 |
| SystemVerilog | 5 of 5 | **5 of 5** | 503,813 |
| HLS | 3 of 5 | **2 of 5** | 300,414 |

This is my fault, not the models'. They optimised against the signal the harness
showed them and met the bar they were given. Both numbers are reported for every
design, and the twelve-product figure is the one that counts.

## Every design, fully graded

Correctness and cycles from twelve products the models never saw; multipliers,
logic and slack from routing the submitted design on the device at 3.333 ns.
A design passes only if it clears every bar.

| Model | Arm | Builds | Tokens | Latency | Interval | DSP | Slack | Passes | Failed bar |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Opus 5 | SPMW | 1 | 107,350 | 33 | 14 | 64 | +1.364 | **yes** | |
| GPT-5.6 Sol | SPMW | 2 | 44,014 | 34 | 14 | 64 | +1.288 | **yes** | |
| DeepSeek | SPMW | 2 | 67,083 | 35 | 25 | 64 | +1.399 | no | interval |
| Kimi K3 | SPMW | 1 | 62,073 | 35 | 25 | 64 | +1.274 | no | interval |
| GLM 5.3 | SPMW | 5 | 205,076 | 53 | 28 | 64 | +1.259 | no | interval |
| Opus 5 | SystemVerilog | 2 | 204,401 | 34 | 8 | 64 | +2.061 | **yes** | |
| Kimi K3 | SystemVerilog | 2 | 122,374 | 32 | 16 | 64 | +2.004 | **yes** | |
| GPT-5.6 Sol | SystemVerilog | 7 | 524,554 | **30** | **8** | 64 | +1.331 | **yes** | |
| GLM 5.3 | SystemVerilog | 5 | 527,420 | 33 | 16 | 64 | +1.763 | **yes** | |
| DeepSeek | SystemVerilog | 4 | 503,813 | 32 | 8 | **0** | +1.050 | no | multipliers |
| Opus 5 | HLS | 3 | 300,414 | 32 | 13 | 64 | +0.819 | **yes** | |
| GLM 5.3 | HLS | 8 | 478,484 | 44 | 15 | 64 | +1.054 | **yes** | |
| DeepSeek | HLS | 2 | 187,897 | 39 | 24 | 64 | +1.031 | no | interval |
| Kimi K3 | HLS | 8 | 317,380 | 74 | 32 | 64 | +1.448 | no | latency, interval |
| GPT-5.6 Sol | HLS | 8 | 296,563 | — | — | — | — | no | never correct |

| Arm | Passes | Median tokens | Median builds |
|---|---|---:|---:|
| SystemVerilog | **4 of 5** | 503,813 | 4 |
| SPMW | 2 of 5 | **67,083** | **2** |
| HLS | 2 of 5 | 300,414 | 8 |

## What this says, honestly

**SystemVerilog produces the best hardware and costs the most to get there.**
Four of five pass, three at the interval floor of 8, and the best latency of 30
is within one cycle of the architectural floor. The median trial spent 503,813
tokens, seven and a half times SPMW's, and two trials exhausted their budget.
Its one failure is instructive: DeepSeek's design clears correctness, latency
and interval but routes to **zero** multipliers, because a plain signed multiply
in SystemVerilog is inferred as logic unless the design says otherwise. Three of
the five hit that at least once; only Opus reached exactly 64 on its first try.

**SPMW is far the cheapest and only two of five pass.** Every SPMW trial reached
a correct design, four of them within two builds, at a median of 67,083 tokens.
Every SPMW design routes with exactly 64 multipliers and over 1.2 ns of slack
without being asked, because the multiplier comes from the language rather than
from inference. All three failures are the same bar: sustained interval.

**The failing SPMW designs differ from the passing ones by where the unit
boundary is drawn.** SPMW's concurrency comes from unit boundaries, so anything
inside one element runs in sequence. Opus and GPT-5.6 Sol put the result drain
in its own unit, which the fabric then runs concurrently with the next product's
accumulation. DeepSeek, Kimi and GLM inlined the drain into the element, which
serialises eight multiply-accumulate steps behind up to seven forwards. GLM is
the sharpest case: its comment says the drain should overlap the next product,
and it wrote the overlap inside the element's own loop, and got the worst
interval of the five. The hand-written reference makes the same mistake.

That is a finding about the documentation as much as the models. The reference
pack says units run concurrently but never says that concurrency follows unit
boundaries, so a reader who keeps one element in one unit gets a correct design
that does not pipeline.

**HLS is the worst arm on every count.** Two of five pass, three ran out of
builds, and one never produced a correct design in eight attempts and 296,563
tokens.

**Stopping early separates models more than language does.** Three trials
submitted after one build. DeepSeek's SystemVerilog trial met the bars at build
1, replaced that design, recovered, broke it again, and exhausted its budget
without ever submitting.

## Architecture review

Done for the four designs that showed an interval of 8. **All four conform**:
64 elements each with one multiplier and one accumulator, operands arriving only
from a neighbour or an edge port, A east and B south one element per cycle,
results leaving through a per-row chain, and the grid instantiated rather than
written out. The three SystemVerilog ones reach interval 8 by double-buffering
the result register, which is legitimate and is exactly the technique the
failing SPMW designs lack.

Eleven designs remain to review, including every SPMW design.

## What is still missing

1. Architecture review for six of the fifteen designs: the four fastest and the
   five SPMW ones have been read, the rest have not. A design that computes the
   right answer without being systolic would still be counted as a pass in the
   table above.
2. `openai/gpt-5.6-sol` in the HLS arm is recorded as never correct. Its final
   design fails to build on the twelve-product set, and the cause has not been
   separated from the eight failures it had during the trial.

## Honesty about the harness

Four launches. The first three were killed by faults of mine, not by any model:
a login shell that could block; the transcript inside the model's own working
directory, where one model read it back for 112,000 tokens; a patch that
accumulated tool time before computing it and killed every agent at its first
build; a missing timescale flag that failed every SystemVerilog design
regardless of quality; error text swallowed so models debugged blind; and an
unset output limit that let one model spend all 131,072 of its output tokens on
reasoning and return nothing. Then the two-product interval above.

All are fixed and recorded in `PREREG.md`. Four trials were repeated, only ever
because the harness rather than the model ended them, and each is named there.
