# Agentic design across three hardware representations: interim results

**Status: all fifteen trials have run and been re-measured over twelve
products. Three checks remain: held-out vectors, authoritative routing, and the
architecture review for eleven of the fifteen designs.** Any of those can still
change a row.

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

## Every design, measured over twelve products

| Model | Arm | Builds | Tokens | Latency | Interval, first | Interval, sustained | Meets bars |
|---|---|---:|---:|---:|---:|---:|---|
| Opus 5 | SPMW | 1 | 107,350 | 33 | 14 | 14 | **yes** |
| GPT-5.6 Sol | SPMW | 2 | 44,014 | 34 | 14 | 14 | **yes** |
| DeepSeek | SPMW | 2 | 67,083 | 35 | 14 | 25 | no |
| Kimi K3 | SPMW | 1 | 62,073 | 35 | 14 | 25 | no |
| GLM 5.3 | SPMW | 5 | 205,076 | 53 | 14 | 28 | no |
| GPT-5.6 Sol | SystemVerilog | 7 | 524,554 | **30** | 8 | **8** | **yes** |
| Opus 5 | SystemVerilog | 2 | 204,401 | 34 | 8 | **8** | **yes** |
| DeepSeek | SystemVerilog | 4 | 503,813 | 32 | 8 | **8** | **yes** |
| Kimi K3 | SystemVerilog | 2 | 122,374 | 32 | 16 | 16 | **yes** |
| GLM 5.3 | SystemVerilog | 5 | 527,420 | 33 | 16 | 16 | **yes** |
| Opus 5 | HLS | 3 | 300,414 | 32 | 13 | 13 | **yes** |
| GLM 5.3 | HLS | 8 | 478,484 | 44 | 15 | 15 | **yes** |
| DeepSeek | HLS | 2 | 187,897 | 39 | 8 | 24 | no |
| Kimi K3 | HLS | 8 | 317,380 | 74 | 32 | 32 | no |
| GPT-5.6 Sol | HLS | 8 | 296,563 | — | — | — | no, never correct |

## What this says, honestly

**SystemVerilog wins on the hardware.** Five of five designs sustain the bars,
three of them at the interval floor of 8, and the best latency of 30 is within
one cycle of the architectural floor. It costs the most: a median of 524,554
tokens against SPMW's 67,083, and two trials exhausted their token budget.

**SPMW is by far the cheapest but only two of five designs sustain.** Every
SPMW trial reached a correct design, four within two builds, at a median of
67,083 tokens, an eighth of the SystemVerilog cost. But three of them sustain an
interval of 25 to 28 rather than the 14 they showed on two products.

**The difference between the passing and failing SPMW designs is one technique.**
Reaching a sustained interval needs a product's drain overlapped with the next
product's accumulation. Opus and GPT-5.6 Sol did that; the other three wrote a
drain chain that serialises after the multiply-accumulate phase, which is also
what the hand-written reference does. So the bar is reachable in SPMW, and three
models did not reach it.

**HLS is the worst arm on every count.** Two of five sustain, three ran out of
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

1. Held-out vector grading for all fifteen. Every design so far has only been
   checked on products it could see while working.
2. Authoritative routing for all fifteen, which decides the timing and area
   bars. Models could route themselves and eleven did at least once, but the
   post-submission route has not been run.
3. Architecture review for eleven of the fifteen.
4. `openai/gpt-5.6-sol` in the HLS arm fails to build on the twelve-product set
   and needs its failure classified.

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
