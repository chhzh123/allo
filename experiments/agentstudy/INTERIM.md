# Agentic design across three hardware representations: interim results

**Status: thirteen of fifteen trials finished, two running. Nothing has been
graded yet.** No trial below has been checked on the held-out vectors, routed at
300 MHz, or read against the architecture requirement, and any of those three
can still overturn a row. Read this as a progress record.

## The question

Five models write the same 8x8 output-stationary int8 systolic array, once each,
in three languages: SPMW, C++ for Vitis HLS, and SystemVerilog. The architecture
is mandated by the specification rather than chosen, so this measures the cost
of expressing a fixed design and getting it right, not the cost of inventing one.

## How a trial is set up

**One prompt template, five slots.** The arms differ only in the language name,
the file to write, the top-level symbol, the reference pack and the build
command. Everything else is byte-identical. This is what a SPMW trial actually
received:

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

The bolded phrases are the only slots. The other two arms read "C++ for AMD
Vitis HLS", `design.cpp`, "a function named `gemm_tile`", and "SystemVerilog",
`design.sv`, "a module named `dut_norm`".

**What the tools do.** `build` compiles, elaborates and simulates the design
against two visible products, reporting wrong values, first-product latency and
steady-state interval, plus per-stage timings. `route` places and routes the
last build on the real device and reports lookup tables, registers, multipliers
and worst slack; it costs about 400 seconds and each trial gets three. `cat` and
`ls` serve only the task, the reference and the design file, and refuse anything
else by name.

**Interface stubs.** Each arm starts from a file with the ports already declared
and the body empty, so no arm spends tokens transcribing a 74-port list.

**The API.** OpenRouter's chat completions endpoint, native tool calling, with a
text-block fallback recorded whenever a model answers in prose instead of a call.
Temperature 0. Each request is bounded at 65,536 output tokens of which at most
32,768 may be reasoning, applied identically to all five and inside the smallest
completion ceiling among them. Billed tokens are counted as OpenRouter reports
them, reasoning included.

| Model | OpenRouter slug |
|---|---|
| Opus 5 | `anthropic/claude-opus-5` |
| GPT-5.6 Sol | `openai/gpt-5.6-sol` |
| Kimi K3 | `moonshotai/kimi-k3` |
| GLM 5.3 | `z-ai/glm-5.3` |
| DeepSeek V4 Pro | `deepseek/deepseek-v4-pro` |

**Documentation parity.** Each arm gets its own complete API reference, within
four percent of the others by size (7,446 / 7,490 / 7,796 characters), plus one
worked example of the same unrelated design, a four-stage scaling chain, in its
own language. No arm gets an example resembling a systolic array.

## What the numbers can be

There is no upper bound on latency or interval: a correct design can be
arbitrarily slow, which is why the task states thresholds. The floors, however,
follow from the ports and the architecture, and they are what the thresholds
were derived from.

**Interval, floor 8 cycles.** Each of the sixteen input ports carries eight
values per product and can transfer one per cycle, so products cannot start
closer together than eight cycles.

**Latency, floor about 29 cycles.** A value moves one element per cycle, so
`PE(7,7)` cannot see its first operands before cycle 15 nor its last before
cycle 22, and `C[7][7]` cannot be final earlier than that. It then leaves
through its row's chain, where it is the eighth value on port 7, so it reaches
the boundary no earlier than about cycle 29.

The thresholds are twice each floor: interval at most 16, latency at most 64.
The specification also states a latency floor of 22, which is the compute-only
bound; it is deliberately conservative, and the architecture review rather than
the floor is what catches a design that moves data further than a neighbour.

**Against that, the best trials are at the limit.** The lowest latency reached
is 30 against a floor of about 29, and an interval of 8 has been reached in six
trials, which is the floor exactly.

## Where the trials stand

Cycles are the best each trial reached in its own builds.

| Model | Arm | Stopped by | Builds | Tokens | First build meeting both bars | Latency | Interval |
|---|---|---|---:|---:|---:|---:|---:|
| Kimi K3 | SPMW | submit | 1 | 62,073 | **1** | 35 | 14 |
| GPT-5.6 Sol | SPMW | submit | 2 | 44,014 | 2 | 34 | 14 |
| DeepSeek | SPMW | submit | 2 | 67,083 | 2 | 35 | 14 |
| Opus 5 | SPMW | submit | 1 | 107,350 | **1** | 33 | 14 |
| GLM 5.3 | SPMW | submit | 5 | 205,076 | 4 | 53 | 14 |
| Kimi K3 | SystemVerilog | submit | 2 | 122,374 | **1** | 32 | 16 |
| Opus 5 | SystemVerilog | submit | 2 | 204,401 | 2 | 34 | **8** |
| GPT-5.6 Sol | SystemVerilog | submit | 7 | 524,554 | 4 | **30** | **8** |
| GLM 5.3 | SystemVerilog | running | 5 | 468,782 | 2 | 31 | 16 |
| DeepSeek | SystemVerilog | out of tokens | 4 | 503,813 | 1 | 31 | **8** |
| Opus 5 | HLS | submit | 3 | 300,414 | 3 | 32 | 13 |
| DeepSeek | HLS | submit | 2 | 187,897 | 2 | 39 | **8** |
| GLM 5.3 | HLS | out of builds | 8 | 478,484 | 8 | 44 | 15 |
| Kimi K3 | HLS | out of builds | 8 | 317,380 | never | 72 | 14 |
| GPT-5.6 Sol | HLS | out of builds | 8 | 296,563 | never | — | — |

## What is visible so far

**Every model reached the bars in SPMW, and four of five did it within two
builds.** Median cost 67,083 tokens. All five submitted.

**SystemVerilog also reaches the bars on every model, and gets closer to the
floors**, latency 30 to 34 with an interval of 8 in three trials, but it costs
far more: median 204,401 tokens, and two trials spent over half a million. Two
models needed five and seven builds.

**HLS is where trials fail.** Three of five ran out of builds. Two never reached
the bars at all, one after eight builds and 317,380 tokens.

**A design difference worth the review's attention.** Every SPMW trial converged
on an interval of 14 and none reached 8, while six trials in the other two arms
did. An interval of 8 requires overlapping a product's drain with the next
product's accumulation; the SPMW designs all chose a simpler drain that does
not. Whether that is a property of the language, of its documentation, or of
how the models read it is not something these numbers answer.

**Stopping early separates models more than language does.** Three trials
submitted after one build. DeepSeek's SystemVerilog trial met both bars at build
1, replaced that design, recovered, broke it again, and exhausted its budget
without submitting.

None of this is a result yet. The interval-8 designs are exactly the ones the
conformance review must scrutinise, because eight cycles is also what a
globally connected multiplier array would achieve, and that is not the mandated
architecture.

## Honesty about the harness

The study took four launches. The first three were killed by faults of mine, not
by any model: a login shell that could block; the transcript sitting inside the
model's own working directory, where one model read it back for 112,000 tokens;
a patch that accumulated tool time before computing it and so killed every agent
at its first build; a missing timescale flag that failed every SystemVerilog
design regardless of quality; error text swallowed so models debugged blind; and
an unset output limit that let one model spend all 131,072 of its output tokens
on reasoning and return nothing.

All are fixed and each is recorded in `PREREG.md` with its amendment. Four
trials were repeated, only ever because the harness rather than the model ended
them, and each repeat is named there. The discarded launches are kept.

## Files

- `PREREG.md` protocol and five amendments
- `task/` specification, conformance criteria, grading testbench, vectors
- `docs/` the three documentation packs
- `arms/` build-and-grade for each language
- `harness/` agent loop, prompt template, stubs, runners
- `designs/` what each trial wrote
- `results/` per-trial and per-build rows
