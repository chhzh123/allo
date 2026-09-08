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
| Budget | 500,000 billed tokens, 8 builds, 3 routing runs, or 4 hours, whichever comes first (amendments 1 and 3) |
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

---

## Amendment 1, before the first graded trial

**What changed.** The budget was 200,000 billed tokens and 25 builds. It is now
500,000 tokens, 20 builds and a four-hour wall cap.

**Why.** A trial run of the harness, thrown away and not graded, showed one
model spending 861 seconds and 50,169 tokens on a single turn of reasoning
before touching a tool. At the original cap a reasoning-heavy model would
exhaust its whole budget in one turn and never reach a build, so the token cap
would have measured reasoning verbosity rather than design ability, and it
would have done so unequally: billed tokens include reasoning tokens, and the
five models differ by an order of magnitude in how many they emit.

**What this means for the analysis.** Builds, not tokens, is now the binding
constraint in most trials, and every model gets the same twenty attempts.
Tokens are reported as a cost rather than used as a gate, and the
budget-sensitivity question is answered by the profile curve, the fraction of
trials passing against tokens spent, which does not depend on where a cap was
placed. The stop reason of every trial is recorded, so a trial stopped by
tokens, by builds or by the clock is distinguishable in the results.

**What did not change.** The task, the architecture requirement, the pass bars
and their derivation, the documentation packs, the models, and the temperature.

## Amendment 2, after a discarded launch

**What happened.** The first launch of the fifteen trials was stopped after 47
minutes and discarded. The harness kept each trial's transcript inside the
model's own working directory, so `ls` showed it and a model read it back:
one trial spent 112,000 tokens in a single turn after `cat transcript.jsonl`,
and had consumed half its budget without ever running a build.

**The fix.** The transcript and the summary now live beside the trial directory
rather than in it, and `ls` and `cat` serve only the three files a trial is
supposed to have: the task, the reference, and the design under construction.
Anything else is refused by name.

**What this does not change.** No graded result came from the discarded launch;
all fifteen trials start again from scratch. The task, the bars, the packs, the
models, the temperature and the budget are unchanged. The discarded transcripts
are kept as `discarded_launch_1/` so the reason for the restart is checkable.

## Amendment 3, before the first graded trial

**Routing is now a tool, not only a final check.** A trial may place and route
what it last built, three times, and see the lookup tables, registers,
multipliers, worst slack and unrouted nets. Measured cost on this device for a
design of this size: 399 seconds. Without it a model could not iterate on the
timing bar at all, and would be failed on a criterion it was given no way to
observe. The final grading still routes the submitted artifact once,
authoritatively, and that run is what the table reports.

**The build budget falls from 20 to 8.** Twenty attempts is more than this task
needs and makes an all-pass table likely, which would say nothing. Eight is
chosen rather than five to hedge one asymmetry: the models have seen a great
deal of SystemVerilog and Vitis HLS and none of SPMW, so a very tight budget
risks flooring the SPMW arm for unfamiliarity and producing a result about
training data rather than about the language.

**How this is reported.** Every trial records the build at which it first met
every cycle bar, so the result is reported as a profile: the fraction of trials
passing within one build, two, and so on to eight. A reading at five builds is
therefore available from the same runs, and reading at any budget does not
depend on where the cap happened to sit. Trials are told their real budget of
eight, so a five-build reading is drawn from models that were pacing for eight,
which makes it a conservative figure rather than a flattering one.

## Amendment 4, during the graded run

**Two trials are being repeated, and this records why, because repeating a
trial is exactly the move that can turn a study into a search for a result.**

`moonshotai/kimi-k3` in the HLS arm died at 3,252 tokens on
`http.client.IncompleteRead`, a transient network fault. The retry in the
agent loop caught three exception classes and that was not one of them, so a
dropped connection ended the trial. That is a harness fault with no bearing on
the model, and the trial is repeated.

`z-ai/glm-5.3` in the SPMW arm spent 615,613 tokens across turns that returned
neither text nor a tool call, then wrote its design after the budget had
already gone. Each such turn cost about 115,000 tokens. The harness noticed
nothing and kept asking. This is closer to model behaviour than to a fault,
but the loop gave it no signal that it was spending a budget on nothing, so it
is treated as a harness deficiency and repeated. The loop now stops after
three consecutive turns that produce nothing, recording `empty_replies` as the
stop reason, and records each turn's finish reason and reasoning length so the
same thing is diagnosable rather than mysterious next time.

**What is not repeated.** `deepseek/deepseek-v4-pro` in the SystemVerilog arm
used its whole token budget across four builds, reaching a correct design at
build 1 and again at build 3 and then replacing it with a wrong one both times.
That is a result, not a fault, and it stands.

**The rule applied here**, stated so it can be checked: a trial is repeated
only when the harness, not the model, ended it. Every repeat is named in this
file with its reason, and the original transcript is kept.

## Amendment 5, during the graded run

**Every request now bounds its own output.** The loop sent no `max_tokens`, so
each provider applied its own default. One model's turns ended at exactly
131,072 completion tokens with the reasoning field consuming all of them, three
turns in a row, so it returned nothing usable and spent 615,613 tokens without
producing a design. Its actual ceiling is 943,718, so the limit was a default
rather than the model's capacity.

Raising the limit alone would let a single turn consume a whole trial budget,
so the fix bounds reasoning instead: `max_tokens` 65,536 with
`reasoning.max_tokens` 32,768, applied identically to all five models and
inside the smallest completion ceiling among them, which is 128,000. A turn can
therefore always emit an answer after thinking, and no turn can cost more than
about an eighth of the trial budget.

**Both GLM trials are repeated under this amendment**, the SPMW one and the
HLS one, because both were running unbounded when they stalled: the HLS trial
reached 412,035 tokens without a single build. Its SystemVerilog trial had
already submitted a correct design before the limit bit, and stands.

### Which already-finished trials the bound would have changed

Amendment 5 arrived mid-run, so the trials that finished before it ran without
a limit. Checking every turn of every finished trial against the new one:

| Trial | Peak completion tokens in a turn | Inside the new 65,536 limit |
|---|---:|---|
| deepseek, SPMW | 20,954 | yes |
| deepseek, HLS | 38,639 | yes |
| deepseek, SystemVerilog | 49,779 | yes |
| kimi, SPMW | 32,289 | yes |
| kimi, SystemVerilog | 33,634 | yes |
| glm, SystemVerilog | **88,159** | **no** |

So the bound is inert for five of the six and would have constrained exactly one
turn in one trial. That trial, `z-ai/glm-5.3` in the SystemVerilog arm, is
repeated under the bound so that every reported trial ran under one set of
request parameters. Its unbounded transcript is kept, and if the repeat gives a
different outcome both are reported.

**Who this affects.** It is a uniform change, but not a neutral one: models that
reason at length are constrained more than terse ones. That is preferable to
the alternative, in which a model that reasons at length produces nothing at
all and is scored as having failed the task.
