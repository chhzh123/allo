# Agent study: one 8x8 systolic array in three languages

Fig. 14 of the paper (arXiv version).

Five models through OpenRouter, at temperature 0, each write the same 8x8
output-stationary int8 systolic array with int32 accumulation, once in SPMW,
once in C++ for Vitis HLS and once in SystemVerilog: fifteen trials, one per
model and language. Each trial gets the specification, an interface stub and
one language reference (7,446 / 7,490 / 7,796 characters), is built and graded
by one shared testbench, and can only write its design file and run `build`,
`route`, `cat` and `ls`. The budget, as every transcript records it, is
500,000 billed tokens, 8 builds, 3 routing runs and 4 hours (the paper's text
says 1M tokens). After a trial
stops, the design it stopped on is simulated over twelve products it never saw
and routed out of context on `xcu280-fsvh2892-2L-e` at 3.333 ns (300 MHz).

## Fig. 14

| Framework | Meets latency | Median tokens | Median builds | Median time (min) |
|---|---:|---:|---:|---:|
| SPMW | 5/5 | 67,083 | 2 | 20.5 |
| C++ HLS | 3/5 | 300,414 | 8 | 37.1 |
| SystemVerilog | 5/5 | 503,813 | 4 | 26.6 |

First-product latency in cycles and billed tokens, per trial (the left panel):

| Model | SPMW | C++ HLS | SystemVerilog |
|---|---|---|---|
| Opus 5 | 33, 107,350 | 32, 300,414 | 34, 204,401 |
| GPT-5.6 Sol | 34, 44,014 | does not build, 296,563 | 30, 524,554 |
| Kimi K3 | 35, 62,073 | **74**, 317,380 | 32, 122,374 |
| GLM-5.3 | 53, 205,076 | 44, 478,484 | 33, 527,420 |
| DeepSeek-V4-Pro | 35, 67,083 | 39, 187,897 | 32, 503,813 |

Every number is a column of [`results/results.csv`](results/results.csv)
(`results.json` holds the same rows):

| Paper | Column | Read from |
|---|---|---|
| meets latency | `steady_result` = CORRECT and 22 <= `steady_latency` <= 64 | `runs/steady_all.log` |
| tokens, builds | `tokens`, `builds_used`: billed, reasoning included, through trial stop | `runs/transcripts/<trial>.summary.json` |
| time | `wall_seconds` / 60: model calls and tools through trial stop | the same |
| 4.48x, 7.51x tokens; 1.81x, 1.29x time | ratios of the medians | |
| all fourteen buildable route at 300 MHz; every SPMW design has 64 DSPs | `routed_wns` >= 0, `routed_dsp` | `runs/steady/<trial>/pnr/{util,timing}.rpt` |

No kept script writes the `routed_*`, `passes` and `failed_bars` columns; they
match the routing reports, and the paper's plot script re-reads those reports
and `runs/steady_all.log` and asserts both against this table.

## Why three trials end above 500,000 tokens

`agent.py` checks the limits before each model call, not during one, and bills
each call for the whole conversation so far (prompt tokens are 70% of all
billed tokens here). The call that crosses 500,000 is counted in full, and its
tool calls still run before the loop stops. Three SystemVerilog trials crossed
the limit on their last call, each from below it:

| Trial | Before last call | Last call (prompt + output) | Total | Last call did |
|---|---:|---:|---:|---|
| DeepSeek-V4-Pro | 454,799 | 35,007 + 14,007 | 503,813 | rewrote `design.sv`, then stopped on tokens |
| GPT-5.6 Sol | 484,395 | 40,030 + 129 | 524,554 | wrote SUBMIT |
| GLM-5.3 | 468,782 | 34,899 + 23,739 | 527,420 | rewrote `design.sv`, then stopped on tokens |

That is why the SystemVerilog median, DeepSeek's 503,813, is above the limit.
For DeepSeek and GLM the graded file is that last rewrite, which no build in
the trial ran:

- **DeepSeek's** last build (4) was WRONG on the visible products. The rewrite
  moves the drain one cycle earlier and adds a capture register, and it is
  the CORRECT 32-cycle design Fig. 14 counts. Graded on its last built design,
  the trial would miss the latency criterion and SystemVerilog would be 4/5.
- **GLM's** last build (5) was CORRECT at 33 cycles. The rewrite only adds
  `use_dsp` attributes (product to a DSP, accumulator to fabric). It routes
  with 64 DSPs; the model's last route, of build 4's design, had 128.

The last built designs are at commit c20b2ee8, in
`experiments/agentstudy/runs/generated/<trial>/sim/design.sv`.

## Two criteria

The paper counts the **latency criterion**: correct on all twelve products and
latency 22 to 64 cycles. The pre-registered bars (`PREREG.md`) also require a
sustained interval of at most 16 cycles (the largest of the last four), exactly
64 DSPs, at most 25,000 LUTs and non-negative slack. Under those, the `passes`
column gives **2/5 SPMW, 2/5 HLS, 4/5 SystemVerilog**: three SPMW and two HLS
designs miss the interval (Kimi's HLS also the latency), DeepSeek's
SystemVerilog routes with 0 DSPs, and GPT-5.6 Sol's HLS does not compile.
`meets_bars_visible` says whether any build met the cycle bars on the two
visible products, the signal the models saw. All fifteen designs conform to the architecture
([`results/CONFORMANCE.md`](results/CONFORMANCE.md), against the five questions
in [`task/CONFORMANCE.md`](task/CONFORMANCE.md)).

## Layout

Trials are named `<provider>_<model>__<arm>`; arm `rtl` is SystemVerilog.

| Path | Contents |
|---|---|
| `task/` | `SPEC.md` (each trial's `TASK.md`), `tb_study.sv` (one testbench for every arm), `CONFORMANCE.md`, `vectors/` (visible: 2 products, held-out: 4, steady: 12) |
| `docs/{spmw,hls,rtl}/REFERENCE.md` | the three language references |
| `arms/{spmw,hls,rtl}/` | each arm's `build`: compile, elaborate, simulate against `tb_study.sv` |
| `designs/<trial>/` | the 15 designs the trials stopped on |
| `runs/transcripts/` | per trial `.transcript.jsonl`, runner `.log` and `.summary.json` (45 files), and `run_all.log` |
| `runs/steady_all.log` | the twelve-product re-measurement of all fifteen |
| `runs/steady/<trial>/` | its inputs and outputs: the design (identical to `designs/`), HLS logs, xsim logs for HLS and SystemVerilog, and `pnr/` (none for GPT-5.6 Sol's HLS) |
| `results/` | `results.csv`, `results.json`, `results_per_build.csv` (60 builds), `CONFORMANCE.md` |
| [`PREREG.md`](PREREG.md) | the pre-registration and its dated amendments, including the four repeated trials |
| [`INTERIM.md`](INTERIM.md) | the full write-up, despite its name final: prompt, tools, every design fully graded, the reading of the results |

| `harness/` | Role |
|---|---|
| `agent.py` | one trial: the model loop, the two tools, the budgets, the transcript |
| `run_trial.sh`, `run_all.sh` | one trial; all fifteen, one model at a time with its three arms concurrently |
| `prompt_common.md`, `arms.json`, `stubs/` | the prompt template, its per-arm slots, the interface stubs |
| `long_vectors.py` | writes the twelve-product `steady.json` and `steady_{a,b,c}.hex` from fixed seeds |
| `steady_all.sh` | simulates every stopped design over those twelve products |
| `pnr_ooc.sh` | out-of-context place and route at 3.333 ns; each `pnr/pnr.tcl` is its output |
| `grade.sh` | held-out vectors, then routing: the grading planned before the re-measurement |
| `collect_trials.py`, `merge_steady.py`, `steady_table.py` | transcripts to `results.csv`; folds `steady_all.log` in; prints the log as a table |

## Reproducing

The harness and the arms name absolute paths on brg-zhang-xcel (only
`agent.py` finds the study root from its own location): the study root
`/scratch/hc676/agentstudy` (this folder, with `trials/` and `steady/` beside
`harness/`), the Allo checkout `/scratch/hc676/allo`, its tool directory
`/scratch/hc676/allo-agent/bin` and the Vitis 2023.2 setup
`/work/shared/common/allo/vitis_2023.2_u280.sh`. Copy the folder there, or edit
those paths first. With `S=/scratch/hc676/agentstudy`:

    bash harness/run_all.sh                          # needs ~/.openrouter_key, mode 600
    python3 harness/collect_trials.py $S/trials $S/results
    python3 harness/long_vectors.py                  # steady.json is byte-identical to the kept one
    bash harness/steady_all.sh > $S/steady_all.log
    bash harness/pnr_ooc.sh $S/steady/<trial>/sim dut_norm $S/steady/<trial>/pnr   # build/sim for SPMW
    python3 harness/merge_steady.py

The first step runs new trials, not these fifteen; the steps after it
re-measure whatever `$S/trials` holds. To re-measure the fifteen here, copy
`designs/<trial>/` to `$S/trials/<trial>/` and start from `long_vectors.py`.

## Not kept

Commit c20b2ee8 has what the cleanup removed: `runs/generated/` (each trial's
build tree, including the routing reports the model was shown),
`runs/rounds/` (per-trial timelines), `runs/transcripts/repeated/` (the four
repeated trials), the grading and salvage logs, duplicate result files under
`runs/`, and 35 one-off harness patch scripts. `PREREG.md`'s
`discarded_launch_1/` was never committed.
