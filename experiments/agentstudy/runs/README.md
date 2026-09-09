# Raw runs: transcripts, logs and generated hardware

Everything the fifteen trials produced that is worth keeping, copied from
`/scratch/hc676/agentstudy` on brg-zhang-xcel. 1,866 files, 36 MB.

## What is here

| Path | Contents |
|---|---|
| `transcripts/` | 15 `.transcript.jsonl`, the complete turn-by-turn record of each trial; the runner's stdout `.log`; the `.summary.json` each trial ended with |
| `transcripts/repeated/` | the four trials that were repeated because a harness fault ended them, named in `PREREG.md` |
| `rounds/` | one Markdown timeline per trial, rendered from its transcript: every model turn, every tool call, every build verdict and every routing result in order |
| `generated/<trial>/` | the design the trial ended on, the hardware the tools generated from it, the simulation logs, and the three routing runs with the reports the model was shown |
| `steady/<trial>/` | the twelve-product re-measurement and its place-and-route: logs and reports only, since it rebuilds the same final design |
| `results*.csv`, `results.json` | the graded tables, also in `../results/` |
| `*.log` | the top-level driver logs for the grading, salvage and steady passes |

Inside `generated/<trial>/`: `route_N/` holds `pnr.log`, `util.rpt`,
`timing.rpt`, `clockInfo.txt` and the `pnr.tcl` that produced them -- `util.rpt`
and `timing.rpt` are where the lookup-table, register, multiplier and slack
columns come from. `sim/` holds the elaboration and simulation logs and, for
the HLS and SPMW arms, the RTL the tools generated. SPMW trials also have
`build/` with the fabric RTL (`spmw_top.sv`, `spmw_fifo.sv`, `spmw_const.sv`)
and `reports/csynth_*.rpt`, one per role.

## What is missing, and why

**The intermediate designs are gone.** Only the file each trial *ended* on
survives. Two things caused this together, and both were the harness, not the
models:

- `write_file` wrote to one path in the working directory, so every round
  overwrote the previous round's file.
- `agent.py` truncated every tool argument to 200 characters before writing
  the transcript, so the transcript holds only the first 200 characters of
  each revision.

So for a trial that built eight times, the eight intermediate designs cannot be
reconstructed. What *is* complete for every round: the model's prose, which
tool it called, the full build verdict including error text, the per-stage
timings, the routing output, and the token and wall-clock accounting. The
`rounds/` timelines carry all of that, and each says at the top that the code
is truncated.

`harness/agent.py` now archives every revision to `<trial>.rounds/rNN_bMM_<file>`
as it is written, so a rerun would preserve what these fifteen lost. Recovering
it for these fifteen would mean running the trials again, which would produce
different trials rather than the missing files from these ones.

**Not copied:** Vitis HLS project internals -- the per-role `.adb`, `.sdb`,
`.bc`, and the thousands of intermediate `.xml` and `.vhd` files. The full
trees are 1.4 GB each for `trials/` and `steady/` and stay on the machine.
Nothing that any reported number was read from was left behind.

## Provenance

Scanned for credentials before export: no OpenRouter key, bearer token or
`Authorization` header appears in any file here. The key was passed to the
agent process through the environment and never written to disk beside a trial.
