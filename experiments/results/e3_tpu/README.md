# E3: the mini-TPU — see `experiments/e3_tpu/README.md`

This directory is a **results bundle**: `results.csv` and `results.json` here
are copies of the experiment's own board rows, and are the point of the bundle.

The README that used to sit here was an assembled snapshot that had gone stale
against its own data -- it carried neither the Gemmini baseline nor the
microbenchmark that compares the two, both of which are now the larger part of
the experiment, and its board timings had moved on. Rather than keep a second,
diverging account, it is replaced by this pointer.

- **The experiment's README** is `experiments/e3_tpu/README.md`.

E3 has three tables rather than one, because it has three things to report,
and they are not the same shape:

| Table | What it holds |
|---|---|
| `results.csv` here, and `experiments/e3_tpu/results.csv` | the two board rows: GPT-2 and Llama, one block each, on the U280 |
| `experiments/e3_tpu/gemmini/results.csv` | Gemmini's MXU-only and MXU+VPU, 4x4 to 32x32, two scale forms: area, timing and cycles |
| `experiments/e3_tpu/micro/results.csv` | the workload both systems run, on three designs -- SPMW's stage engine, a fixed-function SPMW datapath, and Gemmini |

They are kept apart rather than concatenated: the board rows are wall-clock
measurements of a hybrid host/device block, and the other two are cycle and
area measurements of datapaths routed out of context. One table with both in it
would invite a comparison that does not exist.
