# SPMW evaluation bundle, 2026-09-06

The experiments of `SPMW_REMAINING_EXPERIMENTS.md`, run on brg-zhang-xcel.
Every number here was measured; nothing is extrapolated, and a measurement
that did not succeed is present as a row with a status and a reason rather
than absent.

## Where things are

    manifest.json     the packages, their row counts, tools, part, commit
    results.csv/json  every package's rows together, `package` first
    coverage.md       what the plan asked for against what is here
    e1_gemm/          spmw, spmw_mem, autosa, allo
    e2_fft/           spmw, hpfft, allo
    e3_tpu/           complete GPT-2 medium and LLaMA-7B blocks on the board
    e4_feather/       the SPMW port of FEATHER against the original RTL
    e5_compile/       compilation time: shared vs per-instance kernels
    e6_attention/     grouped vs conventional attention, fixed workload
    e7_loc/           design-only source size, HLS against SPMW
    sources/          the Allo commit the runs were made from

Each package holds its own `README.md` -- what was run, the exact commands,
how each metric was measured, and the caveats that belong to it -- along with
`results.csv`, `results.json` and a `reports/` directory with the tool
reports and logs each row was read from. Read a package's README before its
numbers: the packages measure different things and are not interchangeable.

## Fixed settings

| | |
|---|---|
| Board / part | AMD Alveo U280, `xcu280-fsvh2892-2L-e` |
| Tools | Vitis HLS 2023.2, Vivado 2023.2; XRT for the board runs |
| Clock | 300 MHz, a 3.333 ns target, unless a package says otherwise |
| Arithmetic | int8 x int8 into int32 for GEMM and the mini-TPU; complex FP32 for the FFT |
| P&R | out of context (`synth_design -mode out_of_context`), no shell, unless a row says in-context |

## How to read a status

`pass` (or `ok`, in the packages whose collector uses that word) means the run
finished and its check passed: for a cosimulation, that every output token
matched the design's own reference; for an implementation, that it routed with
no unrouted nets and non-negative worst slack. `functional_fail`, `timing_fail`,
`resource_fail`, `timeout`, `unsupported` and `environment_blocked` each carry a
`failure_reason` saying what happened. `pending` and `running` mark rows whose
job had not finished when the bundle was assembled; `manifest.json` records the
assembly time.

## What is measured, in one line each

- **E1** -- an output-stationary int8 GEMM array at 4, 8, 16 and 32 on a side,
  in three systems. The three do not launch the same work: one SPMW launch is
  one tile, AutoSA's kernel folds the tiles of a fixed 16x16x16 problem inside
  the launch, and Allo's folds the tiles of its problem in time. Cycles are
  reported against the workload each row names.
- **E2** -- a radix-2 complex-FP32 FFT at 128 to 1,024 points, folded on all
  three sides: the SPMW single-path delay-feedback pipeline, HP-FFT's
  hand-tuned kernel and Allo's strided in-place transform.
- **E3** -- one complete decoder layer, GPT-2 medium and LLaMA-7B, chained
  launch by launch on one 300 MHz bitstream with every launch checked against
  the integer reference. Hybrid: the GEMMs and softmax run on the device, the
  elementwise operators and all packing on the host.
- **E4** -- FEATHER's GEMM and convolution tilings with general weights, the
  SPMW port against an xsim of FEATHER's own RTL, plus implementations of both.
- **E5** -- compilation time with the hardware held fixed: one HLS project per
  role against one per instance, serial against eight workers, three
  repetitions in randomised order, elapsed and child CPU time reported apart.
- **E6** -- grouped against conventional attention-PV on the same hardware
  budget and the same workload.
- **E7** -- design-only source size on both sides, counted language-aware.

## Provenance

`sources/commit.txt` names the Allo commit the SPMW runs were built from; the
baselines' checkouts and commits are named in their packages' READMEs
(AutoSA, HP-FFT-HLS at c4611b8, FEATHER's reference RTL). The scripts that
produced each package are copied into that package, under `scripts/` where its
run used more than the repository's own entry points.
