# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Allo is

Allo is a Python-embedded, MLIR-based language and compiler for building composable, high-performance hardware accelerators. A Python program is parsed into a custom MLIR dialect (`allo`), transformed by schedule primitives, and then either JIT-compiled to CPU (LLVM) for simulation or emitted as HLS/RTL for FPGAs (AMD/Intel) and AI Engines (AMD Ryzen NPU).

## Build

The build compiles the C++ MLIR backend, so it requires a prebuilt LLVM and several env vars:

```bash
conda activate allo                 # or the environment you set up (see Remote development)
export LLVM_BUILD_DIR=/path/to/llvm-project/build
export PATH=$LLVM_BUILD_DIR/bin:$PATH
python3 -m pip install -v -e .      # builds mlir/ via CMake+Ninja+nanobind, installs allo/ editable
```

- `LLVM_BUILD_DIR` **must** be set or `setup.py` raises. It points at the build of the pinned `externals/llvm-project` submodule.
- Default generator is Ninja. Override with `BUILD_WITH=make`. Parallelism via `NUM_THREADS=N`.
- **Re-run `pip install -v -e .` whenever any C++ file under `mlir/` changes** — the compiled extension (`allo/_mlir`) must be rebuilt. Pure-Python edits under `allo/` need no rebuild (editable install).

## Test & lint

```bash
# Fast path — software simulators only, skip the slow/hardware dataflow suite
python3 -m pytest --ignore=tests/dataflow tests -v

# A single file (preferred — the full suite is slow)
python3 -m pytest tests/test_schedule_compute.py -v

# A single test
python3 -m pytest tests/test_types.py::test_int -v

# Dataflow suite (needs OpenMP threads; skip the AIE hardware tests)
OMP_NUM_THREADS=128 python3 -m pytest --ignore=tests/dataflow/aie tests/dataflow -v

# Lint (license headers, black, clang-format, pylint)
bash scripts/lint/task_lint.sh
```

When writing/running tests, use only software backends: `target="llvm"` (JIT to CPU) or `target="simulator"`. Vitis HLS, Vivado, PYNQ/FPGA, and AIE tests require the hardware toolchain — run those on the remote machine (below) or ask the user, don't attempt them locally.

## Architecture

The flow is **Python AST → `allo` MLIR dialect → schedule transforms → backend codegen**.

- **Frontend / IR building** (`allo/ir/`): `builder.py` walks the Python AST and emits the `allo` MLIR dialect; `infer.py` does type inference; `types.py`/`typing_rule.py` define Allo's fixed-point/int/float type system; `transform.py` applies schedule primitives. `allo/customize.py` (`allo.customize`) is the main entry: it returns a schedule object you mutate with primitives (split, pipeline, partition, compose, stream, etc.).

- **Dataflow model** (`allo/dataflow.py`): the spatial/streaming programming model — `@df.region`, `@df.kernel(mapping=[...])`, `df.get_pid()`, and `Stream[T, depth]` with `.put()/.get()`. `df.build(...)` composes kernels into a top module. This is the layer the current `feat/spmw` branch extends (see `SPMW_IMPLEMENTATION_PLAN.md`); SPMW makes interconnect topology and PE boundary roles first-class instead of hand-written `meta_if` PID chains.

- **Backends** (`allo/backend/`): `llvm.py` (`LLVMModule`, JIT CPU simulation), `hls.py` (`HLSModule`, Vivado/Vitis HLS codegen), `vitis.py`, `pynq.py` (on-board FPGA runtime), `aie/` (AI Engine), plus `catapult.py`, `tapa.py`, `xls.py`, `ip.py` (external IP integration). The backend is selected by the `target=` argument.

- **Frontend importers** (`allo/frontend/pytorch.py`): lowers PyTorch models into Allo programs.

- **C++ MLIR backend** (`mlir/`): the `allo` dialect definition and passes.
  - `mlir/include/allo/Dialect/*.td` — TableGen ops/types/attrs for the dialect.
  - `mlir/lib/Dialect/`, `mlir/lib/Conversion/` (e.g. `FixedPointToInteger`, `LowerBitOps`, `AlloToLLVM`), `mlir/lib/Transforms/` — the passes.
  - `mlir/lib/CAPI/Translation/Emit*HLS.cpp` — the HLS/RTL emitters (Vivado, Intel, Catapult, TAPA, XLS).
  - `mlir/lib/Bindings/` — nanobind Python bindings surfaced as `allo._mlir`.

Key top-level modules: `allo/primitives/` (relay, unify), `allo/autoscheduler/`, `allo/verify.py` (formal equivalence checker), `allo/memory.py` (`Memory`, `Layout`), `allo/library/` (reusable kernels).

## Docs to read for deeper work

- `docs/source/dive/frontend_syntax.rst` — full Allo frontend syntax reference.
- `docs/source/dive/dataflow.rst` — dataflow model (regions, kernels, streams).
- `SPMW_IMPLEMENTATION_PLAN.md` — design + plan for the current `feat/spmw` branch.

## Conventions

- Python frontend code goes in `allo/`; MLIR dialects/passes in `mlir/`; tests in `tests/`; docs in `docs/`.
- Make small, targeted diffs; prefer general solutions over one-off `if/else` patches.
- Every source file carries the Apache-2.0 license header (lint enforces it). Python is formatted with `black==24.8.0`, C++ with `clang-format`.
- Do not change repository structure or install system packages without user confirmation.

## Remote development on `brg-zhang-xcel` (U280 FPGA)

`brg-zhang-xcel` has the AMD/Xilinx Alveo U280 board and the Vitis/Vivado toolchain, so run any HLS/synthesis/on-board test there. Local edits must be synced up before building — the remote has its own checkout at `/scratch/hc676/allo`.

**Connect:**

```bash
ssh brg-zhang-xcel
```

**Environment on the remote:**

```bash
conda activate /scratch/hc676/allo-agent          # the prebuilt Allo conda env
source /work/shared/common/allo/vitis_2023.2_u280.sh   # Vivado/Vitis 2023.2 + U280 platform
```

**Sync local changes → remote** (run locally, from the repo root). Push before every remote build so the remote reflects your working tree:

```bash
rsync -avz --delete \
  --exclude '.git' --exclude 'build' --exclude 'externals' \
  --exclude '__pycache__' --exclude '*.egg-info' \
  ./ brg-zhang-xcel:/scratch/hc676/allo/
```

(`--delete` mirrors deletions; drop it to keep remote-only files. Excluding `externals` avoids copying the LLVM submodule — the remote already has its own.)

**Rebuild on the remote after syncing** — only needed when C++ files under `mlir/` changed; pure-Python edits are picked up automatically by the editable install:

```bash
ssh brg-zhang-xcel
conda activate /scratch/hc676/allo-agent
cd /scratch/hc676/allo
pip install -e .
```

**Run experiments / hardware tests** on the remote after the Vitis env is sourced (e.g. `python3 -m pytest tests/test_vitis.py -v`, `tests/test_pynq.py`, or example scripts targeting the U280).
