## Remote development on `brg-zhang-xcel` (U280 FPGA)

`brg-zhang-xcel` has the AMD/Xilinx Alveo U280 board and the Vitis/Vivado toolchain, so run any HLS/synthesis/on-board test there. Local edits must be synced up before building — the remote has its own checkout at `/scratch/hc676/allo`.

**Connect:**

```bash
ssh brg-zhang-xcel
```

**Environment on the remote.** `conda` is not on hc676's PATH, so activate the prebuilt env by putting its `bin` on PATH directly (equivalent to `conda activate`). `LLVM_BUILD_DIR` must be set at runtime too — the LLVM/simulator backend loads its shared runtime lib from there, or tests fail with `AssertionError: LLVM_BUILD_DIR is not set`:

```bash
export PATH=/scratch/hc676/allo-agent/bin:$PATH        # the prebuilt Allo conda env
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
source /work/shared/common/allo/vitis_2023.2_u280.sh   # Vivado/Vitis 2023.2 + U280 platform (only for HLS/FPGA)
```

Non-interactive SSH does not run `~/.bashrc`, so set these each session, e.g.
`ssh brg-zhang-xcel 'bash -lc "export PATH=/scratch/hc676/allo-agent/bin:\$PATH; export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build; cd /scratch/hc676/allo; python3 -m pytest tests/test_schedule_compute.py -v"'`.

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
export PATH=/scratch/hc676/allo-agent/bin:$PATH
export LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build
cd /scratch/hc676/allo
pip install -e .
```

**Run experiments / hardware tests** on the remote after the Vitis env is sourced (e.g. `python3 -m pytest tests/test_vitis.py -v`, `tests/test_pynq.py`, or example scripts targeting the U280).

# Code style
- Make small, targeted diffs rather than large refactors, and always be concise
- Prefer general solutions instead of one-off `if/else` patches
- Place Python frontend code in `allo/`
- Place MLIR dialects and passes code in `mlir/`
- Add tests and documentation for new features in `tests/` and `docs/`

# Don'ts
- Do not modify repository structure without approval
- Do not install system packages without explicit user confirmation
