# What the compiler emitted for the 256-point lane design, W=2 (Fig. 9(c), arXiv version)

`../source/lanes.py` is the design at every width; `lanes_W2/` is what the
frontend emitted from it at W=2 (`--design fftlanes --size 256 --lanes 2`):

| Directory | Holds |
|---|---|
| `lanes_W2/roles/` | the HLS C++ of each unit: eight butterfly stages (`bfly*`), eight permutations (`perm*`), the output `reorder` |
| `lanes_W2/fabric/` | `spmw_top.sv`, the depth-8 FIFOs that wire the units, the harness and constants |
| `lanes_W2/reports/` | each unit's `csynth.rpt`; II=1 is read where the loop says `yes(flp)` |

Vitis project internals are not here; no reported number comes from them. The
W=2 build's cosimulation and routed reports, and every file of W=4, 8 and 16,
are not in this folder (`../../../README.md`).
