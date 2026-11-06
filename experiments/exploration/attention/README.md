# Attention P V: Table 2 of the arXiv version

This folder holds one attention head's complete `Y[S, 8] = P[S, S] V[S, 8]`,
with int8 operands and int32 results, computed in one launch on a 16x16 array
of 256 MAC PEs. Two mappings are compared, at S = 64 to 1,024. Every run is an
RTL cosimulation of the assembled array through
[`scripts/spmw_build_array.py`](../../../scripts/spmw_build_array.py) with
`--cosim`, and every output token is checked against numpy.

## The two mappings

| | Original (`conv`) | Grouped, G = 2 (`grouped`) |
|---|---|---|
| Topology | the plain 16x16 mesh | two 8-column slabs; slab 0's partial sums run on into slab 1's top |
| Reduction chains | 16, each 16 cells deep | 8, each 32 cells deep |
| Keys a tile | 16 | 32 |
| Tiles | S/16 | S/32 |
| Output lanes | 16: V's 8 columns, plus 8 lanes of zeros that are also checked | 8 |
| Units | 256 `mac`, 16 `seed_sums`, 16 `fold` | 256 `mac`, 8 `seed_sums`, 8 `fold` |

In the original mapping a P row passes across all 16 columns, so every column
sees the same 16 keys. V fills 8 columns and the other 8 hold zeros. Grouping
uses all 256 PEs and halves the tile count.

Both mappings use the same `mac` cell and the same accumulation:

- Each cell holds its V entry for every tile, in a `MemIn` of S/keys entries
  that is loaded before the first query row.
- The S query rows stream through once per tile.
- `fold` sends each row's partial sum into a FIFO S deep. From there
  `seed_sums` feeds it back into the top of the chain on the next tile. The
  last tile's sums are Y.
- Both use HLS flushable pipelines (`spmw_pipeline_style = "flp"`).
- P and V are uniform int8 in [-4, 4), seed 0, one seed a point.

## Results

Cycles run from reset release to the final output token (the testbench in
[`allo/spmw/cosim.py`](../../../allo/spmw/cosim.py)). They include loading V,
pipeline fill and drain, and feedback stalls. All ten runs pass.

| S | Original | Grouped | Speedup | First output, original / grouped |
|---:|---:|---:|---:|---:|
| 64 | 481 | 429 | 1.12x | 343 / 331 |
| 128 | 1,222 | 825 | 1.48x | 1,005 / 663 |
| 256 | 4,302 | 2,294 | 1.88x | 3,957 / 1,997 |
| 512 | 16,606 | 8,446 | 1.97x | 16,005 / 7,893 |
| 1,024 | 65,790 | 33,038 | 1.99x | 64,677 / 31,973 |

Table 2 reports S = 128 to 1,024. The S = 64 pair passes and is in
`results.csv`, but it is not in the table. With HLS's default pipeline style
(`logs/stp_*_64.log`), grouped S = 64 deadlocks: the cosimulation times out
with 0 of 512 outputs. Original S = 64 passes in 468 cycles. The default style
freezes a cell's pipeline while an input is missing, so a short grouped tile
never reaches `fold` to feed the next tile.

Once a tile is longer than a partial sum's trip round the loop, both mappings
stream a row a cycle and take `T(S + 1) + c` cycles for T tiles. For the
original mapping c = 190, exact from S = 128; for the grouped mapping c = 238,
exact from S = 256. Grouped streams S^2/32 rows against S^2/16, so the speedup
tends to 2. Below that point the feedback loop binds: grouped S = 128 is 71
cycles over its form, and both mappings are over it at S = 64.

## What is not measured

- **No place and route**, so these arrays have no clock frequency or area.
- **One seed a point.**
- **No generated code was kept** by these runs. `report/` holds each run's
  `cost.json` (roles, instances, FIFOs, HLS, Vivado and cosimulation times),
  and `logs/` holds its build log.

## Files

| Path | What it is |
|---|---|
| `results.csv` | ten rows, one a mapping and S, made from `logs/` and `report/` by `scripts/make_results.py` |
| `source/test_spmw_attention_passes.py` | [`tests/dataflow/spmw/test_spmw_attention_passes.py`](../../../tests/dataflow/spmw/test_spmw_attention_passes.py) at f6ca5bf5, identical at HEAD. `pv_head` is the design. It imports `grouped_mxu` from `test_spmw_attention.py`, so it runs from `tests/dataflow/spmw/`. Its other designs are a 4x4 study that Table 2 does not use |
| `logs/<kind>_<S>.log` | the build and cosimulation log of each measured run |
| `logs/stp_<kind>_64.log` | the two S = 64 runs with the default pipeline style |
| `logs/driver.log` | the driver's one line a run: return code, wall time, cycles |
| `report/<kind>_<S>_cost.json` | each run's cost record |
| `scripts/attnhead.sh` | the driver |
| `scripts/make_results.py` | builds `results.csv` |

## Reproducing

On brg-zhang-xcel with Vitis HLS and Vivado 2023.2. `ALLO_TREE` is the checkout
to build from (default `/scratch/hc676/simtime/allo`) and `HEAD16_OUT` is where
the runs go (default `/scratch/hc676/e21_attnhead`):

    experiments/exploration/attention/scripts/attnhead.sh 64 128 256 512 1024

Each run is

    python3 scripts/spmw_build_array.py --design attnhead16d8-<grouped|conv>-<S> \
        --size 16 --frequency 300 --jobs 8 --cosim --out <dir>

To rebuild the table, copy each run's log into `logs/<kind>_<S>.log` and its
`cost.json` into `report/<kind>_<S>_cost.json`. Then run:

    python3 experiments/exploration/attention/scripts/make_results.py experiments/exploration/attention

For the default-style runs, delete `head.spmw_pipeline_style = "flp"` in
`pv_head`.
