# E1: int8 GEMM arrays, and exactly what each system was asked to compute

The three systems in this experiment **do not compute the same workload**, and
the difference is in the sources here rather than hidden in the numbers. Read
this before comparing any two cycle counts.

## What each launch computes

| System | Source | One launch computes | K |
|---|---|---|---|
| SPMW, stream-fed mesh | `spmw/spmw_mesh_gemm_int8.py` | one SxS tile | S |
| SPMW, memory-fed kernel | `spmw/spmw_memory_fed_gemm_int8.py` | one SxS tile | S |
| Allo, tile workload | `allo/S<S>_tile/kernel.cpp` | one SxS tile | S |
| Allo, 128 workload | `allo/S<S>_128/kernel.cpp` | a 128x128x128 product, tiles folded in time | 128 |
| AutoSA | `autosa/input/mm<S>i8*/kernel.c` | **a whole 16x16x16 product** (32x32x16 at S=32) | 16 |

AutoSA's input program pins the problem in `kernel.h`:

    #define I 16
    #define J 16
    #define K 16

and the array size is chosen separately by `--sa-sizes`, so the SxS array
iterates the tiles of that fixed problem inside one launch. At S=32 the problem
had to grow to 32x32x16, because AutoSA silently clamps `array_part[32,32,16]`
to the loop bounds of a 16-cubed problem and emits a byte-identical 16x16
design; the clamped attempts are kept in `results_all_variants.csv` upstream
with the diff as evidence.

## Which comparisons are sound

- **SPMW against Allo, tile rows**: sound. Both compute one SxS tile with K=S.
- **SPMW against Allo, 128 rows**: not a cycle comparison. Allo folds 128-cubed
  in time; SPMW would need a host loop over tiles.
- **Anything against AutoSA cycles**: not sound as a cycle comparison. Its
  launch does more work at small S and different work at S=32.
- **Routed resources across all three**: sound at a given S, since every design
  places one multiplier per element, and the table separates the two SPMW
  scopes (mesh alone, and mesh with its memory loaders and drain) because they
  are different amounts of hardware.

The paper's table carries the workload per row for this reason. A reader who
compares the cycle column across systems without reading the workload column
will draw a wrong conclusion, and that is a fault of the reading rather than of
the measurement, but it is worth removing.

## The matched runs

To remove it, `matched/` holds AutoSA inputs with `I = J = K = S`, so its
launch computes exactly one SxS tile with K=S, the same as SPMW and Allo. Those
rows are the apples-to-apples cycle comparison; the fixed-problem rows above
are kept because they are what AutoSA's own tutorial configuration does, and
because they show the tiling behaviour that makes its kernel larger.

## Sources here

- `spmw/` the two SPMW designs, copied from the test suite unchanged. The
  registry names are `gemm8` and `autosa`; the second is structurally matched
  to what AutoSA emits, with chained A and B distribution and a chained C
  drain.
- `autosa/input/<name>/kernel.c` and `kernel.h`, the program AutoSA was given,
  and `out/src/` the code it generated. `autosa_commands.json` records the exact
  command per design, including the space-time transform and the array
  partition.
- `allo/allo_gen.py`, the generator that instantiates Allo's library systolic
  array at a size and workload, and `S<S>_<workload>/kernel.cpp`, the HLS it
  produced.

Names ending `z` use the corrected kernel form, with the accumulator reset
inside the scope. The uncorrected form is functionally wrong: without
`C[i][j] = 0` in the scop the accumulator is never cleared, and the 4x4 design
fails 240 of its 256 outputs. Only the corrected rows are reported.
