# SPMW — what was built, what it does, and what is not done

Branch `hc/spmw-allo-implementation-99c949`, pushed to
`github.com/chhzh123/allo`. This is the review guide: what exists, what is
proven and how, and where the honest edges are.

---

## 1. The short version

`spmw_design_doc.md` specifies a Python frontend in which you write one work unit
against a declared port contract, declare how the copies are wired, and place
components on topologies. That frontend now exists as `allo/spmw/`, and **all
five of the design's worked examples run and produce correct results**:

| Example | What it exercises | ref | simulator | HLS csim |
|---|---|---|---|---|
| §3.1 systolic GEMM | mesh, computed boundaries, loaders and drains | ✅ | ✅ | ✅ |
| §3.2 tiled GEMM | a placed fabric, `shard(dim=)`, per-site tensor views | ✅ | ✅ | ✅ |
| §3.4 FFT | key-form links, block streams, a resident twiddle ROM | ✅ | ✅ | ✅ |
| §3.5 mini-TPU | two placements, `link`, stationary weights, seeded chain | ✅ | ✅ | ✅ |
| §3.6 attention P·V | interior boundaries, split axes, G ∈ {1,2,4} | ✅ | ✅ | ✅ |

On top of that, the design's **load-bearing compilation claim is now realised**:
a spatial design compiles to a number of HLS function bodies that tracks its
*role* count rather than its grid.

**131 passed, 6 skipped** (the skips are the Vitis-gated csim tests when the
toolchain is not sourced). `pylint allo` is 10.00/10, exit 0.

---

## 2. The claim, measured

A 2-D mesh has nine site signatures at any size — interior, four edges, four
corners. The design says nine bodies should reach code generation whatever the
array's size. Measured on the systolic GEMM:

| grid | sites | signatures | **dataflow path**<br>HLS functions | **rolled path**<br>HLS functions |
|---|---|---|---|---|
| 3×3 | 9 | 9 | 16 | **10** |
| 4×4 | 16 | 9 | 25 | **10** |
| 8×8 | 64 | 9 | 81 | **10** |
| 16×16 | 256 | 9 | (289) | **10** |

Both numbers are pinned by tests in `tests/dataflow/spmw/test_spmw_rolled.py`,
side by side, so neither can regress quietly.

The frontend was always flat — signatures and emitted source arms sit at nine
from 3×3 on. What grew was what the *dataflow builder* did with it: it expands
`mapping=[R, C]` into one kernel instance per grid point. The rolled path keeps
the structure as a single `spmw.map` op that survives to code generation.

---

## 3. What to look at, in reading order

**Start here** — the surface, and whether it reads the way the design says:

- `docs/source/dive/spmw.rst` — the guide: the five nouns, both link forms, the
  binding index algebra, the check table.
- `tests/dataflow/spmw/test_spmw_gemm.py` — the flagship, end to end.

**The frontend** (`allo/spmw/`, pure Python, no C++):

| Module | What it holds |
|---|---|
| `ports.py`, `interface.py` | port symbols with protocol/direction/type; the `Interface` metaclass |
| `component.py` | `@spmw.unit`, `@spmw.fabric`, roles, trace-time checks |
| `topology.py` | both link forms, channels, site signatures |
| `placement.py` | `place()`, boundary bundles, axis symbols |
| `index.py` | the `index=` algebra: axis expressions, `...`, slice maps |
| `bricks.py`, `bindings.py` | memory bricks and the mover verbs |
| `graph.py`, `context.py` | fabric elaboration and the design's check table |
| `channels.py` | grouping channels into declarable families |
| `lower_df.py` | lowering to an `allo.dataflow` program |
| `lower_mlir.py` | lowering to the rolled `spmw.map` form |
| `refsim.py` | a reference simulator: one task per site |
| `driver.py` | `spmw.build(target=…)` |

**The dialect** (`mlir/`):

- `mlir/include/allo/Dialect/SPMW/SPMWOps.td`, `SPMWAttrs.td` — `spmw.map` and
  its four attributes.
- `mlir/lib/Dialect/SPMW/SPMWOps.cpp` — the verifier.
- `mlir/lib/Translation/EmitVivadoHLS.cpp`, `emitSPMWMap` — the instantiation.

---

## 4. Design decisions worth checking

**Why the dialect's attributes are not the old branch's.** `feat/spmw` has a
hardware-validated MLIR layer, and the plan was to port it. Three things in it
could not express this frontend:

- *Routing is per site here.* One port can address two channel families, or
  reach its neighbour by a coordinate link at one site and by a key at another —
  which the design explicitly calls one mechanism. The old `peer_link` keyed
  routing per *port*, so only one family survived. `port_map` carries either a
  constant displacement or a per-site slot table.
- *A token may be a block.* `family` separates the array's shape from the shape
  of one token, because an FFT channel carries a complex sample, not a scalar.
- *The class table is materialised.* The old passes computed the coordinate→role
  map and threw it away, keeping only counts. The emitter needs it per site.

**Why role bodies are sibling functions rather than a region.** The HLS emitter
already walks every top-level `func.func`, so O(#roles) bodies come out as C++
for free and only the instantiation is new work.

**Why sites are listed per role rather than branched on.** A role's site set is
not rectangular in general — an interior boundary can open anywhere — and a
constant site table keeps every subscript compile-time constant under unroll.

**Why an unbound slot is −1.** The old lookup defaulted a missing entry to zero,
which corrupts channel zero instead of failing.

---

## 5. What is NOT done

Please read this section before concluding anything from §2.

**The rolled path does not compute yet.** `emitSPMWMap` emits the channels and
the instantiation loops, and the function count is flat — but the role *bodies*
are not transcribed into MLIR. `lower_mlir.render_module` emits them empty, so
the flat-HLS measurement is of a program with the right structure and no
arithmetic. Transcribing bodies is the next step, and the cheapest route is
probably Allo's existing `unroll=False` path, which already deduplicates
instances by control-flow class (measured: 12 functions flat from 3×3 to 8×8).

Consequently **the numerics still come through the dataflow path**, which is
what every correctness test above exercises. The two paths do not yet meet.

**The two analysis passes are not written.** `spmw-role-partition` and
`spmw-resolve-channels` were in the plan, but since this frontend computes what
they would produce, they became *checks* rather than a source of truth, and are
no longer on the critical path. Worth adding as verification.

**Not realised:** `fold` / `unroll` at `place`, and `pack=` / `unpack=`. All four
are refused with a clear message rather than silently ignored.

**A placed fabric expands per site**, so a tiled design emits one kernel per tile
rather than instantiating one engine — correct, but not the hierarchical IP reuse
the model is built for.

---

## 6. How to check it yourself

Everything below assumes `brg-zhang-xcel`. There is no way to run Allo locally
on this machine.

```bash
bash scripts/spmw_remote_check.sh            # sync + the whole SPMW suite
HLS=1 bash scripts/spmw_remote_check.sh      # also the Vitis csim tests
```

If you change any C++, rebuild first — and note the file-descriptor limit, which
otherwise fails the link with a misleading `cannot find -lm`:

```bash
ulimit -n 65536 && cd mlir/build && ninja
```

To see what a fabric lowers to:

```python
import allo.spmw as spmw
print(spmw.source(my_fabric))                       # the dataflow program
from allo.spmw.lower_mlir import render_module
print(render_module(spmw.elaborate(my_fabric)))     # the rolled form
```

---

## 7. On how much the verification is worth

Two review passes and the first real hardware run found 33 issues between them.
Nearly every serious one had the same shape: **a check that existed, looked
right, and could not fire.** A coverage test that was arithmetically a tautology;
a phase rule that collected only writers, so the mixed case it existed for was
unreachable; `check_directions`, fully written and unit-tested, never called by
the library.

Three produced silently wrong numbers rather than errors — the top-level
argument permutation, per-site channel routing, and an elision that discarded a
`get` along with the `put` around it.

The local harnesses (a reference simulator and an interpreter over the generated
program) agreed with each other because they shared my assumptions; the one that
ignored type annotations could not see a kernel parameter typed with a view's
shape while indexed in base coordinates. **The remote run is what settles
things**, and it found five bugs in its first pass, including one where
`spmw.build` was shadowed by its own submodule so the first call worked and every
one after it did not.

`SPMW_PLAN.md` carries the fuller history, including both routes considered for
the rolled path and why the step order changed once the dialect existed.
