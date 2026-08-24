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
| §3.6 attention P·V | interior boundaries, split axes, G ∈ {1,2,4} | ✅ | ✅ | ⚠️ G=1 |

On top of that, the design's **load-bearing compilation claim is now realised**,
and in the form that matters for hardware: **HLS synthesises the unit, RTL builds
the array.** Nine C syntheses and nine IP exports for a 2-D mesh, whatever its
size, and Vivado assembles the array from the exported IPs.

**203 passed, 1 deselected** on `brg-zhang-xcel` with Vitis sourced.
`pylint allo --rcfile=./scripts/lint/pylintrc` is 10.00/10, exit 0.

The deselected test is `test_spmw_attention.py::test_hls_csim_matches[2]`, which
**hangs**. It is not a regression from this work — a run from before it stopped
at the same test — and it is a csim problem rather than a design one: the same
configuration passes both `ref` and `simulator`. The process sits on 75 threads
in `futex_wait_queue` with no subprocess, so Allo's concurrent csim deadlocks
rather than the generated C++ misbehaving. Worth a separate look.

---

## 2. The synthesis flow, measured

This is the part to check first, because it is the architecture and it is now
end to end. Measured on `brg-zhang-xcel`, Vitis/Vivado 2023.2, `xcu280`:

| sites | 9 | 16 | 36 | 64 | 144 | 256 |
|---|---|---|---|---|---|---|
| **whole array**: Allo lowering | 1s | 1s | 2s | 3s | 7s | **578s** |
| **whole array**: `csynth` | 39s | 43s | 60s | 93s | 280s | **807s** |
| whole-array DSP | 45 | 80 | 180 | 320 | 720 | — |
| **nine roles**: `csynth` + `export_design`, concurrent | **40.5s** | 40.5s | 40.5s | 40.5s | 40.5s | 40.5s |
| the same nine, run serially | 332s | 332s | 332s | 332s | 332s | 332s |
| Vivado assembles from the role IPs | 69s | 17s elab | — | — | — | — |

**Per-role cost is flat and parallel.** The unit does not know how large the
array is, so nine roles cost the same whatever the grid — and because roles are
independent *by construction*, which is what having roles means, they synthesise
concurrently. Nine at once take 40.5s of wall clock for 361s of CPU: each role
took 40s alongside eight others against 37s alone, so contention is negligible
at 48 cores.

**That is the second reason for the split, and the stronger one.** A whole-array
`csynth` is one monolithic run — there is nothing to spread across cores. The
role decomposition is what *creates* the parallelism. So the per-role path
matches the array at the smallest grid measured (40.5s against 39s at 9 sites)
and wins everywhere above it, rather than crossing over at ~150.

At 256 sites the array needs 807s of `csynth` and 578s of Allo lowering before
that; the roles need 40.5s and about a second.

### Why nine small roles ever cost more than the whole array

They shouldn't, and the first version of this table made that look intrinsic. It
is not. `csynth_design` on a *single* PE originally took 31.6s, against 39s for
the whole nine-PE array — so one role cost almost as much as the array. The
per-phase log says why:

```
Source Code Analysis and Preprocessing:  23.24s
Compiling Optimization and Transform:     7.45s
Scheduling / Binding / RTL generation:    ~0.5s   (all phases combined)
```

Synthesising a PE is half a second of hardware work behind **23s of C++ header
parsing**. Allo emits a fixed nine-header preamble — `ap_fixed.h`, `hls_math.h`,
`hls_vector.h` and so on — and a streaming float PE uses exactly one of them.
`role_ip.trim_includes` keeps only the headers a unit's body mentions, which took
one role's `csynth` from **31.6s to 8.8s** (source analysis 23.2s → 0.6s) and the
nine-role total from **545s to 332s**, with the cosim still passing.

Process startup is not the culprit: a `vitis_hls` run that opens a project and
exits takes 4s. Nor is batching — nine `csynth_design` calls in one invocation
still cost 295s, because the header parsing is per *call*, not per process.

What is left is `export_design`, now ~26s of each role's 37s. That is IP
packaging, and it is the honest floor of this flow: nine IPs, nine packagings.

**The array path pays the same header tax** — roughly 23s of its 39s at 3×3 — so
trimming Allo's preamble generally would speed up both. That is a change to the
shared HLS backend rather than to SPMW, so it is flagged rather than made.

An earlier version of this document claimed the whole-array program **could not**
be synthesised at any size. **That was wrong**, and the correction matters enough
to state plainly. It failed with

```
ERROR: [HLS 200-979] Argument 'v185' failed dataflow checking:
                     it can only be written in one process function.
```

because the lowering did not partition the tensor arguments. Every site writes
one element of `C`, and HLS dataflow allows a single writer per interface array
*unless its elements are independently addressable*. `feat/spmw` had already
solved this — its rolled top emits `#pragma HLS array_partition variable=C
complete dim=0`, with the comment that the streaming top "needs every element
independently addressable, so no interface is read by more than one dataflow
process". `spmw.build` now partitions every tensor through `Schedule.partition`
before building, and the same program that failed csynths in 43s. Pass
`partition=False` to see the original error.

So the split is a scaling argument, not a feasibility one: flat synthesis cost
and no scalarised interface arrays, against a path that is simpler and faster
until the array gets large.

Reproduce the whole flow with one command:

```bash
python3 scripts/spmw_build_array.py --size 4 --out /scratch/$USER/spmw_array
```

---

## 2a. The rolled HLS path, measured

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
| `ports.py`, `iface.py` | port symbols with protocol/direction/type; the `Interface` metaclass |
| `component.py` | `@spmw.unit`, `@spmw.fabric`, roles, trace-time checks |
| `topology.py` | both link forms, channels, site signatures |
| `placement.py` | `place()`, boundary bundles, axis symbols |
| `index.py` | the `index=` algebra: axis expressions, `...`, slice maps |
| `bricks.py`, `bindings.py` | memory bricks and the mover verbs |
| `graph.py`, `context.py` | fabric elaboration and the design's check table |
| `channels.py` | grouping channels into declarable families |
| `lower_df.py` | lowering to an `allo.dataflow` program |
| `lower_mlir.py` | lowering to the rolled `spmw.map` form |
| `rtl.py` | the structural fabric: FIFOs, role instances, the netlist checks |
| `role_ip.py` | one role as a standalone synthesisable unit, and its wrapper |
| `refsim.py` | a reference simulator: one task per site |
| `driver.py` | `spmw.build(target=…)` |

**The dialect** (`mlir/`):

- `mlir/include/allo/Dialect/SPMW/SPMWOps.td`, `SPMWAttrs.td` — `spmw.map` and
  its four attributes.
- `mlir/lib/Dialect/SPMW/SPMWOps.cpp` — the verifier.
- `mlir/lib/Translation/EmitVivadoHLS.cpp`, `emitSPMWMap` — the instantiation.

---

## 3a. What `feat/spmw` already had, and what it did not

Worth being precise, because the old branch is where this architecture comes
from. `allo/spmw_rtl.py` on `feat/spmw` has exactly this split and it is
hardware-validated: HLS synthesises `pe_interior`, `export_design` packages it,
and Vivado instantiates it `M*N` times in a `generate` nest, never
re-synthesising per grid point. The ap_fifo ABI, the FIFO primitive and the
export/package TCL all carried over here with little change.

What did not carry over is the driver. Its wiring came from a four-entry
literal —

```python
_RTL_WIRING = {
    "west":  _Wire("fa", "i", "j",     "read"),
    "east":  _Wire("fa", "i", "j + 1", "write"),
    "north": _Wire("fb", "i", "j",     "read"),
    "south": _Wire("fb", "i + 1", "j", "write"),
}
```

— and four `NotImplementedError` gates refused anything else: the families had
to be exactly `east/west` + `north/south`, the partition a single compute role
covering the whole grid, the operands a 2-D `A[M,K] @ B[K,N]`, and the port set
the systolic four. It read the grid extents off the IR by regular expression and
then discarded the port list (`del ports  # the systolic wiring is fixed`).
`set_top pe_interior` is likewise hardcoded, so a nine-role mesh could not be
exported.

So the old branch answers "can this architecture work?" — yes, on hardware — and
this branch answers "for which designs?" Here the netlist is read off the
elaborated graph, so the FFT (keyed links, block tokens), the TPU (two placements
joined by `link`) and attention (an interior boundary) all emit a fabric that
Vivado elaborates. Building their *units* is a further step, and one of the three
does not get there — see §5.

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

**Why a `link` is internal but a loader is a port.** Both are "bindings", but a
`link` has both ends inside the fabric and a loader has one end at the DMA. The
test is whether the family's slots cover both directions — the property itself,
not a proxy. `_plan_link` registers one family under *both* placements, so
treating it as a boundary declared the same top-level port twice; `xvlog` caught
that, and the underlying error would have left the TPU's two placements
unconnected.

**Why the unit's parameter mapping is derived and then cross-checked.** Allo
renames every value, so `west` reaches HLS as `v0`. The mapping back is read out
of the generated program's structure (the region's declaration order, and the
call in `top`), which is positional — so it is checked against a second,
independent signal: whether the body reads or writes each parameter must agree
with the port's declared direction. A slipped mapping would wire a reader onto a
writer's FIFO, which is wrong numbers rather than an error.

---

## 5. What is NOT done

Please read this section before concluding anything from §2.

**The rolled MLIR path does not compute.** `emitSPMWMap` emits the channels and
the instantiation loops, and the function count is flat — but the role *bodies*
are not transcribed into MLIR, so `lower_mlir.render_module` emits them empty.
That measurement is of a program with the right structure and no arithmetic.

This matters less than it did. The rolled path was going to be how the design
reached hardware; the RTL path (§2) now is, and it carries real bodies. The
rolled form remains useful as the IR the dialect verifies and as the basis for
whole-array C simulation, and `spmw.map` is what the structural emitter reads.

**The RTL path is now verified numerically, not just structurally.** A 4×4
systolic GEMM built from the nine exported role IPs, glued by the structural
fabric, runs in xsim and produces all sixteen values of C bit-exact against
`spmw.build(target="ref")`:

```
SPMW COSIM PASS (16/16 tokens, 0 errors)
```

The stimulus order is not hand-written — `rtl.boundary_plan` reads it off the
same `binding.imap` the reference simulator evaluates, so a disagreement between
the testbench and the array is a real one rather than a harness bug. Inputs are
small integers, exact in float32, so the comparison is bit-exact rather than a
tolerance. Reproduce with `--cosim`.

Still true: the structural checks (every link on one channel, checked against
`topology.channels`; nothing dangling) are what catch mis-wiring at sizes too
large to simulate.

**One of the five designs does not build as units.** Every role of the GEMM
(9/9), the TPU (10/10), attention (7/7) and the tiled GEMM (16/16) compiles to a
unit and its parameters map back to the fabric's ports. The **FFT does not**: its
butterfly reads its own stage and index, so its sites differ by *position*, not
only by wiring, and one IP cannot stand for them. It is refused with a clear
message rather than compiled — a single-instance kernel's pid is always zero, so
compiling it would run every site as if it were the origin.

Closing it is a known, scoped piece of work. The dialect's calling convention
already says a unit takes its grid coordinates as arguments, and the mechanism is
in hand: Allo refuses a scalar kernel argument but accepts a one-element array,
and coordinates could equally arrive on their own streams, read once and held —
exactly how stationary weights are handled now. That last route needs no new
interface kind and would make the fabric drive each site from a small
constant-source module.

The TPU and attention only build *because* of that stationary handling. In the
array a site reads the parent's weight tensor at its own coordinates,
`local_W[i, j]`, which looked like coordinate dependence but is per-site *data*.
A unit takes it on its own port and holds it, which is what `stationary` already
means.

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
HLS=1 bash scripts/spmw_remote_check.sh      # also the Vitis and Vivado tests
```

The synthesis flow of §2, end to end — nine HLS syntheses and exports, then
Vivado assembling the array from them:

```bash
python3 scripts/spmw_build_array.py --size 4 --out /scratch/$USER/spmw_array --cosim
```

`--cosim` adds the numeric check: it generates the FP cores' simulation models,
runs xsim, and compares every result token against `target="ref"`. Drop it to
stop after Vivado elaboration. `--stage-only` writes the projects and runs
nothing, which is the fast way to read the generated unit and the structural
top.

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
