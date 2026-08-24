# SPMW Implementation Plan

## Context

`spmw_design_doc.md` specifies SPMW (Single-Program, Multiple-Work-unit): a Python frontend
`allo.spmw` in which you write one work-unit program against a declared **port contract**
(`Interface`), declare how copies are replicated and wired (`Topology`), and assemble hierarchy by
**placing** components on topologies. It evolves `allo.dataflow`, whose systolic examples spend
most of their lines re-deriving interconnect by hand (`fifo_A[i, j+1].put(...)` global index math)
and hand-writing boundary variants as `meta_if` PID chains, with nothing checking that the two
agree.

The end goal for this effort: **the systolic GEMM (§3.1), mini-TPU (§3.5), and FFT (§3.4) examples
all run and produce correct results.**

### What already exists

- **`main`** carries PR #555 (`76130c6`): four MLIR ops in the *existing* `allo` dialect —
  `allo.stream_global`, `allo.get_stream_global`, `allo.put_stream_global` (a named array of
  streams indexed by an `AffineMapAttr` over dynamic pids) and `allo.grid_map` (a rolled,
  region-carrying grid op with `$sharding`/`$grid`). Nothing produces or consumes them yet; no
  Python `spmw` module exists. It also refactored `parse_ast` to take a callable, and rewrote
  `_get_global_vars` to walk the whole call stack — both of which a new entry point needs.
- **`feat/spmw`** (unmerged, based on `36bc03e`, one commit behind `main`) is a full prior
  implementation of an *earlier* frontend design: +24k lines, a real `spmw` dialect with
  `spmw.map`, the `spmw-role-partition` / `spmw-resolve-channels` / `spmw-unroll` passes, a rolled
  HLS emitter, a structural SystemVerilog emitter, and ~350 tests validated to hardware.
  Its **frontend surface is string-typed and has no port contract**, so it does not match this
  design doc; its **MLIR layer does** match the doc's §5 compilation model closely.

### The strategic call

Two candidate paths to "runable":

1. Build the rolled `spmw.place` MLIR op + passes + emitter first (the design doc's §5).
2. Elaborate SPMW in pure Python, then **lower generically to `allo.dataflow`**, reusing df's
   proven simulator / Vitis HLS / LLVM backends.

**This plan does (2) first, then (1).** Reasons: (2) needs *zero* C++ and therefore zero rebuild,
which matters because the only machine that can run Allo at all is currently unreachable (see
Constraints); it reaches all three working examples fastest; and it leaves the §5 synthesis-time
win as a well-scoped follow-on rather than a prerequisite.

The critical distinction from `feat/spmw`'s `spmw_datapath.py`: that module *recognized* specific
benchmark shapes (`_recognize_mini_tpu`, `_recognize_fft`, …) and regenerated hand-tailored
dataflow source per shape, so it validated nothing general. **This plan's lowering is a real
compiler pass over the elaborated graph** — topology-driven, shape-agnostic, with no per-benchmark
branches. Anything the elaborator can represent, the lowering can emit.

### Constraints discovered

- **`brg-zhang-xcel` was unreachable throughout** (`ssh` and `ping` both timed out; most likely
  the VPN was not up). Per `CLAUDE.md` and prior project convention, *every* test — including
  the simulator ones — runs there. `scripts/spmw_remote_check.sh` runs the suite in one command
  once it is reachable.
- **Nothing can run Allo locally**: no conda, system `python3` has no numpy, and `allo.ir.types`
  imports the compiled `allo._mlir` extension.
- Mitigation, and a good design property regardless: **`allo/spmw/`'s elaboration core must not
  import `allo._mlir`**. Contracts, topologies, placement, bundles, binding index algebra and all
  of §4's structural checks stay dependency-free, so they are unit-testable under a bare Python
  with only `pytest`. Only the lowering layer imports Allo proper.

---

## Design

### Package layout — `allo/spmw/`

| Module | Contents | Imports `allo._mlir`? |
|---|---|---|
| `ports.py` | `In`/`Out`/`MemIn`/`MemOut`/`Mem` descriptors, `PortSymbol`, accessors | no |
| `iface.py` | `Interface` metaclass, `spmw.interface(...)` functional form, iteration, inheritance | no |
| `component.py` | `@spmw.unit`, `@spmw.fabric`, `.role(unbound=)`, `Site` | no |
| `topology.py` | `Topology`, `to()`, `key()`, `mesh()`, `Grid()`, link validation | no |
| `placement.py` | `place()`, `Placement`, bundles, axis symbols, `split()` | no |
| `bricks.py` | `FIFO`, `RAM`, `mem()`, `banked`/`replicate`/`shared` layouts | no |
| `bindings.py` | `stream_in`, `gather`, `scatter`, `shard`, `stationary`, `copy`, `link`, `phase` | no |
| `index.py` | the `index=` tuple algebra — axis expressions, `...`, lambda escape, bounds checks | no |
| `graph.py` | the elaborated graph (`Site`, `Channel`, `Mover`, `Brick`) + §4 checks | no |
| `lower_df.py` | elaborated graph → `allo.dataflow` program (AST synthesis) | yes |
| `build.py` | `spmw.build(fabric, target=...)` | yes |

`allo/spmw/__init__.py` re-exports the surface so `import allo.spmw as spmw` reads exactly as the
design doc writes it.

### Core mechanisms

**Port symbols.** Members are *assigned* (`west = spmw.In(float32)`), never annotated, so the
declaration constructs a runtime symbol. `__set_name__` binds the name; the metaclass collects
`__ports__`. Class access returns the singleton `PortSymbol` (hashable, identity-comparable, keys
link dicts); instance access returns a site-bound accessor whose `.get()`/`.put()` are what the
unit body calls. Subclasses inherit the *same* symbol objects, so `MacIO.west is NSEW.west`.

**Site signatures drive roles.** After `place`, each site's signature is the set of *bound* ports,
computed by evaluating the link rule at that site. Sites whose `link` dict withholds an edge get an
**interior boundary** — this falls out for free and is what §3.6's slab seams need. Role selection
happens at `place`; whether a binding actually feeds an unbound `In` the body reads is settled at
build, where the set of bindings is known and the diagnostic can name the missing one.

**Channel identity = destination.** A coordinate link `east@(i,j) → west@(i,j+1)` and its inverse
name one channel, canonically identified by `(dst_site, dst_port)`. Grouping channels by unordered
port pair gives a **family** (`east/west`), which becomes one stream array of grid shape. Key-form
links rendezvous on the shared label instead. Both forms produce the same `Channel` records, which
is why the lowering has one path, not two.

### Lowering to `allo.dataflow`

The transcriber synthesizes a Python AST for a complete `@df.region` program and `ast.unparse`s it
into a real module file, which is then imported so `inspect.getsourcelines` works
(`allo/ir/utils.py:144` needs real source). AST-in/AST-out — never f-string templating, which is
what made the old emitters brittle.

Per placement:

```python
fifo_ew: Stream[float32, 2][R, C]        # one array per channel family, indexed by destination

@df.kernel(mapping=[R, C], args=[A, B, C])
def pe(local_A: float32[M, K], ...):
    i, j = df.get_pid()
    with allo.meta_if(_ROLE_pe[i][j] == 0):     # one arm per role/signature class
        <interior body, ports rewritten>
    with allo.meta_elif(_ROLE_pe[i][j] == 1):
        <edge body, unbound Out puts elided>
```

`_ROLE_pe` is a plain nested list in the generated module's globals. This works because
`meta_if`'s condition goes through `ASTResolver.resolve_constant`
(`allo/ir/symbol_resolver.py:132`), which is a bare `eval` against `ctx.global_vars`, and
`_get_global_vars` copies the function's `__globals__` wholesale (`allo/ir/utils.py:48`). Pids are
bound as real ints, so a table lookup evaluates at compile time. **Arm count is O(#roles), not
O(P0·P1)** — the same constant-role-count property the rolled path targets, obtained here at the
source level.

Port rewriting inside a unit body:

| Unit body | Generated |
|---|---|
| `io.west.get()` | `fifo_ew[i, j].get()` (this site is the destination) |
| `io.east.put(a)` | `fifo_ew[i, j+1].put(a)` — **elided entirely** when the far end is off-grid or unmapped |
| `io.w` (`MemIn`, rank-0 residual) | `local_W[i, j]` — the shard's index expression evaluated at this site |
| `io.c = acc` (`MemOut`) + `gather(C, from_=P.c)` | `local_C[i, j] = acc` |

Movers become their own kernels, *not* a halo ring around the compute grid:
`spmw.stream_in(A, into=P.west, index=(P.rows, ...))` emits
`@df.kernel(mapping=[R], args=[A])` looping the `...` axis and putting into `fifo_ew[k, 0]`. This
is what generalizes to §3.6, where loaders must appear at *interior* slab seams that no halo ring
can express.

**Honest scope limit:** this path hands Vitis HLS one function per PID, exactly as `df` does
today, so it does **not** deliver the design doc's §5 synthesis-time win. That is S6's job.

---

## Status

Everything below is implemented and committed on this branch. What is **not** yet
confirmed is the one thing that needs `brg-zhang-xcel`: whether Allo's own tracer
accepts the generated program (`target="simulator"` and `vitis_hls`). The host was
unreachable for the whole session it was written in.

| Piece | State |
|---|---|
| Contracts (`ports`, `interface`, `component`) | done |
| Topologies and placement (`topology`, `placement`, `index`) | done |
| Fabric elaboration and the check table (`bricks`, `bindings`, `graph`) | done |
| Lowering to `allo.dataflow` (`channels`, `lower_df`) | done, **unverified against Allo** |
| Reference simulator (`refsim`, `target="ref"`) | done |
| Hierarchical placement (a fabric placed on a topology) | done, expanded per site |
| Docs (`docs/source/dive/spmw.rst`) | done |
| Rolled MLIR path (the §5 synthesis-time win) | **not started** — see Follow-on |

### The worked examples

All five of the design doc's examples run and match numpy, verified two
independent ways (see Verification):

| Example | Exercises |
|---|---|
| §3.1 systolic GEMM | mesh, computed boundaries, `stream_in`/`gather` |
| §3.2 tiled GEMM | a placed fabric, `shard(dim=)`, per-site tensor views |
| §3.4 FFT | key-form links, block streams, a resident twiddle ROM, lambda maps |
| §3.5 mini-TPU | two placements, `link`, stationary weights, a seeded chain; also the staged form with `copy` under `phase` and `double=True` |
| §3.6 attention P·V | interior boundaries, `split` axes, reduction as wiring, G ∈ {1,2,4} |

### Where the rolling is lost, measured

The frontend's O(#roles) property holds and is visible in the emitted source, but
the dataflow builder unrolls per PID, so HLS still sees one function per grid
point. Measured on the systolic GEMM (`target="vhls"`, counting emitted `void`
functions):

| grid | sites | signatures | source arms | HLS functions |
|---|---|---|---|---|
| 2×2 | 4 | 4 | 4 | 9 |
| 3×3 | 9 | 9 | 9 | 16 |
| 4×4 | 16 | **9** | **9** | 25 |
| 6×6 | 36 | **9** | **9** | 49 |
| 8×8 | 64 | **9** | **9** | 81 |

Signatures and source arms go flat at 9 from 3×3 on — that is the design's claim,
and it is already true. The HLS function count is `sites + 2·n + 1`, i.e. one
body per PID plus the movers and the top. Closing that gap is the whole of the
remaining work, and this table is the baseline to beat.

### A cheaper route than a new dialect, measured

Allo already deduplicates kernel instances by control-flow class: `builder.py`'s
`if not ctx.unroll` branch keys each PID on its `func_predicate_tags` and emits
one `func.func` per distinct tag. That path exists for AIE, but the property it
provides is exactly the design's O(#roles) claim. Driving the SPMW-generated
program through it, on the systolic GEMM:

| grid | `unroll=True` funcs | `unroll=False` funcs |
|---|---|---|
| 3×3 | 16 | **12** |
| 4×4 | 25 | **12** |
| 6×6 | 49 | **12** |
| 8×8 | 81 | **12** |

Flat at twelve, at every size, with no new dialect and no new passes. The site
signatures SPMW computes and the predicate tags Allo derives agree, which is why
this works at all.

**What is still missing is the instantiation.** `_build_top` emits no calls on
this path (`func.call = 0` at every size above), so the bodies exist but nothing
invokes them once per grid point. That rolled instantiation loop is the one
genuinely new piece of work, and it is far smaller than the dialect the plan
originally scoped.

Two routes, and the trade is worth stating: extending the dataflow path touches
shared `allo/dataflow.py`, where a mistake reaches the AIE and HLS backends
everyone else uses; the dialect route stays inside SPMW's own files but is
several times the work and needs a C++ rebuild.

### The dialect's step order, revised once it existed

The port plan assumed the old branch's division of labour, where
`spmw-role-partition` and `spmw-resolve-channels` *compute* what the emitter
needs and write it onto the op. This frontend already computes all of it —
signatures, per-site routing, families and their shapes — so `spmw.map` carries
those directly and the two passes change role: they become **checks** that the
declared tables agree with what the links imply, not the source of them.

That moves the critical path. The step that actually makes HLS emit O(#roles)
functions is the emitter hook, so the order is:

1. **Dialect** — done. `spmw.map`, its attributes, and a verifier.
2. **Frontend emits the op** — `allo/spmw/lower_mlir.py`. Everything it needs is
   already computed in `lower_df.py`'s `_wiring_classes` and `channels.py`'s
   `Family`.
3. **Emitter hook** — the six edits in `Visitor.h`, `EmitBaseHLS.h`,
   `EmitVivadoHLS.{h,cpp}`. This is where the number moves, and it is the only
   genuinely new code: the old branch has no rolled instantiation in C++ at all,
   only a Python emitter that regex-scrapes the IR and recognises two benchmark
   shapes.
4. **The two analysis passes** — worth having as verification, but no longer
   blocking, and best written once the emitter has settled what the op must say.

Note also that the role-selection rule (`missing ⊆ point.missing`, widest wins,
ties are an error) is bit-for-bit this frontend's `Unit.body_for`, so when the
passes are written that part is a transcription rather than a design.

### Follow-on: the rolled MLIR path

The lowering emits a dataflow region, so Vitis HLS still schedules once per
instance. Making the design's §5 claim real means keeping the rolled form to
codegen: a `spmw.place` op (model it on `allo.grid_map` for the region and grid,
and on `allo.get_stream_global` for the `AffineMapAttr` links), then port
`feat/spmw`'s `SPMWRolePartition.cpp` and `SPMWResolveChannels.cpp` — both
already compute what this frontend now declares — plus the five-edit emitter hook
for the instantiation loop. That branch's passes were validated to hardware, so
this is a port, not a rewrite. It needs a C++ rebuild on the remote.

Also open: `fold`/`unroll` are carried on `place` but not realised, and a placed
fabric expands per site rather than instantiating one engine.

## Milestones (as planned)


Each is gated by a test that must pass on `brg-zhang-xcel`, and each is committed separately once
green (per established project convention).

**S1 — Contracts and components.** `ports.py`, `interface.py`, `component.py`. Port symbols,
inheritance identity, closed-set iteration, direction/typo errors at trace time, `@spmw.unit` with
`Site`, `.role(unbound=)` including the "role body may not touch a declared-unbound port" check.
*Gate:* `test_spmw_interface.py` — pure Python, runs without a built Allo.

**S2 — Topologies and placement.** `topology.py`, `placement.py`, `index.py`. Both link forms,
ownership/pairing/type checks at `Topology` construction, `place` legality, site signatures,
bundles with their own dense shape, `split` axis symbols, role coverage.
*Gate:* `test_spmw_topology.py` — mesh yields 9 signatures at 4×4/8×8/16×16; `grouped_mxu`
withheld links produce G slab-west bundles; butterfly key form pairs up.

**S3 — Fabric elaboration.** `bricks.py`, `bindings.py`, `graph.py`. All movers, the `index=`
algebra with `...` and bounds/disjointness checking, phases, the §4 check table.
*Gate:* `test_spmw_elaborate.py` — the §3.1/§3.5/§3.6 fabrics elaborate to the expected channel
and mover counts; every §4 diagnostic fires on a crafted negative case.

**S4 — Lowering + `spmw.build`.** `lower_df.py`, `build.py`. The AST transcriber, role dispatch,
mover kernels, memory bindings.
*Gate:* `test_spmw_gemm.py` — §3.1 GEMM on `target="simulator"` matches numpy, **and** is
bit-identical (`np.array_equal`) to a hand-written `df` oracle kept alongside it.

**S5 — The three examples.**
- `test_spmw_gemm.py` — §3.1 systolic GEMM (extends S4's gate).
- `test_spmw_tpu.py` — §3.5 mini-TPU: `mac` + `act`, two placements wired by `spmw.link`,
  weight-stationary `MemIn` scalar port, psum chain seeded by a rank-0 `stream_in`, int8/int32.
- `test_spmw_fft.py` — §3.4 `fft_spatial` at a small `FFT_N` (grid is `S × N/2`, so 1024 would be
  5120 sites — the simulator makes one OMP section per site). Key-form topology, `csample`
  block streams, bit-reversal lambda, replicated twiddle ROM.
- Stretch, same machinery: §3.6 `attention_pv(G)` for G ∈ {1,2,4}.

*Gate:* each matches numpy on `target="simulator"`; Vitis HLS csim where the toolchain is
available.

**S6 — Rolled MLIR path (follow-on).** Port `feat/spmw`'s validated MLIR layer onto the new
contract model: `spmw.place` op (modelled on `grid_map` for the region/grid parts and
`get_stream_global` for the `AffineMapAttr` link parts), `spmw-role-partition` and
`spmw-resolve-channels` (both port largely verbatim — the affine evaluator and subset-selection
rule are exactly what the doc asks), plus the 5-edit emitter hook for the rolled instantiation
loop. Needs a C++ rebuild on the remote.
*Gate:* HLS role-function count constant as the grid scales 8×8 → 16×16 → 32×32.

---

## Reuse

Do not re-derive what exists:

- `allo/ir/utils.py:144` `parse_ast`, `:79` `get_global_vars` — the tracing entry points.
- `allo/dataflow.py` — `@df.region`/`@df.kernel`/`Stream`/`df.build`; the whole backend fan-out.
- `allo/memory.py` — `Layout`/`Shard`/`Replicate` for shard bindings, `Memory` for `resource=`.
- `allo/template.py` `meta_if`/`meta_elif`/`meta_else` — role dispatch.
- From `feat/spmw` (port, don't rewrite): `SPMWRolePartition.cpp`'s affine evaluator and role
  selection, `SPMWResolveChannels.cpp`'s family grouping and fold→buffer reclassification,
  `allo/transform/f2_*.py` (GF(2) banking — no SPMW dependency at all), the coroutine scheduler in
  `spmw_rollsim.py`, and the `test_spmw_equivalence.py` oracle-diff methodology.

Explicitly **not** reused: `spmw_datapath.py`'s per-benchmark recognizers, and every
regex-over-`str(module)` driver in the old emitters.

---

## Verification

Ladder, per the established project convention:

| Level | Check | How |
|---|---|---|
| L0 | elaboration + checks | pure-Python tests, no built Allo needed |
| L1 | bit-identical to a hand-written `df` oracle | `target="simulator"`, `np.array_equal` |
| L2 | functional vs numpy | `target="vitis_hls", mode="csim"` |
| L3 | synthesis completes, role count flat | `mode="csyn"` (S6) |
| L4 | RTL/board | `mode="hw_emu"` / `mode="hw"` |

Workflow (from `CLAUDE.md`): `rsync` up, then run remotely with
`PATH=/scratch/hc676/allo-agent/bin:$PATH` and
`LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build`; source the Vitis env only for L2+.
`pip install -e .` on the remote **only** when C++ under `mlir/` changes — S1–S5 change no C++.
Tests live in `tests/dataflow/spmw/` so CI picks them up with `OMP_NUM_THREADS=128`.

Lint before each commit: `bash scripts/lint/task_lint.sh` (license headers, `black==24.8.0`,
`pylint`).
