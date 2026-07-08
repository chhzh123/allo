# SPMW rolled lowering — soundness note

This note is the M2 soundness artifact (task2.4): why the rolled `spmw.map` and its role-partition /
channel-resolution passes preserve the per-grid-point behavior of the unrolled form, so the rolled
O(#roles) HLS computes the same result as the fully-expanded one.

## The claim

For a systolic mesh region, the rolled top emitted from `spmw.map` (one `pe_interior` body stamped
across the grid, with `load_a`/`load_b`/`drain` boundary tasks over the FIFO families) is behaviorally
equivalent to the fully-unrolled dataflow program (one kernel instance per grid point). The evidence
ladder: bit-identical simulator equivalence (L1, `test_spmw_equivalence`), then real-Vitis csim
(L2, the rolled top == A@B) and csynth (L3).

## Why role selection is well-defined

`spmw-unroll` and `spmw-role-partition` classify each grid point identically:

1. **Missing ports are a function of the point alone.** A point's missing (off-grid) ports are found
   by evaluating each peer link's affine map at the point's coordinate and testing in-bounds. The
   affine maps are static (dims-only, no symbols), so the result is a deterministic constant per
   point — no runtime state, no ordering dependence.

2. **Role selection is a total, unambiguous function.** A role fits a point when its `missing` set is
   a subset of the point's missing ports; the most specific (largest `missing`) wins; two
   incomparable fits are rejected as an ambiguity at verify/pass time (`selectRole` returns −1 → the
   pass fails). So every point maps to exactly one role, and the verifier (`MapOp::verify`) rejects a
   map that could be ambiguous.

3. **Link-presence classes partition the grid.** Two points share a class iff the same set of links
   is missing; `spmw.link_classes` counts these classes. Their instance counts sum to the grid size
   (checked in tests: 4×4 → sum 16, 8×8 → sum 64), so the classification is a true partition — every
   point is counted once.

## Why the interior datapath is preserved

The interior role func carries the *transcribed* work-unit body (the same `allo.stream_get/put`,
`arith`, and `memref` ops the dataflow frontend emits), PID-parameterized. The IR-driven HLS emitter
(`emit_rolled_hls_ir`) translates exactly those ops to the `pe_interior` body, binding stream ops to
the interior role's declared `ports`. It does not re-derive the computation, so the per-PE arithmetic
— including float accumulation order — is the one the frontend defined. This is why the rolled top is
bit-identical to the hand-written df original at L1 and csim-matches A@B on real Vitis at L2.

## Why the FIFO families are the right channels

`spmw-resolve-channels` groups each peer link and its reciprocal into one undirected channel family
(named by the sorted port-pair) and records the per-family FIFO depth (`spmw.channel_family_depths`),
which the emitter uses for the `hls::stream` array declarations. A link and its reciprocal share one
FIFO in the unrolled form (the `spmw-unroll` channel-sharing), and the verifier enforces reciprocal
symmetry (same depth and element type), so the family FIFO carries the same values in the same order
as the corresponding unrolled channels.

## Scope

This argument covers the funded systolic-mesh floor (a single auto-halo compute role, peer-link
interconnect). The predicate-tag interior split (checkerboard) and key-link/collective channels are
general-model extensions and are out of scope for this note.
