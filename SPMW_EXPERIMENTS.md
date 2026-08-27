# Experiments for an SPMW paper

What to measure to support a claim against native HLS, what is already measured
here, and what is missing. Numbers below are from `brg-zhang-xcel`, Vitis/Vivado
2023.2, `xcu280`, 300 MHz target, unless said otherwise — every one is a real
run in this repo's history, and anything unmeasured is marked **TODO**.

---

## 0. What the claim should be

Worth settling before choosing experiments, because it decides which numbers
matter.

SPMW does **not** produce better hardware than HLS. The processing element is
synthesised by Vitis either way, so per-PE quality of results is identical by
construction. Claiming a QoR win would invite a reviewer to find the place where
it isn't true — and E13 found one: **AutoSA's fp32 PE beats ours by 1.28×**, so
the QoR claim is not merely unsupported, it is false in at least one measured
case.

The defensible claim is about **cost and composition**:

> A spatial design's compilation cost should track the number of *distinct*
> processing elements, not the number of instances. SPMW makes it do so, and
> composes the array structurally so the synthesised element is reused rather
> than re-elaborated.

Everything in §1 supports that. §2 exists to show the abstraction costs nothing,
which is a *defensive* result, not the headline.

---

## 1. Compilation scalability — the headline

### E1. Compile time versus array size ✅ measured

Sweep the grid; report frontend, HLS, and total. The claim is a flat line.

| sites | 9 | 16 | 36 | 64 | 144 | 256 |
|---|---|---|---|---|---|---|
| SPMW, 9 roles concurrent | 43 s | 42 s | — | — | — | **43 s** |
| whole-array `csynth` | 39 s | 43 s | 60 s | 93 s | 280 s | **807 s** |
| whole-array Allo lowering | 1 s | 1 s | 2 s | 3 s | 7 s | **578 s** |

Fill the gaps at 36/64/144 for the SPMW row (expected flat) and extend both to
32×32 = 1024 sites, where the monolithic route should become impractical and the
role path should not move. **TODO: 32×32, and the missing SPMW points.**

*Figure: compile time vs. instance count, log y. Two flat lines and one that
takes off.*

### E2. Where the speedup comes from ✅ measured

Three routes at 16×16, same machine, same code generation:

| | int8 | fp32 |
|---|---|---|
| per-role, 9 concurrent | 1.1 + **43.1** = **44.2 s** | 1.2 + 40.7 = 41.9 s |
| per-role, serial | 1.1 + 372.0 = 373.1 s | 1.2 + 360.0 = 361.2 s |
| whole array, one `csynth` | 719.5 + 536.0 = 1255.5 s | 585.3 + 706.0 = 1291.3 s |

Factors into **decomposition ≈3.4×** (serial roles still beat monolithic) and
**parallelism ≈8.6×**, together **28–31×**. This ablation is worth its own
column in the table: it separates "we generate less work" from "we generate
independent work", and only the first is available to a tool that keeps the
array whole.

### E3. Distinct bodies reaching the backend ✅ measured

A size-independent structural count, cheap to run at sizes too large to
synthesise:

| grid | 3×3 | 4×4 | 8×8 | 16×16 |
|---|---|---|---|---|
| site signatures | 9 | 9 | 9 | 9 |
| SPMW roles synthesised | 9 | 9 | 9 | 9 |
| functions in the array program | 16 | 25 | 81 | 289 |

Pinned by `tests/dataflow/spmw/test_spmw_rolled.py`. Extend to 32×32/64×64,
which costs nothing.

### E4. Where compile time goes ✅ measured

For one small PE, `csynth_design` spends **23.2 s of 31.6 s in "Source Code
Analysis and Preprocessing"** and ~0.5 s in scheduling, binding and RTL. Trimming
Allo's nine-header preamble to what the unit uses took one role from **31.6 s to
8.8 s** and nine roles from 545 s to 332 s.

Useful as a methodological point — a per-PE compiler is dominated by fixed
front-end cost, which is exactly what decomposition amortises badly and
parallelism amortises well. Also an honest caveat: some of the 28× is Vitis
overhead, not algorithmic.

---

## 2. Quality of results — the defensive result

The point is *parity*, and a reviewer will look for a regression.

### E5. Array-level performance and area ✅ measured (synthesis only)

Measured from RTL simulation (cycles) and Vivado synthesis (clock, area) — not
from the HLS report, which describes one unit and never sees the fabric.

16×16, 4096 MACs:

| | int8 | fp32 (`ii=4`) |
|---|---|---|
| array clock | 1.595 ns (WNS +1.738) | 3.215 ns (WNS +0.118) |
| cycles | 52 | 102 |
| wall clock | 82.9 ns | 327.9 ns |
| LUT / FF / DSP | 10,458 / 11,136 / 256 | 139,749 / 120,960 / 768 |

**TODO — the important gap:** this is post-*synthesis*. A paper needs
**place-and-route**: real Fmax, real utilisation, and confirmation that a
256-instance fabric routes. Run `opt_design/place_design/route_design` and report
post-route WNS. Without it a reviewer will discount the clock numbers.

**TODO:** the same designs built by the whole-array HLS route, taken to P&R, to
show the two produce equivalent hardware. This is the parity experiment and it
does not exist yet — currently the two paths are compared only on compile time.

### E6. On-board execution ❌ not done — and blocked on E14

Blocked by the same gap: SPMW emits no memory interface, so there is nothing to
package as an `.xo` yet. AutoSA documents the full path (bitstream, then run
on-board) and its tooling supports `cosim_design` too, so the baseline can go
wherever we can.

Package as an `.xo`, link with `v++`, run under XRT, compare against a CPU
reference. `feat/spmw` has `_vitis_rtl_package_tcl` for the packaging step.
Without this the paper is simulation-only, which is defensible for a compiler
paper but weaker.

---

## 3. Expressiveness

### E7. The design suite ✅ measured

Six designs through one flow, each verified in RTL simulation against the
reference simulator:

| design | what it exercises | roles | instances | cosim |
|---|---|---|---|---|
| systolic GEMM | mesh, computed boundaries | 9 | 9…256 | ✅ |
| int8 GEMM | same mesh, integer arithmetic | 9 | 256 | ✅ 256/256 |
| tiled GEMM | placed fabric, per-site tensor views | 16 | 16 | ✅ 16/16 |
| FFT | keyed links, block tokens, resident ROM | 3 | 12 | ✅ 8/8 |
| mini-TPU | two placements joined by `link` | 10 | 20 | ✅ 24/24 |
| attention P·V | interior boundary, split axes | 7 | 18 | ✅ 12/12 |

**TODO:** a design a reviewer will recognise as an application, not a kernel —
a full attention layer, or a CNN layer with a real dataflow choice. Six kernels
is thin for an expressiveness claim.

### E8. Lines of code ❌ not done

Count the SPMW description against hand-written HLS for the same design, and
against the generated array program (which is a proxy for what you would
otherwise maintain): the 16×16 array program has **289 functions**. Cheap to
produce and reviewers like it, but weak on its own — pair it with E9.

### E9. What changes when the design changes ❌ not done

Stronger than raw LoC. Take one design and vary it — grid size, dataflow
(output-stationary → output-flowing), number format, a boundary condition — and
report the diff in the *source* versus in the generated HLS. SPMW's argument is
that a one-line change should not be a rewrite. `examples/spmw/generated/gemm`
against `gemm8` is exactly this experiment already staged: structurally identical
descriptions, and only the types move.

---

## 4. Ablations

### E10. Roles versus instances ✅ measurable now

Rebuild with the role partitioning disabled (one unit per site) and show
compile time returning to the whole-array curve. This isolates the paper's
central mechanism and is the ablation a reviewer will ask for first. Not yet run
as a clean sweep — **TODO**.

### E11. The scheduling primitive ✅ measured, with a caveat worth publishing

`spmw.pipeline(P, ii=n)` binds the accumulator's adder to `n-1` cycles.

| asked | II | period | ns/MAC | closes 3.33 ns |
|---|---|---|---|---|
| off | 7 | 2.431 ns | 23.3 | yes |
| **4** | **4** | **3.165 ns** | **13.3** | yes |
| 2 | 2 | 6.692 ns | 13.4 | **no** |
| 0 (latency 0) | 1 | 26.270 ns | 26.3 | no |

**1.75× at the unit, and only 1.06–1.09× on the assembled array** (82.0→77.2 ns
at 3×3; 147.1→135.0 ns at 6×6), because the shorter adder lengthens the fabric's
critical path. Publish both numbers. The gap between them is a genuinely useful
observation — unit-level II is a misleading proxy — and hiding it invites the
reviewer who measures the array to find it.

### E12. Why II=1 is unreachable in fp32 ✅ measured

Three variants of one loop separate the causes:

| arithmetic | carries a value? | II |
|---|---|---|
| float | yes, distance 1 | 7 |
| int | yes, distance 1 | **1** |
| float | no | **1** |

II ≥ latency of the recurrence ÷ dependence distance. Removing either condition
gives II=1. Good material for a "what the abstraction exposes" section: the
frontend's dataflow choice, not the compiler, decides whether peak is reachable.

---

## 5. Baselines

### E13. AutoSA ✅ measured

AutoSA (Wang et al., FPGA'21) built from source and configured to match:
`--sa-sizes="{kernel[]->space_time[3];array_part[16,16,16];latency[1,1];simd[1]}"`
gives 256 PEs, and `space_time[3]` was confirmed **output-stationary by reading
the generated PE** — one `local_C[1][1]` accumulator held across the whole `k`
loop, no `fifo_C_in` — rather than by trusting the flag name. `[4]`/`[5]` are
output-flowing. int8 is just the declared C types. AutoSA's SIMD does *not*
auto-pack narrow types; `simd[N]` packs N lanes at any width, so `simd[1]` is the
matched setting.

**Compile time**, same part and clock:

| | AutoSA int8 | SPMW int8 | AutoSA fp32 | SPMW fp32 |
|---|---|---|---|---|
| frontend / codegen | 1.52 s | 1.1 s | 1.55 s | 1.2 s |
| HLS, monolithic | 1652.1 s | 1255.5 s | 1657.0 s | 1291.3 s |
| **HLS, per-role concurrent** | **n/a** | **43.1 s** | **n/a** | **40.7 s** |
| Vivado synth, 256 instances | 595.8 s | 224.9 s | 1401.0 s | 867.4 s |

**Throughput.** Note what this metric is: II × array clock is a *steady-state
per-MAC bound*, not measured end-to-end throughput. It ignores fill, drain and
any array-level stall. It is used here because no comparable end-to-end number
exists — see "cycles" below.

| | II | array clock | ns per MAC | what limits the clock |
|---|---|---|---|---|
| AutoSA fp32 | 4 | 2.510 ns | **10.04** | a PE's `fadd_..._5_no_dsp` |
| SPMW fp32, `ii=4` | 4 | 3.215 ns | 12.86 | a PE's `fadd_..._4_no_dsp` |
| SPMW fp32, default | 7 | 2.411 ns | 16.88 | — |
| AutoSA int8 | 1 | 2.126 ns | 2.13 | **`A_IO_L2_in` — the DRAM I/O network** |
| SPMW int8 | 1 | 1.595 ns | 1.59 | a PE's DSP MAC |

**The int8 row is not a like-for-like comparison and should not be quoted as a
win.** Both designs reach II=1, and the clock difference is not a property of
their compute: AutoSA's critical path is in `A_IO_L2_in_5_U0` — the second level
of its DRAM input network — with 78% of the delay in routing. SPMW has no such
network, so it is not competing for that path. The honest statement is that on
int8 **both reach II=1 and no throughput difference has been demonstrated**.

The fp32 row *is* like-for-like: both critical paths land in a PE's float adder.
There AutoSA is genuinely ahead — and interestingly its adder is the *deeper*
core (latency 5 against our 4), which should bound II at 5 or 6 by the recurrence
argument in E12, yet it reports II=4. Our model does not explain that, and it is
the single most useful thing left to chase.

**Where AutoSA wins, and it matters.** Its fp32 PE reaches II=4 *by default*,
where ours needs `spmw.pipeline(ii=4)` to get there — and it does so at a better
period, so it beats both of our fp32 operating points: **1.28× our tuned one and
1.68× our default**. It gets the interval and the clock together where we had to
trade one for the other. Vitis names the mechanism in our case and not theirs, so
their PE's adder is coming out shallower by default; **finding out why is the
most useful follow-up in this document.**

**Where SPMW wins.** int8 throughput by 1.33×, and compile time — decisively, but
only through role decomposition, which is a structural difference: AutoSA emits
one monolithic kernel that cannot be split across cores. Against the monolithic
routes the margin is a modest 1.3×, and part of even that is explained below.

**Caveats that must travel with these numbers:**

- **AutoSA builds a whole accelerator; SPMW builds a bare array.** AutoSA's
  kernel is ~595 modules: 256 PEs *plus* a C-drain network, a 3-level DRAM I/O
  network, and AXI master + AXI-Lite control. **SPMW's design has no DRAM
  interface at all.** Quote the PE-array-only row (AutoSA int8 15,557 LUT against
  SPMW's 10,458; fp32 157,281 against 139,749 — 1.13–1.49×), never the
  whole-kernel row, which overstates AutoSA's cost by 5×.
- **Which means the compile-time comparison is not clean either**: AutoSA's
  1652 s is synthesising ~2.3× the modules.
- **Cycle counts: both are now cosim, and they still are not comparable.**
  SPMW's 52/102 come from RTL simulation — xsim driving the assembled array,
  counting clocks from reset release to the last output token. AutoSA's int8
  kernel now has a measured number too: **859 cycles, C/RTL co-simulation PASS**.
  But it is a whole accelerator — DRAM reads through AXI, the PE array, and a
  serialised C drain — against our bare fabric fed directly on its edge streams.
  859 against 52 is a scope difference, not a performance one. See E14.

  **Worth publishing in its own right:** AutoSA's *HLS latency estimate* for the
  same design is 401 cycles against 859 measured — **the estimate is 2.1×
  optimistic**. Any comparison built on HLS estimates rather than simulation is
  wrong by about that much, in the tool's own favour. It also means AutoSA's
  autotuner, which optimises against an analytical latency model rather than
  simulation, is tuning against a number with that much slack in it.

  Reproducing it needed one fix: AutoSA emits `m_axi` pragmas with no `depth=`,
  which csynth accepts and cosim rejects (`A depth specification is required for
  MAXI interface port 'gmem_A'`). Depths were added to a *copy*, leaving the
  checked-in artefact exactly as AutoSA emitted it.
- AutoSA fp32 needed `export_design` (1312 s, excluded above) before
  `synth_design` would resolve its FP cores.
- AutoSA's `mm` uses `B[J][K]` transposed, and its stock kernel leaves `C`
  uninitialised — adding the zeroing makes the PE byte-identical bar one
  register.

**Reproducing the build** (~8 min of machine time; ~40 min including diagnosis):
needs NTL and LLVM/Clang 9 in user space, plus two fixes — `src/autogen.sh` must
be run **twice** (the first pass emits no top-level `configure` and exits 0), and
`src/autosa_common.cpp:1231` has `if (index < 0)` on an `isl_multi_pw_aff *`,
which GCC 11 rejects; upstream ppcg has `if (!index)`.

### E15. The daisy-chained mesh — a structural match ✅ measured

`tests/dataflow/spmw/test_spmw_daisy.py` reimplements Allo's
`test_daisy_chain_gemm.py` in SPMW: results are **chained** out of the array
rather than given a port each. Every PE takes the partial column arriving from
the north, drops its own result into its own slot, and passes it south; the
bottom row's columns are what leaves. That is AutoSA's `C_drain_IO_L1_out`
network expressed as a link instead of as generated glue, and it is what makes
the two comparable.

At 16×16 against the plain mesh:

| | plain mesh | daisy-chained |
|---|---|---|
| edge streams | 288 | **48** |
| internal FIFOs | 512 | 768 |
| cosim cycles | 52 | **660** |
| array clock | 1.595 ns | 1.665 ns |
| LUT / FF / DSP | 10,458 / 11,136 / 256 | 147,848 / 150,256 / 256 |

**This reframes the AutoSA comparison, and not in our favour.** The plain mesh's
52 cycles were fast because it drains through 256 parallel output ports — which
is not something a real systolic array does, and not something that could be
fed by any realistic memory system. Chain the drain, as both AutoSA and Allo's
own daisy-chain design do, and the same computation takes **660 cycles**.

Against AutoSA's 859 measured cosim cycles, that is a far more meaningful
number: 660 for a bare fabric with a chained drain, 859 for a whole accelerator
with a chained drain *plus* DRAM reads. The remaining gap is plausibly the
memory system — which is exactly what E14 says to go and build.

**Do not quote the plain mesh against AutoSA.** Quote this.

Remaining mismatches with AutoSA, in order of size:

1. **A and B are still fed directly**, one edge stream per row and column;
   AutoSA daisy-chains them too through its `A_IO_L2_in`/`B_IO_L2_in` networks.
   Extending this design to chain the inputs is the obvious next step and would
   use the same mechanism.
2. **int16 here against AutoSA's int8** — inherited from Allo's original. The
   cycle count should be insensitive to that; the area is not.
3. Still no DRAM interface (E14).

**One general bug this found.** A bare `gather` from a *stream* bundle computed
its positional identity from `placement.grid` rather than the bundle's own
shape, so draining off an edge demanded a tensor shaped like the whole mesh.
`_positional_identity` now uses the bundle. It only shows up when a port is
consumed by a peer link everywhere but one edge, which no earlier design did.

### E14. Making the AutoSA comparison fair — decide this before writing

The comparison as it stands is contaminated in **all three** headline metrics, in
the same direction, by one asymmetry: **AutoSA generates a complete accelerator
and SPMW generates a bare compute fabric.**

| metric | who it flatters | why |
|---|---|---|
| compile time | SPMW | AutoSA synthesises ~2.3× the modules |
| int8 array clock | SPMW | AutoSA's critical path is in `A_IO_L2_in`, which we have no counterpart for |
| whole-kernel area | SPMW | ~5×, almost all of it I/O and drain networks |

None of those is a compute-fabric result. Publishing them as-is invites exactly
the rebuttal that the numbers measure the absence of a feature.

**What AutoSA's I/O actually is**, from its int8 kernel: 256 PEs, an A and B DRAM
input network (`A_IO_L2_in`/`B_IO_L2_in`, O(N) modules each), and a **C-drain
network of 241 wrappers — O(N²), one per PE**. The drain is the large part, and
it is large because collecting one result per PE is intrinsically O(N²) work.

**Two ways to fix it, and a recommendation.**

*Option A — strip AutoSA to its PE array.* Cheap, and their hierarchical
utilisation report already isolates it (int8: 15,557 LUT for PEs + inter-PE FIFOs
against SPMW's 10,458). But it measures a design AutoSA never intended to emit
standalone, its I/O network and latency hiding are *claimed contributions* of the
paper, and it leaves SPMW still undeployable. It also does not fix compile time,
which cannot be meaningfully attributed to "the PE part" of a monolithic csynth.

*Option B — give SPMW a memory interface.* Real work: address generation, AXI
master, a drain collector. But it removes the objection permanently rather than
arguing around it, makes every metric a system-to-system comparison, and unlocks
E6 (on-board), which is the strongest evidence available.

**Recommendation: B, with A as the interim.** For a submission now, report
PE-array-only for QoR on both sides and state the compile-time asymmetry in
words rather than normalising it away. For a strong submission, build the
interface.

**A testable prediction worth stating either way:** SPMW's compile-time claim
should *survive* adding the I/O, because SPMW would generate that network
**structurally in RTL** — the same `generate`-nest treatment the fabric gets —
rather than putting it through HLS scheduling. AutoSA pays csynth for all 595
modules; SPMW would pay csynth for the roles and Vivado elaboration for the rest.
If that holds, the compile-time result stops being an artefact of generating less
and becomes a claim about *how* the I/O is generated. If it does not hold, that
is worth knowing before the paper is written, not after.

**Also unresolved:** whether AutoSA can be asked *not* to emit a DRAM interface
(a stream-fed array), which would make Option A a supported configuration rather
than surgery. Worth checking its flags before doing anything else here.

### The gap this exposes ❌

**SPMW has no memory interface.** AutoSA generates the DRAM I/O and drain
networks that make its array a deployable kernel; ours ends at the fabric's edge
streams. This is a functional gap, not a measurement artefact, and a reviewer
will raise it. Either build the interface, or state plainly that SPMW composes
the compute fabric and delegates I/O — and then compare only the compute fabric,
as the PE-array-only row does.

### Still needed

1. **Hand-written HLS** — what an engineer actually writes: one templated PE plus
   an unrolled instantiation loop. Still missing, and still the most honest
   "native HLS" column.
2. The whole-array route taken to P&R (see E5) as the internal parity control.
3. **SuSy**, **Spatial**, or **HeteroCL** as a second published point.

---

## 6. Threats to validity, to state rather than be asked

- **The monolithic route works.** At every size measured it returns `rc=0`. The
  advantage is cost, not feasibility. Say so; a reviewer who tries it will find
  out.
- **One caveat to that:** the whole-array program does *not* synthesise without
  `#pragma HLS array_partition` on the tensor arguments, because every site
  writes one element of the result and HLS dataflow permits one writer per
  interface array. That is a lowering fix, not an argument for SPMW, and it
  belongs in a footnote rather than the results.
- **Part of the speedup is tool overhead**, not algorithmic (E4).
- **Cycle counts are from small problems** where fill and drain dominate — 52
  cycles for 4096 MACs at 16×16 is ~31% PE utilisation, because the wavefront
  skew (30 cycles) exceeds the steady state (16). Report utilisation honestly or
  use a larger K.
- **Synthesis-only clocks** until E5's P&R is done.
- **One FPGA, one toolchain version.**
- **Against AutoSA, the compile-time margin is 1.3× on the monolithic route** and
  only becomes large through role decomposition — and AutoSA is synthesising
  ~2.3× the modules, because it builds a whole accelerator where SPMW builds a
  bare compute fabric. Present the decomposition as the architectural claim it
  is, not as a raw ratio.
- **SPMW generates no memory interface.** AutoSA's DRAM I/O and drain networks
  are most of its area and none of ours. Compare compute fabrics, and say why.

---

## 7. Suggested figures

1. Compile time vs. instances, log y — the flat line against the rising one (E1).
2. Stacked bar: decomposition vs. parallelism vs. remaining (E2).
3. Distinct bodies vs. grid size — flat at 9 (E3).
4. Per-design table: roles, instances, cycles, clock, area, cosim (E5, E7).
5. Unit II versus array speedup, side by side — the honest one (E11).

---

## 8. Reproducing

```bash
python3 scripts/spmw_build_array.py --design gemm8 --size 16 --synth --out DIR --cosim
python3 scripts/spmw_build_array.py --design gemm --size 16 --jobs 1 --out DIR   # serial
python3 scripts/spmw_dump_generated.py --out examples/spmw/generated             # the artefacts
```

`--tune` sweeps the initiation interval; `--jobs 1` gives the serial baseline;
`--synth` runs Vivado on the whole fabric rather than elaborating it.
`SPMW_SUMMARY.md` carries the measurements and how each was taken.
