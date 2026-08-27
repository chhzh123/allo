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
it isn't true.

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

### E6. On-board execution ❌ not done

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

## 5. Baselines — the biggest gap

Everything above compares SPMW against **Allo's own dataflow expansion**. That is
a fair internal control and it is *not* a baseline a reviewer will accept as
"native HLS".

**Needed, roughly in order of importance:**

1. **Hand-written HLS** for the same designs — what an engineer actually writes:
   one templated PE plus an unrolled instantiation loop, `#pragma HLS dataflow`.
   This is the honest "native HLS" column and it is currently missing entirely.
2. **AutoSA** (Wang et al., FPGA'21) — the closest published systolic-array
   compiler. Comparing compile time and QoR against it is close to mandatory.
3. **SuSy**, **Spatial**, or **HeteroCL** as a second point, depending on the
   framing.
4. The whole-array HLS route taken to P&R (see E5) as the internal parity
   control.

Without at least (1) and (2), the compile-time result reads as "our second
backend is faster than our first".

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
