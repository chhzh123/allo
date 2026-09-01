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

### E16. The matched design — chained inputs, chained drain, int8 ✅ measured

`tests/dataflow/spmw/test_spmw_autosa_match.py` closes the mismatches E15 left.
Operands arrive through daisy-chained distribution networks (AutoSA's
`A_IO_L2_in`/`B_IO_L2_in`), results leave through a chained drain
(`C_drain_IO_L1_out`), and the arithmetic is int8 into int32. Three placements
joined by `link`; 15 roles, constant in the grid; 288 instances at 16×16 and
**18 edge streams**, against 288 for a plain mesh.

**Matching the topology was not enough — the token width mattered more.** The
first version packed each column into one wide token, as Allo's daisy-chain
design does. AutoSA drains a *scalar* (`typedef int C_t1`,
`hls::stream<int>`): each PE emits its own result then forwards what came from
above, one at a time. Changing only that:

| 16×16, 256 PEs | packed column | **scalar (matched)** |
|---|---|---|
| LUT | 273,998 | **40,022** |
| FF | 287,472 | **24,208** |
| cosim cycles | 694 | **104** |
| array clock | 1.715 ns | 1.748 ns |
| Vivado synth | 1342 s | **367 s** |

**6.8× the LUTs, 11.9× the registers and 6.7× the cycles, from one port's
width.** The packed version puts a 512-bit register in every PE and serialises
the drain behind a full column; the scalar version pipelines it. Worth stating
in the paper as a design-space observation in its own right — the topology is
what gets drawn in figures, and it was the smaller half of the decision.

Against AutoSA's int8 16×16:

| | SPMW matched | AutoSA |
|---|---|---|
| cosim cycles | **104** | 859 |
| array clock | 1.748 ns | 2.126 ns |
| wall clock | **182 ns** | 1826 ns |
| LUT (whole kernel) | 40,022 | 55,358 |
| LUT (their PE array only) | — | 15,557 |
| DSP | 256 | 256 |

Read this carefully. Cycles and wall clock still are not like-for-like: AutoSA's
859 includes DRAM reads through AXI, and ours has no memory system to wait for.
The area row is closer to fair — we are 1.38× *smaller* than their whole kernel
and 2.57× *larger* than their PE array alone, and the truth is between those,
because our fabric now contains the distribution and drain networks their PE-only
figure excludes but not the DRAM interface their whole-kernel figure includes.

**Why the cycle counts differ by 8x — it is not the compute.** AutoSA's own
per-module report says where its 859 go:

| module | latency | count |
|---|---|---|
| `PE_wrapper` — **the compute** | **22** | 256 |
| `C_drain_IO_L1_out_wrapper` | 36 | 240 |
| `A`/`B_IO_L2_in` | 56 | 30 |
| **`C_drain_IO_L2_out`** | **321** | 15 |
| **`C_drain_IO_L3_out`** | **267** | 1 |
| `A`/`B_IO_L3_in` | 27 | 2 |

Compute is 22 cycles of it. The **C drain hierarchy is 321 + 267**, and it ends
at an AXI write. Set against the same 4096 MACs measured end to end:

| | cycles |
|---|---|
| SPMW, plain mesh (256 parallel drain ports) | 52 |
| SPMW, matched (chained feed + chained drain) | 104 |
| AutoSA (chained + three levels + AXI) | 859 |

Chaining costs us **+52 cycles**; it costs AutoSA **+837** over its 22-cycle PE.
The difference is not that our PEs are faster — theirs is 22 cycles and ours is
comparable — it is that their drain aggregates through three levels into DRAM
and ours stops at the fabric's edge after one.

**So: the compute matches, and the cycle comparison is measuring their memory
system against our absence of one.** Which is E14's point arriving from a third
direction.

**That gap is now closed — see "The measured end-to-end comparison" below.**
SPMW's fabric reaches DRAM itself, and the matched 16×16 design completes in
**377 cycles** against AutoSA's 859.

(Our own PE reports no static latency, incidentally: the scalar drain's
`for _i in range(row)` has a coordinate-dependent trip count, so HLS gives `?`.
That is correct behaviour and matches what AutoSA's drain does with `idx`/`idy`.)

**What is left to match:** their second-level C drain, which chains once more
across the bottom row where we still export one stream per column (18 masters
against their 3). The DRAM interface itself is built.

### The measured end-to-end comparison

`StructuralEmitter.fabric(memory=True)` builds each binding's mover as an IP
whose tensor argument is an AXI master, so the fabric loads and stores for
itself. `allo.spmw.dram` puts a behavioural AXI4 slave behind every master and
runs the array from `ap_start` to `ap_done`. Both sides now measure the same
thing: DRAM in, DRAM out.

| 16×16 int8 | SPMW matched | AutoSA |
|---|---|---|
| cosim cycles, end to end | **377** | 859 |
| DRAM ports | 18 | 3 |
| HLS runs to build it | **18** (15 roles + 3 movers) | 1 whole-kernel |
| HLS wall clock | **45.3 s** | 1255.5 s |

Where SPMW's 377 go, from the per-master completion times:

| phase | cycles |
|---|---|
| loading A and B from DRAM | 272 |
| compute wavefront and drain | 105 |

**The loads are 72% of the run, and the reason is not the memory.** The loader's
outer loop runs at II=16 because its inner loop copies the block a byte at a
time — `_blk[_b0] = v0[_t][_b0]`, sixteen sequential accesses per token — so
Vitis never widens the master past 8 bits and 256 elements cost 256 cycles.
AutoSA's `A_IO_L3_in` reads 512 bits a beat and covers the same 256 bytes in
four. Emitting a wide read in the mover would take the load phase from 272
cycles towards ~20, and the whole run from 377 towards ~125. That is the
clearest piece of headroom in the flow and it is codegen, not design.

Sensitivity to the memory, sweeping the model's read latency:

| read latency (cycles) | 0 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 4×4 total | 68 | 76 | 100 | 132 | 196 |
| 16×16 total | 377 | 385 | 412 | 540 | 796 |

At 4×4 the cost is exactly `68 + latency`: one burst per master, latency paid
once. At 16×16 it stays flat to about 32 cycles and then grows, because 256
bytes at 4 bytes a beat is four maximum-length bursts rather than one, and only
the first hides behind the packing loop. Both are consequences of the same
narrow master.

### Area, like for like

With the memory interface built, both sides are whole accelerators, and both
numbers now come from **Vivado `synth_design`** on the same part (xcu280) at the
same clock target (3.333 ns against their 3.330 ns). No HLS estimates on either
side, and no PE-array-only row: that row existed to work around the scope
asymmetry and the asymmetry is gone.

| 16×16 int8, Vivado post-synthesis | SPMW matched | AutoSA |
|---|---|---|
| LUT | **54,485** | 55,358 |
| FF | **43,620** | 96,944 |
| CARRY8 | 1,989 | 86 |
| BRAM | 9 | 4.5 |
| URAM | 0 | 0 |
| DSP | 256 | 256 |
| achieved clock | 2.412 ns (WNS +0.921) | **2.126 ns** (WNS +1.204) |
| cosim cycles, DRAM to DRAM | **377** | 859 |
| wall clock | **909 ns** | 1826 ns |
| DRAM ports | 18 | **3** |
| HLS wall clock | **44.7 s** | 1255.5 s |
| Vivado synthesis | **437.5 s** | 595.8 s |

Reading it:

- **LUTs are a tie** — 1.6% apart, which is noise at this scale. The headline
  "1.38× smaller" from before was measuring our missing memory interface.
- **DSPs are identical** at 256 — one per PE on both sides, which is the check
  that the two designs really are doing the same arithmetic.
- **AutoSA holds the better clock**, 2.126 ns against 2.412 ns. We win on wall
  clock anyway, 909 ns against 1826 ns, because 377 cycles against 859 is the
  larger factor.
- **We pay for it in ports.** 18 AXI masters against 3.

#### Where the area actually goes

`report_utilization -hierarchical`, depth 1, summed by module class. Both columns
reconcile exactly to their design totals, so nothing is double-counted.

| SPMW | inst | LUT | FF | SRL | LUTRAM | DSP |
|---|---|---|---|---|---|---|
| PE array (drain fused in) | 256 | 25,293 | 18,432 | 256 | 0 | 256 |
| channel FIFOs | 800 | 14,841 | 3,200 | 0 | 10,656 | 0 |
| movers (AXI masters) | 18 | 13,456 | 19,304 | 2,890 | 0 | 0 |
| feed chain | 32 | 503 | 2,648 | 0 | 0 | 0 |
| top | 1 | 392 | 36 | 2 | 0 | 0 |

| AutoSA | inst | LUT | FF | SRL | LUTRAM | DSP |
|---|---|---|---|---|---|---|
| FIFOs | 1105 | 31,472 | 55,792 | 0 | 148 | 0 |
| PE array | 512 | 9,310 | 5,632 | 512 | 0 | 256 |
| C drain network | 530 | 9,026 | 14,084 | 1 | 0 | 0 |
| A/B I/O network | 66 | 2,813 | 17,916 | 2 | 0 | 0 |
| PE dummy feeds | 64 | 623 | 384 | 0 | 0 | 0 |
| other | 6 | 2,114 | 3,136 | 327 | 0 | 0 |

Three things fall out, and **two of them are against us**. The earlier reading of
the register gap — that it was AutoSA's I/O buffering — was wrong.

1. **The register gap is a FIFO storage choice, not architecture.** Our
   `spmw_fifo` infers **distributed RAM**: 800 FIFOs hold their data in 10,656
   LUTRAM and spend only 3,200 flip-flops. AutoSA's HLS FIFOs are
   register-based: 55,792 flip-flops across 1,105. That single difference is
   52,592 registers against a total gap of 53,324 — **98.6% of it**. Same
   function, different primitive. It is not evidence that our architecture
   carries less state.

2. **Their PE array is 2.7× smaller than ours** — 9,310 LUT against 25,293, for
   the same 256 DSPs. Because we *fused* the drain into the PE: the body's
   `for _i in range(row)` has a coordinate-dependent trip count, so every PE
   carries a counter, a comparator and a variable-latency FSM. AutoSA puts that
   in a separate `C_drain_IO_L1_out` network. Compared like function:

   | | LUT | FF |
   |---|---|---|
   | SPMW PE array (drain inside) | 25,293 | 18,432 |
   | AutoSA PE array + C drain | 18,336 | 19,716 |

   We are **1.38× larger in LUTs** and level on registers. Splitting the drain
   out of the unit is the obvious thing to try.

3. **The I/O is where we are genuinely comparable.** Our 18 masters cost 13,456
   LUT and 19,304 FF; their 3 masters plus the whole L1/L2/L3 hierarchy cost
   ~11,839 LUT and ~32,000 FF. Similar LUTs; they pay ~1.7× the registers for
   the buffering that gets them from 18 ports to 3.

#### CARRY8, and why ours is 23× theirs

CARRY8 is the UltraScale+ CLB's hardened 8-bit carry chain — dedicated silicon
and fast vertical routing for the carry in an adder, subtractor, comparator or
counter, with LUTs computing the bit functions around it. There are 162,960 on
this part, so at 1,989 and 86 neither design is remotely constrained by it. It is
worth reporting only as a *diagnostic*: it measures how much arithmetic is in the
fabric rather than in DSPs.

Ours, by direct cell query (`get_cells -hier -filter {REF_NAME =~ CARRY*}`):

| | CARRY8 |
|---|---|
| PE array | 1,440 |
| movers (AXI masters) | 448 |
| top | 101 |
| channel FIFOs | **0** |
| total | 1,989 |

The FIFOs contribute nothing, which was the first guess and it was wrong. The
matrix arithmetic contributes nothing either — Vitis puts the whole MAC inside
the DSP (`mul → dsp_slice`, `add → dsp_slice` in the role's report), so the
accumulator never touches a carry chain.

What is left is **the fused drain again**: 5.6 CARRY8 per PE for the loop counter
and the comparison against a runtime `row`, where AutoSA's PE is a pure MAC with
a fixed trip count. Plus 25 per mover for 64-bit AXI address arithmetic, which
they pay once for 3 masters and we pay 18 times.

So the LUT anomaly and the CARRY8 anomaly have **one shared root cause**. It is
not, as first guessed, that the drain is fused into the unit.

### Splitting the drain out, and what it actually showed

Four variants at 16×16, all int8, all Vivado post-synthesis, all passing cosim
DRAM to DRAM:

| | fused | split | split + specialised | **fused + specialised** | AutoSA |
|---|---|---|---|---|---|
| LUT | 54,485 | 59,172 | 47,717 | **43,209** | 55,358 |
| FF | 43,620 | 44,644 | 38,308 | **36,788** | 96,944 |
| CARRY8 | 1,989 | 1,989 | 560 | **549** | 86 |
| DSP | 256 | 256 | 256 | 256 | 256 |
| clock | 2.412 ns | 2.341 ns | 2.202 ns | **2.202 ns** | 2.126 ns |
| cosim cycles | 377 | 378 | 378 | 377 | 859 |
| wall clock | 909 ns | 885 ns | 832 ns | **830 ns** | 1826 ns |
| roles | 15 | 18 | 31 | 54 | — |
| instances | 288 | 544 | 544 | 288 | — |
| HLS wall clock | 44.7 s | 45.7 s | 55.0 s | 108.8 s | 1255.5 s |
| Vivado | 437.5 s | 469.8 s | 441.8 s | 433.2 s | 595.8 s |

**Splitting the drain out on its own made things worse** — 59,172 LUT against
54,485, because 256 extra module boundaries mean 256 extra FIFOs, and the CARRY8
total did not move at all. A direct cell query says why: the 1,440 carry chains
that were in the PEs reappeared, to the cell, in the drains (`u_drain_r0` 1,344 +
`u_drain_r2` 96). The work is intrinsic to the drain. Moving it across a module
boundary relocates it and charges for the boundary.

**What the carry chains actually are is a *runtime* trip count.** `for _i in
range(row)` where `row` arrives on a `_pid` stream needs a counter, a comparator
and a variable-latency FSM in every instance. AutoSA does not have this because
it emits **a separate specialised module per instance** — `C_drain_IO_L1_out_314`,
`_330`, `_346` and so on, 1,872 modules in all — so every trip count is a literal
in its own module. That is what its 1255.5 s of HLS buys.

So the real trade is not fused-versus-split. It is:

> A role stands for many sites and is *told* where it is, which keeps the role
> count independent of the grid — and makes everything derived from the position
> runtime logic.

### `place(..., specialise=...)`

Which is now a choice rather than a fixed property. Naming a grid axis makes that
coordinate part of the *role* instead of an input to it: the body sees a literal,
the fabric drives no `_pid`, and a loop bounded by it has a constant trip count.
One drain unit, measured both ways at 16×16:

| drain role | FF | LUT | latency |
|---|---|---|---|
| generic (row on a `_pid` stream) | 70 | 207 | unbounded |
| specialised (row is the role) | **10** | **161** | 9 cycles, known |

Applied to the whole array it is worth **21% of the LUTs, 16% of the registers
and 72% of the carry chains** against the fused baseline, and takes the clock
from 2.412 ns to 2.202 ns — with the cycle count unchanged at 377.

**The price is roles, and roles are concurrent.** Specialising the drain alone
takes 18 roles to 31 and the HLS wall clock from 45.7 s to 55.0 s; specialising
the fused mesh takes 15 to 54 and 44.7 s to 108.8 s. The CPU cost is real — 3608 s
against 793 s — but it parallelises, which is exactly the property the role
decomposition was for. Even the most specialised variant compiles **11.5× faster
than AutoSA** while being 22% smaller and 2.6× lighter on registers.

**And the split was not needed.** `fused + specialised` beats `split +
specialised` on every metric (43,209 LUT against 47,717, 288 instances against
544) because it does not pay for the extra module boundary. The useful change was
specialisation; splitting was a wrong guess that the control run caught.

**What the area numbers leave out, and it favours us:** neither figure includes
the shell's AXI interconnect, and servicing 18 masters costs materially more of
it than servicing 3. On a real platform that gap is not free. Chaining the drain
one more level — the one structural thing still missing — would take us to 3 and
close it.

**What the memory interface cost us**, against the same design without it:

| | bare fabric | memory-mapped |
|---|---|---|
| LUT | 40,022 | 54,485 |
| clock | 1.748 ns | 2.412 ns |

+36% area and +38% period for the AXI masters and their address generation. That
is the honest price of the thing that makes the comparison meaningful.

**Read the 377-vs-859 carefully.** The memory models are not the same: ours is
described in `allo/spmw/dram.py` — one memory per master, no contention, one
outstanding transaction, a beat a cycle — and AutoSA's 859 was measured under
Vitis' own. The defensible claim is that SPMW is now measuring the same *scope*
as AutoSA rather than a strictly smaller one, and that within our model it is
faster at every latency we swept. A controlled comparison would need AutoSA
re-run against this same slave, which is worth doing before the number is
published.

**Where the hierarchy does and does not matter.** The prediction was that
lacking AutoSA's three levels would degrade memory access badly. Half right, and
the half that is right is not the half one expects:

- *Reuse* is not the problem. The distribution chain already gives it: each
  element of A is fetched from DRAM exactly once and broadcast down the chain,
  which is what AutoSA's L2 is for. The chain is an ordinary SPMW placement.
- *Port count* is a real problem, and it is the drain. A plain mesh would need
  2N masters and the chained design needs N+2 — 18 at 16×16, against AutoSA's
  3. A U280 has 32 HBM channels, so 16×16 fits and 32×32 does not.
- *Burst width* is the immediate cost, and it is the mover's codegen rather
  than the hierarchy.

| design | masters at 16×16 | why |
|---|---|---|
| `gemm8`, plain mesh | 32 | one loader per row and per column |
| `daisy`, chained drain | 48 | the same, plus a per-column drain |
| `autosa`, both chained | 18 | two chain heads, plus a per-column drain |
| AutoSA | 3 | one more drain level than we have |

**Two compiler bugs this design found**, both in paths nothing had exercised:

- A bare `gather` from a *stream* bundle computed its positional identity from
  `placement.grid` rather than the bundle's own shape, so draining off an edge
  demanded a tensor shaped like the whole mesh.
- `(slot,) = site.rank` on a *1-D* placement emitted `slot, = _st__pid0`, a
  scalar unpack, because the unit rewriter shaped the value from the grid rank
  rather than from the assignment target.

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
against SPMW's 10,458). Note the provenance mismatch if this row is ever quoted:
15,557 is Vitis HLS's *estimate* from `csynth.rpt`, while SPMW's 10,458 is Vivado
post-synthesis. The whole-kernel row (55,358) is Vivado on both sides and is the
one to use — see "Area, like for like".

Stripping AutoSA also measures a design it never intended to emit
standalone, its I/O network and latency hiding are *claimed contributions* of the
paper, and it leaves SPMW still undeployable. It also does not fix compile time,
which cannot be meaningfully attributed to "the PE part" of a monolithic csynth.

*Option B — give SPMW a memory interface.* Real work: address generation, AXI
master, a drain collector. But it removes the objection permanently rather than
arguing around it, makes every metric a system-to-system comparison, and unlocks
E6 (on-board), which is the strongest evidence available.

**Done: B.** The memory interface is built and measured; see "Area, like for
like" and "The measured end-to-end comparison". A is no longer needed.

*(The original recommendation, kept for the reasoning.)* **B, with A as the interim.** For a submission now, report
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

## E6. On the board: a 16×16 TPU through place and route ✅ measured

Everything above this line is post-*synthesis*. That distinction turned out to
matter more than it sounds, because the first attempt to place the 16×16 TPU
failed outright — and not for any reason synthesis could have told us.

### A fabric has no boundary a chip can present

`place_design` stopped with 103 errors. The design is not too large; it has too
many *pins*:

| port | width × channels | pins |
|---|---|---|
| `mac_w_mem_dout` | 32b × 256 | 8192 |
| `vpu_b_mem_dout` | 64b × 16 | 1024 |
| `mac_op_in_bind_dout` | 32b × 16 | 512 |
| `vpu_y_out_bind_din` | 32b × 16 | 512 |
| … 20 declarations in all | | |
| **total** | | **11,044** |

against roughly 600 usable I/O on a U280. Synthesis never notices, because
synthesis never places anything. **A fabric is not a deployable unit** — its
interface is streams, and streams are an on-chip idea.

So the fabric needs a wrapper. Two of them, for two different questions.

`allo.spmw.shell.harness_sv` answers the smaller one — can the fabric itself
implement, and what does each stage cost. Inputs come from an LFSR and outputs
fold into one register, because tying inputs to constants lets synthesis delete
the array and report a wonderful, meaningless result. The 288 DSPs in the result
below are the check that it did not: 256 MXU cells plus the vector unit's
multipliers, all present.

### Implementation, 16×16 TPU on xcu280 at a 4.000 ns target

| stage | seconds | share |
|---|---|---|
| HLS, 12 roles concurrently | 47.0 | — |
| `synth_design` | 514.2 | 26.6% |
| `opt_design` | 45.6 | 2.4% |
| `place_design` | 709.7 | 36.7% |
| `phys_opt_design` | 27.4 | 1.4% |
| `route_design` | 636.9 | 32.9% |
| **implementation total** | **1933.8** | **32.2 min** |

| | |
|---|---|
| achieved clock | **3.915 ns (255.4 MHz)**, WNS +0.085 |
| unrouted nets | **0** |
| LUT | 109,203 (8.4%) |
| FF | 111,943 (4.3%) |
| DSP | 288 (3.2%) |
| BRAM / URAM | 0 / 0 |
| CARRY8 | 5,056 |

Two things worth reading off that table. **Place and route is two thirds of the
cost** — 1,346 s of 1,934 — and neither scales with the role count, so the flat
HLS time the rest of this document is about buys less at the full-chip level
than it does at the unit level. And **47 s of HLS against 1,934 s of Vivado**:
once a design is real, the compile-time story is a Vivado story.

### The deployable wrapper is a different object

`shell.feeder_cpp` is the other wrapper: one HLS DMA per boundary *family*
rather than per channel, which is what collapses the port count to something a
board can carry.

| | masters |
|---|---|
| one per channel (what a mover would give) | 321 |
| one per family (what the shell gives) | **6** |

Each feeder is an HLS function with an array of stream ports, so Vitis writes
the AXI master; the 256-port case synthesises in 34 s. The host lays each
family's tokens out in the order `rtl.boundary_plan` says its channels consume
them, so the index arithmetic stays on the host and the hardware walks a flat
buffer.

### The same design, both lowerings, at 16×16

The deployment build takes the *whole-array* path — one monolithic kernel for
`v++` — while everything else in this document takes the role path. Running both
on the same 16×16 TPU gives the cleanest compile-time comparison here, because
it is one design and one machine:

| to RTL | role path | whole-array path |
|---|---|---|
| Allo lowering + HLS codegen | 2.0 s | 196.3 s |
| C synthesis | **47.0 s** (12 roles, concurrent) | **2439.3 s** (one `csynth`) |
| **total** | **49.0 s** | **2635.6 s** |
| | | **53.8× slower** |

The 2,439 s is one process; there is nothing to parallelise, because the design
is one function. The 47 s is twelve processes on a 48-core machine, and it is
the *same* 47 s at 4×4 — the role count does not change with the array.

That is the argument this repository is about, measured on a real design at a
real size. What the P&R numbers above add is the other half of it: the role path
wins the front end by 54×, and then both paths hand Vivado the same problem.

### The whole-array path does not implement at 16×16

The deployment attempt went through `v++` on the whole-array kernel, and after
**6 h 13 m** the link failed:

```
ERROR: [Constraints 18-1000] Routing results verification failed due to
  partially-conflicted nets: level0_i/ulp/top_1/inst/buf0_179_U/
  U_top_fifo_w8_d2_S_ShiftReg/... (and 9 more)
ERROR: [Common 17-39] 'route_design' failed due to earlier errors.
WNS = -2.294 ns   TNS = -8302.347 ns   (at the default 300 MHz)
```

The named nets are the kernel's own dataflow FIFOs. Two problems at once:
timing missed by 2.3 ns on a 3.33 ns period, and the router could not resolve
the congestion those thousands of FIFOs create.

Set beside the role path on the identical computation:

| 16×16 TPU | role path | whole-array path |
|---|---|---|
| to RTL | 49.0 s | 2635.6 s |
| implementation | **1933.8 s, completed** | 22,418 s, **failed** |
| LUT | 109,203 | 231,082 |
| FF | 111,943 | 281,943 |
| DSP | 288 | 288 |
| BRAM | 0 | 45.5 |
| result | routed, 255 MHz, 0 unrouted | `route_design` failed |

The DSP counts agreeing at 288 is the check that both really are the same
arithmetic. Everything else differs, and in the same direction: inlining 272
instances into one HLS function costs **2.1× the LUTs and 2.5× the registers**,
and then does not fit through the router.

That is a stronger argument for the role decomposition than any compile-time
number in this document, and it was not one this work set out to make. The
whole-array path had been treated throughout as the slow-but-equivalent
alternative. At 16×16 on a real part it is not equivalent — it is slower to
compile, larger, and it does not close.

*(The retry at a lower kernel clock is what the next section reports.)*

### It runs on the board

The retry at **150 MHz** closed. Routing converged at global iteration 0 with
**zero** overlapping nodes, where the 300 MHz attempt was still at 263,000 after
an hour — the congestion was timing pressure, not capacity.

`v++`, 16×16 TPU, `xilinx_u280_gen3x16_xdma_1_202211_1`:

| step | elapsed |
|---|---|
| Allo lowering + HLS codegen | 195.3 s |
| `v++` HLS (`csynth_design`) | 2439.3 s |
| `v++ system_link` | 29 s |
| `v++ vpl` (synth, place, route, bitstream) | **3 h 35 m 14 s** |
| `rtdgen` + `xclbinutil` + driver | 6 s |
| **v++ total** | **3 h 36 m 21 s** |
| package | 7 s |
| **build to xclbin, end to end** | **4 h 31 m** (16,243.8 s) |

| result | |
|---|---|
| xclbin | 58 MB |
| timing | **WNS 0.000 ns, TNS 0.000, 0 failing endpoints** of 1,146,593 |
| LUT | 348,840 (26.8%) |
| FF | 433,189 (16.6%) |
| DSP | 292 (3.2%) |
| BRAM | 243 (12.1%) |

Those totals are the whole device image — the XDMA shell, the AXI
infrastructure and the kernel — not the kernel alone.

**The Transformer block, executed on the card**, one invocation per step,
compared against the reference for that step:

| step | s | | step | s |
|---|---|---|---|---|
| 1 proj0 (Q) | 4.2 | | 7 softmax | 4.1 |
| 2 proj1 (K) | 4.1 | | 8 attn (P·V) | 4.1 |
| 3 proj2 (V) | 4.1 | | 9 proj_out + residual | 4.0 |
| 4 scores (K·Q′) | 4.0 | | 10 ffn1 (ReLU) | 4.1 |
| 5 row_max | 4.2 | | 11 ffn2 + residual | 4.1 |
| 6 row_sum | 4.0 | | | |

**11/11 steps match.** The ~4 s per step is host round-trip — OpenCL buffer
setup, PCIe transfer and kernel launch for a 16×16 tile — not compute; the array
itself finishes in the low thousands of cycles. Comparing that number to
anything would be measuring XRT.

**A bug this turned up in Allo.** The second and later invocations reuse an
existing xclbin, and that path built the command as `../{bitstream_folder}` with
`bitstream_folder` already absolute, giving `..//scratch/...`. Only the *second*
call ever hit it, so it survived every single-shot test. Fixed in
`allo/backend/hls.py`.

### How to read these numbers

**The workload.** The block is functionally complete — QKV, scaled scores,
softmax, attention, output projection with a residual, and a two-layer FFN with
a residual — at the dimensions a 16×16 array holds:

| | |
|---|---|
| model dimension / sequence / heads / FFN hidden | 16 / 16 / 1 / 16 |
| parameters | 6 × 16×16 int8 = **1,536** |
| arithmetic | **32,768 MACs** (65,536 FLOPs) |
| array MAC slots issued | 90,112 — **36% useful** |
| for scale, one BERT-base block | 4.03 G MACs, 7.1 M parameters |

So this is **1/122,880 of a BERT-base block**. It demonstrates that the
instruction set expresses a Transformer and that the hardware computes it
correctly; it is not a throughput result and none of the timings below should be
read as one. The 36% figure is the cost of the design's own conventions —
`MSKIP` bubbles on plain matmuls, and the three identity passes softmax needs.

**The two paths are alternatives, not stages.** Both lower the *same* SPMW
design; you pick one, you do not run both.

| | role path | whole-array path |
|---|---|---|
| C++ → RTL | 12 `csynth` runs, one per role, concurrent | 1 `csynth` run on one big function |
| RTL → array | Vivado assembles exported IPs | — (already one function) |
| what was done with it | implemented standalone behind a harness | **taken to xclbin and run on the card** |

**The board result came from the whole-array path.** The role path's 32.2 min is
a separate measurement of what implementing that fabric costs; it was *not*
deployed, because deploying it needs the AXI shell packaged as an RTL kernel,
which is generated (`shell.feeder_cpp`) but not taken through `package_xo`. Do
not add 32.2 min to 4 h 31 m — they are two answers to two different questions.

**`csynth_design` and `synth_design` are different tools.** Every FPGA flow runs
both, in this order:

```
C++  --csynth_design (Vitis HLS)-->  RTL  --synth_design (Vivado)-->  netlist
     --place_design-->  --route_design-->  bitstream
```

`csynth_design` is *high-level* synthesis: C++ to Verilog. `synth_design` is
*logic* synthesis: Verilog to LUTs, flip-flops and DSPs. The comparison that is
like-for-like is csynth against csynth:

| | role path | whole-array path |
|---|---|---|
| `csynth_design` | **47.0 s** (12 concurrent) | **2445.9 s** (1 process) |

The two `synth_design` numbers are *not* comparable. The role path's 514 s is
the fabric alone; inside `v++` the equivalent step is 52.3 min of block-level
synthesis covering 149 jobs — the kernel *and* the whole platform, HBM
controllers and AXI crossbars included.

**Where the 4 h 31 m went**, decomposed:

| | elapsed | what it is |
|---|---|---|
| Allo lowering + HLS codegen | 195.3 s | SPMW → one HLS C++ program |
| `v++` HLS | 2445.9 s | C++ → RTL (`csynth_design`) |
| `v++ system_link` | 29 s | kernel metadata into the platform |
| `vpl`: platform + BD setup | ~200 s | |
| `vpl`: block-level synthesis | **52.3 min** | 149 jobs, kernel and platform |
| `vpl`: implementation | **2 h 39 m** | opt, place, route, bitstream |
| `rtdgen` / `xclbinutil` / package | 13 s | |
| **total** | **4 h 31 m** | |

Implementation is 59% of it, HLS 15%. Neither scales with the role count, which
is the limit of what the role decomposition buys once a design is real.

### Performance, and why it is what it is

| | |
|---|---|
| peak | 256 cells × 150 MHz = **38.4 GMAC/s** (76.8 GOPS int8) |
| kernel latency | **9,172 cycles = 61.2 µs** per invocation |
| the block, compute only | 11 × 61.2 µs = **673 µs** → 48.7 MMAC/s, **0.13% of peak** |
| the block, wall clock | **45 s** → 0.73 KMAC/s |

Four losses, multiplicative, largest first.

**1. Host round-trip — 66,903×.** 45 s of wall clock against 673 µs of compute.
Every invocation is a fresh `./top xclbin` process that re-loads and re-programs
a 58 MB bitstream. This is not the accelerator; it is `subprocess` plus PCIe
configuration. Nothing about it should be quoted as a hardware number.

**2. The vector unit runs at II=35 — 17.4×.** `csynth` puts 8,961 of the
kernel's 9,172 cycles in one loop: the VPU's 256 instruction executions, at an
initiation interval of 35 instead of 1.

The cause is one opcode. `RECIP` compiles to `vpu_r0_0_sdiv_32ns_32ns_32_36_1`
— a **36-cycle signed divider** — and the register file is indexed at *runtime*
(`reg[dst]`), so HLS must assume a loop-carried dependency through it. The
divider lands in that recurrence, and every instruction pays for it: a NOP costs
35 cycles because one arm of the dispatch can divide.

Measured, by rebuilding the same unit with that one divide replaced by a shift:

| | iteration latency | II | loop cycles |
|---|---|---|---|
| with `RECIP` | 37 | **35** | 8,961 |
| divide → shift | 5 | **2** | **514** |

This is the fp32-accumulator lesson from §3.1 again — *the interval is set by the
operator in the recurrence* — but with a divider, so 36 cycles instead of 7.

**3. The program starves the array — 8×.** A lane executes 16 instructions per
output while the matrix unit takes 2 steps. Even at II=1 the MXU waits 8:1. The
VPU's program length, not the array, is the throughput limit.

**4. Slot utilisation — 36%.** `MSKIP` bubbles and softmax's three identity
passes, as above.

**What the fixes are worth:**

| | block time | rate | of peak |
|---|---|---|---|
| as built | 673 µs | 48.7 MMAC/s | 0.13% |
| divider out of the recurrence | **53 µs** | 616 MMAC/s | 1.6% |
| …and batched in one process | ~53 µs + one load | — | — |

The divider fix is the cheap one and does not need new hardware: hoist the
reciprocal out of the per-element loop (the row sum is constant across a pass,
so it can be computed once and passed in as the lane's second constant, which
the `b` port already carries), or give `RECIP` a path outside the register-file
recurrence.

**Why even 1.6% is the honest ceiling here.** With d = n = 16, one matmul is
4,096 MACs — sixteen cycles of work for a 256-cell array — against roughly 32
cycles of systolic fill and drain. A 16-deep problem cannot fill a 16-deep
array; the design is latency-bound, not throughput-bound, and no amount of
tuning changes that. Efficiency needs the sequence dimension to be much longer
than the array, which is what a real TPU arranges and what §E1 (32×32 and
beyond) would have to measure.

### Removing the host round-trip

The 4.1 s per step was never the accelerator. `allo.backend.hls` runs the host
as a *fresh subprocess per call*, so every invocation re-loaded a 58 MB bitstream
and reprogrammed the device. `scripts/spmw_board_run.py` opens the device once
and reuses it, which is what a host would actually do:

| | block | per step | rate | of peak |
|---|---|---|---|---|
| a process per step | 45 s | 4.1 s | 0.73 KMAC/s | 0.000002% |
| **xclbin loaded once** | **2.200 ms** | 0.200 ms | 14.9 MMAC/s | 0.039% |
| steady state (steps 3–11) | 1.29 ms | **0.117 ms** | 25.5 MMAC/s | 0.066% |
| kernel compute alone (`csynth`) | 0.673 ms | 0.061 ms | 48.7 MMAC/s | 0.127% |

**20,500×**, and none of it hardware. Device open, xclbin load and programming
is **0.50 s, paid once**. All 11 steps still match.

The gap between 0.117 ms observed and 0.061 ms of kernel is the per-call cost
that remains: six buffer syncs, an enqueue and a PCIe round trip. That is what
batching the whole block into one invocation would remove.

### Why the block is not one invocation

It could be, and it is the right thing to want — but it needs one piece of a TPU
this design does not have.

*What is easy.* The weight file is `NW = 4` tiles and the block uses nine
distinct matrices (Wq, Wk, Wv, Wo, W1, W2, the identity, plus the computed Qᵀ
and V). Raising `NW` is a parameter, paid in storage per cell.

*What is not.* **Step k's activations are step k−1's outputs.** Today the VPU's
results leave the fabric and the host hands them back as the next `A`. Doing
that on-chip means routing `V.y_out` back to `P.a_in` — a link from the vector
unit to the matrix unit, which closes a **cycle** in a dataflow graph that is
feed-forward by construction. HLS dataflow will not schedule a cycle, and SPMW's
elaborator models the connection as a stream, which cannot break one.

A TPU solves this with a **unified buffer**: results are written to on-chip
SRAM and the next instruction reads them back. That is a memory between the
units, not a channel — and it is also what the two remaining host-side
operations need, the requantise-and-clip (trivial, a VPU instruction) and the
transpose of P (a different read order out of that same buffer).

So "batch the instructions" is not an instruction-set limitation. The ISA
already sequences a whole block. What is missing is the **memory the sequence
would flow through**, and adding it is the one substantial piece of TPU
architecture still absent here.

*What it would be worth.* Of the 1.29 ms steady-state block, 0.673 ms is kernel
and ~0.62 ms is per-call host cost. One invocation removes the latter — about
**1.9×**. Getting the divider out of the VPU's recurrence is worth **12×** and
is a program change, not a hardware one. That is the order to do them in.

### Getting the divider out of the recurrence

The `RECIP` opcode is gone; `LOADR` replaces it. The divisor was always a
*per-lane constant* — the row sum, fixed across a pass — so the divide never
belonged inside a per-element loop. It now happens once in the lane's prologue
and `LOADR` reads the result.

Measured on the same unit, same build:

| VPU lane | iteration latency | II | loop cycles | lane total |
|---|---|---|---|---|
| `RECIP` in the dispatch | 37 | **35** | 8,961 | 9,005 |
| reciprocal in the prologue | 5 | **3** | 769 | **840** |

**10.7× on the lane**, and the kernel was 98% lane. It is an ISA change, not a
hardware one: nothing was added, one opcode moved from the loop to the prologue.

The general lesson is the one from §3.1 for a third time — *the interval is set
by the operator in the recurrence* — with the sharper corollary that **an
operator only has to be reachable to cost you**. Not one program in the block
executed `RECIP` more than once per output, and NOPs still paid 35 cycles for it.

### Could this run BERT-base?

**The mechanism, yes** — `test_the_array_computes_a_matmul_four_times_its_size`
checks it: a 16×16 array computes a 64-deep, 64-wide matmul by holding four
reduction tiles in the cell weight file, summing them with four `ACCZ`, and
passing four times over the output columns. Tiling is a program.

**Usefully, no.** With one BERT-base encoder block at n=512, d=768, f=3072 and
the engine rebuilt with `steps=2048`, `outs=512`:

| op | invocations | MACs |
|---|---|---|
| Q, K, V | 3 × 576 | 906 M |
| QKᵀ, P·V | 384 + 384 | 403 M |
| output projection | 576 | 302 M |
| FFN1, FFN2 | 2 × 2304 | 2416 M |
| **per block** | **7,680** | **4.03 G** |

| | |
|---|---|
| per block | 1.26 s compute + 0.43 s host = **1.69 s** |
| one sequence, 12 blocks | **~20 s** |
| utilisation | 8.3% of the array's 38.4 GMAC/s |

A GPU does the same forward pass in a few milliseconds, so this is roughly
10⁴ times slower. Two reasons, and only one of them is interesting.

**The array is small.** 256 cells against a TPUv1's 65,536. On this part the
DSP ceiling is 9,024, so 96×96 = 9,216 cells is the largest square that fits —
36× more, and 1.38 TMAC/s. The same arithmetic would be 35 ms per pass at peak,
~0.44 s at the same 8%. *The role count does not change with the array*, so
SPMW's front end is indifferent to this; place and route is not.

**The vector unit starves the matrix unit, 12:1.** 16 instructions at II=3 is 48
cycles per output while the MXU takes 4 steps. Shortening a plain requantise to
seven instructions would take it to 5:1 and utilisation to ~19%.

Three things are structural rather than programmable and would need a rebuild:
`steps` and `outs` (loop bounds), `NW` (the weight file, which caps how much
reduction depth one load covers), and the array's own size. None is a
limitation of the instruction set — the ISA sequences all of it — and that is
the honest summary: **this is a mechanism demonstrator, and the mechanism
scales; the instance does not.**

### Why the interval is 3 and not 1

Vitis names the recurrence exactly:

```
[HLS 200-880] Unable to enforce a carried dependence constraint
  (II = 1, distance = 1) between 'store' of variable 'reg[3]'
  and 'load' on local variable 'reg_3_3'
[HLS 200-1470] Final II = 3, Depth = 5
```

The register file is indexed at *runtime*, so iteration i+1's read may alias
iteration i's write. That makes the whole ALU a distance-1 recurrence, and the
interval is whatever the slowest operator in it takes. With `RECIP` gone the
slowest is `MUL` — Vitis binds it to `mul_32s_32s_32_2_1`, a **two-cycle**
multiplier, and the operand widening makes it a 64-bit product.

Measured, by rebuilding the same unit with that one multiply replaced by an add:

| VPU lane | II | loop cycles |
|---|---|---|
| with `MUL` | 3 | 769 |
| multiply → add | **1** | **259** |

So II=1 is reachable, and the price is stated: **every operator the dispatch can
reach has to finish in one cycle.** The multiply does not need to be 32×32 — in
the softmax normalise the operands are an exponential ≤ 2^8 and a reciprocal
≤ 2^14 — so narrowing `MUL` to 16×16 would buy II=1 at the cost of a documented
range limit on one opcode. That is a real ISA decision rather than a free win,
which is why it is recorded here rather than taken.

The general shape, now seen three times: **an operator only has to be reachable
from the dispatch to set the interval for every instruction.** A divider cost 35,
a multiplier costs 3, an adder would cost 1.

### Floorplanning, and what the clock was actually limited by

`shell.floorplan_xdc` writes the floorplan from the grid, which is the one thing
a regular design should not have to be told: the emitter already knows the mesh
is 16×16 and which instance is at which site. It makes two decisions —

* keep the whole array inside **one SLR**, because every net in a systolic array
  is neighbour-to-neighbour and an SLR crossing turns one of those into a trip
  through Laguna;
* give each mesh **row its own pblock**, stacked in logical order, so a partial
  sum travelling south travels south on the die.

16 pblocks, all 256 cells, generated.

**But first, a correction to the 255 MHz.** The critical path of the routed
16×16 array was:

```
Slack (MET) 0.085ns
  Source:      lfsr_reg[19]_rep__0/C
  Destination: dut/u_mac_r4_5_0/.../v0_read_reg[19]/D
  Data Path Delay: 3.957ns  (logic 0.080ns 2.0%, route 3.877ns 98.0%)
```

The source is `lfsr_reg` — the **harness**, not the fabric. One shared LFSR fed
every channel of every family, so it fanned out to hundreds of loads across the
die and became the longest path in the design. 98% of it was route.

That measurement was of the test harness. `harness_sv` now gives every channel
its own small LFSR, so each source sits beside its load, and the floorplan above
constrains the array. What that combination reaches at a 3.333 ns target is the
run this section will report; the honest statement until it lands is that
**255 MHz was a harness number and the fabric's own paths were never measured.**

### 300 MHz, and where the clock went instead

Two changes — a source per channel in the harness, and the generated floorplan —
taken to place and route at a **3.333 ns** target:

| stage | 250 MHz, no floorplan | 300 MHz + floorplan |
|---|---|---|
| synth | 514.2 | 580.3 |
| opt | 45.6 | 47.7 |
| place | 709.7 | **937.1** |
| phys_opt | 27.4 | 30.3 |
| route | 636.9 | **1172.1** |
| total | 1933.8 s (32.2 min) | 2767.5 s (**46.1 min**, +43%) |

| | |
|---|---|
| achieved | **3.204 ns = 312 MHz**, WNS +0.129 |
| unrouted | **0** |

The constrained placement costs 43% more implementation time, which is the
honest price of the floorplan.

**And the critical path moved where it should be:**

| | source → destination | logic / route |
|---|---|---|
| before | `lfsr_reg` → a MAC cell | 0.080 / **3.877 ns** |
| after | `lshr_ln1_reg` → `reg_3_3` *inside a VPU lane* | — |

It is now the vector unit's own ALU recurrence — a shift feeding the register
file — which is the same dependency that sets the II. The array is no longer
limited by anything outside itself, and the next clock improvement is the same
work as the next II improvement: narrow what the dispatch can reach.

**Attribution is still open.** Two things changed at once, and the old critical
path *was* the shared LFSR, so the harness fix alone may account for all of it.
A control with per-channel sources and no floorplan is running; until it lands,
the honest claim is 312 MHz for the pair, not for the floorplan.

### The reciprocal fix, end to end

The whole-array kernel, rebuilt with the reciprocal in the prologue:

| | cycles | at 150 MHz | block (11 invocations) |
|---|---|---|---|
| `RECIP` in the dispatch | 9,172 | 61.2 µs | 673 µs |
| reciprocal hoisted | **1,137** | **7.58 µs** | **83.4 µs** |

**8.07×** on the kernel, and the vector unit inside it went from 9,005 cycles to
327. Arithmetic throughput goes from 48.7 to **393 MMAC/s**, 0.13% → **1.02%**
of peak.

At that point the per-invocation host cost (~56 µs measured) is seven times the
kernel again, which is the argument for the unified buffer.

### The floorplan did not help — the harness fix did

The control settles the attribution, and against the hypothesis:

| | impl time | achieved | | route |
|---|---|---|---|---|
| shared LFSR, no floorplan | 1934 s | 3.915 ns | 255 MHz | 636.9 s |
| per-channel sources **+ floorplan** | 2768 s | 3.204 ns | 312 MHz | 1172.1 s |
| per-channel sources, **no floorplan** | **2314 s** | **3.147 ns** | **318 MHz** | **805.4 s** |

The floorplan cost **20% more implementation time and 6 MHz**. All of the
255 → 318 MHz came from giving each channel its own source; none of it came from
placing the array by hand. Area is a wash (109,590 vs 109,832 LUT).

Why, in hindsight: a systolic array's netlist *already* expresses the locality a
floorplan would impose — every net is neighbour-to-neighbour, and Vivado's placer
reads that directly. Row-per-clock-region pblocks then only take options away.
The generated floorplan is kept (`shell.floorplan_xdc`) because it is the right
tool for a design that *does* need pinning — one spanning SLRs, or sharing a die
with a shell that has already taken the good regions — but on this design, on
this part, at this size, it is a pessimisation.

Both runs end with the critical path inside a VPU lane, `lshr…_reg` → `reg_…`:
the vector unit's ALU recurrence, the same dependency that sets the II. Once the
harness stopped being the longest path, the array became limited by its own
arithmetic, which is where a design should be.

### On the board with the reciprocal hoisted

Rebuilt and measured, one process, device opened once. **11/11 steps match.**

| step | ms | | step | ms |
|---|---|---|---|---|
| 1 proj0 | 0.173 | | 7 softmax | 0.066 |
| 2 proj1 | 0.097 | | 8 attn | 0.071 |
| 3 proj2 | 0.069 | | 9 proj_out | 0.068 |
| 4 scores | 0.068 | | 10 ffn1 | 0.068 |
| 5 row_max | 0.068 | | 11 ffn2 | 0.067 |
| 6 row_sum | 0.067 | | **block** | **0.883** |

| | block | steady/step | rate | of peak |
|---|---|---|---|---|
| a process per step, `RECIP` | 45 s | — | 0.73 KMAC/s | 0.000002% |
| one process, `RECIP` | 2.200 ms | 0.117 ms | 14.9 MMAC/s | 0.039% |
| one process, **reciprocal hoisted** | **0.883 ms** | **0.0675 ms** | **37.1 MMAC/s** | **0.097%** |

**2.5× on the measured block**, on top of the 20,500× from removing the
per-invocation process. Total, from where this started: **51,000×**, none of it
from a faster array.

And the composition has inverted. A steady step is 67.5 µs, of which the kernel
is **7.58 µs — 11%**. The other 89% is six buffer syncs, an enqueue and a PCIe
round trip. Speeding up the hardware further would now move a ninth of the
number; the remaining work is all in how the host talks to it, which is what the
unified buffer in the section above is for.

### How long a BERT-base block would take

Not measured end to end — that needs a rebuild with `steps=2048, outs=512` and a
four-hour `v++` run. But the two dominant terms *are* measured, so this is one
extrapolation rather than three.

**Measured inputs.** The units at BERT tile shape (512 rows, four reduction
tiles, an eight-instruction program):

| | cycles | at 150 MHz |
|---|---|---|
| MXU cell, `steps=2048` | 2,061 | 13.7 µs |
| VPU lane, `outs=512`, `vprog_len=8` | **12,344** | **82.3 µs** |

VPU-bound by 6:1, as at the small size. And the host cost per invocation,
measured on the board: **59.9 µs** fixed, plus 16 µs to move 193 KB.

**The arithmetic.** One invocation covers 512 output rows × 16 output columns ×
64 reduction depth = 524,288 MACs. A BERT-base encoder block is 4.03 G MACs, so
**7,680 invocations**.

| | |
|---|---|
| kernel | 82.3 µs |
| host | 76.4 µs |
| **per invocation** | **158.7 µs** |
| **per block** | **1.22 s** |
| **12 blocks, one 512-token sequence** | **~15 s** |

Utilisation is 16.6% while the kernel runs and 8.6% overall — the difference
being that the host is on the critical path for half of every invocation.

**Against the floor.** 4.03 G MACs at 256 MACs/cycle and 150 MHz is **105 ms**
per block if the array never idled; 1.26 s for twelve. So the gap to a perfect
16×16 array is 11.6×, and the gap to a GPU (a few ms for the whole pass) is
about 10⁴ — almost all of which is that 256 cells is a small array.

What the two known fixes are worth, on the same arithmetic:

| | per block | 12 blocks |
|---|---|---|
| as it stands | 1.22 s | 14.6 s |
| II=1 (narrow `MUL` to 16×16) | 0.80 s | 9.6 s |
| …and one invocation per matmul (unified buffer) | 0.29 s | 3.4 s |

Neither closes the real gap, which is the array: 96×96 is the U280's DSP ceiling
and would be 36× the cells. **The honest ceiling for this design on this part is
around 0.1 s per sequence, and that needs both software fixes and a 36× bigger
array.**

### The shape belongs in the instruction stream, not the netlist

A TPU is fixed silicon that takes shapes as instructions, and this design failed
that test: `steps`, `outs` and the program length were loop bounds in the unit
bodies, so a different shape meant a different netlist. Saying "BERT needs a
rebuild with `steps=2048`" was describing a design flaw, not a property of the
approach.

The instruction stream is now self-describing. Its first word is a header — the
MXU's says how many instructions follow, the VPU's carries
`[outputs:16 | program length:16]` — and every shape-bearing loop is bounded by
what was read rather than by what was compiled in. Read off the generated C++:

| unit | loop | bound |
|---|---|---|
| MXU | unpack the weight vector | `4` (= `NW`) |
| MXU | **the step loop** | **`v12`, runtime** |
| VPU | zero the instruction buffer | `16` (= `NPROG`) |
| VPU | zero the register file | `4` (= `REGS`) |
| VPU | unpack the bias | `2` (= `NB`) |
| VPU | **load the program** | **`v19`, runtime** |
| VPU | **drain the buffer padding** | **`v28`, runtime** |
| VPU | **the output loop** | **`v43`, runtime** |
| VPU | **execute the program** | **`v46`, runtime** |

Every remaining literal is the size of something physically there.
`test_the_shape_is_data_and_the_buffers_are_hardware` asserts exactly that —
the set of literal loop bounds equals `{NW, NB, NPROG, REGS}` and nothing else —
so the split cannot quietly rot back.

That is the same division a real machine makes: the systolic array's size, the
register file, the instruction buffer and the weight file are hardware; the
matmul's shape is an instruction field.

**Correcting the BERT estimate.** The rebuild with `steps=2048, outs=512` was
never necessary. The deployed engine can run a BERT-shaped block *today* by
tiling it into more invocations, exactly as
`test_the_array_computes_a_matmul_four_times_its_size` does at 4× the array. The
bigger buffer is an **optimisation** — larger tiles mean fewer invocations and so
less host overhead — not a precondition. What the ~1.2 s/block figure measures
is that optimised configuration; the same block on the engine that is on the
board now would take more invocations of the same total arithmetic, so more host
overhead and a somewhat worse number.

**What is still hardware, and honestly so:** the array is 16×16, a lane has four
registers, its buffer holds sixteen instructions, and a cell holds four weights.
A program that wants more of any of those needs a new netlist — as it would on
any machine.
