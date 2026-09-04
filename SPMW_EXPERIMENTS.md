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

## Floorplanning, and why it does not help this array

AutoBridge [FPGA'21] and RapidStream [FPGA'22] argue that an HLS design misses
its clock because the compiler cannot see how far a wire will go, and that the
fix is to floorplan the dataflow graph **and** pipeline the connections that
cross the floorplan's boundaries. Either half alone is worthless. An earlier
floorplan here did only the first half and measured *worse* than no floorplan --
0.085 ns of slack against 0.186 ns -- which is exactly the ablation those papers
argue against. So this is the coupled experiment.

### What a spatial program already knows

Most of both tools' machinery recovers structure a spatial program never lost.
AutoBridge builds a dataflow graph out of C++, estimates each function's area
from HLS reports, and solves an ILP to partition it; then, because "each vertex
is an FSM and the firing rate is not fixed and can have complex pattern", it
adopts a conservative approach -- an SDC over cut-sets to rebalance reconvergent
paths, and a refusal to pipeline anything on a dependency cycle. The paper names
SDF and latency-insensitive theory as the models that would make the problem
easy and says it cannot assume them.

A `Topology` is that model, so the same questions are answered by reading the
declaration:

| the question | AutoBridge | here |
|---|---|---|
| what is the graph? | recovered from C++ | `Topology` is the graph |
| how big is each vertex? | HLS area *estimates* | few roles, each synthesised once, exact |
| where may latency be added? | SDC over cut-sets | a family *is* a cut-set |
| which nets cross? | ILP, then per net | `offset` says the direction, the index says where |
| is there a cycle? | conservative refusal | checked: the fabric is a DAG |

`crossing_families` returns 48 of the 256 southward channels on the 16×16 engine
-- those landing on rows 4, 8 and 12, the three band boundaries. Anchoring the
whole family instead would have built 768 registers to do the work of 48.

### The measurement

16×16 TPU, `xcu280`, 3.333 ns target, LFSR harness, identical in every other
respect:

| configuration | achieved | WNS | frequency |
|---|---|---|---|
| no floorplan, no anchors | 3.071 ns | +0.262 ns | **325.6 MHz** |
| 4 slots + anchors on the crossings | 3.134 ns | +0.199 ns | **319.1 MHz** |

**The coupled version is 2% slower.** Not a wash in favour, a small loss.

### Why, and it is not subtle

The critical path is not a link between units. It is inside one:

```
base      dut/u_vpu_r0_12/u/grp_..._Pipeline_VITIS_LOOP_84_6_.../lshr_ln1_reg_1295_reg[1]
       -> dut/u_vpu_r0_12/u/grp_..._Pipeline_VITIS_LOOP_84_6_.../reg_0_3_fu_126_reg[29]
          2.825 ns: logic 0.691 (24%), route 2.134 (76%)
```

Source and destination are the same instance. It is the lane's shift-and-write
back into a register file indexed at runtime -- the same `reg[dst]` that made the
initiation interval 35 before it was cut to 3. Three quarters of the delay is
route, *inside a unit*, because a runtime index is a mux over the whole file.

Anchoring the southward links pipelines wires that had slack to spare. The
floorplan then takes placement freedom away from the one module that needed it,
and the path moves to `g_vpu_z_in_bind[15].u/count_reg -> u_vpu_r1_15` -- the
accumulator FIFO into the last lane, still not a mesh link.

**AutoBridge's premise does not hold here.** Its gains come from designs whose
critical paths are long inter-module wires, especially ones crossing dies: 43
configurations, 147 MHz to 297 MHz. This array is 272 instances, 8% of the
device, and sits inside one SLR with every link joining physical neighbours. The
placer was already doing the thing a floorplan would tell it to do.

That is worth stating as a limit rather than filed as a failure. The mechanism
is cheaper here than in either paper -- no ILP, no SDC, no area estimates, and
the anchors are transparent by construction (the eleven-step block replay passes
11/11 on the anchored netlist with cycle counts identical to the un-anchored
one). It would pay on an array too large for one SLR, which at this cell size
means roughly 96×96, the U280's DSP ceiling. It does not pay on this one, and
the reason is legible in one timing path.

**What would actually help** is the thing both critical paths point at: the
runtime-indexed register file inside the lane. That is an ISA decision, not a
placement one.

### And the floorplan cannot go on the card at all

Two independent reasons, and they point the same way.

The array's clock is set inside a unit, so pblocks cannot help it — that is the
measurement above. Separately, a Vitis platform puts the kernel inside a
*reconfigurable partition*, and a user pblock inside one is an overlap as far as
`HD.RECONFIGURABLE` DRC is concerned:

```
ERROR: [DRC HDPR-66] Reconfigurable Pblock must not overlap other pblocks.:
  HD.RECONFIGURABLE Pblock 'pblock_dynamic_region' and Pblock
  'level0_i_ulp_spmw_kernel_1_inst_pb_mac_slot0' overlap.
```

The fix is one line — `set_property PARENT pblock_dynamic_region` makes the
pblock a child rather than an intruder, and `floorplan_xdc` now emits it when
given a `parent` — but it buys a design that measured 2% slower standalone and
whose critical path a floorplan cannot reach. So the deployed kernel keeps the
**anchors** (verified transparent: 11/11 on the block replay with identical
cycle counts) and drops the **pblocks**, which is `--no-pblocks`.

That is the whole AutoBridge result for this design, stated plainly: the cheap
half is free and does nothing here, the expensive half is both harmful and
awkward, and the thing actually limiting the clock is an ISA decision inside the
vector lane.

## The role path on the card

The role path had never been deployed. `SPMW_EXPERIMENTS.md` said why: "deploying
it needs the AXI shell packaged as an RTL kernel, which is generated
(`shell.feeder_cpp`) but not taken through `package_xo`". `allo/spmw/kernel.py`
does that now.

| | whole-array kernel (deployed before) | role-path kernel |
|---|---|---|
| LUT | 348,840 (26.8%) | **109,931 (8.4%)** |
| FF | 433,189 (16.6%) | **113,988 (4.4%)** |
| DSP | — | 304 (3.4%) |
| kernel clock closed at | 150 MHz | targeted 250 MHz |

**3.2× fewer LUTs for the same design**, which is the strongest form of the
argument the role decomposition has been making all along: it is not merely
faster to compile, it is a smaller and better netlist.

### Five things that are not in any error message

Each of these cost a build, so they are written down.

* Vitis names a **one-element** array of streams `out_r`, not `out_0`. Every
  other size counts from zero.
* `-m_axi_max_widen_bitwidth 512` is a ceiling, not a promise. A feeder whose
  index is computed (`src[t * channels + c]`) does not burst, so the master is
  32 bits wide however wide the request; the kernel's ports have to say what was
  built, so the width is read back from the netlist.
* Every AXI interface must appear in the clock's `ASSOCIATED_BUSIF`. Without it
  the system linker stops with `Invalid option value '' specified for 'objects'`,
  which mentions neither clocks nor interfaces. The real message only appears
  once a memory bank is named explicitly: *Could not identify a clock source pin*.
* A scoped XDC added by a path outside the IP is referenced as
  `../floorplan.xdc` and is simply **not shipped in the `.xo`**, surfacing four
  steps later as "failed to deliver one or more file(s)".
* `package_xo` refuses to overwrite an existing `.xo` and needs `-force`. Without
  it the stale one links, and the symptom reads exactly like the packager
  silently dropping metadata.

### Two bugs the fabric's own cosim could not have found

The kernel synthesised, linked, loaded and then hung. On a card that is the
whole diagnosis: pyxrt exposes no way to read a kernel's control map, so
"did not finish" is all there is. `allo.spmw.kernel.kernel_testbench` puts the
same kernel against `dram.ram_module`'s behavioural AXI slaves, where a stall
can be counted rather than guessed at.

**Every feeder ran twice.** The kernel's `ap_start` has to stay high until the
*slowest* feeder is done, and an `ap_ctrl_hs` IP takes the job again the moment
it frees up. Gating on `ap_done` does not help -- the IP accepts on `ap_ready`
and asserts both in the same cycle. The probe made it obvious:

```
mac_w_mem      written=2  read=1      (the family has one step)
vpu_op_in_bind written=21 read=17
```

The array survived being over-fed for a while and then stalled somewhere
downstream, with the extra tokens the only evidence. The start is a one-shot
now: raised by a rising edge on `ap_start`, dropped when that feeder accepts.

**The edge FIFOs were four deep, and a systolic array is skewed.** A feeder is
one sequential loop -- every channel's step-*t* token before any channel's
*t+1*. The last row of a 16x16 mesh is fifteen steps behind the first, because
its partial sums have to come down through fourteen cells. So the last channel's
FIFO filled, the feeder blocked on it, and the first row starved waiting for a
token the feeder could not deliver until the last row moved, which it could not,
because it was waiting on the first:

```
edge_a15  written=4  read=0     <- full, nobody reading
edge_a0   written=5  read=5     <- consumed, wants more, feeder is stuck
```

Every other FIFO empty, the whole array halted, and nothing anywhere saying why.

**Neither could have been caught by the cosim that passes 11/11.** That
testbench gives every channel its own driver, so a lagging channel never blocks
a leading one, and the fabric has no `ap_ctrl_hs` in it at all. The bugs live
exactly in the part the cosim replaces with an idealisation. Sizing each edge
FIFO to its family's own step count -- what the design was elaborated with --
lets a feeder lay down a whole pass and the array take it in its own order.

### The shape is data, checked three ways on one netlist

With both fixed, the same kernel runs three tile sizes, each byte-exact against
a reference computed by an engine *elaborated at that tile*:

| tile | control-map polls to completion | result |
|---|---|---|
| `outs=16` | 734 | 0 of 1024 bytes wrong |
| `outs=8` | 402 | 0 of 512 bytes wrong |
| `outs=4` | 325 | 0 of 256 bytes wrong |

The poll counts scale with the tile, which is the claim rather than a restatement
of it: a smaller tile is genuinely less work on the same hardware.

One accident along the way is worth keeping. An early version of the bench wrote
the *elaborated* counts for every tile, so a `outs=8` run was fed 32 activations
instead of 16 -- and the array still produced its eight correct results. The
shape really does come from the instruction stream; only the feeders, told to
move more than the array wanted, failed to finish.

## The role path on the card, measured

`xilinx_u280_gen3x16_xdma_1_202211_1`, kernel clock 250 MHz, one CU.

| stage | elapsed |
|---|---|
| HLS, 12 roles + 6 feeders, concurrent | 184 s |
| `package_xo` | 26 s |
| `v++ -l` (synth 26 m, place 50 m, route, bitstream) | **10,464 s = 2 h 54 m** |
| xclbin | 52.4 MB |

Timing closed: WNS **+0.016 ns**, TNS 0, WHS +0.006, THS 0.

### Resources

| | whole-array kernel (deployed previously) | **role-path kernel** |
|---|---|---|
| LUT | 348,840 (26.8%) | **109,931 (8.4%)** |
| FF | 433,189 (16.6%) | **113,988 (4.4%)** |
| DSP | — | 304 (3.4%) |
| BRAM | — | 3.5 |
| kernel clock closed | 150 MHz | **250 MHz** |

**3.2× fewer LUTs and 1.67× the clock**, for the same design expressed the same
way. That is the strongest form of the argument the role decomposition has been
making: it is not only faster to compile, it is a better netlist.

### The shape is data, on the card

Three tile sizes, one `.xclbin`, nothing reprogrammed between them, each checked
byte-for-byte against a reference computed by an engine elaborated at *that*
tile:

| tile | counts written to the control map | result | fresh call | queued | restarted |
|---|---|---|---|---|---|
| `outs=16` | 32,33,1,17,16,1 | **matches** | 59.7 µs | 44.8 µs | **17.3 µs** |
| `outs=8` | 16,17,1,17,8,1 | **matches** | 58.6 µs | 44.5 µs | **17.1 µs** |
| `outs=4` | 8,9,1,17,4,1 | **matches** | 59.2 µs | 45.6 µs | **17.5 µs** |

A different tiling is six integers, not a new bitstream.

### BERT-base, and why it is slow

One layer is 931,135,488 MACs, which at 16×16×16 per invocation is **227,328
invocations**. All of them were run:

| | per layer | 12 layers | rate |
|---|---|---|---|
| **measured on the card** | **4.261 s** | **51.1 s** | 0.219 GMAC/s (0.34% of peak) |
| the array alone, dispatch removed | 1.333 s | 16.0 s | 0.699 GMAC/s (1.1% of peak) |

Two losses, and they are different in kind.

**1. XRT dispatch — 2.9×.** The three tiles above cost the *same* wall time
despite doing four times the arithmetic, which settles where the time goes
before any profiling: it is not compute. Simulation gives the array's own cost:

| tile | kernel cycles | at 250 MHz |
|---|---|---|
| `outs=16` | 1,466 | 5.86 µs |
| `outs=8` | 802 | 3.21 µs |
| `outs=4` | 649 | 2.60 µs |

Against 17.3 µs measured, **11.4 µs per invocation is XRT** — command
submission and completion. Reusing one `xrt::run` and rewriting only the two
pointers that move takes a fresh call's 59.7 µs down to 17.3 µs; the rest needs
the host out of the loop entirely, which means a kernel that walks a descriptor
list rather than one tile per start.

**2. The tile is too small — 90×.** Even with dispatch free, 1,466 cycles for
4,096 MACs is 2.8 MACs per cycle out of 256. The invocation moves 3,844 bytes
to do 4,096 multiply-accumulates: roughly **a byte per MAC**, which no array
can hide. The largest single transfer is not the weights (1,024 B) but the
*instruction stream* (2,112 B), because the MXU program is replicated across
all sixteen rows in memory even though every row runs the same thing.

Neither is a property of the spatial description; both are the shell around it.
A broadcast loader for the program, and a descriptor list so one start covers
many tiles, are the two changes the measurement asks for -- in that order.

### LLaMA-7B on the same bitstream

No rebuild, no reprogram: the shape is the counts and the instruction stream.

| | per layer | 32 layers |
|---|---|---|
| 26,038,239,232 MAC/layer → 6,356,992 invocations | | |
| measured (227,328 invocations sampled, 18.6 µs each) | 118.4 s | **63 min** |
| the array alone | 37.3 s | 20 min |

This is a 16×16 array of int8 MACs -- 64 GMAC/s of peak against an A100's
~600 TOPS. The interesting claim was never that it is fast; it is that a 7B
model's layer shape and BERT's run on the same netlist, differing only in what
the host writes into six registers and one instruction stream.

### Getting a pblock to stick inside a Vitis reconfigurable partition

Four builds, and the third is the one worth remembering.

| attempt | mechanism | outcome |
|---|---|---|
| 1 | scoped XDC in the packaged IP | pblocks spanned X0..X7 into `pblock_blp`, the static shell -- DRC overlap |
| 2 | the same, plus `PARENT` in a `catch` | the scoped rename broke the lookup; `catch` hid it |
| 3 | `--vivado.prop ...OPT_DESIGN.TCL.PRE` | **built a perfect xclbin containing zero pblocks** |
| 4 | inject into the platform's `preopt.tcl` | 4 pblocks, 64 cells each, correct parent |

**The third failure is the dangerous one.** `vpl` generates its own
`_vivado_impl_props.tcl` claiming every implementation hook it uses -- INIT,
OPT, PLACE and WRITE_BITSTREAM, pre and post -- and overwrites anything
`--vivado.prop` sets. The build ran three hours, closed timing, passed every
DRC, and was indistinguishable from success. The only thing that caught it was
asking the placed checkpoint directly:

```
open_checkpoint level0_wrapper_placed.dcp
get_pblocks pb_mac_slot*    ->  0
```

A floorplan that silently does not apply produces a *working design*. Nothing
errors, because nothing was asked for.

**Two things had to be true at once.** The pblocks must be six clock-region
columns wide, not eight -- `pblock_dynamic_region` is `X0Y4:X5Y10` and column
X7 belongs to the static shell, which is what the overlap DRC was really
objecting to; parenting was a red herring. And the constraint has to run at the
top level, where the full hierarchy and the platform's own pblocks are visible.

The answer was in the platform's own hook file, which carries a commented-out
workaround for CR-1038346 noting that the XSA "cannot register the parent
PBLOCKS in the EARLY XDC processing stage", and demonstrating the pattern that
does work: `add_cells_to_pblock` with full paths like `level0_i/ulp/SLR1`.

`preopt.tcl` is copied fresh from the platform every time `v++` sets up, so the
injection cannot be prepared in advance -- it has to land after setup and
before `opt_design`, a window synthesis leaves open for about half an hour.
`scripts/build_floorplanned.sh` waits for the file and injects into it.

**Verified in the placed design, not inferred from the build succeeding:**

```
pb_mac_slot0  CLOCKREGION_X0Y4:X5Y4  parent=pblock_dynamic_region  cells=64
pb_mac_slot1  CLOCKREGION_X0Y5:X5Y5  parent=pblock_dynamic_region  cells=64
pb_mac_slot2  CLOCKREGION_X0Y6:X5Y6  parent=pblock_dynamic_region  cells=64
pb_mac_slot3  CLOCKREGION_X0Y7:X5Y7  parent=pblock_dynamic_region  cells=64
```

256 mesh cells, four bands, one SLR.

## The floorplanned array on the card

It builds, it closes, it runs, and it changes nothing.

| | unfloorplanned | **floorplanned** |
|---|---|---|
| pblocks in the placed design | 0 | **4, 64 cells each** |
| `v++ -l` | 10,464 s | 10,383 s |
| xclbin | 52.4 MB | 56.4 MB |
| WNS at 250 MHz | +0.016 ns | **+0.016 ns** |
| TNS | 0 | 0 |
| LUT / FF / DSP | 109,931 / 113,988 / 304 | *identical* -- same netlist |

The resources are identical by construction: a floorplan is an implementation
constraint, not a change to the design. Only placement differs.

**On the board, three tile sizes byte-exact on each bitstream**, and the same
speed:

| invocation cost, three runs of 60,000 | | | |
|---|---|---|---|
| unfloorplanned | 17.44 µs | 17.05 µs | 17.09 µs |
| floorplanned | 17.05 µs | 17.19 µs | 22.79 µs |

A single full-layer run gave 3.829 s floorplanned against 4.261 s
unfloorplanned, which looks like an 11% win and is not one -- the repeat above
shows the spread covers it. The measurement is dispatch-bound at ~17 µs against
the array's own 5.86 µs, so it could not resolve a clock difference even if
there were one, and there is not: both close at exactly +0.016 ns.

BERT-base is therefore **~3.9 s/layer, ~46 s for twelve** either way, and
LLaMA-7B **~108 s/layer** on the same bitstream with nothing reprogrammed.

### What the whole floorplanning exercise showed

Three measurements, and they agree.

1. **Standalone, behind the harness**: 325.6 MHz without, 319.1 MHz with -- a
   2% loss.
2. **The critical path**: inside a VPU lane, `lshr_ln1_reg -> reg_0_3_fu_126`,
   0.691 ns of logic and 2.134 ns of route within one instance. Not a link
   between units, so no floorplan can reach it.
3. **On the card**: no difference at all, because the kernel clock is set by
   the platform and both designs meet it with the same slack.

The mechanism is sound and cheaper than the papers' -- no ILP, no area
estimates, no SDC, and the anchors are transparent (11/11 on the block replay
with identical cycle counts). It is the *design* that does not need it: 272
instances, 8% of the device, one SLR, every link joining physical neighbours.
The placer was already doing what the floorplan would have told it.

AutoBridge's own numbers are for designs that must span dies. This one fits in
a third of one. The honest conclusion is not that the technique fails but that
this array is too small to need it -- and the array that would need it, at the
U280's DSP ceiling of roughly 96x96, is a different experiment.

## 32×32: the floorplan costs a third of the clock

The 16×16 measurements found floorplanning worth −2%, and the explanation was
that the critical path sat *inside* a vector lane where no floorplan could reach
it, in an array that fitted in one SLR with every link joining a neighbour. The
prediction was that a bigger array would change this: at 32×32 the mesh must
cross from SLR1 into SLR2, an unregistered Laguna hop costs 1–2 ns, and the
critical path should move onto a southward link — where anchors would help.

**That prediction was wrong.** Both halves of it.

| 32×32, 3.333 ns target | no floorplan | 8 bands + 224 anchors |
|---|---|---|
| P&R | 9,261 s | 9,945 s |
| achieved | **3.333 ns = 300.0 MHz** | **5.109 ns = 195.7 MHz** |
| WNS | +0.000 (met) | **−1.776 (violated)** |
| LUT | 336,442 (25.8%) | 341,971 (26.2%) |
| router congestion reports | 5 | 1 |

The floorplan costs **35% of the clock** — not 2%, and not in the direction the
congestion numbers suggest. It congests *less* and runs much slower.

### Neither critical path is a link

```
no floorplan   g_mac_p_out_p_in[525].u/count_reg[0]  ->  .../mem_reg_0_1_28_31/RAMA/WE
               3.042 ns   logic 0.217 (7%)   route 2.825 (93%)

floorplan      g_vpu_z_in_bind[15].u/count_reg[1]    ->  g_vpu_z_in_bind[15].u/count_reg[0]/D
               5.087 ns   logic 0.344 (7%)   route 4.743 (93%)
```

Source and destination are **the same FIFO instance** in both. The second is a
FIFO's own occupancy counter — `count_reg[1]` to `count_reg[0]`, two bits of one
small register — taking 4.7 ns of route. That only happens if the FIFO's own
cells were placed far apart.

### A partial floorplan is worse than none

The floorplan constrains the 1,024 mesh cells and nothing else. The design has
**3,136 FIFOs** and 32 vector lanes, all unconstrained. Packing the mesh into
eight thin bands leaves everything else to fill the gaps, and a FIFO whose two
counter bits land in different regions is a 4.7 ns path. Left alone, the placer
keeps each FIFO's cells together; the floorplan takes that freedom away without
replacing it.

So the mechanism is not the one AutoBridge describes at all. Its cost model is
slot-crossings weighted by channel width — it assumes the things that matter are
the declared inter-module channels. Here the channels are fine and the *FIFOs
implementing them* are the critical resource, and they are invisible to a
floorplan written over the placement grid.

**What a floorplan for this design would have to do:** constrain the FIFO
belonging to a link into the same band as the units it joins. That is expressible
— the emitter knows which sites a channel connects — but it is a different
constraint from the one `floorplan_xdc` writes, which is about roles on a grid.

### The array scales well without any of it

16×16 reached 325.6 MHz and 32×32 reaches 300.0 MHz — **4× the instances for 8%
of the clock**, with the placer left alone. The role path's compile time is also
flat in the array's side: 12 roles at both sizes, ~3 minutes of HLS either way.
Only Vivado grows, 46 min to 2 h 34 m.

### The FIFO-aware floorplan is worse still

The 32×32 diagnosis was that a *partial* floorplan is the problem: 1,024 mesh
cells pinned, 3,039 FIFOs and 32 vector lanes left to fill the gaps. So the
floorplan was extended to place every unit and every internal FIFO — the FIFO in
the band of the site that reads it, the chain in the band of the mesh row it is
linked from.

**It is worse.** The hypothesis is refuted.

| 32×32, 3.333 ns target | no floorplan | mesh only | every unit + FIFO |
|---|---|---|---|
| achieved | **3.333 ns / 300.0 MHz** | 5.109 ns / 195.7 MHz | **6.015 ns / 166.3 MHz** |
| WNS | +0.000 | −1.776 | −2.682 |
| P&R | 9,261 s | 9,945 s | 10,155 s |
| LUT | 336,442 | 341,971 | 340,017 |

The last build's reported path ends at `sig_reg`, which is the *harness's*
signature register, so that number is contaminated. Asking the routed
checkpoints for the worst path with both ends inside `dut` removes the harness
and gives the array's own clock:

| | array-internal slack | endpoint |
|---|---|---|
| no floorplan | **0.000** | `g_mac_p_out_p_in[525].u/mem_reg.../RAMA/WE` |
| mesh only | −1.776 | `g_vpu_z_in_bind[15].u/count_reg[0]/D` |
| every unit + FIFO | **−2.264** | `u_vpu_r0_4/u/.../reg_1_3_fu_130_reg[14]/CE` |

Still worst, and the endpoint is the giveaway: it is the vector lane's
register-file writeback — the same intra-unit path that limited the 16×16 array
at 2.825 ns, back again.

### Why constraining more made it worse

A band is one clock-region row by six-to-eight columns: very wide and very thin.
A vector lane is ~2,882 cells and wants to be compact. Packing 1,900 cells into a
1×6 strip smears every unit across the die's width, so the *intra-unit* routing —
which is what the critical path is made of — gets longer, not shorter.

The floorplan is shaped like the mesh's logic (rows of a 2-D grid, flattened to
1-D stripes) rather than like the silicon (a 2-D plane). Constraining by mesh
geometry actively harms units whose critical paths are internal.

**Four measurements, all negative:** −2% at 16×16 standalone, no change at 16×16
on the card, −35% at 32×32 with the mesh floorplanned, −40% with everything
floorplanned. Compile time is worse each time too (9,261 → 9,945 → 10,155 s), so
there is no placement-time win either.

**The one hypothesis left with a mechanism behind it** is that the bands are the
wrong *shape*: 2-D tiles, each holding a square block of the mesh, would keep a
unit's own cells compact instead of smearing them. That is a different floorplan
from the one written here, and it is the last version worth a build.

**And the compile-time win SPMW does have is not floorplanning at all.** It is
the role decomposition: 12 roles synthesised concurrently, flat in the array's
side — ~3 minutes of HLS at 16×16 and at 32×32 alike, while the whole-array route
goes from 807 s to impractical. That result is already measured, large, and does
not depend on placement constraints.

## FFT-256: why the reference is vectorised, in numbers

`fft_stream_of(256, 8)` is the butterfly network at the size the reference is
written for: 8 stages of 128 butterflies, **1,024 units and 3 roles**, elaborated
in 0.1 s and correct against `numpy.fft` to 9.3e-06 in float32.

Each butterfly reaches **II=1** at N=256 exactly as it does at N=8 — same role,
same depth 21, because the role is size-independent. Per butterfly, `csynth` at a
3.333 ns target with BATCH=8:

| | |
|---|---|
| latency / interval | 68 / 69 cycles |
| pipelined loop | Target II = 1, **Final II = 1**, Depth = 21 |
| DSP | 24 |
| LUT | 2,490 |
| FF | 3,742 |

### The fully spatial form does not fit, by 2.7×

| | per butterfly | × 1,024 | U280 | |
|---|---|---|---|---|
| DSP | 24 | 24,576 | 9,024 | **272%** |
| LUT | 2,490 | 2,549,760 | 1,303,680 | **196%** |
| FF | 3,742 | 3,831,808 | 2,607,360 | 147% |

A float32 butterfly is four multiplies and six adds; at 24 DSPs each, a thousand
of them is nearly three times the device. **This is the answer to why the
reference folds.** Fold factor against a U280:

| fold | butterflies | DSP | LUT |
|---|---|---|---|
| 1× | 1,024 | 272% | 196% |
| 2× | 512 | 136% | 98% |
| **4×** | **256** | **68%** | **49%** |
| 8× | 128 | 34% | 24% |
| 32× | 32 | 9% | 6% |

The reference's WIDTH=32 lanes is a 32× fold, which is 9% of the DSPs — room for
the buffers, the twiddle ROM and the shell.

### And that is where XOR banking comes from

A fully spatial FFT has **no bank conflicts to solve**: every butterfly owns its
own links and there is no shared memory. Folding is what introduces one. With
WIDTH lanes and N/WIDTH vectors, the stages where `STRIDE >= WIDTH` have both
butterfly operands landing in the same bank, and the reference's F2 swizzle

```
bank(idx) = (idx & (WIDTH-1)) ^ (((idx >> s) & 1) << (LOG2_WIDTH-1))
```

is what separates them. So XOR banking is not an optimisation on top of the
spatial design -- it is a *consequence* of folding, and folding is forced by the
DSP budget above.

**SPMW cannot express it today.** `spmw.xor_bank` is exported and documented as
"the conflict-free layout for butterfly access sets", and it does nothing:
`Brick.layout` is assigned in `bricks.py` and never read anywhere in the
repository, and `bank_fn` and `stride_bit` appear only in `__slots__` and the
constructor. The whole layout vocabulary -- `banked`, `xor_bank`, `replicate`,
`shared` -- is declarative; the lowering ignores all of it. Making the folded FFT
real means implementing that first.

## Against HP-FFT, 256-point FP32 at 250 MHz

HP-FFT (Wang, Zhang, Wu, Cong — UCLA) generates FFT architectures from HLS and
reports a 256-point radix-2 DIT RN FFT in FP32 at 250 MHz on an AMD Versal
VPK180, Vitis/Vivado 24.2. Its unrolling factor `UF` sweeps the parallelism.

| design | II | latency | DSP | LUT | FF |
|---|---|---|---|---|---|
| HP-FFT, no SP | 2085 | 2084 | 8 | 551 | 844 |
| HP-FFT, SP, UF=1 | 128 | 1239 | 24 | 18,937 | 13,421 |
| HP-FFT, SP, UF=8 | 16 | 240 | 158 | 166,047 | 119,219 |
| **HP-FFT, SP, UF=32** | **4** | **126** | **618** | **655,458** | **401,229** |
| **SPMW, fully spatial** | **1** | ~184 | **24,576** | **2,549,760** | **3,831,808** |

Per butterfly, measured at their clock: latency 68, interval 69, II=1 on the
pipelined loop, depth 21, **24 DSP**, 2,490 LUT, 3,742 FF, 0 BRAM.

**Four times the throughput for forty times the DSPs.** As efficiency:

| | points/cycle | DSP | points/cycle/DSP |
|---|---|---|---|
| HP-FFT UF=32 | 64 | 618 | **0.1036** |
| SPMW spatial | 256 | 24,576 | **0.0104** |

**Ten times worse per DSP.** Three causes, and only the last is SPMW's to fix.

**The device is not the same.** VPK180 is Versal: its DSP58 has hardened FP32,
so a float multiply is one DSP. The U280 is UltraScale+, whose DSP48E2 has no
float support at all — every FP32 multiply is built from integer DSPs and logic.
"Same setting" holds for N, data type and clock, and does not hold for the part.
This is most of the 2.41-vs-24 DSP-per-butterfly gap and it is not an
architectural result.

**Fully spatial is past the knee.** HP-FFT's own sweep shows the shape: UF=1 to
UF=32 buys 32× the II for 26× the DSPs, roughly linear. The spatial design is
UF=128 with every stage resident — 1,024 butterflies against UF=32's 256 — and
it buys 4× for 40×. The last doubling of a parallelism sweep is always the worst
one, and this is two doublings past where HP-FFT stopped.

**37% of the multipliers are multiplying by one.** At N=256 the twiddle for
stage `s`, butterfly `b` is `k = (b mod 2^s) · (128 / 2^s)`, and

| stage | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| trivial twiddles | 128 | 128 | 64 | 32 | 16 | 8 | 4 | 2 |

**382 of 1,024 butterflies** have a twiddle of (1,0) or (0,−1) — a multiply by
one or by −i, needing no multiplier at all. Stages 0 and 1 are *entirely*
trivial. The reference snaps near-zero twiddles to exact zero precisely so HLS
constant-folds them away; this design reads its twiddle from a replicated ROM at
run time, so all 24 DSPs are spent whatever the value.

That last one is the interesting gap, because it is what SPMW's role
specialisation is *for*: the twiddle is a function of the site, known at
elaboration. A butterfly whose twiddle is (1,0) should specialise into a role
with no multiplier, and 37% of the array should cost nothing to multiply with.
It does not today because the twiddle is a memory read rather than a constant.

## HP-FFT rerun on our part: the device was most of the story

The comparison above used HP-FFT's *published* numbers, taken on an AMD Versal
VPK180. That leaves the device as a confound, so HP-FFT was cloned and rerun
here: their sources, their `common.tcl`, their `-unsafe_math_optimizations`, the
same 250 MHz and the same 256-point FP32 configuration. The only change is
`TARGET_PART_NUM`, from `xcvp1802-lsvc4072-3HP-e-S` to `xcu280-fsvh2892-2L-e`.
Vitis 2023.2 here against their 2024.2.

| config | II (U280) | II (VPK180) | DSP (U280) | DSP (VPK180) | LUT (U280) | LUT (VPK180) |
|---|---|---|---|---|---|---|
| no SP | 3045 | 2085 | 48 | 8 | 3,808 | 551 |
| UF1 | 145 | 128 | 72 | 24 | 23,513 | 18,937 |
| UF2 | 82 | 64 | 144 | 44 | 48,960 | 40,551 |
| UF4 | 424 | 32 | 258 | 82 | 94,182 | 86,909 |
| UF8 | 65 | 16 | 423 | 158 | 177,466 | 166,047 |
| UF16 | 57 | 8 | 789 | 310 | 342,060 | 328,225 |
| **UF32** | **53** | **4** | **1,515** | **618** | 672,598 | 655,458 |

**The logic ports; the arithmetic does not.** LUTs and registers land within 3%
of the published Versal numbers — 672,598 against 655,458 at UF=32 — so the
generator's structure is device-independent. DSPs are 2.5× higher, and the
initiation interval is **thirteen times** worse.

**And the parallelism stops paying past UF=8**: II goes 65 → 57 → 53 while DSPs
go 423 → 789 → 1,515. Four times the arithmetic for a 1.2× improvement. Vitis
names the reason:

```
o R_Pair_loop_R_Group_loop  |  II  |  Iteration Latency 24  |  Interval 9
```

A butterfly loop that reaches II=1 on Versal reaches II=9 here. Versal's DSP58
has hardened FP32; the U280's DSP48E2 has none, so every float multiply is
integer DSPs plus logic and the recurrence will not close at rate. The published
II=4 is a property of the part, not of the generator.

### Which reverses the earlier comparison

| on xcu280, 250 MHz | II | points/cycle | DSP | points/cycle/DSP |
|---|---|---|---|---|
| HP-FFT UF=32 | 53 | 4.83 | 1,515 | 0.00319 |
| SPMW spatial | 1 | 256 | 24,576 | **0.01042** |

**3.3× better per DSP on the same part** — where the published-number comparison
made SPMW look 10× worse. The whole of that 33× swing was the device.

Two honest qualifications. The spatial design does not *fit*: 2.55M LUTs against
the U280's 1.30M, where HP-FFT's UF=32 fits in 51.6%. And SPMW's II=1 is
structural — every butterfly's loop is measured at II=1, and one transform per
cycle follows from each stage consuming a transform's worth per cycle — while
HP-FFT's is a measured whole-design interval. A rolled SPMW design that fits is
the like-for-like, and is the next thing to build.

## Rolling the FFT: where a buffer becomes unavoidable

To compare with HP-FFT at UF=32 the design has to be *rolled*: 8 stages of 32
units, each unit handling 128/32 = 4 butterflies per transform, 256 units rather
than 1,024. The question is whether the inter-stage wiring stays static, because
if it does the rolled design needs no memory at all -- SPMW's key-form links
carry it.

Folding is a choice of which bits of the butterfly index become the unit and
which become the sub-step. Searching all of them:

* **Every producing unit sends its 8 output lanes to at most 2 destination
  ports**, at every stage. So two output ports always suffice.
* With `unit = b >> 2`, stages 3–7 are static and stages 1–2 are not. With the
  fold the per-stage search prefers, stages 6–7 are not. **No consistent chain
  of bit-selection folds makes all eight stages static.**
* The obstruction is precise: for 64 of 224 producing units, the two output
  ports want *different* sub-step orders, and one sequential loop cannot emit
  both. Stages 2–6 want the identity; the failures are entirely at the
  boundaries where the stride and the fold width interact.

**Which is the reference's architecture, derived rather than copied.** HP-FFT
splits N=256, WIDTH=32 into "intra stages 0–4, no conflicts" and "inter stages
5–7, 2-D buffers with F2 XOR-swizzle banking". Their fold puts the sub-step in
the *high* bits, so the awkward stages are 5–7; the fold above puts it in the
low bits, so they are 1–2. **The number of stages needing a reorder is
invariant; only which ones moves.** That is what the swizzle is for, and it is
why a rolled FFT cannot be all wires.

Two ways to carry those two boundaries:

* **Extra ports** -- give the unit two inputs per side and select on the
  sub-step. Expressible in SPMW today; costs FIFOs, no memory.
* **A banked buffer** -- what the reference does, and what `xor_bank` now
  describes. Needs a *shared* brick with several readers, which SPMW's memory
  model does not have: `shard` gives one brick per site, and a butterfly's two
  operands are in different banks by construction, so no unit can read both
  from its own.

The first is the buildable one and is the remaining step for a like-for-like
UF=32 comparison. The second is the honest gap: SPMW can now *describe* a
conflict-free banked layout and lower it for a per-site brick, and still cannot
express the shared buffer that makes banking worth having.

### Correcting the device claim: the II gap is banking, not DSP58

The section above attributed HP-FFT's II=53-on-U280 against II=4-on-Versal to the
device. That was too quick. There is no Vitis 2024.2 on this machine — only
2023.2 — so their toolchain cannot be reproduced, and two tool-shaped effects
had to be ruled out first.

**Pragmas are being dropped in 2023.2, and it was worth checking.**

```
WARNING: [HLS 207-5573] Unroll pragma is ignored, because 'factor' is not
                        positive integer  (./FFT.cpp:225:27)
```

The pragma is `#pragma HLS unroll factor=UF>>(stage-1)` inside
`template<int stage> void FFT_stage_spatial_unroll(...)`. `stage` is a template
parameter, so the factor *is* a compile-time constant; 2023.2's pragma parser
will not evaluate the expression. Three `performance` pragmas are dropped too,
for not being in a loop. So the hypothesis was right that pragmas are lost.

**But it is not what limits the design.** Per-stage intervals at UF=32:

| stage | 1 | 2 | 3 | 4 | 5 | 6 | **7** |
|---|---|---|---|---|---|---|---|
| interval | 11 | 11 | 19 | 19 | 19 | 19 | **52** |

The loops carrying the ignored unroll all reach II=1 regardless. One loop sets
the whole design's II=53: stage 7's inner loop, at **interval 9**, the one whose
`performance target_ti=4` Vitis reports as failed.

**And it is a bank conflict that no HLS pragma can express.** `data_ld` and
`data_st` are plain `complex<float>[256]` with no partitioning, so adding some
looked like the obvious fix. It changes nothing — II stays 53, stage 7 stays at
interval 9 — and the arithmetic says why:

```
stage 7: i1 - i0 = bflyStep = 128,  cyclic banks = 32,  128 mod 32 = 0
```

Both butterfly operands land in the **same** cyclic bank, for every butterfly,
at any factor. That is exactly `test_cyclic_banking_collides_on_every_pair` in
`test_spmw_banking.py`, and exactly what the XOR swizzle exists to fix:
`(i & 31) ^ (((i >> 7) & 1) << 4)` separates them because bit 7 is what differs.

So the corrected reading:

* the **2.5× DSP** gap is a device effect — DSP58 has hardened FP32, DSP48E2 does not;
* the **13× II** gap is a **banking** effect, and HP-FFT's HLS C++ has no swizzle
  in it. Their *Allo* version does, through `s.f2_layout` — the module ported
  here. Whether 2024.2 infers a better layout unprompted is untestable on this
  machine, and is the one open question.

The comparison numbers stand as measured. What changes is the attribution, and
it lands on the mechanism this branch has been building: a conflict-free layout
is the thing that makes the last stage of a folded FFT keep rate.

### Settled: the II gap is the tool, the DSP gap is the device

zhang26 has Vitis 2024.2, so HP-FFT could be run in its own toolchain. Three
points, 256-point FP32 at 250 MHz:

| tool | part | UF8 II | UF32 II | UF32 latency | UF32 DSP |
|---|---|---|---|---|---|
| **2024.2** | Versal VP1802 | **16** | **4** | 126 | 618 |
| **2024.2** | UltraScale+ U55C | **16** | **4** | 152 | ~1854 |
| 2023.2 | UltraScale+ U280 | 65 | 53 | 183 | 1515 |

The first row reproduces the published table exactly — II 16 and 4, latency 240
and 126 — so the setup is sound and the paper's numbers replicate.

**The initiation interval is entirely a tool effect.** On UltraScale+ with the
right compiler HP-FFT reaches II=4, the same as on Versal. The 53 measured
earlier is Vitis 2023.2 failing to schedule what 2024.2 schedules; both of the
explanations offered before -- DSP58's hardened FP32, then bank conflicts -- were
wrong. The pragma evidence was real (`unroll factor=UF>>(stage-1)` is dropped in
2023.2) but was not the mechanism either, since that loop reaches II=1 anyway.

**The DSP count is a device effect**, and only that: ~1854 on UltraScale+ against
618 on Versal for the same design and tool, about 3×, which is what a float
multiply costs without DSP58's hardened FP32. Latency moves a little too (152
against 126).

The lesson worth keeping is procedural. Three explanations were offered for one
number, two of them confidently, before the obvious experiment -- run their code
in their toolchain -- was possible. The tool version was in `env.sh` all along.

*(U55C rather than U280: the U280's device files are not in that install. Same
family, same `fsvh2892` package, same `2L` speed grade, same DSP48E2.)*

## §7.2 — the FFT twiddle experiment

The draft asks what constant-folding per-site twiddle indices buys. A butterfly
is four real multiplies and six adds; when the twiddle is (1,0) or (0,−1) the
multiplies vanish, but only if the value is a compile-time constant. The spatial
FFT reads it from a replicated ROM, so it pays for them whatever the value.

Measured per butterfly, `xcu280` at 250 MHz, FP32, batch 64, II=1:

| twiddle | DSP | FF | LUT |
|---|---|---|---|
| from a ROM (what the design does) | **24** | 3,296 | 1,775 |
| constant (1,0) — multiply by one | **8** | 1,554 | 939 |
| constant (0,−1) — multiply by −i | **8** | 1,554 | 939 |
| constant, general | 24 | 3,104 | 1,711 |

Folding a *general* constant saves nothing — the multiplies are still there. The
saving is entirely in the trivial cases, and the eight that remain are the float
adds, which also land on DSPs without hardened FP32.

At N=256 the twiddle for stage `s`, butterfly `b` is `k = (b mod 2^s)(128/2^s)`,
so the trivial ones are:

| stage | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| trivial | 128 | 128 | 64 | 32 | 16 | 8 | 4 | 2 |

**382 of 1,024, or 37%** — stages 0 and 1 entirely. So for the spatial FFT-256:

| | DSP |
|---|---|
| twiddle selection dynamic | **24,576** |
| per-site twiddle constant-folded | **18,464** |
| | **25% fewer** |

**And it costs compile time, which is the interesting part.** Constant-folding
requires the twiddle to differ per role, so roles split by distinct twiddle
value: stage `s` has 2^s of them, and the design goes from **3 roles to 255**.
The saving and the compile-time reuse pull against each other — 25% of the DSPs
for 85× the HLS invocations. Neither number is free, and the design has a real
choice to make rather than a strictly better option.

## §7.3 Synthesis scalability: the 4²–32² sweep

Systolic GEMM on `xcu280`, 300 MHz target (3.333 ns), role path, measured on
brg-zhang-xcel with Vitis 2023.2. Each row is one full run from Python source to
a placed-and-routed array; the stage timings are separated so the flat part and
the growing part can be told apart.

| array | units | FIFOs | roles | elaborate | HLS wall | HLS CPU | Vivado | total | achieved |
|-------|------:|------:|------:|----------:|---------:|--------:|-------:|------:|---------:|
| 4×4   |    16 |    32 |     9 |     1.0 s |   82.5 s | 383.6 s | 130.3 s |  215.7 s | 1.584 ns |
| 8×8   |    64 |   128 |     9 |     1.0 s |   81.5 s | 375.6 s | 148.7 s |  232.5 s | 1.589 ns |
| 16×16 |   256 |   512 |     9 |     1.2 s |   82.5 s | 380.5 s | 249.7 s |  334.8 s | 1.595 ns |
| 32×32 | 1,024 | 2,048 |     9 |     1.7 s |   82.4 s | 383.3 s | 955.3 s | 1,040.7 s | 1.671 ns |
| ratio |   64× |   64× |  1.00× |     1.7× |   1.00× |   1.00× |   7.3× |    4.8× |   +5.5% |

Three readings.

**HLS time is exactly flat.** 82.5 s at 16 instances and 82.4 s at 1,024 — the
0.1 s spread across a 64× range is noise. The same 9 role projects are compiled
regardless of array size, and the CPU figures (383.6 → 383.3 s) confirm it is
the same work, not the same wall clock hiding more parallelism.

**Elaboration is not a bottleneck at any size.** 1.0 → 1.7 s for 64× the
instances. The structural front end walks every instance, so this does grow, but
from a base small enough not to matter.

**Vivado is the whole cost at scale.** 130.3 → 955.3 s, a 7.3× rise for 64× the
units — sublinear, but it moves from 60% to **92% of total wall time**. Any
future compile-time work has to aim there; there is nothing left to win in HLS.

Timing degrades gently and never fails: 1.584 → 1.671 ns achieved against the
3.333 ns target, so even 32×32 closes with +1.662 ns of slack.

The FFT shows the same shape more sharply — 1,024 butterflies compile as 3
roles, elaborated in 0.1 s.

Not run: the no-reuse ablation (one HLS project per instance), which would give
the speedup denominator, and the per-PE / factored-HLS comparison against prior
tools.

## HP-FFT attribution, re-verified

Every cell re-read from the `csynth.rpt` files rather than from notes, because
this comparison had been mis-attributed twice (first to the device, then to
banking). N=256, 250 MHz target.

| tool | part | UF8 lat / II | UF32 lat / II | UF32 DSP |
|------|------|-------------:|--------------:|---------:|
| 2024.2 | Versal VP1802 (`xcvp1802-lsvc4072-3`) | 240 / 16 | 126 / 4 | 618 |
| 2024.2 | UltraScale+ U55C (`xcu55c-fsvh2892-2`) | 276 / 16 | 152 / 4 | 1,854 |
| 2023.2 | UltraScale+ U280 | 307 / 65 | 183 / 53 | 1,515 |

Row 1 reproduces the published table exactly. Holding the tool at 2024.2 and
changing only the device, **II is identical** and latency rises 15–20%. Holding
the device family and dropping to 2023.2, II collapses 13×. So the interval is a
tool effect; the 3× DSP gap is the genuine device effect (DSP58 hardened FP32).

The 2023.2 UF32 point also **missed timing** (slack −0.19 ns), so it was never a
valid design point to quote.

## §7.2–7.5, run against the draft's figure and table specifications

### Fig. 10, SPMW series: square GEMM on the deployed array

The 16×16 int8 MXU bitstream at 250 MHz, U280, one invocation per 16×16×16
tile. Every size was checked byte-exact against the reference simulator before
being timed.

| GEMM | invocations | µs/invocation | GMAC/s | GOP/s | % of the array's peak |
|------|------------:|--------------:|-------:|------:|----------------------:|
| 64³   |          64 | 17.55 | 0.233 | 0.467 | 0.36% |
| 128³  |         512 | 17.18 | 0.238 | 0.477 | 0.37% |
| 256³  |       4,096 | 17.15 | 0.239 | 0.478 | 0.37% |
| 512³  |      32,768 | 17.28 | 0.237 | 0.474 | 0.37% |
| 1024³ |     262,144 | 19.09 | 0.215 | 0.429 | 0.34% |

Throughput is flat in problem size and sits at **0.34–0.37% of the 64 GMAC/s
the array could do**. The reason is visible in the same table: 17 µs per
invocation against 16 cycles -- 64 ns -- of actual array work. The design is
bound by the host round trip, not by the architecture, because the fabric has
no DRAM interface of its own and each tile is a separate kernel invocation over
PCIe. That is the `A_IO_L3`/AXI-master gap E14 already names.

This matters for how Fig. 10 can be read. As deployed, the number measures
Xilinx XRT's command latency; it is not evidence about the quality of the
generated architecture, and quoting it as "% of theoretical peak" would say
almost nothing about SPMW. The architecture-level comparison below is the
apples-to-apples one.

### Fig. 10, hand-written Vitis HLS baseline: the architecture comparison

`examples/spmw/baselines/hls/gemm_systolic.cpp`, the same int8 weight-stationary
16×16 array, synthesised at 300 MHz for `xcu280`:

| | hand-written HLS | SPMW role path |
|---|---:|---:|
| LUT | 114,401 | 109,931 |
| FF | 94,742 | 113,988 |
| DSP | 262 | 304 |
| HLS wall time | 867 s | 82 s |

The areas agree within 4% on LUT, which is the fidelity claim §7.2 wants. The
compile times do not agree at all, and that is the §7.3 claim: one monolithic
HLS run over the whole array costs **10.6× the wall time** of nine role
projects compiled concurrently.

### Fig. 11, no-reuse ablation

Same RTL architecture, same worker limit; the only change is one HLS project per
site instead of one per role.

| array | instances | roles | reuse wall | reuse CPU | no-reuse wall | no-reuse CPU | wall | CPU |
|-------|----------:|------:|-----------:|----------:|--------------:|-------------:|-----:|----:|
| 4×4   |    16 | 9 | 43.4 s |   389 s |    45.8 s |     723 s |  1.1× |  1.9× |
| 8×8   |    64 | 9 | 42.7 s |   383 s |   142.2 s |   3,050 s |  3.3× |  8.0× |
| 16×16 |   256 | 9 | 43.5 s |   388 s |   563.8 s |  13,104 s | 13.0× | 33.8× |

Reuse is worth little at 4×4, where there are 16 instances against 9 roles, and
grows with the array: at 16×16 it saves **33.8× of CPU and 13.0× of wall time**.
The reuse column is flat at ~43 s throughout, which is the same fact §7.3
reports from the other direction.

### Table 4, seeded defects: 43%, not the near-100% a pilot suggested

Fourteen single-line mutations across six bug classes, each design elaborated
unmutated first as a control (all five controls pass).

| Bug class | Seeded | Caught statically |
|---|---:|---:|
| Mismatched channel endpoints | 4 | 1 |
| Off-by-one link arithmetic | 3 | 1 |
| Missing boundary variant | 1 | 0 |
| Type mismatch across a channel | 1 | 1 |
| Unsynchronized shared memory | 2 | 2 |
| Port-capacity violation | 3 | 1 |
| **Total** | **14** | **6 (43%)** |

An earlier 12-mutation pilot scored 12/12, and that was misleading: it only
seeded defects in the classes SPMW checks. The broader set exposes where the
representation is genuinely blind, and the reason is structural rather than an
oversight:

**Dangling channels are silently dropped.** `_resolve_coordinate` and
`_resolve_keyed` both skip any channel with no writer or no readers. They have
to -- at a mesh edge `spmw.key(r, c + 1)` names a key nobody reads, and that is
how an array terminates. The consequence is that **a typo'd link is
indistinguishable from an intentional boundary**, so most endpoint and boundary
mutations pass elaboration. This is not fixable by adding a check; it needs the
boundary to be declared rather than inferred.

**`stationary()` skips its shape check when `index=None`.** `bind_check` is
called only inside `if index is not None`, so binding a D×D tensor to a
D-element port is accepted silently. This one *is* a bug, and it is the same
shape as the recurring one: a check that exists but cannot fire.

**Nothing rejects two bindings of the same memory port.** There is no
duplicate-client check, so a second `shard` onto a bound port is accepted.

What SPMW does catch, it catches well: both unsynchronised-shared-memory
mutations are rejected by `_check_phase_writers`, the type mismatch by the port
checker, and an out-of-range gather by `bind_check`.

### Table 5, SYCL/oneAPI column: the flow no longer exists

`examples/spmw/baselines/sycl/gemm_systolic.cpp` is a complete, idiomatic
oneAPI FPGA implementation -- one kernel per PE, connected by pipes -- but it
cannot be built on the evaluation machine, and not for a fixable reason:

- `icpx -fintelfpga` in oneAPI 2026.1: *"option '-fintelfpga' is not supported
  and has been removed from the compiler"*.
- `sycl/ext/intel/fpga_extensions.hpp` is not present in the install at all.

Intel has discontinued the SYCL FPGA flow, and it targeted Intel devices rather
than an AMD Alveo in any case. The SYCL column of Table 5 can be filled with
line counts; the SYCL series of Fig. 10 cannot be measured on this hardware.

### Table 3, GEMM mesh sweep: 16 shapes, one source design

`scripts/spmw_sweep_table3.py`. Rows and columns vary independently; K is held
at 16 and every point is measured on the same fixed 256³ problem. Area is the
sum over roles of that role's HLS estimate times the sites it covers, which is
what the split backend instantiates -- an HLS estimate, not a post-route number.

| shape | instances | roles | cycles | LUT | FF | DSP | elaborate |
|-------|----------:|------:|-------:|----:|---:|----:|----------:|
| 4×4   |    16 | 9 | 1,572,864 |   3,344 |   1,648 |   16 | 1.34 s |
| 4×8   |    32 | 9 |   917,504 |   6,760 |   3,296 |   32 | 1.19 s |
| 4×16  |    64 | 9 |   589,824 |  13,592 |   6,592 |   64 | 1.00 s |
| 4×32  |   128 | 9 |   425,984 |  27,256 |  13,184 |  128 | 1.05 s |
| 8×4   |    32 | 9 |   917,504 |   6,760 |   3,296 |   32 | 0.94 s |
| 8×8   |    64 | 9 |   524,288 |  13,664 |   6,592 |   64 | 0.95 s |
| 8×16  |   128 | 9 |   327,680 |  27,472 |  13,184 |  128 | 1.03 s |
| 8×32  |   256 | 9 |   229,376 |  55,088 |  26,368 |  256 | 1.10 s |
| 16×4  |    64 | 9 |   589,824 |  13,592 |   6,592 |   64 | 0.97 s |
| 16×8  |   128 | 9 |   327,680 |  27,472 |  13,184 |  128 | 1.11 s |
| 16×16 |   256 | 9 |   196,608 |  55,232 |  26,368 |  256 | 1.11 s |
| 16×32 |   512 | 9 |   131,072 | 110,752 |  52,736 |  512 | 1.36 s |
| 32×4  |   128 | 9 |   425,984 |  27,256 |  13,184 |  128 | 1.16 s |
| 32×8  |   256 | 9 |   229,376 |  55,088 |  26,368 |  256 | 1.15 s |
| 32×16 |   512 | 9 |   131,072 | 110,752 |  52,736 |  512 | 1.41 s |
| 32×32 | 1,024 | 9 |    81,920 | 222,080 | 105,472 | 1024 | 1.79 s |

**Points 16, area span 66.4×, on the frontier 16.**

Two things are worth pulling out. **Every one of the sixteen shapes is
Pareto-optimal** -- no configuration is dominated, because rows and columns
trade latency against area monotonically and the two rectangular orientations
of a shape cost exactly the same. The knob spans a real design space rather
than a handful of useful points and a tail of bad ones.

And **the role count is 9 at every shape**, not just at every size. A 4×4 and a
32×32 mesh have the same nine wiring classes -- interior, four edges, four
corners -- so the HLS cost of exploring this entire 66× span is sixteen runs of
nine roles, and elaboration never exceeds 1.8 s.

### Table 5, complete: lines of code per design

`scripts/spmw_loc.py`. `cloc` is not installed on the machine, so the script
applies cloc's rule directly -- a line counts when it is neither blank nor
wholly a comment -- and only the design is counted, not test harnesses or
reference implementations. The reduction is against the larger of the two
baselines.

| Design | Vitis HLS | SYCL/oneAPI | SPMW | Reduction |
|--------|----------:|------------:|-----:|----------:|
| Systolic GEMM (16×16) | 99 | 119 | 32 | 4.0× |
| Multi-cache GEMM | 117 | 146 | 57 | 2.6× |
| Tiled GEMM (2-level) | 113 | 132 | 33 | 4.0× |
| FFT-1024 (spatial + folded) | 407 | 130 | 127 | 3.2× |
| Mini-TPU MXU | 117 | 132 | 40 | 3.3× |
| Attention-PV (G ∈ {1,2,4}) | 107 | 133 | 55 | 2.4× |

Reductions run **2.4×–4.0×**, which is a good deal smaller than a headline
number picked from the best row would suggest. Three caveats belong with the
table rather than under it.

**The FFT row is not symmetric.** Its Vitis HLS figure is HP-FFT-HLS
`n1024/UF32`, a published, expert-tuned implementation, while its SYCL figure is
a compact folded implementation written here. That is why SYCL (130) and SPMW
(127) come out nearly equal on that row while HLS is 407: the HLS baseline
expands its stages explicitly and the other two do not. Comparing 407 against
127 compares two different amounts of hand-expansion, not two languages.

**The SYCL column cannot be compiled.** oneAPI 2026.1 removed `-fintelfpga` and
ships no FPGA extension headers, so these are complete and idiomatic but
unverified programs. The HLS column is verified: every file synthesises for
`xcu280`.

**The mapping needed a correction.** The weight-stationary kernel with ReLU and
a shift is the mini-TPU MXU, not the plain systolic GEMM; they were conflated in
a first pass, which inflated the MXU's SPMW count from 40 to 143 and made the
row read as a *negative* reduction.

### Fig. 11 complete: the no-reuse ablation through 32×32

| array | instances | roles | reuse wall | reuse CPU | no-reuse wall | no-reuse CPU | wall | CPU |
|-------|----------:|------:|-----------:|----------:|--------------:|-------------:|-----:|----:|
| 4×4   |    16 | 9 | 43.4 s | 389 s |    45.8 s |     723 s |  1.1× |   1.9× |
| 8×8   |    64 | 9 | 42.7 s | 383 s |   142.2 s |   3,050 s |  3.3× |   8.0× |
| 16×16 |   256 | 9 | 43.5 s | 388 s |   563.8 s |  13,104 s | 13.0× |  33.8× |
| 32×32 | 1,024 | 9 | 47.7 s | 422 s | 2,195.2 s |  52,170 s | 46.0× | 123.8× |

At 32×32 reuse is worth **123.8× of CPU and 46.0× of wall clock**: 9 HLS
projects instead of 1,024, for the same RTL architecture under the same worker
limit. The reuse column stays at 43–48 s across the whole range.

### Table 3, attention-PV: the group count, and a design space with one winner

| G | roles | instances | LUT | DSP | cycles (modelled) | elaborate |
|---|------:|----------:|----:|----:|------------------:|----------:|
| 1  | 10 | 272 | 53,008 | 240 | 1,536 | 1.75 s |
| 2  | 10 | 264 | 51,640 | 248 |   768 | 1.84 s |
| 4  | 10 | 260 | 49,708 | 252 |   384 | 1.31 s |
| 8  |  7 | 258 | 46,246 | 254 |   192 | 1.02 s |
| 16 |  4 | 257 | 39,523 | 255 |    96 | 0.60 s |

**Points 5, area span 1.3×, on the frontier 1.**

This sweep behaves unlike the mesh one, and the difference is the interesting
part. Raising G makes the design **both faster and smaller**, so every point
except G=16 is dominated and the frontier has a single member. Smaller because
the periphery shrinks with G: the head dimension d = 16/G sets the width of the
activation unit and the number of seeded and drained edges, so more grouping
means fewer boundary units and more of the work done by the psum chain that
already exists. The role count falls with it, 10 → 4, because at large G the
distinct boundary cases collapse.

The cycle column is **modelled, not measured** -- passes × (steps + rows + cols)
-- so it is exactly 2× per doubling by construction. The measured figure comes
from RTL cosimulation and is reported separately; the draft's 1.94× model
prediction should be checked against that, not against this column.

### Table 3, hierarchical GEMM: hierarchy multiplies roles

A `tiles × tiles` grid of `pe × pe` engines, measured the same way as the mesh.

| configuration | instances | roles | LUT | DSP | cycles (modelled) | elaborate |
|---------------|----------:|------:|----:|----:|------------------:|----------:|
| 2×2 of 2 |    16 |  16 |   3,200 |   16 | 1,572,864 |  1.78 s |
| 2×2 of 4 |    64 |  36 |  13,376 |   64 |   524,288 |  4.22 s |
| 2×2 of 8 |   256 |  36 |  54,656 |  256 |   196,608 |  4.56 s |
| 4×4 of 2 |    64 |  64 |  12,800 |   64 |   524,288 |  7.87 s |
| 4×4 of 4 |   256 | 144 |  53,504 |  256 |   196,608 | 17.88 s |
| 4×4 of 8 | 1,024 | 144 | 218,624 | 1024 |    81,920 | 23.47 s |

**Points 6, area span 68.3×, on the frontier 4.**

The result that matters here is the one that cuts against the reuse claim
elsewhere. **Hierarchy multiplies roles.** A 4×4 grid of 8×8 engines has 1,024
instances and **144 roles**; the flat 32×32 mesh has the same 1,024 instances
and **9**. Elaboration follows: 23.5 s against 1.8 s.

The reason is that a tile's own boundary interacts with the grid's, so a PE's
wiring class is the product of where it sits inside its engine and where its
engine sits in the grid, instead of the sum. Nesting a fabric is free in the
source, but it is not free in the number of distinct kernels the backend has to
compile, and §7.3's flat-mesh numbers should not be read as covering the
hierarchical case.

One limitation of this row: the cycle column is modelled from the array's outer
side length, so it cannot distinguish `2×2 of 4` from `4×4 of 2` -- both have a
side of 8 and both come out at 524,288. The draft's claim of a latency
difference between designs with the same PE count needs a model that captures
the staging between levels, which this one does not.

### Fig. 12, measured: grouping is worth 1.99× at M=4096

The draft's comparison cannot be expressed by varying `G` alone. In the fabric
the head dimension is *derived* -- `d = cols // groups` -- so raising G shrinks
the output width instead of filling idle columns, and the naive sweep measures a
different problem at every point. Run that way, cycles go **up** with G (37, 45,
73 for G = 1, 2, 4) because the serpentine chain lengthens while the array stays
the same size and the work stays constant at 96 MACs.

The comparison the draft intends holds the head dimension fixed and changes how
many columns the array has. With d = 2 throughout: the ungrouped design is a 4×2
array covering 4 rows of the reduction per pass, and the grouped one a 4×4 array
in two slabs covering 8. Both were cosimulated and both agree with the reference
bit for bit.

| M (sequence) | ungrouped, span 4 | grouped G=2, span 8 | speedup |
|--------------|------------------:|--------------------:|--------:|
| 6    |    29 cycles |    45 cycles | 1.29× |
| 64   |   107 cycles |   123 cycles | 1.74× |
| 4096 | 5,483 cycles | 5,499 cycles | **1.994×** |

The speedup is `2 × C_ungrouped / C_grouped`, since the grouped design needs
half the passes. Both configurations have the same slope -- 1.345 cycles per
sequence step -- and differ only in fill: about 21 cycles against 37. That is
the whole story of the figure. At short sequences the deeper serpentine
dominates and grouping is worth only 1.29×; by M=4096 the fill has amortised and
the measured **1.994×** sits within 2.9% of the draft's 1.94× model, slightly
*better* than the model predicts rather than worse.

### Table 3, FFT row, and the completed table

The knob the draft names is "spatial/fold factor". Only the spatial half is
buildable: the folded form needs a brick several butterflies share, which
`shard` cannot express. This sweeps the spatial size instead, which changes area
and throughput without touching the butterfly body.

| N | instances | roles | LUT | DSP | elaborate |
|---|----------:|------:|----:|----:|----------:|
|  8 |  12 | 3 |  28,392 |   288 | 0.48 s |
| 16 |  32 | 3 |  76,704 |   768 | 0.42 s |
| 32 |  80 | 3 | 192,320 | 1,920 | 0.45 s |
| 64 | 192 | 3 | 462,720 | 4,608 | 0.48 s |

**Three roles at every size**, and elaboration flat at ~0.45 s while instances
grow 16×.

#### Table 3 as the draft asks for it

| Design | Structural knob | Points | Area span | Pareto |
|--------|-----------------|-------:|----------:|-------:|
| GEMM mesh | rows × columns | 16 | 66.4× | 16 |
| Hierarchical GEMM | tile and stage size | 6 | 68.3× | 4 |
| FFT | spatial size | 4 | 16.3× | 1 |
| Attention-PV | group count G | 5 | 1.3× | 1 |
| **Total** | | **31** | | **22** |

Area is an HLS estimate summed over roles times sites, and the cycle axis of the
frontier is modelled rather than measured except for attention-PV, which was
cosimulated. Both are labelled that way deliberately: 31 post-route runs would
be the honest version of this table and were not done.

### Baseline verification

All five hand-written HLS baselines synthesise for `xcu280` at 300 MHz:

| baseline | rc | csynth | DSP | LUT |
|----------|---:|-------:|----:|----:|
| systolic GEMM (output-stationary) | 0 | 1,495 s | — | — |
| multi-cache GEMM | 0 | 1,177 s | — | — |
| tiled GEMM | 0 | 1,557 s | — | — |
| mini-TPU MXU | 0 | 867 s | 262 | 114,401 |
| attention-PV | 0 | 1,028 s | 256 | 112,924 |

The attention baseline is worth a note because it first passed in **49 s with 9
DSPs**, which is not a 16×16 array. Its `DATAFLOW` pragma sat inside a bare
block, where Vitis emits `HLS 207-5571` and synthesises the region sequentially
-- a success return for the wrong architecture. Moving the region into its own
function gives the 260 dataflow processes the design wants. A green return code
was not evidence the baseline was the intended design, and neither the line
count nor the reduction in Table 5 would have been wrong -- only the claim that
the baseline had been verified.

### The HLS baseline deadlocked on hardware, and why

The hand-written baseline synthesised cleanly, closed timing, and built a
bitstream in 10,116 s -- and then hung on the card: `ERT_CMD_STATE_TIMEOUT` on
a two-tile invocation. Synthesis success was not evidence the design ran.

Cosimulation named the cause directly:

    WARNING: [HLS 200-656] Deadlocks can occur since process seed_p is
    instantiated in a dataflow region with ap_ctrl_none or without start
    propagation and contains an auto-rewind pipeline.

`seed_p` only *wrote* -- it pushed a zero into the north edge of every column
and read nothing. A process with no input is free-running, Vitis auto-rewinds
its pipeline, and inside a dataflow region that can deadlock. Cosim also asked
for depth 8 on the partial-sum FIFOs feeding the drain, against the 4 they had.

Both are the same underlying mistake: an operand reaching PE (r, c) travels c
hops east while its partial sum travels r hops south, so the array needs
buffering on the order of its own side length, and a process that participates
in that pipeline cannot be free-running.

The fix removes `seed_p` rather than constraining it: row 0 gets its own PE
variant that starts the accumulation at zero, so there is no write-only process
at all, and the stream depths go to 32. Re-synthesis is clean with **zero**
`200-656` warnings.

Correcting the design changes the Figure 10 comparison, so the earlier numbers
are superseded:

| implementation | LUT | FF | DSP |
|---|---:|---:|---:|
| SPMW role path | 109,931 | 113,988 | 304 |
| hand-written HLS (deadlocking version) | 114,401 | 94,742 | 262 |
| **hand-written HLS (corrected)** | **121,563** | **94,515** | **246** |
| AutoSA | 168,680 | 110,895 | 256 |

Against the version that actually runs, SPMW is **9.6% smaller on LUT**, not the
4% the broken one suggested. The direction of the claim is unchanged and the
margin is larger; the point is that the first number was measured on a design
that could not execute.

### Fig. 10 completed: the corrected baseline on the card

The rebuilt bitstream links clean (10,115 s) and **runs**: `correctness on 2
tiles: matches`. The deadlock is gone.

Measured on the U280, both at SPMW's invocation granularity and at the
granularity this baseline can actually use:

| | µs per 16×16×16 tile | GOP/s | % of that design's peak |
|---|---:|---:|---:|
| SPMW role path (250 MHz) | 17.15 | 0.478 | 0.37% |
| hand-written HLS, one tile per launch (300 MHz) | 40.53 | 0.202 | 0.13% |
| hand-written HLS, 1,024 tiles per launch | 2.392 | 3.425 | 2.2% |

Two readings, and they point opposite ways, so both belong in the paper.

**At the same invocation granularity SPMW is 2.4× faster than the hand-written
baseline** -- 17.15 µs against 40.53 µs per tile. That is a like-for-like
comparison: one 16×16×16 tile per kernel launch on both sides.

**Given the batching the baseline can do and SPMW cannot, the baseline is 7.2×
faster** -- 3.425 GOP/s against 0.478. The SPMW kernel takes one tile per
invocation because the fabric has no DRAM interface of its own, so it cannot
amortise the launch over many tiles the way this baseline does. The
host-interface effect measured directly on the baseline is **17×**: 40.53 µs
per launch against 2.392 µs per tile when batched.

Neither design is anywhere near its array's peak, and for different reasons.
SPMW is bound by the PCIe round trip -- 17 µs of launch against 64 ns of array
work. The batched baseline reaches only 2.2% because it reloads all 256
stationary weights per tile: `load_w` is 330 cycles against the tile's own ~16,
so 718 of its 718 cycles per tile are mostly weight traffic. A GEMM that reused
one weight tile across many activation tiles would not pay that, and neither
implementation currently expresses the reuse.

The honest summary for Fig. 10 is that **the throughput axis measures memory
and launch structure, not the generated array**, on both sides. The area and
timing comparison is what carries the fidelity claim.

### Fig. 10, the AutoSA comparison done properly: all three placed and routed

Same part, same 3.333 ns (300 MHz) target, same Vivado flow through
`route_design`.

| design | arithmetic | LUT | FF | DSP | achieved | WNS | 300 MHz |
|--------|-----------|----:|---:|----:|---------:|----:|:-------:|
| SPMW daisy chain | int16 | 140,515 | 147,353 | 256 | 2.942 ns (340 MHz) | +0.391 ns | yes |
| SPMW AutoSA-matched | int8→int32 | **39,471** | **24,112** | 256 | 2.846 ns (351 MHz) | +0.487 ns | yes |
| AutoSA `mm16i8` | int8→int32 | 53,253 | 96,913 | 256 | 2.790 ns (358 MHz) | +0.543 ns | yes |

**The daisy chain closes 300 MHz** with 0.391 ns of margin, which was the
question. At matched arithmetic SPMW uses **26% fewer LUTs and 75% fewer
registers** than AutoSA, with all three at exactly 256 DSPs -- one per PE -- so
nothing is being traded between LUTs and DSPs.

Two corrections to what this file said earlier, both from comparing an HLS
estimate against a post-route number:

**AutoSA does not miss timing.** The −0.25 ns slack quoted before is Vitis HLS's
*estimate* at the 300 MHz target. Placed and routed it makes **+0.543 ns** --
more margin than either SPMW design. Any claim that SPMW closes timing where
AutoSA does not is wrong.

**AutoSA is not 168,680 LUTs.** That is the HLS estimate; post-route it is
**53,253**, 3.2× lower. The earlier "SPMW is 35% smaller than AutoSA" compared a
post-route SPMW number against an HLS-estimated AutoSA one and is not a valid
comparison. The honest figure is the 26% above, measured the same way on both
sides.

Two asymmetries remain, and they pull in opposite directions, so the 26% is a
range rather than a point:

- **AutoSA carries a DRAM interface and SPMW does not.** `A_IO_L3_in` and the
  AXI masters are inside AutoSA's 53,253; the SPMW array ends at its edge
  streams. This inflates AutoSA.
- **SPMW carries a harness and AutoSA does not.** AutoSA's `kernel0` exposes
  1,289 I/O ports -- more than the part has pins -- so `place_design` refused it
  outright (`IO Placement failed due to overutilization`) and it had to be run
  out-of-context, which inserts no I/O buffers and adds no driver logic. The
  SPMW numbers include their per-channel LFSR drivers. This inflates SPMW.

The second was chosen deliberately in AutoSA's favour: if SPMW still comes out
smaller with its own overhead counted and AutoSA's excluded, the direction is
safe even though the magnitude is not exact.

The int16 daisy row is not comparable to AutoSA -- it is the arithmetic of the
Allo reference (`tests/dataflow/test_daisy_chain_gemm.py`), and its 140,515 LUTs
are mostly int16 multipliers and the 256-bit packed column its drain chain
carries. It is here because it answers the frequency question for that design,
not because it belongs in the same row as an int8 array.

### The daisy chain in AutoSA's arithmetic: one variable moved, and it cost 6.5×

`--design daisy8` is `daisy_of(16, operand=int8, accum=int32)`: the same chained
packed-column drain as the int16 daisy, the same int8→int32 as AutoSA and the
AutoSA-matched design. Placed and routed at the same 3.333 ns target:

| design | arithmetic | drain | LUT | FF | DSP | achieved | WNS |
|--------|-----------|-------|----:|---:|----:|---------:|----:|
| SPMW daisy (int8→int32) | int8→int32 | packed column, `int32[16]` = 512 b/token | **257,904** | 277,321 | 256 | 3.327 ns | **+0.006 ns** |
| SPMW AutoSA-matched | int8→int32 | scalars forwarded, reverse row order | 39,471 | 24,112 | 256 | 2.846 ns | +0.487 ns |
| AutoSA `mm16i8` (OOC) | int8→int32 | scalars (`C_drain_IO_L1_out`) | 53,253 | 96,913 | 256 | 2.790 ns | +0.543 ns |

With arithmetic held fixed, the only thing that moved between the first two
rows is how results leave the array, and it moved the LUT count by **6.5×** and
ate all of the timing margin: 300 MHz closes by 6 ps, after a congestion
iteration (`iter_3_CongestedCLBsAndNets.txt`), and Vivado took 4,417 s against
1,053 s. The cost is structural: every one of the 256 PEs carries a 512-bit
token through its drain FIFO, and there are 768 FIFOs. The int16 daisy's 256-bit
token was already the widest thing in the design; doubling the accumulator
doubled it again.

So the answer to "can the daisy chain reach 300 MHz" is yes, barely, and the
answer to "is it a fair AutoSA comparison" is that it is the *wrong structure*
for one: AutoSA does not pass packed columns, and neither should a design that
wants to be compared with it. The AutoSA-matched row is the like-for-like
comparison, and it stands at 26% fewer LUTs and 75% fewer registers.

## GPT-2 medium (arXiv 2312.15159) on SPMW

### The target, and what the reference actually is

The Allo paper's LLM is GPT-2 355M: 24 layers, 16 heads of 64, hidden 1024,
FFN 4096, sequence 128, W8A8, on a U280 at 245 MHz. Its accelerator is three
Vitis kernels, one per SLR, joined by AXI streams: region 1 holds three 8×16
int8 systolic arrays for Q, K and V (two MACs per DSP by packing the weight
into 17 bits); region 2 holds K and V in URAM, an 8×8 attention array and an 8×8
context array per head, float softmax, an 8×16 array for the output projection,
the residual and a float LayerNorm; region 3 holds two more 8×16 arrays for the
FFN with a piecewise-linear GELU between them, the residual and the second
LayerNorm. Its host times a whole layer with OpenCL profiling on the last
kernel's event, single-layer and 24-layer; there is no per-stage timing.

Two facts about the artifact worth knowing before reproducing it. The `llm/`
Makefile builds `Bert_layer_dataflow_region_*` from `bert_region_*.cpp` -- the
GPT kernels are in the directory but nothing builds them -- and `llm/reports/`
is BERT's implementation (437k LUT, 1,776 DSP, 47 URAM across three SLRs), not
GPT's. The GPT numbers in the paper are relative to DFX and an A100; no absolute
latency is tabulated.

Per layer the model is 12·128·1024² + 2·128²·1024 = **1,644,167,168 MACs**. At
the paper's M = 256 MACs/cycle and 245 MHz the ideal is 6.4M cycles, about
26 ms/layer. That MAC rate is exactly one 16×16 array -- the same array the
deployed SPMW engine already is. The reference reaches it by spreading six
smaller arrays over three SLRs so every stage runs concurrently; nothing about
the arithmetic needs more than one array running all the time.

### Baseline: GPT-2 medium tiled on the deployed engine, as it stands

`spmw_board_model.py --model gpt2-medium`, the existing 16×16 MXU + VPU
bitstream at 250 MHz, every tile byte-exact against the reference simulator:

| | |
|---|---:|
| MACs per layer | 1,644,167,168 |
| launches per layer | 401,408 of 16×16×16 |
| per launch | 20.64 µs |
| **per layer** | **8.287 s** |
| **24 layers** | **198.9 s** |
| of the array's 64 GMAC/s | 0.31% |

This is the number to beat, and the reason it is bad is structural rather than
a slow array: a pass can only use the one 16×16 weight tile resident in the
cells, so a K=1024 reduction is 64 separate launches per 16×16 output tile,
each carrying 16 cycles of work behind ~20 µs of PCIe. The array is idle 99.7%
of the time waiting for the host.

### The design: a stage engine, and why it is the efficient one

The reference reaches M = 256 MACs/cycle by spreading six 8×16 arrays (two
MACs per DSP by bit-packing) over three SLRs so that every stage of a layer
runs concurrently. Nothing about the arithmetic needs six arrays; it needs one
array of 256 MACs that never waits. The deployed SPMW engine is that array. What
it lacked was a way to keep it fed: a pass could only use the one weight tile
resident in the cells, so K=1024 was 64 launches per output tile and the array
idled 99.7% of the time behind PCIe.

`tests/dataflow/spmw/test_spmw_gpt_stage.py` is the same MXU and VPU with two
changes that are *instructions*, not structure:

- the cell's weight file is 256 tiles deep -- K=4096 in one file, or four
  16-column output slabs of K=1024 -- and `MSWEEP base count` makes a cell run
  `count` steps selecting tiles `base..base+count-1` in turn, so one program
  word is a whole reduction;
- the lane's `ACCN n` folds `n` partial sums in a tight loop at one per cycle.
  Dispatching one `ACCZ` per psum ran the lane at II=2 and would have made the
  vector unit the bottleneck of a matrix engine.

A launch is therefore 32,768 activation steps producing 512 rows (four slabs of
a K=1024 projection) or 128 rows (one slab of the K=4096 FFN2), on one netlist,
told apart by two numbers in the stream. GPT-2 medium's layer becomes 192 such
launches plus 48 small attention ones, against 401,408 before.

**HLS, 16×16 (Vitis 2023.2, 250 MHz):** 12 roles, 272 instances, 18 units with
the six DMA feeders. The cell's sweep loop is **II=1** over 0–65,535 trips; the
lane's `ACCN` loop is **II=1**; the weight-file load is 256 trips at II=1. The
cells cost 55 s of HLS each and the lanes 45 s, concurrently: 158 s for the
whole kernel's HLS.

Two things the packaging path had to learn on the way. A cell's weight file is
a 2,048-bit token, wider than any scalar the generated DMA feeder knew, so it
is now moved as four 512-bit AXI beats and reassembled -- the host already lays
tokens out as contiguous little-endian bytes, so there is no order to agree on.
And "one whole pass" is no longer a sane edge-FIFO depth: at 32,768 steps it was
half a megabyte of block RAM to cover a sixteen-step skew, so the depth is
capped at 1,024.

Not on the device in this version, and reported as such rather than folded in:
the softmax between the score and context GEMMs, GELU, and the two LayerNorms.
The reference runs those in float on the FPGA. Here the accelerator's number
covers the eight GEMMs, which are 99% of the layer's MACs; the omission is
named in the host's output beside the stages it does run.

### The packaged stage kernel, simulated before it is linked

`spmw_kernel_sim.py` on the `--sim` output, with the per-head score launch --
2,048 activation steps, 512 result rows, a full 256-tile weight file moved as
512-bit beats, and counts smaller than every buffer:

    SPMW TB: done after 16443 poll(s)
    SPMW TB RESULT 0 of 32768 byte(s) wrong

Bit-exact against the reference simulator, in 31 s of xsim over 124 source
files (the kernel, twelve role wrappers, their exported netlists and six
feeders). Two things this checks that no unit test could: the 2,048-bit weight
token survives the beat split and reassembly on both sides of the AXI, and a
feeder that outruns the 1,024-deep edge FIFO -- this launch is 2,048 steps --
stalls and resumes rather than deadlocking, which was the whole argument for
capping the depth.

That is the gate for the link. `v++ -l` at 250 MHz is running.
