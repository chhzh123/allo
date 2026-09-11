# Fixing the HP-FFT baseline on Vitis 2023.2 / xcu280

E2 reported HP-FFT at 91.1% / 85.9% / 8.0% of its ideal interval `FFT_NUM/(2*UF)`
at UF1 / UF2 / UF4, with the UF4 row collapsing to an interval of 400 cycles
against an ideal of 32. This directory holds the investigation of that collapse
and the pragma changes that remove it. UF8, not measured before, turned out to
have a second and unrelated defect.

Everything below is read out of the per-module **Interval** column of
`csynth.rpt` (Vitis 2023.2, `xcu280-fsvh2892-2L-e`, 3.333 ns) or out of the
cosimulation, never inferred from a cycle count.

## What was actually wrong

The top-level number is a dataflow region's interval, which is the interval of
its **slowest process**. Reading the per-loop column rather than the top line
shows that at UF4 *every loop meets its `target_ti` of 32* -- the Performance
Pragma Report says `TI met: yes` for all of them -- while the enclosing module
`reverse_input_stream_UF4` reports interval 423 and `Pipelined: no`.

Its five loop nests run back to back:

| loop nest in `reverse_input_stream_UF4` | latency |
|---|---:|
| `Pipeline_2 / Loop 1` (trip 8, II=4) | 33 |
| `READ_STREAM_INPUT` | 34 |
| `FROM_BLOCK_TO_CYCLIC` | 35 |
| `STREAM_OUT_REVERSE` | 34 |
| `Loop 1` (trip 8, not pipelined) | 279 |
| sum + handshake overhead | **423** |

Two of those -- 33 and 279, i.e. **312 of the 423 cycles** -- are not FFT work.
They are the `complex<float>` default constructor initialising the two
non-`static` scratch arrays `data_rev_stream[UF*2][FFT_NUM/(UF*2)]` and
`data_in_cyclic[...]` on every call.

**The N=256 UF4 source is the only one of the four that pays this.** UF1 and UF8
declare the same two arrays `static`; UF2 does not have them at all, because its
reverse stage is already split into three process functions
(`reverse_read_stream_input`, `reverse_from_block_to_cyclic`,
`reverse_write_stream_output`) that take the buffers as parameters -- the
canonical dataflow form, which is also what the commented-out lines in the UF1
source sketch. UF4 is the outlier: one monolithic function, automatic arrays, no
dataflow pragma.

## The two fixes

### UF4: `static` scratch arrays, then pipeline the reverse stage as a function

```c
 void reverse_input_stream_UF4 (...) {
+    #pragma HLS pipeline II=FFT_NUM/(2*UF)
-    complex<float> data_rev_stream[UF*2][FFT_NUM/(UF*2)];
-    complex<float> data_in_cyclic[UF*2][FFT_NUM/(UF*2)];
+    static complex<float> data_rev_stream[UF*2][FFT_NUM/(UF*2)];
+    static complex<float> data_in_cyclic[UF*2][FFT_NUM/(UF*2)];
     ...
     READ_STREAM_INPUT: for (...) {
-        #pragma HLS pipeline          // and the same on the other two loops:
                                       // function pipelining unrolls them, so
                                       // the per-loop pragmas are superseded
```

`static` alone takes the FFT_TOP interval from **424 to 109**. The remaining 108
is the three real loops still running in sequence. Pipelining the whole function
at II = beats-per-transform takes it to **52**, with
`reverse_input_stream_UF4` reporting interval **32 = exactly its trip count**
and latency 60.

That is the same structure upstream's own build has. On Vitis **2024.2** /
`xcvp1802` the shipped UF4 reports `reverse_input_stream_UF4` at latency 61,
interval 32, `Pipelined: yes` -- 2024.2 infers the function pipeline by itself
and 2023.2 does not, so the pragma just asks for it explicitly.

### UF8: one wrong cyclic partition factor

```c
-    #pragma HLS array_partition variable=data_2 type=cyclic factor=UF*2 dim=1
+    #pragma HLS array_partition variable=data_2 type=cyclic factor=UF*4 dim=1
```

At UF8 the reverse stage is already fine (2023.2 function-pipelines it on its
own, interval 16). The bottleneck is `FFT_stage_spatial_unroll_5_s` at interval
66 with its inner loop at **II=3**, and the tool names the cause:

> `[HLS 200-885] The II Violation in module 'FFT_stage_spatial_unroll_5_s' ...
> due to limited memory ports (II = 1). Please consider using a memory core with
> more ports or partitioning the array`

on `data_2`. This is the banking conflict the butterfly's `i` / `i + (1 << s)`
access pair creates. `data_2` is read by stage 5 (stride 16) and written by
stage 4 (stride 8); at `factor=UF*2` = 16 banks the stride-16 read pair lands in
the same bank, and at `factor=UF` = 8 banks the stride-8 write pair does.
`factor=UF*4` = 32 banks satisfies both. FFT_TOP goes **67 to 36**.

`bind_storage type=RAM_T2P` -- the tool's other suggestion -- changes nothing.

## Results, N=256

FFT_TOP interval from `csynth.rpt`:

| UF | ideal | before | after | change |
|---|---:|---:|---:|---|
| 1 | 128 | 147 | 147 | none found |
| 2 | 64 | 84 | 84 | none found |
| 4 | 32 | **424** | **52** | `static` + function pipeline |
| 8 | 16 | **67** | **36** | `data_2` factor `UF*2` -> `UF*4` |

Steady interval from the **cosimulation**, 32 transforms of numpy stimulus,
`min` and `max` equal in every row (dead flat, no jitter):

| UF | ideal | before | after | of ideal, before -> after | speedup |
|---|---:|---:|---:|---:|---:|
| 1 | 128 | 140.5 | 140.5 | 91.1% | 1.00x |
| 2 | 64 | 74.5 | 74.5 | 85.9% | 1.00x |
| 4 | 32 | **400.0** | **42.0** | 8.0% -> **76.2%** | **9.52x** |
| 8 | 16 | **57.0** | **26.0** | 28.1% -> **61.5%** | **2.19x** |

Latency falls with it: UF4 first output 782 -> 377 and full transform 813 ->
408; UF8 284 -> 254 and 299 -> 269.

The RTL result is checked against `numpy.fft.fft` on every one of the 32
transforms, and the max absolute error is **identical before and after** --
`5.372450323953043e-06` for all four configurations. The fix changed the
schedule and nothing else.

Cost: UF4 keeps its 258 DSPs, spends FF 85,651 -> 104,962 and saves LUT 92,237
-> 78,012. UF8 goes DSP 423 -> 486, FF 165,953 -> 180,336, LUT 143,535 ->
156,983. (csynth estimates; the fixed designs were not put through
place-and-route, so the resource columns in `results.csv` are left blank for
them rather than filled in with pre-P&R numbers.)

## The same defect at N=1024

`n1024/UF4` has the same bug spelled differently: nine function-scope arrays
(`data_in_cyclic` plus `data_rev_stream_0..7`), all non-`static`, in a
monolithic reverse stage. Each pays a 129-cycle constructor loop, and the shipped
build reports FFT_TOP interval **1576 against an ideal of 128 -- 8.1%**, the same
signature as N=256 UF4. So this is a property of the shipped UF4 sources rather
than a one-off in one file, and `mkvariant.py`'s `static` edit covers both: it
promotes every `complex<float>` array declared at function scope in the reverse
stage, and deliberately leaves the loop-body scratch arrays (`block_data`,
`cyclic_data`) automatic, since those are rewritten every iteration.

## Why none of them reach ideal, and why that is not a pragma problem

Every butterfly stage is a separate non-pipelined process in the dataflow
region, containing one pipelined loop at II=1. Such a process cannot overlap its
own successive invocations, so its interval is its latency:

> **stage interval = trip count + iteration latency**

The fit is exact at every unroll factor, with the iteration latency being the
float butterfly's pipeline depth and essentially constant:

| UF | trip | iteration latency | stage interval |
|---:|---:|---:|---:|
| 1 | 128 | 18 | 146 |
| 2 | 64 | 18 | 82 |
| 4 | 32 | 18 | 50 |
| 8 | 16 | 19 | 35 |

The trip count *is* the ideal interval, so the shortfall is a fixed ~19 cycles
whatever the width -- which is why the percentage of ideal necessarily falls as
UF rises, from 87% at UF1 to 44% at UF8. After the fixes all four unroll factors
sit within one cycle of that floor. **The baseline is now as fast as this
structure permits.**

Three routes out of the floor were tried and all failed:

- **Over-unrolling** the butterfly (`unroll factor=UF*2`) halves the trip count
  but the loop II rises 1 -> 2 in exact compensation: interval stays 146 at UF1
  and 52 -> 53 at UF4. Scaling the partition factor with the unroll
  (`cyclic factor=UF*2`) does not change that.
- **Dropping the `cyclic factor=UF dim=2` partition** on `data_in_cyclic` -- the
  original prime suspect -- changes nothing at all (109 either way).
- **Removing the `bind_op impl=fabric` bindings** makes it worse: the butterfly's
  iteration latency rises 7 -> 9.

Reaching ideal needs the stage's address mapping changed so that `i` and
`i + (1 << s)` never share a bank at any stage -- an XOR swizzle -- which is an
algorithmic change to the butterfly indexing, outside the scope of this fix.
Recorded here as a finding rather than made.

## Reproducing

`mkvariant.py` derives each variant from a baseline source by named pragma
edits, and every edit asserts that it matched, so a variant that silently failed
to apply is impossible:

```
python3 mkvariant.py <src_dir> <dst_dir> static fnpipe
python3 mkvariant.py <src_dir> <dst_dir> "repart:data_2:cyclic factor=UF*4"
```

`testbench_check.cpp` is the correctness gate: it runs three transforms through
`FFT_TOP` and compares against a double-precision DFT, returning non-zero if the
max absolute error exceeds 2e-3. Every configuration in the table above passes
it at 1.3293e-05. The shipped testbench only prints its output and never
asserts, so "csim passed" meant nothing before.
