# HP-FFT's corrected UF4 and UF8: the w = 8 and 16 HP-FFT rows of Fig. 9(c,d), arXiv version

Built on Vitis HLS 2023.2 for `xcu280-fsvh2892-2L-e` at 3.333 ns, the shipped
UF4 and UF8 sources (UCLA-VAST/HP-FFT-HLS @ `c4611b8`) fall far short of their
ideal interval for two unrelated reasons. Both Fig. 9 panels use the corrected
sources below, at 256 points (`N256/source/fix_UF4_a8`, `fix_UF8_b4`) and with
the same edits at 1,024 points (`../../n1024/hpfft/UF4_a8`, `UF8_b4`). UF1 and
UF2 are as shipped. Intervals below are read from the per-module **Interval**
column of `csynth.rpt` or from the cosimulation, never inferred from a cycle
count.

## UF4: static scratch arrays, and the bit-reversal stage pipelined as a function

A dataflow region's interval is that of its slowest process. At UF4 every loop
meets its target, but `reverse_input_stream_UF4` is not pipelined and its loop
nests run back to back. Two of them are not FFT work: they are the
`complex<float>` default constructor initialising the stage's non-`static`
scratch arrays (`data_rev_stream`, `data_in_cyclic`; at 1,024 points
`data_in_cyclic` and `data_rev_stream_0..7`) on every call. UF1 and UF8 declare
them `static`; UF2 splits the stage into three dataflow processes. The fix:

```c
 void reverse_input_stream_UF4 (...) {
+    #pragma HLS pipeline II=FFT_NUM/(2*UF)
-    complex<float> data_rev_stream[UF*2][FFT_NUM/(UF*2)];
-    complex<float> data_in_cyclic[UF*2][FFT_NUM/(UF*2)];
+    static complex<float> data_rev_stream[UF*2][FFT_NUM/(UF*2)];
+    static complex<float> data_in_cyclic[UF*2][FFT_NUM/(UF*2)];
     ...
-        #pragma HLS pipeline          // on each loop: function pipelining
                                       // unrolls them, so these are superseded
```

`static` removes the constructor loops; the function pipeline overlaps the
three real loops. Upstream's Vitis 2024.2 build infers that function pipeline
by itself; 2023.2 does not, so the pragma asks for it. In
`N256/report/fix_UF4_a8/csynth.rpt`, `reverse_input_stream_UF4` is pipelined,
latency 60, interval 32 (its trip count), and `FFT_TOP` is at interval 52.

## UF8: one partition factor

```c
-    #pragma HLS array_partition variable=data_2 type=cyclic factor=UF*2 dim=1
+    #pragma HLS array_partition variable=data_2 type=cyclic factor=UF*4 dim=1
```

At UF8 the bit-reversal stage is already function-pipelined by 2023.2. The
bottleneck is `FFT_stage_spatial_unroll_5_s`, whose inner loop runs at II=3,
and the tool names the cause on `data_2`:

> `[HLS 200-885] The II Violation in module 'FFT_stage_spatial_unroll_5_s' ...
> due to limited memory ports`

`data_2` is written by stage 4 (stride 8) and read by stage 5 (stride 16). At
`factor=UF*2` (16 banks) the stride-16 read pair shares a bank; at `factor=UF`
(8 banks) the stride-8 write pair does; `factor=UF*4` (32 banks) satisfies
both. In `N256/report/fix_UF8_b4/csynth.rpt`, `FFT_TOP` is at interval 36 and
`reverse_input_stream_UF8` at 16.

## Results at 256 points

| config | ideal | csynth interval | cosimulated interval | of ideal | latency |
|---|---:|---:|---:|---:|---:|
| UF4, fixed | 32 | 52 | 42.0 | 76.2% | 408 |
| UF8, fixed | 16 | 36 | 26.0 | 61.5% | 269 |

From `cosim_events.json`: 32 transforms complete, every completion interval
equal (min = max). `validate_rtl_vcd.json` checks all 32 against
`numpy.fft.fft` and passes, max absolute error 5.37e-6 (1.8e-7 of the largest
output) for both. The edits change the schedule, not the arithmetic.

The variant folders keep only `FFT.cpp`; `FFT.h` and `project.tcl` are the
shipped UF4 and UF8 ones, which `mkvariant.py` copies unchanged.

**Timing.** csynth flags both fixed builds at slack -0.00 ns. Routed, UF4
closes (+0.119 ns) and UF8 misses (-0.093 ns) in `../results.csv`; their routed
reports are not in this folder. At 1,024 points the UF4 fix does not survive
routing: `../../n1024/README.md`.

## Why neither reaches ideal

Every butterfly stage is a separate non-pipelined process holding one II=1
loop, so it cannot overlap its own successive invocations and its interval is
the loop's trip count, `N / (2 UF)` beats, plus the float butterfly's pipeline
depth:

| UF | trip | stage interval | difference | read from |
|---:|---:|---:|---:|---|
| 1 | 128 | 146 | 18 | `N256/report/UF1/csynth.rpt` |
| 4 | 32 | 50 | 18 | `N256/report/fix_UF4_a8/csynth.rpt` |
| 8 | 16 | 35 | 19 | `N256/report/fix_UF8_b4/csynth.rpt` |

The trip count is the ideal interval, so the shortfall is a near-constant and
its share grows with UF. Removing it needs a bank mapping in which `i` and
`i + (1 << s)` never share a bank at any stage, an XOR swizzle: a change to the
algorithm's indexing, not a pragma, and not made here.

## Reproducing

`../scripts/hpfft/mkvariant.py` derives a variant from a source directory by
named edits; each edit asserts that it matched, so a variant cannot silently
build the baseline:

    python3 mkvariant.py <src_dir> <dst_dir> static fnpipe
    python3 mkvariant.py <src_dir> <dst_dir> "repart:data_2:cyclic factor=UF*4"

`static` promotes only the `complex<float>` arrays declared at function scope in
the bit-reversal stage, ahead of its first loop; the loop-body scratch arrays
(`block_data`, `cyclic_data`) are rewritten every iteration and stay automatic.

`../scripts/hpfft/testbench_check.cpp` is a csim gate: three transforms through
`FFT_TOP` against a double-precision DFT, non-zero exit above 2e-3 max absolute
error. The shipped testbench only prints its output. The build, cosimulation
and route scripts are listed in `../README.md`.
