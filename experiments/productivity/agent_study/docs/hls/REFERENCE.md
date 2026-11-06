# Vitis HLS reference

Vitis HLS compiles C++ into a hardware block. You describe the computation as
functions and loops, and pragmas tell the tool how to turn them into hardware:
which loops become pipelines, which run concurrently, and how arrays are stored.

    #include <hls_stream.h>
    #include <ap_int.h>

## Execution model

By default a function's statements run in sequence. Two things change that.

**Pipelining** overlaps successive iterations of one loop. A loop with
`#pragma HLS pipeline II=1` starts a new iteration every cycle, so a loop of
`n` iterations finishes in about `n` cycles plus the depth of its body, rather
than `n` times that depth. The tool reports the initiation interval it actually
achieved, which may be worse than you asked for if an iteration depends on a
value the previous one produced too late.

**Dataflow** runs several functions concurrently. Inside a region marked
`#pragma HLS dataflow`, each called function becomes its own process, and they
run at the same time, communicating only through the channels between them.
This is how you build an array of communicating elements: write the element as
a function, call it once per position inside a dataflow region, and connect the
calls with streams.

    void top(...) {
    #pragma HLS dataflow
      hls::stream<ap_int<8> > link;
    #pragma HLS stream variable=link depth=2
      producer(in, link);
      consumer(link, out);
    }

Within a dataflow region a variable written by one process and read by another
must be a stream, and each stream must have exactly one writer and one reader.

## Streams

`hls::stream<T>` is a blocking channel.

| Call | Meaning |
|---|---|
| `s.read()` | take the next value, blocking until one is there |
| `s.write(v)` | append a value, blocking until there is room |
| `s.empty()`, `s.full()` | non-blocking status |

Its depth defaults to 2 and is set with `#pragma HLS stream variable=s depth=n`.
A stream that is a function parameter becomes a handshaked port on the generated
hardware, with `dout`, `empty_n` and `read` signals for an input and `din`,
`full_n` and `write` for an output.

## Types

`ap_int<N>` and `ap_uint<N>` are integers of exactly `N` bits, and arithmetic on
them keeps the width you assign into. Use them rather than `int` when the width
matters: `ap_int<32> p = a * b;` with `ap_int<8>` operands keeps the full
product.

## Arrays and partitioning

An array becomes a memory, and a memory has a limited number of ports, so a
loop that reads four elements of the same array in one cycle cannot be
pipelined at one iteration per cycle. `#pragma HLS array_partition variable=A
complete` splits an array into individual registers, removing the limit at the
cost of area. `dim=1` or `dim=2` partitions one axis of a two-dimensional array.
This pragma is usually what stands between a design and the initiation interval
it wants.

## Interface pragmas

`#pragma HLS interface ap_ctrl_none port=return` removes the start and done
handshake, so the block runs continuously from reset and processes inputs as
they arrive. Without it the block waits to be started once per invocation.

`#pragma HLS bind_op variable=v op=mul impl=dsp` puts a multiplication in a
digital signal processing block rather than in logic.

## Reading the report

Synthesis prints, per function and per loop, the latency in cycles, the
initiation interval, and estimated lookup tables, registers and DSP blocks. A
loop that reports no initiation interval was not pipelined. A loop whose
interval is worse than you asked for reports the dependence that caused it.

## Getting one process per position

Inside a dataflow region, a loop that calls a function repeatedly becomes
separate processes only when the tool can unroll it completely and prove each
call touches different channels. Two reliable forms:

    for (int s = 0; s < NS; s++) {
    #pragma HLS unroll
      stage(s, link[s], link[s + 1]);
    }

or writing the calls out. If neither holds, the tool reports that the region
was not split and the calls run in sequence instead, which is the single most
common reason a design that looks parallel is slow.

An array of streams, `hls::stream<T> link[N];`, is the usual way to name the
channels between positions. Declare it inside the dataflow region.

## Why an initiation interval comes out worse than you asked for

The report names the reason; these are the frequent ones.

| Cause | What it looks like | What removes it |
|---|---|---|
| memory ports | a loop reads two elements of one array per iteration | `array_partition` on that array |
| loop-carried dependence | an iteration reads what the previous wrote | restructure, or `#pragma HLS dependence variable=x inter false` when you know the accesses do not overlap |
| a long operation in the recurrence | an accumulator whose adder is too slow | narrow the type, or split the accumulation |
| a blocking read that may not have data | a stream read inside a conditional | read unconditionally where the protocol allows |

## Other pragmas worth knowing

`#pragma HLS inline` folds a function into its caller, removing its interface
overhead; `#pragma HLS inline off` keeps it separate, which is what you want for
a function you are instantiating many times. `#pragma HLS latency min=n max=m`
constrains a region. `#pragma HLS aggregate` packs a struct into one wide word.
`#pragma HLS array_reshape` widens an array's elements instead of splitting it
into registers, which suits memories read in fixed groups.

## Numeric care

Assigning a product to a narrower type truncates silently. `ap_int<8> a, b;
ap_int<32> c = a * b;` is safe because the operands promote, but
`ap_int<16> t = a * b; ap_int<32> c = t;` has already lost the top bits.
Accumulators should be declared at the width the final sum needs.

## Worked example: a four-stage scaling chain

Not related to your task. It shows a dataflow region, one function instantiated
per position, streams between them, and a pipelined inner loop.

    #include <hls_stream.h>
    #include <ap_int.h>

    #define NS 4          // stages
    #define NT 8          // tokens per launch

    static void stage(int k, hls::stream<ap_int<32> > &x_in,
                             hls::stream<ap_int<32> > &x_out) {
      for (int t = 0; t < NT; t++) {
    #pragma HLS pipeline II=1
        x_out.write(x_in.read() * k);
      }
    }

    void line_engine(hls::stream<ap_int<32> > &X, hls::stream<ap_int<32> > &Y) {
    #pragma HLS interface ap_ctrl_none port=return
    #pragma HLS dataflow
      hls::stream<ap_int<32> > link[NS - 1];
    #pragma HLS stream variable=link depth=2
      stage(2, X, link[0]);
      stage(3, link[0], link[1]);
      stage(4, link[1], link[2]);
      stage(5, link[2], Y);
    }

Each call becomes its own process, so the four stages run concurrently and a
token moves one stage per cycle once the chain is full. Note that the calls are
written out: a loop calling `stage` inside a dataflow region is only unrolled
into separate processes when its trip count is a compile-time constant and the
tool can prove each call touches different streams, so writing the calls
explicitly, or using `#pragma HLS unroll` on the calling loop, is the reliable
way to get one process per position.

## Building

    build

runs synthesis, exports the generated hardware, and simulates it against the
visible vectors. It prints the tool's own errors if synthesis fails, then the
number of wrong values and the cycle counts.
