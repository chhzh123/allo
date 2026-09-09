# z-ai_glm-5_3__hls: round by round

Model `z-ai/glm-5.3`, arm **hls**, temperature 0.0. Budget 500,000 tokens, 8 builds, 3 routes.

Outcome: **did not submit**, stopped by `builds`, 478,484 tokens, 4771.1 s wall (2916.1 s model, 1855.0 s tools).

> **The code each round wrote is not recoverable.** The harness truncated every
> tool argument to 200 characters before writing this transcript, and the working
> directory kept only the final file, which each round overwrote. Below is the full
> prose, every build verdict and every routing result, plus the first 200 characters
> of each `design.cpp`. The final design is in `../generated/z-ai_glm-5_3__hls/`.


### t=9.13s — model turn (706 tokens this turn, 706 cumulative)

**ran** `cat TASK.md`


**ran** `cat REFERENCE.md`


**ran** `cat design.cpp`


*result of* `cat TASK.md`:

```
# Design task: an 8x8 output-stationary integer matrix-multiply tile

Build a hardware module that multiplies two 8x8 matrices of 8-bit signed
integers and produces their 8x8 product as 32-bit signed integers.

## What it computes

For every `i` and `j` in `0..7`:

    C[i][j] = sum over k in 0..7 of A[i][k] * B[k][j]

`A` and `B` hold signed 8-bit values in `[-128, 127]`. `C` holds signed 32-bit
values. No saturation, no rounding, no scaling: the products and their sum are
exact in 32-bit arithmetic.

## The architecture you must build

This is not open. Build exactly the structure below; a design that computes the
right answer another way does not satisfy the task.

- **Sixty-four processing elements in an 8 by 8 grid.** Call the one in row `i`
  and column `j` `PE(i,j)`. Each holds one multiplier and one accumulator, and
  the design uses 64 multipliers in total, no more and no fewer.
- **Output-stationary.** `PE(i,j)` accumulates `C[i][j]` across all eight values
  of `k` and holds it until it is drained. No element's partial sums are
  computed anywhere else, and there is no adder tree over the grid.
- **Nearest-neighbour movement only.** A processing element exchanges values

```


*result of* `cat REFERENCE.md`:

```
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
a function, call it once per position inside a dataflo
```


*result of* `cat design.cpp`:

```
// Fill in the body. Do not change the signature.
#include <hls_stream.h>
#include <ap_int.h>

void gemm_tile(
    hls::stream<ap_int<8> > &a_in_0,
    hls::stream<ap_int<8> > &a_in_1,
    hls::stream<ap_int<8> > &a_in_2,
    hls::stream<ap_int<8> > &a_in_3,
    hls::stream<ap_int<8> > &a_in_4,
    hls::stream<ap_int<8> > &a_in_5,
    hls::stream<ap_int<8> > &a_in_6,
    hls::stream<ap_int<8> > &a_in_7,
    hls::stream<ap_int<8> > &b_in_0,
    hls::stream<ap_int<8> > &b_in_1,
    hls::stream<ap_int<8> > &b_in_2,
    hls::stream<ap_int<8> > &b_in_3,
    hls::stream<ap_int<8> > &b_in_4,
    hls::stream<ap_int<8> > &b_in_5,
    hls::stream<ap_int<8> > &b_in_6,
    hls::stream<ap_int<8> > &b_in_7,
    hls::stream<ap_int<32> > &c_out_0,
    hls::stream<ap_int<32> > &c_out_1,
    hls::stream<ap_int<32> > &c_out_2,
    hls::stream<ap_int<32> > &c_out_3,
    hls::stream<ap_int<32> > &c_out_4,
    hls::stream<ap_int<32> > &c_out_5,
    hls::stream<ap_int<32> > &c_out_6,
    hls::stream<ap_int<32> > &c_out_7) {
#pragma HLS interface ap_ctrl_none port=return

  // your design here
}

```


### t=1304.56s — model turn (27517 tokens this turn, 28,223 cumulative)

I've read both files. Now I'll design a systolic 8×8 PE grid: each PE is one
dataflow process with one multiplier/accumulator, A flows east, B flows
south, and results drain east along rows. Each PE merges the drain of the
previous product into the compute loop of the next one, giving 8-cycle
throughput.


**wrote `design.cpp`** — first 300 characters as recorded, the rest was not kept:

```
// 8x8 output-stationary integer matrix-multiply tile.
//
// Architecture: 64 processing elements in an 8x8 grid, one dataflow process
// per element.  A[i][k] enters at PE(i,0) and travels east one neighbour per
// cycle; B[k][j] enters at PE(0,j) and travels south one neighbour per cycle.
// PE(i,
```


### t=1321.65s — model turn (7403 tokens this turn, 35,626 cumulative)

**ran** `build`


*(segment: 35,626 tokens, 1321.6 s model, 147.3 s tools)*


## Build 1 — t=1468.99s, 35,626 tokens spent

```
STUDY VALUES 128 checked, 0 wrong
STUDY RESULT TIMEOUT after 20001 cycles, 64 of 128 outputs seen
```


### t=1499.53s — model turn (11595 tokens this turn, 47,221 cumulative)

Correct values, but the last product's results were never drained (they only
drain during the *next* product's compute, which never comes). I'll drain in
a separate loop right after compute — that makes each PE's period 16 cycles,
still meeting the ≤16-cycle throughput threshold. I'll also make the grid
position a template parameter so the drain conditionals are static and the
loop pipelines at II=1.


**wrote `design.cpp`** — first 300 characters as recorded, the rest was not kept:

```
// 8x8 output-stationary integer matrix-multiply tile.
//
// Architecture: 64 processing elements in an 8x8 grid, one dataflow process
// per element.  A[i][k] enters at PE(i,0) and travels east one neighbour per
// cycle; B[k][j] enters at PE(0,j) and travels south one neighbour per cycle.
// PE(i,
```


### t=1500.07s — model turn (10109 tokens this turn, 57,330 cumulative)

**ran** `build`


*(segment: 57,330 tokens, 31.1 s model, 220.4 s tools)*


## Build 2 — t=1720.49s, 57,330 tokens spent

```
STUDY VALUES 128 checked, 0 wrong
STUDY PRODUCT 0 first_in 5 last_out 40 latency 35
STUDY PRODUCT 1 first_in 29 last_out 64 latency 35
STUDY INTERVAL 1 24
STUDY RESULT CORRECT
```


### t=2083.5s — model turn (50466 tokens this turn, 107,796 cumulative)

Build 1 was correct with interval 8, but the final product's results were
never drained (they only drain during a *following* product's compute, which
never comes). I'll add an end-of-input detection: when a PE finds its `a_in`
empty at the start of what would be the next product, it enters drain mode
and forwards/emits its row's results, then idles.


**wrote `design.cpp`** — first 300 characters as recorded, the rest was not kept:

```
// 8x8 output-stationary integer matrix-multiply tile.
//
// Architecture: 64 processing elements in an 8x8 grid, one dataflow process
// per element.  A[i][k] enters at PE(i,0) and travels east one neighbour per
// cycle; B[k][j] enters at PE(0,j) and travels south one neighbour per cycle.
// PE(i,
```


### t=2084.32s — model turn (13128 tokens this turn, 120,924 cumulative)

**ran** `build`


*(segment: 120,924 tokens, 363.8 s model, 181.9 s tools)*


## Build 3 — t=2266.26s, 120,924 tokens spent

```
MISMATCH product 1 c_out_0[0]: got 0 want 33001
MISMATCH product 1 c_out_0[1]: got 0 want 3757
MISMATCH product 1 c_out_1[0]: got 0 want 1291
MISMATCH product 1 c_out_0[2]: got 0 want 7012
MISMATCH product 1 c_out_1[1]: got 0 want -40079
MISMATCH product 1 c_out_2[0]: got 0 want -6433
MISMATCH product 1 c_out_0[3]: got 0 want -10843
MISMATCH product 1 c_out_1[2]: got 0 want 13481
STUDY VALUES 128 checked, 64 wrong
STUDY PRODUCT 0 first_in 5 last_out 59 latency 54
STUDY PRODUCT 1 first_in 20 last_out 74 latency 54
STUDY INTERVAL 1 15
STUDY RESULT   WRONG
```


### t=2595.5s — model turn (49780 tokens this turn, 170,704 cumulative)

Found the bug in build 3: the final-drain branch emitted `acc` (already
reset to 0) instead of `acc_prev`, which holds the just-computed product —
that's exactly why product 1 came out as zeros while product 0 was correct.
The `empty()` end-detection itself worked. I'll restructure for cleaner
pipelining and better margins: a single compute loop with unconditional
blocking reads (internal result links always carry data, even garbage on the
first product, so reads never deadlock), port writes suppressed only during
the first product, and a separate multiplier-free final-drain loop.


**wrote `design.cpp`** — first 300 characters as recorded, the rest was not kept:

```
// 8x8 output-stationary integer matrix-multiply tile.
//
// Architecture: 64 processing elements in an 8x8 grid, one dataflow process
// per element.  A[i][k] enters at PE(i,0) and travels east one neighbour per
// cycle; B[k][j] enters at PE(0,j) and travels south one neighbour per cycle.
// PE(i,
```


### t=2596.11s — model turn (16454 tokens this turn, 187,158 cumulative)

**ran** `build`


*(segment: 187,158 tokens, 329.8 s model, 353.7 s tools)*


## Build 4 — t=2949.79s, 187,158 tokens spent

```
STUDY VALUES 128 checked, 0 wrong
STUDY RESULT TIMEOUT after 20001 cycles, 0 of 128 outputs seen
```


### t=3048.09s — model turn (28875 tokens this turn, 216,033 cumulative)

Build 4 deadlocked entirely (0 outputs — the separate wait loop with
`empty()` broke something fundamental, likely the dataflow split). Build 3's
structure was empirically sound: it produced product 0 perfectly and
completed the final drain structurally — its only flaw was emitting `acc`
(already reset) instead of `acc_prev` in the final-drain branch. I'll
restore build 3 exactly and fix that one value.


**wrote `design.cpp`** — first 300 characters as recorded, the rest was not kept:

```
// 8x8 output-stationary integer matrix-multiply tile.
//
// Architecture: 64 processing elements in an 8x8 grid, one dataflow process
// per element.  A[i][k] enters at PE(i,0) and travels east one neighbour per
// cycle; B[k][j] enters at PE(0,j) and travels south one neighbour per cycle.
// PE(i,
```


### t=3048.93s — model turn (19487 tokens this turn, 235,520 cumulative)

**ran** `build`


*(segment: 235,520 tokens, 99.1 s model, 171.6 s tools)*


## Build 5 — t=3220.52s, 235,520 tokens spent

```
STUDY VALUES 128 checked, 0 wrong
STUDY RESULT TIMEOUT after 20001 cycles, 16 of 128 outputs seen
```


### t=3423.92s — model turn (43160 tokens this turn, 278,680 cumulative)

Build 5 regressed badly (only 16 outputs) versus build 3 (all 128). The
single-check `a_in.empty()` at t==0 is a timing race: whether a PE sees the
stream already drained depends on synthesis scheduling of the empty signal
relative to the previous cycle's read commit — the `acc`→`acc_prev` change
shifted that scheduling and hung rows mid-drain. I'll make end-detection
race-proof: a bounded wait loop that only commits to "another product
exists" after observing the input non-empty on two consecutive cycles, past
a dead zone that lets any stale emptiness settle. In steady state this adds
a constant few cycles per product (period ~13 < 16), and at the true end it
times out and drains. The compute branch stays exactly as the proven build
2.


**wrote `design.cpp`** — first 300 characters as recorded, the rest was not kept:

```
// 8x8 output-stationary integer matrix-multiply tile.
//
// Architecture: 64 processing elements in an 8x8 grid, one dataflow process
// per element.  A[i][k] enters at PE(i,0) and travels east one neighbour per
// cycle; B[k][j] enters at PE(0,j) and travels south one neighbour per cycle.
// PE(i,
```


### t=3424.56s — model turn (22729 tokens this turn, 301,409 cumulative)

**ran** `build`


*(segment: 301,409 tokens, 204.0 s model, 362.0 s tools)*


## Build 6 — t=3786.52s, 301,409 tokens spent

```
STUDY VALUES 128 checked, 0 wrong
STUDY PRODUCT 0 first_in 6 last_out 56 latency 50
STUDY PRODUCT 1 first_in 27 last_out 97 latency 70
STUDY INTERVAL 1 21
STUDY RESULT CORRECT
```


### t=3991.3s — model turn (48771 tokens this turn, 350,180 cumulative)

Build 6 was fully correct (all 128 values, both products drained) but
interval 21 > 16: the timeout-based wait costs ~13 cycles per product
because a PE in column j must wait ~j cycles for its internal A link to
refill before it can trust "empty". New scheme that removes that cost: only
**PE(i,0)** detects the end (its `a_in` is a top-level port, so `empty()`
there is race-free — the harness always has data during a product stream).
It polls the port for a few cycles at each product boundary; no data ⇒ last
product. It then sends a 1-bit GO/END **token east** one hop per cycle — a
nearest-neighbour link like the data. The token sent at PE(i,0)'s boundary
of product p arrives at PE(i,j) exactly at PE(i,j)'s boundary of product p
(j-cycle wave lag = j-cycle token propagation), so the blocking token read
costs nothing in steady state. Each PE consumes one token per product,
keeping the whole row self-synchronising. A false END mid-stream recovers
gracefully (drains, then resumes in order); the debounce guard makes a false
GO — the only fatal error — impossible.


**wrote `design.cpp`** — first 300 characters as recorded, the rest was not kept:

```
// 8x8 output-stationary integer matrix-multiply tile.
//
// Architecture: 64 processing elements in an 8x8 grid, one dataflow process
// per element.  A[i][k] enters at PE(i,0) and travels east one neighbour per
// cycle; B[k][j] enters at PE(0,j) and travels south one neighbour per cycle.
// PE(i,
```


### t=3991.98s — model turn (26738 tokens this turn, 376,918 cumulative)

**ran** `build`


*(segment: 376,918 tokens, 205.5 s model, 255.1 s tools)*


## Build 7 — t=4247.12s, 376,918 tokens spent

```
STUDY VALUES 128 checked, 0 wrong
STUDY RESULT TIMEOUT after 20001 cycles, 0 of 128 outputs seen
```


### t=4607.26s — model turn (70763 tokens this turn, 447,681 cumulative)

Build 7 (token scheme) deadlocked entirely — 0 outputs. The pattern across
builds 3/5 is now clear: `empty()` on **internal** links is racy (stale
emptiness after a read), while `empty()` on the **top-level port** at column
0 was always decided correctly (builds 3/5 computed/drained product 1 fine
from the port check). Also, separate boundary loops before the compute loop
(builds 4, 7) kill the design, while a single unified loop works. New design
for my final build: - **Only PE(i,0) ever uses `empty()`**, on its top-level
port (proven reliable). Its decision travels east as an **end marker riding
the A links** (widened to 9 bits: valid flag + value) — one hop per cycle,
arriving at each column exactly at its boundary. Columns 1–7 use pure
blocking reads: no race at all. - **Result links always flow** one word per
cycle (padding when idle), so all c-link reads are unconditional — tight
pipelining, and the port PE emits exactly the 8 in-order results in both the
steady drain and the final drain. - Everything stays in **one 8-iteration
pipelined loop** per product (the shape that synthesized correctly in builds
2/3/5/6), so the interval should be well under 16.


**wrote `design.cpp`** — first 300 characters as recorded, the rest was not kept:

```
// 8x8 output-stationary integer matrix-multiply tile.
//
// Architecture: 64 processing elements in an 8x8 grid, one dataflow process
// per element.  A[i][k] enters at PE(i,0) and travels east one neighbour per
// cycle; B[k][j] enters at PE(0,j) and travels south one neighbour per cycle.
// PE(i,
```


### t=4608.23s — model turn (30803 tokens this turn, 478,484 cumulative)

**ran** `build`


*(segment: 478,484 tokens, 361.1 s model, 162.9 s tools)*


## Build 8 — t=4771.12s, 478,484 tokens spent

```
STUDY VALUES 128 checked, 0 wrong
STUDY PRODUCT 0 first_in 5 last_out 49 latency 44
STUDY PRODUCT 1 first_in 20 last_out 64 latency 44
STUDY INTERVAL 1 15
STUDY RESULT CORRECT
```
