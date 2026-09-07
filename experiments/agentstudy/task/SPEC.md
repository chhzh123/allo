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
  only with the elements directly north, south, east and west of it. Nothing is
  broadcast: no value reaches more than one element in the same cycle, and no
  element reads an input port other than the ones described below.
- **`A` travels east.** `A[i][k]` enters the grid at `PE(i,0)` and passes from
  each element to its eastern neighbour.
- **`B` travels south.** `B[k][j]` enters the grid at `PE(0,j)` and passes from
  each element to its southern neighbour.
- **Results leave along the rows.** Each row's accumulators reach output port
  `i` through its own chain of elements, so no element drives an output port
  other than through its neighbours.

Because a value takes one step per element, the element in the far corner
cannot finish before the wave has crossed the grid. That is a property of this
architecture and the harness checks for it.

## The interface your module must present

The module has 24 handshaked ports. Every port carries one value per transfer
and uses the same two-signal handshake described below. Your module chooses when
to transfer; the surrounding test harness always has input data available and
always accepts output data.

| Port | Direction | Width | Carries |
|---|---|---|---|
| `a_in_0` .. `a_in_7` | input | 8 bits, signed | port `i` carries `A[i][0]`, `A[i][1]`, ... `A[i][7]`, in that order |
| `b_in_0` .. `b_in_7` | input | 8 bits, signed | port `j` carries `B[0][j]`, `B[1][j]`, ... `B[7][j]`, in that order |
| `c_out_0` .. `c_out_7` | output | 32 bits, signed | port `i` carries `C[i][0]`, `C[i][1]`, ... `C[i][7]`, in that order |

Each input port therefore receives exactly 8 values, and each output port
produces exactly 8 values, for one 8x8 by 8x8 product.

### The handshake

An input port `p` has `p_dout` (data, driven by the harness), `p_empty_n` (the
harness has a value, driven by the harness) and `p_read` (take it, driven by
your module). A value transfers on a rising clock edge when `p_empty_n` and
`p_read` are both high.

An output port `p` has `p_din` (data, driven by your module), `p_full_n` (the
harness can accept, driven by the harness) and `p_write` (here is a value,
driven by your module). A value transfers on a rising clock edge when `p_full_n`
and `p_write` are both high.

There is no start signal and no done signal. The module runs continuously from
reset and processes products back to back: after the last value of one product
it accepts the first value of the next.

Clock is `ap_clk`, rising edge. Reset is `ap_rst_n`, active low, asserted for at
least eight cycles before the first transfer.

## What your design must achieve

| Requirement | Threshold |
|---|---|
| Correctness | every one of the 64 output values exact, on held-out inputs |
| Correct structure | the architecture above, checked by reading your design |
| Latency | between 22 and 64 clock cycles from the first input transfer to the last output transfer of the **first** product after reset |
| Throughput | at most 16 clock cycles between the starts of consecutive products, once running |
| Clock | routes on the target device at a 3.333 ns period with non-negative worst slack |
| Multipliers | exactly 64 DSP blocks, one per processing element |
| Logic | at most 25,000 lookup tables |

The latency has a floor as well as a ceiling. A value entering at `PE(i,0)`
needs seven steps to reach the eastern edge and seven more southward, so the
far corner cannot have its last operand before cycle 22, and a design that
produces every result sooner than that is moving data further than one
neighbour per cycle. Both upper thresholds are twice the lower bound the ports
themselves impose. Each
port carries eight values per product, so no design can start products closer
together than eight cycles, and none can finish one in fewer than about
twenty-four. A design that computes the 512 multiplications one at a time meets
neither; a design that finishes one product completely before accepting the
next will meet the latency threshold but not the throughput one.

## How you will be evaluated

Latency is measured on the first product only, with the design empty, so a
design that overlaps products is not charged for the ones queued behind.
Throughput is measured between the starts of later products.

A test harness compiles your design, elaborates it, and drives it with input
vectors you have not seen, in a register-transfer-level simulation. It reports
the number of mismatching values and the latency in clock cycles. You may run
this harness as often as your budget allows.

Resource and timing figures come from synthesis while you work; the final
routing check runs once, after you submit, so treat the synthesis estimate as
your guide and leave margin.

## Rules

- Write only the design. The test harness and its vectors are provided.
- Do not read or write any file outside your working directory.
- You have no network access.
- When you believe the design meets every requirement, say `SUBMIT` on a line by
  itself. Your last passing artifact is what gets routed.
