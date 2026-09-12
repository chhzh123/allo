# E2: a radix-2 complex FP32 FFT, in three systems

N = 256 unless a row says otherwise. Everything below is routed out of context
at 3.333 ns on `xcu280-fsvh2892-2L-e` with nothing unrouted, and every cycle
figure comes from RTL cosimulation. `results.csv` is the full table, including
variants this page does not show.

## The result

SPMW's `lanes` design against HP-FFT's best configuration at each datapath
width. HP-FFT's boundary carries `2*UF` complex samples a beat and SPMW's
carries `W`, so `W = 2*UF` is the matched pairing and both share the ideal
interval `N / (samples per cycle)`.

| samples/cyc | system | latency | interval | of ideal | LUT | FF | DSP | BRAM18 | slack | MHz |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | HP-FFT UF1 | 1,527 | 140.5 | 91.1% | 22,403 | 22,244 | 72 | 82 | +0.283 | 328 |
| | **SPMW lanes W=2** | **645** | **128.0** | **100%** | 24,012 | 28,614 | 72 | **10** | **+0.443** | **346** |
| 4 | HP-FFT UF2 | 766 | 74.5 | 85.9% | 45,482 | 37,809 | 132 | 104 | +0.317 | 332 |
| | **SPMW lanes W=4** | **376** | **64.0** | **100%** | 45,711 | 54,192 | 144 | **20** | +0.241 | 323 |
| 8 | HP-FFT UF4 | 408 | 42.0 | 76.2% | 91,524 | 90,201 | 246 | 166 | +0.119 | 311 |
| | **SPMW lanes W=8** | **260** | **32.0** | **100%** | **87,021** | 102,464 | 258 | **40** | **+0.292** | **329** |
| 16 | HP-FFT UF8 | 269 | 26.0 | 61.5% | 179,275 | 148,768 | 474 | 288 | **-0.093** | 292 |
| | **SPMW lanes W=16** | **200** | **16.0** | **100%** | **167,847** | 195,716 | 486 | **64** | **+0.079** | **307** |

Three things to take from it.

**SPMW holds exactly 100% of ideal interval at every width. HP-FFT does not,
and gets worse as it widens** -- 91.1%, 85.9%, 76.2%, 61.5%.

**The two land on nearly the same silicon.** At 8 and 16 samples a cycle the
lookup tables and multipliers agree within 5%, and SPMW uses a quarter of the
block RAM. It pays 14% to 32% more registers. So the performance gap is not
bought with area.

**HP-FFT's widest configuration does not close timing** (-0.093 ns). Its
interval of 26.0 is real and cosimulated, but there is no implementation
behind it at 300 MHz. SPMW closes at every width.

Both HP-FFT rows at 8 and 16 samples a cycle are the **fixed** builds; the
shipped sources reach 8.0% and 28.1% of ideal on this target, for reasons in
`hpfft/fix/README.md`. Comparing against the shipped ones would flatter SPMW
by a factor of ten and is not done here.

## Accounting for every difference

One cause explains most of the table: **what sits between two stages.**

HP-FFT declares eight whole transforms' worth of array between its stages --
`static complex<float> data_1[FFT_NUM] ... data_8[FFT_NUM]` -- inside a
`#pragma HLS dataflow` region. Array channels in a dataflow region are **PIPO**,
ping-pong: the consumer task cannot start until the producer task has finished,
and the buffer is doubled so the next transform can fill one half while this one
drains the other.

SPMW connects its units with **depth-8 FIFOs carrying one complex sample each**
-- `spmw_fifo #(.DW(64), .DEPTH(8))`, 32 of them in the W=8 fabric, read out of
the generated `spmw_top.sv`. A consumer starts on the first token, not the
last, and the channel holds eight samples rather than a transform.

That is also **not** the mechanism the reference Allo design uses, which is
worth stating because the two are easy to conflate. `feature/allo-fft`'s
`tests/dataflow/test_fft.py` streams a whole `WIDTH`-element vector per token
(`Stream[float32[WIDTH], 2]`) and unrolls a lane loop inside the stage, so the
swizzled indices `il`/`iu` fold to constants. This design has **no shared
buffer and no bank arithmetic at all**: the `W` lanes are `W` separate scalar
streams, a stage is `W/2` sites, and site `m` reads the two lanes it needs on
two ports. The lane index is a *placement coordinate*, which is a stronger
compile-time constant than an unrolled loop index. Where the partner is `D >= W`
away -- the same lane, several beats back, which no site can read -- a
permutation element in front of the stage turns that beat distance into a lane
distance.

Everything below follows from that, and each row is checked rather than
asserted.

| | SPMW | HP-FFT | ratio | why |
|---|---:|---:|---:|---|
| interval | 32.0 | 42.0 | **0.76x** | HP-FFT's stage modules restart per transform |
| latency | 260 | 408 | **0.64x** | PIPO serialises HP-FFT's stages within a transform |
| BRAM18 | 40 | 166 | **0.24x** | eight partitioned, ping-ponged inter-stage arrays |
| LUT | 87,021 | 91,524 | 0.95x | same arithmetic; within noise |
| DSP | 258 | 246 | 1.05x | same butterfly count; within noise |
| FF | 102,464 | 90,201 | **1.14x** | SPMW holds in registers what HP-FFT holds in BRAM |

(8 samples a cycle. At 16 the pattern is the same: 0.62x interval, 0.74x
latency, 0.22x BRAM, 0.94x LUT, 1.03x DSP, 1.32x FF.)

### Interval: a refill HP-FFT pays once per transform

HP-FFT's *inner loops* all reach II=1 -- `FFT_Stage1` at iteration latency 7,
the spatial-unroll stages at 17 to 19, each reporting `Interval 1`. Its stage
*modules* do not: `FFT_stage_spatial_unroll_4_s` is latency 51, interval 51. A
module whose interval equals its latency restarts from empty every transform.

The refill is a constant, and it is the whole efficiency curve:

| config | ideal | measured | overhead |
|---|---:|---:|---:|
| UF1 | 128 | 140.5 | +12.5 |
| UF2 | 64 | 74.5 | +10.5 |
| UF4 | 32 | 42.0 | +10.0 |
| UF8 | 16 | 26.0 | +10.0 |

Ten cycles at every width. The ideal interval halves each step and the constant
does not, so its share grows from 9% to 38% -- 91.1% of ideal down to 61.5%.
SPMW's units are persistent processes at II=1 that never restart, so a stage's
interval is exactly its trip count.

### Latency: PIPO makes the stages sequential inside one transform

This is a different mechanism from the interval, and it is the larger effect.
Because the channels are PIPO, stage `k+1` waits for stage `k` to *finish the
whole transform*. A transform's latency is therefore the **sum** of the stage
latencies, and HP-FFT's own csynth says so:

    60 + 39 + 40 + 49 + 51 + 50 + 50 + 50 = 389        measured 408

The 19-cycle remainder is the output stage and the handshakes. Stages overlap
only *across* transforms, which is why the interval (42) is an eighth of the
latency (408).

SPMW's stages overlap *within* a transform: a consumer starts on the first
token, so latency is the sum of the stage pipeline *depths* plus the beats of
one transform, not the sum of whole stage runtimes. At W=8 that is 260 for 32
beats -- about 28 cycles of depth per stage, which is one butterfly's float
pipeline plus its lane buffer.

So the two systems differ on latency for the same reason a bucket brigade beats
carrying one load at a time to the end of the line and back.

### BRAM: the cost is banks, not capacity

Eight arrays of 256 complex FP32 is 131,072 bits; ping-pong doubles it to
262,144. That is **14 RAMB18 of capacity**, and UF4 uses **166**.

The difference is `#pragma HLS array_partition ... cyclic factor=UF` (and
`factor=UF*4` on `data_2`). Partitioning splits each array into banks so the
unrolled butterflies can read it in parallel, and **each bank becomes its own
block RAM however little it holds** -- a bank of 16 complex floats is 1 Kb in a
device primitive that holds 18. Eight arrays, ten-odd banks each, doubled by
ping-pong, is the 166.

SPMW has no inter-stage arrays to partition. Its block RAM is the twiddle ROMs
and nothing else, which is why it does not grow the same way: 10, 20, 40, 64
across the four widths against HP-FFT's 82, 104, 166, 288.

### Registers: the same trade, seen from the other side

SPMW is 14% to 32% *higher* on flip-flops, and it is the direct counterpart of
the block RAM result. The lane design's per-stage vector buffers and delay lines
live in registers and SRL; HP-FFT's inter-stage data lives in block RAM. Neither
is storing much more than the other -- they are storing it in different device
resources.

### Lookup tables and multipliers: no difference to explain

Within 5% at both widths, in both directions. Both designs instantiate one
butterfly per stage per lane at matched throughput, and both now bind their
float adders to fabric, so the arithmetic is the same hardware. **This is the
part of the comparison where the two systems agree**, and it is what makes the
other four rows worth reporting: the performance gap is not paid for in logic.

## Three SPMW designs, and why more than one is kept

They are identical on interval -- 100% of ideal at every width -- and differ in
latency and in what they spend. At 2 samples a cycle:

| design | latency | LUT | FF | DSP | BRAM18 | structure |
|---|---:|---:|---:|---:|---:|---|
| rolled W=2 | **581** | 24,318 | 55,109 | 318 | 12 | explicit `spmw.to` links, one unit per lane per stage |
| paired W=2 | 919 | 21,266 | 24,582 | 72 | 44 | one unit a stage on a one-site grid, own buffer |
| lanes W=2 | 645 | 24,012 | 28,614 | 72 | **10** | lane axis unrolled, bank index a compile-time constant |

**`rolled` is fastest to a first result at 2 and 4 samples a cycle** (581 and
368 against the lane design's 645 and 376), because it streams rather than
buffering a vector. **`lanes` overtakes it at 8 and 16** (260 and 200 against
271 and 222) and costs a third to a fifth of the multipliers throughout.

The reason `rolled` needs those multipliers: a butterfly consumes two samples,
its unit is fed one a cycle, so each multiplier is half idle and the design
needs twice as many. Its DSP-per-throughput barely moves across a 16x widening,
159 to 147, because nothing in it ever pairs a butterfly's operands. The lane
design's doubles cleanly -- 72, 144, 258, 486 -- because every multiplier stays
busy.

**`lanes` is the one to quote.** It is the only form that gets the multiplier
count, the block RAM and the interval at once.

## The other transform sizes

Measured for the original single-lane design only, one sample a cycle against
HP-FFT's narrowest:

| N | SPMW latency | interval | HP-FFT UF1 latency | interval |
|---:|---:|---:|---:|---:|
| 128 | **581** | **128.0** | 742 | 76.0 |
| 256 | **1,158** | **256.0** | 1,527 | 140.5 |
| 512 | **2,310** | **512.0** | 3,208 | 269.0 |
| 1,024 | **4,614** | **1,024.0** | 6,809 | 525.5 |

SPMW's interval is exactly N at every size, 100% of ideal for a one-sample
boundary; HP-FFT's is near N/2 because its boundary carries two samples. On the
matched-width comparison at the top, that difference is removed.

## Allo, the third system

Allo's strided in-place FFT computes one transform per call, so its interval is
its latency: 26,220 cycles at N=256 against SPMW's 256. It is a different kind
of design -- not a streaming pipeline -- and is reported for completeness
rather than as a throughput competitor. It is the smallest of the three by a
wide margin: 4,422 LUT, 16 DSP.

## How the numbers are defined

- **Latency** is a transform's last output beat minus the launch's first input
  beat: `first_out + N/W - 1 - first_in` from the cosimulation's counters.
- **Interval** is the steady spacing between consecutive transform completions
  when transforms stream back to back, the median over 32 of them.
- **BRAM18** is `2 x Block RAM Tile` for every row. Reading a `RAMB18E2` line
  instead gives 6 for HP-FFT UF4, whose memory is mostly RAMB36, where the
  true 18K-equivalent is 166. This trap has been hit three times in this work.
- **Adders in fabric on both sides.** HP-FFT binds its butterfly's float adds
  with `bind_op ... impl=fabric`; SPMW does the same through
  `spmw_bind_fabric`. Without it SPMW's DSP count is 2.4x higher and the
  comparison is of who remembered the pragma.

## Reproducing

    # SPMW, any design and width
    python3 scripts/spmw_build_array.py --design fftlanes --size 256 --lanes 8 --pnr
    python3 scripts/spmw_build_array.py --design fftrolled --size 256 --lanes 8 --cosim

    # HP-FFT, including the fixed variants
    bash hpfft/fix/mkvariant.py            # generates the variants
    bash /scratch/hc676/hpfft_pnr2.sh UF4_a8   # routes one, IP cores included

Designs are in `spmw/source/`, the generated hardware and every csynth report
in `spmw/generated/`, and HP-FFT's investigation in `hpfft/fix/README.md`.

## What is in results.csv but not above

The shipped (unfixed) HP-FFT rows; SPMW's `folded SDF` and `rolled W=1`, which
are the same design at one sample a cycle; the `paired` variants including a
banked-buffer form that reaches only 512 cycles of interval and is kept as the
measurement that rejected it; and HP-FFT's static-only UF4 fallback, which was
routed in case the full fix missed timing and was not needed.
