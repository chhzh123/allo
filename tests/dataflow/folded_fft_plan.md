# Folded FFT-256: Transformation Plan

## Goal

The user writes a **simple, fully-spatial kernel** and the compiler automatically
transforms it into a high-performance HLS design matching the hand-optimized
vectorized FFT-256 (latency=99, interval=19, DSP=328).

```python
# User writes this:
@df.kernel(mapping=[LOG2_N, HALF_N], fold={1: FOLD}, args=[rev_re, rev_im, out_re, out_im])
def butterfly(buf_re, buf_im, res_re, res_im):
    s, b = df.get_pid()
    stride: ConstExpr[int32] = _stride(s)
    upper: int32 = ((b >> s) << _s_plus_1(s)) | (b & mask)
    lower: int32 = upper | stride
    # ... simple butterfly ...
    res_re[upper] = a_re + bw_re
    res_re[lower] = a_re - bw_re
```

The compiler should produce performance equivalent to the manually-constructed
`fft_256` with 10 separate stage kernels chained by streams.

---

## Architecture Gap Analysis

### Reference (`fft_256`) — Latency: 99, Interval: 19, DSP: 328

```
inp_re/im (stream)
  → [bit_rev_stage] → s_re[0], s_im[0]
  → [intra_0] → s_re[1] (STRIDE=1)
  → [intra_1] → s_re[2] (STRIDE=2)
  → [intra_2] → s_re[3] (STRIDE=4)
  → [intra_3] → s_re[4] (STRIDE=8)
  → [intra_4] → s_re[5] (STRIDE=16)
  → [inter_0] → s_re[6] (STRIDE=32, dataflow: load/compute/write)
  → [inter_1] → s_re[7] (STRIDE=64)
  → [inter_2] → s_re[8] (STRIDE=128)
  → [output_stage] → out_re/im (stream)
```

Key properties:
- Each stage has **dedicated** input/output streams (single writer, single reader)
- `#pragma HLS dataflow` overlaps all stages (interval = max stage latency = 19)
- Each stage processes WIDTH=32 elements per clock (vectorized)
- Inter stages use sub-function dataflow (LOAD || COMPUTE || WRITE)

### Folded (`get_fft_256_folded(FOLD=128)`) — Current: Latency: 2264, Interval: 2265

```
inp_re/im (m_axi)
  → [load_buf0..3]
  → [bit_rev_0, bit_rev_1]
  → [butterfly_0_0] ─┐
  → [butterfly_1_0]  │ ALL share same buf2/buf3 (input)
  → [butterfly_2_0]  │ ALL share same buf4/buf5 (output)
  → ...              │
  → [butterfly_7_0] ─┘
  → [store_res2..5]
  → out_re/im (m_axi)
```

Problems:
1. **Shared buffers**: All 8 butterfly stages read/write the same arrays → violates
   HLS dataflow's single-writer/single-reader constraint
2. **Sequential execution**: Without dataflow, stages run one after another
   (198 cycles × 8 stages = 1584 + load/store overhead = 2264)
3. **No stage pipelining**: Interval = total latency (no overlap)

---

## Synthesis Results Comparison

| Metric | Reference (fft_256) | Folded (no dataflow) | Ratio |
|--------|-------------------|---------------------|-------|
| Latency | 99 cycles | 2,264 cycles | 22.9× |
| Interval | 19 cycles | 2,265 cycles | 119× |
| DSP | 328 (2%) | 32 (~0%) | 0.10× |
| FF | 257,407 (3%) | 95,158 (3%) | 0.37× |
| LUT | 277,125 (8%) | 109,192 (8%) | 0.39× |
| BRAM | 0 | 18 (~0%) | — |
| Timing | +0.08 slack | -0.56 slack | ✗ |

Target device: reference uses xcvp1802 (Versal Premium), folded uses xcu280 (U280).

---

## Required Compiler Transformation: Buffer Chain Insertion

### The Problem

When `mapping=[S, B], fold={1: F}` expands into S × (B/F) PE instances, the
builder currently wires ALL instances along dim-0 (stages) to the same region
arrays:

```
butterfly_0_0(rev_re, rev_im, out_re, out_im)
butterfly_1_0(rev_re, rev_im, out_re, out_im)  ← same arrays!
butterfly_2_0(rev_re, rev_im, out_re, out_im)  ← same arrays!
...
```

### The Solution

The builder should detect that consecutive instances along the **non-folded
stage dimension** form a sequential pipeline, and insert intermediate arrays:

```
inter_01_re: float32[N]
inter_01_im: float32[N]
inter_12_re: float32[N]
inter_12_im: float32[N]
...
inter_67_re: float32[N]
inter_67_im: float32[N]

butterfly_0_0(rev_re,     rev_im,     inter_01_re, inter_01_im)
butterfly_1_0(inter_01_re, inter_01_im, inter_12_re, inter_12_im)
butterfly_2_0(inter_12_re, inter_12_im, inter_23_re, inter_23_im)
...
butterfly_7_0(inter_67_re, inter_67_im, out_re,      out_im)
```

Now each intermediate array has exactly one writer and one reader → HLS dataflow
works → stages pipeline → interval ≈ max(stage latency).

### API Design

```python
@df.kernel(
    mapping=[LOG2_N, HALF_N],
    fold={1: FOLD},
    chain=0,                    # chain along dim 0 (stages)
    args=[rev_re, rev_im, out_re, out_im],
)
def butterfly(buf_re, buf_im, res_re, res_im):
    ...
```

`chain=0` tells the builder:
1. For the non-folded stage dimension (dim 0), create intermediate arrays
   between consecutive stage instances
2. The kernel's formal parameters are split into inputs (`buf_re`, `buf_im`)
   and outputs (`res_re`, `res_im`)
3. Wire: instance_s reads from instance_{s-1}'s outputs, writes to its own outputs
4. First instance reads from region input args, last writes to region output args

The input/output split is determined by analyzing load/store patterns on each
formal parameter: parameters that are only loaded from are inputs, parameters
that are stored to are outputs.

### Implementation Steps

#### Step 1: `chain` keyword parsing
- Parse `chain=dim_index` in `@df.kernel` decorator (builder.py)
- Validate: chained dimension must not be folded
- Store chain info in kernel context

#### Step 2: Intermediate array allocation
- During PE expansion in `build_FunctionDef`:
  - For chained dimension with S values, create (S-1) intermediate array pairs
  - Each intermediate array has the same shape/type as the chained formal params
  - Emit `memref.alloc` ops in the top function body

#### Step 3: Instance wiring
- When inserting function calls in `_build_top` or the builder's PE expansion:
  - Instance 0: reads from region input args, writes to intermediate[0]
  - Instance s (0 < s < S-1): reads from intermediate[s-1], writes to intermediate[s]
  - Instance S-1: reads from intermediate[S-2], writes to region output args

#### Step 4: Optimization pass
- After chain insertion, apply standard optimizations:
  - `s.auto_f2()` on all intermediate arrays (bank-conflict-free partitioning)
  - HLS dataflow is now valid (single writer/reader per buffer)
  - Pipeline/unroll the fold loops within each stage

#### Step 5: HLS pragma generation
- Intermediate arrays get `partition`, `bind_storage`, `dependence` pragmas
  (handled by auto_f2)
- Top function gets `#pragma HLS dataflow` (already emitted for regions)

### Actual Synthesis Results After Chain Insertion

#### Config A: Full Unroll (FOLD=128, unroll=128)
- Latency: 668, Interval: 256, DSP: **13,714 (151%)** — over budget!
- Each butterfly: 16 cycles (fast), but 128 parallel multipliers × 8 stages
- FF: 2.9M (110%), LUT: 2.4M (183%)

#### Config B: Pipeline II=1 (FOLD=128, pipeline, no unroll)
- **Latency: 1700, Interval: 256, DSP: 122 (1%)** — fits on xcu280!
- Each butterfly: 145 cycles (128 iter pipelined, iter_latency=17)
- FF: 759K (29%), LUT: 418K (32%), BRAM: 8
- Interval=256 dominated by m_axi load/store (266 cycles each)

#### Analysis: Gap to Reference (Latency=94, Interval=19)
- Reference uses `hls::stream<hls::vector<float,32>>` I/O → 8 iterations → ~19 cycles
- Folded uses scalar `float32[256]` → m_axi load/store → 256 cycles per burst
- **Root cause**: I/O architecture difference, not compute efficiency
- To match reference interval: need vectorized stream I/O (future work)

---

## Progress

### Completed ✅

1. **Grid folding** (`fold={axis: factor}`) — builder, infer, utils
2. **Auto-F2 + fold integration** — auto_f2 detects fold loop patterns
3. **Folded FFT-256 frontend** — `get_fft_256_folded(FOLD)` compiles
4. **HLS codegen** — generates valid C++ with unrolled fold loops
5. **Synthesis baseline** — folded design synthesizes (without dataflow)
6. **Buffer chain insertion** (`chain=dim`) — implemented
   - `allo/ir/builder.py`: Parse `chain=` keyword, validate, store on ctx
   - `allo/dataflow.py`: `_build_top` allocates intermediate memrefs and wires calls
   - `allo/customize.py`: Propagates chain info from builder ctx to Schedule
   - Region-local arrays detected and allocated as fresh memrefs (not top-level args)
   - Chain intermediate allocs get `name` attributes for valid C++ names
7. **Auto-optimization pass** — pipeline fold loops + complete partition + twiddle partition
8. **HLS codegen verified** — `test_fft_256_folded_hls` passes
9. **Schedule API: pipeline replaces unroll** — `s.pipeline()` removes conflicting `unroll` attr
10. **Performance validation** — csyn completed, pipeline config fits on FPGA

### Blocked / Future 🔲

11. **Vectorized stream I/O** — needed to match reference's 19-cycle interval
12. **Multi-PE chain** — handle FOLD < N//2 (multiple PEs per stage)
13. **Partial unroll** — intermediate configs between full-pipeline and full-unroll

---

## Files Modified

| File | Change |
|------|--------|
| `allo/ir/builder.py` | Parse `chain=` keyword, store chain info on original ctx |
| `allo/dataflow.py` | `_build_top` chain detection, intermediate array alloc, call wiring, region-local array handling |
| `allo/customize.py` | Propagate `ctx._kernel_chain_dim` to `Schedule.chain_info`; `s.pipeline()` removes conflicting unroll |
| `tests/dataflow/test_fft.py` | Add `chain=0` to butterfly, `_apply_folded_optimizations` with pipeline + partition, tests |
