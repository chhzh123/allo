# Allo Dataflow FFT Features

This document tracks compiler features added to enable high-performance FFT
generation targeting Vitis HLS, guided by the F2 linear layout synthesis plan.

---

## 1. F2 Linear Layout Solver (`allo/transform/f2_layout.py`)

**What it does:**
Implements Phases 1–3 of `plan.md`: given an FFT size N, vector width W, and a
set of butterfly stride bits, automatically synthesizes a conflict-free memory
bank swizzle matrix S over GF(2).

**Algorithm:**
- Represent every memory address as an n-bit vector over GF(2), where
  `n = log2(N)`.
- Identify the conflict subspace: stride-2^s butterfly pairs share the same
  lower `bank_bits` address bits whenever `s >= bank_bits`, causing bank
  conflicts under the default cyclic assignment.
- Solve for a minimal-XOR swizzle matrix S (heuristic: XOR the MSB bank row
  with the conflicting stride bit) such that every non-zero conflict vector
  maps to a distinct bank address.
- Verify correctness: assert `S @ delta != 0` for every stride delta.

**Key classes:**

| Class / Function | Description |
|---|---|
| `F2LayoutSolver(n_bits, bank_bits)` | Main solver; call `.solve(stride_bits)` |
| `SwizzleHelper` | Wraps solved S; provides `bank_expr`, `offset_expr`, `dims`, `swizzle_bank` |
| `fft_swizzle(N, WIDTH, stride_bit)` | Convenience factory for FFT inter-vector stages |

**Example (N=256, WIDTH=32, STRIDE=32):**
```python
from allo.transform.f2_layout import fft_swizzle

helper = fft_swizzle(N=256, WIDTH=32, stride_bit=5)
print(helper.dims())        # (32, 8) → 2D buffer shape [32][8]
print(helper.bank_expr("i"))
# ((i >> 0) & 1) | (((i >> 1) & 1) << 1) | ... | (((i >> 4) & 1) ^ ((i >> 5) & 1)) << 4)
```

Resulting swizzle matrix S for stride_bit=5:
```
[[1 0 0 0 0 0 0 0]
 [0 1 0 0 0 0 0 0]
 [0 0 1 0 0 0 0 0]
 [0 0 0 1 0 0 0 0]
 [0 0 0 0 1 1 0 0]]   ← bank bit 4 = addr[4] XOR addr[5]
```

This eliminates bank conflicts for STRIDE=32 butterflies: elements `idx` and
`idx XOR 32` map to different banks.

---

## 2. `_build_top` Deduplication Fix (`allo/dataflow.py`)

**Problem fixed:**
When a dataflow region had shared I/O arrays used by multiple kernels (e.g.,
`bit_rev_stage` and `output_stage` both referencing `out_re`/`out_im`), the
compiler incorrectly deduplicated them by formal parameter name (`local_re`,
`local_im`) instead of region argument name. This collapsed 4 distinct I/O
arrays into 2, causing an assertion failure.

**Fix:**
Changed the deduplication key in `_build_top` from `dtensor.name` to
`dtensor.top_name` (the region-level argument name such as `inp_re`, `inp_im`,
`out_re`, `out_im`).

---

## 3. New Schedule Primitives (`allo/customize.py`)

### `s.bind_storage(target, impl, storage_type="")`

Emits `#pragma HLS bind_storage variable=<buf> type=<storage_type> impl=<impl>`.

| Argument | Example values |
|---|---|
| `impl` | `"bram"`, `"uram"`, `"lutram"`, `"srl"` |
| `storage_type` | `""` (auto), `"ram_1p"`, `"ram_2p"`, `"ram_t2p"`, `"ram_s2p"` |

**Example:**
```python
s.bind_storage("inter_5_0:in_re", impl="lutram", storage_type="ram_2p")
```
Emits: `#pragma HLS bind_storage variable=in_re type=ram_2p impl=lutram`

`ram_2p` enables dual-port access, which is required for HLS DATAFLOW ping-pong
buffering (one port writes new data while another reads previous call's data).

### `s.dependence(target, dep_type="inter", direction="false")`

Emits `#pragma HLS dependence variable=<buf> inter false`.

Suppresses conservative false loop-carried dependency analysis by HLS, enabling
II=1 for loops that compute butterfly indices via XOR expressions that HLS cannot
easily prove are conflict-free.

**Example:**
```python
s.dependence("inter_5_0:out_re_b")
```
Emits: `#pragma HLS dependence variable=out_re_b inter false`

---

## 5. Vectorized FFT-256 with F2 Swizzle (`tests/dataflow/test_fft.py`)

**What it is:**
A complete, annotated Allo dataflow implementation of a radix-2 DIT FFT for
N=256 with WIDTH=32 SIMD parallelism, producing HLS code equivalent to
`gemini-fft.prj/kernel.cpp`.

### Pipeline structure

```
bit_rev_stage → intra_0 → intra_1 → intra_2 → intra_3 → intra_4
              → inter_5 → inter_6 → inter_7 → output_stage
```

- **`bit_rev_stage`**: Loads input in bit-reversed order into block streams
  of width 32.
- **`intra_0..4`** (5 stages): Intra-vector butterfly stages (STRIDE < WIDTH).
  Operate entirely within each 32-element SIMD block; no bank conflicts.
- **`inter_5..7`** (3 stages): Inter-vector butterfly stages (STRIDE >= WIDTH).
  Use F2-swizzled 2D buffers to avoid bank conflicts.
- **`output_stage`**: Drains block streams to the output arrays.

### Memory layout for inter-vector stages

Each inter-vector kernel uses four 2D local buffers:
```python
in_re:   float32[32, 8]   # [num_banks=WIDTH, depth=N/WIDTH]
in_im:   float32[32, 8]
out_re_b: float32[32, 8]
out_im_b: float32[32, 8]
```
With HLS pragma (injected via `s.partition`):
```cpp
#pragma HLS array_partition variable=in_re complete dim=1
```

This matches the structure in `gemini-fft.prj/kernel.cpp`.

### XOR swizzle formulas (inline in kernel)

For each inter-vector stage with stride bit `s`:
```python
bank = (idx & (WIDTH - 1)) ^ (((idx >> s) & 1) << (LOG2_WIDTH - 1))
offset = idx >> LOG2_WIDTH
```

Stage-specific:
- `inter_5`: `bank = k ^ ((i & 1) << 4)`         (stride_bit=5, STRIDE=32)
- `inter_6`: `bank = k ^ (((i >> 1) & 1) << 4)`  (stride_bit=6, STRIDE=64)
- `inter_7`: `bank = k ^ (((i >> 2) & 1) << 4)`  (stride_bit=7, STRIDE=128)

### HLS codegen verification

The generated HLS C++ code includes:
- `hls::vector<float, 32>` block-stream types
- `#pragma HLS array_partition variable=... complete dim=1`
- XOR arithmetic for conflict-free bank indexing
- `#pragma HLS pipeline II=1` on inner loops

### Tests

| Test | What it checks |
|---|---|
| `test_fft_8` | Scalar HP-FFT correctness (N=8) via simulator |
| `test_fft_256_vectorized` | Vectorized N=256 FFT correctness via simulator |
| `test_fft_256_hls_codegen` | HLS codegen has `array_partition`, `hls::vector`, XOR ops |
| `test_fft_256_csyn` | Full Vitis HLS C-synthesis (skipped if HLS not available) |

---

## 6. Scalar HP-FFT (`tests/dataflow/test_fft.py`)

A simple reference implementation using the Allo dataflow `mapping` API:
- `input_loader`: `mapping=[N]` — one PE per element, handles bit-reversal
- `butterfly`: `mapping=[LOG2_N, HALF_N]` — one PE per butterfly instance
- `output_store`: `mapping=[N]`

This serves as the functional baseline for the vectorized implementation.

---

## 7. Full Optimization Pass (`_apply_f2_optimizations`)

```python
def _apply_f2_optimizations(s):
    # 1. ARRAY_PARTITION complete dim=1 on all inter-stage 2D buffers
    _apply_f2_partitions(s)
    # 2. BIND_STORAGE ram_2p lutram + DEPENDENCE inter false on buffers
    for kn in inter_kernels:
        for bn in bufs:
            s.bind_storage(f"{kn}:{bn}", impl="lutram", storage_type="ram_2p")
            s.dependence(f"{kn}:{bn}")
    # 3. DATAFLOW on inter-stage kernels (sub-function pipeline, II=8)
    for kn in inter_kernels:
        s.dataflow(kn)
    # 4. PIPELINE II=1 on all outer loops + UNROLL on all inner k loops
    # (bit_rev, intra_0..4, inter_5..7, output_stage)
```

This achieves:
- Intra stages: II=1 (8 cycles per call, pipeline on outer `_i` loop)
- Inter stages: II=8 (dataflow sub-pipelining of load/compute/write)
- Bit-rev stage: II=256 (load) + II=8 (write)
- Overall throughput: 1 FFT per ~N*LOG2_N/WIDTH = 64 cycles

---

## Performance Target

The goal is HLS output matching `gemini-fft.prj/kernel.cpp`:
- Throughput: 1 FFT per `N * LOG2_N / WIDTH` cycles = 256 * 8 / 32 = 64 cycles
- II = 1 for all pipelined inner loops
- No bank conflicts (verified by F2 solver)
- Full `complete dim=1` partitioning on all inter-vector buffers
