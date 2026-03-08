# Allo Dataflow FFT Features

This document tracks compiler features added to enable high-performance FFT
generation targeting Vitis HLS, guided by the F2 linear layout synthesis plan.

---

## 1. F2 Linear Layout Solver (`allo/transform/f2_layout.py`)

**What it does:**
Implements Phases 1-3 of `plan.md`: given an FFT size N, vector width W, and a
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
| `apply_f2_layout(module, ...)` | MLIR pass: rewrites 1D memref to 2D with computed bank/offset indices |

**Example (N=256, WIDTH=32, STRIDE=32):**
```python
from allo.transform.f2_layout import fft_swizzle

helper = fft_swizzle(N=256, WIDTH=32, stride_bit=5)
print(helper.dims())        # (32, 8) -> 2D buffer shape [32][8]
print(helper.bank_expr("i"))
# ((i >> 0) & 1) | (((i >> 1) & 1) << 1) | ... | (((i >> 4) & 1) ^ ((i >> 5) & 1)) << 4)
```

Resulting swizzle matrix S for stride_bit=5:
```
[[1 0 0 0 0 0 0 0]
 [0 1 0 0 0 0 0 0]
 [0 0 1 0 0 0 0 0]
 [0 0 0 1 0 0 0 0]
 [0 0 0 0 1 1 0 0]]   <- bank bit 4 = addr[4] XOR addr[5]
```

This eliminates bank conflicts for STRIDE=32 butterflies: elements `idx` and
`idx XOR 32` map to different banks.

### `apply_f2_layout`: Automatic 1D-to-2D Buffer Transform

The MLIR pass `apply_f2_layout()` rewrites a 1D `memref<N x f32>` allocation
and all its uses to a 2D `memref<num_banks x depth x f32>` with computed
bank/offset indices.  Two banking modes are supported:

| Mode | Bank formula | Offset formula | Use case |
|---|---|---|---|
| `"cyclic"` | `(idx & (W-1)) ^ (xor_bit << (bank_bits-1))` | `idx >> bank_bits` | Inter-vector butterfly stages |
| `"block"` | `idx >> offset_bits` | `idx & (depth-1)` | Bit-reversal stage |

**Cyclic mode optimization:** The pass computes `xor_bit` from the offset
(`offset >> (stride_bit - bank_bits)) & 1`) instead of from the full index,
avoiding a redundant shift chain.

**Redundant index_cast elimination:** When the MLIR index value comes from an
`arith.index_cast(i32 -> index)`, the pass reuses the original i32 SSA value
instead of casting back `index -> i32`, eliminating double-cast overhead that
can confuse HLS optimization.

---

## 2. `s.f2_layout()` Schedule Primitive (`allo/customize.py`)

**What it does:**
A single schedule call that applies the full F2 bank-conflict-free transform
to a 1D buffer: rewrites to 2D, then automatically applies `partition`,
`bind_storage`, and `dependence` pragmas.

```python
@wrapped_apply
def f2_layout(self, target, n_bits, bank_bits, stride_bit=None, banking="cyclic"):
    """Apply conflict-free bank partitioning to a 1D buffer."""
    apply_f2_layout(self.module, func_name, buf_name, self.func_args,
                    n_bits, bank_bits, stride_bit, banking=banking)
    self.partition(target_buf, partition_type=Partition.Complete, dim=1)
    self.bind_storage(f"{func_name}:{buf_name}", impl="lutram", storage_type="ram_2p")
    self.dependence(f"{func_name}:{buf_name}")
```

**Usage in test_fft.py:**
```python
def _apply_f2_partitions(s):
    # Inter stages: cyclic F2 XOR swizzle
    for stage in range(3):
        stride_bit = stage + 5
        for buf in ["in_re", "in_im", "out_re_b", "out_im_b"]:
            s.f2_layout(f"inter_{stage}:{buf}",
                n_bits=LOG2_N, bank_bits=LOG2_WIDTH, stride_bit=stride_bit)

    # Bit-rev stage: block banking
    for bn in ["buf_re", "buf_im"]:
        s.f2_layout(f"bit_rev_stage_0:{bn}",
            n_bits=LOG2_N, bank_bits=LOG2_WIDTH, banking="block")
```

The generated HLS pragmas per buffer:
```cpp
#pragma HLS array_partition variable=in_re complete dim=1
#pragma HLS bind_storage variable=in_re type=ram_2p impl=lutram
#pragma HLS dependence variable=in_re inter false
```

---

## 3. Writing HLS-Friendly Index Expressions

**This is the most critical performance consideration for achieving II=1.**

When a loop is unrolled (`#pragma HLS unroll`), HLS must statically prove that
each unrolled iteration accesses a **different bank**. If it cannot, it
serializes the accesses and II degrades (e.g., II=16 instead of II=1).

### The problem: opaque index chains

A generic F2 swizzle applied to a complex linear index produces:

```
bank = (il & 31) ^ (((il >> stride_bit) & 1) << 4)
```

where `il` involves multiple shifts, masks, and ORs of loop variables. HLS sees
a long chain of dependent bit operations and **cannot simplify** this to prove
that `bank(k=0) != bank(k=1) != ... != bank(k=15)`.

### The solution: expose the unrolled variable in bank-select bits

Restructure the index so the unrolled loop variable `k` directly occupies the
lower bank-select bits **by construction**:

```python
# COMPUTE loop: k is unrolled (0..15), i is the pipelined outer loop
raw_bank: uint32 = k | ((i & 1) << 4)    # bits [4:0] = {i&1, k[3:0]}
il: uint32 = (offset << LOG2_WIDTH) | raw_bank
```

Now when the F2 transform computes `bank = il & 31`:

```
il & 31 = ((offset << 5) | raw_bank) & 31
        = raw_bank & 31
        = raw_bank
        = k | ((i & 1) << 4)
```

Since `k` is the unrolled variable (0..15), the lower 4 bits are just `k` --
**trivially distinct** across unrolled iterations. HLS proves this with basic
bit-level reasoning.

The XOR swizzle only flips bit 4: `bank = raw_bank ^ (xor_bit << 4)`.
Bits [3:0] remain `k`, so distinctness is preserved:

```
bank[3:0] = k            <- distinct for k=0..15
bank[4]   = (i&1) ^ xor_bit  <- same across all k values
```

### Rules for writing bank-conflict-free kernels

1. **The unrolled loop variable must be the bank index (lower bits).**
   Write `idx = (offset << LOG2_WIDTH) | k`, not `idx = offset * WIDTH + k`
   followed by `bank = idx & (WIDTH-1)`.

2. **Separate offset computation from bank computation.** Compute the offset
   from the outer loop variable, then OR in the bank bits. Don't mix them into
   a single arithmetic expression.

3. **Use `uint32` types for bitwise index expressions.** This forces the Allo
   builder to use `memref.load/store` (non-affine) instead of `affine.load/store`,
   which is required for `<<`, `|`, `&`, `^` operations. The LOAD/WRITE loops
   can still use affine-compatible `i * WIDTH + k` since they access sequentially.

4. **Avoid redundant temporaries.** Each `x: int32 = expr` in Allo creates a
   `memref<i32>` alloca + store + load. Subsequent operations on the loaded
   value may get sign-extended to `i64` by MLIR. Minimize intermediate variables
   and prefer direct expressions with loop variables where possible.

### Performance impact

| Index pattern | HLS bank proof | Achieved II |
|---|---|---|
| `bank = (complex_linear_idx & 31) ^ (...)` | Cannot prove | II=16 |
| `bank = k \| ((i & 1) << 4)` (k in lower bits) | Trivially provable | II=1 |

---

## 4. `_build_top` Deduplication Fix (`allo/dataflow.py`)

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

## 5. Other Schedule Primitives (`allo/customize.py`)

### `s.bind_storage(target, impl, storage_type="")`

Emits `#pragma HLS bind_storage variable=<buf> type=<storage_type> impl=<impl>`.

| Argument | Example values |
|---|---|
| `impl` | `"bram"`, `"uram"`, `"lutram"`, `"srl"` |
| `storage_type` | `""` (auto), `"ram_1p"`, `"ram_2p"`, `"ram_t2p"`, `"ram_s2p"` |

`ram_2p` enables dual-port access, required for HLS DATAFLOW ping-pong
buffering (one port writes new data while another reads previous call's data).

### `s.dependence(target, dep_type="inter", direction="false")`

Emits `#pragma HLS dependence variable=<buf> inter false`.

Suppresses conservative false loop-carried dependency analysis by HLS, enabling
II=1 for loops that compute butterfly indices via XOR expressions that HLS cannot
easily prove are conflict-free.

### `s.partition_global(name_prefix)`

Marks all `memref.GlobalOp` constants whose `sym_name` starts with the given
prefix for complete array partitioning.  Used for twiddle ROMs (`twr`, `twi`).

---

## 6. `pipeline_outer` Mode in `create_data_movement` (`allo/ir/transform.py`)

**Problem:**
The default `create_data_movement` emits a single loop nest where the innermost
loop is pipelined.  For a 2D array `[8][32]`, this generates a 256-iteration
pipelined loop (II=1, 256 cycles), not the desired 8-iteration pipelined outer
loop with 32-way unrolled inner (8 cycles).

**Fix:**
A 4-element mapping tuple `(loop_bounds, src_pattern, dst_pattern, True)` enables
`pipeline_outer` mode: the outer loop gets `pipeline_ii=1 rewind` and the
innermost loop gets `unroll` (full unroll, factor=0).

**Important:** `unroll` must be `IntegerAttr(UInt32, 0)`, **not** `UnitAttr`.
`EmitVivadoHLS.cpp` calls `dyn_cast<IntegerAttr>(factor).getValue()` -- a
`UnitAttr` causes a null-dereference segfault.

---

## 7. Vectorized FFT-256 (`tests/dataflow/test_fft.py`)

### Architecture

The frontend uses **plain 1D arrays** (`float32[N]`) inside kernels. The
compiler's `s.f2_layout()` transform automatically rewrites them to 2D
bank-conflict-free layouts for HLS.

### Pipeline structure

```
bit_rev_stage -> intra_0 -> intra_1 -> intra_2 -> intra_3 -> intra_4
              -> inter_0 -> inter_1 -> inter_2 -> output_stage
```

- **`bit_rev_stage`**: 1D `float32[N]` buffers, inline 8-bit reversal.
  Compiler transforms to 2D block-banked layout (`bank = addr >> 3`).
- **`intra_0..4`** (5 stages): Intra-vector butterflies (STRIDE < WIDTH).
  Operate within each 32-element chunk; no bank conflicts.
  Trivial/minus-j butterfly elimination (`tw_k == 0` / `tw_k == 64`) saves
  35 DSPs across intra stages.
- **`inter_0..2`** (3 stages): Inter-vector butterflies (STRIDE >= WIDTH).
  1D `float32[N]` buffers with restructured index expressions (see Section 3).
  Compiler transforms to 2D cyclic F2 XOR-swizzled layout.
- **`output_stage`**: Drains final stream to output.

### I/O interface

Top-level uses `Stream[float32[WIDTH], 2]` (block streams of 32 floats).
No `load_buf`/`store_res` wrappers needed.

### Inter-vector kernel structure (1D frontend code)

```python
@df.kernel(mapping=[3])
def inter():
    s_rel = df.get_pid()
    in_re: float32[N]       # 1D -- compiler transforms to float32[32, 8]
    out_re_b: float32[N]

    # LOAD: affine-compatible indexing (i * WIDTH + k)
    for i in range(NUM_VECS):
        for k in range(WIDTH):
            in_re[i * WIDTH + k] = chunk_re[k]

    # COMPUTE: restructured index with k in lower bits
    for i in range(NUM_VECS):
        for k in range(16):
            raw_bank: uint32 = k | ((i & 1) << 4)
            il: uint32 = (off_il << LOG2_WIDTH) | raw_bank
            iu: uint32 = (off_iu << LOG2_WIDTH) | raw_bank
            # ... butterfly using in_re[il], in_re[iu], etc.

    # WRITE: affine-compatible indexing (i * WIDTH + k)
    for i in range(NUM_VECS):
        for k in range(WIDTH):
            chunk_re_out[k] = out_re_b[i * WIDTH + k]
```

Key design choices:
- LOAD/WRITE use `i * WIDTH + k` (affine-compatible: `*` and `+` are valid
  affine ops in MLIR). This avoids `AffineDimExpr` errors from `<<` and `|`.
- COMPUTE uses `uint32` types with `<<`, `|`, `&` (forces non-affine
  `memref.load/store` fallback), with `k` in the lower bits for bank
  distinctness.

### Bit-reversal kernel (1D frontend code)

```python
@df.kernel(mapping=[1])
def bit_rev_stage():
    buf_re: float32[N]      # 1D -- compiler transforms to float32[32, 8]

    for ii in range(NUM_VECS):
        for kk in range(WIDTH):
            idx: uint32 = (ii << LOG2_WIDTH) | kk
            rev: uint32 = <8-bit reversal of idx>
            buf_re[rev] = chunk_in_re[kk]

    for jj in range(NUM_VECS):
        for mm in range(WIDTH):
            chunk_re[mm] = buf_re[jj * WIDTH + mm]
```

Block banking (`bank = addr >> 3`, `offset = addr & 7`) works because
bit-reversal scatters writes across all 32 banks (a bijection), and the
sequential read phase accesses contiguous blocks.

### Tests

| Test | What it checks |
|---|---|
| `test_fft_8` | Scalar HP-FFT correctness (N=8) via simulator |
| `test_fft_256_vectorized` | HLS codegen: streams, vectors, partitions, XOR ops |
| `test_fft_256_hls_codegen` | Full HLS pragma verification (all pragmas present) |
| `test_fft_256_simulator` | Functional correctness via LLVM simulator |
| `test_fft_256_csyn` | Full Vitis HLS C-synthesis (skipped if HLS not available) |

---

## 8. Full Optimization Pass (`_apply_f2_optimizations`)

```python
def _apply_f2_optimizations(s):
    # 1. F2 layout: 1D -> 2D with partition + bind_storage + dependence
    _apply_f2_partitions(s)
    # 2. Partition twiddle ROMs for parallel access from unrolled k-loops
    s.partition_global("twr")
    s.partition_global("twi")
    # 3. DATAFLOW on inter stages + bit_rev (sub-function pipeline)
    for kn in ["inter_0", "inter_1", "inter_2"]:
        s.dataflow(kn)
    s.dataflow("bit_rev_stage_0")
    # 4. PIPELINE II=1 on outer loops + UNROLL on inner k loops
    # (bit_rev, intra_0..4, inter_0..2, output_stage)
```

---

## 9. HLS Codegen Fixes

### `ap_int<65>` -> `int64_t` (`mlir/lib/Translation/EmitVivadoHLS.cpp`)

MLIR conservatively widens signed integer arithmetic: `i64 + i32` produces an
`i65` result type to prevent overflow.  The previous HLS emitter mapped any
non-standard-width integer to `ap_int<N>`, generating `ap_int<65>` for all
index arithmetic -- expensive in HLS (uses 2 DSPs and is slow to synthesize).

**Fix**: integers with width 33-65 are now emitted as `int64_t`/`uint64_t`.

### `disable_start_propagation` on DATAFLOW functions

Function-level `#pragma HLS dataflow` now always includes
`disable_start_propagation`, matching the reference `kernel.cpp` behavior.

### Redundant zero-initializer removal

All `= 0` initializers on arrays that are fully written before being read have
been removed. These create init loops inside `#pragma HLS pipeline` loops,
preventing II=1.

---

## 10. Synthesis Results (Vitis HLS 2023.2)

Target device: `xcvp1802-lsvc4072-3HP-e-S` (Versal Premium)

| Metric | Value |
|---|---|
| **Latency** | **94 cycles** |
| **Interval** | **19 cycles** |
| **DSP** | 328 (2%) |
| **BRAM** | 0 |
| **FF** | 262,072 (3%) |
| **LUT** | 276,402 (8%) |

All pipelined loops achieve II=1. Inter-stage compute loops achieve II=1
thanks to the restructured index expressions (Section 3) and F2 XOR swizzle.

Per-stage breakdown:
- `bit_rev_stage_0`: 26 cycles (dataflow: load 10 + write 10, overlapped)
- `intra_0..1`: 14 cycles each (no twiddle multiply)
- `intra_2..4`: 18 cycles each (with twiddle multiply, 32/48/56 DSPs)
- `inter_0..2`: 40 cycles each (dataflow: load 10 + compute 18 + write 10)
- `output_stage_0`: 10 cycles

HLS project: `tests/dataflow/fft_hls_prj/fft_256_prj/`

---

## 11. Auto-F2: Automatic Bank-Conflict-Free Partitioning

### Overview

`s.auto_f2()` is a schedule primitive that **automatically detects** bank
conflict patterns in 1D buffers and applies the correct F2 layout transform
— eliminating the need for manual `s.f2_layout()` calls.

The engine uses **symbolic execution over F2** (the binary field) to trace
index expressions as affine functions of loop variables. This is a general
technique that works for any kernel with power-of-2-sized buffers and bitwise
index expressions (not limited to FFT).

### Key Files

| File | Description |
|---|---|
| `allo/transform/f2_symbolic.py` | F2Symbol class and conflict subspace builder |
| `allo/transform/auto_f2.py` | MLIR SSA walker and auto-F2 analysis pass |
| `allo/customize.py` | `s.auto_f2()` schedule primitive |

### Usage

```python
s = df.customize(fft_256)

# Before: 14+ manual calls
# s.f2_layout("inter_0:in_re", n_bits=8, bank_bits=5, stride_bit=5)
# s.f2_layout("inter_0:in_im", n_bits=8, bank_bits=5, stride_bit=5)
# ... 12 more calls ...
# s.f2_layout("bit_rev_stage_0:buf_re", n_bits=8, bank_bits=5, banking="block")
# s.f2_layout("bit_rev_stage_0:buf_im", n_bits=8, bank_bits=5, banking="block")

# After: single call
s.auto_f2()  # analyzes all kernels, detects all 14 buffers automatically
```

### How It Works

#### 1. F2 Symbolic Execution (`f2_symbolic.py`)

Each value is represented as an affine function over F2:

$$\text{addr\_bits}[i] = \bigoplus_j A[i][j] \cdot \text{input\_bits}[j] \oplus c[i]$$

The `F2Symbol` class tracks this matrix `A` and constant `c` through
operations:

| Operation | F2 Linear? | Symbolic Handling |
|---|---|---|
| `a ^ b` (XOR) | Yes | Row-wise XOR of matrices |
| `a << k` (constant shift) | Yes | Shift rows |
| `a >> k` (constant shift) | Yes | Shift rows |
| `a & constant` | Yes | Zero out rows |
| `a \| b` (non-overlapping) | Yes | Same as XOR |
| `a + b` (non-overlapping) | Yes | Same as XOR (no carries) |
| `a * power_of_2` | Yes | Same as shift |
| Non-linear ops | No | Conservative opaque fallback |

#### 2. Conflict Subspace Construction

Two sources of parallel accesses that cause bank conflicts:

**Source 1 — Intra-iteration**: Multiple load/store ops in the same loop body
are always simultaneous (even in pipelined loops). The XOR of their symbolic
addresses gives conflict vectors.

**Source 2 — Inter-iteration**: When a loop is unrolled, all iterations execute
simultaneously. The columns of the access matrix at the unrolled variable's
bit positions are conflict vectors.

$$P = \text{span}(V_{\text{intra}} \cup V_{\text{inter}})$$

#### 3. MLIR SSA Walker (`auto_f2.py`)

`symbolize_mlir_index()` walks the MLIR SSA def-use chain backwards from each
array index, building the F2Symbol. It handles:
- `arith.constant`, `arith.xori`, `arith.shli`, `arith.shrui`, `arith.andi`
- `arith.ori` (checks non-overlapping), `arith.addi` (checks non-overlapping)
- `arith.muli` (power-of-2 constant), `arith.index_cast`, `arith.ext*`
- Allo scalar variable pattern: `memref.alloc + affine.store + affine.load`
  (traces through to the stored value)

#### 4. Banking Mode Detection

The banking mode emerges from the conflict subspace structure:
- **Cyclic + XOR swizzle**: Conflict bits span lower bits plus one high stride
  bit (e.g., FFT inter-stage buffers)
- **Block**: All conflict bits are in the upper bit positions (e.g., bit-reversal
  buffers)

### Example: Auto-detection on FFT-256

For `inter_0:in_re` (float32[256]), the engine:

1. Finds `memref.load %buf[%il]` and `memref.load %buf[%iu]` in the compute loop
2. Symbolizes: `il = (off_il << 5) | (k | ((i & 1) << 4))`, `iu = (off_iu << 5) | bank`
   where `off_iu = off_il | (1 << s_rel)`
3. V_inter from k bits (0-3): {e_0, e_1, e_2, e_3}
4. V_intra from delta(il, iu): stride at bit 5 → {e_5}
5. **P = span(e_0, e_1, e_2, e_3, e_5)**, dim=5 → needs 32 banks
6. Solver: S with bank[4] = addr[4] XOR addr[5] → cyclic, stride_bit=5

This matches the manual `s.f2_layout(..., stride_bit=5)` exactly.

For `bit_rev_stage_0:buf_re`, the engine detects block banking (all conflict
bits in upper positions) and applies `banking="block"` automatically.

### Common Pattern: Cyclic Partition

The engine also handles standard cyclic partition patterns:

```python
for i in range(N):        # pipelined, II=1
    A[i * 2] = ...        # addr = i << 1
    A[i * 2 + 1] = ...    # addr = (i << 1) | 1
```

Conflict subspace: P = span(e_0), dim=1 → bank = addr[0] = **cyclic factor 2**.

### Tests

| Test | What it checks |
|---|---|
| `test_fft_256_auto_f2` | Auto-F2 produces same HLS pragmas as manual f2_layout |
| `f2_symbolic.py __main__` | 53 unit tests for F2Symbol operations and conflict subspace |
