# Auto-F2 Layout via Grid Folding: Design Plan

## Goal

Users write **simple, fully-spatial kernel code** (one PE per operation) and
specify **folding factors** to control how many PEs are merged into temporal
loops. The compiler automatically:

1. **Folds** grid dimensions into loops (which are then unrolled for spatial parallelism)
2. **Symbolically executes** index expressions over F2 to build access matrices
3. **Constructs** the conflict subspace P from all parallel accesses
4. **Solves** for the partition matrix S using the F2LayoutSolver
5. **Applies** the 1D→2D layout transform with correct banking and emits all HLS pragmas

Users never write `s.f2_layout()` — the compiler derives everything from the
access pattern and folding factors. The technique is **general**, not limited
to FFT — any kernel with power-of-2-sized buffers and bitwise index expressions.

---

## Conceptual Framework (from Linear Layouts paper + plan.md)

### F2 Linear Layout

A **linear layout** is a matrix $L \in \mathbb{F}_2^{m \times n}$ that maps
hardware resource bits to logical tensor coordinate bits. In HLS:

- Input space = (Bank, Offset, TimeStep) bits
- Output space = logical array address bits
- **Bank** = which physical memory bank (partitioned)
- **Offset** = address within that bank
- **TimeStep** = which cycle within the pipeline (folded iteration)

### Grid Folding = Controlling the TimeStep Dimension

In the scalar FFT:
```python
@df.kernel(mapping=[LOG2_N, HALF_N])
def butterfly():
    s, b = df.get_pid()   # HALF_N separate PEs for b
```

With **folding factor F=2** on dimension 1:
```python
# Compiler generates:
@df.kernel(mapping=[LOG2_N, HALF_N // 2])
def butterfly():
    s, b_outer = df.get_pid()
    for b_inner in range(2):  # unrolled → 2 parallel accesses
        b = b_outer * 2 + b_inner
        # ... original butterfly body using b ...
```

The F=2 iterations execute **simultaneously** (unrolled). The F2 analysis
ensures their buffer accesses map to different banks.

---

## Core Engine: Symbolic Execution over F2

### Why Symbolic Execution?

Pattern-matching (looking for XOR, stride, bit-reversal) is fragile and
application-specific. Instead, we need a **general** approach that works for
any index expression built from bitwise operations.

Key observation: all array sizes are powers of 2, and HLS index expressions
use operations that are **linear over F2** (the field {0,1} with XOR as
addition). We can symbolically trace each index expression as an affine
function over F2 bit vectors:

$$\text{addr\_bits} = A \cdot \text{input\_var\_bits} \oplus c$$

where $A$ is a binary matrix and $c$ is a constant vector. The columns of $A$
corresponding to the **parallel (unrolled) loop variable** directly reveal
which address bits vary across simultaneous accesses — **this IS the conflict
subspace**, with no heuristics needed.

### F2 Linearity of Common Operations

| Operation | Linear over F2? | F2 Representation |
|---|---|---|
| `a ^ b` (XOR) | Yes | Row-wise XOR of matrices |
| `a << k` (shift left by constant) | Yes | Shift rows up, zero-fill |
| `a >> k` (shift right by constant) | Yes | Shift rows down, truncate |
| `a & constant_mask` | Yes | Zero out rows where mask bit = 0 |
| `a \| b` (non-overlapping bits) | Yes | Same as XOR when bits don't overlap |
| `a + b` (non-overlapping bits) | Yes | Same as XOR (no carries) |
| `a * power_of_2` | Yes | Same as left shift |
| `a + b` (overlapping bits) | **No** | Carry propagation breaks linearity |
| `a * non_power_of_2` | **No** | Distributes into shifts + adds with carries |
| `a & b` (both symbolic) | **No** | AND = multiplication in F2 (bilinear) |

In practice, well-written HLS kernels use the linear subset for index
computation (shifts, masks, XOR, non-overlapping OR), making the F2
representation exact. When non-linear ops are encountered, we conservatively
treat the result as **opaque** (all output bits independent of tracked inputs).

### `F2Symbol` Class

```python
class F2Symbol:
    """Symbolic bit vector over F2.

    Represents a value as an affine function:
        bits[i] = (sum_j matrix[i][j] * input_var_bits[j]) XOR constant[i]

    where all arithmetic is mod 2 (XOR for +, AND for *).
    """
    def __init__(self, matrix, constant, n_bits):
        self.matrix = matrix     # numpy array, shape (n_bits, n_input_vars), dtype=uint8
        self.constant = constant # numpy array, shape (n_bits,), dtype=uint8
        self.n_bits = n_bits

    @classmethod
    def variable(cls, var_id, var_bits, total_input_bits):
        """Create symbol for a loop variable occupying specific input bit positions."""
        # Identity columns at the variable's bit positions
        ...

    @classmethod
    def constant_val(cls, value, n_bits, total_input_bits):
        """Create symbol for a compile-time constant."""
        # Zero matrix, constant = bit decomposition of value
        ...

    def xor(self, other):
        """a ^ b: row-wise XOR of matrices and constants."""
        return F2Symbol(self.matrix ^ other.matrix, self.constant ^ other.constant, ...)

    def shift_left(self, k):
        """a << k: shift rows up by k, zero-fill bottom k rows."""
        new_matrix = np.zeros_like(self.matrix)
        new_matrix[k:] = self.matrix[:-k] if k > 0 else self.matrix
        new_const = np.zeros_like(self.constant)
        new_const[k:] = self.constant[:-k] if k > 0 else self.constant
        return F2Symbol(new_matrix, new_const, ...)

    def shift_right(self, k):
        """a >> k: shift rows down by k, truncate top k rows."""
        ...

    def and_constant(self, mask):
        """a & constant: zero out rows where mask bit is 0."""
        for i in range(self.n_bits):
            if not (mask >> i) & 1:
                new_matrix[i] = 0
                new_const[i] = 0
        ...

    def or_nonoverlapping(self, other):
        """a | b where a and b have non-overlapping bits.
        Equivalent to XOR when bits don't overlap.
        Verifiable: (self.matrix[i] & other.matrix[i]).any() == False for all i.
        """
        return self.xor(other)  # Safe when non-overlapping

    def truncate(self, n_bits):
        """Keep only the lower n_bits."""
        ...

    def conflict_columns(self, var_bit_indices):
        """Extract columns corresponding to specific input variable bits.

        Returns the sub-matrix A[:, var_bit_indices] — these columns span
        the set of address-bit differences when the variable ranges over
        all values. This IS the conflict subspace basis.
        """
        return self.matrix[:, var_bit_indices]
```

### Symbolic Execution of MLIR SSA Chains

```python
def symbolize_index(ssa_value, loop_var_map, n_addr_bits, n_input_bits):
    """Convert an MLIR SSA value to an F2Symbol.

    Args:
        ssa_value: MLIR SSA value (the array index)
        loop_var_map: {ssa_value: (var_id, start_bit, n_bits)}
            Maps loop induction variables to their bit positions in the input space
        n_addr_bits: Number of address bits (log2 of array size)
        n_input_bits: Total input variable bits

    Returns:
        F2Symbol representing the index expression
    """
    if ssa_value in loop_var_map:
        var_id, start_bit, n_bits = loop_var_map[ssa_value]
        return F2Symbol.variable(start_bit, n_bits, n_input_bits, n_addr_bits)

    op = ssa_value.owner

    if is_constant_op(op):
        return F2Symbol.constant_val(get_constant(op), n_addr_bits, n_input_bits)

    if is_xori_op(op):  # arith.xori
        lhs = symbolize_index(op.operands[0], ...)
        rhs = symbolize_index(op.operands[1], ...)
        return lhs.xor(rhs)

    if is_shli_op(op):  # arith.shli by constant
        val = symbolize_index(op.operands[0], ...)
        shift = get_constant(op.operands[1])
        return val.shift_left(shift)

    if is_shrui_op(op):  # arith.shrui by constant
        val = symbolize_index(op.operands[0], ...)
        shift = get_constant(op.operands[1])
        return val.shift_right(shift)

    if is_andi_op(op):  # arith.andi with constant mask
        if is_constant(op.operands[1]):
            val = symbolize_index(op.operands[0], ...)
            mask = get_constant(op.operands[1])
            return val.and_constant(mask)
        # AND of two symbolic values → non-linear → opaque
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    if is_ori_op(op):  # arith.ori
        lhs = symbolize_index(op.operands[0], ...)
        rhs = symbolize_index(op.operands[1], ...)
        if lhs.is_nonoverlapping(rhs):  # Verify non-overlapping
            return lhs.or_nonoverlapping(rhs)
        # Overlapping OR → non-linear → opaque
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    if is_index_cast_op(op):  # arith.index_cast
        return symbolize_index(op.operands[0], ...)

    if is_addi_op(op):  # arith.addi
        lhs = symbolize_index(op.operands[0], ...)
        rhs = symbolize_index(op.operands[1], ...)
        if lhs.is_nonoverlapping(rhs):
            return lhs.or_nonoverlapping(rhs)  # add = xor when no carries
        # Overlapping add → non-linear → opaque
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    if is_muli_op(op):  # arith.muli by constant
        if is_constant(op.operands[1]):
            c = get_constant(op.operands[1])
            if c & (c - 1) == 0:  # power of 2
                val = symbolize_index(op.operands[0], ...)
                return val.shift_left(int(np.log2(c)))
        return F2Symbol.opaque(n_addr_bits, n_input_bits)

    # Unknown op → opaque
    return F2Symbol.opaque(n_addr_bits, n_input_bits)
```

### Conflict Subspace Construction

There are **two distinct sources** of parallel memory accesses that can cause
bank conflicts. The conflict subspace P must account for both:

#### Source 1: Intra-iteration parallelism (multiple accesses in the same loop body)

Even in a **pipelined** (non-unrolled) loop, all memory accesses within the
same iteration body execute in the same clock cycle. If a loop body contains
`A[i*2]` and `A[i*2+1]`, these two accesses are always simultaneous.

```python
for i in range(N):     # pipelined, II=1
    A[i*2] = ...       # access 1 } both in same cycle
    A[i*2+1] = ...     # access 2 }
```

The conflict vector is the XOR of their symbolic addresses:
$\delta = \text{sym}(\texttt{i*2}) \oplus \text{sym}(\texttt{i*2+1})$

For this example: `i*2 = i << 1` has constant=`[0,0,...]`, `i*2+1 = (i<<1)|1`
has constant=`[1,0,...]`. So $\delta_{\text{const}} = e_0$, meaning bit 0
always differs. This contributes $e_0$ to the conflict subspace.

#### Source 2: Inter-iteration parallelism (from unrolling or folding)

When a loop is **unrolled** (or created by grid folding), multiple iterations
execute simultaneously. The address varies with the unrolled loop variable,
and all values of that variable produce concurrent accesses.

```python
for i in range(2):     # UNROLLED → i=0 and i=1 execute simultaneously
    A[i*2] = ...       # i=0: A[0], i=1: A[2]  ← must be different banks
    A[i*2+1] = ...     # i=0: A[1], i=1: A[3]  ← must be different banks
```

The conflict vectors come from the columns of the access matrix A
corresponding to the unrolled variable bits: how the address changes when
the unrolled variable bit flips.

For `i*2 = i << 1`: column for i[0] is $e_1$. This means flipping i changes
addr bit 1. So $e_1 \in P$.

#### Combined algorithm

```python
def build_conflict_subspace(func_op, alloc_op, loop_var_map, parallel_var_bits):
    """Build the conflict subspace P for a buffer.

    The conflict subspace captures ALL address differences between accesses
    that happen in the SAME clock cycle. Two sources of parallelism:

    1. Intra-iteration: multiple load/store ops in the same loop body
       (always parallel, even in pipelined loops)
    2. Inter-iteration: unrolled loop iterations that execute simultaneously
       (from folding or explicit s.unroll())

    Args:
        func_op: The kernel function
        alloc_op: The 1D buffer allocation
        loop_var_map: maps SSA values → (var_id, start_bit, n_bits)
        parallel_var_bits: input bit indices for unrolled/folded variables
            (empty list if the loop is pipelined, not unrolled)

    Returns:
        P: numpy array — basis vectors of conflict subspace (columns)
    """
    accesses = collect_load_store_ops(alloc_op)
    symbols = []
    conflict_vectors = []

    # Symbolize all accesses
    for acc in accesses:
        sym = symbolize_index(get_index(acc), loop_var_map, n_addr_bits, n_input_bits)
        symbols.append(sym)

    # SOURCE 1: Intra-iteration conflicts
    # Any two accesses in the same loop body execute simultaneously.
    # Their address difference (over F2) is a conflict vector.
    for (sym_a, sym_b) in all_pairs(symbols):
        # Constant part: fixed address difference
        delta_const = sym_a.constant ^ sym_b.constant
        if delta_const.any():
            conflict_vectors.append(delta_const)

        # Matrix part: variable-dependent address differences.
        # For each input variable bit, if the two accesses depend on it
        # differently, that bit contributes a conflict vector.
        delta_matrix = sym_a.matrix ^ sym_b.matrix
        for col_idx in range(delta_matrix.shape[1]):
            col = delta_matrix[:, col_idx]
            if col.any():
                conflict_vectors.append(col)

    # SOURCE 2: Inter-iteration conflicts (only if loop is unrolled)
    # When parallel_var_bits is non-empty, different values of the unrolled
    # variable execute simultaneously. The address variation across these
    # values must map to different banks.
    if parallel_var_bits:
        for sym in symbols:
            for bit_idx in parallel_var_bits:
                col = sym.matrix[:, bit_idx]
                if col.any():
                    conflict_vectors.append(col)

    # Build basis of P by row-reducing (Gaussian elimination over F2)
    if conflict_vectors:
        P = row_reduce_f2(np.column_stack(conflict_vectors))
    else:
        P = np.zeros((n_addr_bits, 0), dtype=np.uint8)
    return P
```

#### Worked example: `for i in range(2): A[i*2]=...; A[i*2+1]=...`

**Case A: Loop is pipelined (not unrolled)**

`parallel_var_bits = []` (no inter-iteration parallelism)

Source 1 only:
- sym(i*2):   matrix=`[[0],[1]]`, const=`[0,0]`
- sym(i*2+1): matrix=`[[0],[1]]`, const=`[1,0]`
- delta_const = `[1,0]` = e_0 → **P = span(e_0)**, dim=1

With 2 banks (bank_bits=1): S = `[[1, 0]]` → bank = addr[0].
This is **cyclic partitioning with factor 2**: even indices in bank 0,
odd indices in bank 1. Each cycle writes one even and one odd address — no conflict.

**Case B: Loop is unrolled (folding factor = 2)**

`parallel_var_bits = [0]` (i has 1 bit at position 0)

Source 1 (same as above): delta_const = e_0
Source 2: column for i[0] in sym(i*2) = `[0,1]` = e_1
          column for i[0] in sym(i*2+1) = `[0,1]` = e_1

**P = span(e_0, e_1)**, dim=2

With 2 banks: dim(P)=2 > bank_bits=1 → **impossible!** Need 4 banks.
With 4 banks: S = `[[1,0],[0,1]]` → bank = addr[1:0] → complete partitioning.
This correctly reflects that 4 simultaneous accesses (A[0], A[1], A[2], A[3])
need 4 distinct banks.

This distinction between pipelined and unrolled is critical and emerges
naturally from whether `parallel_var_bits` is empty or not.

### From Conflict Subspace to Partition Matrix

```python
def solve_partition(P, n_addr_bits, n_banks):
    """Solve for partition matrix S given conflict subspace P.

    S must satisfy: for all v in P, v != 0 → S·v != 0
    i.e., ker(S) ∩ P = {0}

    Args:
        P: basis of conflict subspace (n_addr_bits × dim_P matrix)
        n_addr_bits: log2(buffer_size)
        n_banks: number of memory banks (= 2^bank_bits)

    Returns:
        S: partition matrix (bank_bits × n_addr_bits), rows define bank index
        mode: "cyclic" or "block"
    """
    bank_bits = int(np.log2(n_banks))

    # Feasibility check: dim(P) must be <= bank_bits
    # (need enough banks to separate all conflicting accesses)
    if P.shape[1] > bank_bits:
        raise ValueError(
            f"Conflict subspace dim={P.shape[1]} > bank_bits={bank_bits}. "
            f"Need more banks or reduce folding factor."
        )

    # Use F2LayoutSolver with the conflict subspace basis as stride_bits
    # The solver finds S such that S @ v != 0 for all v in P
    solver = F2LayoutSolver(n_addr_bits, bank_bits)

    # Extract stride bits from P's basis vectors
    # Each basis vector of P that is a unit vector e_s gives stride_bit = s
    # Non-unit vectors require the generalized solver
    stride_bits = extract_stride_bits_from_subspace(P)

    return solver.solve(stride_bits)
```

### Banking Mode Detection (General)

Instead of hardcoded "cyclic" vs "block" heuristics, the banking mode emerges
naturally from the conflict subspace structure:

- **Cyclic banking** (addr lower bits = bank): The conflict subspace P has
  basis vectors concentrated in the lower bit positions. The swizzle matrix
  S starts as identity on lower bits and adds XOR corrections.

- **Block banking** (addr upper bits = bank): The conflict subspace P spans
  the full address space, but the *write* access pattern is a permutation
  (bijection). Block banking assigns `bank = addr >> offset_bits`.

Detection: if the write-side access matrix has full column rank (every input
combination produces a unique address), and the conflict subspace would
require more banks than available under cyclic mode, try block banking.

---

## User-Facing API

### Before (manual, current):
```python
# User writes complex vectorized kernels with hand-crafted index expressions
@df.kernel(mapping=[3])
def inter():
    s_rel = df.get_pid()
    in_re: float32[N]
    # ... 60 lines of careful index arithmetic ...

# User manually specifies F2 layout for each buffer
s.f2_layout("inter_0:in_re", n_bits=8, bank_bits=5, stride_bit=5)
# ... 14 more calls ...
```

### After (auto, proposed):
```python
# User writes simple scalar kernel
@df.kernel(mapping=[LOG2_N, HALF_N])
def butterfly():
    s, b = df.get_pid()
    upper = get_upper_idx(s, b)
    lower = get_lower_idx(s, b)
    # ... simple butterfly using stage[s, upper/lower] ...

# User specifies folding factors only
s.fold("butterfly", dim=1, factor=WIDTH)
# Compiler automatically:
#   1. Creates WIDTH-iteration inner loop (unrolled)
#   2. Symbolic-executes index expressions over F2
#   3. Builds conflict subspace from parallel access matrices
#   4. Solves for optimal partition matrix S
#   5. Transforms 1D arrays to 2D with S-derived bank/offset indexing
#   6. Emits partition + bind_storage + dependence pragmas
```

---

## Implementation Plan

### Step 1: F2 Symbolic Execution Engine (`allo/transform/f2_symbolic.py`)

Core class `F2Symbol` and MLIR SSA walker `symbolize_index()`.

**Test strategy**: Unit tests with known index expressions:
```python
# Test: il = (off << 5) | (k | ((i & 1) << 4))
# With k as folded var (5 bits), expect conflict columns = identity on bits 0-4
sym = symbolize_index(il_ssa, {k_ssa: (0, 0, 5), i_ssa: (1, 5, 3)}, 8, 8)
assert sym.conflict_columns([0,1,2,3,4]) == I_5  # identity
```

### Step 2: Conflict Subspace Builder (`allo/transform/auto_f2.py`)

Uses `F2Symbol` to analyze all accesses to each buffer, builds P, solves S.

### Step 3: Schedule Primitives (`allo/customize.py`)

- `s.auto_f2(kernel_names=None)`: Auto-partition all 1D buffers
- `s.fold(kernel, dim, factor)`: Fold grid dimension, then auto-partition

### Step 4: Update test_fft.py

Replace `_apply_f2_partitions(s)` with `s.auto_f2()`, verify same results.

### Step 5: Update FEATURES.md

---

## Worked Example: FFT Butterfly with Folding

### Input
```python
@df.kernel(mapping=[LOG2_N, HALF_N])  # mapping=[8, 128]
def butterfly():
    s, b = df.get_pid()
    upper = get_upper_idx(s, b)   # = group*2*stride + pos
    lower = get_lower_idx(s, b)   # = upper + stride
    a = stage[s, upper].get()
    c = stage[s, lower].get()
    # ... butterfly ...
    stage[s+1, upper].put(a + c*tw)
    stage[s+1, lower].put(a - c*tw)

s.fold("butterfly", dim=1, factor=16)
```

### After folding (compiler generates):
```python
@df.kernel(mapping=[LOG2_N, HALF_N // 16])  # mapping=[8, 8]
def butterfly():
    s, b_outer = df.get_pid()
    for b_inner in range(16):  # unrolled: 16 parallel accesses
        b = b_outer * 16 + b_inner
        upper = get_upper_idx(s, b)
        lower = get_lower_idx(s, b)
        a = stage[s, upper].get()
        c = stage[s, lower].get()
        stage[s+1, upper].put(a + c*tw)
        stage[s+1, lower].put(a - c*tw)
```

### Symbolic execution (for fixed stage s=5, stride=32):

Input variables: `b_inner` (4 bits: positions 0-3), `b_outer` (3 bits: positions 4-6)

```
b = b_outer * 16 + b_inner
  = (b_outer << 4) | b_inner       (non-overlapping, linear)
  → A_b = I_7 (identity on all 7 input bits)

upper(s=5, b):
  group = b >> 5 = b >> s           → bits [6:5] of b → input bits 5,6
  pos   = b & 31 = b & (stride-1)  → bits [4:0] of b → input bits 0-4
  upper = group * 64 + pos          → (group << 6) | pos

  → upper_sym.matrix (8 addr bits × 7 input bits):
      addr[0] = b_inner[0]    (input bit 0)
      addr[1] = b_inner[1]    (input bit 1)
      addr[2] = b_inner[2]    (input bit 2)
      addr[3] = b_inner[3]    (input bit 3)
      addr[4] = b_outer[0]    (input bit 4)
      addr[5] = 0             (stride bit, always 0 in upper)
      addr[6] = b_outer[1]    (input bit 5)
      addr[7] = b_outer[2]    (input bit 6)

lower = upper + stride = upper XOR (1 << 5) = upper XOR 32
  → lower_sym.matrix = upper_sym.matrix  (same)
  → lower_sym.constant = [0,0,0,0,0,1,0,0] (bit 5 flipped)
```

### Conflict subspace P:

**V_space** (from b_inner bits 0-3 in upper and lower):
```
Column 0: [1,0,0,0,0,0,0,0]  = e_0
Column 1: [0,1,0,0,0,0,0,0]  = e_1
Column 2: [0,0,1,0,0,0,0,0]  = e_2
Column 3: [0,0,0,1,0,0,0,0]  = e_3
```

**V_pattern** (delta between upper and lower in same iteration):
```
delta_const = upper.constant XOR lower.constant = e_5
delta_matrix = 0 (same matrix)
```

So: **P = span(e_0, e_1, e_2, e_3, e_5)** — dim = 5

### Solving for S:

Need S ∈ F2^{5×8} (5 bank bits for 32 banks) such that S·v ≠ 0 for all v ∈ P.

F2LayoutSolver produces:
```
S = [[1,0,0,0,0,0,0,0],   ← bank[0] = addr[0]
     [0,1,0,0,0,0,0,0],   ← bank[1] = addr[1]
     [0,0,1,0,0,0,0,0],   ← bank[2] = addr[2]
     [0,0,0,1,0,0,0,0],   ← bank[3] = addr[3]
     [0,0,0,0,1,1,0,0]]   ← bank[4] = addr[4] XOR addr[5]
```

The XOR in row 4 resolves the stride conflict: `e_5` maps to `S·e_5 = [0,0,0,0,1]` ≠ 0.

This matches the manually-specified `stride_bit=5` result exactly!

---

## Generality Beyond FFT

The symbolic execution approach handles any application with power-of-2-sized
buffers. The key is that bank conflicts emerge from the conflict subspace P,
which the engine constructs uniformly regardless of the application.

### Example 1: Cyclic Partition from Interleaved Access (common HLS pattern)

```python
for i in range(N):        # pipelined, II=1
    A[i * 2] = ...        # even address
    A[i * 2 + 1] = ...    # odd address
```

**Symbolic execution** (i has log2(N) bits):
- `i*2` = `i << 1`:     matrix row 0 = 0, row 1 = i[0], row 2 = i[1], ...
                         constant = `[0, 0, ...]`
- `i*2+1` = `(i<<1)|1`: same matrix, constant = `[1, 0, ...]`

**Conflict subspace** (pipelined, no unrolling → Source 1 only):
- delta_const = `[1,0,...] XOR [0,0,...] = e_0`
- delta_matrix = 0 (same dependence on i)
- **P = span(e_0)**, dim = 1

**Solver output**: S = `[[1, 0, ...]]` → bank = addr[0].
This IS **cyclic partition with factor 2** — even addresses in bank 0,
odd addresses in bank 1. Exactly what HLS `#pragma array_partition cyclic factor=2` does.

### Example 2: Interleaved Access with Unrolling

```python
for i in range(N):        # UNROLLED fully (N iterations in parallel)
    A[i * 2] = ...
    A[i * 2 + 1] = ...
```

With N=4, i has 2 bits:

**Conflict subspace** (Sources 1 + 2):
- Source 1 (intra-iteration): delta_const = e_0
- Source 2 (inter-iteration): columns for i[0] in sym(i*2) = e_1, in sym(i*2+1) = e_1
                               columns for i[1] in sym(i*2) = e_2, in sym(i*2+1) = e_2
- **P = span(e_0, e_1, e_2)**, dim = 3 → needs 8 banks (complete partition)

This is correct: 4 iterations × 2 accesses = 8 simultaneous writes to
A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7] — all must be in different banks.

### Example 3: Block Access (contiguous chunks)

```python
for i in range(NUM_BLOCKS):   # pipelined
    for k in range(BLOCK_SIZE):  # unrolled → BLOCK_SIZE parallel accesses
        B[i * BLOCK_SIZE + k] = ...
```

With BLOCK_SIZE=4, k has 2 bits, i has remaining bits:

**Symbolic execution**: `i * 4 + k = (i << 2) | k` (non-overlapping):
- matrix: addr[0] = k[0], addr[1] = k[1], addr[2] = i[0], addr[3] = i[1], ...

**Conflict subspace** (Source 2 only, single access per iteration):
- Columns for k[0]: e_0
- Columns for k[1]: e_1
- **P = span(e_0, e_1)**, dim = 2

**Solver output**: S = `[[1,0,...],[0,1,...]]` → bank = addr[1:0] = k.
This IS **cyclic partition with factor 4** — or equivalently, complete
partition of the first dimension if reshaped to `B[NUM_BLOCKS][BLOCK_SIZE]`.

### Example 4: Matrix Transpose (scatter write)

```python
for j in range(N):    # unrolled
    buf[j * N + i] = inp[i * N + j]   # write at transposed index
```

With N=8, j has 3 bits:

**Symbolic execution**: `j * 8 + i = (j << 3) | i` (non-overlapping):
- j bits map to addr[5:3], i bits map to addr[2:0]

**Conflict subspace** (Source 2, j is unrolled):
- Columns for j[0..2]: e_3, e_4, e_5
- **P = span(e_3, e_4, e_5)**, dim = 3

**Solver output**: S picks addr[5:3] as bank bits → bank = j.
This IS **block partition** — each row in separate bank.

### Example 5: Stencil (non-linear — conservative fallback)

```python
for i in range(N):   # pipelined
    out[i] = buf[i-1] + buf[i] + buf[i+1]
```

**Symbolic execution**:
- `buf[i]`: linear, addr = i
- `buf[i-1]`: `i - 1` has carry propagation → **non-linear** → opaque
- `buf[i+1]`: same, opaque

**Conflict subspace**: Since opaque symbols have independent bits, all
address bits are potentially in P. The solver conservatively requires
complete partitioning (all elements in separate banks).

For stencils, the user would need to either:
1. Accept complete partitioning (if buffer is small enough), or
2. Manually specify a sliding-window partition scheme via `s.f2_layout()`

Note: `i+1` and `i-1` could be handled with a **modular arithmetic extension**
to F2Symbol (representing addition with carries as a sequence of XOR+AND
operations). This is a future enhancement, not needed for the core engine.

---

## Risk Mitigation

1. **Correctness**: Unit-test F2Symbol against known index expressions.
   Cross-check auto-detected partitions against manual s.f2_layout() on FFT.

2. **Non-linear fallback**: When symbolic execution hits a non-linear op,
   return opaque (conservative). The user sees a clear error if the conflict
   subspace is too large for the available banks.

3. **Performance**: Verify via csyn that auto-F2 produces the same HLS
   pragmas and achieves the same II=1 as manual F2 on the FFT benchmark.

4. **Incremental delivery**:
   - Step 1: F2Symbol + unit tests (pure Python, no MLIR dependency)
   - Step 2: MLIR SSA walker + conflict subspace builder
   - Step 3: Integration with s.auto_f2() schedule primitive
   - Step 4: s.fold() grid folding (frontend changes)

---

## Connection to Linear Layouts Paper (§5.4)

| Paper Concept | Our FPGA Adaptation |
|---|---|
| Register bits | Bank bits (which partition) |
| Thread bits | Offset bits (address within bank) |
| Layout matrix L | Access matrix A (index expr → addr bits) |
| Conflict: P = span(M_Vec ∪ A_Thr) | P = span(V_space ∪ V_pattern) |
| Find H s.t. P ∩ span(H) = {0} | Find S s.t. ker(S) ∩ P = {0} |
| Optimal swizzled memory layout M | Partition matrix S |
| E ⊕ F construction for H | XOR row heuristic in F2LayoutSolver |

The paper's optimal swizzling (§5.4) solves the same core problem: given
simultaneously-accessed addresses (from multiple threads / multiple unrolled
iterations), find a bank-to-address mapping that avoids conflicts. Their
"find largest H disjoint from P" is equivalent to our "find S whose kernel
is disjoint from P".
