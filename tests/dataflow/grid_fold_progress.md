# Grid Folding Implementation Progress

## Architecture

**PE expansion** (`allo/ir/builder.py`):
- `mapping=[M, N]` → `np.ndindex(M, N)` → M*N separate kernel functions
- Each function has `df.p0, df.p1` set to constant integers
- `get_pid()` returns these constants (inlined at compile time)

**Grid folding**: `fold={dim: factor}` reduces PE count by `factor`, inserting
an unrolled loop of `factor` iterations per PE.

Example: `mapping=[8, 128], fold={1: 32}` → 8×4 = 32 PEs instead of 1024.
Each PE has `for _fold_1 in range(32): ...` loop, with `pid = outer*32 + _fold_1`.

## Implementation (All Steps Complete)

### Step 1: Builder PE expansion with fold ✅
- Parse `fold` keyword from `@df.kernel` decorator via `eval()`
- Compute `reduced_mapping` by dividing folded dims by factor
- Use `reduced_mapping` for `np.ndindex` iteration
- Store `_FoldInfo(outer_val, factor, fold_var_name)` in `ctx.global_vars`
- `_wrap_body_with_fold_loops()` wraps kernel body AST in fold `for` loops
- Updated call insertion to use `reduced_mapping` via `ctx._kernel_fold_info`

### Step 2: Dynamic get_pid() handling ✅
- Modified both `get_pid()` code paths (assignment form + call form)
- For folded dims: emits `arith.constant(outer*factor) + fold_iv` MLIR ops
- Result cast from `index` to `i32` to match `MockConstant` convention
- Put result in symbol table as `MockArg` (not `global_vars`)
- Non-folded dims: unchanged (constant via `global_vars`)

### Step 3: Unroll annotation ✅
- Fold loops get `unroll=True` in synthesized AST
- Builder translates to `unroll = 0 : ui32` attribute on `affine.for`
- Auto-F2 can detect unrolled loops for conflict subspace construction

### Step 4: Type inference (infer.py) ✅
- Parse `fold` keyword in type inference pass
- Use `reduced_mapping` for PE iteration in `visit_FunctionDef`

### Step 5: ConstExpr guard ✅
- Folded pid dimensions are dynamic (not in `global_vars`)
- ConstExpr evaluation catches `NameError` and raises clear error with hint
- Test: `test_fold_constexpr_guard`

### Step 6: Declaration hoisting ✅
- `_is_array_decl()` identifies uninitialized array declarations
- Array decls hoisted outside fold loops so buffers are shared across iterations
- Essential for auto_f2 conflict detection

### Step 7: Fold + auto_f2 integration ✅
- auto_f2 correctly analyzes folded kernel buffers
- `apply_f2_layout` handles buffers inside folded functions
- Fixed: `_compute_bank_indices` guard for `stride_bit < bank_bits`
- Fixed: Block object handling in `defining_op` and `use.owner`
- Test: `test_fold_with_auto_f2` (32-elem buf → 8×4 partitioned)

## Files Modified
- `allo/ir/builder.py` — PE expansion, `_FoldInfo`, `_wrap_body_with_fold_loops`, get_pid()
- `allo/ir/infer.py` — fold-aware type inference
- `allo/ir/utils.py` — `_ast_to_value` dict support
- `allo/transform/f2_layout.py` — cyclic mode guard, Block object handling, MockArg guard
- `allo/transform/auto_f2.py` — simple cyclic stride_bit=0 handling
- `tests/dataflow/test_grid_fold.py` — 5 tests (all passing)
- `tests/dataflow/FEATURES.md` — Section 12: Grid Folding documentation

## Status: COMPLETE ✅
All 5 grid fold tests pass. Existing FFT auto_f2 test unaffected.
