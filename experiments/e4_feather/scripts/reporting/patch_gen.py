"""E4b: extend e4_feather_gen.py (the previous agent's generator) in three places.

1. A `mixed` operand pattern: the stored byte ranges over all of [0, 256)
   whatever the zero point, so d = u - zp is negative for u < zp. On the
   RTL that is where the unsigned datapath parts from the signed
   mathematics; the checker already reports it (`signed_equals_rtl`).
2. Convolution programs for AW = 8 and 16 (the drivers ship four for AW =
   4): a depth-first search over the BIRRD stages with the ports' column
   sets as the state, verified by the selftest against the drivers'
   semantics (feather_ref) and against the RTL model under the mapping.
3. The conv tiling for any AW (RS padded up to a multiple of AH).
"""
import re
import sys

path = sys.argv[1]
src = open(path).read()


def sub(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:80])
    src = src.replace(old, new)


# -- 1. the mixed pattern --------------------------------------------------------
sub('''    hi = 256 - zp
    return {"small": min(8, hi), "mid": min(32, hi), "full": hi, "sparse": hi}[pattern]''',
    '''    hi = 256 - zp
    # `mixed`: the stored byte covers all of [0, 256) whatever the zero point, so
    # d = u - zp is negative for u < zp -- the operands the RTL's unsigned
    # datapath cannot handle (the checker reports `signed_equals_rtl`).
    return {"small": min(8, hi), "mid": min(32, hi), "full": hi, "sparse": hi, "mixed": 256}[pattern]''')

# the stored byte wraps explicitly (numpy's int64 -> uint8 cast wraps too, but say so)
sub('''    A = (rng.integers(0, rng_hi, size=(M, K)) + zpa).astype(np.uint8)
    B = (rng.integers(0, rng_hi, size=(K, Nn)) + zpw).astype(np.uint8)''',
    '''    A = ((rng.integers(0, rng_hi, size=(M, K)) + zpa) % 256).astype(np.uint8)
    B = ((rng.integers(0, rng_hi, size=(K, Nn)) + zpw) % 256).astype(np.uint8)''')
sub('''    x = (rng.integers(0, rng_hi, size=(H, Wd, Cin)) + zpa).astype(np.uint8)  # HWC
    w = (rng.integers(0, rng_hi, size=(M, Cin, R, S)) + zpw).astype(np.uint8)''',
    '''    x = ((rng.integers(0, rng_hi, size=(H, Wd, Cin)) + zpa) % 256).astype(np.uint8)  # HWC
    w = ((rng.integers(0, rng_hi, size=(M, Cin, R, S)) + zpw) % 256).astype(np.uint8)''')
sub('''        iActs = (iActs + zpa).astype(np.uint8)
        weights = (weights + zpw).astype(np.uint8)''',
    '''        iActs = ((iActs + zpa) % 256).astype(np.uint8)
        weights = ((weights + zpw) % 256).astype(np.uint8)''')

# -- 2. conv programs for any AW ------------------------------------------------
sub('''def conv_insts():
    return [
        np.array([[AL, AL], [AL, PS], [PS, PS]], dtype=np.int8),
        np.array([[AL, AL], [AL, PS], [SW, PS]], dtype=np.int8),
        np.array([[AL, AL], [AR, PS], [PS, PS]], dtype=np.int8),
        np.array([[AL, AL], [AR, PS], [PS, SW]], dtype=np.int8),
    ]
''',
    '''def conv_insts(AW=4):
    """One program per output position in a line: every column sum of the
    tile into output column `pos`. AW = 4: the drivers' four programs
    (examples/feather/convolution.py). AW = 8 and 16: found by
    `find_reduce_program` under the same semantics, checked by the selftest."""
    if AW == 4:
        return [
            np.array([[AL, AL], [AL, PS], [PS, PS]], dtype=np.int8),
            np.array([[AL, AL], [AL, PS], [SW, PS]], dtype=np.int8),
            np.array([[AL, AL], [AR, PS], [PS, PS]], dtype=np.int8),
            np.array([[AL, AL], [AR, PS], [PS, SW]], dtype=np.int8),
        ]
    if AW not in _CONV_CACHE:
        _CONV_CACHE[AW] = [find_reduce_program(AW, pos) for pos in range(AW)]
    return _CONV_CACHE[AW]


_CONV_CACHE = {}


def find_reduce_program(AW, pos):
    """A BIRRD program (the drivers' semantics: PS/AR/AL/SW, the bit-reversal
    links of `stage_bits`) whose output column `pos` carries the sum of all AW
    column sums. Depth-first over the stages; the state is which columns each
    port holds (a bit set a port), merges are tried first, and a (stage, state)
    pair is visited once. The other output columns are unconstrained."""
    import itertools

    P0, P1 = birrd_shape(AW)
    full = (1 << AW) - 1
    seen = set()

    def dest(s, q):
        return q if s == P0 - 1 else reverse_bits(q, stage_bits(s, AW))

    def dfs(s, masks, prog):
        if s == P0:
            return prog if masks[pos] == full else None
        key = (s, masks)
        if key in seen:
            return None
        seen.add(key)
        opts = []
        for j in range(P1):
            l, r = masks[2 * j], masks[2 * j + 1]
            if l and r:
                opts.append([(AL, l | r, r), (AR, l, l | r), (PS, l, r), (SW, r, l)])
            elif l or r:
                opts.append([(PS, l, r), (SW, r, l)])
            else:
                opts.append([(PS, 0, 0)])
        for combo in itertools.product(*opts):
            nxt = [0] * AW
            for j, (_c, ol, orr) in enumerate(combo):
                nxt[dest(s, 2 * j)] = ol
                nxt[dest(s, 2 * j + 1)] = orr
            found = dfs(s + 1, tuple(nxt), prog + [[c for (c, _, _) in combo]])
            if found is not None:
                return found
        return None

    prog = dfs(0, tuple(1 << q for q in range(AW)), [])
    assert prog is not None, (AW, pos)
    inst = np.array(prog, dtype=np.int8)
    assert inst.shape == (P0, P1), inst.shape
    return inst
''')

# make_tiles: conv<k> programs at any N
sub('''        elif args.program.startswith("conv"):
            inst = conv_insts()[int(args.program[4:])]''',
    '''        elif args.program.startswith("conv"):
            inst = conv_insts(N)[int(args.program[4:])]''')
sub('''            if N == 4:
                inst = conv_insts()[t % 4]
            elif t % 2 == 0:''',
    '''            if N == 4:
                inst = conv_insts(N)[t % 4]
            elif t % 2 == 0:''')

# -- 3. the conv tiling for any AW ---------------------------------------------
sub('''    assert AW == 4, "the drivers' conv programs exist for AW = 4"
    P, Q = H + 2 * pad - R + 1, Wd + 2 * pad - S + 1''',
    '''    P, Q = H + 2 * pad - R + 1, Wd + 2 * pad - S + 1''')
sub('''    insts = conv_insts()
    tiles = []
    for p in range(P):''',
    '''    insts = conv_insts(AW)
    tiles = []
    for p in range(P):''')

# -- selftest: the conv programs at 8 and 16, on tiles ----------------------------
sub('''    for N in (4, 8, 16):
        progs = [gemm_insts(N)] + (conv_insts() if N == 4 else []) + [
            rng.integers(0, 4, size=birrd_shape(N)).astype(np.int8) for _ in range(5)
        ]
        for inst in progs:''',
    '''    for N in (4, 8, 16):
        # every conv program sums all N column sums into its output column
        for pos, inst in enumerate(conv_insts(N)):
            iActs = rng.integers(-128, 128, size=(N, N)).astype(np.int8)
            weights = rng.integers(-128, 128, size=(N, N, N)).astype(np.int8)
            ref = feather_ref(iActs, weights, inst, N, N).astype(np.int64)
            cols = np.einsum("kj,ijk->ij", iActs.astype(np.int64), weights.astype(np.int64))
            assert np.array_equal(ref[:, pos], cols.sum(axis=1)), ("conv program", N, pos)
        progs = [gemm_insts(N)] + list(conv_insts(N)) + [
            rng.integers(0, 4, size=birrd_shape(N)).astype(np.int8) for _ in range(5)
        ]
        for inst in progs:''')

open(path, "w").write(src)
print("patched", path)
