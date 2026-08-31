# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dump the block's eleven invocations as raw bytes, for a numpy-free host.
Run under the Allo environment::

    python3 scripts/spmw_board_trace.py /scratch/$USER/trace16

Pairs with `spmw_board_run.py`, which runs under the *system* Python
because that is what pyxrt is built against.
"""
import os, sys

sys.path.insert(0, "tests/dataflow/spmw")
import numpy as np
from test_spmw_transformer import BIG, _params, _ref_block, transformer_block

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
params = _params(shape=BIG)
out, eng = transformer_block(*params, shape=BIG)
assert np.array_equal(out, _ref_block(*params)), "reference disagrees"

DTYPE = {
    "A": np.int8,
    "MProg": np.int32,
    "VProg": np.int32,
    "W": np.int8,
    "Bias": np.int32,
    "Y": np.int32,
}
names = []
for i, s in enumerate(eng.trace):
    names.append(s["name"])
    for key, dt in DTYPE.items():
        a = np.ascontiguousarray(s[key], dtype=dt)
        with open(os.path.join(out_dir, f"s{i}_{key}.bin"), "wb") as f:
            f.write(a.tobytes())
with open(os.path.join(out_dir, "steps.txt"), "w") as f:
    f.write("\n".join(names))
print(f"wrote {len(names)} steps to {out_dir}")
