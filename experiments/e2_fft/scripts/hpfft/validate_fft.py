#!/scratch/hc676/allo-agent/bin/python3
"""Validate dumped FFT outputs against double-precision numpy FFT of the dumped inputs.
usage: validate_fft.py <inputs.txt> <outputs.txt> <N> <NT> [natural|bitrev] -> JSON on stdout
Files hold one 'rehex imhex' IEEE-754 single pair per line, NT*N lines, transform-major."""
import sys, json, numpy as np
fi, fo, N, NT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
order = sys.argv[5] if len(sys.argv) > 5 else "natural"
def load(fn):
    w = np.loadtxt(fn, dtype=str, ndmin=2)
    re_ = np.array([int(x, 16) for x in w[:, 0]], dtype=np.uint32).view(np.float32)
    im_ = np.array([int(x, 16) for x in w[:, 1]], dtype=np.uint32).view(np.float32)
    v = (re_.astype(np.float64) + 1j * im_.astype(np.float64))
    return v[: (len(v) // N) * N].reshape(-1, N)
x, y = load(fi), load(fo)
K = min(len(x), len(y), NT)          # transforms that can be checked (the RTL may not flush the last one)
x, y = x[:K], y[:K]
ref = np.fft.fft(x, axis=1)
if order == "bitrev":
    L = N.bit_length() - 1
    perm = np.array([int(format(i, f"0{L}b")[::-1], 2) for i in range(N)])
    ref = ref[:, perm]   # hardware output index k holds X[bitrev(k)]
err = np.abs(y - ref)
scale = np.abs(ref).max(axis=1)
res = {"N": N, "NT": NT, "transforms_checked": int(K), "order": order,
       "validation_pass": bool(np.allclose(y, ref, atol=1e-4, rtol=1e-4)),
       "max_abs_error": float(err.max()),
       "max_norm_error": float((err.max(axis=1) / scale).max()),
       "max_norm_error_def": "max over transforms of max|y-ref| / max|ref| (per transform)",
       "rel_rms_error": float(np.sqrt((err**2).sum() / (np.abs(ref)**2).sum())),
       "max_input_abs": float(np.abs(x).max()), "max_ref_abs": float(scale.max()),
       "per_transform_max_abs": [float(v) for v in err.max(axis=1)][:8],
       "nan_or_inf": bool(~np.isfinite(y).all())}
print(json.dumps(res))
