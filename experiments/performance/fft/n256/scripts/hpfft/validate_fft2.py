#!/scratch/hc676/allo-agent/bin/python3
"""Validate dumped FFT outputs against double-precision numpy FFT of the dumped inputs, overall and per seed.
usage: validate_fft2.py <inputs.txt> <outputs.txt> <N> <NT> [natural|bitrev] [--seeds <stimulus.json>] -> JSON on stdout
Files hold one 'rehex imhex' IEEE-754 single pair per line, NT*N lines, transform-major. Pass criterion:
numpy.allclose(y, ref, atol=1e-4, rtol=1e-4), ref = numpy.fft.fft(x) in complex128 (permuted to bit-reversed
index order when 'bitrev' is given: hardware output index k holds X[bitrev(k)])."""
import sys, json, numpy as np
a = sys.argv[1:]; seeds_fn = None
if "--seeds" in a:
    i = a.index("--seeds"); seeds_fn = a[i + 1]; del a[i:i + 2]
fi, fo, N, NT = a[0], a[1], int(a[2]), int(a[3])
order = a[4] if len(a) > 4 else "natural"
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
    ref = ref[:, perm]
err = np.abs(y - ref); scale = np.abs(ref).max(axis=1)
def stats(sel):
    if not sel.any(): return {"transforms_checked": 0, "validation_pass": None, "max_abs_error": None, "max_norm_error": None}
    return {"transforms_checked": int(sel.sum()), "validation_pass": bool(np.allclose(y[sel], ref[sel], atol=1e-4, rtol=1e-4)),
            "max_abs_error": float(err[sel].max()), "max_norm_error": float((err[sel].max(axis=1) / scale[sel]).max())}
res = {"N": N, "NT": NT, "order": order, "criterion": "numpy.allclose(atol=1e-4, rtol=1e-4) against numpy.fft.fft (complex128) of the dumped inputs",
       "max_norm_error_def": "max over transforms of max|y-ref| / max|ref| (per transform)"}
res.update(stats(np.ones(K, bool)))
res["rel_rms_error"] = float(np.sqrt((err ** 2).sum() / (np.abs(ref) ** 2).sum()))
res["max_input_abs"] = float(np.abs(x).max()); res["max_ref_abs"] = float(scale.max())
res["per_transform_max_abs"] = [float(v) for v in err.max(axis=1)][:40]
res["nan_or_inf"] = bool(~np.isfinite(y).all())
if seeds_fn:
    sm = json.load(open(seeds_fn)); so = np.array(sm["seed_of_transform"][:K])
    res["per_seed"] = {str(s): stats(so == s) for s in sm["seeds"]}
    res["seeds_checked"] = sorted(set(int(v) for v in so))
    res["seeds_all_pass"] = all(v["validation_pass"] for v in res["per_seed"].values() if v["validation_pass"] is not None)
    res["unchecked_transforms"] = [int(j) for j in range(K, NT)]
print(json.dumps(res))
