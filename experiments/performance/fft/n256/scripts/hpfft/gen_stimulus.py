#!/scratch/hc676/allo-agent/bin/python3
"""E2 stimulus: NT transforms of N complex samples, re and im ~ Uniform[-1,1) as float32, drawn from
numpy.random.RandomState(seed) for seeds 0, 1, 2; seed s owns the s-th block of ceil(NT/3) consecutive transforms.
usage: gen_stimulus.py <N> <NT> <out_prefix>
writes <out_prefix>.txt  : one 'rehex imhex' line per sample (IEEE-754 single), transform-major
       <out_prefix>.json : seeds, seed of every transform, distribution"""
import sys, json, numpy as np
N, NT, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
SEEDS = [0, 1, 2]
per = -(-NT // len(SEEDS))
seed_of = [SEEDS[min(j // per, len(SEEDS) - 1)] for j in range(NT)]
blocks = {s: seed_of.count(s) for s in SEEDS}
re, im = [], []
for s in SEEDS:
    if blocks[s] == 0: continue
    x = np.random.RandomState(s).uniform(-1.0, 1.0, size=(blocks[s], N, 2)).astype(np.float32)
    re.append(x[:, :, 0]); im.append(x[:, :, 1])
re, im = np.concatenate(re), np.concatenate(im)
with open(out + ".txt", "w") as f:
    for j in range(NT):
        f.writelines("%08x %08x\n" % (a, b) for a, b in zip(re[j].view(np.uint32), im[j].view(np.uint32)))
json.dump({"N": N, "NT": NT, "seeds": SEEDS, "seed_of_transform": seed_of, "transforms_per_seed": blocks,
           "distribution": "re, im ~ Uniform[-1,1) as float32: numpy.random.RandomState(seed).uniform(-1,1,(count,N,2)).astype(float32)"},
          open(out + ".json", "w"))
print("stimulus", out, "N", N, "NT", NT, "per seed", blocks)
