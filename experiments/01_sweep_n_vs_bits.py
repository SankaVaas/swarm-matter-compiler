"""Phase diagram v0: transport progress vs robot count N and memory states K."""
import sys, json; sys.path.insert(0, ".")
from swarmmatter import evolve
Ns, Ks = [3, 10, 30], [1, 2, 4, 8]   # K=1 -> 0 bits of memory
res = {}
for K in Ks:
    for N in Ns:
        g, f = evolve(K, N, gens=30, pop=16)
        res[f"K{K}_N{N}"] = f
        print(f"K={K} (bits={K.bit_length()-1}) N={N} progress={f:.3f}", flush=True)
json.dump(res, open("experiments/phase_v0.json", "w"), indent=1)
