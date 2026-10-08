import numpy as np
from .controller import random_genome, mutate
from .sim import simulate

def fitness(g, N, seeds=(0, 1, 2)):
    return float(np.mean([simulate(g, N=N, seed=s) for s in seeds]))

def evolve(K, N, gens=60, pop=24, seed=0):
    """(mu+lambda) evolution of FSM tables. Returns best genome and fitness."""
    rng = np.random.default_rng(seed)
    P = [random_genome(K, rng) for _ in range(pop)]
    F = [fitness(g, N) for g in P]
    for _ in range(gens):
        order = np.argsort(F)[::-1][: pop // 4]
        kids = [mutate(P[i], rng) for i in order for _ in range(3)]
        kF = [fitness(k, N) for k in kids]
        P = [P[i] for i in order] + kids
        F = [F[i] for i in order] + kF
        keep = np.argsort(F)[::-1][:pop]
        P, F = [P[i] for i in keep], [F[i] for i in keep]
    return P[0], F[0]
