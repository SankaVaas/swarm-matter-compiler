"""(mu+lambda) evolution of tiny FSM controllers, with logging + history."""
import logging
import numpy as np
from .controller import FSM
from .sim import Swarm

log = logging.getLogger("matterc.evolve")

def fitness(fsm, n_robots=10, seeds=(0, 1, 2), steps=400):
    return float(np.mean([Swarm(n_robots, fsm, seed=s).run(steps) for s in seeds]))

def evolve(n_states=2, n_robots=10, gens=30, pop=16, seed=0, steps=400, verbose=True):
    """Returns (best_score, best_fsm, history_of_best_per_gen)."""
    rng = np.random.default_rng(seed)
    P = [FSM(n_states, rng=rng) for _ in range(pop)]
    best, hist = (-1e9, None), []
    for g in range(gens):
        scored = sorted(((fitness(f, n_robots, steps=steps), i, f) for i, f in enumerate(P)), key=lambda x: -x[0])
        if scored[0][0] > best[0]: best = (scored[0][0], scored[0][2])
        hist.append(best[0])
        log.info("gen %d best=%.4f gen_best=%.4f", g, best[0], scored[0][0])
        elite = [s[2] for s in scored[:max(1, pop // 4)]]
        P = elite + [elite[rng.integers(len(elite))].mutate(rng) for _ in range(pop - len(elite))]
    return best[0], best[1], hist
