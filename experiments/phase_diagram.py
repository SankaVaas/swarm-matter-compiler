"""PHASE 1: sweep memory bits x N robots. Saves heatmap + data + logs."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from matterc.runs import Run
from matterc.evolve import evolve
from matterc import viz

Ns, states = [3, 10, 30], [1, 2, 4]
run = Run("phase_diagram", dict(Ns=Ns, states=states, gens=10, pop=12))
M = np.zeros((len(states), len(Ns)))
for i, s in enumerate(states):
    for j, n in enumerate(Ns):
        M[i, j], _, _ = evolve(s, n, gens=10, pop=12)
        run.log.info("states=%d N=%d -> %.3f", s, n, M[i, j])
run.save_array("phase.npy", M)
viz.plot_heatmap(M, Ns, [int(np.log2(s)) for s in states], run.path("figures", "phase.png"))
run.log_metric("phase", M)
