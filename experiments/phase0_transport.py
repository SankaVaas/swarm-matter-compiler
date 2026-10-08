"""PHASE 0: evolve a minimal FSM swarm to push an object. Saves logs, metrics, figures, genome."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from matterc.runs import Run
from matterc.evolve import evolve
from matterc.sim import Swarm
from matterc import viz

cfg = dict(n_states=2, n_robots=10, gens=20, pop=16, steps=400, seed=0)
run = Run("phase0_transport", cfg)
score, fsm, hist = evolve(cfg["n_states"], cfg["n_robots"], cfg["gens"], cfg["pop"], cfg["seed"], cfg["steps"])
run.log_metric("best_fitness", score); run.log_metric("history", hist)
run.save_array("best_genome.npy", fsm.table)
viz.plot_fitness(hist, run.path("figures", "fitness.png"))
sw = Swarm(cfg["n_robots"], fsm, seed=123); disp = sw.run(cfg["steps"], record_every=10)
run.log_metric("heldout_displacement", disp)
viz.plot_snapshots(sw.frames, sw.R, sw.r, run.path("figures", "snapshots.png"))
viz.plot_object_path(sw.frames, run.path("figures", "object_path.png"))
run.log.info("done -> %s", run.dir)
