import sys; sys.path.insert(0, ".")
import numpy as np
from swarmmatter import simulate, random_genome
def test_runs():
    g = random_genome(2, np.random.default_rng(0))
    assert np.isfinite(simulate(g, N=5))
