import numpy as np
from matterc import Swarm, FSM

def test_runs_and_finite():
    assert np.isfinite(Swarm(5, FSM(2, rng=np.random.default_rng(0)), seed=0).run(50))

def test_deterministic_given_seed():
    f = FSM(1, rng=np.random.default_rng(1))
    assert Swarm(5, f, seed=3).run(50) == Swarm(5, f, seed=3).run(50)

def test_no_overlap_explosion():
    s = Swarm(20, FSM(2, rng=np.random.default_rng(2)), seed=1); s.run(200)
    assert np.all(np.abs(s.pos) < 50)

def test_record_frames():
    s = Swarm(5, FSM(2, rng=np.random.default_rng(0)), seed=0); s.run(50, record_every=10)
    assert len(s.frames) == 5
