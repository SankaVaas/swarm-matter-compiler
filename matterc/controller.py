"""Minimal robot brain: FSM with 2 input bits: contact (touching object/robot) and light
(is my heading within 90 deg of the global goal light, +x). Input index = contact*2 + light.
Genome: table[state, contact] -> (next_state, action).
Actions: 0=forward, 1=turn left, 2=turn right, 3=stay.
Memory bits = log2(n_states).
"""
import numpy as np

class FSM:
    def __init__(self, n_states=2, table=None, rng=None):
        self.n = n_states
        rng = rng or np.random.default_rng()
        # table[s, c] = (next_state, action)
        self.table = table if table is not None else np.stack(
            [rng.integers(0, n_states, (n_states, 4)),
             rng.integers(0, 4, (n_states, 4))], axis=-1)

    def mutate(self, rng, p=0.15):
        t = self.table.copy()
        for s in range(self.n):
            for c in range(4):
                if rng.random() < p:
                    t[s, c, 0] = rng.integers(0, self.n)
                if rng.random() < p:
                    t[s, c, 1] = rng.integers(0, 4)
        return FSM(self.n, t)

    def step(self, state, contact):
        nxt = self.table[state, contact, 0]
        act = self.table[state, contact, 1]
        return nxt, act
