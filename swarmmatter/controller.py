"""Minimal FSM controller. Inputs: contact bit c, light bit l (is robot facing goal).
Memory = K states -> log2(K) bits. Actions: 0 fwd, 1 turn left, 2 turn right, 3 stop."""
import numpy as np

def random_genome(K, rng):
    return {"K": K,
            "next": rng.integers(0, K, size=(K, 4)),
            "act": rng.integers(0, 4, size=(K, 4))}

def mutate(g, rng, rate=0.1):
    h = {"K": g["K"], "next": g["next"].copy(), "act": g["act"].copy()}
    m = rng.random(h["next"].shape) < rate
    h["next"][m] = rng.integers(0, h["K"], size=m.sum())
    m = rng.random(h["act"].shape) < rate
    h["act"][m] = rng.integers(0, 4, size=m.sum())
    return h
