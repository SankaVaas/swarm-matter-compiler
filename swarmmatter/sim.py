"""Overdamped 2D sim: N robots push one disc object. Object is the only coordination channel."""
import numpy as np

def simulate(g, N=10, steps=400, goal=(1.0, 0.0), seed=0, v=0.01, turn=0.3,
             R=0.08, r=0.02, alpha=0.5, return_traj=False):
    rng = np.random.default_rng(seed)
    goal = np.array(goal, float)
    obj = np.zeros(2)
    ang = rng.uniform(0, 2*np.pi, N)
    pos = obj + (R + 0.15) * np.c_[np.cos(ang), np.sin(ang)] * rng.uniform(0.5, 1.5, (N, 1))
    head = rng.uniform(0, 2*np.pi, N)
    state = np.zeros(N, int)
    traj = [obj.copy()]
    gdir = np.arctan2(*(goal - obj)[::-1])
    for _ in range(steps):
        d = pos - obj
        dist = np.linalg.norm(d, axis=1) + 1e-9
        pen = np.maximum(0, R + r - dist)
        c = (pen > 0).astype(int)
        l = (np.cos(head - gdir) > 0).astype(int)
        inp = c + 2 * l
        act = g["act"][state, inp]
        state = g["next"][state, inp]
        head = head + turn * ((act == 1) * 1 - (act == 2) * 1)
        pos = pos + v * (act == 0)[:, None] * np.c_[np.cos(head), np.sin(head)]
        d = pos - obj
        dist = np.linalg.norm(d, axis=1) + 1e-9
        pen = np.maximum(0, R + r - dist)
        n = d / dist[:, None]
        pos = pos + n * pen[:, None]          # robots pushed out of object
        obj = obj - alpha * (n * pen[:, None]).sum(0)  # object pushed by all contacts
        gdir = np.arctan2(*(goal - obj)[::-1])
        traj.append(obj.copy())
    prog = obj @ (goal / np.linalg.norm(goal))
    return (prog, np.array(traj)) if return_traj else prog
