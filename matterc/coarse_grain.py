"""Coarse-grain robot positions/velocities into continuum fields rho(x,t), v(x,t)."""
import numpy as np

def fields(pos_t, grid=16, L=3.0):
    """pos_t: (T, N, 2). Returns rho (T,g,g) and v (T,g,g,2) via finite differences."""
    edges = np.linspace(-L, L, grid + 1)
    rho = np.stack([np.histogram2d(p[:, 0], p[:, 1], bins=[edges, edges])[0] for p in pos_t])
    vel = np.zeros((len(pos_t), grid, grid, 2))
    for t in range(len(pos_t) - 1):
        dp = pos_t[t + 1] - pos_t[t]
        for k in range(2):
            s = np.histogram2d(pos_t[t][:, 0], pos_t[t][:, 1], bins=[edges, edges], weights=dp[:, k])[0]
            vel[t, ..., k] = s / np.maximum(rho[t], 1)
    return rho, vel
