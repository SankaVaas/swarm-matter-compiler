import numpy as np
from matterc.evolve import evolve
from matterc.discover import discover
from matterc.coarse_grain import fields

def test_evolution_monotone_and_improves():
    score, fsm, hist = evolve(2, 8, gens=6, pop=12, steps=200, verbose=False)
    assert all(b >= a for a, b in zip(hist, hist[1:]))
    assert score > 0

def test_sindy_recovers_diffusion():
    X, T, D = 64, 150, 0.5
    x = np.linspace(0, 2*np.pi, X, endpoint=False); dx = x[1]-x[0]; dt = 0.2*dx**2/D
    rho = np.zeros((T, X)); rho[0] = np.exp(-(x-np.pi)**2)
    for t in range(T-1):
        rho[t+1] = rho[t] + dt*D*(np.roll(rho[t],1)-2*rho[t]+np.roll(rho[t],-1))/dx**2
    c = discover(rho, dt, dx)
    assert abs(c["rho_xx"] - D) < 0.05

def test_coarse_grain_conserves_count():
    pos = np.random.default_rng(0).uniform(-2, 2, (3, 50, 2))
    rho, vel = fields(pos, grid=8)
    assert np.allclose(rho.sum((1, 2)), 50)
