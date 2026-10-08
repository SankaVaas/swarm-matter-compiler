"""Sanity check: SINDy recovers a known diffusion law. Saves figure + metrics."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matterc.runs import Run
from matterc.discover import discover
run = Run("sindy_demo")
X, T, D = 64, 200, 0.5
x = np.linspace(0, 2*np.pi, X, endpoint=False); dx = x[1]-x[0]; dt = 0.2*dx**2/D
rho = np.zeros((T, X)); rho[0] = np.exp(-(x-np.pi)**2)
for t in range(T-1):
    lap = (np.roll(rho[t],1)-2*rho[t]+np.roll(rho[t],-1))/dx**2
    rho[t+1] = rho[t] + dt*D*lap
coefs = discover(rho, dt, dx); run.log_metric("coefs", coefs); run.log_metric("true_D", D)
fig, ax = plt.subplots(figsize=(5,3.5)); ax.imshow(rho.T, aspect="auto", origin="lower"); ax.set_title("synthetic rho(x,t)")
fig.savefig(run.path("figures","rho.png"), dpi=120)
