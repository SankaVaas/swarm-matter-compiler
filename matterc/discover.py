"""Minimal SINDy (sequentially thresholded least squares) in pure NumPy.
Discovers sparse continuum law: d(target)/dt = Theta(library) @ xi.
"""
import numpy as np

def library(rho, dx=1.0):
    """Candidate terms for 1D-reduced density: rho, rho^2, d_x rho, d_xx rho, rho*d_x rho."""
    dx_ = np.gradient(rho, dx, axis=-1)
    dxx = np.gradient(dx_, dx, axis=-1)
    cols = {"rho": rho, "rho^2": rho**2, "rho_x": dx_, "rho_xx": dxx, "rho*rho_x": rho*dx_}
    return list(cols), np.stack([c.ravel() for c in cols.values()], 1)

def stlsq(Theta, y, thresh=0.05, iters=10):
    xi = np.linalg.lstsq(Theta, y, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(xi) < thresh
        xi[small] = 0
        big = ~small
        if big.any():
            xi[big] = np.linalg.lstsq(Theta[:, big], y, rcond=None)[0]
    return xi

def discover(rho_series, dt=1.0, dx=1.0, thresh=0.05):
    """rho_series: (T, X). Returns dict term->coef for rho_t = sum xi_k * term_k."""
    rho_t = np.gradient(rho_series, dt, axis=0).ravel()
    names, Theta = library(rho_series, dx)
    return dict(zip(names, stlsq(Theta, rho_t, thresh)))
