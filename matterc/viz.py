"""Plots saved to files (headless-safe)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_fitness(history, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(history, marker="o"); ax.set_xlabel("generation"); ax.set_ylabel("best fitness (object +x)")
    ax.set_title("Evolution progress"); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

def plot_snapshots(frames, R, r, path, n=4):
    """frames: list of dict(pos, obj). Shows n snapshots across time."""
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    for ax, i in zip(axs, idx):
        f = frames[i]
        ax.add_patch(plt.Circle(f["obj"], R, color="tab:orange", alpha=.6))
        ax.scatter(*f["pos"].T, s=8, color="tab:blue")
        ax.set_xlim(-2.5, 4.5); ax.set_ylim(-2.5, 2.5); ax.set_aspect("equal"); ax.set_title(f"step {f['t']}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

def plot_object_path(frames, path):
    p = np.array([f["obj"] for f in frames])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(p[:, 0], p[:, 1]); ax.scatter(*p[0], c="g", label="start"); ax.scatter(*p[-1], c="r", label="end")
    ax.set_aspect("equal"); ax.legend(); ax.set_title("Object path"); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

def plot_heatmap(M, xlabels, ylabels, path, title="Phase diagram (object displacement)"):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    im = ax.imshow(M, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels); ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_xlabel("N robots"); ax.set_ylabel("memory bits"); ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
