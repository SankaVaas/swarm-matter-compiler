"""2D overdamped swarm simulator (pure NumPy, runs on CPU).
N robots (discs) + one passive object (disc). No communication, no map.
Inputs: contact bit + global light bit. Only coupling between robots: steric contact and the shared object.
The goal is a global 1-bit light: robots never see it; fitness only
measures object displacement toward the goal direction (+x).
"""
import numpy as np

class Swarm:
    def __init__(self, n_robots=10, fsm=None, obj_radius=0.5, robot_radius=0.05,
                 arena=3.0, seed=0, step_len=0.01, turn=0.3, obj_mobility=1.0):
        self.rng = np.random.default_rng(seed)
        self.N, self.fsm = n_robots, fsm
        self.r, self.R = robot_radius, obj_radius
        self.step_len, self.turn, self.mu = step_len, turn, obj_mobility
        self.reset()

    def reset(self):
        self.obj = np.zeros(2)
        ang = self.rng.uniform(0, 2*np.pi, self.N)
        rad = self.R + self.r + self.rng.uniform(0.05, 1.0, self.N)
        self.pos = np.stack([rad*np.cos(ang), rad*np.sin(ang)], 1)
        self.th = self.rng.uniform(0, 2*np.pi, self.N)
        self.state = np.zeros(self.N, dtype=int)
        self.contact = np.zeros(self.N, dtype=int)

    def _resolve(self):
        # robot-object: push object, push robot out
        d = self.pos - self.obj
        dist = np.linalg.norm(d, axis=1) + 1e-9
        pen = (self.R + self.r) - dist
        hit = pen > 0
        self.contact = hit.astype(int)
        if hit.any():
            n = d[hit] / dist[hit, None]
            self.obj -= self.mu * (pen[hit, None] * n).sum(0) * 0.5
            self.pos[hit] += n * pen[hit, None] * 0.5
        # robot-robot soft repulsion
        diff = self.pos[:, None] - self.pos[None]
        dd = np.linalg.norm(diff, axis=2) + np.eye(self.N)
        ov = np.clip(2*self.r - dd, 0, None)
        np.fill_diagonal(ov, 0)
        self.pos += 0.5 * (diff / dd[..., None] * ov[..., None]).sum(1)

    def step(self):
        light = (np.cos(self.th) > 0).astype(int)
        idx = self.contact * 2 + light
        nxt, act = self.fsm.table[self.state, idx, 0], self.fsm.table[self.state, idx, 1]
        self.state = nxt
        self.th += np.where(act == 1, self.turn, 0) - np.where(act == 2, self.turn, 0)
        move = (act == 0)
        self.pos[move] += self.step_len * np.stack([np.cos(self.th[move]), np.sin(self.th[move])], 1)
        self._resolve()

    def run(self, steps=500, record_every=0):
        """Returns object x-displacement. If record_every>0, stores self.frames."""
        self.frames = []
        for t in range(steps):
            self.step()
            if record_every and t % record_every == 0:
                self.frames.append(dict(t=t, pos=self.pos.copy(), obj=self.obj.copy()))
        return self.obj[0]
