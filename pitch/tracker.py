import math

import numpy as np

SIG_E = 0.004
SIG_P = 0.05


class HandTracker:
    def __init__(self, layout, cfg):
        self.lay = layout
        self.cfg = cfg
        self.S = layout.S
        self.active = False
        self.age = 0.0
        self.n_starts = 0
        self.e = np.zeros(self.S)
        self.v = np.zeros(self.S)
        self.Pm = np.tile(np.eye(2), (self.S, 1, 1))
        self._pos = np.array([0.0, 0.3])
        self._J = None
        self._solved = False

    def start(self, pos, e=None):
        if e is None:
            e = self.lay.excess(pos[0], pos[1])
        self.e = np.array(e, float)
        self.v = np.zeros(self.S)
        self.Pm = np.tile(np.diag([0.02 ** 2, 0.4 ** 2]), (self.S, 1, 1))
        self.active = True
        self.n_starts += 1
        self.age = 0.0
        self._pos = np.array(pos, float)
        self._solved = False

    def stop(self):
        self.active = False

    def set_xy(self, pos, vel, sigma=0.02):
        x, y = float(pos[0]), float(pos[1])
        e, J = self.lay.excess_jac(x, y)
        self.e = np.array(e, float)
        self.v = J @ np.asarray(vel, float)
        for s in range(self.S):
            self.Pm[s] = np.diag([sigma ** 2, 0.1 ** 2])
        self._pos = np.array([x, y])
        self._J = J
        self._solved = True

    def excess(self):
        return self.e.copy()

    def _gn(self, x, y, e, prior, iters):
        lay = self.lay
        w = (SIG_E / SIG_P) ** 2
        for _ in range(iters):
            ee, J = lay.excess_jac(x, y)
            r = ee - e
            A = J.T @ J + (w + 1e-6) * np.eye(2)
            g = J.T @ r + w * np.array([x - prior[0], y - prior[1]])
            step = -np.linalg.solve(A, g)
            n = math.hypot(step[0], step[1])
            if n > 0.08:
                step *= 0.08 / n
            x += step[0]
            y += step[1]
            if n < 2e-5:
                break
        ee, _ = lay.excess_jac(x, y)
        cost = float(np.sum((ee - e) ** 2)) / SIG_E ** 2 + ((x - prior[0]) ** 2 + (y - prior[1]) ** 2) / SIG_P ** 2
        return x, y, cost

    def _solve(self):
        if self._solved:
            return
        lay, cfg = self.lay, self.cfg
        x, y = float(self._pos[0]), float(self._pos[1])
        e = self.e
        if self.S == 1:
            for _ in range(4):
                ee, J = lay.excess_jac(x, y)
                g = J[0]
                step = -(ee[0] - e[0]) * g / max(float(g @ g), 1e-9)
                step[0] = -x
                n = math.hypot(step[0], step[1])
                if n > 0.08:
                    step *= 0.08 / n
                x += step[0]
                y += step[1]
        else:
            prior = (x, y)
            best = None
            for sx, sy in ((x, y), (x, max(y, 0.30)), (0.0, 0.35)):
                cx, cy, cost = self._gn(sx, sy, e, prior, 8)
                valid = cy > 0.07 and cfg.x_limits[0] - 0.05 < cx < cfg.x_limits[1] + 0.05
                if best is None or (not valid, cost) < best[0]:
                    best = ((not valid, cost), cx, cy)
            x, y = best[1], best[2]
        x = min(max(x, cfg.x_limits[0]), cfg.x_limits[1])
        y = min(max(y, cfg.y_limits[0]), cfg.y_limits[1])
        self._pos = np.array([x, y])
        self._J = lay.excess_jac(x, y)[1]
        self._solved = True

    @property
    def pos(self):
        self._solve()
        return self._pos.copy()

    @property
    def vel(self):
        self._solve()
        J = self._J
        if self.S == 1:
            g = J[0]
            return np.array([0.0, self.v[0] / g[1] if abs(g[1]) > 1e-6 else 0.0])
        try:
            return np.linalg.solve(J, self.v)
        except np.linalg.LinAlgError:
            return np.zeros(2)

    def cov(self):
        self._solve()
        J = self._J
        if self.S == 1:
            g = J[0][1] if abs(J[0][1]) > 1e-6 else 1.0
            return np.diag([1e-6, self.Pm[0, 0, 0] / g ** 2])
        try:
            Ji = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            return np.eye(2) * 0.01
        return Ji @ np.diag(self.Pm[:, 0, 0]) @ Ji.T
