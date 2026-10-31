import math

import numpy as np


class Layout:
    def __init__(self, geom, stereo=True):
        self.mic = np.array(geom.mic, float)
        if stereo:
            self.spk = np.array([geom.spk_left, geom.spk_right], float)
        else:
            self.spk = np.array([np.mean([geom.spk_left, geom.spk_right], axis=0)], float)
        self.S = len(self.spk)
        self.z = float(geom.hand_z)
        self.direct = np.linalg.norm(self.spk - self.mic, axis=1)
        self._mic_t = tuple(float(v) for v in self.mic)
        self._spk_t = [tuple(float(v) for v in p) for p in self.spk]
        self.p_ref = self.direct.copy()
        self._iso_cache = {}
        self._inner = {}

    def point(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return np.stack([x, y, np.full(np.broadcast(x, y).shape, self.z)], axis=-1)

    def paths(self, x, y):
        P = self.point(x, y)[..., None, :]
        d1 = np.linalg.norm(P - self.spk, axis=-1)
        d2 = np.linalg.norm(P[..., 0, :] - self.mic, axis=-1)[..., None]
        return d1 + d2

    def excess(self, x, y):
        return self.paths(x, y) - self.p_ref

    def excess_jac(self, x, y):
        z = self.z
        mx, my, mz = self._mic_t
        dxm, dym, dzm = x - mx, y - my, z - mz
        r2 = math.sqrt(dxm * dxm + dym * dym + dzm * dzm) or 1e-9
        e = np.empty(self.S)
        J = np.empty((self.S, 2))
        for s, (sx, sy, sz) in enumerate(self._spk_t):
            dx, dy, dz = x - sx, y - sy, z - sz
            r1 = math.sqrt(dx * dx + dy * dy + dz * dz) or 1e-9
            e[s] = r1 + r2 - self.p_ref[s]
            J[s, 0] = dx / r1 + dxm / r2
            J[s, 1] = dy / r1 + dym / r2
        return e, J

    def jacobian(self, x, y):
        P = np.array([x, y, self.z])
        u1 = (P - self.spk) / np.linalg.norm(P - self.spk, axis=1)[:, None]
        u2 = (P - self.mic) / np.linalg.norm(P - self.mic)
        return (u1 + u2[None])[:, :2]

    def solve(self, e, x0=(0.0, 0.3), iters=12):
        e = np.asarray(e, float)
        ok = np.isfinite(e)
        if ok.sum() == 0:
            return None
        x, y = float(x0[0]), float(x0[1])
        for _ in range(iters):
            r = (self.excess(x, y) - e)[ok]
            J = self.jacobian(x, y)[ok]
            if ok.sum() == 1:
                g = J[0]
                step = -r[0] * g / max(g @ g, 1e-9)
            else:
                A = J.T @ J + 1e-6 * np.eye(2)
                step = -np.linalg.solve(A, J.T @ r)
            n = np.linalg.norm(step)
            if n > 0.1:
                step *= 0.1 / n
            x += step[0]
            y += step[1]
            y = max(y, 0.0)
            if n < 1e-5:
                break
        return np.array([x, y])

    def solve2(self, e, x0, iters=6, tol=1e-5):
        if self.S != 2 or not (math.isfinite(e[0]) and math.isfinite(e[1])):
            return self.solve(e, x0=x0)
        x, y = float(x0[0]), float(x0[1])
        e0, e1 = float(e[0]), float(e[1])
        for _ in range(iters):
            ee, J = self.excess_jac(x, y)
            r0, r1 = ee[0] - e0, ee[1] - e1
            a, b, c, d = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
            det = a * d - b * c
            if abs(det) < 1e-9:
                return self.solve(e, x0=(x, y))
            dx = -(d * r0 - b * r1) / det
            dy = -(-c * r0 + a * r1) / det
            n = math.hypot(dx, dy)
            if n > 0.1:
                dx, dy = dx * 0.1 / n, dy * 0.1 / n
            x += dx
            y = max(y + dy, 0.0)
            if n < tol:
                break
        return np.array([x, y])

    def iso_curve(self, s, path_len, n=220):
        key = (s, int(round(path_len * 1000)), n)
        if key not in self._iso_cache:
            if len(self._iso_cache) > 4000:
                self._iso_cache.clear()
            self._iso_cache[key] = self._iso_curve(s, key[1] / 1000.0, n)
        return self._iso_cache[key]

    def _iso_curve(self, s, path_len, n):
        cx, cy = self._inner_point(s)
        if self.paths(cx, cy)[s] >= path_len:
            return None
        sx, sy, sz = self._spk_t[s]
        mx, my, mz = self._mic_t
        dz1, dz2 = (self.z - sz) ** 2, (self.z - mz) ** 2

        def path(x, y):
            return np.sqrt((x - sx) ** 2 + (y - sy) ** 2 + dz1) + np.sqrt((x - mx) ** 2 + (y - my) ** 2 + dz2)

        th = np.linspace(0, 2 * np.pi, n)
        ux, uy = np.cos(th), np.sin(th)
        lo = np.zeros(n)
        hi = np.full(n, 0.05)
        for _ in range(12):
            f = path(cx + hi * ux, cy + hi * uy) - path_len
            if np.all(f > 0):
                break
            hi = np.where(f > 0, hi, hi * 2.0)
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            inside = path(cx + mid * ux, cy + mid * uy) < path_len
            lo = np.where(inside, mid, lo)
            hi = np.where(inside, hi, mid)
        r = 0.5 * (lo + hi)
        return np.stack([cx + r * ux, cy + r * uy], axis=1)

    def _inner_point(self, s):
        if s not in self._inner:
            xs = np.linspace(-0.4, 0.4, 81)
            ys = np.linspace(-0.2, 0.5, 71)
            X, Y = np.meshgrid(xs, ys)
            p = self.paths(X, Y)[..., s]
            i = np.unravel_index(np.argmin(p), p.shape)
            self._inner[s] = (float(X[i]), float(Y[i]))
        return self._inner[s]
