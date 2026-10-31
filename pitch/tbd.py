import math

import numpy as np


class GridTBD:
    def __init__(self, layout, fe, cfg, dt):
        self.lay = layout
        self.cfg = cfg
        self.dt = dt
        step = cfg.tbd_cell_m
        self.xs = np.arange(-cfg.tbd_xmax, cfg.tbd_xmax + 1e-9, step)
        self.ys = np.arange(cfg.tbd_ymin, min(cfg.y_limits[1], cfg.tbd_ymax) + 1e-9, step)
        self.X, self.Y = np.meshgrid(self.xs, self.ys)
        self.S = layout.S
        self.mpt = fe.m_per_tap
        self.rebuild()
        self.score = np.zeros(self.X.shape)
        self.occ = np.zeros(self.X.shape)
        self.occ_n = 0
        self.llr = np.zeros(self.X.shape)

    def rebuild(self):
        ny, nx = self.X.shape
        e = np.empty((self.S, ny, nx))
        for j in range(ny):
            for i in range(nx):
                e[:, j, i] = self.lay.excess(self.X[j, i], self.Y[j, i])
        self.valid = np.all((e > self.cfg.gate_min) & (e < self.cfg.gate_max), axis=0)
        self.idx = np.clip(np.round(e / self.mpt).astype(int), 0, None)

    def _log_snr(self, snr):
        cfg = self.cfg
        ls = [np.clip(np.log(np.maximum(snr[s][np.minimum(self.idx[s], snr.shape[1] - 1)], 1e-6)),
                      cfg.tbd_clip_lo, cfg.tbd_clip) for s in range(self.S)]
        return np.sum(ls, axis=0)

    def step(self, snr, protect=None):
        cfg = self.cfg
        llr = self._log_snr(snr)
        llr -= self.S * cfg.tbd_bias
        llr[~self.valid] = -10.0
        llr -= cfg.tbd_near_prior * self.Y
        self.occ_n += 1
        a = max(1.0 / self.occ_n, self.dt / cfg.occ_tau)
        w = np.ones(self.X.shape)
        if protect is not None:
            px, py, pr = protect
            w = ((self.X - px) ** 2 + (self.Y - py) ** 2 > pr * pr).astype(float)
        self.occ += (a * w) * (np.maximum(llr, 0.0) - self.occ)
        llr = llr - w * self.occ
        self.llr = llr

        sc = self.score
        m = sc.copy()
        m[1:, :] = np.maximum(m[1:, :], sc[:-1, :])
        m[:-1, :] = np.maximum(m[:-1, :], sc[1:, :])
        m2 = m.copy()
        m2[:, 1:] = np.maximum(m2[:, 1:], m[:, :-1])
        m2[:, :-1] = np.maximum(m2[:, :-1], m[:, 1:])
        self.score = np.maximum(0.0, cfg.tbd_lambda * m2 + llr)

        j, i = np.unravel_index(int(np.argmax(self.score)), self.score.shape)
        best = float(self.score[j, i])
        j0, j1 = max(j - 1, 0), min(j + 2, self.score.shape[0])
        i0, i1 = max(i - 1, 0), min(i + 2, self.score.shape[1])
        w = self.score[j0:j1, i0:i1]
        if w.sum() > 0:
            x = float((w * self.X[j0:j1, i0:i1]).sum() / w.sum())
            y = float((w * self.Y[j0:j1, i0:i1]).sum() / w.sum())
        else:
            x, y = float(self.X[j, i]), float(self.Y[j, i])
        return x, y, best

    def peaks(self, k, sep, thr):
        sc = self.score
        m = sc.copy()
        m[1:, :] = np.maximum(m[1:, :], sc[:-1, :])
        m[:-1, :] = np.maximum(m[:-1, :], sc[1:, :])
        m2 = m.copy()
        m2[:, 1:] = np.maximum(m2[:, 1:], m[:, :-1])
        m2[:, :-1] = np.maximum(m2[:, :-1], m[:, 1:])
        jj, ii = np.nonzero((sc >= m2) & (sc > thr))
        if len(jj) == 0:
            return []
        out = []
        for o in np.argsort(-sc[jj, ii]):
            px, py = float(self.X[jj[o], ii[o]]), float(self.Y[jj[o], ii[o]])
            if all((px - q[0]) ** 2 + (py - q[1]) ** 2 >= sep * sep for q in out):
                out.append((px, py, float(sc[jj[o], ii[o]])))
                if len(out) >= k:
                    break
        return out

    def evidence(self, snr):
        llr = self._log_snr(snr) - self.S * self.cfg.tbd_bias - self.cfg.tbd_near_prior * self.Y
        llr[~self.valid] = -10.0
        return llr

    def gated_peak(self, M, px, py, sx2, sy2, gate2, rmin, rmax, prior_w):
        dx = float(self.xs[1] - self.xs[0])
        r = min(rmax, max(rmin, math.sqrt(gate2 * max(sx2, sy2))))
        ri = int(math.ceil(r / dx))
        i = int(round((px - self.xs[0]) / dx))
        j = int(round((py - self.ys[0]) / dx))
        i0, i1 = max(i - ri, 0), min(i + ri + 1, len(self.xs))
        j0, j1 = max(j - ri, 0), min(j + ri + 1, len(self.ys))
        if i0 >= i1 or j0 >= j1:
            return None
        X = self.xs[i0:i1][None, :] - px
        Y = self.ys[j0:j1][:, None] - py
        rr = X * X + Y * Y
        d2 = X * X / sx2 + Y * Y / sy2
        inside = (rr <= r * r) & ((d2 <= gate2) | (rr <= rmin * rmin))
        w = M[j0:j1, i0:i1]
        sc = np.where(inside, w - 0.5 * prior_w * np.minimum(d2, gate2), -np.inf)
        k = int(np.argmax(sc))
        if not np.isfinite(sc.flat[k]):
            return None
        jj, ii = divmod(k, i1 - i0)
        jc, ic = jj + j0, ii + i0
        ev = float(M[jc, ic])
        a0, a1 = max(jc - 1, 0), min(jc + 2, len(self.ys))
        b0, b1 = max(ic - 1, 0), min(ic + 2, len(self.xs))
        ww = np.maximum(M[a0:a1, b0:b1], 0.0)
        s = float(ww.sum())
        self.last_cell = (jc, ic)
        if s > 0:
            return (float((ww * self.X[a0:a1, b0:b1]).sum() / s), float((ww * self.Y[a0:a1, b0:b1]).sum() / s), ev)
        return float(self.X[jc, ic]), float(self.Y[jc, ic]), ev
