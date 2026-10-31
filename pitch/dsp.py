import numpy as np


class FrontEnd:
    def __init__(self, cfg, table, n_mics):
        self.N = table.n
        self.fs = table.fs
        self.S, self.K = table.S, table.K
        self.M = n_mics
        self.bins = table.bins
        self.kf = table.bins.astype(np.float64)
        self.invX = 1.0 / table.X
        self.step = table.bin_step
        self.L = int(cfg.cir_len)
        self.win = np.kaiser(self.K, cfg.kaiser_beta)
        self.win_norm = float(self.win.sum())
        self.recenter = np.exp(-1j * np.pi * (self.K - 1) * np.arange(self.L) / self.L)
        self.samples_per_tap = self.N / (self.step * self.L)
        self.m_per_tap = self.samples_per_tap * cfg.c / self.fs
        self.alias_m = self.N / self.step * cfg.c / self.fs
        self.lam = cfg.c / table.fc
        gmax = min(cfg.gate_max, 0.85 * self.alias_m)
        self.gate_lo = int(np.ceil(cfg.gate_min / self.m_per_tap))
        self.gate_hi = int(np.floor(gmax / self.m_per_tap))
        self.shift = np.ones((self.S, self.M, self.K), complex)
        self.eps = 0.0
        self._E = None
        self._Efull = None
        self._taper = {}
        bw = self.K * self.step * self.fs / self.N
        self.resolution_m = 0.89 * cfg.c / bw * (1.0 + 0.11 * cfg.kaiser_beta)

    def set_drift(self, eps):
        q = float(eps)
        if self._Efull is not None and abs(q - self.eps) < 0.75e-6:
            return
        self.eps = q
        n = np.arange(self.N)[:, None]
        f = (self.bins.reshape(-1) * (1.0 - q))[None, :]
        self._Efull = np.exp(-2j * np.pi * n * f / self.N).astype(np.complex64)
        self._E = None if abs(q) < 2e-7 else self._Efull

    def dft(self, frames, starts):
        B, M = frames.shape[0], frames.shape[1]
        if self._E is None:
            Yb = np.fft.rfft(frames, axis=-1)[:, :, self.bins]
        else:
            Yb = (frames.astype(np.complex64) @ self._E).astype(np.complex128)
            Yb = Yb.reshape(B, M, self.S, self.K)
        Yb = np.moveaxis(Yb, 2, 1)
        off = (np.asarray(starts) % self.N).astype(np.float64)
        ph = np.exp(-2j * np.pi * off[:, None, None] * self.kf[None] / self.N)
        return Yb, ph

    def channel(self, frames, starts):
        Yb, ph = self.dft(frames, starts)
        return Yb * (ph * self.invX[None])[:, :, None, :]

    def set_reference(self, tau0):
        tau0 = np.array(tau0, dtype=np.float64)
        self.shift = np.exp(2j * np.pi * self.kf[:, None, :] * tau0[:, :, None] / self.N)

    def delay(self, H, delta):
        d = np.asarray(delta, dtype=np.float64)
        d = d.reshape(d.shape + (1, 1, 1))
        return H * np.exp(2j * np.pi * self.kf[:, None, :] * d / self.N)

    def estimate_delay(self, H, Href, lag_win=0.0):
        C = H * np.conj(Href)[None]
        d = self._kay(C)
        if lag_win > 0:
            n = int(min(self.L, 1 << int(np.ceil(np.log2(4 * self.K)))))
            taper = self._lag_taper(lag_win, n)
            for _ in range(2):
                Cs = self.delay(C, d)
                Cw = np.fft.fft(np.fft.ifft(Cs * self.win, n=n, axis=-1) * taper, axis=-1)[..., :self.K]
                d = d + self._kay(Cw)
        return d

    def _kay(self, C):
        R = np.sum(C[..., 1:] * np.conj(C[..., :-1]), axis=(1, 2, 3))
        return -np.angle(R) * self.N / (2.0 * np.pi * self.step)

    def _lag_taper(self, lag_win, n):
        key = (float(lag_win), int(n))
        if key not in self._taper:
            t = np.arange(n)
            lag = np.minimum(t, n - t) * (self.N / (self.step * n))
            w = np.clip((2.0 * key[0] - lag) / key[0], 0.0, 1.0)
            self._taper[key] = 0.5 - 0.5 * np.cos(np.pi * w)
        return self._taper[key]

    @staticmethod
    def estimate_gain(H, Href):
        num = np.sum(H * np.conj(Href)[None], axis=-1)
        den = np.sum(np.abs(Href) ** 2, axis=-1)
        return num / np.maximum(den, 1e-30)[None]

    def wrap_delay(self, d):
        half = self.N / (2.0 * self.step)
        return (np.asarray(d) + half) % (2 * half) - half

    def cir(self, H):
        h = np.fft.ifft(H * self.win, n=self.L, axis=-1) * (self.L / self.win_norm)
        return h * self.recenter

    def find_reference_taps(self, h):
        mag = np.abs(h)
        S, M, L = mag.shape
        out = np.zeros((S, M))
        look = int(0.45 / self.m_per_tap)
        for s in range(S):
            for m in range(M):
                x = mag[s, m]
                p = int(np.argmax(x))
                best = p
                for back in range(2, look):
                    i = (p - back) % L
                    if x[i] >= 0.5 * x[p] and x[i] >= x[(i - 1) % L] and x[i] >= x[(i + 1) % L]:
                        best = i
                frac = parabolic(x[(best - 1) % L], x[best], x[(best + 1) % L])
                out[s, m] = (best + frac) * self.samples_per_tap
        return out


def parabolic(a, b, c):
    den = a - 2 * b + c
    if den >= 0 or not np.isfinite(den):
        return 0.0
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def db(x, floor=1e-12):
    return 10.0 * np.log10(np.maximum(x, floor))
