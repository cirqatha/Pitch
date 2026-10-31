import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class ProbeTable:
    n: int
    fs: int
    bins: np.ndarray
    X: np.ndarray
    wave: np.ndarray

    @property
    def S(self):
        return self.bins.shape[0]

    @property
    def K(self):
        return self.bins.shape[1]

    @property
    def freqs(self):
        return self.bins * (self.fs / self.n)

    @property
    def fc(self):
        return self.freqs.mean(axis=1)

    @property
    def bin_step(self):
        if self.K < 2:
            return 1
        return int(self.bins[0, 1] - self.bins[0, 0])


def silence(n, fs):
    return ProbeTable(n, fs, np.zeros((0, 0), int), np.zeros((0, 0), complex), np.zeros((n, 2), np.float32))


def tone_bins(fs, n, f_lo, f_hi):
    df = fs / n
    lo = int(np.ceil(f_lo / df))
    hi = min(int(np.floor(f_hi / df)), n // 2 - 2)
    return np.arange(lo, hi + 1)


def random_phases(n, bins, seed, iters=80):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, len(bins))
    spec = np.zeros(n // 2 + 1, complex)
    for _ in range(iters):
        spec[:] = 0
        spec[bins] = np.exp(1j * theta)
        x = np.fft.irfft(spec, n)
        lim = 1.5 * np.sqrt(np.mean(x * x))
        theta = np.angle(np.fft.rfft(np.clip(x, -lim, lim))[bins])
    return theta


def make_probe(fs, n, f_lo, f_hi, level, stereo=True):
    ks = tone_bins(fs, n, f_lo, f_hi)
    nspk = 2 if stereo else 1
    ks = ks[:len(ks) - (len(ks) % nspk)]
    if len(ks) < 2 * nspk:
        raise ValueError(f"band {f_lo:.0f}-{f_hi:.0f} Hz is too narrow")
    bins = np.stack([ks[s::nspk] for s in range(nspk)])
    S, K = bins.shape
    wave = np.zeros((n, 2))
    X = np.zeros((S, K), complex)
    for s in range(S):
        spec = np.zeros(n // 2 + 1, complex)
        spec[bins[s]] = np.exp(1j * random_phases(n, bins[s], seed=7919 * (s + 1) + K))
        x = np.fft.irfft(spec, n)
        x *= level / np.max(np.abs(x))
        if stereo:
            wave[:, s] = x
        else:
            wave[:, 0] = x
            wave[:, 1] = x
        X[s] = np.fft.rfft(x)[bins[s]]
    return ProbeTable(n, fs, bins, X, wave.astype(np.float32))


class ProbePlayer:
    def __init__(self, table, xfade=1024, fade_in=4096):
        self.n = table.n
        self.cur = table
        self.prev = None
        self.xfade_left = 0
        self.xfade = xfade
        self.pos = 0
        self.gain = 0.0
        self.gain_step = 1.0 / fade_in
        self.target_gain = 1.0
        self.volume = 1.0
        self._pending = None
        self._lock = threading.Lock()

    def request(self, table):
        with self._lock:
            self._pending = table

    def fade_out(self):
        self.target_gain = 0.0

    def render(self, frames):
        with self._lock:
            pend, self._pending = self._pending, None
        if pend is not None and pend is not self.cur:
            self.prev = self.cur
            self.cur = pend
            self.xfade_left = self.xfade
        idx = (self.pos + np.arange(frames)) % self.n
        out = self.cur.wave[idx].astype(np.float32)
        if self.prev is not None and self.xfade_left > 0:
            done = self.xfade - self.xfade_left
            r = np.clip((done + np.arange(frames)) / self.xfade, 0.0, 1.0)[:, None].astype(np.float32)
            out = r * out + (1.0 - r) * self.prev.wave[idx]
            self.xfade_left = max(0, self.xfade_left - frames)
            if self.xfade_left == 0:
                self.prev = None
        g = self.gain + self.gain_step * np.arange(1, frames + 1) * np.sign(self.target_gain - self.gain)
        g = np.clip(g, min(self.gain, self.target_gain), max(self.gain, self.target_gain)).astype(np.float32)
        self.gain = float(g[-1])
        out *= (g * self.volume)[:, None]
        self.pos += frames
        return out
