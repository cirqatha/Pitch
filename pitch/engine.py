import threading
import time
from collections import deque

import numpy as np

from .dsp import FrontEnd, db, parabolic
from .geometry import Layout
from .probe import make_probe, silence
from .tbd import GridTBD
from .tracker import HandTracker

STEPS = [
    ("noise", "Listening to the room"),
    ("sweep", "Sweeping the speakers"),
    ("background", "Mapping the static echoes"),
    ("anchor", "Palm 20 cm in front of the screen"),
]

DISP_MAX_M = 0.9
WATERFALL_W = 480
LEAK_RATIO = 4.0


class Ring:
    def __init__(self, M, cap):
        self.buf = np.zeros((M, cap), np.float32)
        self.cap = cap
        self.total = 0

    def push(self, x):
        n = len(x)
        if n >= self.cap:
            x = x[-self.cap:]
            self.total += n - self.cap
            n = self.cap
        i = self.total % self.cap
        a = min(n, self.cap - i)
        self.buf[:, i:i + a] = x[:a].T
        if a < n:
            self.buf[:, :n - a] = x[a:].T
        self.total += n

    def frames(self, starts, N):
        idx = (starts[:, None] + np.arange(N)[None]) % self.cap
        return np.transpose(self.buf[:, idx], (1, 0, 2))


class XYHist:
    def __init__(self, n):
        self.a = np.zeros((max(int(n), 2), 2))
        self.i = 0
        self.n = 0

    def append(self, p):
        self.a[self.i] = p
        self.i = (self.i + 1) % len(self.a)
        self.n = min(self.n + 1, len(self.a))

    def __len__(self):
        return self.n

    def last(self, m):
        m = min(int(m), self.n)
        j = self.i - m
        if j >= 0:
            return self.a[j:self.i]
        return np.concatenate([self.a[j:], self.a[:self.i]])


def smoothing_kernel(n, sigma):
    t = np.arange(n)
    t = np.minimum(t, n - t)
    k = np.exp(-0.5 * (t / max(sigma, 1e-3)) ** 2)
    return np.fft.rfft(k / k.sum())


class Engine:
    def __init__(self, cfg, fs, n_mics):
        self.cfg = cfg
        self.fs = int(fs)
        self.N = cfg.n_fft
        self.hop = cfg.hop
        self.dt = self.hop / self.fs
        self.M = int(n_mics)
        self.info = {}
        self.lock = threading.Lock()
        self.ring = Ring(self.M, 16 * self.N)
        self.level = cfg.probe_level
        self._make_sweep()
        self.silence = silence(self.N, self.fs)
        self.version = 0
        self.desired = self.silence
        self.proc_ms = deque(maxlen=200)
        self._wall = deque(maxlen=200)
        self._dr_last = None
        self._glitch_left = 0
        self._anchor_gap = 0
        self._dop_mats = {}
        self.last_profile = None
        self.restart()

    def restart(self):
        with self.lock:
            if abs(self.level - self.cfg.probe_level) > 1e-9:
                self.level = self.cfg.probe_level
                self._make_sweep()
            self.state = "noise"
            self.state_t0 = self.ring.total
            self.next_frame = self.ring.total + int(0.15 * self.fs)
            self._set_output(self.silence)
            self.step_status = {k: "pending" for k, _ in STEPS}
            self.step_status["noise"] = "active"
            self.message = "Hands away from the laptop and keep still"
            self.error = None
            self.warnings = []
            self.progress = 0.0
            self.acc = []
            self.noise_pow = None
            self.latency_s = None
            self.request_at = None
            self.onset_at = None
            self.response = None
            self.band = None
            self.fe = None
            self.layout = None
            self.tracker = None
            self.tbd = None
            self.bg_retries = 0
            self.anchor_skip = False
            self.anchor_buf = []
            self.anchor_ok = None
            self._anc_avg = None
            self.det = None
            self.drift_ppm = 0.0
            self.eps = 0.0
            self._stack_eps = 0.0
            self._skew_d = 0.0
            self._skew_n = 0
            self.delta_acc = 0.0
            self._held = None
            self._lost = None
            self._ended = None
            self._dop_spec = None
            self._dop_den = None
            self._last_dop = None
            self._too_loud_at = None
            self._level_refined = False
            self._ps_acc = None
            self.frame_i = 0
            self.wf = None
            self.wf_ptr = 0
            self.wf_track = None

    def _make_sweep(self):
        cfg = self.cfg
        hi = min(cfg.sweep_hi, 0.485 * self.fs)
        self.sweep_table = make_probe(self.fs, self.N, cfg.sweep_lo, hi, self.level, True)
        self.fe_sweep = FrontEnd(cfg, self.sweep_table, self.M)

    def _band_powers(self, frames):
        P = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[-1]), axis=-1)) ** 2
        k = lambda f: int(f * self.N / self.fs)
        lo = self.cfg.sweep_lo
        return (float(np.median(P[..., k(min(2000, 0.3 * lo)):k(0.85 * lo)])),
                float(np.median(P[..., k(lo):k(min(self.cfg.sweep_hi, 0.48 * self.fs))])))

    def skip_anchor(self):
        self.anchor_skip = True

    def reanchor(self):
        with self.lock:
            if self.state != "tracking":
                return False
            self.anchor_skip = False
            self.anchor_buf = []
            self._anc_avg = None
            self.tracker.stop()
            self.step_status["anchor"] = "active"
            self._goto("anchor", "Hold your open palm 20 cm in front of the middle of the screen, still")
            return True

    def _set_output(self, table):
        self.desired = table
        self.version += 1
        self.request_at = self.ring.total

    def _goto(self, state, message=""):
        self.state = state
        self.state_t0 = self.ring.total
        self.acc = []
        if state in self.step_status:
            self.step_status[state] = "active"
        if message:
            self.message = message

    def _fail(self, msg, hints):
        self.state = "error"
        self.error = {"message": msg, "hints": hints}
        self.message = msg
        self._set_output(self.silence)

    def feed(self, block):
        block = block[:, :self.M]
        t0 = time.perf_counter()
        with self.lock:
            self.ring.push(block.astype(np.float32, copy=False))
            avail = self.ring.total - self.N
            if avail - self.next_frame > int(0.25 * self.fs):
                skip = (avail - self.next_frame) // self.hop * self.hop
                self.next_frame += skip
            if self.next_frame < self.ring.total - self.ring.cap + self.N:
                self.next_frame = self.ring.total - self.ring.cap + self.N
            nb = 0
            while self.next_frame <= avail:
                count = min(8, (avail - self.next_frame) // self.hop + 1)
                starts = self.next_frame + self.hop * np.arange(count)
                frames = self.ring.frames(starts, self.N)
                self._process(frames, starts)
                self.next_frame += count * self.hop
                nb += count
            if nb:
                self.proc_ms.append((time.perf_counter() - t0) * 1e3 / nb)
                self._wall.append((time.perf_counter(), nb))

    def _elapsed(self):
        return (self.ring.total - self.state_t0) / self.fs

    def _process(self, frames, starts):
        st = self.state
        if st == "noise":
            self._st_noise(frames)
        elif st == "sweep":
            self._st_sweep(frames, starts)
        elif st == "background":
            self._st_background(frames, starts)
        elif st in ("anchor", "tracking"):
            self._st_track(frames, starts)

    def _st_noise(self, frames):
        Y = np.fft.rfft(frames, axis=-1)[:, :, self.sweep_table.bins]
        self.acc.append((np.abs(Y) ** 2, frames.std(axis=-1)))
        self.progress = min(1.0, self._elapsed() / self.cfg.t_noise)
        if self._elapsed() < self.cfg.t_noise:
            return
        P = np.concatenate([a for a, _ in self.acc])
        rms = np.concatenate([b for _, b in self.acc])
        if np.max(rms) < 1e-6:
            self._fail("The mic is only giving silence",
                       ["Windows: Settings > Privacy & security > Microphone, allow desktop apps",
                        "check the mic isn't muted",
                        "or pick another input: --list-devices, then --in N"])
            return
        self.noise_pow = np.mean(P, axis=0) + 1e-20
        self.noise_low = self._band_powers(frames)[0]
        self.level_checked = False
        self.step_status["noise"] = "done"
        self._set_output(self.sweep_table)
        self.onset_at = None
        self._goto("sweep", "Hands away, sweeping the speakers")

    def _st_sweep(self, frames, starts):
        fe = self.fe_sweep
        if self.onset_at is None:
            Y = np.fft.rfft(frames, axis=-1)[:, :, self.sweep_table.bins]
            ratio = np.median((np.abs(Y) ** 2) / self.noise_pow[None], axis=(1, 2, 3))
            hit = np.nonzero(ratio > 8.0)[0]
            if len(hit):
                self.onset_at = int(starts[hit[0]] + self.N)
                self.latency_s = (self.onset_at - self.request_at) / self.fs
            elif self._elapsed() > self.cfg.latency_max:
                self._fail("The mic can't hear the speakers",
                           ["unmute, set the volume to 50-80% and press R",
                            "unplug headphones / turn off bluetooth audio",
                            "turn off 'Audio enhancements' for the speakers and the mic",
                            "the mic should record at 44.1 or 48 kHz (Sound settings > Advanced)"])
            self.progress = 0.1 * min(1.0, self._elapsed() / self.cfg.latency_max)
            return
        settle = self.onset_at + int(0.12 * self.fs)
        keep = starts >= settle
        if not self.level_checked:
            if self.ring.total < settle + 5 * self.N:
                return
            fr4 = self.ring.frames(settle + self.N * np.arange(4), self.N)
            peak = float(np.max(np.abs(fr4)))
            low, tone = self._band_powers(fr4)
            dist = (low - self.noise_low) / max(tone, 1e-20)
            X = np.fft.rfft(fr4, axis=-1)[..., self.sweep_table.bins]
            good = np.mean(np.abs(X) ** 2, axis=0) > 100.0 * self.noise_pow
            if good.sum() >= 20:
                Xg = X[:, good]
                coh = float(np.mean([abs(np.vdot(Xg[i], Xg[i + 1])) /
                                     (np.linalg.norm(Xg[i]) * np.linalg.norm(Xg[i + 1]) + 1e-30) for i in range(3)]))
            else:
                coh = 1.0
            distorted = dist > 1e-2 and low > 10 * self.noise_low
            if peak > 0.8 or distorted or coh < 0.95:
                self._too_loud_at = self.level
                lvl = self.level * (0.25 if peak > 0.97 else 0.4)
                if lvl < 0.004 and peak <= 0.8:
                    self.level_checked = True
                    self.warnings.append("Couldn't get a clean probe, tracking might be noisy")
                    return
                if lvl < 0.004:
                    self._fail("The speakers or the mic are overloaded",
                               ["turn the volume down to 30-50% and press R",
                                "turn off mic boost (Sound settings > Recording > Levels)",
                                "turn off audio effects (Nahimic, Dolby, loudness)"])
                    return
                self.level = lvl
                self._set_level_warning(lvl, peak)
                self._make_sweep()
                self._set_output(self.sweep_table)
                self.onset_at = self.request_at + int(((self.latency_s or 0.15) + 0.03) * self.fs) + self.N
                self.acc = []
                self.state_t0 = self.ring.total
                self.message = "Finding the right probe level"
                return
            if not self._level_refined and self._too_loud_at:
                if 1e-4 < peak < 0.6 * self.cfg.level_target_peak / 0.5:
                    up = min(self.cfg.level_target_peak / peak, 0.8 * self._too_loud_at / self.level)
                else:
                    up = 1.0
                if up > 1.15:
                    self._level_refined = True
                    old = f"{self.level:.3f}"
                    self.level *= up
                    self.warnings = [w.replace(old, f"{self.level:.3f}", 1) if w.startswith("Probe level") else w
                                     for w in self.warnings]
                    self._make_sweep()
                    self._set_output(self.sweep_table)
                    self.onset_at = self.request_at + int(((self.latency_s or 0.15) + 0.03) * self.fs) + self.N
                    self.acc = []
                    self.state_t0 = self.ring.total
                    self.message = "Finding the right probe level"
                    return
            self.level_checked = True
        if np.any(keep):
            self.acc.append((fe.channel(frames[keep], starts[keep]), starts[keep]))
        got = sum(len(a[1]) for a in self.acc) * self.hop / self.fs
        self.progress = 0.1 + 0.9 * min(1.0, got / self.cfg.t_sweep)
        if got < self.cfg.t_sweep:
            return
        H = np.concatenate([a[0] for a in self.acc])
        st = np.concatenate([a[1] for a in self.acc])
        Havg, Hn = self._align_stack(fe, H, st)
        self.eps = self._stack_eps
        Xp = np.abs(self.sweep_table.X)[:, None, :] ** 2
        g_abs = np.mean(np.abs(fe.estimate_gain(H, Havg)), axis=0)
        sig = np.abs(Havg) ** 2 * Xp * (g_abs ** 2)[..., None]
        mic_noise = np.moveaxis(self.noise_pow, 1, 0)
        snr = db(np.mean(sig, axis=1) / np.mean(mic_noise, axis=1))
        freqs = self.sweep_table.freqs
        self.response = {"freqs": freqs.copy(), "snr_db": snr.copy(), "band": None}
        h = np.abs(fe.cir(Havg)).mean(axis=1)
        good = [np.median(snr[s]) > 6.0 for s in range(2)]
        stereo = all(good)
        if stereo:
            a, b = h[0], h[1]
            corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            pa = int(np.argmax(a)) * fe.samples_per_tap
            pb = int(np.argmax(b)) * fe.samples_per_tap
            if corr > 0.995 and abs(pa - pb) < 0.6:
                stereo = False
                self.warnings.append("Left and right sound the same (mono output?), only tracking distance")
        elif any(good):
            self.warnings.append("Only one speaker is loud enough, only tracking distance")
        else:
            self._fail("The speakers are too quiet above 14 kHz",
                       ["turn the volume up and press R",
                        "turn off audio effects (Dolby, loudness, enhancements)",
                        "unplug headphones / turn off bluetooth audio"])
            return
        band = self._choose_band(freqs, snr if stereo else snr[[int(np.argmax([np.median(x) for x in snr]))]])
        if band is None:
            self._fail("Couldn't find a usable band, the speakers/mic roll off too early",
                       ["turn the volume up and press R",
                        "set the mic to 48 kHz (Sound settings > Recording > Properties > Advanced)",
                        "turn off noise suppression / enhancements on the mic"])
            return
        self.band = band
        self.response["band"] = band
        if band[0] < self.cfg.quiet_above:
            self.warnings.append(f"Probe at {band[0] / 1000:.1f}-{band[1] / 1000:.1f} kHz, these speakers need "
                                 "audible frequencies so a soft hiss is normal (--quiet avoids it)")
        self.probe_table = make_probe(self.fs, self.N, band[0], band[1], self.level, stereo)
        self.fe = FrontEnd(self.cfg, self.probe_table, self.M)
        self.fe.set_drift(self.eps)
        self.drift_ppm = self.eps * 1e6
        self.layout = Layout(self.cfg.geometry, stereo)
        self.tracker = HandTracker(self.layout, self.cfg)
        self.step_status["sweep"] = "done"
        self._set_output(self.probe_table)
        self._goto("background", "Hands away, mapping the static echoes")

    def _set_level_warning(self, lvl, peak):
        self.warnings = [w for w in self.warnings if not w.startswith("Probe level")]
        if peak > 0.8 and lvl < 0.1:
            self.warnings.append(f"Probe level lowered to {lvl:.3f} because the mic is set way too sensitive, so "
                                 "the echoes are weak. Press M (or Settings > System > Sound > Input), set the mic "
                                 "input volume to about 20% and press R, then it can play louder and track a lot better")
        elif peak > 0.8:
            self.warnings.append(f"Probe level lowered to {lvl:.3f} (the mic was clipping)")
        else:
            self.warnings.append(f"Probe level lowered to {lvl:.3f} (the speakers are compressing it)")

    def _choose_band(self, freqs, snr):
        cfg = self.cfg
        f = freqs.mean(axis=0) if freqs.ndim == 2 else freqs
        worst = snr.min(axis=0)
        best = None
        for w in np.arange(cfg.band_target, cfg.band_min - 1, -500.0):
            for lo in np.arange(cfg.band_floor, f.max() - w + 1, 250.0):
                sel = (f >= lo) & (f <= lo + w)
                if sel.sum() < 6:
                    continue
                q = np.percentile(worst[sel], 20)
                if q < cfg.snr_band_req:
                    continue
                score = (4.0 * min(w, 6000.0) / 1000.0 + 0.3 * (q - cfg.snr_band_req)
                         - cfg.hiss_penalty_db * max(0.0, cfg.quiet_above - lo) / 1000.0
                         + 0.0005 * w)
                if best is None or score > best[0]:
                    best = (score, lo, lo + w)
        return None if best is None else (float(best[1]), float(best[2]))

    def _align_stack(self, fe, H, starts):
        starts = np.asarray(starts)
        good = np.ones(len(H), bool)
        for ref_i in ((0, len(H) // 2) if len(H) >= 8 else (0,)):
            ref = H[ref_i]
            d = fe.estimate_delay(H, ref)
            Ha = fe.delay(H, d)
            if len(H) < 8:
                break
            g = fe.estimate_gain(Ha, Ha[ref_i])
            ga = np.abs(g).mean(axis=(1, 2))
            Hn1 = Ha / g[..., None]
            mu = Hn1.mean(axis=0)
            r = np.mean(np.abs(Hn1 - mu[None]) ** 2, axis=(1, 2, 3))
            gm = np.median(ga)
            bad = (r > 6.0 * np.median(r)) | (ga < 0.7 * gm) | (ga > 1.4 * gm)
            if bad.sum() <= len(H) // 2:
                good = ~bad
                break
        if not good.all():
            H, Ha, d, starts = H[good], Ha[good], d[good], starts[good]
        ref2 = Ha.mean(axis=0)
        d2 = fe.estimate_delay(Ha, ref2)
        Ha = fe.delay(Ha, d2)
        g = fe.estimate_gain(Ha, Ha[0])
        Hn = Ha / g[..., None]
        tot = d + d2
        self._last_stack_delay = float(fe.wrap_delay(tot[-1]))
        if len(starts) > 8:
            dd = np.diff(tot)
            ds = np.diff(starts).astype(float)
            ok = np.abs(dd) < 0.5
            self._stack_eps = float(np.median(dd[ok] / ds[ok])) if ok.sum() > 4 else 0.0
            if ok.sum() > 8:
                self._stack_eps = float(np.sum(dd[ok]) / np.sum(ds[ok]))
        else:
            self._stack_eps = 0.0
        return Hn.mean(axis=0), Hn

    def _st_background(self, frames, starts):
        fe = self.fe
        cfg = self.cfg
        lat = self.latency_s if self.latency_s is not None else 0.3
        settle = self.request_at + int((lat + 0.15) * self.fs)
        keep = starts >= settle
        if np.any(keep):
            self.acc.append((fe.channel(frames[keep], starts[keep]), starts[keep]))
        got = sum(len(a[1]) for a in self.acc) * self.hop / self.fs
        self.progress = min(1.0, got / cfg.t_background)
        if got < cfg.t_background:
            return
        H = np.concatenate([a[0] for a in self.acc])
        st = np.concatenate([a[1] for a in self.acc])
        Havg, Hn = self._align_stack(fe, H, st)
        if abs(self._stack_eps - self.eps) < 300e-6:
            self.eps = 0.5 * (self.eps + self._stack_eps)
        fe.set_drift(self.eps)
        self.drift_ppm = self.eps * 1e6
        tau0 = fe.find_reference_taps(fe.cir(Havg))
        fe.set_reference(tau0)
        Hbg = Havg * fe.shift
        D = Hn * fe.shift[None] - Hbg[None]
        p = np.mean(np.abs(fe.cir(D)) ** 2, axis=2)
        sig_t = cfg.speckle_smooth_m / fe.m_per_tap
        self.smooth_k = smoothing_kernel(fe.L, sig_t)
        p = np.fft.irfft(np.fft.rfft(p, axis=-1) * self.smooth_k, n=fe.L, axis=-1)
        var = p.mean(axis=0)
        g0, g1 = fe.gate_lo, fe.gate_hi
        gv = var[:, g0:g1]
        near = var[:, g0:int(0.6 / fe.m_per_tap)]
        motion = float(np.max(near / np.median(gv, axis=1, keepdims=True)))
        if motion > 60.0 and self.bg_retries < 2:
            self.bg_retries += 1
            self.message = "Something moved, keep your hands away and stay still"
            self.acc = []
            self.state_t0 = self.ring.total
            self.request_at = self.ring.total - int((lat + 0.15) * self.fs)
            return
        if motion > 60.0:
            self.warnings.append("Something moved during calibration, press R with your hands away if it's jumpy")
        floor = np.median(gv, axis=1, keepdims=True)
        self.noise = 2.0 * np.maximum(var, 0.3 * floor)
        self.Hbg = Hbg
        self.hbg = fe.cir(Hbg)
        te = np.arange(fe.L) * fe.m_per_tap
        te = np.where(te > fe.alias_m / 2, te - fe.alias_m, te)
        self.tap_e = te
        self.direct_w = np.nonzero(np.abs(te) < 0.04)[0]
        self.visit = np.full((self.layout.S, fe.L), -1e9)
        self._dop_L = int(min(fe.L, fe.gate_hi + 16))
        self._dop_buf = np.zeros((cfg.dop_win, self.layout.S, self.M, self._dop_L), np.complex64)
        self._dop_n = 0
        self._dop_i = -1
        self._dop_nf = 1 << int(np.ceil(np.log2(self._dop_L + 8 * max(sig_t, 1.0) + 2)))
        self._dop_smooth_k = smoothing_kernel(self._dop_nf, sig_t)
        self._dop_ref = {}
        self.delta_acc = self._last_stack_delay
        self.disp_taps = int(DISP_MAX_M / fe.m_per_tap) // 2 * 2
        self.wf = np.full((self.layout.S, WATERFALL_W, self.disp_taps // 2), -10.0, np.float32)
        self.wf_track = np.full((self.layout.S, WATERFALL_W), np.nan, np.float32)
        self.wf_ptr = 0
        self.det = [None] * self.layout.S
        self.tbd = GridTBD(self.layout, fe, cfg, self.dt)
        self._cands, self._gst = [], None
        self.step_status["background"] = "done"
        if not self.anchor_skip:
            self._goto("anchor", "Hold your open palm 20 cm in front of the middle of the screen, still")
            self.anchor_buf = []
            self._anc_avg = None
        else:
            self._start_tracking()

    def _start_tracking(self):
        if self.step_status["anchor"] == "active":
            self.step_status["anchor"] = "done" if self.anchor_ok else "skipped"
        self._goto("tracking", "Tracking")
        self.progress = 1.0

    def _st_track(self, frames, starts):
        fe = self.fe
        S = self.layout.S
        cfg = self.cfg
        B = len(frames)
        Yb, ph = fe.dft(frames, starts)
        fwd = (ph * fe.invX[None])[:, :, None, :] * fe.shift[None]
        fwd = fe.delay(fwd, self.delta_acc)
        H = Yb * fwd
        dr = fe.estimate_delay(H, self.Hbg, cfg.timing_lag_win)

        good = np.ones(B, bool)
        last = self._dr_last
        for b in range(B):
            ref = last if b == 0 else dr[b - 1]
            if ref is not None and abs(float(fe.wrap_delay(dr[b] - ref))) > cfg.glitch_samples:
                self._glitch_left = self.N // self.hop + 1
            if self._glitch_left > 0:
                good[b] = False
                self._glitch_left -= 1
        self._dr_last = 0.0

        al = 1.0 - np.exp(-self.dt / cfg.timing_tau)
        acc, out = 0.0, np.empty(B)
        for i in range(B):
            e_i = float(dr[i]) - acc
            acc += (e_i if abs(e_i) > cfg.glitch_samples else al * e_i)
            out[i] = acc
        dr = out
        H = fe.delay(H, dr)

        c = fe.cir(H)
        w = self.direct_w
        num = np.sum(c[..., w] * np.conj(self.hbg[None][..., w]), axis=-1)
        den = np.sum(np.abs(self.hbg[..., w]) ** 2, axis=-1)
        g = num / np.maximum(den, 1e-30)[None]
        Hn = H / g[..., None]
        cn = c / g[..., None]

        prev = self.delta_acc
        self.delta_acc = float(fe.wrap_delay(self.delta_acc + dr[-1]))
        jump = float(fe.wrap_delay(self.delta_acc - prev))
        if abs(jump) < 0.5:
            self._skew_d += jump
            self._skew_n += B * self.hop
        if self._skew_n >= self.fs:
            est = self._skew_d / self._skew_n
            if abs(est - self.eps) < 400e-6:
                self.eps += 0.3 * (est - self.eps)
                self.drift_ppm = self.eps * 1e6
                fe.set_drift(self.eps)
            self._skew_d = 0.0
            self._skew_n = 0

        D = Hn - self.Hbg[None]
        d = cn - self.hbg[None]
        p = np.mean(np.abs(d) ** 2, axis=2)
        p_s = np.fft.irfft(np.fft.rfft(p, axis=-1) * self.smooth_k, n=fe.L, axis=-1)
        g0, g1 = fe.gate_lo, fe.gate_hi
        p_s = self._tsmooth(p_s)
        den = np.maximum(self.noise[None], cfg.cfar_kappa * np.median(p_s[:, :, g0:g1], axis=-1, keepdims=True))
        snr = np.maximum(p_s, 1e-30) / den

        tracking = self.state == "tracking"
        for b in range(B):
            if not good[b]:
                continue
            sd = self._doppler_detect(d[b])
            if tracking and sd is not None:
                self._last_dop = sd
                self._frame_mht(sd, sd)
                self._waterfall(sd)
            elif tracking:
                self._frame_mht(snr[b], p_s[b])
                self._waterfall(snr[b])
            else:
                dets = self._acquire(snr[b], p_s[b])
                self._waterfall(snr[b])
                self._anchor_step(dets, d[b])
                if self.state != "anchor":
                    tracking = True

        tb = B * self.dt
        anchoring = self.state == "anchor"
        centres = [np.nan] * S
        if self.tracker.active:
            centres = list(self.tracker.excess())
        elif self.det is not None and anchoring:
            centres = [x["e"] if x else np.nan for x in self.det]
        if not anchoring:
            tau = cfg.ref_tau if self.tracker.active else cfg.bg_tau_idle
            if good.any():
                self.Hbg = self.Hbg + (1 - np.exp(-tb / tau)) * D[good].mean(axis=0)
        now = self.ring.total / self.fs
        for s in range(S):
            if np.isfinite(centres[s]):
                rel = self.tap_e - centres[s]
                self.visit[s, (rel > -0.12) & (rel < cfg.protect_behind_m)] = now
        quiet = (now - self.visit) > cfg.bg_quiet_s
        if anchoring:
            a = np.zeros((S, fe.L))
        else:
            fast = 1 - np.exp(-tb / (cfg.bg_tau_far if self.tracker.active else cfg.bg_tau_idle))
            a = np.where(quiet, fast, 1 - np.exp(-tb / cfg.bg_tau_track))
        if good.any():
            self.hbg = self.hbg + a[:, None, :] * d[good].mean(axis=0)
        pm = p_s[good].mean(axis=0) if good.any() else self.noise
        if anchoring:
            pm = self.noise
        dn = 1 - np.exp(-tb / cfg.floor_down_tau)
        up = 1 - np.exp(-tb / cfg.floor_up_tau)
        self.noise = np.where(pm < self.noise, self.noise + dn * (pm - self.noise),
                              self.noise + up * (np.minimum(pm, 10 * self.noise) - self.noise))
        self.noise = np.maximum(self.noise, 1e-30)
        last = self._last_dop if self.state == "tracking" else None
        if last is None:
            last = snr[-1]
        self.last_profile = db(last[:, :self.disp_taps])

    def _doppler_detect(self, d):
        cfg, fe = self.cfg, self.fe
        W, Lg = cfg.dop_win, self._dop_L
        i = self._dop_i = (self._dop_i + 1) % W
        self._dop_buf[i] = d[..., :Lg]
        self._dop_n += 1
        if self._dop_n < W:
            return None
        g0, g1 = fe.gate_lo, fe.gate_hi
        nf = self._dop_nf
        best = None
        for Wn in (W, cfg.dop_win_fast):
            X = self._dop_dft(Wn, W, i) @ self._dop_buf.reshape(W, -1)
            X = X.reshape((Wn,) + self._dop_buf.shape[1:])
            F = (X.real * X.real + X.imag * X.imag).sum(axis=2)
            if Wn == W:
                self._dop_spec = F
            pdop = F[1:].max(axis=0)
            ps = np.fft.irfft(np.fft.rfft(pdop, n=nf, axis=-1) * self._dop_smooth_k, n=nf, axis=-1)[:, :Lg]
            ps = np.maximum(ps, 1e-30)
            lp = np.log(ps)
            ref = self._dop_ref.get(Wn)
            if ref is None:
                ref = lp.copy()
            else:
                ref += (self.dt / cfg.dop_ref_tau) * (lp - ref)
            self._dop_ref[Wn] = ref
            seg = ps[:, g0:g1]
            m2 = seg.shape[1] // 2
            if seg.shape[1] % 2:
                med = np.partition(seg, m2, axis=-1)[:, m2:m2 + 1]
            else:
                pp = np.partition(seg, (m2 - 1, m2), axis=-1)
                med = 0.5 * (pp[:, m2 - 1:m2] + pp[:, m2:m2 + 1])
            den = np.maximum(np.exp(ref), cfg.dop_kappa * med)
            snr = ps / den
            best = snr if best is None else np.maximum(best, snr)
            if Wn == W:
                self._dop_den = den
        out = np.zeros((best.shape[0], fe.L))
        out[:, :Lg] = best
        return out

    def _dop_dft(self, Wn, W, i):
        key = (Wn, W)
        if key not in self._dop_mats:
            t = np.arange(Wn)
            Dw = np.exp(-2j * np.pi * np.outer(t, t) / Wn) * np.hanning(Wn)[None, :]
            G = Dw - Dw.sum(axis=1, keepdims=True) / Wn
            mats = np.zeros((W, Wn, W), np.complex64)
            for ii in range(W):
                for j in range(W):
                    age = (ii - j) % W
                    if age < Wn:
                        mats[ii, :, j] = G[:, Wn - 1 - age]
            self._dop_mats[key] = mats
        return self._dop_mats[key][i]

    def _tsmooth(self, p):
        k = 1 - np.exp(-self.dt / self.cfg.det_time_tau)
        acc = self._ps_acc
        if acc is None or acc.shape != p.shape[1:]:
            acc = p[0].copy()
        out = np.empty_like(p)
        for b in range(len(p)):
            acc = acc + k * (p[b] - acc)
            out[b] = acc
        self._ps_acc = acc
        return out

    def _detect(self, snr_s, x, mode, leak=None, e_pred=None, width=None, thr_db=None, guard_db=None):
        fe, cfg = self.fe, self.cfg
        g0, g1 = fe.gate_lo, fe.gate_hi
        if mode == "track":
            t = e_pred / fe.m_per_tap
            w = width / fe.m_per_tap
            lo, hi = int(max(g0, t - w)), int(min(g1, t + w))
            if hi - lo < 3:
                return None
            a0, a1 = max(lo - 1, 0), min(hi + 1, len(x))
            seg = x[a0:a1]
            pk = np.nonzero((seg[1:-1] >= seg[:-2]) & (seg[1:-1] > seg[2:]))[0] + 1 + a0
            pk = pk[(pk >= lo) & (pk < hi)]
            pk = pk[snr_s[pk] >= 10 ** (cfg.det_track_db / 10)]
            if leak is not None:
                pk = pk[x[pk] * LEAK_RATIO >= leak[pk]]
            if len(pk) == 0:
                return None
            sig_w = max(width / 2.5, 0.02) / fe.m_per_tap
            score = np.log(np.maximum(x[pk], 1e-30)) - 0.5 * ((pk - t) / sig_w) ** 2
            k = int(pk[int(np.argmax(score))])
        else:
            seg = x[g0 - 1:g1 + 1]
            thr = 10 ** ((cfg.det_acq_db if thr_db is None else thr_db) / 10)
            pk = np.nonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]))[0] + g0
            pk = pk[snr_s[pk] >= thr]
            if leak is not None:
                pk = pk[x[pk] * LEAK_RATIO >= leak[pk]]
            if len(pk) == 0:
                return None
            vals = x[pk]
            lim = vals.max() / 10 ** ((cfg.sidelobe_guard_db if guard_db is None else guard_db) / 10)
            k = int(pk[np.nonzero(vals >= lim)[0][0]])
        a = np.sqrt(max(x[k - 1], 0)) if k > 0 else 0.0
        b = np.sqrt(max(x[k], 0))
        c = np.sqrt(max(x[k + 1], 0)) if k + 1 < len(x) else 0.0
        frac = parabolic(a, b, c)
        return {"e": float((k + frac) * fe.m_per_tap), "snr": float(snr_s[k])}

    @staticmethod
    def _leak(pw, s):
        if len(pw) < 2:
            return None
        return np.max(np.delete(pw, s, axis=0), axis=0)

    def _acquire(self, snr, pw):
        self.frame_i += 1
        self.det = [self._detect(snr[s], pw[s], "acq", self._leak(pw, s)) for s in range(self.layout.S)]
        return self.det

    def _refine(self, snr, pw, lx, ly):
        lay, cfg = self.layout, self.cfg
        e_l = lay.excess(lx, ly)
        e = np.empty(lay.S)
        for s in range(lay.S):
            x = self._detect(snr[s], pw[s], "track", self._leak(pw, s), e_l[s], cfg.refine_win_m)
            if x is None:
                return None
            e[s] = x["e"]
        p = lay.solve2(e, x0=(lx, ly))
        if p is None or np.hypot(p[0] - lx, p[1] - ly) > cfg.refine_agree_m:
            return None
        return np.asarray(p, float)

    def _new_hyp(self, pos):
        cfg = self.cfg
        n = int(max(cfg.conf_s, cfg.stale_s) / self.dt) + 2
        p2, v2 = cfg.kf_init_pos ** 2, cfg.kf_init_vel ** 2
        return {"pos": np.array(pos, float), "vel": np.zeros(2), "low": 0.0, "age": 0.0,
                "hist": XYHist(n), "vh": deque(maxlen=cfg.dop_win), "P": [[p2, 0.0, v2], [p2, 0.0, v2]],
                "ev": 0.0, "score": 0.0, "vote": 0.0, "edge": 0.0, "dom": 0.0, "conf": False, "ev_s": 0.0,
                "front": 0.0, "behind": False, "shown": None}

    def _dop_matched_llr(self, st):
        F, den = self._dop_spec, self._dop_den
        cfg, fe, lay = self.cfg, self.fe, self.layout
        if F is None or den is None:
            return None
        W, Lg = F.shape[0], F.shape[2]
        vh = st["vh"]
        v = vh[-(W // 2 + 1)] if len(vh) > W // 2 else st["vel"]
        e, J = lay.excess_jac(float(st["pos"][0]), float(st["pos"][1]))
        kk = np.fft.fftfreq(W) * W
        out = np.zeros((lay.S, Lg))
        useful = False
        scale = W * self.dt
        for s in range(lay.S):
            k = -float(J[s] @ v) / fe.lam[s] * scale
            sr = float(np.sqrt(J[s, 0] ** 2 * st["P"][0][2] + J[s, 1] ** 2 * st["P"][1][2]))
            tol = max(cfg.dop_match_tol, 2.0 * sr / fe.lam[s] * scale)
            dk = (kk - k + W / 2) % W - W / 2
            sel = (np.abs(dk) <= tol) & (np.abs(kk) >= 1)
            if not sel.any():
                return None
            useful |= abs(k) > tol + 1
            out[s] = F[sel, s, :].max(axis=0)
        if not useful:
            return None
        nf = self._dop_nf
        ps = np.fft.irfft(np.fft.rfft(out, n=nf, axis=-1) * self._dop_smooth_k, n=nf, axis=-1)[:, :Lg]
        snr = np.zeros((lay.S, fe.L))
        snr[:, :Lg] = np.maximum(ps, 1e-30) / den
        return self.tbd.evidence(snr)

    def _kf_step(self, st, snr, pw, refine, M=None):
        cfg, dt, tb = self.cfg, self.dt, self.tbd
        pos, vel, P = st["pos"], st["vel"], st["P"]
        v0 = vel.copy()
        p0 = pos.copy()
        q = cfg.kf_accel ** 2
        dt2 = dt * dt
        for ax in (0, 1):
            a, b, c = P[ax]
            pos[ax] += vel[ax] * dt
            P[ax] = [a + 2 * b * dt + c * dt2 + 0.25 * q * dt2 * dt2, b + c * dt + 0.5 * q * dt2 * dt, c + q * dt2]
        rx2, ry2 = cfg.kf_meas_x ** 2, cfg.kf_meas_y ** 2
        gmax = cfg.kf_gate_max + cfg.reach_v * max(st["low"] - cfg.coast_s, 0.0)
        z = tb.gated_peak(tb.llr if M is None else M, pos[0], pos[1], P[0][0] + rx2, P[1][0] + ry2, cfg.kf_gate ** 2,
                          cfg.kf_gate_min, min(gmax, 2 * cfg.kf_gate_max), cfg.kf_prior_w)
        hit = z is not None and z[2] > cfg.kf_det
        st["ev"] = (z[2] if M is None else float(tb.llr[tb.last_cell])) if z is not None else 0.0
        if hit:
            zx, zy = z[0], z[1]
            k = max(1.0, (cfg.kf_ev_ref / max(z[2], 1e-3)) ** 2)
            if refine:
                ref = self._refine(snr, pw, zx, zy)
                if ref is not None:
                    zx, zy, k = float(ref[0]), float(ref[1]), k * cfg.kf_refine_scale ** 2
            for ax, zz, r2 in ((0, zx, rx2 * k), (1, zy, ry2 * k)):
                a, b, c = P[ax]
                S = a + r2
                k1, k2 = a / S, b / S
                inn = zz - pos[ax]
                pos[ax] += k1 * inn
                vel[ax] += k2 * inn
                P[ax] = [(1 - k1) * a, (1 - k1) * b, c - k2 * b]
            st["low"] = 0.0
            dv = self._dop_velocity(pos)
            if dv is not None:
                for ax, vz, r2 in ((0, dv[0], cfg.kf_dop_sx ** 2), (1, dv[1], cfg.kf_dop_sy ** 2)):
                    a, b, c = P[ax]
                    S = c + r2
                    inn = vz - vel[ax]
                    if inn * inn > 9.0 * S:
                        continue
                    k1, k2 = b / S, c / S
                    pos[ax] += k1 * inn
                    vel[ax] += k2 * inn
                    P[ax] = [a - k1 * b, b - k1 * c, c - k2 * c]
        else:
            st["low"] += dt
            vel *= np.exp(-dt / cfg.kf_coast_tau)
        dv = vel - v0
        n = float(np.hypot(dv[0], dv[1]))
        if n > cfg.kf_amax * dt:
            vel[:] = v0 + dv * (cfg.kf_amax * dt / n)
        sp = float(np.hypot(vel[0], vel[1]))
        if sp > cfg.kf_vmax:
            vel *= cfg.kf_vmax / sp
        d = pos - p0
        n = float(np.hypot(d[0], d[1]))
        if n > cfg.kf_vmax * dt:
            pos[:] = p0 + d * (cfg.kf_vmax * dt / n)
        if self.layout.S == 1:
            pos[0] = vel[0] = 0.0
        for ax, lo, hi in ((0, tb.xs[0], tb.xs[-1]), (1, tb.ys[0], tb.ys[-1])):
            if pos[ax] < lo or pos[ax] > hi:
                pos[ax] = min(max(pos[ax], lo), hi)
                vel[ax] = 0.0
        st["age"] += dt
        st["hist"].append(pos.copy())
        st["vh"].append(vel.copy())

    def _dop_velocity(self, pos):
        lay, fe, cfg = self.layout, self.fe, self.cfg
        F = self._dop_spec
        if F is None or lay.S < 2:
            return None
        e, J = lay.excess_jac(float(pos[0]), float(pos[1]))
        W = F.shape[0]
        kk = np.fft.fftfreq(W) * W
        rates = np.empty(lay.S)
        for s in range(lay.S):
            k = int(round(e[s] / fe.m_per_tap))
            if k < 3 or k + 4 > F.shape[2]:
                return None
            spec = F[:, s, k - 3:k + 4].sum(axis=-1)
            spec[np.abs(kk) < 1] = 0.0
            i = int(np.argmax(spec))
            pk = float(spec[i])
            if pk < cfg.kf_dop_q * float(np.partition(spec, W // 2)[W // 2]):
                return None
            a, c = spec[(i - 1) % W], spec[(i + 1) % W]
            la, lb, lc = np.log(a + 1e-30), np.log(pk + 1e-30), np.log(c + 1e-30)
            den = la - 2 * lb + lc
            frac = float(np.clip(0.5 * (la - lc) / den, -0.5, 0.5)) if den < 0 else 0.0
            rates[s] = -(kk[i] + frac) / (W * self.dt) * fe.lam[s]
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-6:
            return None
        return np.array([(J[1, 1] * rates[0] - J[0, 1] * rates[1]) / det,
                         (J[0, 0] * rates[1] - J[1, 0] * rates[0]) / det])

    def _travel(self, st, window_s):
        h = st["hist"]
        n = int(window_s / self.dt)
        if len(h) < max(2, int(self.cfg.conf_min_s / self.dt)):
            return 0.0
        P = h.last(len(h))
        if len(P) <= n:
            d = P - P[0]
        else:
            d = P[n:] - P[:-n]
        return float(np.sqrt((d * d).sum(axis=1)).max())

    def _extent(self, st, seconds):
        n = int(seconds / self.dt)
        h = st["hist"]
        if len(h) < max(2, int(self.cfg.conf_min_s / self.dt)):
            return 0.0
        P = h.last(n)
        d = P.max(axis=0) - P.min(axis=0)
        return float(np.hypot(d[0], d[1]))

    def _frame_mht(self, snr, pw):
        lay, trk, cfg, dt, tb = self.layout, self.tracker, self.cfg, self.dt, self.tbd
        S = lay.S
        self.frame_i += 1
        now = self.frame_i * dt
        prot = None
        if trk.active and self._gst is not None:
            prot = (float(self._gst["pos"][0]), float(self._gst["pos"][1]), cfg.occ_protect_m)
        x, y, sc = tb.step(snr, protect=prot)
        if trk.active and (self._gst is None or self._gst["age"] > trk.age + 1e-9):
            self._gst = self._new_hyp(trk.pos)
            self._gst["score"] = cfg.mht_show
        disp = self._gst if trk.active else None
        dist = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))
        edge_m = 2.01 * cfg.tbd_cell_m

        hyps = ([disp] if disp is not None else []) + self._cands
        for h in hyps:
            M = self._dop_matched_llr(h) if h is disp else None
            self._kf_step(h, snr, pw, h is disp, M=M)
            ev = h["ev"] if h["low"] == 0.0 else min(h["ev"], 0.0)
            h["score"] = cfg.mht_lambda * h["score"] + float(np.clip(ev, cfg.mht_ev_lo, cfg.tbd_clip * S))
            h["ev_s"] = h["ev_s"] + (dt / cfg.mht_recent_tau) * (ev - h["ev_s"])
            edge = (h["pos"][0] <= tb.xs[0] + edge_m or h["pos"][0] >= tb.xs[-1] - edge_m
                    or h["pos"][1] <= tb.ys[0] + edge_m or h["pos"][1] >= tb.ys[-1] - edge_m)
            h["edge"] += (dt / cfg.mht_edge_s) * ((1.0 if edge else 0.0) - h["edge"])

        a = dt / cfg.mht_vote_tau
        near = None
        if sc > cfg.tbd_on and hyps:
            dd = [dist((x, y), h["pos"]) for h in hyps]
            k = int(np.argmin(dd))
            if dd[k] < cfg.mht_vote_m:
                near = hyps[k]
        for h in hyps:
            h["vote"] = (1 - a) * h["vote"] + a * (1.0 if h is near else 0.0)

        if disp is not None:
            st = disp
            trk.age = st["age"]
            pos, vel = st["pos"], st["vel"]
            q = st["score"] * (1 - cfg.mht_lambda)
            trk.set_xy(pos, vel, sigma=float(np.clip(0.03 / max(q, 0.1), 0.005, 0.08)))
            far = pos[1] > cfg.far_y
            stale = st["age"] > cfg.stale_s and \
                self._extent(st, cfg.stale_s) < (cfg.stale_m * cfg.far_conf if far else cfg.stale_m)
            at_edge = st["edge"] > cfg.mht_edge_frac
            if st["low"] > cfg.coast_s or stale or at_edge:
                self._ended = (pos.copy(), now)
                if not stale:
                    self._lost = (pos.copy(), now)
                if trk.age > 0.5 and not (stale and far) and not at_edge:
                    self._held = (pos.copy(), self.ring.total, cfg.hold_s if stale else cfg.hold_lost_s)
                trk.stop()
                self._gst = None
                if not stale and not at_edge:
                    st["shown"] = now
                    self._cands.append(st)
                disp = None

        keep = []
        for h in sorted(self._cands, key=lambda h: (h["shown"] is None, -h["score"])):
            dormant = h["shown"] is not None and now - h["shown"] < cfg.reacq_s
            if (h["low"] > cfg.coast_s and not dormant) or h["score"] < cfg.mht_min_score or h["edge"] > cfg.mht_edge_frac:
                continue
            if disp is not None and dist(h["pos"], disp["pos"]) < cfg.mht_merge_m:
                continue
            if any(dist(h["pos"], k["pos"]) < cfg.mht_merge_m for k in keep):
                continue
            keep.append(h)
        keep = keep[:cfg.mht_max]

        for px, py, psc in tb.peaks(cfg.n_cands, cfg.cand_sep_m, cfg.tbd_on):
            if len(keep) >= cfg.mht_max:
                break
            p = (px, py)
            if disp is not None and dist(p, disp["pos"]) < cfg.cand_sep_m:
                continue
            if any(dist(p, k["pos"]) < cfg.cand_sep_m for k in keep):
                continue
            h = self._new_hyp(p)
            if px <= tb.xs[0] + edge_m or px >= tb.xs[-1] - edge_m or py <= tb.ys[0] + edge_m or py >= tb.ys[-1] - edge_m:
                h["edge"] = cfg.mht_edge_frac - 0.1
            keep.append(h)
        self._cands = keep

        lost, ended = self._lost, self._ended
        for h in keep:
            if h["low"] > 0.0:
                h["conf"] = False
                continue
            again = (lost is not None and now - lost[1] < cfg.reacq_s
                     and dist(h["pos"], lost[0]) < cfg.reacq_m and h["age"] >= 0.05)
            if h["shown"] is not None and now - h["shown"] < cfg.reacq_s:
                again = True
            need = cfg.conf_m
            ref = None
            if disp is not None:
                ref, el = disp["pos"], 0.0
            elif ended is not None:
                ref, el = ended[0], now - ended[1]
            if ref is not None and el < cfg.reach_s and dist(h["pos"], ref) > cfg.reach_r0 + cfg.reach_v * el:
                need = cfg.conf_m * cfg.reach_conf
            if h["pos"][1] > cfg.far_y:
                need = max(need, cfg.conf_m * cfg.far_conf)
            moved = self._travel(h, cfg.conf_win_s)
            h["conf"] = bool(again or (h["age"] >= 0.15 and h["edge"] < 0.3 and moved >= need))

        new = None
        if disp is not None:
            eD = lay.excess_jac(float(disp["pos"][0]), float(disp["pos"][1]))[0]
            for h in keep:
                d = lay.excess_jac(float(h["pos"][0]), float(h["pos"][1]))[0] - eD
                h["behind"] = bool(d.min() > -cfg.arm_same_m and d.max() > cfg.arm_behind_m
                                   and disp["ev_s"] > cfg.mht_weak)
                front = (h["low"] == 0.0 and h["conf"] and d.max() < cfg.arm_same_m and d.min() < -cfg.arm_front_m
                         and dist(h["pos"], disp["pos"]) < cfg.arm_max_m
                         and h["ev_s"] > max(cfg.kf_det, cfg.arm_ev_frac * disp["ev_s"]))
                h["front"] = h["front"] + dt if front else 0.0
            ref = max(disp["score"], 1e-3)
            for h in keep:
                if h["behind"]:
                    h["dom"] = 0.0
                    continue
                if h["front"] > cfg.arm_switch_s:
                    new = h
                    break
                far = dist(h["pos"], disp["pos"]) > cfg.reach_r0
                ratio, need_s = (cfg.mht_switch_far, cfg.mht_switch_s_far) if far else (cfg.mht_switch, cfg.mht_switch_s)
                better = h["score"] > ratio * ref
                voted = h["vote"] > cfg.mht_vote_on and disp["vote"] < cfg.mht_vote_off
                weak = disp["ev_s"] < cfg.mht_weak and h["ev_s"] > cfg.mht_strong
                if h["conf"] and h["score"] > cfg.mht_show and (better or voted or weak):
                    h["dom"] = h["dom"] + dt
                else:
                    h["dom"] = 0.0
                if weak and not far:
                    need_s = min(need_s, cfg.mht_weak_s)
                if h["dom"] > need_s and (new is None or h["score"] > new["score"]):
                    new = h
        else:
            for h in keep:
                if h["conf"] and h["score"] > cfg.mht_show and (new is None or h["score"] > new["score"]):
                    new = h
        if new is not None:
            if disp is not None:
                disp["dom"] = 0.0
                self._cands = [k for k in self._cands if k is not new] + [disp]
            else:
                self._cands = [k for k in self._cands if k is not new]
            new["shown"] = None
            trk.start(new["pos"], e=lay.excess(*new["pos"]))
            trk.age = new["age"]
            self._gst = new
            self._held = None
            trk.set_xy(new["pos"], new["vel"])

        dets = [None] * S
        if trk.active:
            e = lay.excess(*self._gst["pos"])
            dets = [{"e": float(e[s]),
                     "snr": float(snr[s][min(int(round(e[s] / self.fe.m_per_tap)), snr.shape[1] - 1)])}
                    for s in range(S)]
        self.det = dets

    def _waterfall(self, snr):
        n = self.disp_taps
        v = db(snr[:, :n]).reshape(len(snr), n // 2, 2).max(axis=2)
        self.wf[:, self.wf_ptr] = v
        if self.tracker.active:
            self.wf_track[:, self.wf_ptr] = self.tracker.excess() / DISP_MAX_M
        else:
            self.wf_track[:, self.wf_ptr] = np.nan
        self.wf_ptr = (self.wf_ptr + 1) % WATERFALL_W

    def _anchor_detect(self, d):
        fe = self.fe
        a = 1 - np.exp(-self.dt / 0.5)
        p = np.mean(np.abs(d) ** 2, axis=1)
        p = np.fft.irfft(np.fft.rfft(p, axis=-1) * self.smooth_k, n=fe.L, axis=-1)
        if self._anc_avg is None:
            self._anc_avg = p.copy()
            self._anc_n = 0
        self._anc_avg = self._anc_avg + a * (p - self._anc_avg)
        self._anc_n += 1
        if self._anc_n * self.dt < 0.4:
            return [None] * self.layout.S
        snr = np.maximum(self._anc_avg, 1e-30) / np.maximum(self.noise, 1e-30)
        return [self._detect(snr[s], self._anc_avg[s], "acq", thr_db=self.cfg.anchor_det_db, guard_db=6.0)
                for s in range(self.layout.S)]

    def _anchor_step(self, dets, d):
        dc = self._anchor_detect(d)
        if all(x is not None for x in dc):
            dets = dc
        if self.anchor_skip or self._elapsed() > self.cfg.t_anchor_max:
            self.anchor_ok = False
            self.message = "Skipped the anchor, using the default geometry"
            self._start_tracking()
            return
        if all(x is not None for x in dets):
            self.anchor_buf.append([x["e"] for x in dets])
            self._anchor_gap = 0
        else:
            self._anchor_gap += 1
            if self._anchor_gap * self.dt > 0.4:
                self.anchor_buf = []
        need = int(0.7 / self.dt)
        self.progress = min(1.0, len(self.anchor_buf) / need)
        if len(self.anchor_buf) < need:
            return
        e = np.array(self.anchor_buf[-need:])
        if np.max(np.std(e, axis=0)) > 0.025:
            self.anchor_buf = self.anchor_buf[-need // 2:]
            self.message = "Hold your palm still..."
            return
        lay = self.layout
        ax, ay = self.cfg.anchor_pos
        p_ref = lay.paths(ax, ay) - np.median(e, axis=0)
        if np.all(np.abs(p_ref - lay.direct) < 0.35):
            lay.p_ref = p_ref
            self.tbd.rebuild()
            self.anchor_ok = True
            self.message = "Anchored"
        else:
            self.anchor_ok = False
            self.warnings.append("The anchor looked wrong, using the default geometry")
        self._cands, self._gst = [], None
        self.tracker.start((ax, ay))
        self._start_tracking()

    def snapshot(self):
        with self.lock:
            snap = {
                "state": self.state,
                "message": self.message,
                "progress": self.progress,
                "steps": [(k, label, self.step_status[k]) for k, label in STEPS],
                "error": self.error,
                "warnings": list(self.warnings),
                "band": self.band,
                "latency_s": self.latency_s,
                "drift_ppm": self.drift_ppm,
                "response": None if self.response is None else dict(self.response),
                "info": self.info,
            }
            if self.proc_ms:
                snap["proc_ms"] = float(np.mean(self.proc_ms))
            if len(self._wall) > 2:
                tw = self._wall[-1][0] - self._wall[0][0]
                nf = sum(n for _, n in list(self._wall)[1:])
                snap["dsp_hz"] = nf / tw if tw > 0 else 0.0
            if self.fe is not None and self.state in ("anchor", "tracking"):
                fe = self.fe
                snap["resolution_m"] = fe.resolution_m
                snap["gate"] = (self.cfg.gate_min, min(self.cfg.gate_max, 0.85 * fe.alias_m))
                snap["p_ref"] = self.layout.p_ref.copy()
                snap["det"] = [None if x is None else dict(x) for x in (self.det or [])]
                trk = self.tracker
                snap["hand"] = None
                if trk.active:
                    self._held = None
                    snap["hand"] = {"pos": trk.pos, "vel": trk.vel, "cov": trk.cov(), "id": trk.n_starts}
                elif self._held is not None:
                    age = (self.ring.total - self._held[1]) / self.fs
                    if age < self._held[2]:
                        snap["hand"] = {"pos": self._held[0].copy(), "vel": np.zeros(2), "held": True,
                                        "fade": 1.0 - age / self._held[2]}
                snap["tbd"] = {"score": self.tbd.score.astype(np.float32).copy(),
                               "x": (float(self.tbd.xs[0]), float(self.tbd.xs[-1])),
                               "y": (float(self.tbd.ys[0]), float(self.tbd.ys[-1])),
                               "on": float(self.cfg.tbd_on)}
                snap["profile_m"] = np.arange(self.disp_taps) * fe.m_per_tap
                snap["profile"] = None if self.last_profile is None else self.last_profile.copy()
        return snap

    def waterfall(self):
        with self.lock:
            if self.wf is None:
                return None
            return (np.roll(self.wf, -self.wf_ptr, axis=1).copy(),
                    np.roll(self.wf_track, -self.wf_ptr, axis=1).copy())

    def output_table(self):
        return self.desired, self.version
