import math
import sys
import time
from collections import deque

import numpy as np
import pygame

from .engine import DISP_MAX_M

BLACK = (0, 0, 0)
BAR = (40, 40, 40)
BAR_HI = (64, 64, 64)
LINE = (90, 90, 90)
WHITE = (230, 230, 230)
GREY = (160, 160, 160)
DIM = (105, 105, 105)
YELLOW = (255, 215, 0)
LEFT = (0, 200, 255)
RIGHT = (255, 140, 0)
TOWARD = (255, 60, 60)
AWAY = (60, 120, 255)
GREEN = (0, 200, 0)
RED = (255, 60, 60)

GREYS = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)


def to_db(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return -60.0
    return 10 * math.log10(x) if x > 1e-6 else -60.0


def trail_colour(v_toward):
    a = min(1.0, abs(v_toward) / 0.35)
    tgt = TOWARD if v_toward > 0 else AWAY
    return tuple(int(WHITE[i] + (tgt[i] - WHITE[i]) * a) for i in range(3))


class Text:
    def __init__(self):
        self.path = None
        for name in ("arial", "dejavusans", "liberationsans"):
            p = pygame.font.match_font(name)
            if p:
                self.path = p
                break
        self.fonts = {}
        self.cache = {}

    def font(self, size, bold=False):
        f = self.fonts.get((size, bold))
        if f is None:
            f = pygame.font.Font(self.path, size) if self.path else pygame.font.SysFont(None, size)
            f.set_bold(bold)
            self.fonts[(size, bold)] = f
        return f

    def surf(self, s, size, colour, bold=False):
        key = (s, size, colour, bold)
        t = self.cache.get(key)
        if t is None:
            if len(self.cache) > 3000:
                self.cache.clear()
            t = self.font(size, bold).render(s, True, colour)
            self.cache[key] = t
        return t

    def draw(self, dst, s, pos, size=15, colour=WHITE, bold=False, anchor="topleft"):
        t = self.surf(s, size, colour, bold)
        r = t.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
        dst.blit(t, r)
        return r

    def wrap(self, s, size, width):
        f = self.font(size)
        out, line = [], ""
        for w in s.split():
            t = (line + " " + w).strip()
            if line and f.size(t)[0] > width:
                out.append(line)
                line = w
            else:
                line = t
        if line:
            out.append(line)
        return out


class App:
    def __init__(self, io, cfg):
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        pygame.init()
        pygame.display.set_caption("Pitch")
        info = pygame.display.Info()
        w = int(min(1600, max(1000, info.current_w * 0.86))) if info.current_w > 0 else 1440
        h = int(min(980, max(600, info.current_h * 0.84))) if info.current_h > 0 else 900
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self.io = io
        self.eng = io.engine
        self.cfg = cfg
        self.text = Text()
        self.clock = pygame.time.Clock()
        self.trail = deque(maxlen=400)
        self.track_id = None
        self.ink = []
        self.drawing = False
        self.show_ellipses = True
        self.show_sonogram = True
        self.show_evidence = False
        self.fullscreen = False
        self.toast = None
        self.prof_s = None
        self.grid_key = None
        self.frame_no = 0
        self.side_cache = None
        self.bars_cache = None
        self.field_bg = None
        self.laptop_key = None
        self.fps = deque(maxlen=60)
        self.set_layout(self.screen.get_size())

    def set_layout(self, size):
        W, H = size
        self.W = W
        self.u = u = max(0.75, min(1.4, H / 900))
        top = int(58 * u)
        bot = int(34 * u)
        side = int(np.clip(W * 0.31, 380, 540))
        pad = int(14 * u)
        self.r_top = pygame.Rect(0, 0, W, top)
        self.r_bot = pygame.Rect(0, H - bot, W, bot)
        self.r_field = pygame.Rect(0, top, W - side, H - top - bot)
        self.r_side = pygame.Rect(W - side, top, side, H - top - bot)
        x0 = self.r_side.x + pad
        w = side - 2 * pad
        y = top + pad
        hh = self.r_side.h - 2 * pad
        self.r_panels = []
        for f in (0.19, 0.19, 0.17, 0.17):
            h = int(hh * f)
            self.r_panels.append(pygame.Rect(x0, y, w, h))
            y += h + int(10 * u)
        self.r_read = pygame.Rect(x0, y, w, top + pad + hh - y)
        f = self.r_field
        xr, yr = (-0.44, 0.44), (-0.075, 0.70)
        self.s = min((f.w - 30 * u) / (xr[1] - xr[0]), (f.h - 20 * u) / (yr[1] - yr[0]))
        self.ox = f.centerx
        self.oy = f.bottom + yr[0] * self.s - 6 * u
        self.grid_key = None
        self.overlay = pygame.Surface((f.w, f.h), pygame.SRCALPHA)
        self.build_fan()

    def w2s(self, x, y):
        return (self.ox + x * self.s, self.oy - y * self.s)

    def mic_xy(self):
        g = self.cfg.geometry
        return g.mic[0], g.mic[1]

    def build_fan(self):
        f = self.r_field
        self.fan_r = 0.68
        self.fan_half = math.radians(64)
        ax, ay = self.mic_xy()
        pts = [self.w2s(ax, ay)]
        for a in np.linspace(-self.fan_half, self.fan_half, 90):
            pts.append(self.w2s(ax + self.fan_r * math.sin(a), ay + self.fan_r * math.cos(a)))
        self.fan_pts = [(x - f.x, y - f.y) for x, y in pts]
        g = pygame.Surface((f.w, f.h), pygame.SRCALPHA)
        for r_cm in range(10, 70, 10):
            r = r_cm / 100
            arc = [self.w2s(ax + r * math.sin(a), ay + r * math.cos(a))
                   for a in np.linspace(-self.fan_half, self.fan_half, 70)]
            pygame.draw.aalines(g, (*DIM, 70), False, [(x - f.x, y - f.y) for x, y in arc])
        for a in np.linspace(-self.fan_half, self.fan_half, 5):
            p0 = self.w2s(ax + 0.06 * math.sin(a), ay + 0.06 * math.cos(a))
            p1 = self.w2s(ax + self.fan_r * math.sin(a), ay + self.fan_r * math.cos(a))
            pygame.draw.aaline(g, (*DIM, 45), (p0[0] - f.x, p0[1] - f.y), (p1[0] - f.x, p1[1] - f.y))
        pygame.draw.aalines(g, (*LINE, 255), True, self.fan_pts)
        self.fan = g

    def draw_laptop(self, dst, lay):
        g = self.cfg.geometry
        u = self.u
        f = self.r_field
        ox, oy = self.ox, self.oy
        self.ox, self.oy = ox - f.x, oy - f.y
        hw = g.screen_width / 2
        x0, y0 = self.w2s(-hw, 0.0)
        x1, y1 = self.w2s(hw, g.deck_depth)
        dashed_rect(dst, pygame.Rect(int(x0), int(y1), int(x1 - x0), int(y0 - y1)), DIM, int(6 * u))
        a = self.w2s(-hw, -0.012)
        b = self.w2s(hw, -0.012)
        pygame.draw.line(dst, (120, 128, 138), a, b, max(3, int(5 * u)))
        self.text.draw(dst, "screen", (b[0] + 8 * u, b[1]), int(13 * u), DIM, anchor="midleft")
        if lay is not None:
            cols = [LEFT, RIGHT] if lay.S == 2 else [WHITE]
            names = ["L", "R"] if lay.S == 2 else ["L+R"]
            for s in range(lay.S):
                px, py = self.w2s(lay.spk[s][0], lay.spk[s][1])
                pygame.draw.circle(dst, cols[s], (int(px), int(py)), max(4, int(6 * u)))
                pygame.draw.circle(dst, BLACK, (int(px), int(py)), max(2, int(3 * u)))
                self.text.draw(dst, names[s], (px, py + 10 * u), int(14 * u), cols[s], anchor="midtop")
        mx, my = self.w2s(*self.mic_xy())
        pygame.draw.circle(dst, WHITE, (int(mx), int(my)), max(3, int(4 * u)))
        self.text.draw(dst, "mic", (mx, my - 8 * u), int(13 * u), GREY, anchor="midbottom")
        self.ox, self.oy = ox, oy

    def make_grid(self, snap, lay):
        key = (tuple(np.round(lay.p_ref, 5)), self.r_field.size, lay.S)
        if key == self.grid_key:
            return
        self.grid_key = key
        f = self.r_field
        cell = max(3, int(4 * self.u))
        gw, gh = f.w // cell, f.h // cell
        X = (f.x + (np.arange(gw) + 0.5) * cell - self.ox) / self.s
        Y = (self.oy - (f.y + (np.arange(gh) + 0.5) * cell)) / self.s
        XX, YY = np.meshgrid(X, Y, indexing="ij")
        e = lay.excess(XX, YY)
        n = len(snap["profile_m"])
        idx = np.round(e / snap["profile_m"][1]).astype(int)
        valid = ((e >= snap["gate"][0]) & (idx < n) & (idx >= 0)).all(axis=-1)
        ax, ay = self.mic_xy()
        infan = (np.abs(np.arctan2(XX - ax, YY - ay)) <= self.fan_half) & (np.hypot(XX - ax, YY - ay) <= self.fan_r)
        self.g_idx = np.clip(idx, 0, n - 1)
        self.g_valid = valid & infan
        self.g_shape = (gw, gh)
        dead = np.zeros((gw, gh, 3), np.uint8)
        dead[infan & ~valid & (YY > -0.02)] = (18, 20, 24)
        self.dead_surf = pygame.transform.scale(pygame.surfarray.make_surface(dead), f.size)

    def draw_sonogram(self, dst, snap, lay):
        prof = snap["profile"]
        self.make_grid(snap, lay)
        if self.prof_s is None or self.prof_s.shape != prof.shape:
            self.prof_s = prof.astype(np.float32)
        else:
            self.prof_s = 0.55 * self.prof_s + 0.45 * prof
        vals = np.zeros(self.g_shape, np.float32)
        for s in range(lay.S):
            vals += self.prof_s[s][self.g_idx[..., s]]
        vals /= lay.S
        k = np.clip((vals - 4.0) / 30.0, 0, 1)
        k[~self.g_valid] = 0
        surf = pygame.surfarray.make_surface(GREYS[(k * 255).astype(np.uint8)])
        surf = pygame.transform.smoothscale(surf, self.r_field.size)
        dst.blit(self.dead_surf, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        dst.blit(surf, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    def draw_evidence(self, dst, tb):
        sc = tb["score"]
        top = float(sc.max())
        if top < tb["on"]:
            return
        base = float(np.median(sc))
        k = np.clip((sc - base) / max(top - base, 1e-6), 0, 1) ** 1.5
        ny, nx = k.shape
        rgb = np.zeros((nx, ny, 3), np.uint8)
        kk = k.T[:, ::-1] ** 0.7
        rgb[..., 0] = (40 * kk).astype(np.uint8)
        rgb[..., 1] = (210 * kk).astype(np.uint8)
        rgb[..., 2] = (200 * kk).astype(np.uint8)
        f = self.r_field
        x0, y0 = self.w2s(tb["x"][0], tb["y"][1])
        x1, y1 = self.w2s(tb["x"][1], tb["y"][0])
        w, h = int(abs(x1 - x0)), int(abs(y1 - y0))
        if w < 2 or h < 2:
            return
        surf = pygame.transform.smoothscale(pygame.surfarray.make_surface(rgb), (w, h))
        dst.blit(surf, (int(min(x0, x1) - f.x), int(min(y0, y1) - f.y)), special_flags=pygame.BLEND_RGB_ADD)

    def run(self):
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.screen = pygame.display.set_mode((max(1000, ev.w), max(660, ev.h)), pygame.RESIZABLE)
                    self.set_layout(self.screen.get_size())
                elif ev.type == pygame.KEYDOWN:
                    running = self.key(ev.key)
            snap = self.eng.snapshot()
            self.update(snap)
            self.draw(snap)
            pygame.display.flip()
            self.fps.append(self.clock.tick(12 if self.io.busy() else 30))
        pygame.quit()

    def key(self, k):
        if k in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if k == pygame.K_r:
            self.eng.restart()
            self.trail.clear()
            self.prof_s = None
            self.say("Recalibrating")
        elif k == pygame.K_SPACE:
            self.eng.skip_anchor()
        elif k == pygame.K_n:
            if self.eng.reanchor():
                self.say("Hold your palm 20 cm in front of the screen, still")
        elif k == pygame.K_d:
            self.drawing = not self.drawing
            self.say("Drawing on" if self.drawing else "Drawing off")
        elif k == pygame.K_c:
            self.ink = []
            self.trail.clear()
            self.say("Cleared")
        elif k == pygame.K_e:
            self.show_ellipses = not self.show_ellipses
        elif k == pygame.K_i:
            self.show_sonogram = not self.show_sonogram
        elif k == pygame.K_v:
            self.show_evidence = not self.show_evidence
            self.say("Evidence map on" if self.show_evidence else "Evidence map off")
        elif k == pygame.K_f:
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            else:
                self.screen = pygame.display.set_mode((1440, 900), pygame.RESIZABLE)
            self.set_layout(self.screen.get_size())
        elif k == pygame.K_s:
            name = time.strftime("pitch_%Y%m%d_%H%M%S.png")
            pygame.image.save(self.screen, name)
            self.say(f"Saved {name}")
        elif k == pygame.K_m:
            if sys.platform.startswith("win"):
                try:
                    import subprocess
                    subprocess.Popen(["control", "mmsys.cpl,,1"])
                    self.say("Microphone Array > Properties > Levels: set it to about 20, OK, then press R")
                except Exception:
                    self.say("Sound settings > Input: set the mic volume to about 20%, then press R")
            else:
                self.say("Set the mic input level to about 20% in your sound settings, then press R")
        elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_MINUS, pygame.K_KP_MINUS):
            p = self.io.player
            if p is not None:
                up = k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS)
                p.volume = float(np.clip(p.volume * (1.25 if up else 0.8), 0.1, 2.5))
                self.say(f"Probe level {p.volume * 100:.0f}%")
        return True

    def say(self, msg):
        self.toast = (msg, time.perf_counter())

    def update(self, snap):
        now = time.perf_counter()
        hand = snap.get("hand")
        if hand is not None and not hand.get("held"):
            x, y = hand["pos"]
            if hand["id"] != self.track_id and self.trail:
                self.trail.append((now, math.nan, math.nan, 0.0))
                if self.drawing and self.ink and self.ink[-1] is not None:
                    self.ink.append(None)
            self.track_id = hand["id"]
            self.trail.append((now, x, y, -hand["vel"][1]))
            if self.drawing:
                if not self.ink or self.ink[-1] is None:
                    self.ink.append([])
                self.ink[-1].append((x, y))
        elif self.drawing and self.ink and self.ink[-1] is not None:
            self.ink.append(None)

    def draw(self, snap):
        scr = self.screen
        self.frame_no += 1
        k = self.frame_no
        scr.fill(BLACK, self.r_field)
        self.draw_field(snap)
        size = scr.get_size()
        if self.side_cache is None or k % 3 == 1 or self.side_cache[0] != size:
            self.draw_side(snap)
            self.side_cache = (size, scr.subsurface(self.r_side).copy())
        else:
            scr.blit(self.side_cache[1], self.r_side.topleft)
        if self.bars_cache is None or k % 6 == 1 or self.bars_cache[0] != size:
            self.draw_top(snap)
            self.draw_bottom()
            self.bars_cache = (size, scr.subsurface(self.r_top).copy(), scr.subsurface(self.r_bot).copy())
        else:
            scr.blit(self.bars_cache[1], self.r_top.topleft)
            scr.blit(self.bars_cache[2], self.r_bot.topleft)
        if snap["state"] != "tracking":
            self.draw_calibration(snap)
        if self.toast and time.perf_counter() - self.toast[1] < 2.6:
            u = self.u
            t = self.text.surf(self.toast[0], int(16 * u), WHITE)
            r = t.get_rect(midbottom=(self.r_field.centerx, self.r_field.bottom - 16 * u))
            pygame.draw.rect(scr, BAR_HI, r.inflate(20 * u, 10 * u))
            scr.blit(t, r)

    def draw_field(self, snap):
        f = self.r_field
        fs = self.screen.subsurface(f)
        lay = self.eng.layout
        ready = snap["state"] in ("anchor", "tracking") and lay is not None
        u = self.u
        bkey = (f.size, ready, self.show_sonogram, self.show_evidence, id(lay))
        if self.field_bg is not None and self.field_bg[0] == bkey and self.frame_no % 2 == 0:
            fs.blit(self.field_bg[1], (0, 0))
        else:
            if ready and self.show_sonogram and snap.get("profile") is not None:
                self.draw_sonogram(fs, snap, lay)
            if ready and self.show_evidence:
                self.draw_evidence(fs, snap["tbd"])
            fs.blit(self.fan, (0, 0))
            ax, ay = self.mic_xy()
            for r_cm in (10, 20, 30, 40, 50, 60):
                x, y = self.w2s(ax + r_cm / 100 * math.sin(self.fan_half), ay + r_cm / 100 * math.cos(self.fan_half))
                self.text.draw(fs, f"{r_cm}", (x - f.x + 8 * u, y - f.y), int(12 * u), DIM, anchor="midleft")
            x, y = self.w2s(ax + 0.6 * math.sin(self.fan_half), ay + 0.6 * math.cos(self.fan_half))
            self.text.draw(fs, "cm from mic", (x - f.x + 8 * u, y - f.y + 16 * u), int(12 * u), DIM, anchor="midleft")
            key = (id(lay), f.size)
            if self.laptop_key != key:
                self.laptop_img = pygame.Surface(f.size, pygame.SRCALPHA)
                self.draw_laptop(self.laptop_img, lay)
                self.laptop_key = key
            fs.blit(self.laptop_img, (0, 0))
            self.field_bg = (bkey, fs.copy())
        if not ready:
            return
        ov = self.overlay
        ov.fill((0, 0, 0, 0))
        now = time.perf_counter()
        if self.show_ellipses:
            cols = [LEFT, RIGHT] if lay.S == 2 else [WHITE]
            for s, d in enumerate(snap["det"]):
                if d is None:
                    continue
                pts = lay.iso_curve(s, d["e"] + snap["p_ref"][s], n=160)
                if pts is None:
                    continue
                run = []
                for x, y in pts:
                    if y > 0.0:
                        run.append((self.ox + x * self.s - f.x, self.oy - y * self.s - f.y))
                    else:
                        if len(run) > 1:
                            pygame.draw.aalines(ov, (*cols[s], 150), False, run)
                        run = []
                if len(run) > 1:
                    pygame.draw.aalines(ov, (*cols[s], 150), False, run)
        for stroke in self.ink:
            if stroke and len(stroke) > 1:
                sp = [(self.ox + x * self.s - f.x, self.oy - y * self.s - f.y) for x, y in stroke]
                pygame.draw.lines(ov, (*WHITE, 235), False, sp, max(2, int(3 * u)))
        pts = [p for p in self.trail if now - p[0] < 1.6]
        for i in range(1, len(pts)):
            t0, x0, y0, _ = pts[i - 1]
            t1, x1, y1, v = pts[i]
            if not (math.hypot(x1 - x0, y1 - y0) <= 0.08):
                continue
            age = (now - t1) / 1.6
            pygame.draw.line(ov, (*trail_colour(v), int(255 * (1 - age) ** 1.6)),
                             (self.ox + x0 * self.s - f.x, self.oy - y0 * self.s - f.y),
                             (self.ox + x1 * self.s - f.x, self.oy - y1 * self.s - f.y),
                             max(2, int(4 * u * (1 - 0.6 * age))))
        hand = snap.get("hand")
        if hand is not None and not hand.get("held"):
            vx, vy = hand["vel"]
            hx, hy = self.w2s(*hand["pos"])
            hx -= f.x
            hy -= f.y
            w, V = np.linalg.eigh(hand["cov"])
            w = np.sqrt(np.maximum(w, 1e-8)) * 2
            th = np.linspace(0, 2 * np.pi, 48)
            E = V @ np.stack([w[0] * np.cos(th), w[1] * np.sin(th)])
            pygame.draw.aalines(ov, (*WHITE, 110), True, [(hx + E[0, i] * self.s, hy - E[1, i] * self.s)
                                                          for i in range(len(th))])
            c = trail_colour(-vy)
            if math.hypot(vx, vy) > 0.04:
                tx, ty = hx + vx * 0.25 * self.s, hy - vy * 0.25 * self.s
                pygame.draw.line(ov, (*c, 230), (hx, hy), (tx, ty), max(2, int(2 * u)))
                ang = math.atan2(ty - hy, tx - hx)
                L = 9 * u
                for da in (2.6, -2.6):
                    pygame.draw.line(ov, (*c, 230), (tx, ty), (tx + L * math.cos(ang + da), ty + L * math.sin(ang + da)),
                                     max(2, int(2 * u)))
        fs.blit(ov, (0, 0))
        if hand is not None:
            x, y = hand["pos"]
            hx, hy = self.w2s(x, y)
            hx -= f.x
            hy -= f.y
            if hand.get("held"):
                k = max(0.0, min(1.0, hand["fade"]))
                col = tuple(int(v * (0.35 + 0.65 * k)) for v in YELLOW)
                pygame.draw.circle(fs, col, (int(hx), int(hy)), max(6, int(11 * u)), max(1, int(2 * u)))
                pygame.draw.circle(fs, col, (int(hx), int(hy)), max(2, int(3 * u)))
                self.text.draw(fs, f"x {x * 100:+.1f}   y {y * 100:.1f} cm  (still)", (hx + 18 * u, hy - 18 * u),
                               int(14 * u), col, anchor="bottomleft")
            else:
                pygame.draw.circle(fs, trail_colour(-hand["vel"][1]), (int(hx), int(hy)), max(6, int(10 * u)))
                pygame.draw.circle(fs, WHITE, (int(hx), int(hy)), max(6, int(10 * u)), 1)
                self.text.draw(fs, f"x {x * 100:+.1f}   y {y * 100:.1f} cm", (hx + 18 * u, hy - 18 * u),
                               int(14 * u), YELLOW, anchor="bottomleft")
        elif snap["state"] == "tracking":
            self.text.draw(fs, "Move your hand in front of the screen", (f.w / 2, self.w2s(0, 0.45)[1] - f.y),
                           int(18 * u), GREY, anchor="center")
        if self.drawing:
            self.text.draw(fs, "Drawing (C clears)", (f.w - 16 * u, 14 * u), int(14 * u), YELLOW, anchor="topright")
        if snap["state"] == "tracking":
            tips = [w for w in snap["warnings"] if "input volume" in w]
            if tips:
                lines = self.text.wrap(tips[0], int(13 * u), min(f.w - 40 * u, 760 * u))
                y0 = f.h - 14 * u - 18 * u * len(lines)
                for i, ln in enumerate(lines):
                    self.text.draw(fs, ln, (f.w / 2, y0 + 18 * u * i), int(13 * u), YELLOW, anchor="midtop")

    def draw_top(self, snap):
        u = self.u
        r = self.r_top
        pygame.draw.rect(self.screen, BAR, r)
        t = self.text.draw(self.screen, "Pitch", (22 * u, r.centery), int(24 * u), WHITE, bold=True, anchor="midleft")
        self.text.draw(self.screen, "- hand tracking with the laptop's speakers + mics (sonar)",
                       (t.right + 12 * u, r.centery + 2 * u), int(14 * u), GREY, anchor="midleft")
        st = snap["state"]
        if st == "tracking":
            label, col = ("Tracking your hand", GREEN) if snap.get("hand") is not None else ("Listening, no hand", GREY)
        elif st == "error":
            label, col = "Needs attention", RED
        else:
            label, col = "Calibrating", YELLOW
        self.text.draw(self.screen, "Status: " + label, (r.right - 20 * u, r.centery), int(15 * u), col, anchor="midright")

    def draw_bottom(self):
        u = self.u
        r = self.r_bot
        pygame.draw.rect(self.screen, BAR, r)
        keys = [("R", "recalibrate"), ("N", "re-anchor"), ("D", "draw"), ("C", "clear"), ("E", "ellipses"),
                ("I", "sonogram"), ("V", "evidence"), ("M", "mic level"), ("+/-", "level"), ("S", "screenshot"),
                ("F", "full screen"), ("Esc", "quit")]
        x = 18 * u
        for k, lab in keys:
            x = self.text.draw(self.screen, f"{k}: {lab}", (x, r.centery), int(13 * u), WHITE, anchor="midleft").right + 14 * u
            if x > self.W - 120 * u:
                break
        fps = 1000 / max(1, np.mean(self.fps)) if self.fps else 0
        self.text.draw(self.screen, f"{fps:.0f} fps", (self.W - 16 * u, r.centery), int(13 * u), DIM, anchor="midright")

    def panel(self, r, title, sub=None):
        u = self.u
        pygame.draw.rect(self.screen, LINE, r.inflate(8 * u, 6 * u), 1)
        t = self.text.draw(self.screen, title, (r.x, r.y), int(14 * u), WHITE)
        if sub:
            self.text.draw(self.screen, sub, (r.right, r.y + 1 * u), int(12 * u), DIM, anchor="topright")
        return pygame.Rect(r.x, t.bottom + 6 * u, r.w, r.bottom - t.bottom - 6 * u)

    def draw_side(self, snap):
        pygame.draw.rect(self.screen, (20, 20, 20), self.r_side)
        p = self.r_panels
        wf = self.eng.waterfall() if snap["state"] in ("anchor", "tracking") else None
        lay = self.eng.layout
        stereo = lay is not None and lay.S == 2
        names = ["Left speaker echoes", "Right speaker echoes"] if stereo else ["Echoes", ""]
        cols = [LEFT, RIGHT] if stereo else [WHITE, WHITE]
        for s in range(2):
            inner = self.panel(p[s], names[s], "extra path vs time" if s == 0 else None)
            if wf is not None and s < wf[0].shape[0]:
                self.echo_plot(inner, wf[0][s], wf[1][s], cols[s])
            else:
                self.empty(inner, "Waiting for calibration")
        inner = self.panel(p[2], "Moving echoes now", "Doppler signal-to-noise vs extra path")
        self.profile_plot(inner, snap, cols)
        inner = self.panel(p[3], "Speaker to mic response", "chosen probe band shaded")
        self.response_plot(inner, snap, cols)
        self.readouts(self.r_read, snap)

    def empty(self, r, msg):
        pygame.draw.rect(self.screen, BLACK, r)
        self.text.draw(self.screen, msg, r.center, int(13 * self.u), DIM, anchor="center")

    def echo_plot(self, r, img, track, col):
        u = self.u
        pygame.draw.rect(self.screen, BLACK, r)
        lab_w = int(30 * u)
        plot = pygame.Rect(r.x + lab_w, r.y, r.w - lab_w, r.h)
        k = np.clip((img - 3.0) / 33.0, 0, 1)
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(GREYS[(k * 255).astype(np.uint8)][:, ::-1]))
        self.screen.blit(pygame.transform.smoothscale(surf, plot.size), plot.topleft)
        W = len(track)
        pts = []
        for i in range(0, W, 3):
            v = track[i]
            if np.isfinite(v) and 0 <= v <= 1:
                pts.append((plot.x + i / (W - 1) * plot.w, plot.bottom - v * plot.h))
            else:
                if len(pts) > 1:
                    pygame.draw.aalines(self.screen, col, False, pts)
                pts = []
        if len(pts) > 1:
            pygame.draw.aalines(self.screen, col, False, pts)
        for cm in (0, 20, 40, 60, 80):
            y = plot.bottom - cm / (DISP_MAX_M * 100) * plot.h
            self.text.draw(self.screen, f"{cm}", (plot.x - 5 * u, y), int(11 * u), DIM, anchor="midright")
        self.text.draw(self.screen, "now", (plot.right - 3 * u, plot.bottom - 2 * u), int(11 * u), DIM, anchor="bottomright")

    def axes(self, r, xlim, ylim, xticks, yticks, xunit):
        u = self.u
        pygame.draw.rect(self.screen, BLACK, r)
        plot = pygame.Rect(r.x + int(30 * u), r.y + int(4 * u), r.w - int(36 * u), r.h - int(22 * u))

        def X(v):
            return plot.x + (v - xlim[0]) / (xlim[1] - xlim[0]) * plot.w

        def Y(v):
            return plot.bottom - (v - ylim[0]) / (ylim[1] - ylim[0]) * plot.h

        for t in yticks:
            pygame.draw.line(self.screen, (24, 27, 31), (plot.x, Y(t)), (plot.right, Y(t)))
            self.text.draw(self.screen, f"{t:.0f}", (plot.x - 5 * u, Y(t)), int(11 * u), DIM, anchor="midright")
        for t in xticks:
            lab = f"{t:.0f} {xunit}" if t == xticks[-1] else f"{t:.0f}"
            self.text.draw(self.screen, lab, (X(t), plot.bottom + 3 * u), int(11 * u), DIM, anchor="midtop")
        return plot, X, Y

    def profile_plot(self, r, snap, cols):
        prof = snap.get("profile")
        if prof is None:
            self.empty(r, "Shows up once the sonar is running")
            return
        plot, X, Y = self.axes(r, (0, 90), (-5, 45), (0, 20, 40, 60, 80), (0, 20, 40), "cm")
        m = snap["profile_m"] * 100
        y = Y(10 * math.log10(math.exp(self.cfg.tbd_bias)))
        for xx in range(int(plot.x), int(plot.right), 8):
            pygame.draw.line(self.screen, YELLOW, (xx, y), (min(xx + 4, plot.right), y))
        self.text.draw(self.screen, "counts as evidence", (plot.right, y - 2 * self.u), int(11 * self.u), YELLOW,
                       anchor="bottomright")
        pygame.draw.rect(self.screen, (16, 18, 21), pygame.Rect(plot.x, plot.y, X(snap["gate"][0] * 100) - plot.x, plot.h))
        step = max(1, len(m) // 220)
        for s in range(prof.shape[0]):
            pygame.draw.aalines(self.screen, cols[s], False, list(zip(X(m[::step]), Y(np.clip(prof[s][::step], -5, 45)))))
        for s, d in enumerate(snap["det"]):
            if d is not None:
                pygame.draw.circle(self.screen, cols[s], (int(X(d["e"] * 100)), int(Y(min(44, to_db(d["snr"]))))),
                                   max(3, int(4 * self.u)))

    def response_plot(self, r, snap, cols):
        resp = snap.get("response")
        if resp is None:
            self.empty(r, "Measured during calibration")
            return
        f = resp["freqs"]
        snr = resp["snr_db"]
        plot, X, Y = self.axes(r, (9, 24), (0, 75), (9, 12, 15, 18, 21, 24), (0, 25, 50, 75), "kHz")
        band = resp["band"]
        if band:
            x0, x1 = X(band[0] / 1000), X(band[1] / 1000)
            pygame.draw.rect(self.screen, (40, 36, 24), pygame.Rect(x0, plot.y, x1 - x0, plot.h))
            self.text.draw(self.screen, f"probe {band[0] / 1000:.1f}-{band[1] / 1000:.1f} kHz",
                           ((x0 + x1) / 2, plot.y + 2), int(11 * self.u), YELLOW, anchor="midtop")
        for s in range(snr.shape[0]):
            ff = f[s] / 1000
            keep = (ff >= 9) & (ff <= 24)
            pts = list(zip(X(ff[keep]), Y(np.clip(snr[s][keep], 0, 75))))
            if len(pts) > 1:
                pygame.draw.aalines(self.screen, cols[s], False, pts)

    def readouts(self, r, snap):
        u = self.u
        pygame.draw.rect(self.screen, LINE, r.inflate(8 * u, 6 * u), 1)
        rows = []
        hand = snap.get("hand")
        if hand is not None:
            x, y = hand["pos"]
            vx, vy = hand["vel"]
            v = math.hypot(vx, vy)
            dirn = "toward the laptop" if vy < -0.03 else ("away" if vy > 0.03 else "sideways")
            rows.append(("Hand", f"x {x * 100:+.1f} cm,  y {y * 100:.1f} cm"))
            rows.append(("Speed", f"{v:.2f} m/s {dirn}" if v > 0.04 else "holding still"))
        else:
            rows.append(("Hand", "not detected"))
            rows.append(("Speed", "-"))
        dets = snap.get("det") or []
        if dets:
            names = ["L", "R"] if len(dets) == 2 else ["mono"]
            rows.append(("Echo strength", ",  ".join(f"{n} {to_db(d['snr']):.0f} dB" if d else f"{n} -"
                                                     for n, d in zip(names, dets))))
        if snap["band"]:
            b = snap["band"]
            line = f"{b[0] / 1000:.1f}-{b[1] / 1000:.1f} kHz"
            if snap.get("resolution_m"):
                line += f", {snap['resolution_m'] * 100:.0f} cm resolution"
            if b[0] < self.cfg.quiet_above - 1:
                line += "  (soft hiss)"
            rows.append(("Probe", line))
        if snap["latency_s"] is not None:
            rows.append(("Round trip", f"{snap['latency_s'] * 1000:.0f} ms,  clock skew {snap['drift_ppm']:+.1f} ppm"))
        if snap.get("dsp_hz"):
            line = f"{snap['dsp_hz']:.0f} updates/s,  {snap.get('proc_ms', 0):.2f} ms each"
            if self.io.skipped_s > 0.5:
                line += f"  (too busy, skipped {self.io.skipped_s:.0f} s)"
            rows.append(("Signal processing", line))
        info = snap["info"]
        if info.get("in_name"):
            rows.append(("Microphone", str(info["in_name"])[:38]))
        if info.get("out_name"):
            rows.append(("Speakers", str(info["out_name"])[:38]))
        y = r.y
        lh = int(19 * u)
        for lab, val in rows:
            if y + lh > r.bottom:
                break
            self.text.draw(self.screen, lab, (r.x, y), int(13 * u), GREY)
            self.text.draw(self.screen, val, (r.x + int(132 * u), y), int(13 * u), WHITE)
            y += lh

    def card(self, w, h, top=False, dim=150):
        u = self.u
        f = self.r_field
        shade = pygame.Surface(f.size, pygame.SRCALPHA)
        shade.fill((0, 0, 0, dim))
        self.screen.blit(shade, f.topleft)
        c = pygame.Rect(0, 0, int(w * u), int(h * u))
        if top:
            c.midtop = (f.centerx, f.y + int(18 * u))
        else:
            c.center = (f.centerx, f.y + f.h * 0.46)
        pygame.draw.rect(self.screen, BAR, c)
        pygame.draw.rect(self.screen, LINE, c, 1)
        return c

    def draw_calibration(self, snap):
        u = self.u
        st = snap["state"]
        if st == "error":
            err = snap["error"] or {}
            hints = err.get("hints", [])
            c = self.card(640, 150 + 30 * len(hints))
            y = c.y + 26 * u
            self.text.draw(self.screen, err.get("message", "Something went wrong"), (c.x + 28 * u, y), int(21 * u), RED)
            y += 44 * u
            for h in hints:
                self.text.draw(self.screen, h, (c.x + 28 * u, y), int(15 * u), WHITE)
                y += 30 * u
            self.text.draw(self.screen, "Press R to try again", (c.x + 28 * u, c.bottom - 34 * u), int(15 * u), YELLOW)
            return
        anchoring = st == "anchor"
        wl = [ln for w in snap["warnings"] for ln in self.text.wrap(w, int(13 * u), 540 * u)]
        c = self.card(600, 330 + 20 * len(wl), top=anchoring, dim=60 if anchoring else 150)
        if anchoring:
            ax, ay = self.cfg.anchor_pos
            x, y = self.w2s(ax, ay)
            pygame.draw.circle(self.screen, YELLOW, (int(x), int(y)), int(28 * u), 2)
            self.text.draw(self.screen, "palm here, hold it still", (x, y + 38 * u), int(15 * u), YELLOW, anchor="midtop")
        x = c.x + 30 * u
        y = c.y + 26 * u
        self.text.draw(self.screen, "Setting up the sonar", (x, y), int(22 * u), WHITE)
        y += 40 * u
        self.text.draw(self.screen, snap["message"], (x, y), int(17 * u), YELLOW)
        y += 42 * u
        for i, (k, label, status) in enumerate(snap["steps"]):
            mark = {"done": "[x]", "active": "[..]", "skipped": "[-]"}.get(status, "[ ]")
            col = {"done": GREEN, "active": YELLOW}.get(status, GREY)
            self.text.draw(self.screen, mark, (x, y + 1 * u), int(16 * u), col)
            self.text.draw(self.screen, f"{i + 1}. {label}", (x + 36 * u, y + 1 * u), int(16 * u),
                           WHITE if status in ("active", "done") else GREY)
            if status == "skipped":
                self.text.draw(self.screen, "skipped", (c.right - 30 * u, y + 1 * u), int(14 * u), DIM, anchor="topright")
            y += 34 * u
        bar = pygame.Rect(x, y + 6 * u, c.w - 60 * u, 8 * u)
        pygame.draw.rect(self.screen, BAR_HI, bar)
        fill = bar.copy()
        fill.w = int(bar.w * float(np.clip(snap["progress"], 0, 1)))
        pygame.draw.rect(self.screen, YELLOW, fill)
        y += 26 * u
        if anchoring:
            self.text.draw(self.screen, "Space skips this (positions will be a bit off)", (x, y), int(14 * u), GREY)
            y += 26 * u
        for ln in wl:
            self.text.draw(self.screen, ln, (x, y), int(13 * u), YELLOW)
            y += 20 * u


def dashed_rect(dst, r, col, dash):
    def seg(a, b):
        ax, ay = a
        bx, by = b
        n = max(1, int(math.hypot(bx - ax, by - ay) / dash))
        for i in range(0, n, 2):
            t0, t1 = i / n, min(1, (i + 1) / n)
            pygame.draw.line(dst, col, (ax + (bx - ax) * t0, ay + (by - ay) * t0),
                             (ax + (bx - ax) * t1, ay + (by - ay) * t1), 1)

    seg(r.topleft, r.topright)
    seg(r.topright, r.bottomright)
    seg(r.bottomright, r.bottomleft)
    seg(r.bottomleft, r.topleft)
