"""Render the five storyboard shots: 2.5D camera moves, depth of field, glows, colour grade.

  python render.py shot 1 2 3 4 5      render shots to raw frame files
  python render.py ends 1 2 3 4 5      re-render only the dissolve ends of those shots
  python render.py check               report any shot that samples outside the frame
  python render.py still 3 2.5 out.jpg render one frame of a shot (for look-dev)
"""
import math
import os
import sys

import cv2
import numexpr as ne
import numpy as np

from layers import CACHE, HRH, HRW, RADII, WORK, smoothstep

W, H, FPS = 1920, 1080, 30
K0 = W / HRW                      # output px per source px at zoom 1
SHOTS = os.path.join(WORK, "shots")
XFADE = 18                        # frames of overlap between shots
LEVELS = np.arange(len(RADII), dtype=np.float32)
UNIT = np.float32(1 / 65025)

U, V = np.meshgrid(np.arange(W, dtype=np.float32) - (W - 1) / 2,
                   np.arange(H, dtype=np.float32) - (H - 1) / 2)


def ss(t, a, b):
    """Smooth 0..1 ramp between times a and b."""
    x = min(max((t - a) / (b - a), 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def lerp(a, b, u):
    return a + (b - a) * u


# --------------------------------------------------------------------------------------
# Shot definitions. Each returns the camera / lens / light state at time t (seconds).
#   cx, cy   look-at point (fraction of the frame)      z      zoom
#   px, py   parallax in output px per unit of depth     kappa  dolly gain (near grows faster)
#   d0       depth that stays put under parallax         Df     focus depth, Ap aperture
#   extra    extra defocus in source px (focus pulls, dissolves)
# --------------------------------------------------------------------------------------
def shot1(t, dur):
    x = lerp(-1, 1, 0.5 * (t / dur) + 0.5 * ss(t, 0, dur))
    # focus lands on the DURIANS sign, then finds her, then him
    df = lerp(lerp(0.31, 0.49, ss(t, 3.5, 4.4)), 0.56, ss(t, 6.0, 6.9))
    return dict(cx=0.489 + 0.017 * x, cy=0.505, z=lerp(1.075, 1.09, ss(t, 0, dur)),
                px=-30 * x, py=0, kappa=0, d0=0.5, Df=df, Ap=70, flat=0.3,
                extra=44 * (1 - ss(t, 1.4, 2.9)))


def shot2(t, dur):
    u = ss(t, 0, dur)
    tilt = ss(t, 2.3, 4.3)
    return dict(cx=0.5, cy=lerp(0.52, 0.47, tilt), z=lerp(1.10, 1.22, u),
                px=lerp(8, -8, u), py=0, kappa=lerp(-0.05, 0.05, u), d0=0.5,
                Df=lerp(0.64, 0.355, ss(t, 2.6, 3.7)), Ap=34, flat=1.0)


def shot3(t, dur):
    x = lerp(-1, 1, 0.5 * (t / dur) + 0.5 * ss(t, 0, dur))
    to_vendor = ss(t, 1.9, 2.7) - ss(t, 3.9, 4.7)
    return dict(cx=0.5 + 0.022 * x, cy=0.475, z=lerp(1.10, 1.125, ss(t, 0, dur)),
                px=-22 * x, py=0, kappa=0, d0=0.6,
                Df=lerp(0.734, 0.15, to_vendor), Ap=26, flat=0.25)


def shot4(t, dur):
    u = 1 - (1 - min(t / 4.8, 1.0)) ** 2            # already moving on the cut, long settle
    return dict(cx=0.5, cy=lerp(0.585, 0.50, u), z=lerp(1.36, 1.075, u),
                px=lerp(-6, 6, t / dur), py=0, kappa=lerp(0.06, -0.04, u), d0=0.5,
                Df=0.53, Ap=30, flat=0.3)


def shot5(t, dur):
    u = ss(t, 0, dur)
    return dict(cx=lerp(0.50, 0.492, u), cy=lerp(0.50, 0.468, u), z=lerp(1.06, 1.19, u),
                px=lerp(10, -10, u), py=0, kappa=lerp(-0.05, 0.05, u), d0=0.6,
                Df=lerp(0.12, 0.719, ss(t, 0.5, 1.9)), Ap=32, flat=0.25,
                extra=44 * ss(t, 5.6, 7.5))


# look: light direction (screen), rim/halo colour, light wash position, grade
LOOK = {
    1: dict(light=(-0.55, -0.83), rim=(1.0, 0.94, 0.78), amb=0.25, wash=(0.36, -0.08),
            wash_col=(1.0, 0.90, 0.66), warm=0.00, sat=1.15),
    2: dict(light=(-0.45, -0.89), rim=(1.0, 0.93, 0.76), amb=0.25, wash=(0.30, -0.10),
            wash_col=(1.0, 0.88, 0.64), warm=0.01, sat=1.15),
    3: dict(light=(0.55, -0.83), rim=(1.0, 0.90, 0.74), amb=0.25, wash=(0.72, -0.08),
            wash_col=(1.0, 0.86, 0.62), warm=0.02, sat=1.15),
    4: dict(light=(0.60, -0.80), rim=(1.0, 0.89, 0.70), amb=0.25, wash=(0.86, -0.02),
            wash_col=(1.0, 0.84, 0.58), warm=0.03, sat=1.15),
    5: dict(light=(0.0, -1.0), rim=(1.0, 0.72, 0.50), amb=0.65, wash=(0.50, 0.36),
            wash_col=(1.0, 0.66, 0.42), warm=0.02, sat=1.15, rim_gain=1.35),
}
SHOT = {1: (shot1, 8.5), 2: (shot2, 5.6), 3: (shot3, 5.8), 4: (shot4, 5.2), 5: (shot5, 8.2)}
# animated sparkles: (source x, y, depth) anchors
SPARKS = {
    2: [(0.478, 0.50, 0.6), (0.497, 0.485, 0.6), (0.519, 0.505, 0.6), (0.505, 0.535, 0.6),
        (0.488, 0.548, 0.6), (0.530, 0.53, 0.6)],
    5: [(0.493, 0.455, 0.7), (0.515, 0.418, 0.7), (0.535, 0.452, 0.7), (0.523, 0.488, 0.7),
        (0.505, 0.515, 0.7)],
}


def params(n, t):
    fn, dur = SHOT[n]
    p = fn(t, dur)
    p.setdefault("extra", 0.0)
    if n > 1:                                    # every dissolve happens out of focus
        p["extra"] += 34 * (1 - ss(t, 0.0, 1.0))
    if n < 5:
        p["extra"] += 34 * ss(t, dur - 1.0, dur)
    p["extra"] = min(p["extra"], 44.0)
    # a little life in the camera, and focus breathing
    p["cx"] += 0.0009 * math.sin(2 * math.pi * 0.21 * t + n) + 0.0005 * math.sin(2 * math.pi * 0.47 * t + 2.1 * n)
    p["cy"] += 0.0008 * math.sin(2 * math.pi * 0.17 * t + 1.7 * n) + 0.0004 * math.sin(2 * math.pi * 0.39 * t + 0.6 * n)
    p["z"] *= 1 + 0.02 * (p["Df"] - fn(0, dur)["Df"])
    return p


# --------------------------------------------------------------------------------------
class Scene:
    def __init__(self, n):
        z = np.load(f"{CACHE}/s{n}_layers.npz")
        self.n = n
        self.L0, self.L1, self.box1 = z["L0"], z["L1"], z["box1"]
        self.L2 = z["L2"] if "L2" in z else None
        self.box2 = z["box2"] if "L2" in z else None
        self.D0c, self.D0w, self.D1, self.D1med = z["D0c"], z["D0w"], z["D1"], z["D1med"]
        self.glow = cv2.resize(z["glow"], (W, H), interpolation=cv2.INTER_LINEAR)
        self._cache = {}
        rng = np.random.default_rng(100 + n)
        self.dust = rng.random((42, 6)).astype(np.float32)
        self.spark_phase = rng.random((16, 3)).astype(np.float32)
        self.overshoot = 0.0

    def dof(self, name, stack, depth, p, tol=1.2):
        """Blend the blur stack per pixel from the circle of confusion."""
        key = (round(p["Df"], 4), round(p["Ap"], 2), round(p["extra"], 2), round(p.get("flat", 1), 2))
        hit = self._cache.get(name)
        if hit and hit[0] == key:
            return hit[1]
        rho = np.maximum(p["Ap"] * np.abs(depth - p["Df"]) - tol, 0) + p["extra"]
        idx = np.minimum(np.interp(rho, RADII, LEVELS), len(RADII) - 1.001).astype(np.float32)
        base = idx.astype(np.uint8)
        frac = ((idx - base) * 255 + 0.5).astype(np.uint8)
        inv = cv2.subtract(255, frac)
        ch = stack.shape[-1]
        acc = np.zeros(stack.shape[1:], np.float32)           # in units of 255*255
        for k in range(int(base.min()), int(base.max()) + 2):
            w = cv2.bitwise_and(inv, cv2.compare(base, k, cv2.CMP_EQ))
            w = cv2.bitwise_or(w, cv2.bitwise_and(frac, cv2.compare(base, k - 1, cv2.CMP_EQ)))
            cv2.accumulateProduct(stack[k], cv2.merge([w] * ch), acc)
        self._cache[name] = (key, acc)
        return acc

    def src_map(self, depth, p):
        """Where each output pixel samples the source, with depth-dependent parallax."""
        zk = p["z"] * K0
        cx, cy = p["cx"] * HRW, p["cy"] * HRH
        sx, sy = cx + U / zk, cy + V / zk
        for _ in range(2):
            d = cv2.remap(depth, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) - p["d0"]
            m = zk * (1 + p["kappa"] * d)
            sx, sy = cx + (U - p["px"] * d) / m, cy + (V - p["py"] * d) / m
        return sx, sy

    def project(self, x, y, d, p):
        """Source point (fractions) -> output pixel."""
        dd = d - p["d0"]
        m = p["z"] * K0 * (1 + p["kappa"] * dd)
        return (m * (x * HRW - p["cx"] * HRW) + p["px"] * dd + (W - 1) / 2,
                m * (y * HRH - p["cy"] * HRH) + p["py"] * dd + (H - 1) / 2)

    def frame(self, t):
        n = self.n
        p = params(n, t)
        look = LOOK[n]

        # ---- layers through the lens
        bg = self.dof("L0", self.L0, self.D0c, p)
        sx, sy = self.src_map(self.D0w, p)
        self.overshoot = max(self.overshoot, -sx.min(), sx.max() - (HRW - 1), -sy.min(), sy.max() - (HRH - 1))
        comp = cv2.remap(bg, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) * UNIT

        x0, y0, x1, y1 = (int(v) for v in self.box1)
        d1 = self.D1med + p["flat"] * (self.D1 - self.D1med)
        fg = self.dof("L1", self.L1, d1[y0:y1, x0:x1], p)
        fx_, fy_ = self.src_map(self.D1med + 0.6 * (self.D1 - self.D1med), p)
        fg = cv2.remap(fg, fx_ - np.float32(x0), fy_ - np.float32(y0), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) * UNIT
        a1 = fg[..., 3]
        fg = fg[..., :3]

        if self.L2 is not None:
            x0, y0, x1, y1 = (int(v) for v in self.box2)
            nr = self.dof("L2", self.L2, self.D0c[y0:y1, x0:x1], p)
            nr = cv2.remap(nr, sx - np.float32(x0), sy - np.float32(y0), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) * UNIT
            a2 = nr[..., 3]
        else:
            nr, a2 = None, np.zeros((H, W), np.float32)

        # ---- character glow: halo behind, rim light on the lit edges, soft diffusion
        pulse = 1 + 0.10 * math.sin(2 * math.pi * t / 3.4 + n)
        focus = 1 - 0.7 * min(p["extra"] / 30, 1)           # glow eases off when defocused
        ag = cv2.resize(a1 * self.glow, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        soft = cv2.GaussianBlur(ag, (0, 0), 3.0)
        gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=3)
        facing = np.clip(-(gx * look["light"][0] + gy * look["light"][1]) / (np.sqrt(gx * gx + gy * gy) + 1e-4), 0, 1)
        band = ag * np.clip(1 - cv2.GaussianBlur(ag, (0, 0), 4.0), 0, 1) * 2.0
        rim = band * (look["amb"] + (1 - look["amb"]) * facing ** 1.5)
        halo = cv2.GaussianBlur(ag, (0, 0), 16) * (1 - ag)
        rim = cv2.resize(rim, (W, H), interpolation=cv2.INTER_LINEAR)
        halo = cv2.resize(halo, (W, H), interpolation=cv2.INTER_LINEAR)
        rim_col = np.array(look["rim"], np.float32)

        comp = 1 - (1 - comp) * (1 - (0.30 * pulse * focus * halo)[..., None] * rim_col)
        comp = fg + comp * (1 - a1[..., None])
        comp = 1 - (1 - comp) * (1 - (0.42 * look.get("rim_gain", 1) * pulse * focus * rim)[..., None] * rim_col)
        diff = cv2.GaussianBlur(cv2.resize(fg * self.glow[..., None], (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 5)
        comp = 1 - (1 - comp) * (1 - 0.09 * focus * cv2.resize(diff, (W, H), interpolation=cv2.INTER_LINEAR))
        if nr is not None:
            comp = nr[..., :3] + comp * (1 - a2[..., None])

        # ---- atmosphere
        comp = self.light_wash(comp, look, t)
        comp = self.sparkles(comp, p, t)
        comp = self.dust_motes(comp, p, t, look)
        comp = bloom(comp)
        comp = grade(comp, look)
        return (np.clip(comp, 0, 1) * 255 + 0.5).astype(np.uint8)

    def light_wash(self, comp, look, t):
        lx = (look["wash"][0] + 0.01 * math.sin(0.5 * t)) * W
        ly = look["wash"][1] * H
        small = (W // 8, H // 8)
        u = cv2.resize(U + (W - 1) / 2, small) - lx
        v = cv2.resize(V + (H - 1) / 2, small) - ly
        r = np.sqrt(u * u + v * v) / W
        g = np.exp(-(r / 0.42) ** 2) * (0.11 + 0.015 * math.sin(0.9 * t + self.n))
        g = cv2.resize(g.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        return 1 - (1 - comp) * (1 - g[..., None] * np.array(look["wash_col"], np.float32))

    def sparkles(self, comp, p, t):
        pts = SPARKS.get(self.n)
        if not pts:
            return comp
        amp = max(0.0, 1 - p["extra"] / 14)
        if amp <= 0:
            return comp
        for i, (x, y, d) in enumerate(pts):
            ph = self.spark_phase[i]
            b = max(0.0, math.sin(2 * math.pi * (t * (0.55 + 0.5 * ph[0]) + ph[1]))) ** 2.2
            if b < 0.02:
                continue
            ox, oy = self.project(x, y, d, p)
            add_sprite(comp, star(int(13 + 12 * ph[2]), 0.25 * t * (ph[0] - 0.5)), ox, oy,
                       np.array([1.0, 0.95, 0.78], np.float32) * (1.0 * b * amp))
        return comp

    def dust_motes(self, comp, p, t, look):
        col = np.array(look["rim"], np.float32)
        for x, y, zp, ph, sp, tw in self.dust:
            r = 1.6 + 15 * zp ** 2.2
            a = (0.34 - 0.27 * zp) * (0.6 + 0.4 * math.sin(2 * math.pi * (0.13 + 0.2 * tw) * t + 6.28 * ph))
            px = (x * 1.2 - 0.1) * W + (5 + 22 * zp) * (0.5 + sp) * t + 9 * math.sin(0.6 * t + 6.28 * ph) + p["px"] * (zp - 0.4) * 1.5
            py = (y * 1.2 - 0.1) * H - (3 + 12 * zp) * (0.5 + sp) * t + 6 * math.cos(0.5 * t + 6.28 * tw)
            px = (px + 0.1 * W) % (1.2 * W) - 0.1 * W
            py = (py + 0.1 * H) % (1.2 * H) - 0.1 * H
            add_sprite(comp, disc(r), px, py, col * a)
        return comp


_sprites = {}


def disc(r):
    key = ("d", round(r * 2) / 2)
    if key not in _sprites:
        R = int(math.ceil(r * 1.6)) + 2
        y, x = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
        d = np.sqrt(x * x + y * y)
        s = smoothstep(r * 1.35, r * 0.55, d) if r > 5 else np.exp(-(d / (0.75 * r)) ** 2)
        _sprites[key] = s.astype(np.float32)
    return _sprites[key]


def star(size, rot):
    """Four-point twinkle with a soft core."""
    R = size + 2
    y, x = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    c, s = math.cos(rot), math.sin(rot)
    xr, yr = x * c + y * s, -x * s + y * c
    w = max(1.0, size * 0.075)
    arm = np.exp(-np.abs(xr) / (size * 0.36)) * np.exp(-(yr / w) ** 2) + \
          np.exp(-np.abs(yr) / (size * 0.36)) * np.exp(-(xr / w) ** 2)
    core = np.exp(-(xr * xr + yr * yr) / (2 * (size * 0.16) ** 2))
    glow = 0.22 * np.exp(-(xr * xr + yr * yr) / (2 * (size * 0.5) ** 2))
    return np.clip(arm * 0.8 + core + glow, 0, 1.5).astype(np.float32)


def add_sprite(img, spr, cx, cy, col):
    """Screen-blend a sprite into img (in place), nearest-pixel placement with sub-pixel shift."""
    h, w = spr.shape
    fx, fy = cx - w / 2, cy - h / 2
    ix, iy = int(math.floor(fx)), int(math.floor(fy))
    m = np.float32([[1, 0, fx - ix], [0, 1, fy - iy]])
    spr = cv2.warpAffine(spr, m, (w + 1, h + 1), flags=cv2.INTER_LINEAR)
    x0, y0, x1, y1 = max(ix, 0), max(iy, 0), min(ix + w + 1, W), min(iy + h + 1, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - iy:y1 - iy, x0 - ix:x1 - ix, None] * col
    reg = img[y0:y1, x0:x1]
    reg[:] = 1 - (1 - reg) * (1 - np.clip(s, 0, 1))


def bloom(comp):
    lum = comp @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    bright = comp * smoothstep(0.70, 1.0, lum)[..., None]
    q = cv2.resize(bright, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b = 0.6 * cv2.GaussianBlur(q, (0, 0), 5) + 0.4 * cv2.GaussianBlur(q, (0, 0), 16)
    return 1 - (1 - comp) * (1 - 0.15 * cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR))


_vig = None


def grade(x, look):
    """Gentle filmic contrast, warm highlights / cool shadows, vibrance, vignette."""
    global _vig
    if _vig is None:
        r = np.sqrt((U / (W / 2)) ** 2 + (V / (H / 2)) ** 2) / math.sqrt(2)
        _vig = (1 - 0.30 * smoothstep(0.45, 1.05, r)).astype(np.float32)
    np.clip(x, 0, 1, out=x)
    x += np.float32(0.34) * (x * x * (3 - 2 * x) - x)              # S-curve
    lum = cv2.transform(x, LUMA)
    sh, hi = (1 - lum) ** 2, lum * lum
    wm = look["warm"]
    c = cv2.split(x)
    mx, mn = cv2.max(cv2.max(c[0], c[1]), c[2]), cv2.min(cv2.min(c[0], c[1]), c[2])
    vib = 1 + np.float32((look["sat"] - 1) * 2.2) * (1 - (mx - mn) / (mx + 1e-4))   # lift muted colours more
    tone = ((-0.012, 0.030 + wm), (0.004, 0.010), (0.030, -0.030 - wm))
    out = [(lum + (c[i] + sh * np.float32(tone[i][0]) + hi * np.float32(tone[i][1]) - lum) * vib) * _vig for i in range(3)]
    return cv2.merge(out)


LUMA = np.float32([[0.2126, 0.7152, 0.0722]])


def render_shot(n, ranges=None):
    """Render a shot to SHOTS/sN.npy. ranges=[(a, b), ...] re-renders only those frames."""
    os.makedirs(SHOTS, exist_ok=True)
    sc = Scene(n)
    frames = int(round(SHOT[n][1] * FPS))
    path = f"{SHOTS}/s{n}.npy"
    if ranges:
        out = np.load(path, mmap_mode="r+")
        todo = [i for a, b in ranges for i in range(max(a, 0), min(b, frames))]
    else:
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(frames, H, W, 3))
        todo = list(range(frames))
    for j, i in enumerate(todo):
        out[i] = sc.frame(i / FPS)
        if j % 15 == 0:
            print(f"shot {n} frame {i} ({j}/{len(todo)})", flush=True)
    out.flush()
    print(f"shot {n} done, max source overshoot {sc.overshoot:.1f}px", flush=True)


if __name__ == "__main__":
    cv2.setNumThreads(1)
    ne.set_num_threads(1)
    if sys.argv[1] == "shot":
        for n in map(int, sys.argv[2:]):
            render_shot(n)
    elif sys.argv[1] == "ends":            # re-render only the first / last second of each shot
        for n in map(int, sys.argv[2:]):
            f = int(round(SHOT[n][1] * FPS))
            render_shot(n, ([(0, 32)] if n > 1 else []) + ([(f - 32, f)] if n < 5 else []))
    elif sys.argv[1] == "check":           # does any shot sample outside the source frame?
        for n in SHOT:
            sc = Scene(n)
            worst = 0.0
            for i in range(0, int(SHOT[n][1] * FPS), 6):
                sx, sy = sc.src_map(sc.D0w, params(n, i / FPS))
                worst = max(worst, -sx.min(), sx.max() - (HRW - 1), -sy.min(), sy.max() - (HRH - 1))
            print(f"shot {n}: overshoot {worst:.1f}px")
    elif sys.argv[1] == "still":
        n, t, path = int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
        sc = Scene(n)
        import time
        t0 = time.time()
        img = sc.frame(t)
        print(f"{time.time() - t0:.2f}s, overshoot {sc.overshoot:.1f}")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
