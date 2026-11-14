"""Render the five storyboard shots: character animation, 2.5D camera moves, depth of field, glows, grade.

  python render.py shot 1 2 3 4 5      render shots to raw frame files
  python render.py check               report any shot that samples outside the frame
  python render.py still 3 2.5 out.jpg render one frame of a shot (for look-dev)
"""
import math
import os
import sys

import cv2
import numexpr as ne
import numpy as np

from layers import CACHE, HRH, HRW, RADII, SCENES, WORK, smoothstep

W, H, FPS = 1920, 1080, 30
K0 = W / HRW                      # output px per source px at zoom 1
SHOTS = os.path.join(WORK, "shots")
XFADE = 6                         # frames of overlap between shots
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
# Shot definitions. Each returns the camera / lens state at time t (seconds).
#   cx, cy     look-at point (fraction of the frame)      z      zoom
#   px, py     parallax in output px per unit of depth     kappa  dolly gain (near grows faster)
#   d0         depth that stays put under parallax
#   Df         focus depth. It stays on the two leads: they and whatever they touch are always sharp
#   far, near  blur in source px per unit of depth behind / in front of the focal plane (kept gentle)
#   extra      defocus of the whole frame in source px (only the closing shot uses it)
# --------------------------------------------------------------------------------------
def shot1(t, dur):
    x = lerp(-1, 1, 0.5 * (t / dur) + 0.5 * ss(t, 0, dur))
    return dict(cx=0.489 + 0.017 * x, cy=0.505, z=lerp(1.075, 1.09, ss(t, 0, dur)),
                px=-30 * x, py=0, kappa=0, d0=0.5, Df=0.52, far=16, near=9)


def shot2(t, dur):
    u = ss(t, 0, dur)
    tilt = ss(t, 2.3, 4.3)
    return dict(cx=0.5, cy=lerp(0.52, 0.47, tilt), z=lerp(1.10, 1.22, u),
                px=lerp(8, -8, u), py=0, kappa=lerp(-0.05, 0.05, u), d0=0.5,
                Df=0.354, far=30, near=9)


def shot3(t, dur):
    x = lerp(-1, 1, 0.5 * (t / dur) + 0.5 * ss(t, 0, dur))
    return dict(cx=0.5 + 0.022 * x, cy=0.475, z=lerp(1.10, 1.125, ss(t, 0, dur)),
                px=-22 * x, py=0, kappa=0, d0=0.6, Df=0.734, far=10, near=9)


def shot4(t, dur):
    u = 1 - (1 - min(t / 4.8, 1.0)) ** 2            # already moving on the cut, long settle
    return dict(cx=0.5, cy=lerp(0.585, 0.50, u), z=lerp(1.36, 1.075, u),
                px=lerp(-6, 6, t / dur), py=0, kappa=lerp(0.06, -0.04, u), d0=0.5,
                Df=0.53, far=14, near=8)


def shot5(t, dur):
    u = ss(t, 0, dur)
    return dict(cx=lerp(0.50, 0.492, u), cy=lerp(0.50, 0.468, u), z=lerp(1.06, 1.19, u),
                px=lerp(10, -10, u), py=0, kappa=lerp(-0.05, 0.05, u), d0=0.6,
                Df=0.719, far=16, near=9, extra=40 * ss(t, 6.1, 7.7))    # melts into bokeh for the end card


# --------------------------------------------------------------------------------------
# Acting. Per shot and actor a list of (start time, key pose, seconds the change takes); the first
# entry is the opening pose. Scenes with a split have "her" and "him", the others act as "all".
# --------------------------------------------------------------------------------------
def blink(t):
    return [(t, "blink", 0.07), (t + 0.13, "base", 0.11)]


ACT = {
    1: {"her": [(0, "base", 0)] + blink(1.7) + blink(4.4) + blink(4.75) + blink(7.3),
        "him": [(0, "base", 0), (5.5, "lookup", 0.32)]},               # he notices the durians
    2: {"all": [(0, "a0", 0), (0.5, "a", 0.6), (1.1, "base", 0.22),      # reach ... touch!
                (3.2, "c", 0.4)]},                                       # eyes meet
    3: {"her": [(0, "base", 0), (1.3, "a", 0.25), (3.6, "base", 0.25)],  # she peeks at him
        "him": [(0, "base", 0), (2.0, "a", 0.25), (4.4, "base", 0.3)]},  # he cracks up
    4: {"her": [(0, "base", 0), (1.9, "a", 0.35), (4.0, "base", 0.35)],  # lowers the piece, another bite
        "him": [(0, "base", 0), (0.9, "a", 0.3), (3.5, "base", 0.3)]},   # fans the smell away
    5: {"her": [(0, "base", 0)] + blink(1.5) + blink(3.6) + [(4.9, "a", 0.35)],
        "him": [(0, "base", 0), (4.5, "a", 0.35)]},                      # he looks up, she smiles back
}
TOUCH = 1.3                                          # shot 2: the moment the hands land


def acting(n, actor, t):
    """-> (pose A, pose B, u): the actor is u of the way from A to B."""
    track = ACT[n][actor]
    cur = track[0][1]
    state = (cur, cur, 0.0)
    for start, pose, dur in track[1:]:
        if t < start:
            break
        state = (cur, pose, ss(t, start, start + dur)) if t < start + dur else (pose, pose, 0.0)
        cur = pose
    if n == 4 and actor == "him" and 1.2 < t < 3.5:  # his hand keeps fanning: warp part-way back and forth
        env = ss(t, 1.2, 1.5) * (1 - ss(t, 3.0, 3.5))
        return ("a", "base", 0.45 * env * (0.5 - 0.5 * math.cos(2 * math.pi * 3.0 * (t - 1.2))), "warp")
    return state


def giggle(t, a, b, amp=3.5, rate=4.3):
    return amp * ss(t, a, a + 0.25) * (1 - ss(t, b - 0.3, b)) * math.sin(2 * math.pi * rate * (t - a))


WALK = (760.0, 4.6, 150.0)                           # shot 1: distance he covers, seconds, step length


def motion(n, actor, t):
    """Whole-body movement of an actor in source px: offset x, y, then lean and bounce at head height."""
    ox = oy = lean = bounce = 0.0
    if n == 1 and actor == "him":                    # strolls in, eyes on his list, and slows to a stop
        dist, secs, step = WALK
        left = dist * (1 - min(t / secs, 1.0)) ** 1.7
        ox = -left
        oy = 5.0 * min(left / 60, 1.0) * (0.5 - 0.5 * math.cos(2 * math.pi * left / step))
        lean = 6.0 * min(left / 200, 1.0) - 4.0 * ss(t, 5.5, 5.9)
    if n == 1 and actor == "her":
        lean = 5.0 * ss(t, 4.9, 6.0)                 # leans towards the stall
    if n == 2 and t > TOUCH:                         # startled little jump
        bounce = -9.0 * math.exp(-5.0 * (t - TOUCH)) * math.cos(2 * math.pi * 2.6 * (t - TOUCH))
    if n == 3:
        bounce = giggle(t, 0.0, 1.4) + giggle(t, 3.7, 5.8) if actor == "her" else giggle(t, 2.1, 4.5, 4.0)
    if n == 4:
        bounce = giggle(t, 0.0, 2.0) + giggle(t, 4.1, 5.2) if actor == "her" else giggle(t, 0.3, 5.2, 2.6, 3.7)
    return ox, oy, lean, bounce


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
    # a little life in the camera
    p["cx"] += 0.0009 * math.sin(2 * math.pi * 0.21 * t + n) + 0.0005 * math.sin(2 * math.pi * 0.47 * t + 2.1 * n)
    p["cy"] += 0.0008 * math.sin(2 * math.pi * 0.17 * t + 1.7 * n) + 0.0004 * math.sin(2 * math.pi * 0.39 * t + 0.6 * n)
    return p


# --------------------------------------------------------------------------------------
class Scene:
    def __init__(self, n):
        z = np.load(f"{CACHE}/s{n}_layers.npz")
        self.n = n
        self.L0, self.box1 = z["L0"], [int(v) for v in z["box1"]]
        self.poses = {k[2:]: z[k] for k in z.files if k.startswith("P_")}     # blur stack per key pose
        self.rims = {k[2:]: z[k] for k in z.files if k.startswith("A_")}      # alpha the glow follows
        self.flows = {tuple(k[2:].split("__")): z[k].astype(np.float32) for k in z.files if k.startswith("F_")}
        self.L2 = z["L2"] if "L2" in z else None
        self.box2 = [int(v) for v in z["box2"]] if "L2" in z else None
        self.D0c, self.D0w = z["D0c"], z["D0w"]
        self.Dsub = z["D1med"] + 0.6 * (z["D1"] - z["D1med"])                 # depth the characters ride on
        self.glow = cv2.resize(z["glow"], (W, H), interpolation=cv2.INTER_LINEAR)
        self.split = SCENES[n].get("split")
        self.actors = ["her", "him"] if self.split else ["all"]
        self.foot = self.box1[3] - 60                                         # body motion grows from here up
        self.tall = max(self.box1[3] - self.box1[1] - 120, 1)
        self._cache = {}
        rng = np.random.default_rng(100 + n)
        self.dust = rng.random((42, 6)).astype(np.float32)
        self.spark_phase = rng.random((16, 3)).astype(np.float32)
        self.overshoot = 0.0

    def dof(self, name, stack, depth, p, tol=2.0):
        """Blend the blur stack per pixel from the circle of confusion."""
        key = (round(p["Df"], 4), p["far"], p["near"], round(p["extra"], 2))
        hit = self._cache.get(name)
        if hit and hit[0] == key:
            return hit[1]
        gain = np.where(depth < p["Df"], np.float32(p["far"]), np.float32(p["near"]))
        rho = np.maximum(gain * np.abs(depth - p["Df"]) - tol, 0) + p["extra"]
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

    def level(self, pose, rho):
        """A key pose blurred by rho source px (the characters are flat cut-outs: one blur for all of them)."""
        stack = self.poses[pose]
        if len(stack) == 1 or rho < 0.05:
            return stack[0]
        key = ("lv", pose)
        hit = self._cache.get(key)
        if hit and hit[0] == round(rho, 2):
            return hit[1]
        idx = min(float(np.interp(rho, RADII, LEVELS)), len(stack) - 1.001)
        k = int(idx)
        img = cv2.addWeighted(stack[k], 1 - (idx - k), stack[k + 1], idx - k, 0)
        self._cache[key] = (round(rho, 2), img)
        return img

    def sample(self, pose, x, y, rho):
        """One key pose at source coords x, y -> premultiplied RGBA and the alpha its glow follows (0..1)."""
        mx, my = x - np.float32(self.box1[0]), y - np.float32(self.box1[1])
        rgba = cv2.remap(self.level(pose, rho), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rgba = rgba.astype(np.float32) * np.float32(1 / 255)
        if pose not in self.rims:
            return rgba, rgba[..., 3]
        rim = cv2.remap(self.rims[pose], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return rgba, rim.astype(np.float32) * np.float32(1 / 255)

    def perform(self, state, x, y, rho):
        """The characters part-way between two key poses: warp both along the optical flow and blend."""
        a, b, u = state[:3]
        if a == b or u <= 0.001:
            return self.sample(a, x, y, rho)
        warp_only = len(state) > 3
        if u >= 0.999 and not warp_only:
            return self.sample(b, x, y, rho)
        if (a, b) not in self.flows:                             # no flow for this pair: quick dissolve
            (ra, ga), (rb, gb) = self.sample(a, x, y, rho), self.sample(b, x, y, rho)
            return ra + (rb - ra) * np.float32(u), ga + (gb - ga) * np.float32(u)
        lx, ly = x * np.float32(0.5) - np.float32(0.25), y * np.float32(0.5) - np.float32(0.25)
        f01 = cv2.remap(self.flows[(a, b)], lx, ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        f10 = cv2.remap(self.flows[(b, a)], lx, ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fa = np.float32(-(1 - u) * u) * f01 + np.float32(u * u) * f10
        ra, ga = self.sample(a, x + fa[..., 0], y + fa[..., 1], rho)
        if warp_only:
            return ra, ga
        fb = np.float32((1 - u) ** 2) * f01 - np.float32(u * (1 - u)) * f10
        rb, gb = self.sample(b, x + fb[..., 0], y + fb[..., 1], rho)
        w = np.float32(ss(u, 0.2, 0.8))                          # shapes travel the whole way, drawing swaps mid-way
        return ra + (rb - ra) * w, ga + (gb - ga) * w

    def cast(self, p, t):
        """All characters of the shot, acted and moved -> premultiplied RGBA, glow alpha."""
        n = self.n
        fx, fy = self.src_map(self.Dsub, p)
        hgt = np.clip((self.foot - fy) * np.float32(1 / self.tall), 0, 1.3)
        xn = fx * np.float32(1 / HRW)
        # idle life: slow sway and breathing, out of phase from left to right
        dx = hgt * np.sin(xn * np.float32(2.5) + np.float32(2 * math.pi * 0.19 * t + 1.3 * n)) * np.float32(3.0)
        dy = hgt * np.sin(xn * np.float32(3.1) + np.float32(2 * math.pi * 0.27 * t + 0.7 * n)) * np.float32(2.4)
        todo = {}
        for actor in self.actors:
            ox, oy, lean, bounce = motion(n, actor, t)
            if self.split:
                w = smoothstep(self.split - 0.012, self.split + 0.012, xn - np.float32(ox / HRW))
                w = w if actor == "him" else 1 - w
                dx += w * hgt * np.float32(lean)
                dy += w * hgt * np.float32(bounce)
            else:
                w = None
                dx += hgt * np.float32(lean)
                dy += hgt * np.float32(bounce)
            key = (acting(n, actor, t), ox, oy)
            todo[key] = w if key not in todo or w is None else todo[key] + w
        out = None
        for (state, ox, oy), w in todo.items():
            rgba, rim = self.perform(state, fx - np.float32(ox) - dx, fy - np.float32(oy) - dy, p["extra"])
            if len(todo) == 1:
                return rgba, rim
            rgba *= w[..., None]
            rim = rim * w
            out = (rgba, rim) if out is None else (out[0] + rgba, out[1] + rim)
        return out

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

        fg, rim_a = self.cast(p, t)
        a1 = fg[..., 3]
        fg = fg[..., :3]

        if self.L2 is not None:
            x0, y0, x1, y1 = self.box2
            nr = self.dof("L2", self.L2, self.D0c[y0:y1, x0:x1], p)
            nr = cv2.remap(nr, sx - np.float32(x0), sy - np.float32(y0), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) * UNIT
            a2 = nr[..., 3]
        else:
            nr, a2 = None, np.zeros((H, W), np.float32)

        # ---- character glow: halo behind, rim light on the lit edges, soft diffusion
        pulse = 1 + 0.10 * math.sin(2 * math.pi * t / 3.4 + n)
        focus = 1 - 0.7 * min(p["extra"] / 30, 1)           # glow eases off when defocused
        ag = cv2.resize(rim_a * self.glow, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
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

        comp = 1 - (1 - comp) * (1 - (0.22 * pulse * focus * halo)[..., None] * rim_col)
        comp = fg + comp * (1 - a1[..., None])
        comp = 1 - (1 - comp) * (1 - (0.32 * look.get("rim_gain", 1) * pulse * focus * rim)[..., None] * rim_col)
        diff = cv2.GaussianBlur(cv2.resize(fg * self.glow[..., None], (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 5)
        comp = 1 - (1 - comp) * (1 - 0.07 * focus * cv2.resize(diff, (W, H), interpolation=cv2.INTER_LINEAR))
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
        amp = max(0.0, 1 - p["extra"] / 14) * (ss(t, TOUCH + 0.05, TOUCH + 0.5) if self.n == 2 else 1.0)
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
