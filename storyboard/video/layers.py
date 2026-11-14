"""Split each storyboard frame into depth layers and pre-blur them for depth of field.

Layers per scene
  L0  background plate  - the whole frame with the subject painted out
  P_* subject           - the characters (clean matte), premultiplied RGBA, one per key pose
  L2  near foreground   - things closer to camera than the subject, premultiplied RGBA
  F_* optical flow between key poses, used to warp one pose into the next

Every layer is stored as a stack of disk-blurred copies (RADII, in source pixels).
The renderer picks a blur level per pixel from the circle of confusion.
"""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("DURIAN_WORK", "/tmp/work")
PREP = os.path.join(WORK, "prep")      # upscaled frames from prep.py
CACHE = os.path.join(WORK, "cache")
MAPS = os.path.join(HERE, "maps")
POSES = os.path.join(HERE, "poses")

HRW, HRH = 2752, 1536
RADII = np.array([0, 2, 4, 7, 11, 16, 23, 32, 44], np.float32)

# Per-scene recipe. All coordinates are fractions of the frame.
#   t_lo / t_hi   subject = matte & depth inside this window (depth: 0 far .. 1 near)
#   ymax          ignore matte below this line
#   ysplit+poly   below ysplit only keep matte inside poly
#   strict        boxes where the depth test is relaxed less (keeps the vendor out)
#   exact         boxes where the depth test is not relaxed at all
#   exclude       boxes that never belong to the subject
#   depth_add     (box, lo, hi): also take pixels in this depth range (parts the matte missed)
#   near          (lo, hi) depth ramp for the near-foreground layer, near_box limits it
#   glow          boxes the character glow is limited to (None = whole subject)
#   hb            highlight boost for bokeh
#   poses         name -> (edited frame in poses/, boxes the change is limited to or None)
#   morphs        pose pairs that get optical flow, so one can be warped into the other
#   split         x that separates "her" (left) from "him" (right); without it they act as one
#   defocus       the shot throws the characters out of focus, so they need a blur stack too
#   plate         a clean, people-free version of the frame, used as the background
SCENES = {
    1: dict(t_lo=0.38, t_hi=0.72, ymax=0.90, near=(0.68, 0.80), hb=1.0, exact=[(0.84, 0.60, 1.0, 0.92)],
            depth_add=[((0.80, 0.30, 1.0, 0.68), 0.57, 0.72), ((0.84, 0.68, 0.935, 0.90), 0.45, 0.72)],
            poses={"blink": ("s1_b_blink.png", [(0.08, 0.20, 0.27, 0.50)]),
                   "lookup": ("s1_c_lookup.png", [(0.74, 0.02, 1.0, 0.50)])},
            morphs=[("base", "lookup")], split=0.5, plate="s1_plate.png"),
    2: dict(t_lo=0.18, ysplit=0.60,
            poly=[(0.30, 0.59), (0.36, 0.655), (0.415, 0.675), (0.42, 0.80), (0.45, 0.90),
                  (0.50, 0.915), (0.55, 0.90), (0.585, 0.80), (0.59, 0.69), (0.66, 0.665),
                  (0.72, 0.59)],
            near=(0.78, 0.90), hb=1.0,
            poses={"a0": ("s2_a0_start.png", None), "a": ("s2_a_reach.png", None),
                   "c": ("s2_c_eyes_meet.png", None)},
            morphs=[("a0", "a"), ("a", "base"), ("base", "c")]),
    3: dict(t_lo=0.45, strict=[(0.42, 0.15, 0.565, 0.67)], hb=1.0,
            glow=[(0.0, 0.08, 0.37, 1.0), (0.53, 0.0, 0.96, 1.0)],
            poses={"a": ("s3_a_glance.png", None)}, morphs=[("base", "a")], split=0.48),
    4: dict(t_lo=0.33, strict=[(0.42, 0.0, 0.61, 0.53)], exclude=[(0.0, 0.0, 0.245, 1.0)],
            near=(0.58, 0.70), near_box=(0.83, 0.62, 1.0, 1.0), hb=1.0,
            glow=[(0.26, 0.10, 0.80, 1.0)],
            poses={"a": ("s4_a_laugh.png", None)}, morphs=[("base", "a")], split=0.50),
    5: dict(t_lo=0.45, hb=5.0, defocus=True,
            poses={"a": ("s5_a_eyes_meet.png", None),
                   "blink": ("s5_b_blink.png", [(0.28, 0.25, 0.50, 0.60)])},
            morphs=[("base", "a")], split=0.52),
}


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def box_mask(boxes, w, h, feather=0):
    m = np.zeros((h, w), np.float32)
    for x0, y0, x1, y1 in boxes:
        m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 1
    if feather:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m


def disk_kernel(r):
    R = int(math.ceil(r)) + 1
    y, x = np.mgrid[-R:R + 1, -R:R + 1]
    k = np.clip(r + 0.5 - np.sqrt(x * x + y * y), 0, 1).astype(np.float32)
    return k / k.sum()


def disk_blur(img, r):
    """Disk (bokeh-shaped) blur. Large radii are done at reduced size."""
    if r <= 0:
        return cv2.GaussianBlur(img, (0, 0), 0.6)
    f = 1
    while r / f > 6:
        f *= 2
    if f == 1:
        return cv2.filter2D(img, -1, disk_kernel(r), borderType=cv2.BORDER_REFLECT)
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)
    small = cv2.filter2D(small, -1, disk_kernel(r / f), borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def subject_matte(n, matte, depth):
    """Clean subject alpha at map resolution from the saliency matte and the depth map."""
    c = SCENES[n]
    h, w = matte.shape
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    d_relaxed = cv2.dilate(depth, k5)
    if "strict" in c:
        s = box_mask(c["strict"], w, h) > 0
        d_relaxed = np.where(s, cv2.dilate(depth, k2), d_relaxed)
    if "exact" in c:
        d_relaxed = np.where(box_mask(c["exact"], w, h) > 0, depth, d_relaxed)
    b = (matte > 0.5) & (d_relaxed > c["t_lo"])
    if "t_hi" in c:
        b &= depth < c["t_hi"]
    yy = np.arange(h)[:, None] / h
    if "ymax" in c:
        b &= yy < c["ymax"]
    if "ysplit" in c:
        poly = np.zeros((h, w), np.uint8)
        pts = np.array([[x * w, y * h] for x, y in c["poly"]], np.int32)
        cv2.fillPoly(poly, [pts], 1)
        b &= (yy < c["ysplit"]) | (poly > 0)
    if "exclude" in c:
        b &= box_mask(c["exclude"], w, h) == 0
    added = np.zeros_like(b)
    for box, lo, hi in c.get("depth_add", []):
        added |= (box_mask([box], w, h) > 0) & (depth > lo) & (depth < hi) & ~b
    b = (b | added).astype(np.uint8)
    # drop specks, keep sizeable blobs
    num, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    keep = np.zeros(num, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > 0.0015 * h * w
    b = keep[lab].astype(np.uint8)
    # fill pin holes, and bigger holes that sit at subject depth (matte dropouts, not see-through gaps)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(1 - b, connectivity=4)
    fill = np.zeros(num, bool)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw == w or y + bh == h:
            continue
        fill[i] = area < 160 or np.median(depth[lab == i]) > c["t_lo"] + 0.04
    b = np.maximum(b, fill[lab].astype(np.uint8))
    core = cv2.erode(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    reach = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    a = smoothstep(0.22, 0.78, matte) * reach
    a = np.maximum(a, cv2.GaussianBlur((added & (b > 0)).astype(np.float32), (0, 0), 0.8))
    return np.maximum(a, core.astype(np.float32)), b


def nearest_fill(values, valid):
    """Fill invalid pixels with the value of the nearest valid pixel."""
    _, labels = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(valid)
    lut = np.zeros(labels.max() + 1, values.dtype)
    lut[labels[ys, xs]] = values[ys, xs]
    return lut[labels]


def build_stack(rgb_lin, alpha, hb, gamma=2.2, radii=RADII):
    """Highlight-weighted blur stack. Returns list of uint8 arrays (RGB or premultiplied RGBA).

    rgb_lin is sRGB ** gamma. gamma 2.2 blurs in linear light (bright bokeh); gamma 1 keeps
    defocused faces from turning into white-eyed blobs."""
    lum = rgb_lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    wgt = 1 + hb * smoothstep(0.45, 0.95, lum) ** 2
    wa = wgt if alpha is None else wgt * alpha
    src = np.dstack([rgb_lin * wa[..., None], wa]).astype(np.float32)
    out = []
    for r in radii:
        b = disk_blur(src, float(r))
        col = np.clip(b[..., :3] / np.maximum(b[..., 3:4], 1e-5), 0, 1) ** (1 / gamma)
        if alpha is None:
            out.append((col * 255 + 0.5).astype(np.uint8))
        else:
            a = np.clip(disk_blur(alpha, float(r)), 0, 1)
            out.append((np.dstack([col * a[..., None], a]) * 255 + 0.5).astype(np.uint8))
    return out


def bbox(a, pad):
    ys, xs = np.nonzero(a > 0.004)
    h, w = a.shape
    return (max(0, xs.min() - pad), max(0, ys.min() - pad),
            min(w, xs.max() + pad + 1), min(h, ys.max() + pad + 1))


def load_pose(n, name):
    """Upscaled frame, saliency matte, depth and change mask of a pose ('base' = the storyboard frame)."""
    tag = f"s{n}" if name == "base" else f"s{n}_{name}"
    hr = cv2.cvtColor(cv2.imread(f"{PREP}/{tag}_hr.png"), cv2.COLOR_BGR2RGB)
    matte = cv2.imread(f"{MAPS}/{tag}_matte.png", 0).astype(np.float32) / 255
    d = cv2.imread(f"{MAPS}/{tag}_depth.png", -1)
    depth = d.astype(np.float32) / (65535 if d.dtype == np.uint16 else 255)
    mask = None if name == "base" else cv2.imread(f"{MAPS}/{tag}_mask.png", 0).astype(np.float32) / 255
    return hr, matte, depth, mask


def morph_flow(a, b, zone):
    """Optical flow a->b and b->a (map resolution, in upscaled px), kept only where both directions agree."""
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(1)
    dis.setVariationalRefinementIterations(5)
    f01, f10 = dis.calc(ga, gb, None), dis.calc(gb, ga, None)
    h, w = ga.shape
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = []
    for f, g in ((f01, f10), (f10, f01)):
        back = cv2.remap(g, x + f[..., 0], y + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err = np.sqrt(((f + back) ** 2).sum(axis=2))
        conf = cv2.GaussianBlur((1 - smoothstep(2.0, 6.0, err)) * zone, (0, 0), 3)
        f = cv2.GaussianBlur(f, (0, 0), 1.2) * conf[..., None]
        out.append((f * (HRW / w)).astype(np.float16))
    return out


def build(n, debug=False):
    c = SCENES[n]
    names = ["base"] + list(c["poses"])
    hr, matte, depth, _ = load_pose(n, "base")
    h, w = matte.shape
    a_base, b1 = subject_matte(n, matte, depth)

    # every pose keeps the base cut-out except where the drawing really changed
    frames, alpha_lo, solid, masks = {"base": hr}, {"base": a_base}, {"base": b1}, {"base": np.zeros_like(a_base)}
    for name in names[1:]:
        phr, pmatte, pdepth, m = load_pose(n, name)
        a_raw, b_raw = subject_matte(n, pmatte, pdepth)
        zone = np.clip(cv2.GaussianBlur(cv2.dilate(m, np.ones((9, 9), np.uint8)), (0, 0), 2), 0, 1)
        frames[name], masks[name] = phr, m
        alpha_lo[name] = a_base * (1 - zone) + a_raw * zone
        solid[name] = np.where(zone > 0.5, b_raw, b1).astype(np.uint8)
    alpha = {}
    for name in names:
        a = np.clip(cv2.resize(alpha_lo[name], (HRW, HRH), interpolation=cv2.INTER_CUBIC), 0, 1)
        alpha[name] = smoothstep(0.12, 0.88, a).astype(np.float32)
    a_any = np.max([alpha[k] for k in names], axis=0)
    D = cv2.resize(depth, (HRW, HRH), interpolation=cv2.INTER_CUBIC)

    # subject depth: taken from inside the matte and pushed outwards over the edge
    core = cv2.erode(b1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    d1_lo = nearest_fill(depth, core) if core.any() else depth
    D1 = cv2.GaussianBlur(cv2.resize(d1_lo, (HRW, HRH), interpolation=cv2.INTER_LINEAR), (0, 0), 5)
    # one representative depth per connected blob, so each character moves as one piece
    num, lab = cv2.connectedComponents(b1, connectivity=8)
    med_lo = np.zeros_like(depth)
    for i in range(1, num):
        sel = (lab == i) & core
        if sel.any():
            med_lo[lab == i] = np.median(depth[sel])
            print(f"  scene {n} blob {i}: depth {np.median(depth[sel]):.3f}, area {sel.mean():.3f}")
    med_lo = nearest_fill(med_lo, core) if core.any() else depth
    D1med = cv2.GaussianBlur(cv2.resize(med_lo, (HRW, HRH), interpolation=cv2.INTER_LINEAR), (0, 0), 5)

    # background plate: what is behind the characters
    carry = {}
    if c.get("plate"):                      # a clean, people-free painting of the same view
        plate = cv2.cvtColor(cv2.imread(f"{PREP}/s{n}_plate_hr.png"), cv2.COLOR_BGR2RGB).astype(np.float32)
        d0_lo = cv2.imread(f"{MAPS}/s{n}_plate_depth.png", -1).astype(np.float32) / 65535
    else:                                   # paint the characters out
        hole = cv2.dilate(b1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        small = cv2.resize(hr, (w, h), interpolation=cv2.INTER_AREA)
        plate_lo = cv2.inpaint(small, hole, 7, cv2.INPAINT_TELEA)
        d0_lo = cv2.inpaint((depth * 255).astype(np.uint8), hole, 7, cv2.INPAINT_TELEA).astype(np.float32) / 255
        d0_lo = np.where(hole > 0, d0_lo, depth)
        hole_hr = cv2.GaussianBlur(cv2.resize(hole.astype(np.float32), (HRW, HRH)), (0, 0), 2.5)[..., None]
        plate = hr * (1 - hole_hr) + cv2.resize(plate_lo, (HRW, HRH), interpolation=cv2.INTER_CUBIC) * hole_hr
        # where a pose uncovers background the plate never had (a hand moved away, sparkles not
        # there yet), the pose brings its own background along
        for name in names[1:]:
            q = cv2.resize(frames[name], (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
            differs = cv2.GaussianBlur(np.abs(q - plate_lo).max(axis=2), (0, 0), 1.5) > 26
            need = (masks[name] > 0.5) & (solid[name] == 0) & ((hole > 0) | differs)
            need = cv2.morphologyEx(need.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            need = cv2.dilate(need, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32)
            need = cv2.GaussianBlur(cv2.resize(need, (HRW, HRH), interpolation=cv2.INTER_LINEAR), (0, 0), 3.0)
            carry[name] = np.clip(need * 1.6, 0, 1)
    D0 = cv2.resize(d0_lo, (HRW, HRH), interpolation=cv2.INTER_CUBIC)

    # near-foreground layer
    if c.get("near"):
        a2 = smoothstep(c["near"][0], c["near"][1], cv2.GaussianBlur(D, (0, 0), 3))
        if "near_box" in c:
            a2 = a2 * box_mask([c["near_box"]], HRW, HRH, feather=25)
        a2 = (a2 * (1 - a_any)).astype(np.float32)
    else:
        a2 = None

    glow = np.ones((HRH // 8, HRW // 8), np.float32)
    if c.get("glow"):
        glow = box_mask(c["glow"], HRW // 8, HRH // 8, feather=5)

    if debug:
        tiles = []
        for name in names:
            vis = frames[name].astype(np.float32) * (0.25 + 0.75 * alpha[name][..., None])
            if a2 is not None:
                vis = vis * (1 - 0.6 * a2[..., None]) + np.array([40, 90, 255], np.float32) * 0.6 * a2[..., None]
            if name in carry:
                vis = vis * (1 - 0.5 * carry[name][..., None]) + np.array([255, 60, 60], np.float32) * 0.5 * carry[name][..., None]
            tiles.append(cv2.resize(vis.astype(np.uint8), (1376, 768)))
        tiles.append(cv2.resize(np.clip(plate, 0, 255).astype(np.uint8), (1376, 768)))
        cv2.imwrite(f"{CACHE}/dbg_s{n}.jpg", cv2.cvtColor(np.vstack(tiles), cv2.COLOR_RGB2BGR))
        return

    lin = (hr.astype(np.float32) / 255) ** 2.2
    plate_lin = (np.clip(plate, 0, 255).astype(np.float32) / 255) ** 2.2
    data = dict(D0c=cv2.GaussianBlur(D0, (0, 0), 4), D0w=cv2.GaussianBlur(D0, (0, 0), 12),
                D1=D1, D1med=D1med, glow=glow)
    data["L0"] = np.stack(build_stack(plate_lin, None, c["hb"]))
    del plate_lin, plate
    x0, y0, x1, y1 = bbox(a_any, 60)
    data["box1"] = np.array([x0, y0, x1, y1])
    for name in names:                      # the characters only need the blur stack if the shot defocuses them
        srgb = frames[name][y0:y1, x0:x1].astype(np.float32) / 255
        a = alpha[name][y0:y1, x0:x1]
        if name in carry:                   # glow follows the character only, not the background it carries
            data["A_" + name] = (a * 255 + 0.5).astype(np.uint8)
            a = np.maximum(a, carry[name][y0:y1, x0:x1])
        stack = build_stack(srgb, a, 0.0, gamma=1.0, radii=RADII if c.get("defocus") else RADII[:1])
        data["P_" + name] = np.stack(stack)
    if a2 is not None:
        x0, y0, x1, y1 = bbox(a2, 60)
        data["box2"] = np.array([x0, y0, x1, y1])
        data["L2"] = np.stack(build_stack(lin[y0:y1, x0:x1], a2[y0:y1, x0:x1], c["hb"]))
    for a, b in c.get("morphs", []):
        small = [cv2.resize(frames[k], (w, h), interpolation=cv2.INTER_AREA) for k in (a, b)]
        zone = cv2.dilate(np.maximum(masks[a], masks[b]), np.ones((15, 15), np.uint8))
        data[f"F_{a}__{b}"], data[f"F_{b}__{a}"] = morph_flow(small[0], small[1], zone)
    np.savez(f"{CACHE}/s{n}_layers.npz", **data)


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    dbg = "--debug" in sys.argv
    for n in [int(a) for a in sys.argv[1:] if a.isdigit()] or [1, 2, 3, 4, 5]:
        build(n, debug=dbg)
        print("scene", n, "done", flush=True)
