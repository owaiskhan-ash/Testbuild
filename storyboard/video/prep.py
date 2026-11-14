"""One-off preparation of each storyboard frame and its key poses: 2x upscale, matte, depth.

A key pose is an edited copy of a storyboard frame (same framing, different expression or
gesture). Only the pixels that really changed are taken from it, so everything else stays
rock steady when the animation switches pose.

Needs three model files under $DURIAN_WORK (default /tmp/work), see README.md:
  esr/realesr-animevideov3-x2.{param,bin}   Real-ESRGAN anime model (ncnn)
  models/isnet_medium.onnx                  ISNet saliency matte
  models/da2_small_int8.onnx                Depth Anything V2 small

  python prep.py [scene numbers]      (skips anything already prepared)
"""
import os
import subprocess
import sys

import cv2
import ncnn
import numpy as np
import onnxruntime as ort

from layers import MAPS, POSES, PREP, SCENES, WORK, box_mask

BOARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NAMES = ["scene-1-arrival", "scene-2-the-reach", "scene-3-the-standoff", "scene-4-the-share", "scene-5-sunset"]


class Models:
    def __init__(self):
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = 2
        self.net.load_param(f"{WORK}/esr/realesr-animevideov3-x2.param")
        self.net.load_model(f"{WORK}/esr/realesr-animevideov3-x2.bin")
        self.isnet = ort.InferenceSession(f"{WORK}/models/isnet_medium.onnx", providers=["CPUExecutionProvider"])
        self.da = ort.InferenceSession(f"{WORK}/models/da2_small_int8.onnx", providers=["CPUExecutionProvider"])

    def upscale(self, rgb, tiles=(2, 2), pad=16):
        h, w = rgb.shape[:2]
        out = np.zeros((h * 2, w * 2, 3), np.float32)
        ys = np.linspace(0, h, tiles[0] + 1).astype(int)
        xs = np.linspace(0, w, tiles[1] + 1).astype(int)
        for i in range(tiles[0]):
            for j in range(tiles[1]):
                y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
                py0, py1, px0, px1 = max(0, y0 - pad), min(h, y1 + pad), max(0, x0 - pad), min(w, x1 + pad)
                x = np.ascontiguousarray(rgb[py0:py1, px0:px1].transpose(2, 0, 1).astype(np.float32) / 255)
                ex = self.net.create_extractor()
                ex.input("data", ncnn.Mat(x))
                t = np.clip(np.array(ex.extract("output")[1]).transpose(1, 2, 0), 0, 1)
                oy, ox = (y0 - py0) * 2, (x0 - px0) * 2
                out[y0 * 2:y1 * 2, x0 * 2:x1 * 2] = t[oy:oy + (y1 - y0) * 2, ox:ox + (x1 - x0) * 2]
        return (out * 255 + 0.5).astype(np.uint8)

    def matte(self, rgb):
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)
        m = self.isnet.run(["output"], {"input": ((x - 128) / 256).transpose(2, 0, 1)[None]})[0][0, 0]
        return cv2.resize(np.clip(m, 0, 1), (w, h), interpolation=cv2.INTER_LINEAR)

    def depth(self, rgb):
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (1106, 616), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None]
        d = self.da.run(None, {"pixel_values": x})[0][0]
        return np.clip(cv2.resize((d - d.min()) / (d.max() - d.min()), (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)


def read_rgb(path, size=None):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if size and img.shape[:2] != size:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return img


def write_rgb(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def colour_match(base, pose):
    """Remove any overall tone drift of an edited frame, fitted on the pixels it left alone."""
    d = np.abs(pose.astype(np.float32) - base.astype(np.float32)).max(axis=2)
    same = cv2.erode((cv2.GaussianBlur(d, (0, 0), 2) < 14).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    out = pose.astype(np.float32)
    for ch in range(3):
        k, b = np.polyfit(out[..., ch][same][::7], base[..., ch][same][::7].astype(np.float32), 1)
        out[..., ch] = out[..., ch] * k + b
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def change_mask(base, pose, base_matte, boxes):
    """Soft mask of the pixels an edited frame really changed, kept near the characters."""
    h, w = base.shape[:2]
    d = np.abs(pose.astype(np.float32) - base.astype(np.float32)).max(axis=2)
    d = cv2.GaussianBlur(d, (0, 0), 1.5)
    m = (d > 16 + 2 * np.median(d)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    near = cv2.dilate((base_matte > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121)))
    m = m & near
    if boxes:
        m = m & (box_mask(boxes, w, h) > 0).astype(np.uint8)
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.5)


def prepare(n, item):
    """item: 'base', 'plate' or a pose name. One item per process - the upscaler is memory hungry."""
    os.makedirs(PREP, exist_ok=True)
    os.makedirs(MAPS, exist_ok=True)
    models = Models()
    c = SCENES[n]
    base = read_rgb(f"{BOARD}/{NAMES[n - 1]}.png")
    if item == "base":
        write_rgb(f"{PREP}/s{n}_hr.png", models.upscale(base))
        cv2.imwrite(f"{MAPS}/s{n}_matte.png", (models.matte(base) * 255 + 0.5).astype(np.uint8))
        cv2.imwrite(f"{MAPS}/s{n}_depth.png", (models.depth(base) * 65535).astype(np.uint16))
    elif item == "plate":
        plate = read_rgb(f"{POSES}/{c['plate']}", base.shape[:2])
        write_rgb(f"{PREP}/s{n}_plate_hr.png", models.upscale(plate))
        cv2.imwrite(f"{MAPS}/s{n}_plate_depth.png", (models.depth(plate) * 65535).astype(np.uint16))
    else:
        fname, boxes = c["poses"][item]
        base_matte = cv2.imread(f"{MAPS}/s{n}_matte.png", 0).astype(np.float32) / 255
        pose = colour_match(base, read_rgb(f"{POSES}/{fname}", base.shape[:2]))
        m = change_mask(base, pose, base_matte, boxes)
        merged = (base * (1 - m[..., None]) + pose * m[..., None] + 0.5).astype(np.uint8)
        write_rgb(f"{PREP}/s{n}_{item}_lr.png", merged)
        write_rgb(f"{PREP}/s{n}_{item}_hr.png", models.upscale(merged))
        cv2.imwrite(f"{MAPS}/s{n}_{item}_mask.png", (m * 255 + 0.5).astype(np.uint8))
        cv2.imwrite(f"{MAPS}/s{n}_{item}_matte.png", (models.matte(merged) * 255 + 0.5).astype(np.uint8))
        cv2.imwrite(f"{MAPS}/s{n}_{item}_depth.png", (models.depth(merged) * 255 + 0.5).astype(np.uint8))
        print(f"scene {n} pose {item}: {100 * (m > 0.5).mean():.1f}% of the frame changed", flush=True)


def main(which):
    for n in which:
        c = SCENES[n]
        todo = [("base", f"{PREP}/s{n}_hr.png")]
        if c.get("plate"):
            todo.append(("plate", f"{PREP}/s{n}_plate_hr.png"))
        todo += [(name, f"{MAPS}/s{n}_{name}_depth.png") for name in c["poses"]]
        for item, product in todo:
            if os.path.exists(product):
                continue
            subprocess.run([sys.executable, os.path.abspath(__file__), "--item", str(n), item], check=True)
        print("scene", n, "prepared", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--item"]:
        prepare(int(sys.argv[2]), sys.argv[3])
    else:
        main([int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5])
