"""One-off preparation of each storyboard frame: 2x upscale, saliency matte, depth map.

Needs three model files under $DURIAN_WORK (default /tmp/work), see README.md:
  esr/realesr-animevideov3-x2.{param,bin}   Real-ESRGAN anime model (ncnn)
  models/isnet_medium.onnx                  ISNet saliency matte
  models/da2_small_int8.onnx                Depth Anything V2 small

  python prep.py [scene numbers]
"""
import os
import sys

import cv2
import ncnn
import numpy as np
import onnxruntime as ort

from layers import MAPS, PREP, WORK

BOARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NAMES = ["scene-1-arrival", "scene-2-the-reach", "scene-3-the-standoff", "scene-4-the-share", "scene-5-sunset"]


def upscale(net, rgb, tiles=(2, 2), pad=16):
    h, w = rgb.shape[:2]
    out = np.zeros((h * 2, w * 2, 3), np.float32)
    ys = np.linspace(0, h, tiles[0] + 1).astype(int)
    xs = np.linspace(0, w, tiles[1] + 1).astype(int)
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
            py0, py1, px0, px1 = max(0, y0 - pad), min(h, y1 + pad), max(0, x0 - pad), min(w, x1 + pad)
            x = np.ascontiguousarray(rgb[py0:py1, px0:px1].transpose(2, 0, 1).astype(np.float32) / 255)
            ex = net.create_extractor()
            ex.input("data", ncnn.Mat(x))
            t = np.clip(np.array(ex.extract("output")[1]).transpose(1, 2, 0), 0, 1)
            oy, ox = (y0 - py0) * 2, (x0 - px0) * 2
            out[y0 * 2:y1 * 2, x0 * 2:x1 * 2] = t[oy:oy + (y1 - y0) * 2, ox:ox + (x1 - x0) * 2]
    return out


def main(which):
    os.makedirs(PREP, exist_ok=True)
    os.makedirs(MAPS, exist_ok=True)
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.load_param(f"{WORK}/esr/realesr-animevideov3-x2.param")
    net.load_model(f"{WORK}/esr/realesr-animevideov3-x2.bin")
    isnet = ort.InferenceSession(f"{WORK}/models/isnet_medium.onnx", providers=["CPUExecutionProvider"])
    da = ort.InferenceSession(f"{WORK}/models/da2_small_int8.onnx", providers=["CPUExecutionProvider"])
    for n in which:
        rgb = cv2.cvtColor(cv2.imread(f"{BOARD}/{NAMES[n - 1]}.png"), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        hr = upscale(net, rgb)
        cv2.imwrite(f"{PREP}/s{n}_hr.png", cv2.cvtColor((hr * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))

        x = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32)
        m = isnet.run(["output"], {"input": ((x - 128) / 256).transpose(2, 0, 1)[None]})[0][0, 0]
        m = cv2.resize(np.clip(m, 0, 1), (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f"{MAPS}/s{n}_matte.png", (m * 255 + 0.5).astype(np.uint8))

        x = cv2.resize(rgb, (1106, 616), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32).transpose(2, 0, 1)[None]
        d = da.run(None, {"pixel_values": x})[0][0]
        d = cv2.resize((d - d.min()) / (d.max() - d.min()), (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{MAPS}/s{n}_depth.png", (np.clip(d, 0, 1) * 65535).astype(np.uint16))
        print("scene", n, "prepared", flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5])
