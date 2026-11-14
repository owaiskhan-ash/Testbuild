"""Cut the rendered shots together: dissolves, titles, grain, fades, H.264 encode.

  python assemble.py [out.mp4]
"""
import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from layers import WORK
from render import FPS, H, SHOT, SHOTS, W, XFADE, ss

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(WORK, "fonts")
CREAM = (255, 245, 224)


def tracked_text(draw, xy_center, text, font, tracking, fill):
    """Draw centred text with letter spacing."""
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = xy_center[0] - total / 2
    for ch, w in zip(text, widths):
        draw.text((x, xy_center[1]), ch, font=font, fill=fill, anchor="ls")
        x += w + tracking


def title_card(lines):
    """lines: [(text, font file, size, tracking, baseline y)] -> premultiplied float RGBA with soft shadow."""
    txt = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(txt)
    for text, font, size, tracking, y in lines:
        tracked_text(d, (W / 2, y), text, ImageFont.truetype(os.path.join(FONTS, font), size), tracking, 255)
    a = np.asarray(txt, np.float32) / 255
    shadow = np.asarray(txt.filter(ImageFilter.GaussianBlur(16)), np.float32) / 255
    shadow = np.clip(shadow * 1.6, 0, 1) * 0.55
    return a, shadow


def scrim():
    """Soft dark oval behind the title so cream type reads over a bright blurred frame."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / (W * 0.48)) ** 2 + ((y - H * 0.5) / (H * 0.42)) ** 2)
    return np.clip(1 - r, 0, 1) ** 1.2 * 0.42


def put_title(frame, card, amount, scale=1.0, scrim_map=None):
    if amount <= 0:
        return frame
    a, shadow = card
    if scale != 1.0:
        m = cv2.getRotationMatrix2D((W / 2, H / 2), 0, scale)
        a = cv2.warpAffine(a, m, (W, H), flags=cv2.INTER_LINEAR)
        shadow = cv2.warpAffine(shadow, m, (W, H), flags=cv2.INTER_LINEAR)
    dark = shadow * amount
    if scrim_map is not None:
        dark = 1 - (1 - dark) * (1 - scrim_map * amount)
    frame = frame * (1 - dark[..., None])
    aa = (a * amount)[..., None]
    return frame * (1 - aa) + np.array(CREAM, np.float32) * aa


def main(out_path):
    shots = {n: np.load(f"{SHOTS}/s{n}.npy", mmap_mode="r") for n in SHOT}
    starts, pos = {}, 0
    for n in sorted(shots):
        starts[n] = pos
        pos += len(shots[n]) - XFADE
    total = pos + XFADE
    print("frames", total, "seconds", total / FPS, starts, flush=True)

    opening = title_card([("THE LAST DURIAN", "DMSerifDisplay-Regular.ttf", 138, 10, 548),
                          ("A  MARKET  MEET-CUTE", "Poppins-Medium.ttf", 30, 11, 628)])
    ending = title_card([("The End", "DMSerifDisplay-Italic.ttf", 120, 2, 575)])
    scr = scrim()

    rng = np.random.default_rng(7)
    grains = [cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.7) for _ in range(6)]
    grains = [g / g.std() for g in grains]

    ff = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-vf", "scale=out_color_matrix=bt709:out_range=tv",
         "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
         "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
         "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)

    end_start = starts[5]
    for f in range(total):
        t = f / FPS
        frame = None
        for n in sorted(shots):
            i = f - starts[n]
            if 0 <= i < len(shots[n]):
                img = shots[n][i].astype(np.float32)
                if frame is None:
                    frame = img
                else:                                  # dissolve: second shot fades up over the first
                    k = np.float32(ss(i + 0.5, 0, XFADE))
                    frame = frame * (1 - k) + img * k
        # titles
        frame = put_title(frame, opening, ss(t, 0.6, 1.5) * (1 - ss(t, 2.3, 3.1)), 1 + 0.012 * t, scr)
        te = t - end_start / FPS
        frame = put_title(frame, ending, ss(te, 6.3, 7.1), 1 + 0.01 * max(te - 6.3, 0))
        # grain (a touch stronger in the mid-tones), then fades
        g = grains[rng.integers(len(grains))]
        if rng.random() < 0.5:
            g = g[:, ::-1]
        if rng.random() < 0.5:
            g = g[::-1]
        lum = frame.mean(axis=2) / 255
        frame += (g * (1.8 * (0.45 + 2.2 * lum * (1 - lum))))[..., None]
        frame *= np.float32(ss(t, 0.0, 0.9) * (1 - ss(te, 7.45, 8.15)))
        ff.stdin.write(np.clip(frame + 0.5, 0, 255).astype(np.uint8).tobytes())
        if f % 60 == 0:
            print("frame", f, flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", out_path, os.path.getsize(out_path) / 1e6, "MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "the-last-durian.mp4"))
