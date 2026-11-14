# The Last Durian – storyboard video

`the-last-durian.mp4` is a 32 second, 1920×1080, 30 fps animated cut of the five storyboard
frames in the parent folder. It has no audio.

No video model is involved. The characters are animated pose to pose: every scene has a few key
poses in `poses/` (edited copies of the storyboard frame with a different gesture or expression),
and the renderer warps one pose into the next along optical flow. On top of that the characters
breathe, sway, bounce when they laugh and, in the first shot, he walks in. Each frame is also
split into depth layers and filmed with a virtual camera.

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `prep.py` | For each frame and key pose: keeps only the pixels the pose really changed, 2× upscale, saliency matte and depth map (saved in `maps/`) |
| 2 | `layers.py` | Splits the frame into background plate, characters (one cut-out per key pose) and near foreground, pre-blurs the layers with a bokeh-shaped kernel, computes the flow between poses |
| 3 | `render.py` | Per shot: acting, parallax camera move, depth of field, character rim light and halo, light wash, dust, sparkles, bloom, colour grade |
| 4 | `assemble.py` | Short dissolves between shots, titles, film grain, fades, H.264 encode |

Where to change things in `render.py`:

- `ACT` is the acting: per shot and character, which key pose is reached when and how fast.
  `motion()` holds the body movement (walk, startle, laughter).
- `shot1` … `shot5` are the camera moves and the lens. Focus stays on the two leads for the whole
  film, `far` / `near` set how soft things get behind and in front of them. Only the closing shot
  defocuses the whole frame, under the fade-out.
- `LOOK` is the light and grade of each scene.

## Re-rendering

```bash
python -m venv /tmp/venv
/tmp/venv/bin/pip install numpy opencv-python-headless onnxruntime ncnn numexpr pillow imageio-ffmpeg

/tmp/venv/bin/python prep.py                 # only what is missing; delete maps/ and $DURIAN_WORK/prep to redo
/tmp/venv/bin/python layers.py
/tmp/venv/bin/python render.py shot 1 2 3 4 5
/tmp/venv/bin/python assemble.py
```

Working files (about 8 GB) go to `$DURIAN_WORK`, default `/tmp/work`. `render.py still 3 2.5 out.jpg`
renders a single frame of shot 3 at 2.5 s for look-dev.

`prep.py` and `assemble.py` expect these files under `$DURIAN_WORK`; they are not in the repo:

| File | Source |
| --- | --- |
| `esr/realesr-animevideov3-x2.{param,bin}` | `models/` inside the `realesrgan-ncnn-py` wheel on PyPI |
| `models/isnet_medium.onnx` | the `/models/medium` chunks listed in `dist/resources.json` of the npm package `@imgly/background-removal-node@1.4.5`, concatenated |
| `models/da2_small_int8.onnx` | `plugin/models/depth_anything_v2_small_int8.onnx` in the GitHub repo `rydersd/ill-tool` |
| `fonts/DMSerifDisplay-Regular.ttf`, `DMSerifDisplay-Italic.ttf`, `Poppins-Medium.ttf` | `ofl/dmserifdisplay` and `ofl/poppins` in the GitHub repo `google/fonts` |
